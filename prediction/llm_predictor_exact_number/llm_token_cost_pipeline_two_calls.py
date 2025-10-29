# llm_token_cost_pipeline_two_calls.py
"""
Two-call pipeline for estimating token & tool usage costs with LLMs, incorporating repository file trees.

This version addresses the context length limitation by making two LLM calls:
1. First call: Generate a summary of the repository structure relevant to the problem
2. Second call: Make predictions using the summary instead of the full file tree

This approach allows handling much larger repositories while staying within context limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import logging
import os
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Optional: Anthropic client
try:
    import anthropic
except ImportError:
    anthropic = None

###############################################################################
# Part 1 ──────────────────────────────────────────────────────────────────────
# Configuration dataclass
###############################################################################

class Setting(str, Enum):
    """The experimental settings we support with repository information."""

    # New settings with repository information
    ZERO_PS_TOOL_DIFF_REPO = "ZeroShot_ProblemStatement_ToolReasoning_Difficulty_RepoInfo"
    FEW_PS_TOOL_DIFF_REPO = "FewShot_ProblemStatement_ToolReasoning_Difficulty_RepoInfo"


@dataclass
class Config:
    """Centralised runtime configuration."""

    # Core paths - updated to use the enhanced dataset with repo stats and file trees (exact number prediction)
    dataset_path: Path = Path("./prediction/llm_predictor_exact_number/simplified_with_tool_avgs.csv")
    results_root: Path = Path("./prediction/llm_predictor_exact_number/results_two_calls")

    # Experiment controls
    sample_size: int = 490
    few_shot_k: int = 4
    random_seed: int = 42

    # LLM settings
    model_name: str = "claude-3-7-sonnet-20250219"
    temperature: float = 0.7
    max_tokens: int = 10240
    api_key_env: str = "ANTHROPIC_API_KEY"

    # Settings to run - focus on the new repo file tree settings
    settings: List[Setting] = field(
        default_factory=lambda: [
            Setting.ZERO_PS_TOOL_DIFF_REPO,
            Setting.FEW_PS_TOOL_DIFF_REPO,
        ]
    )

    @property
    def api_key(self) -> str:
        key = os.getenv(self.api_key_env)
        if key:
            return key
        
        hardcoded_key = ""
        if hardcoded_key:
            return hardcoded_key
            
        raise RuntimeError(f"{self.api_key_env} not found in environment and no hardcoded key available")

###############################################################################
# Part 2 ──────────────────────────────────────────────────────────────────────
# Helper functions (reused from original pipeline)
###############################################################################

def sample_dataframe(df: pd.DataFrame, n: int, exclude_indices: List[int], seed: int) -> pd.DataFrame:
    """Return *n* random rows from *df* excluding given indices."""
    rng = np.random.default_rng(seed)
    available_indices = [i for i in range(len(df)) if i not in exclude_indices]
    if n > len(available_indices):
        raise ValueError("Requested sample larger than population w/o excluded rows.")
    chosen = rng.choice(available_indices, size=n, replace=False)
    return df.iloc[chosen].copy()

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-9, None))) * 100.0

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def corrcoef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])

def get_llm_response(prompt: str, system_prompt: str, cfg: Config) -> Dict[str, Any]:
    """Call the LLM and return a dict with `text` and cost metadata."""
    if anthropic is None:
        raise ImportError("anthropic package not installed.")

    client = anthropic.Client(api_key=cfg.api_key)

    try:
        response = client.messages.create(
            model=cfg.model_name,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            messages=[{"role": "user", "content": prompt}],
            system=system_prompt
        )
        text_out: str = response.content[0].text.strip()
        meta = {
            "input_tokens": getattr(response.usage, "input_tokens", None) if hasattr(response, "usage") else None,
            "output_tokens": getattr(response.usage, "output_tokens", None) if hasattr(response, "usage") else None,
            "latency_s": getattr(response, "response_ms", None) and response.response_ms / 1000.0,
        }
        return {"text": text_out, "meta": meta}
    except Exception as exc:
        logging.exception("LLM call failed")
        raise exc

def parse_estimates(text: str, expect_tools: bool) -> Dict[str, int]:
    """Extract integer estimates wrapped in XML‑style tags."""
    tags = [
        "INPUT_TOKEN_ESTIMATE",
        "OUTPUT_TOKEN_ESTIMATE",
        "TOOL_USE_EXECUTE_BASH_ESTIMATE",
        "TOOL_USE_STR_REPLACE_EDITOR_ESTIMATE",
        "TOOL_USE_THINK_ESTIMATE",
    ] if expect_tools else [
        "INPUT_TOKEN_ESTIMATE",
        "OUTPUT_TOKEN_ESTIMATE",
    ]
    
    out: Dict[str, int] = {}
    for tag in tags:
        match = re.search(rf"<{tag}>(.*?)</{tag}>", text)
        if match is None:
            raise ValueError(f"Tag <{tag}> not found in output: {text[:120]}...")
        try:
            content = match.group(1).strip()
            content = re.sub(r'^#+', '', content)
            out[tag] = int(round(float(content)))
        except ValueError as err:
            raise ValueError(f"Non‑numeric value for <{tag}>: {match.group(1)}") from err
    return out

def plot_pred_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, title: str, save_path: Path | None = None) -> None:
    """Scatter plot of predictions vs. ground truth after sorting by GT."""
    idx_sorted = np.argsort(y_true)
    y_true_sorted = y_true[idx_sorted]
    y_pred_sorted = y_pred[idx_sorted]

    plt.figure(figsize=(8, 5))
    plt.plot(y_true_sorted, label="Ground Truth")
    plt.plot(y_pred_sorted, label="Prediction")
    plt.title(title)
    plt.xlabel("Problem index (sorted by GT)")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.close()

###############################################################################
# Part 3 ──────────────────────────────────────────────────────────────────────
# Two-call prompt builder utilities
###############################################################################

# OpenHands system prompt and tool descriptions
OPENHANDS_SYSTEM_PROMPT = """Your primary role is to assist users by executing commands, modifying code, and solving technical problems effectively. You should be thorough, methodical, and prioritize quality over speed.\n* If the user asks a question, like \"why is X happening\", don't try to fix the problem. Just give an answer to the question.\n\n<EFFICIENCY>\n* Each action you take is somewhat expensive. Wherever possible, combine multiple actions into a single action, e.g. combine multiple bash commands into one, using sed and grep to edit/view multiple files at once.\n* When exploring the codebase, use efficient tools like find, grep, and git commands with appropriate filters to minimize unnecessary operations.\n</EFFICIENCY>\n\n<FILE_SYSTEM_GUIDELINES>\n* When a user provides a file path, do NOT assume it's relative to the current working directory. First explore the file system to locate the file before working on it.\n* If asked to edit a file, edit the file directly, rather than creating a new file with a different filename.\n* For global search-and-replace operations, consider using `sed` instead of opening file editors multiple times.\n</FILE_SYSTEM_GUIDELINES>\n\n<CODE_QUALITY>\n* Write clean, efficient code with minimal comments. Avoid redundancy in comments: Do not repeat information that can be easily inferred from the code itself.\n* When implementing solutions, focus on making the minimal changes needed to solve the problem.\n* Before implementing any changes, first thoroughly understand the codebase through exploration.\n* If you are adding a lot of code to a function or file, consider splitting the function or file into smaller pieces when appropriate.\n</CODE_QUALITY>\n\n<VERSION_CONTROL>\n* When configuring git credentials, use \"openhands\" as the user.name and \"openhands@all-hands.dev\" as the user.email by default, unless explicitly instructed otherwise.\n* Exercise caution with git operations. Do NOT make potentially dangerous changes (e.g., pushing to main, deleting repositories) unless explicitly asked to do so.\n* When committing changes, use `git status` to see all modified files, and stage all files necessary for the commit. Use `git commit -a` whenever possible.\n* Do NOT commit files that typically shouldn't go into version control (e.g., node_modules/, .env files, build directories, cache files, large binaries) unless explicitly instructed by the user.\n* If unsure about committing certain files, check for the presence of .gitignore files or ask the user for clarification.\n</VERSION_CONTROL>\n\n<PULL_REQUESTS>\n* When creating pull requests, create only ONE per session/issue unless explicitly instructed otherwise.\n* When working with an existing PR, update it with new commits rather than creating additional PRs for the same issue.\n* When updating a PR, preserve the original PR title and purpose, updating description only when necessary.\n</PULL_REQUESTS>\n\n<PROBLEM_SOLVING_WORKFLOW>\n1. EXPLORATION: Thoroughly explore relevant files and understand the context before proposing solutions\n2. ANALYSIS: Consider multiple approaches and select the most promising one\n3. TESTING:\n   * For bug fixes: Create tests to verify issues before implementing fixes\n   * For new features: Consider test-driven development when appropriate\n   * If the repository lacks testing infrastructure and implementing tests would require extensive setup, consult with the user before investing time in building testing infrastructure\n4. IMPLEMENTATION: Make focused, minimal changes to address the problem\n5. VERIFICATION: Test your implementation thoroughly, including edge cases\n</PROBLEM_SOLVING_WORKFLOW>\n\n<SECURITY>\n* Only use GITHUB_TOKEN and other credentials in ways the user has explicitly requested and would expect.\n* Use APIs to work with GitHub or other platforms, unless the user asks otherwise or your task requires browsing.\n</SECURITY>\n\n<ENVIRONMENT_SETUP>\n* When user asks you to run an application, don't stop if the application is not installed. Instead, please install the application and run the command again.\n* If you encounter missing dependencies:\n  1. First, look around in the repository for existing dependency files (requirements.txt, pyproject.toml, package.json, Gemfile, etc.)\n  2. If dependency files exist, use them to install all dependencies at once (e.g., `pip install -r requirements.txt`, `npm install`, etc.)\n  3. Only install individual packages directly if no dependency files are found or if only specific packages are needed\n* Similarly, if you encounter missing dependencies for essential tools requested by the user, install them when possible.\n</ENVIRONMENT_SETUP>\n\n<TROUBLESHOOTING>\n* If you've made repeated attempts to solve a problem but tests still fail or the user reports it's still broken:\n  1. Step back and reflect on 5-7 different possible sources of the problem\n  2. Assess the likelihood of each possible cause\n  3. Methodically address the most likely causes, starting with the highest probability\n  4. Document your reasoning process\n* When you run into any major issue while executing a plan from the user, please don't try to directly work around it. Instead, propose a new plan and confirm with the user before proceeding.\n</TROUBLESHOOTING>"""

TOOL_USE_PROMPT = """[{"type": "function", "function": {"name": "execute_bash", "description": "Execute a bash command in the terminal within a persistent shell session.\n\n### Command Execution\n* One command at a time: You can only execute one bash command at a time. If you need to run multiple commands sequentially, use `&&` or `;` to chain them together.\n* Persistent session: Commands execute in a persistent shell session where environment variables, virtual environments, and working directory persist between commands.\n* Timeout: Commands have a soft timeout of 120 seconds, once that's reached, you have the option to continue or interrupt the command (see section below for details)\n\n### Running and Interacting with Processes\n* Long running commands: For commands that may run indefinitely, run them in the background and redirect output to a file, e.g. `python3 app.py > server.log 2>&1 &`.\n* Interact with running process: If a bash command returns exit code `-1`, this means the process is not yet finished. By setting `is_input` to `true`, you can:\n  - Send empty `command` to retrieve additional logs\n  - Send text (set `command` to the text) to STDIN of the running process\n  - Send control commands like `C-c` (Ctrl+C), `C-d` (Ctrl+D), or `C-z` (Ctrl+Z) to interrupt the process\n\n### Best Practices\n* Directory verification: Before creating new directories or files, first verify the parent directory exists and is the correct location.\n* Directory management: Try to maintain working directory by using absolute paths and avoiding excessive use of `cd`.\n\n### Output Handling\n* Output truncation: If the output exceeds a maximum length, it will be truncated before being returned.\n", "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The bash command to execute. Can be empty string to view additional logs when previous exit code is `-1`. Can be `C-c` (Ctrl+C) to interrupt the currently running process. Note: You can only execute one command at a time. If you need to run multiple commands sequentially, you can use `&&` or `;` to chain them together."}, "is_input": {"type": "string", "description": "If True, the command is an input to the running process. If False, the command is a bash command to be executed in the terminal. Default is False.", "enum": ["true", "false"]}}, "required": ["command"]}}}, {"type": "function", "function": {"name": "think", "description": "Use the tool to think about something. It will not obtain new information or make more changes to the repository, but just log the thought. Use it when complex reasoning or brainstorming is needed.\n\nCommon use cases:\n1. When exploring a repository and discovering the source of a bug, call this tool to brainstorm several unique ways of fixing the bug, and assess which change(s) are likely to be simplest and most effective.\n2. After receiving test results, use this tool to brainstorm ways to fix failing tests.\n3. When planning a complex refactoring, use this tool to outline different approaches and their tradeoffs.\n4. When designing a new feature, use this tool to think through architecture decisions and implementation details.\n5. When debugging a complex issue, use this tool to organize your thoughts and hypotheses.\n\nThe tool simply logs your thought process for better transparency and does not execute any code or make changes.", "parameters": {"type": "object", "properties": {"thought": {"type": "string", "description": "The thought to log."}}, "required": ["thought"]}}}, {"type": "function", "function": {"name": "finish", "description": "Signals the completion of the current task or conversation.\n\nUse this tool when:\n- You have successfully completed the user's requested task\n- You cannot proceed further due to technical limitations or missing information\n\nThe message should include:\n- A clear summary of actions taken and their results\n- Any next steps for the user\n- Explanation if you're unable to complete the task\n- Any follow-up questions if more information is needed\n\nThe task_completed field should be set to True if you believed you have completed the task, and False otherwise.\n", "parameters": {"type": "object", "required": ["message", "task_completed"], "properties": {"message": {"type": "string", "description": "Final message to send to the user"}, "task_completed": {"type": "string", "enum": ["true", "false", "partial"], "description": "Whether you have completed the task."}}}}}, {"type": "function", "function": {"name": "str_replace_editor", "description": "Custom editing tool for viewing, creating and editing files in plain-text format\n* State is persistent across command calls and discussions with the user\n* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep\n* The `create` command cannot be used if the specified `path` already exists as a file\n* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`\n* The `undo_edit` command will revert the last edit made to the file at `path`\n\n\nBefore using this tool:\n1. Use the view tool to understand the file's contents and context\n2. Verify the directory path is correct (only applicable when creating new files):\n   - Use the view tool to verify the parent directory exists and is the correct location\n\nWhen making edits:\n   - Ensure the edit results in idiomatic, correct code\n   - Do not leave the code in a broken state\n   - Always use absolute file paths (starting with /)\n\nCRITICAL REQUIREMENTS FOR USING THIS TOOL:\n\n1. EXACT MATCHING: The `old_str` parameter must match EXACTLY one or more consecutive lines from the file, including all whitespace and indentation. The tool will fail if `old_str` matches multiple locations or doesn't match exactly with the file content.\n\n2. UNIQUENESS: The `old_str` must uniquely identify a single instance in the file:\n   - Include sufficient context before and after the change point (3-5 lines recommended)\n   - If not unique, the replacement will not be performed\n\n3. REPLACEMENT: The `new_str` parameter should contain the edited lines that replace the `old_str`. Both strings must be different.\n\nRemember: when making multiple file edits in a row to the same file, you should prefer to send all edits in a single message with multiple calls to this tool, rather than multiple messages with a single call each.\n", "parameters": {"type": "object", "properties": {"command": {"description": "The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.", "enum": ["view", "create", "str_replace", "insert", "undo_edit"], "type": "string"}, "path": {"description": "Absolute path to file or directory, e.g. `/workspace/file.py` or `/workspace`.", "type": "string"}, "file_text": {"description": "Required parameter of `create` command, with the content of the file to be created.", "type": "string"}, "old_str": {"description": "Required parameter of `edit` command containing the string in `path` to replace.", "type": "string"}, "new_str": {"description": "Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.", "type": "string"}, "insert_line": {"description": "Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.", "type": "integer"}, "view_range": {"description": "Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.", "items": {"type": "integer"}, "type": "array"}}, "required": ["command", "path"]}}}]"""

# System prompts for the two-call approach
SYSTEM_PROMPTS: Dict[Setting, Dict[str, str]] = {
    Setting.ZERO_PS_TOOL_DIFF_REPO: {
        "summary": """You are **RepositoryAnalyzer**, an expert at analyzing code repositories and understanding their structure.

Your task is to analyze the provided repository file tree and problem statement, then create a concise summary that captures the key aspects relevant to solving the problem.

Focus on:
1. **Repository Structure**: Key directories, file organization patterns
2. **Technology Stack**: Programming languages, frameworks, build tools
3. **Complexity Indicators**: Size, depth, and organization complexity
4. **Problem-Relevant Files**: Files or directories that seem most relevant to the given task
5. **Architecture Patterns**: Any clear architectural patterns or design principles

Provide a structured summary that would help someone understand the repository context for this specific problem. Keep it concise but informative.""",
        
        "prediction": f"""You are **TokenEstimator**, not a coding agent.

ALWAYS do exactly FOUR things, in order:
1. **Repository Context Analysis**: Review the provided repository summary and understand the codebase structure and complexity.
2. **Task Planning**: Based on the repository context and the specific task, outline a high-level approach for solving the problem.
3. **Token and Tool Estimation**: Estimate how many tokens the assistant *would* emit and how many tool calls it *would* invoke to solve the task, taking into account the difficulty level and repository structure. For each step, provide detailed reasoning for your estimates.  
   Additionally, for tool use, break down how many times *each specific tool* would likely be invoked (e.g., `execute_bash`: 10, `str_replace_editor`: 20, `think`: 5).

IMPORTANT: The assistant will be using the OpenHands system with the following capabilities and workflow:

{OPENHANDS_SYSTEM_PROMPT}

Available Tools:
{TOOL_USE_PROMPT}

4. Output five lines in *this* exact format — and nothing else:
   <INPUT_TOKEN_ESTIMATE>####</INPUT_TOKEN_ESTIMATE>
   <OUTPUT_TOKEN_ESTIMATE>####</OUTPUT_TOKEN_ESTIMATE>
   <TOOL_USE_EXECUTE_BASH_ESTIMATE>##</TOOL_USE_EXECUTE_BASH_ESTIMATE>
   <TOOL_USE_STR_REPLACE_EDITOR_ESTIMATE>##</TOOL_USE_STR_REPLACE_EDITOR_ESTIMATE>
   <TOOL_USE_THINK_ESTIMATE>##</TOOL_USE_THINK_ESTIMATE>

Do **NOT** execute the task, do **NOT** call any tool, do **NOT** write code."""
    },

    Setting.FEW_PS_TOOL_DIFF_REPO: {
        "summary": """You are **RepositoryAnalyzer**, an expert at analyzing code repositories and understanding their structure.

Your task is to analyze the provided repository file tree and problem statement, then create a concise summary that captures the key aspects relevant to solving the problem.

Focus on:
1. **Repository Structure**: Key directories, file organization patterns
2. **Technology Stack**: Programming languages, frameworks, build tools
3. **Complexity Indicators**: Size, depth, and organization complexity
4. **Problem-Relevant Files**: Files or directories that seem most relevant to the given task
5. **Architecture Patterns**: Any clear architectural patterns or design principles

Provide a structured summary that would help someone understand the repository context for this specific problem. Keep it concise but informative.""",
        
        "prediction": f"""You are **TokenEstimator**, not a coding agent.

ALWAYS do exactly FOUR things, in order:
1. **Repository Context Analysis**: Review the provided repository summary and understand the codebase structure and complexity.
2. **Task Planning**: Based on the repository context and the specific task, outline a high-level approach for solving the problem.
3. **Token and Tool Estimation**: Estimate how many tokens the assistant *would* emit and how many tool calls it *would* invoke to solve the task, taking into account the difficulty level and repository structure. Use the provided examples to inform your estimates and provide detailed reasoning for each step.  
   Additionally, for tool use, break down how many times *each specific tool* would likely be invoked (e.g., `execute_bash`: 10, `str_replace_editor`: 20, `think`: 5).

IMPORTANT: The assistant will be using the OpenHands system with the following capabilities and workflow:

{OPENHANDS_SYSTEM_PROMPT}

Available Tools:
{TOOL_USE_PROMPT}

4. Output five lines in *this* exact format — and nothing else:
   <INPUT_TOKEN_ESTIMATE>####</INPUT_TOKEN_ESTIMATE>
   <OUTPUT_TOKEN_ESTIMATE>####</OUTPUT_TOKEN_ESTIMATE>
   <TOOL_USE_EXECUTE_BASH_ESTIMATE>##</TOOL_USE_EXECUTE_BASH_ESTIMATE>
   <TOOL_USE_STR_REPLACE_EDITOR_ESTIMATE>##</TOOL_USE_STR_REPLACE_EDITOR_ESTIMATE>
   <TOOL_USE_THINK_ESTIMATE>##</TOOL_USE_THINK_ESTIMATE>

Do **NOT** execute the task, do **NOT** call any tool, do **NOT** write code."""
    }
}

# Setting flags for the new settings
SETTING_FLAGS: Dict[Setting, Dict[str, bool]] = {
    Setting.ZERO_PS_TOOL_DIFF_REPO: {"tool": True, "difficulty": True, "fewshot": False, "repo_info": True},
    Setting.FEW_PS_TOOL_DIFF_REPO: {"tool": True, "difficulty": True, "fewshot": True, "repo_info": True},
}

def load_file_tree(repo_name: str) -> str:
    """Load the file tree for a given repository."""
    # Convert repo name from "owner/repo" format to "owner_repo" format
    file_tree_name = repo_name.replace('/', '_')
    file_tree_path = Path(f"dataset/repo_stats_analysis/repo_analyses/file_trees/{file_tree_name}_file_tree.txt")
    
    if not file_tree_path.exists():
        return f"File tree not found for repository: {repo_name}"
    
    try:
        with open(file_tree_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
        
    except Exception as e:
        return f"Error loading file tree for {repo_name}: {str(e)}"

def truncate_file_tree_for_repo(repo_name: str) -> None:
    """Truncate the file tree by removing the deepest layer."""
    # Convert repo name from "owner/repo" format to "owner_repo" format
    file_tree_name = repo_name.replace('/', '_')
    file_tree_path = Path(f"dataset/repo_stats_analysis/repo_analyses/file_trees/{file_tree_name}_file_tree.txt")
    
    if not file_tree_path.exists():
        logging.warning(f"Cannot truncate file tree for {repo_name}: file not found")
        return
    
    try:
        with open(file_tree_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        # Find the separator line (=====)
        separator_idx = -1
        for i, line in enumerate(lines):
            if line.startswith('=' * 20):
                separator_idx = i
                break
        
        if separator_idx < 0:
            logging.warning(f"Cannot truncate file tree for {repo_name}: no separator found")
            return
        
        # Keep header (repository info and separator)
        header_lines = lines[:separator_idx + 1]
        file_tree_lines = lines[separator_idx + 1:]
        
        # Find the maximum depth in the file tree
        max_depth = 0
        for line in file_tree_lines:
            if line.strip():
                # Count leading spaces/tabs to determine depth
                depth = len(line) - len(line.lstrip())
                max_depth = max(max_depth, depth)
        
        if max_depth == 0:
            logging.warning(f"Cannot truncate file tree for {repo_name}: no depth found")
            return
        
        # Remove lines at the deepest level
        filtered_lines = []
        removed_count = 0
        for line in file_tree_lines:
            if line.strip():
                line_depth = len(line) - len(line.lstrip())
                if line_depth < max_depth:
                    filtered_lines.append(line)
                else:
                    removed_count += 1
            else:
                filtered_lines.append(line)  # Keep empty lines
        
        # Reconstruct the content
        truncated_content = '\n'.join(header_lines + filtered_lines)
        truncated_content += f"\n\n[FILE TREE TRUNCATED: Removed {removed_count} lines at depth {max_depth} due to token limit]"
        
        # Write back to file
        with open(file_tree_path, 'w', encoding='utf-8') as f:
            f.write(truncated_content)
        
        logging.info(f"Truncated file tree for {repo_name}: removed {removed_count} lines at depth {max_depth}")
        
    except Exception as e:
        logging.error(f"Error truncating file tree for {repo_name}: {str(e)}")

def format_repository_info_for_summary(row: pd.Series) -> str:
    """Format repository information for the summary generation call."""
    repo_name = row.get('repo', 'Unknown')
    
    # Load the file tree
    file_tree_content = load_file_tree(repo_name)
    
    # Get basic stats from the row
    total_files = row.get('total_files', 0)
    total_lines = row.get('total_lines', 0)
    
    repo_info = f"""Repository Information:
- Repository: {repo_name}
- Total Files: {total_files:,}
- Total Lines of Code: {total_lines:,}

Repository File Structure:
{file_tree_content}"""
    
    return repo_info

def build_summary_prompt(row: pd.Series) -> str:
    """Build the prompt for the first LLM call to generate repository summary."""
    problem_prompt = f"Task: {row.problem_statement}\nDifficulty Level: {row.difficulty}"
    
    repo_info = format_repository_info_for_summary(row)
    
    summary_prompt = f"""Please analyze the following repository and problem statement to create a concise summary.

{repo_info}

{problem_prompt}

Based on this information, provide a structured summary that captures the key aspects of the repository relevant to solving this problem. Focus on structure, technology stack, complexity, and problem-relevant components."""

    return summary_prompt

def build_prediction_prompt(
    row: pd.Series,
    setting: Setting,
    repo_summary: str,
    few_shot_examples: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build the prompt for the second LLM call to make predictions."""
    
    flags = SETTING_FLAGS[setting]
    
    # Build the problem prompt with difficulty
    problem_prompt = f"Task: {row.problem_statement}\nDifficulty Level: {row.difficulty}"
    
    # Use the generated summary instead of full repo info
    repo_info = f"""Repository Summary:
{repo_summary}"""
    
    # Build examples section if few-shot
    examples_section = ""
    if flags["fewshot"] and few_shot_examples:
        examples = []
        for i, ex in enumerate(few_shot_examples, 1):
            example_text = f"""Example {i}:

Repository Summary:
{ex["repo_summary"]}

Task: {ex["problem_statement"]}
Difficulty Level: {ex["difficulty"]}

<INPUT_TOKEN_ESTIMATE>####{ex["total_prompt_tokens_mean"]}</INPUT_TOKEN_ESTIMATE>
<OUTPUT_TOKEN_ESTIMATE>####{ex["total_completion_tokens_mean"]}</OUTPUT_TOKEN_ESTIMATE>
<TOOL_USE_EXECUTE_BASH_ESTIMATE>##{ex["avg_tool_usage_execute_bash"]}</TOOL_USE_EXECUTE_BASH_ESTIMATE>
<TOOL_USE_STR_REPLACE_EDITOR_ESTIMATE>##{ex["avg_tool_usage_str_replace_editor"]}</TOOL_USE_STR_REPLACE_EDITOR_ESTIMATE>
<TOOL_USE_THINK_ESTIMATE>##{ex["avg_tool_usage_think"]}</TOOL_USE_THINK_ESTIMATE>"""
            
            examples.append(example_text)
        
        examples_section = "\n\nBelow are examples that can help you understand the task and better estimate the token usage:\n\n" + "\n\n".join(examples)
    
    # Build the complete user prompt
    user_prompt = f"""Below is the task description:

{repo_info}

{problem_prompt}

{examples_section}

Please provide your estimates in the following format:

<INPUT_TOKEN_ESTIMATE>####</INPUT_TOKEN_ESTIMATE>
<OUTPUT_TOKEN_ESTIMATE>####</OUTPUT_TOKEN_ESTIMATE>
<TOOL_USE_EXECUTE_BASH_ESTIMATE>##</TOOL_USE_EXECUTE_BASH_ESTIMATE>
<TOOL_USE_STR_REPLACE_EDITOR_ESTIMATE>##</TOOL_USE_STR_REPLACE_EDITOR_ESTIMATE>
<TOOL_USE_THINK_ESTIMATE>##</TOOL_USE_THINK_ESTIMATE>"""

    return user_prompt

###############################################################################
# Part 4 ──────────────────────────────────────────────────────────────────────
# Enhanced orchestrator with two-call approach
###############################################################################

def select_few_shot_examples(df: pd.DataFrame, exclude_indices: List[int], seed: int) -> List[pd.Series]:
    """Select few-shot examples ensuring different repos and difficulty levels."""
    rng = random.Random(seed)
    few_shot_examples: List[pd.Series] = []
    
    # Get unique repositories and difficulty levels
    available_df = df[~df.index.isin(exclude_indices)]
    unique_repos = available_df['repo'].unique()
    difficulty_levels = ["<15 min fix", "1-4 hours"]
    
    # Create a matrix of repo x difficulty combinations
    combinations = []
    for repo in unique_repos:
        for difficulty in difficulty_levels:
            repo_diff_df = available_df[(available_df['repo'] == repo) & (available_df['difficulty'] == difficulty)]
            if len(repo_diff_df) > 0:
                combinations.append((repo, difficulty))
    
    # Randomly shuffle combinations and select up to 4
    rng.shuffle(combinations)
    selected_combinations = combinations[:4]
    
    for repo, difficulty in selected_combinations:
        repo_diff_df = available_df[(available_df['repo'] == repo) & (available_df['difficulty'] == difficulty)]
        selected_idx = rng.choice(repo_diff_df.index)
        few_shot_examples.append(df.loc[selected_idx])
        exclude_indices.append(selected_idx)
    
    return few_shot_examples

def generate_repository_summaries(
    df: pd.DataFrame,
    cfg: Config,
    setting: Setting,
    few_shot_examples: Optional[List[pd.Series]] = None
) -> Dict[str, str]:
    """Generate repository summaries for all examples (including few-shot if applicable)."""
    summaries = {}
    
    # Generate summary for few-shot examples if applicable
    if SETTING_FLAGS[setting]["fewshot"] and few_shot_examples:
        print("Generating summaries for few-shot examples...")
        for ex in few_shot_examples:
            uid = ex["instance_id"]
            summary_prompt = build_summary_prompt(ex)
            system_prompt = SYSTEM_PROMPTS[setting]["summary"]
            
            try:
                llm_out = get_llm_response(summary_prompt, system_prompt, cfg)
                summaries[uid] = llm_out["text"]
                
                # Debug: Print token usage for few-shot summary generation
                input_tokens = llm_out["meta"].get("input_tokens", "N/A")
                output_tokens = llm_out["meta"].get("output_tokens", "N/A")
                total_tokens = (input_tokens + output_tokens) if isinstance(input_tokens, int) and isinstance(output_tokens, int) else "N/A"
                print(f"  Generated few-shot summary for {uid} - Tokens: {input_tokens} in + {output_tokens} out = {total_tokens} total")
            except Exception as e:
                logging.error(f"Failed to generate summary for {uid}: {e}")
                # Use a fallback summary
                summaries[uid] = f"Repository: {ex.get('repo', 'Unknown')} - Summary generation failed"
    
    return summaries

def run_setting(cfg: Config, setting: Setting) -> None:
    """Run inference for one setting using the two-call approach."""
    
    # Create output directory
    out_dir = cfg.results_root / setting.value
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(cfg.dataset_path)
    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")

    # ─── Few‑shot example selection ────────────────────────────────────
    exclude_pos = []
    few_shot_examples: List[pd.Series] = []
    
    if SETTING_FLAGS[setting]["fewshot"]:
        few_shot_examples = select_few_shot_examples(df, exclude_pos, cfg.random_seed)
        print(f"Selected {len(few_shot_examples)} few-shot examples from different repos and difficulty levels")
        for i, ex in enumerate(few_shot_examples):
            print(f"  Example {i+1}: {ex['repo']} - {ex['difficulty']}")

    # ─── Generate repository summaries ─────────────────────────────────
    print("Generating repository summaries...")
    repo_summaries = generate_repository_summaries(df, cfg, setting, few_shot_examples)

    # ─── Sampling test rows ───────────────────────────────────────────
    test_df = sample_dataframe(df, cfg.sample_size, exclude_pos, cfg.random_seed)
    print(f"Sampled {len(test_df)} test rows")

    # ─── Iterate over test rows ───────────────────────────────────────
    preds: Dict[str, Dict[str, int]] = {}
    raw_outputs: Dict[str, str] = {}
    all_summaries: Dict[str, str] = {}
    
    # Debug: Create file to save input/output history
    debug_file = out_dir / "debug_conversation_history.txt"
    with open(debug_file, 'w', encoding='utf-8') as f:
        f.write(f"=== DEBUG: Input/Output History for {setting.value} ===\n\n")

    pbar = tqdm(
        test_df.iterrows(), 
        total=len(test_df),
        desc=f"Running {setting.value}", 
        unit="row",
        ncols=100,
        leave=True,
        position=0
    )
    
    for _, row in pbar:
        uid = row["instance_id"]
        
        # First call: Generate repository summary with retry logic for token limits
        repo_summary = None
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Build prompt
                summary_prompt = build_summary_prompt(row)
                summary_system_prompt = SYSTEM_PROMPTS[setting]["summary"]
                
                # Debug: Save input to file
                with open(debug_file, 'a', encoding='utf-8') as f:
                    f.write(f"=== INSTANCE {uid} - CALL 1: SUMMARY GENERATION (Attempt {attempt + 1}) ===\n")
                    f.write(f"System Prompt:\n{summary_system_prompt}\n\n")
                    f.write(f"User Prompt:\n{summary_prompt}\n\n")
                    f.write("=" * 80 + "\n\n")
                
                summary_out = get_llm_response(summary_prompt, summary_system_prompt, cfg)
                repo_summary = summary_out["text"]
                all_summaries[uid] = repo_summary
                
                # Debug: Save output to file and print token usage
                with open(debug_file, 'a', encoding='utf-8') as f:
                    f.write(f"LLM Response:\n{repo_summary}\n\n")
                    f.write("=" * 80 + "\n\n")
                
                input_tokens = summary_out["meta"].get("input_tokens", "N/A")
                output_tokens = summary_out["meta"].get("output_tokens", "N/A")
                total_tokens = (input_tokens + output_tokens) if isinstance(input_tokens, int) and isinstance(output_tokens, int) else "N/A"
                print(f"  Generated summary for {uid} ({len(repo_summary)} chars) - Tokens: {input_tokens} in + {output_tokens} out = {total_tokens} total")
                break  # Success, exit retry loop
                
            except Exception as e:
                error_msg = str(e)
                
                # Check if it's a token limit error
                if "prompt is too long" in error_msg or "tokens >" in error_msg:
                    logging.warning(f"Token limit exceeded for {uid} (attempt {attempt + 1}), truncating file tree")
                    
                    # Debug: Save retry info to file
                    with open(debug_file, 'a', encoding='utf-8') as f:
                        f.write(f"Token limit exceeded, truncating file tree\n")
                        f.write(f"Error: {error_msg}\n\n")
                    
                    # Truncate the file tree by removing the deepest layer
                    truncate_file_tree_for_repo(row.get('repo', 'Unknown'))
                    
                    if attempt < max_retries - 1:  # Not the last attempt
                        continue
                    else:
                        # Last attempt failed, use fallback
                        logging.error(f"Failed to generate summary for {uid} after {max_retries} attempts: {e}")
                        repo_summary = f"Repository: {row.get('repo', 'Unknown')} - Summary generation failed after {max_retries} attempts"
                        all_summaries[uid] = repo_summary
                        
                        # Debug: Save final error to file
                        with open(debug_file, 'a', encoding='utf-8') as f:
                            f.write(f"=== INSTANCE {uid} - CALL 1: FINAL ERROR ===\n")
                            f.write(f"Error: {str(e)}\n\n")
                            f.write("=" * 80 + "\n\n")
                else:
                    # Other error, don't retry
                    logging.error(f"Failed to generate summary for {uid}: {e}")
                    repo_summary = f"Repository: {row.get('repo', 'Unknown')} - Summary generation failed"
                    all_summaries[uid] = repo_summary
                    
                    # Debug: Save error to file
                    with open(debug_file, 'a', encoding='utf-8') as f:
                        f.write(f"=== INSTANCE {uid} - CALL 1: ERROR ===\n")
                        f.write(f"Error: {str(e)}\n\n")
                        f.write("=" * 80 + "\n\n")
                    break
        
        # Second call: Make predictions using the summary
        try:
            # Prepare few-shot examples with summaries if applicable
            few_shot_with_summaries = None
            if SETTING_FLAGS[setting]["fewshot"] and few_shot_examples:
                few_shot_with_summaries = []
                for ex in few_shot_examples:
                    ex_uid = ex["instance_id"]
                    ex_summary = repo_summaries.get(ex_uid, f"Repository: {ex.get('repo', 'Unknown')}")
                    few_shot_with_summaries.append({
                        "repo_summary": ex_summary,
                        "problem_statement": ex["problem_statement"],
                        "difficulty": ex["difficulty"],
                        "total_prompt_tokens_mean": ex["total_prompt_tokens_mean"],
                        "total_completion_tokens_mean": ex["total_completion_tokens_mean"],
                        "avg_tool_usage_execute_bash": ex["avg_tool_usage_execute_bash"],
                        "avg_tool_usage_str_replace_editor": ex["avg_tool_usage_str_replace_editor"],
                        "avg_tool_usage_think": ex["avg_tool_usage_think"],
                    })
            
            prediction_prompt = build_prediction_prompt(row, setting, repo_summary, few_shot_with_summaries)
            prediction_system_prompt = SYSTEM_PROMPTS[setting]["prediction"]
            
            # Debug: Save input to file
            with open(debug_file, 'a', encoding='utf-8') as f:
                f.write(f"=== INSTANCE {uid} - CALL 2: PREDICTION ===\n")
                f.write(f"System Prompt:\n{prediction_system_prompt}\n\n")
                f.write(f"User Prompt:\n{prediction_prompt}\n\n")
                f.write("=" * 80 + "\n\n")
            
            llm_out = get_llm_response(prediction_prompt, prediction_system_prompt, cfg)
            text_resp = llm_out["text"]
            estimates = parse_estimates(text_resp, expect_tools=SETTING_FLAGS[setting]["tool"])
            preds[uid] = estimates
            raw_outputs[uid] = text_resp
            
            # Debug: Save output to file and print token usage
            with open(debug_file, 'a', encoding='utf-8') as f:
                f.write(f"LLM Response:\n{text_resp}\n\n")
                f.write(f"Parsed Estimates: {estimates}\n\n")
                f.write("=" * 80 + "\n\n")
            
            input_tokens = llm_out["meta"].get("input_tokens", "N/A")
            output_tokens = llm_out["meta"].get("output_tokens", "N/A")
            total_tokens = (input_tokens + output_tokens) if isinstance(input_tokens, int) and isinstance(output_tokens, int) else "N/A"
            print(f"  Made prediction for {uid} - Tokens: {input_tokens} in + {output_tokens} out = {total_tokens} total")
            
        except Exception as e:
            logging.error(f"Failed to make prediction for {uid}: {e}")
            
            # Debug: Save error to file
            with open(debug_file, 'a', encoding='utf-8') as f:
                f.write(f"=== INSTANCE {uid} - CALL 2: ERROR ===\n")
                f.write(f"Error: {str(e)}\n\n")
                f.write("=" * 80 + "\n\n")
            
            continue
        
        pbar.set_postfix({"Completed": len(preds)})
    
    pbar.close()
    
    # Debug: Close the debug file
    print(f"Debug conversation history saved to: {debug_file}")

    # ─── Save predictions csv ─────────────────────────────────────────
    preds_df = pd.DataFrame.from_dict(preds, orient="index")
    preds_df["raw_llm_output"] = pd.Series(raw_outputs)
    preds_df["repo_summary"] = pd.Series(all_summaries)
    
    # Add total tool calls prediction
    preds_df["TOTAL_TOOL_CALLS_PREDICTED"] = (
        preds_df["TOOL_USE_EXECUTE_BASH_ESTIMATE"] + 
        preds_df["TOOL_USE_STR_REPLACE_EDITOR_ESTIMATE"] + 
        preds_df["TOOL_USE_THINK_ESTIMATE"]
    )
    
    # Add ground truth values
    gt_df = test_df.set_index("instance_id")[[
        "total_prompt_tokens_mean",
        "total_completion_tokens_mean",
        "avg_tool_usage_execute_bash",
        "avg_tool_usage_str_replace_editor",
        "avg_tool_usage_think",
        "total_tool_usages_mean",
    ]]
    
    # Join predictions with ground truth
    final_df = preds_df.join(gt_df, how="inner")
    
    preds_csv_path = out_dir / "predictions.csv"
    final_df.to_csv(preds_csv_path, index_label="instance_id")
    print(f"Saved predictions to {preds_csv_path}")

    # ─── Compute metrics & plots ──────────────────────────────────────
    all_metrics = {}
    
    # Token metrics
    for col_gt, col_pred in [
        ("total_prompt_tokens_mean", "INPUT_TOKEN_ESTIMATE"),
        ("total_completion_tokens_mean", "OUTPUT_TOKEN_ESTIMATE"),
    ]:
        y_true = final_df[col_gt].values.astype(float)
        y_pred = final_df[col_pred].values.astype(float)
        metrics = {
            "mape": mape(y_true, y_pred),
            "rmse": rmse(y_true, y_pred),
            "corr": corrcoef(y_true, y_pred),
        }
        all_metrics[col_pred] = metrics

        plot_path = out_dir / f"plot_{col_pred}.png"
        plot_pred_vs_actual(y_true, y_pred, f"{setting} – {col_pred}", plot_path)

    # Tool usage metrics
    for col_gt, col_pred in [
        ("avg_tool_usage_execute_bash", "TOOL_USE_EXECUTE_BASH_ESTIMATE"),
        ("avg_tool_usage_str_replace_editor", "TOOL_USE_STR_REPLACE_EDITOR_ESTIMATE"),
        ("avg_tool_usage_think", "TOOL_USE_THINK_ESTIMATE"),
    ]:
        y_true = final_df[col_gt].values.astype(float)
        y_pred = final_df[col_pred].values.astype(float)
        metrics = {
            "mape": mape(y_true, y_pred),
            "rmse": rmse(y_true, y_pred),
            "corr": corrcoef(y_true, y_pred),
        }
        all_metrics[col_pred] = metrics

        plot_path = out_dir / f"plot_{col_pred}.png"
        plot_pred_vs_actual(y_true, y_pred, f"{setting} – {col_pred}", plot_path)
    
    # Total tool calls metrics
    y_true_total = final_df["total_tool_usages_mean"].values.astype(float)
    y_pred_total = final_df["TOTAL_TOOL_CALLS_PREDICTED"].values.astype(float)
    total_metrics = {
        "mape": mape(y_true_total, y_pred_total),
        "rmse": rmse(y_true_total, y_pred_total),
        "corr": corrcoef(y_true_total, y_pred_total),
    }
    all_metrics["TOTAL_TOOL_CALLS_PREDICTED"] = total_metrics

    plot_path = out_dir / f"plot_TOTAL_TOOL_CALLS_PREDICTED.png"
    plot_pred_vs_actual(y_true_total, y_pred_total, f"{setting} – Total Tool Calls", plot_path)

    # Save metrics
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2))
    print(f"Saved metrics to {metrics_path}")

    logging.info("Finished setting %s → results in %s", setting, out_dir)

###############################################################################
# Entry‑point
###############################################################################

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = Config()

    for setting in cfg.settings:
        logging.info("Running %s", setting)
        run_setting(cfg, setting)

if __name__ == "__main__":
    main() 