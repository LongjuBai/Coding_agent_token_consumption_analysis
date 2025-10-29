# llm_token_cost_pipeline.py
"""
A modular pipeline for estimating token & tool usage costs with LLMs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple
import json
import logging
import os
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Optional: Anthropic client (swap out for OpenAI, etc.)
try:
    import anthropic
except ImportError:  # library may not be installed in all environments
    anthropic = None  # type: ignore

###############################################################################
# Part 1 ──────────────────────────────────────────────────────────────────────
# Configuration dataclass
###############################################################################


class Setting(str, Enum):
    """The six experimental settings we support."""

    ZERO_PS = "ZeroShot_ProblemStatement"
    ZERO_PS_TOOL = "ZeroShot_ProblemStatement_ToolReasoning"
    ZERO_PS_TOOL_DIFF = "ZeroShot_ProblemStatement_ToolReasoning_Difficulty"
    FEW_PS = "FewShot_ProblemStatement"
    FEW_PS_TOOL = "FewShot_ProblemStatement_ToolReasoning"
    FEW_PS_TOOL_DIFF = "FewShot_ProblemStatement_ToolReasoning_Difficulty"


@dataclass
class Config:
    """Centralised runtime configuration."""

    # Core paths
    dataset_path: Path = Path("./prediction/llm_predictor_exact_number/simplified_with_tool_avgs.csv")
    results_root: Path = Path("./prediction/llm_predictor_exact_number/results")

    # Experiment controls
    sample_size: int = 490  # default N rows sampled for inference
    few_shot_k: int = 4  # how many examples to insert for few‑shot settings
    random_seed: int = 42

    # LLM settings
    model_name: str = "claude-3-7-sonnet-20250219"
    temperature: float = 0.7
    max_tokens: int = 10240
    api_key_env: str = "ANTHROPIC_API_KEY"  # environment var name

    # Settings to run
    settings: List[Setting] = field(
        default_factory=lambda: [
            Setting.ZERO_PS,
            Setting.ZERO_PS_TOOL,
            Setting.ZERO_PS_TOOL_DIFF,
            Setting.FEW_PS,
            Setting.FEW_PS_TOOL,
            Setting.FEW_PS_TOOL_DIFF,
        ]
    )

    # Convenience ─ resolved at runtime
    @property
    def api_key(self) -> str:
        # Try environment variable first
        key = os.getenv(self.api_key_env)
        if key:
            return key
        
        # Fallback to hardcoded key if environment variable not found
        hardcoded_key = ""
        if hardcoded_key:
            return hardcoded_key
            
        raise RuntimeError(f"{self.api_key_env} not found in environment and no hardcoded key available")


###############################################################################
# Part 2 ──────────────────────────────────────────────────────────────────────
# Helper functions (data sampling, metrics, LLM I/O, parsing, plotting)
###############################################################################


# 2.1 ─ Data sampling

def sample_dataframe(df: pd.DataFrame, n: int, exclude_indices: List[int], seed: int) -> pd.DataFrame:
    """Return *n* random rows from *df* excluding given indices (by positional index)."""
    rng = np.random.default_rng(seed)
    available_indices = [i for i in range(len(df)) if i not in exclude_indices]
    if n > len(available_indices):
        raise ValueError("Requested sample larger than population w/o excluded rows.")
    chosen = rng.choice(available_indices, size=n, replace=False)
    return df.iloc[chosen].copy()


# 2.2 ─ Metrics

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-9, None))) * 100.0


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def corrcoef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


# 2.3 ─ LLM response wrapper


def get_llm_response(prompt: str, system_prompt: str, cfg: Config) -> Dict[str, Any]:
    """Call the LLM and return a dict with `text` and cost metadata.

    *Abstracts away* Anthropic‑specific client logic so we can later support
    OpenAI, Together, etc.
    """

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
        # Anthropic Python SDK for messages returns .content list
        text_out: str = response.content[0].text.strip()
        # The SDK provides token usage metadata in different fields per version
        meta = {
            "input_tokens": getattr(response.usage, "input_tokens", None) if hasattr(response, "usage") else None,
            "output_tokens": getattr(response.usage, "output_tokens", None) if hasattr(response, "usage") else None,
            "latency_s": getattr(response, "response_ms", None) and response.response_ms / 1000.0,
        }
        return {"text": text_out, "meta": meta}
    except Exception as exc:  # noqa: BLE001
        logging.exception("LLM call failed")
        raise exc


# 2.4 ─ Prediction parser

TOK_TAGS_TOOL = [
    "INPUT_TOKEN_ESTIMATE",
    "OUTPUT_TOKEN_ESTIMATE",
    "TOOL_USE_EXECUTE_BASH_ESTIMATE",
    "TOOL_USE_STR_REPLACE_EDITOR_ESTIMATE",
    "TOOL_USE_THINK_ESTIMATE",
]

TOK_TAGS_TOKEN_ONLY = [
    "INPUT_TOKEN_ESTIMATE",
    "OUTPUT_TOKEN_ESTIMATE",
]


def parse_estimates(text: str, expect_tools: bool) -> Dict[str, int]:
    """Extract integer estimates wrapped in XML‑style tags.

    Parameters
    ----------
    text
        Raw assistant output.
    expect_tools
        Whether tool‑specific estimates are required.
    """

    tags = TOK_TAGS_TOOL if expect_tools else TOK_TAGS_TOKEN_ONLY
    out: Dict[str, int] = {}
    for tag in tags:
        match = re.search(rf"<{tag}>(.*?)</{tag}>", text)
        if match is None:
            raise ValueError(f"Tag <{tag}> not found in output: {text[:120]}...")
        try:
            # Extract the content and clean it up
            content = match.group(1).strip()
            # Remove any #### or ## prefixes that might be left from the template
            content = re.sub(r'^#+', '', content)
            # Convert to float first (allows decimal values), then round to integer
            out[tag] = int(round(float(content)))
        except ValueError as err:
            raise ValueError(f"Non‑numeric value for <{tag}>: {match.group(1)}") from err
    return out


# 2.5 ─ Sorting + plotting

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
# Prompt builder utilities
###############################################################################

# OpenHands system prompt and tool descriptions
OPENHANDS_SYSTEM_PROMPT = """Your primary role is to assist users by executing commands, modifying code, and solving technical problems effectively. You should be thorough, methodical, and prioritize quality over speed.\n* If the user asks a question, like \"why is X happening\", don't try to fix the problem. Just give an answer to the question.\n\n<EFFICIENCY>\n* Each action you take is somewhat expensive. Wherever possible, combine multiple actions into a single action, e.g. combine multiple bash commands into one, using sed and grep to edit/view multiple files at once.\n* When exploring the codebase, use efficient tools like find, grep, and git commands with appropriate filters to minimize unnecessary operations.\n</EFFICIENCY>\n\n<FILE_SYSTEM_GUIDELINES>\n* When a user provides a file path, do NOT assume it's relative to the current working directory. First explore the file system to locate the file before working on it.\n* If asked to edit a file, edit the file directly, rather than creating a new file with a different filename.\n* For global search-and-replace operations, consider using `sed` instead of opening file editors multiple times.\n</FILE_SYSTEM_GUIDELINES>\n\n<CODE_QUALITY>\n* Write clean, efficient code with minimal comments. Avoid redundancy in comments: Do not repeat information that can be easily inferred from the code itself.\n* When implementing solutions, focus on making the minimal changes needed to solve the problem.\n* Before implementing any changes, first thoroughly understand the codebase through exploration.\n* If you are adding a lot of code to a function or file, consider splitting the function or file into smaller pieces when appropriate.\n</CODE_QUALITY>\n\n<VERSION_CONTROL>\n* When configuring git credentials, use \"openhands\" as the user.name and \"openhands@all-hands.dev\" as the user.email by default, unless explicitly instructed otherwise.\n* Exercise caution with git operations. Do NOT make potentially dangerous changes (e.g., pushing to main, deleting repositories) unless explicitly asked to do so.\n* When committing changes, use `git status` to see all modified files, and stage all files necessary for the commit. Use `git commit -a` whenever possible.\n* Do NOT commit files that typically shouldn't go into version control (e.g., node_modules/, .env files, build directories, cache files, large binaries) unless explicitly instructed by the user.\n* If unsure about committing certain files, check for the presence of .gitignore files or ask the user for clarification.\n</VERSION_CONTROL>\n\n<PULL_REQUESTS>\n* When creating pull requests, create only ONE per session/issue unless explicitly instructed otherwise.\n* When working with an existing PR, update it with new commits rather than creating additional PRs for the same issue.\n* When updating a PR, preserve the original PR title and purpose, updating description only when necessary.\n</PULL_REQUESTS>\n\n<PROBLEM_SOLVING_WORKFLOW>\n1. EXPLORATION: Thoroughly explore relevant files and understand the context before proposing solutions\n2. ANALYSIS: Consider multiple approaches and select the most promising one\n3. TESTING:\n   * For bug fixes: Create tests to verify issues before implementing fixes\n   * For new features: Consider test-driven development when appropriate\n   * If the repository lacks testing infrastructure and implementing tests would require extensive setup, consult with the user before investing time in building testing infrastructure\n4. IMPLEMENTATION: Make focused, minimal changes to address the problem\n5. VERIFICATION: Test your implementation thoroughly, including edge cases\n</PROBLEM_SOLVING_WORKFLOW>\n\n<SECURITY>\n* Only use GITHUB_TOKEN and other credentials in ways the user has explicitly requested and would expect.\n* Use APIs to work with GitHub or other platforms, unless the user asks otherwise or your task requires browsing.\n</SECURITY>\n\n<ENVIRONMENT_SETUP>\n* When user asks you to run an application, don't stop if the application is not installed. Instead, please install the application and run the command again.\n* If you encounter missing dependencies:\n  1. First, look around in the repository for existing dependency files (requirements.txt, pyproject.toml, package.json, Gemfile, etc.)\n  2. If dependency files exist, use them to install all dependencies at once (e.g., `pip install -r requirements.txt`, `npm install`, etc.)\n  3. Only install individual packages directly if no dependency files are found or if only specific packages are needed\n* Similarly, if you encounter missing dependencies for essential tools requested by the user, install them when possible.\n</ENVIRONMENT_SETUP>\n\n<TROUBLESHOOTING>\n* If you've made repeated attempts to solve a problem but tests still fail or the user reports it's still broken:\n  1. Step back and reflect on 5-7 different possible sources of the problem\n  2. Assess the likelihood of each possible cause\n  3. Methodically address the most likely causes, starting with the highest probability\n  4. Document your reasoning process\n* When you run into any major issue while executing a plan from the user, please don't try to directly work around it. Instead, propose a new plan and confirm with the user before proceeding.\n</TROUBLESHOOTING>"""

TOOL_USE_PROMPT = """[{"type": "function", "function": {"name": "execute_bash", "description": "Execute a bash command in the terminal within a persistent shell session.\n\n### Command Execution\n* One command at a time: You can only execute one bash command at a time. If you need to run multiple commands sequentially, use `&&` or `;` to chain them together.\n* Persistent session: Commands execute in a persistent shell session where environment variables, virtual environments, and working directory persist between commands.\n* Timeout: Commands have a soft timeout of 120 seconds, once that's reached, you have the option to continue or interrupt the command (see section below for details)\n\n### Running and Interacting with Processes\n* Long running commands: For commands that may run indefinitely, run them in the background and redirect output to a file, e.g. `python3 app.py > server.log 2>&1 &`.\n* Interact with running process: If a bash command returns exit code `-1`, this means the process is not yet finished. By setting `is_input` to `true`, you can:\n  - Send empty `command` to retrieve additional logs\n  - Send text (set `command` to the text) to STDIN of the running process\n  - Send control commands like `C-c` (Ctrl+C), `C-d` (Ctrl+D), or `C-z` (Ctrl+Z) to interrupt the process\n\n### Best Practices\n* Directory verification: Before creating new directories or files, first verify the parent directory exists and is the correct location.\n* Directory management: Try to maintain working directory by using absolute paths and avoiding excessive use of `cd`.\n\n### Output Handling\n* Output truncation: If the output exceeds a maximum length, it will be truncated before being returned.\n", "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The bash command to execute. Can be empty string to view additional logs when previous exit code is `-1`. Can be `C-c` (Ctrl+C) to interrupt the currently running process. Note: You can only execute one bash command at a time. If you need to run multiple commands sequentially, you can use `&&` or `;` to chain them together."}, "is_input": {"type": "string", "description": "If True, the command is an input to the running process. If False, the command is a bash command to be executed in the terminal. Default is False.", "enum": ["true", "false"]}}, "required": ["command"]}}}, {"type": "function", "function": {"name": "think", "description": "Use the tool to think about something. It will not obtain new information or make any changes to the repository, but just log the thought. Use it when complex reasoning or brainstorming is needed.\n\nCommon use cases:\n1. When exploring a repository and discovering the source of a bug, call this tool to brainstorm several unique ways of fixing the bug, and assess which change(s) are likely to be simplest and most effective.\n2. After receiving test results, use this tool to brainstorm ways to fix failing tests.\n3. When planning a complex refactoring, use this tool to outline different approaches and their tradeoffs.\n4. When designing a new feature, use this tool to think through architecture decisions and implementation details.\n5. When debugging a complex issue, use this tool to organize your thoughts and hypotheses.\n\nThe tool simply logs your thought process for better transparency and does not execute any code or make changes.", "parameters": {"type": "object", "properties": {"thought": {"type": "string", "description": "The thought to log."}}, "required": ["thought"]}}}, {"type": "function", "function": {"name": "finish", "description": "Signals the completion of the current task or conversation.\n\nUse this tool when:\n- You have successfully completed the user's requested task\n- You cannot proceed further due to technical limitations or missing information\n\nThe message should include:\n- A clear summary of actions taken and their results\n- Any next steps for the user\n- Explanation if you're unable to complete the task\n- Any follow-up questions if more information is needed\n\nThe task_completed field should be set to True if you believed you have completed the task, and False otherwise.\n", "parameters": {"type": "object", "required": ["message", "task_completed"], "properties": {"message": {"type": "string", "description": "Final message to send to the user"}, "task_completed": {"type": "string", "enum": ["true", "false", "partial"], "description": "Whether you have completed the task."}}}}}, {"type": "function", "function": {"name": "str_replace_editor", "description": "Custom editing tool for viewing, creating and editing files in plain-text format\n* State is persistent across command calls and discussions with the user\n* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep\n* The `create` command cannot be used if the specified `path` already exists as a file\n* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`\n* The `undo_edit` command will revert the last edit made to the file at `path`\n\n\nBefore using this tool:\n1. Use the view tool to understand the file's contents and context\n2. Verify the directory path is correct (only applicable when creating new files):\n   - Use the view tool to verify the parent directory exists and is the correct location\n\nWhen making edits:\n   - Ensure the edit results in idiomatic, correct code\n   - Do not leave the code in a broken state\n   - Always use absolute file paths (starting with /)\n\nCRITICAL REQUIREMENTS FOR USING THIS TOOL:\n\n1. EXACT MATCHING: The `old_str` parameter must match EXACTLY one or more consecutive lines from the file, including all whitespace and indentation. The tool will fail if `old_str` matches multiple locations or doesn't match exactly with the file content.\n\n2. UNIQUENESS: The `old_str` must uniquely identify a single instance in the file:\n   - Include sufficient context before and after the change point (3-5 lines recommended)\n   - If not unique, the replacement will not be performed\n\n3. REPLACEMENT: The `new_str` parameter should contain the edited lines that replace the `old_str`. Both strings must be different.\n\nRemember: when making multiple file edits in a row to the same file, you should prefer to send all edits in a single message with multiple calls to this tool, rather than multiple messages with a single call each.\n", "parameters": {"type": "object", "properties": {"command": {"description": "The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.", "enum": ["view", "create", "str_replace", "insert", "undo_edit"], "type": "string"}, "path": {"description": "Absolute path to file or directory, e.g. `/workspace/file.py` or `/workspace`.", "type": "string"}, "file_text": {"description": "Required parameter of `create` command, with the content of the file to be created.", "type": "string"}, "old_str": {"description": "Required parameter of `str_replace` command containing the string in `path` to replace.", "type": "string"}, "new_str": {"description": "Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.", "type": "string"}, "insert_line": {"description": "Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.", "type": "integer"}, "view_range": {"description": "Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.", "items": {"type": "integer"}, "type": "array"}}, "required": ["command", "path"]}}}]"""

SYSTEM_PROMPTS: Dict[Setting, str] = {
    Setting.ZERO_PS: """You are **TokenEstimator**, not a coding agent.

ALWAYS do exactly three things, in order:
1. Estimate how many tokens the assistant *would* emit to solve the task shown below. Provide detailed reasoning for your estimates.
2. Output two lines in *this* exact format — and nothing else:
   <INPUT_TOKEN_ESTIMATE>####</INPUT_TOKEN_ESTIMATE>
   <OUTPUT_TOKEN_ESTIMATE>####</OUTPUT_TOKEN_ESTIMATE>
3. Do **NOT** execute the task, do **NOT** call any tool, do **NOT** write code.""",

    Setting.ZERO_PS_TOOL: """You are **TokenEstimator**, not a coding agent.

ALWAYS do exactly three things, in order:
1. Estimate how many tokens the assistant *would* emit and how many tool calls it *would* invoke to solve the task shown below. For each step, provide detailed reasoning for your estimates.  
   Additionally, for tool use, break down how many times *each specific tool* would likely be invoked (e.g., `execute_bash`: 10, `str_replace_editor`: 20, `think`: 5).
2. Output five lines in *this* exact format — and nothing else:
   <INPUT_TOKEN_ESTIMATE>####</INPUT_TOKEN_ESTIMATE>
   <OUTPUT_TOKEN_ESTIMATE>####</OUTPUT_TOKEN_ESTIMATE>
   <TOOL_USE_EXECUTE_BASH_ESTIMATE>##</TOOL_USE_EXECUTE_BASH_ESTIMATE>
   <TOOL_USE_STR_REPLACE_EDITOR_ESTIMATE>##</TOOL_USE_STR_REPLACE_EDITOR_ESTIMATE>
   <TOOL_USE_THINK_ESTIMATE>##</TOOL_USE_THINK_ESTIMATE>
3. Do **NOT** execute the task, do **NOT** call any tool, do **NOT** write code.""",

    Setting.ZERO_PS_TOOL_DIFF: """You are **TokenEstimator**, not a coding agent.

ALWAYS do exactly three things, in order:
1. Estimate how many tokens the assistant *would* emit and how many tool calls it *would* invoke to solve the task shown below, taking into account the difficulty level. For each step, provide detailed reasoning for your estimates.  
   Additionally, for tool use, break down how many times *each specific tool* would likely be invoked (e.g., `execute_bash`: 10, `str_replace_editor`: 20, `think`: 5).
2. Output five lines in *this* exact format — and nothing else:
   <INPUT_TOKEN_ESTIMATE>####</INPUT_TOKEN_ESTIMATE>
   <OUTPUT_TOKEN_ESTIMATE>####</OUTPUT_TOKEN_ESTIMATE>
   <TOOL_USE_EXECUTE_BASH_ESTIMATE>##</TOOL_USE_EXECUTE_BASH_ESTIMATE>
   <TOOL_USE_STR_REPLACE_EDITOR_ESTIMATE>##</TOOL_USE_STR_REPLACE_EDITOR_ESTIMATE>
   <TOOL_USE_THINK_ESTIMATE>##</TOOL_USE_THINK_ESTIMATE>
3. Do **NOT** execute the task, do **NOT** call any tool, do **NOT** write code.""",

    Setting.FEW_PS: """You are **TokenEstimator**, not a coding agent.

ALWAYS do exactly three things, in order:
1. Estimate how many tokens the assistant *would* emit to solve the task shown below. Use the provided examples to inform your estimates and provide detailed reasoning.
2. Output two lines in *this* exact format — and nothing else:
   <INPUT_TOKEN_ESTIMATE>####</INPUT_TOKEN_ESTIMATE>
   <OUTPUT_TOKEN_ESTIMATE>####</OUTPUT_TOKEN_ESTIMATE>
3. Do **NOT** execute the task, do **NOT** call any tool, do **NOT** write code.""",

    Setting.FEW_PS_TOOL: """You are **TokenEstimator**, not a coding agent.

ALWAYS do exactly three things, in order:
1. Estimate how many tokens the assistant *would* emit and how many tool calls it *would* invoke to solve the task shown below. Use the provided examples to inform your estimates and provide detailed reasoning for each step.  
   Additionally, for tool use, break down how many times *each specific tool* would likely be invoked (e.g., `execute_bash`: 10, `str_replace_editor`: 20, `think`: 5).
2. Output five lines in *this* exact format — and nothing else:
   <INPUT_TOKEN_ESTIMATE>####</INPUT_TOKEN_ESTIMATE>
   <OUTPUT_TOKEN_ESTIMATE>####</OUTPUT_TOKEN_ESTIMATE>
   <TOOL_USE_EXECUTE_BASH_ESTIMATE>##</TOOL_USE_EXECUTE_BASH_ESTIMATE>
   <TOOL_USE_STR_REPLACE_EDITOR_ESTIMATE>##</TOOL_USE_STR_REPLACE_EDITOR_ESTIMATE>
   <TOOL_USE_THINK_ESTIMATE>##</TOOL_USE_THINK_ESTIMATE>
3. Do **NOT** execute the task, do **NOT** call any tool, do **NOT** write code.""",

    Setting.FEW_PS_TOOL_DIFF: """You are **TokenEstimator**, not a coding agent.

ALWAYS do exactly three things, in order:
1. Estimate how many tokens the assistant *would* emit and how many tool calls it *would* invoke to solve the task shown below, taking into account the difficulty level. Use the provided examples to inform your estimates and provide detailed reasoning for each step.  
   Additionally, for tool use, break down how many times *each specific tool* would likely be invoked (e.g., `execute_bash`: 10, `str_replace_editor`: 20, `think`: 5).
2. Output five lines in *this* exact format — and nothing else:
   <INPUT_TOKEN_ESTIMATE>####</INPUT_TOKEN_ESTIMATE>
   <OUTPUT_TOKEN_ESTIMATE>####</OUTPUT_TOKEN_ESTIMATE>
   <TOOL_USE_EXECUTE_BASH_ESTIMATE>##</TOOL_USE_EXECUTE_BASH_ESTIMATE>
   <TOOL_USE_STR_REPLACE_EDITOR_ESTIMATE>##</TOOL_USE_STR_REPLACE_EDITOR_ESTIMATE>
   <TOOL_USE_THINK_ESTIMATE>##</TOOL_USE_THINK_ESTIMATE>
3. Do **NOT** execute the task, do **NOT** call any tool, do **NOT** write code.""",
}

# Few‑shot templates.

FEWSHOT_TEMPLATES: Dict[int, str] = {
    1: (
        "For the following task ##{problem_statement}##, "
        "<INPUT_TOKEN_ESTIMATE>####{input_tokens}</INPUT_TOKEN_ESTIMATE>\n"
        "<OUTPUT_TOKEN_ESTIMATE>####{output_tokens}</OUTPUT_TOKEN_ESTIMATE>"
    ),
    2: (
        "For the following task ##{problem_statement}##, "
        "<INPUT_TOKEN_ESTIMATE>####{input_tokens}</INPUT_TOKEN_ESTIMATE>\n"
        "<OUTPUT_TOKEN_ESTIMATE>####{output_tokens}</OUTPUT_TOKEN_ESTIMATE>\n"
        "   <TOOL_USE_EXECUTE_BASH_ESTIMATE>##{bash}</TOOL_USE_EXECUTE_BASH_ESTIMATE>\n"
        "   <TOOL_USE_STR_REPLACE_EDITOR_ESTIMATE>##{replace}</TOOL_USE_STR_REPLACE_EDITOR_ESTIMATE>\n"
        "   <TOOL_USE_THINK_ESTIMATE>##{think}</TOOL_USE_THINK_ESTIMATE>"
    ),
    3: (
        "For the following task ##{problem_statement}## with a difficulty level as {difficulty}, "
        "<INPUT_TOKEN_ESTIMATE>####{input_tokens}</INPUT_TOKEN_ESTIMATE>\n"
        "<OUTPUT_TOKEN_ESTIMATE>####{output_tokens}</OUTPUT_TOKEN_ESTIMATE>\n"
        "   <TOOL_USE_EXECUTE_BASH_ESTIMATE>##{bash}</TOOL_USE_EXECUTE_BASH_ESTIMATE>\n"
        "   <TOOL_USE_STR_REPLACE_EDITOR_ESTIMATE>##{replace}</TOOL_USE_STR_REPLACE_EDITOR_ESTIMATE>\n"
        "   <TOOL_USE_THINK_ESTIMATE>##{think}</TOOL_USE_THINK_ESTIMATE>"
    ),
}


# Mapping from Setting → whether we expect tool reasoning and difficulty label.

SETTING_FLAGS: Dict[Setting, Dict[str, bool]] = {
    Setting.ZERO_PS: {"tool": False, "difficulty": False, "fewshot": False},
    Setting.ZERO_PS_TOOL: {"tool": True, "difficulty": False, "fewshot": False},
    Setting.ZERO_PS_TOOL_DIFF: {"tool": True, "difficulty": True, "fewshot": False},
    Setting.FEW_PS: {"tool": False, "difficulty": False, "fewshot": True},
    Setting.FEW_PS_TOOL: {"tool": True, "difficulty": False, "fewshot": True},
    Setting.FEW_PS_TOOL_DIFF: {"tool": True, "difficulty": True, "fewshot": True},
}


def build_user_prompt(
    row: pd.Series,
    setting: Setting,
    few_shot_examples: List[pd.Series] | None = None,
) -> str:
    """Construct the full *user* prompt based on the setting and optional few‑shot examples."""

    flags = SETTING_FLAGS[setting]
    
    # Build the problem prompt
    if flags["difficulty"]:
        problem_prompt = f"Task: {row.problem_statement}\nDifficulty Level: {row.difficulty}"
    else:
        problem_prompt = f"Task: {row.problem_statement}"
    
    # Build examples section if few-shot
    examples_section = ""
    if flags["fewshot"] and few_shot_examples:
        template_id = 1 if not flags["tool"] else 3 if flags["difficulty"] else 2
        template = FEWSHOT_TEMPLATES[template_id]
        
        examples = []
        for i, ex in enumerate(few_shot_examples, 1):
            example_text = template.format(
                problem_statement=ex["problem_statement"],
                difficulty=ex["difficulty"],
                input_tokens=ex["total_prompt_tokens_mean"],
                output_tokens=ex["total_completion_tokens_mean"],
                bash=ex["avg_tool_usage_execute_bash"],
                replace=ex["avg_tool_usage_str_replace_editor"],
                think=ex["avg_tool_usage_think"],
            )
            examples.append(f"Example {i}:\n{example_text}")
        
        examples_section = "\n\nBelow is a few examples that can help you understand the task and better estimate the token usage:\n\n" + "\n\n".join(examples)
    
    # Build the complete user prompt
    user_prompt = f"""Below is the task description:
{OPENHANDS_SYSTEM_PROMPT}{examples_section}

Below is the task whose cost you must estimate:
{problem_prompt}

Here are the available tools:
{TOOL_USE_PROMPT}

Please provide your estimates in the following format:"""

    # Add the expected output format
    if flags["tool"]:
        user_prompt += """
<INPUT_TOKEN_ESTIMATE>####</INPUT_TOKEN_ESTIMATE>
<OUTPUT_TOKEN_ESTIMATE>####</OUTPUT_TOKEN_ESTIMATE>
   <TOOL_USE_EXECUTE_BASH_ESTIMATE>##</TOOL_USE_EXECUTE_BASH_ESTIMATE>
   <TOOL_USE_STR_REPLACE_EDITOR_ESTIMATE>##</TOOL_USE_STR_REPLACE_EDITOR_ESTIMATE>
   <TOOL_USE_THINK_ESTIMATE>##</TOOL_USE_THINK_ESTIMATE>"""
    else:
        user_prompt += """
<INPUT_TOKEN_ESTIMATE>####</INPUT_TOKEN_ESTIMATE>
<OUTPUT_TOKEN_ESTIMATE>####</OUTPUT_TOKEN_ESTIMATE>"""

    return user_prompt


###############################################################################
# Part 4 ──────────────────────────────────────────────────────────────────────
# Orchestrator: iterate over settings, run predictions, save results
###############################################################################


def run_setting(cfg: Config, setting: Setting) -> None:
    """Run inference for one setting and persist outputs/plots/metrics."""

    # Create output directory
    out_dir = cfg.results_root / setting.value
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(cfg.dataset_path)

    # ─── Few‑shot example selection ────────────────────────────────────
    rng = random.Random(cfg.random_seed)
    few_shot_examples: List[pd.Series] = []
    if SETTING_FLAGS[setting]["fewshot"]:
        # Select one example from each difficulty level
        difficulty_levels = ["<15 min fix", "15 min - 1 hour", "1-4 hours", ">4 hours"]
        exclude_pos = []
        
        for difficulty in difficulty_levels:
            # Filter dataframe for this difficulty level
            difficulty_df = df[df["difficulty"] == difficulty]
            if len(difficulty_df) > 0:
                # Randomly select one example from this difficulty level
                selected_idx = rng.choice(difficulty_df.index)
                few_shot_examples.append(df.iloc[selected_idx])
                exclude_pos.append(selected_idx)
            else:
                logging.warning(f"No examples found for difficulty level: {difficulty}")
    else:
        exclude_pos = []

    # ─── Sampling test rows ───────────────────────────────────────────
    test_df = sample_dataframe(df, cfg.sample_size, exclude_pos, cfg.random_seed)

    # ─── Iterate over test rows ───────────────────────────────────────
    preds: Dict[str, Dict[str, int]] = {}
    raw_outputs: Dict[str, str] = {}

    # Configure tqdm for better display with logging
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
        user_prompt = build_user_prompt(row, setting, few_shot_examples)
        system_prompt = SYSTEM_PROMPTS[setting]

        llm_out = get_llm_response(user_prompt, system_prompt, cfg)
        text_resp = llm_out["text"]
        estimates = parse_estimates(text_resp, expect_tools=SETTING_FLAGS[setting]["tool"])
        preds[uid] = estimates
        raw_outputs[uid] = text_resp
        
        # Update progress bar description with current progress
        pbar.set_postfix({"Completed": len(preds)})
    
    pbar.close()

    # ─── Save predictions csv ─────────────────────────────────────────
    preds_df = pd.DataFrame.from_dict(preds, orient="index")
    # Add raw output column
    preds_df["raw_llm_output"] = pd.Series(raw_outputs)
    
    # Add total tool calls prediction for tool reasoning settings
    if SETTING_FLAGS[setting]["tool"]:
        preds_df["TOTAL_TOOL_CALLS_PREDICTED"] = (
            preds_df["TOOL_USE_EXECUTE_BASH_ESTIMATE"] + 
            preds_df["TOOL_USE_STR_REPLACE_EDITOR_ESTIMATE"] + 
            preds_df["TOOL_USE_THINK_ESTIMATE"]
        )
    
    preds_csv_path = out_dir / "predictions.csv"
    preds_df.to_csv(preds_csv_path, index_label="instance_id")

    # ─── Compute metrics & plots ──────────────────────────────────────
    # Token metrics
    gt = test_df.set_index("instance_id")[[
        "total_prompt_tokens_mean",
        "total_completion_tokens_mean",
    ]]
    merged = gt.join(preds_df, how="inner")
    
    # Collect all metrics in one dictionary
    all_metrics = {}
    
    # Token metrics
    for col_gt, col_pred in [
        ("total_prompt_tokens_mean", "INPUT_TOKEN_ESTIMATE"),
        ("total_completion_tokens_mean", "OUTPUT_TOKEN_ESTIMATE"),
    ]:
        y_true = merged[col_gt].values.astype(float)
        y_pred = merged[col_pred].values.astype(float)
        metrics = {
            "mape": mape(y_true, y_pred),
            "rmse": rmse(y_true, y_pred),
            "corr": corrcoef(y_true, y_pred),
        }
        all_metrics[col_pred] = metrics

        plot_path = out_dir / f"plot_{col_pred}.png"
        plot_pred_vs_actual(y_true, y_pred, f"{setting} – {col_pred}", plot_path)

    # Tool usage metrics (only for settings that predict tools)
    if SETTING_FLAGS[setting]["tool"]:
        tool_gt = test_df.set_index("instance_id")[[
            "avg_tool_usage_execute_bash",
            "avg_tool_usage_str_replace_editor", 
            "avg_tool_usage_think",
            "total_tool_usages_mean",  # Add total tool usages for comparison
        ]]
        tool_merged = tool_gt.join(preds_df, how="inner")
        
        # Calculate total predicted tool calls
        tool_merged["TOTAL_TOOL_CALLS_PREDICTED"] = (
            tool_merged["TOOL_USE_EXECUTE_BASH_ESTIMATE"] + 
            tool_merged["TOOL_USE_STR_REPLACE_EDITOR_ESTIMATE"] + 
            tool_merged["TOOL_USE_THINK_ESTIMATE"]
        )
        
        for col_gt, col_pred in [
            ("avg_tool_usage_execute_bash", "TOOL_USE_EXECUTE_BASH_ESTIMATE"),
            ("avg_tool_usage_str_replace_editor", "TOOL_USE_STR_REPLACE_EDITOR_ESTIMATE"),
            ("avg_tool_usage_think", "TOOL_USE_THINK_ESTIMATE"),
        ]:
            y_true = tool_merged[col_gt].values.astype(float)
            y_pred = tool_merged[col_pred].values.astype(float)
            metrics = {
                "mape": mape(y_true, y_pred),
                "rmse": rmse(y_true, y_pred),
                "corr": corrcoef(y_true, y_pred),
            }
            all_metrics[col_pred] = metrics

            plot_path = out_dir / f"plot_{col_pred}.png"
            plot_pred_vs_actual(y_true, y_pred, f"{setting} – {col_pred}", plot_path)
        
        # Calculate metrics for total tool calls
        y_true_total = tool_merged["total_tool_usages_mean"].values.astype(float)
        y_pred_total = tool_merged["TOTAL_TOOL_CALLS_PREDICTED"].values.astype(float)
        total_metrics = {
            "mape": mape(y_true_total, y_pred_total),
            "rmse": rmse(y_true_total, y_pred_total),
            "corr": corrcoef(y_true_total, y_pred_total),
        }
        all_metrics["TOTAL_TOOL_CALLS_PREDICTED"] = total_metrics

        # Plot for total tool calls
        plot_path = out_dir / f"plot_TOTAL_TOOL_CALLS_PREDICTED.png"
        plot_pred_vs_actual(y_true_total, y_pred_total, f"{setting} – Total Tool Calls", plot_path)

    # Save all metrics in one JSON file
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2))

    logging.info("Finished setting %s → results in %s", setting, out_dir)


###############################################################################
# Entry‑point helper
###############################################################################


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = Config()

    for setting in cfg.settings:
        logging.info("Running %s", setting)
        run_setting(cfg, setting)


if __name__ == "__main__":
    main()
