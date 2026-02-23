#!/usr/bin/env python3
"""Generate per-interaction round summaries from raw completion JSONs.

This unifies the old `rounds_analysis` and `rounds_analysis_withCache` logic.

Output modes:
- `basic`: write `summary_rounds.json`
- `cache`: write `summary_rounds_withCache.json`
- `both`: write both files
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DEFAULT_RUN_DIR_TEMPLATE = "claude-3-7-sonnet-20250219_maxiter_100_N_v0.31.0-no-hint-juan-inst-t1-run_{i}"


def parse_run_ids(raw: str) -> List[int]:
    """Parse comma-separated run ids."""
    out = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    if not out:
        raise ValueError("No run ids provided")
    return out


def count_tokens(text: str) -> int:
    """Rough token estimator using max(word-like chunks, chars/4)."""
    if not text:
        return 0
    word_like = len(re.findall(r"\S+", text))
    char_est = math.ceil(len(text) / 4.0)
    return max(word_like, char_est)


def latest_tool_message(messages: Iterable[dict]) -> Optional[dict]:
    """Return last non-finish tool message."""
    for msg in reversed(list(messages)):
        if msg.get("role") == "tool" and msg.get("name") and msg.get("name") != "finish":
            return msg
    return None


def interaction_sort_key(key: str) -> int:
    """Extract numeric interaction index for sorting."""
    match = re.search(r"interaction_(\d+)", key)
    if not match:
        return 10**9
    return int(match.group(1))


def parse_interaction_json(path: Path) -> Dict[str, object]:
    """Extract interaction-level metrics from one completion JSON file."""
    defaults: Dict[str, object] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost": 0.0,
        "tool_output_tokens": 0,
        "tool_executed_name": "none",
        "cached_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }

    if not path.exists():
        return defaults

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return defaults

    usage = (data.get("response") or {}).get("usage", {})
    prompt_details = usage.get("prompt_tokens_details", {}) if isinstance(usage, dict) else {}

    try:
        defaults["cost"] = float(data.get("cost", 0.0) or 0.0)
    except (TypeError, ValueError):
        defaults["cost"] = 0.0

    defaults["prompt_tokens"] = int(usage.get("prompt_tokens", 0) or 0)
    defaults["completion_tokens"] = int(usage.get("completion_tokens", 0) or 0)
    defaults["cached_tokens"] = int(prompt_details.get("cached_tokens", 0) or 0)
    defaults["cache_creation_input_tokens"] = int(usage.get("cache_creation_input_tokens", 0) or 0)
    defaults["cache_read_input_tokens"] = int(usage.get("cache_read_input_tokens", 0) or 0)

    tmsg = latest_tool_message(data.get("messages", []))
    if tmsg:
        defaults["tool_executed_name"] = tmsg.get("name", "unknown_tool")
        content = tmsg.get("content", "")

        if isinstance(content, str):
            defaults["tool_output_tokens"] = count_tokens(content)
        elif isinstance(content, list):
            joined = "\n".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
            defaults["tool_output_tokens"] = count_tokens(joined)

    return defaults


def build_enhanced_rounds(rounds_info: Dict[str, dict], orig_dir: Path, include_cache: bool) -> Dict[str, dict]:
    """Build augmented rounds dictionary for one instance."""
    enhanced: Dict[str, dict] = {}
    interactions = sorted(rounds_info.keys(), key=interaction_sort_key)

    for interaction_key in interactions:
        base_name = interaction_key.split("__", 1)[1] if "__" in interaction_key else interaction_key
        interaction_path = orig_dir / base_name
        parsed = parse_interaction_json(interaction_path)

        row = {
            **rounds_info[interaction_key],
            "prompt_tokens": parsed["prompt_tokens"],
            "completion_tokens": parsed["completion_tokens"],
            "cost": parsed["cost"],
            "tool_output_tokens": parsed["tool_output_tokens"],
            "tool_executed_name": parsed["tool_executed_name"],
        }

        if include_cache:
            row.update(
                {
                    "cached_tokens": parsed["cached_tokens"],
                    "cache_creation_input_tokens": parsed["cache_creation_input_tokens"],
                    "cache_read_input_tokens": parsed["cache_read_input_tokens"],
                }
            )

        enhanced[interaction_key] = row

    return enhanced


def write_summary(instance_dir: Path, instance_id: str, run_id: int, rounds: Dict[str, dict], include_cache: bool) -> Path:
    """Write one summary file and return output path."""
    output_name = "summary_rounds_withCache.json" if include_cache else "summary_rounds.json"
    out_file = instance_dir / output_name

    out_file.write_text(
        json.dumps(
            {
                "instance_id": instance_id,
                "run_id": run_id,
                "rounds": rounds,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate per-interaction round summaries")
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("./sonet_openhands"),
        help="Base directory containing run folders",
    )
    parser.add_argument(
        "--run-dir-template",
        default=DEFAULT_RUN_DIR_TEMPLATE,
        help="Template for run folders, must contain '{i}'",
    )
    parser.add_argument(
        "--extract-dir-template",
        default="",
        help="Template for extract folders; default is run_dir_template + '_all_interaction_extract'",
    )
    parser.add_argument(
        "--run-ids",
        default="1,2,3,4",
        help="Comma-separated run ids",
    )
    parser.add_argument(
        "--mode",
        choices=["basic", "cache", "both"],
        default="both",
        help="Output mode",
    )
    args = parser.parse_args()

    run_ids = parse_run_ids(args.run_ids)
    extract_template = args.extract_dir_template or (args.run_dir_template + "_all_interaction_extract")

    for run_id in run_ids:
        run_dir = args.base / args.run_dir_template.format(i=run_id)
        extract_dir = args.base / extract_template.format(i=run_id)

        if not run_dir.exists() or not extract_dir.exists():
            print(f"Run {run_id}: expected dirs missing, skipping")
            continue

        print(f"\\nProcessing run {run_id}")

        for instance_dir in sorted(extract_dir.iterdir()):
            if not instance_dir.is_dir():
                continue

            summary_file = instance_dir / "summary.json"
            if not summary_file.exists():
                continue

            try:
                summary = json.loads(summary_file.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"Failed to read {summary_file}: {exc}")
                continue

            instance_id = summary.get("instance_id")
            rounds_info = summary.get("tool_calls_per_interaction", {})
            if not instance_id or not isinstance(rounds_info, dict):
                continue

            orig_dir = run_dir / "llm_completions" / instance_id
            if not orig_dir.exists():
                continue

            need_basic = args.mode in {"basic", "both"}
            need_cache = args.mode in {"cache", "both"}

            if need_basic:
                rounds_basic = build_enhanced_rounds(rounds_info, orig_dir, include_cache=False)
                out = write_summary(instance_dir, instance_id, run_id, rounds_basic, include_cache=False)
                print(f"  wrote {out}")

            if need_cache:
                rounds_cache = build_enhanced_rounds(rounds_info, orig_dir, include_cache=True)
                out = write_summary(instance_dir, instance_id, run_id, rounds_cache, include_cache=True)
                print(f"  wrote {out}")

    print("\\nAll runs finished")


if __name__ == "__main__":
    main()
