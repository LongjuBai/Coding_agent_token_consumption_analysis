#!/usr/bin/env python
"""
Augment per-interaction stats and write summary_rounds_withCache.json next to summary.json

The script expects the same directory layout you used earlier:

    <BASE>/claude-…-run_{i}
    <BASE>/claude-…-run_{i}_all_interaction_extract
"""

import json
import math
import re
from collections import defaultdict
from pathlib import Path

# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------
BASE = Path("./sonet_openhands")
RUN_IDS = range(1, 5)  # runs 1 … 4

RUN_DIR_TMPL = (
    "claude-3-7-sonnet-20250219_maxiter_100_N_v0.31.0-no-hint-juan-inst-t1-run_{i}"
)
EXTRACT_DIR_TMPL = RUN_DIR_TMPL + "_all_interaction_extract"
# --------------------------------------------------------------------


def count_tokens(text: str) -> int:
    """
    Rough but safer token estimator:

    1. regex `\\S+`   → word/symbol chunks
    2. len(text)/4     (≈ OpenAI rule-of-thumb 4 chars per token)
    Return the *larger* estimate to avoid under-counting.
    """
    if not text:
        return 0
    word_like = len(re.findall(r"\S+", text))
    char_est = math.ceil(len(text) / 4.0)
    return max(word_like, char_est)


def latest_tool_message(messages):
    """
    Return the *last* message whose role == "tool" **excluding** the 'finish'
    bookkeeping tool.  If none, return None.
    """
    for msg in reversed(messages):
        if msg.get("role") == "tool":
            if msg.get("name") and msg["name"] != "finish":
                return msg
    return None


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
for run_id in RUN_IDS:
    run_dir = BASE / RUN_DIR_TMPL.format(i=run_id)
    extract_dir = BASE / EXTRACT_DIR_TMPL.format(i=run_id)

    if not run_dir.exists() or not extract_dir.exists():
        print(f"Run {run_id}: expected dirs missing – skipping")
        continue

    print(f"\n▶ Processing run {run_id}")

    # ----------------------------------------------------------------
    # iterate over every problem instance
    # ----------------------------------------------------------------
    for instance_dir in sorted(extract_dir.iterdir()):
        if not instance_dir.is_dir():
            continue

        summary_file = instance_dir / "summary.json"
        if not summary_file.exists():
            print(f"{summary_file.relative_to(BASE)} missing – skipped")
            continue

        with summary_file.open() as fh:
            summary = json.load(fh)

        instance_id = summary["instance_id"]
        rounds_info = summary["tool_calls_per_interaction"]

        orig_dir = run_dir / "llm_completions" / instance_id
        if not orig_dir.exists():
            print(f"original completions missing for {instance_id} – skipped")
            continue

        # ---------- sort keys numerically (interaction_01 …) ----------
        interactions_sorted = sorted(
            rounds_info.keys(),
            key=lambda k: int(re.search(r"interaction_(\d+)", k).group(1)),
        )

        enhanced_rounds = {}

        # ---------- walk through each interaction ----------
        for interaction_key in interactions_sorted:
            base_name = interaction_key.split("__", 1)[1]
            this_json = orig_dir / base_name

            prompt_tokens = completion_tokens = 0
            cost_val = 0.0 
            tool_output_tokens = 0
            tool_executed_name = "none"
            
            # New cache-related token counts
            cached_tokens = 0
            cache_creation_input_tokens = 0
            cache_read_input_tokens = 0

            if this_json.exists():
                try:
                    with this_json.open() as fh:
                        data_this = json.load(fh)

                    # usage from assistant completion
                    usage = data_this.get("response", {}).get("usage", {})
                    cost_val = data_this.get("cost", 0.0)
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    
                    # Extract cache-related token counts
                    prompt_tokens_details = usage.get("prompt_tokens_details", {})
                    cached_tokens = prompt_tokens_details.get("cached_tokens", 0)
                    cache_creation_input_tokens = usage.get("cache_creation_input_tokens", 0)
                    cache_read_input_tokens = usage.get("cache_read_input_tokens", 0)

                    # look for *last* tool message in this file
                    tmsg = latest_tool_message(data_this.get("messages", []))
                    if tmsg:
                        tool_executed_name = tmsg.get("name", "unknown_tool")
                        content = tmsg.get("content", "")
                        if isinstance(content, str):
                            tool_output_tokens = count_tokens(content)
                        else:  # list of blocks
                            joined = "\n".join(
                                blk.get("text", "")
                                for blk in content
                                if blk.get("type") == "text"
                            )
                            tool_output_tokens = count_tokens(joined)

                except Exception as exc:
                    print(f"cannot parse {this_json.name}: {exc}")

            # assemble record
            enhanced_rounds[interaction_key] = {
                **rounds_info[interaction_key],  # count + tools list
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost": cost_val,
                "tool_output_tokens": tool_output_tokens,
                "tool_executed_name": tool_executed_name,
                # New cache-related fields
                "cached_tokens": cached_tokens,
                "cache_creation_input_tokens": cache_creation_input_tokens,
                "cache_read_input_tokens": cache_read_input_tokens,
            }

        # ---------- write summary_rounds_withCache.json ----------
        out_file = instance_dir / "summary_rounds_withCache.json"
        with out_file.open("w") as fh:
            json.dump(
                {
                    "instance_id": instance_id,
                    "run_id": run_id,
                    "rounds": enhanced_rounds,
                },
                fh,
                indent=2,
            )
        print(f"wrote {out_file.relative_to(BASE)}")

print("\nAll runs finished")
