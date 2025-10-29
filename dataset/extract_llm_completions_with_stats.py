import os
import re
import json
from pathlib import Path

def extract_all_interactions():
    input_root = Path("./sonet_openhands")
    run_name = "claude-3-7-sonnet-20250219_maxiter_100_N_v0.31.0-no-hint-juan-inst-t1-run_4"
    completions_path = input_root / run_name / "llm_completions"
    
    output_root = input_root / f"{run_name}_all_interaction_extract"
    output_root.mkdir(parents=True, exist_ok=True)

    global_summary = {}

    timestamp_pattern = re.compile(r"-(\d+\.\d+)\.json$")

    for instance_dir in completions_path.iterdir():
        if not instance_dir.is_dir():
            continue

        instance_id = instance_dir.name
        output_instance_dir = output_root / instance_id
        output_instance_dir.mkdir(parents=True, exist_ok=True)

        # total_prompt = 0
        # total_completion = 0
        # total_calls = 0

        json_files = list(instance_dir.glob("*.json"))

        if not json_files:
            continue  # skip if no interaction files

        # Sort files using timestamp from filename
        def extract_timestamp(filename):
            match = timestamp_pattern.search(filename.name)
            return float(match.group(1)) if match else float('inf')

        json_files = sorted(json_files, key=extract_timestamp)

        # Aggregated usage stats
        # Initialize aggregates
        usage_agg = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cached_tokens": 0
        }

        total_tool_calls = 0
        # tool_usage_count = 0
        # tool_usage_per_name = {}
        tool_calls_per_interaction = {}
        total_cost = 0.0


        # Process sorted files (oldest to newest)
        # json_files.sort()  # ensure chronological order by filename

        for idx, json_file in enumerate(json_files):
            with open(json_file, "r") as f:
                data = json.load(f)

            # Track interaction filenames
            interaction_filename = f"interaction_{idx+1:02d}__{json_file.name}"

            # Save for tool call counts (declared in assistant response)
            tool_call_count = 0
            tool_call_names = []

            assistant_msg = data.get("response", {}).get("choices", [{}])[0].get("message")
            if assistant_msg:
                tool_calls = assistant_msg.get("tool_calls")
                if isinstance(tool_calls, list):
                    tool_call_count = len(tool_calls)
                    tool_call_names = [call.get("function", {}).get("name", "unknown") for call in tool_calls]

            total_tool_calls += tool_call_count
            tool_calls_per_interaction[interaction_filename] = {
                "count": tool_call_count,
                "tools": tool_call_names
            }

            # Save extracted messages
            messages = data.get("messages", [])
            if assistant_msg:
                messages.append(assistant_msg)

            with open(output_instance_dir / interaction_filename, "w") as f_out:
                json.dump(messages, f_out, indent=2)

            # Aggregate token usage
            usage = data.get("response", {}).get("usage", {})
            usage_agg["prompt_tokens"] += usage.get("prompt_tokens", 0)
            usage_agg["completion_tokens"] += usage.get("completion_tokens", 0)
            usage_agg["total_tokens"] += usage.get("total_tokens", 0)
            usage_agg["cache_creation_input_tokens"] += usage.get("cache_creation_input_tokens", 0)
            usage_agg["cache_read_input_tokens"] += usage.get("cache_read_input_tokens", 0)
            prompt_details = usage.get("prompt_tokens_details") or {}
            usage_agg["cached_tokens"] += prompt_details.get("cached_tokens", 0)
            total_cost += data.get("cost", 0.0)

        final_data = None
        if json_files:
            final_file = json_files[-1]
            with open(final_file, "r") as f:
                final_data = json.load(f)

        tool_usage_count = 0
        tool_usage_per_name = {}

        if final_data:
            for msg in final_data.get("messages", []):
                if msg.get("role") == "tool":
                    tool_usage_count += 1
                    tool_name = msg.get("name", "unknown")
                    tool_usage_per_name[tool_name] = tool_usage_per_name.get(tool_name, 0) + 1

        # Add the final "finish" tool
        tool_usage_count += 1
        tool_usage_per_name["finish"] = tool_usage_per_name.get("finish", 0) + 1

        # Final summary
        instance_summary = {
            "instance_id": instance_id,
            "total_interaction_rounds": len(json_files),
            "total_tool_usages": tool_usage_count,
            "tool_usage_by_name": tool_usage_per_name,
            "tool_calls_declared": total_tool_calls,
            "tool_calls_per_interaction": tool_calls_per_interaction,
            "total_cost": total_cost,
            **usage_agg
        }



        with open(output_instance_dir / "summary.json", "w") as f_summary:
            json.dump(instance_summary, f_summary, indent=2)

        global_summary[instance_id] = instance_summary

    # Global summary
    with open(output_root / "all_instance_summaries.json", "w") as f_global:
        json.dump(global_summary, f_global, indent=2)

    print(f"Extraction complete. Output saved to: {output_root}")

if __name__ == "__main__":
    extract_all_interactions()
