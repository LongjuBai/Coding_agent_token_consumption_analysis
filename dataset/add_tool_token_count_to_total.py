#!/usr/bin/env python3
"""Add per-tool token-cost aggregates and averages to dataset CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


DEFAULT_RUN_DIR_TEMPLATE = "claude-3-7-sonnet-20250219_maxiter_100_N_v0.31.0-no-hint-juan-inst-t1-run_{i}"
DEFAULT_TOOLS = ["str_replace_editor", "execute_bash", "think"]


def parse_run_ids(raw: str) -> List[int]:
    """Parse comma-separated run ids."""
    out: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    if not out:
        raise ValueError("No run ids provided")
    return out


def parse_tools(raw: str) -> List[str]:
    """Parse comma-separated tool names."""
    tools = [t.strip() for t in raw.split(",") if t.strip()]
    if not tools:
        raise ValueError("No tools provided")
    return tools


def to_float(value: object) -> float:
    """Best-effort conversion to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def add_run_tool_token_columns(
    df: pd.DataFrame,
    extract_base: Path,
    extract_template: str,
    run_ids: List[int],
    tools: List[str],
    summary_filename: str,
) -> None:
    """Populate total_tool_usage_<tool>_token_cost_runX columns."""
    for run_id in run_ids:
        for tool in tools:
            col = f"total_tool_usage_{tool}_token_cost_run{run_id}"
            if col not in df.columns:
                df[col] = 0.0

        extract_root = extract_base / extract_template.format(i=run_id)
        if not extract_root.exists():
            print(f"Extract dir missing for run {run_id}: {extract_root}")
            continue

        for idx, row in df.iterrows():
            instance_id = str(row.get("instance_id", ""))
            if not instance_id:
                continue

            summary_path = extract_root / instance_id / summary_filename
            if not summary_path.exists():
                continue

            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
                rounds = payload.get("rounds", {})
            except Exception:
                continue

            token_totals = {tool: 0.0 for tool in tools}
            for round_data in rounds.values():
                if not isinstance(round_data, dict):
                    continue
                tool_name = round_data.get("tool_executed_name")
                tool_tokens = to_float(round_data.get("tool_output_tokens", 0))
                if tool_name in token_totals:
                    token_totals[tool_name] += tool_tokens

            for tool in tools:
                col = f"total_tool_usage_{tool}_token_cost_run{run_id}"
                df.at[idx, col] = token_totals[tool]


def add_average_columns(df: pd.DataFrame, run_ids: List[int], tools: List[str]) -> None:
    """Add row-wise averages and per-call token-cost columns."""
    for tool in tools:
        token_cols = [f"total_tool_usage_{tool}_token_cost_run{r}" for r in run_ids if f"total_tool_usage_{tool}_token_cost_run{r}" in df.columns]
        call_cols = [f"total_tool_usage_{tool}_run{r}" for r in run_ids if f"total_tool_usage_{tool}_run{r}" in df.columns]

        avg_token_col = f"avg_token_cost_{tool}"
        avg_usage_col = f"avg_tool_usage_{tool}"
        per_call_col = f"avg_token_cost_{tool}_per_call"

        df[avg_token_col] = df[token_cols].mean(axis=1) if token_cols else 0.0
        df[avg_usage_col] = df[call_cols].mean(axis=1) if call_cols else 0.0

        total_tokens = df[token_cols].sum(axis=1) if token_cols else 0.0
        total_calls = df[call_cols].sum(axis=1) if call_cols else 0.0

        if isinstance(total_tokens, (int, float)) or isinstance(total_calls, (int, float)):
            df[per_call_col] = 0.0
        else:
            df[per_call_col] = np.where(total_calls > 0, total_tokens / total_calls, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add tool token-cost columns to dataset")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("swe_bench_token_cost_aggregated_total.csv"),
        help="Input CSV path",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("swe_bench_token_cost_aggregated_total_with_tool_avgs.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--extract-base",
        type=Path,
        default=Path("./sonet_openhands"),
        help="Base directory containing *_all_interaction_extract folders",
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
        "--tools",
        default=",".join(DEFAULT_TOOLS),
        help="Comma-separated tool names",
    )
    parser.add_argument(
        "--summary-filename",
        default="summary_rounds.json",
        help="Summary filename under each instance folder (e.g., summary_rounds.json)",
    )
    args = parser.parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    run_ids = parse_run_ids(args.run_ids)
    tools = parse_tools(args.tools)
    extract_template = args.extract_dir_template or (args.run_dir_template + "_all_interaction_extract")

    df = pd.read_csv(args.input_csv)
    if "instance_id" not in df.columns:
        raise ValueError(f"Missing required column 'instance_id' in {args.input_csv}")

    add_run_tool_token_columns(
        df=df,
        extract_base=args.extract_base,
        extract_template=extract_template,
        run_ids=run_ids,
        tools=tools,
        summary_filename=args.summary_filename,
    )
    add_average_columns(df=df, run_ids=run_ids, tools=tools)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print(f"Wrote {args.output_csv}")
    print("Global average token cost per call:")
    for tool in tools:
        col = f"avg_token_cost_{tool}_per_call"
        if col not in df.columns:
            continue
        valid = df.loc[df[col] > 0, col]
        mean_val = float(valid.mean()) if len(valid) > 0 else 0.0
        print(f"  {tool}: {mean_val:,.1f}")


if __name__ == "__main__":
    main()
