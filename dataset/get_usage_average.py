#!/usr/bin/env python3
"""Add run-wise mean usage columns to a token-cost CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List


def to_float(value: str) -> float:
    """Convert to float; invalid values become 0.0."""
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def add_mean_columns(input_file: Path, output_file: Path) -> int:
    """Read input CSV, append mean columns, and write output CSV."""
    csv.field_size_limit(sys.maxsize)

    with input_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header")

        fieldnames: List[str] = list(reader.fieldnames)
        new_cols = [
            "total_prompt_tokens_mean",
            "total_completion_tokens_mean",
            "total_tool_usages_mean",
        ]
        out_fields = fieldnames + [c for c in new_cols if c not in fieldnames]

        rows: List[Dict[str, str]] = []
        for row in reader:
            prompt_vals = [to_float(row.get(f"total_prompt_tokens_run{i}")) for i in range(1, 5)]
            completion_vals = [to_float(row.get(f"total_completion_tokens_run{i}")) for i in range(1, 5)]
            tool_vals = [to_float(row.get(f"total_tool_usages_run{i}")) for i in range(1, 5)]

            row["total_prompt_tokens_mean"] = str(sum(prompt_vals) / 4.0)
            row["total_completion_tokens_mean"] = str(sum(completion_vals) / 4.0)
            row["total_tool_usages_mean"] = str(sum(tool_vals) / 4.0)
            rows.append(row)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add mean usage columns to dataset CSV")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset/swe_bench_token_cost_aggregated_total_with_accuracy.csv"),
        help="Input CSV path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/swe_bench_token_cost_aggregated_total_with_accuracy_col_mean.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    row_count = add_mean_columns(args.input, args.output)
    print("Added columns: total_prompt_tokens_mean, total_completion_tokens_mean, total_tool_usages_mean")
    print(f"Rows processed: {row_count}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
