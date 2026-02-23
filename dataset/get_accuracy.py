#!/usr/bin/env python3
"""Add run-level accuracy columns using report.json resolved_ids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd


DEFAULT_RUN_DIR_TEMPLATE = "claude-3-7-sonnet-20250219_maxiter_100_N_v0.31.0-no-hint-juan-inst-t1-run_{i}"


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


def load_resolved_ids(report_path: Path) -> Set[str]:
    """Load resolved ids from one report.json file."""
    if not report_path.exists():
        return set()

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return set()

    resolved = payload.get("resolved_ids", [])
    if not isinstance(resolved, list):
        return set()
    return {str(x) for x in resolved}


def build_report_paths(base: Path, run_template: str, run_ids: List[int]) -> Dict[int, Path]:
    """Build report.json paths for all runs."""
    return {
        run_id: base / run_template.format(i=run_id) / "report.json"
        for run_id in run_ids
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Add accuracy columns from report.json files")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("swe_bench_token_cost_aggregated_total.csv"),
        help="Input CSV path",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("swe_bench_token_cost_aggregated_total_with_accuracy.csv"),
        help="Output CSV path",
    )
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
        "--run-ids",
        default="1,2,3,4",
        help="Comma-separated run ids",
    )
    args = parser.parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    run_ids = parse_run_ids(args.run_ids)
    report_paths = build_report_paths(args.base, args.run_dir_template, run_ids)

    df = pd.read_csv(args.input_csv)
    if "instance_id" not in df.columns:
        raise ValueError(f"Missing required column 'instance_id' in {args.input_csv}")

    for run_id in run_ids:
        resolved_ids = load_resolved_ids(report_paths[run_id])
        col = f"acc_run{run_id}"
        df[col] = df["instance_id"].astype(str).isin(resolved_ids).astype(int)

        resolved_count = int(df[col].sum())
        total = len(df)
        rate = (resolved_count / total * 100.0) if total > 0 else 0.0
        print(f"{col}: {resolved_count}/{total} ({rate:.2f}%)")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
