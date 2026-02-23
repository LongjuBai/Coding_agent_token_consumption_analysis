#!/usr/bin/env python3
"""Build a success-subset CSV across multiple models.

A problem instance is in the success subset only if every model has at least one
successful run (`acc_runX == 1`). For each model-instance pair, averages are
computed from successful runs only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

def model_prefix(model_name: str) -> str:
    """Normalize model keys for output column names."""
    return model_name.replace(".", "_").replace("-", "_")


def default_model_paths(dataset_root: Path) -> Dict[str, Path]:
    """Discover model CSV paths under a dataset root."""
    model_paths: Dict[str, Path] = {}
    pattern = "*/swe_bench_token_cost_aggregated_total_with_accuracy.csv"
    for csv_path in sorted(dataset_root.glob(pattern)):
        model_name = csv_path.parent.name
        model_paths[model_name] = csv_path
    return model_paths


def parse_model_paths(values: Iterable[str]) -> Dict[str, Path]:
    """Parse --model-csv args in the form model=path."""
    out: Dict[str, Path] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Invalid --model-csv value '{raw}'. Expected model=path.")
        model, path_str = raw.split("=", 1)
        out[model.strip()] = Path(path_str.strip())
    return out


def successful_run_values(row: pd.Series, value_prefix: str) -> List[float]:
    """Collect values from successful runs for one metric prefix."""
    values: List[float] = []
    for run in range(1, 5):
        acc_key = f"acc_run{run}"
        value_key = f"{value_prefix}_run{run}"
        if acc_key not in row.index or value_key not in row.index:
            continue
        if pd.isna(row[acc_key]) or int(row[acc_key]) != 1:
            continue
        if pd.isna(row[value_key]):
            continue
        try:
            values.append(float(row[value_key]))
        except (TypeError, ValueError):
            continue
    return values


def has_any_success(row: pd.Series) -> bool:
    """Return True if at least one run is successful."""
    for run in range(1, 5):
        key = f"acc_run{run}"
        if key in row.index and not pd.isna(row[key]) and int(row[key]) == 1:
            return True
    return False


def build_success_subset(model_frames: Dict[str, pd.DataFrame], model_names: List[str]) -> List[str]:
    """Return instance ids where all models have at least one successful run."""
    id_sets = [set(df["instance_id"].astype(str).tolist()) for df in model_frames.values()]
    common_ids = set.intersection(*id_sets) if id_sets else set()

    success_ids: List[str] = []
    for instance_id in sorted(common_ids):
        all_success = True
        for model in model_names:
            row = model_frames[model].loc[model_frames[model]["instance_id"].astype(str) == instance_id]
            if row.empty or not has_any_success(row.iloc[0]):
                all_success = False
                break
        if all_success:
            success_ids.append(instance_id)
    return success_ids


def build_output_rows(
    model_frames: Dict[str, pd.DataFrame],
    model_names: List[str],
    success_ids: List[str],
) -> pd.DataFrame:
    """Build output dataframe for success subset."""
    rows = []

    for instance_id in success_ids:
        first_model = model_names[0]
        first_row = model_frames[first_model].loc[
            model_frames[first_model]["instance_id"].astype(str) == instance_id
        ].iloc[0]

        out = {
            "problem_id": instance_id,
            "problem_statement": "",
        }
        if "problem_statement" in first_row.index and not pd.isna(first_row["problem_statement"]):
            out["problem_statement"] = str(first_row["problem_statement"])

        for model in model_names:
            row = model_frames[model].loc[
                model_frames[model]["instance_id"].astype(str) == instance_id
            ].iloc[0]
            pfx = model_prefix(model)

            in_vals = successful_run_values(row, "total_prompt_tokens")
            out_vals = successful_run_values(row, "total_completion_tokens")
            round_vals = successful_run_values(row, "total_interaction_rounds")
            cost_vals = successful_run_values(row, "total_cost")

            out[f"{pfx}_avg_input_token"] = float(np.mean(in_vals)) if in_vals else np.nan
            out[f"{pfx}_avg_output_token"] = float(np.mean(out_vals)) if out_vals else np.nan
            out[f"{pfx}_avg_rounds"] = float(np.mean(round_vals)) if round_vals else np.nan
            out[f"{pfx}_cost"] = float(np.mean(cost_vals)) if cost_vals else np.nan

        rows.append(out)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create success-subset CSV across models")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset"),
        help="Root directory containing per-model subdirectories",
    )
    parser.add_argument(
        "--model-csv",
        action="append",
        default=[],
        help="Optional model CSV override in form model=path (repeatable)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/success_subset_data.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    model_paths = default_model_paths(args.dataset_root)
    if args.model_csv:
        model_paths.update(parse_model_paths(args.model_csv))

    model_names = sorted([m for m, p in model_paths.items() if p.exists()])
    if not model_names:
        raise FileNotFoundError("No model CSVs found. Check --dataset-root or --model-csv.")

    frames: Dict[str, pd.DataFrame] = {}
    for model in model_names:
        df = pd.read_csv(model_paths[model])
        if "instance_id" not in df.columns:
            raise ValueError(f"Missing 'instance_id' column in {model_paths[model]}")
        frames[model] = df

    success_ids = build_success_subset(frames, model_names)
    output_df = build_output_rows(frames, model_names, success_ids)

    ordered_cols: List[str] = ["problem_id", "problem_statement"]
    for model in model_names:
        pfx = model_prefix(model)
        ordered_cols.extend(
            [
                f"{pfx}_avg_input_token",
                f"{pfx}_avg_output_token",
                f"{pfx}_avg_rounds",
                f"{pfx}_cost",
            ]
        )

    output_df = output_df.reindex(columns=ordered_cols)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False)

    print(f"Models used: {', '.join(model_names)}")
    print(f"Success subset size: {len(success_ids)}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
