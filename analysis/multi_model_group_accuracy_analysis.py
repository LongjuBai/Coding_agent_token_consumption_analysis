#!/usr/bin/env python3
"""Multi-model grouped accuracy analysis.

This script aggregates grouped cost/accuracy statistics across models and
(optionally) fits mixed-effects models:

  group_accuracy ~ standardized_group_cost + (1 | model)

No visualization is generated; outputs are CSV tables only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    from statsmodels.regression.mixed_linear_model import MixedLM
except ImportError:  # pragma: no cover
    MixedLM = None

ACC_COLS = [f"acc_run{i}" for i in range(1, 5)]
METRIC_CONFIG = {
    "Total Prompt Tokens": [f"total_prompt_tokens_run{i}" for i in range(1, 5)],
    "Total Completion Tokens": [f"total_completion_tokens_run{i}" for i in range(1, 5)],
    "Total Tool Use": [f"total_tool_usages_run{i}" for i in range(1, 5)],
}


def default_model_paths(dataset_root: Path) -> Dict[str, Path]:
    """Discover per-model CSV paths from dataset root."""
    model_paths: Dict[str, Path] = {}
    pattern = "*/swe_bench_token_cost_aggregated_total_with_accuracy.csv"
    for csv_path in sorted(dataset_root.glob(pattern)):
        model_name = csv_path.parent.name
        model_paths[model_name] = csv_path
    return model_paths


def parse_model_paths(values: Iterable[str]) -> Dict[str, Path]:
    """Parse model=path overrides."""
    out: Dict[str, Path] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Invalid --model-csv value '{raw}'. Expected model=path.")
        model, path_str = raw.split("=", 1)
        out[model.strip()] = Path(path_str.strip())
    return out


def load_model_frames(model_paths: Dict[str, Path]) -> Dict[str, pd.DataFrame]:
    """Load available model CSVs."""
    frames: Dict[str, pd.DataFrame] = {}
    for model, path in sorted(model_paths.items()):
        if not path.exists():
            continue
        df = pd.read_csv(path)
        missing = [col for col in ACC_COLS if col not in df.columns]
        if missing:
            raise ValueError(f"{path} missing accuracy columns: {missing}")
        df = df.copy()
        df["model"] = model
        frames[model] = df
    return frames


def grouped_rows_for_metric(
    model_name: str,
    df_model: pd.DataFrame,
    metric_cols: List[str],
    group_size: int,
    n_groups: int,
) -> List[Dict[str, float]]:
    """Build grouped cost/accuracy rows for one model and one metric."""
    values = df_model[metric_cols].astype(float)

    lower = values.stack().quantile(0.01)
    upper = values.stack().quantile(0.99)
    values_clipped = values.clip(lower=lower, upper=upper)

    mean_per_instance = values_clipped.mean(axis=1)
    sorted_idx = mean_per_instance.argsort()
    sorted_mean = mean_per_instance.iloc[sorted_idx].reset_index(drop=True)

    rows: List[Dict[str, float]] = []
    for group in range(n_groups):
        start = group * group_size
        end = (group + 1) * group_size
        if end > len(sorted_idx):
            continue

        selected = sorted_idx[start:end]
        group_acc = df_model[ACC_COLS].iloc[selected].values.flatten().astype(float)
        group_accuracy = float(np.mean(group_acc))
        group_cost = float(sorted_mean[start:end].mean())

        rows.append(
            {
                "model": model_name,
                "group": group,
                "group_cost": group_cost,
                "group_accuracy": group_accuracy,
                "n_instances": int(len(selected)),
            }
        )

    return rows


def build_group_dataframe(
    frames: Dict[str, pd.DataFrame],
    group_size: int,
    n_groups: int,
) -> pd.DataFrame:
    """Create full grouped dataframe for all metrics and models."""
    rows: List[Dict[str, float]] = []

    for metric_name, metric_cols in METRIC_CONFIG.items():
        models = list(frames.keys())

        for model in models:
            if not all(col in frames[model].columns for col in metric_cols):
                continue
            metric_rows = grouped_rows_for_metric(
                model,
                frames[model],
                metric_cols,
                group_size=group_size,
                n_groups=n_groups,
            )
            for row in metric_rows:
                row["metric"] = metric_name
            rows.extend(metric_rows)

    return pd.DataFrame(rows)


def fit_mixed_effects(group_df: pd.DataFrame) -> pd.DataFrame:
    """Fit one mixed model per metric if statsmodels is installed."""
    summary_rows: List[Dict[str, str]] = []

    for metric_name in sorted(group_df["metric"].unique()):
        sub = group_df[group_df["metric"] == metric_name].copy()
        if sub.empty:
            continue

        cost_mean = float(sub["group_cost"].mean())
        cost_std = float(sub["group_cost"].std())
        if cost_std == 0 or np.isnan(cost_std):
            summary_rows.append(
                {
                    "metric": metric_name,
                    "n_rows": str(len(sub)),
                    "fixed_effect_std": "",
                    "fixed_effect_original_scale": "",
                    "intercept": "",
                    "p_value": "",
                    "status": "skipped_zero_variance",
                }
            )
            continue

        sub["cost_std"] = (sub["group_cost"] - cost_mean) / cost_std

        if MixedLM is None:
            summary_rows.append(
                {
                    "metric": metric_name,
                    "n_rows": str(len(sub)),
                    "fixed_effect_std": "",
                    "fixed_effect_original_scale": "",
                    "intercept": "",
                    "p_value": "",
                    "status": "skipped_statsmodels_not_installed",
                }
            )
            continue

        try:
            model = MixedLM.from_formula("group_accuracy ~ cost_std", sub, groups=sub["model"])
            result = model.fit(reml=True)

            slope_std = float(result.fe_params.get("cost_std", np.nan))
            slope_orig = slope_std / cost_std
            intercept = float(result.fe_params.get("Intercept", np.nan))
            p_value = float(result.pvalues.get("cost_std", np.nan))

            summary_rows.append(
                {
                    "metric": metric_name,
                    "n_rows": str(len(sub)),
                    "fixed_effect_std": f"{slope_std:.8f}",
                    "fixed_effect_original_scale": f"{slope_orig:.12f}",
                    "intercept": f"{intercept:.8f}",
                    "p_value": f"{p_value:.8g}",
                    "status": "ok",
                }
            )
        except Exception as exc:  # pragma: no cover
            summary_rows.append(
                {
                    "metric": metric_name,
                    "n_rows": str(len(sub)),
                    "fixed_effect_std": "",
                    "fixed_effect_original_scale": "",
                    "intercept": "",
                    "p_value": "",
                    "status": f"fit_failed: {exc}",
                }
            )

    return pd.DataFrame(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grouped multi-model accuracy analysis")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset"),
        help="Root directory containing model subdirectories",
    )
    parser.add_argument(
        "--model-csv",
        action="append",
        default=[],
        help="Optional model CSV override in form model=path (repeatable)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=25,
        help="Instances per group",
    )
    parser.add_argument(
        "--n-groups",
        type=int,
        default=20,
        help="Maximum groups per model",
    )
    parser.add_argument(
        "--group-output",
        type=Path,
        default=Path("analysis/multi_model_group_accuracy_groups.csv"),
        help="Output CSV for grouped rows",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("analysis/multi_model_group_accuracy_mixed_effects.csv"),
        help="Output CSV for mixed-effects summary",
    )
    args = parser.parse_args()

    model_paths = default_model_paths(args.dataset_root)
    if args.model_csv:
        model_paths.update(parse_model_paths(args.model_csv))

    frames = load_model_frames(model_paths)
    if not frames:
        raise FileNotFoundError("No model CSV files found")

    group_df = build_group_dataframe(frames, args.group_size, args.n_groups)
    if group_df.empty:
        raise ValueError("No grouped rows were created. Check input columns and files.")

    summary_df = fit_mixed_effects(group_df)

    args.group_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    group_df.to_csv(args.group_output, index=False)
    summary_df.to_csv(args.summary_output, index=False)

    print(f"Models loaded: {', '.join(frames.keys())}")
    print(f"Grouped rows: {len(group_df)}")
    print(f"Wrote {args.group_output}")
    print(f"Wrote {args.summary_output}")


if __name__ == "__main__":
    main()
