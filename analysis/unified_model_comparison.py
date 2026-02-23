#!/usr/bin/env python3
"""Build a unified tabular comparison across models.

This script computes core metrics without drawing figures:
1. Mean total tokens and mean total cost over all instances/runs
2. Mean total tokens and mean total cost over success subset runs only
3. Self-prediction correlations and cost-ratio metrics (if prediction CSVs exist)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

def safe_float(value: object) -> Optional[float]:
    """Best-effort numeric conversion."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def corr(x: List[float], y: List[float]) -> Optional[float]:
    """Pearson correlation with defensive checks."""
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return None
    if len(set(x)) <= 1 or len(set(y)) <= 1:
        return None
    c = float(np.corrcoef(np.array(x), np.array(y))[0, 1])
    if np.isnan(c):
        return None
    return c


def parse_model_paths(values: Iterable[str]) -> Dict[str, Path]:
    """Parse model=path pairs."""
    out: Dict[str, Path] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Invalid --model-csv value '{raw}'. Expected model=path.")
        model, path_str = raw.split("=", 1)
        out[model.strip()] = Path(path_str.strip())
    return out


def default_model_paths(dataset_root: Path) -> Dict[str, Path]:
    """Discover per-model token/cost CSV paths from dataset root."""
    model_paths: Dict[str, Path] = {}
    pattern = "*/swe_bench_token_cost_aggregated_total_with_accuracy.csv"
    for csv_path in sorted(dataset_root.glob(pattern)):
        model_name = csv_path.parent.name
        model_paths[model_name] = csv_path
    return model_paths


def normalize_name(text: str) -> str:
    """Normalize model/file names for loose matching."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def find_prediction_csv(prediction_root: Path, model_name: str) -> Path:
    """Find the best-matching *_predictions.csv for a model name."""
    candidates = sorted(prediction_root.glob("*_predictions.csv"))
    model_norm = normalize_name(model_name)

    for path in candidates:
        stem_norm = normalize_name(path.stem.replace("_predictions", ""))
        if stem_norm == model_norm:
            return path

    for path in candidates:
        stem_norm = normalize_name(path.stem)
        if model_norm in stem_norm:
            return path

    return prediction_root / f"{model_name}_predictions.csv"


def has_any_success(row: pd.Series) -> bool:
    """True if row has at least one acc_runX == 1."""
    for run in range(1, 5):
        key = f"acc_run{run}"
        if key in row.index and pd.notna(row[key]) and int(row[key]) == 1:
            return True
    return False


def build_success_subset(frames: Dict[str, pd.DataFrame], model_names: List[str]) -> List[str]:
    """Instances where all models have at least one success."""
    id_sets = [set(df["instance_id"].astype(str).tolist()) for df in frames.values()]
    common_ids = set.intersection(*id_sets) if id_sets else set()

    out: List[str] = []
    for instance_id in sorted(common_ids):
        ok = True
        for model in model_names:
            row = frames[model].loc[frames[model]["instance_id"].astype(str) == instance_id]
            if row.empty or not has_any_success(row.iloc[0]):
                ok = False
                break
        if ok:
            out.append(instance_id)
    return out


def mean_total_tokens_all(df: pd.DataFrame) -> float:
    """Mean over all non-zero run-level total tokens."""
    vals: List[float] = []
    for run in range(1, 5):
        p = f"total_prompt_tokens_run{run}"
        c = f"total_completion_tokens_run{run}"
        if p not in df.columns or c not in df.columns:
            continue
        total = pd.to_numeric(df[p], errors="coerce").fillna(0) + pd.to_numeric(df[c], errors="coerce").fillna(0)
        vals.extend([float(v) for v in total if v > 0])
    return float(np.mean(vals)) if vals else np.nan


def mean_total_cost_all(df: pd.DataFrame) -> float:
    """Mean over all non-zero run-level costs."""
    vals: List[float] = []
    for run in range(1, 5):
        col = f"total_cost_run{run}"
        if col not in df.columns:
            continue
        cost = pd.to_numeric(df[col], errors="coerce").fillna(0)
        vals.extend([float(v) for v in cost if v > 0])
    return float(np.mean(vals)) if vals else np.nan


def mean_total_tokens_success(df: pd.DataFrame, success_ids: List[str]) -> float:
    """Mean token total over successful runs within success subset."""
    vals: List[float] = []
    sub = df[df["instance_id"].astype(str).isin(success_ids)]

    for _, row in sub.iterrows():
        for run in range(1, 5):
            acc = row.get(f"acc_run{run}")
            if pd.isna(acc) or int(acc) != 1:
                continue
            p = safe_float(row.get(f"total_prompt_tokens_run{run}"))
            c = safe_float(row.get(f"total_completion_tokens_run{run}"))
            if p is None or c is None:
                continue
            total = p + c
            if total > 0:
                vals.append(total)

    return float(np.mean(vals)) if vals else np.nan


def mean_total_cost_success(df: pd.DataFrame, success_ids: List[str]) -> float:
    """Mean total cost over successful runs within success subset."""
    vals: List[float] = []
    sub = df[df["instance_id"].astype(str).isin(success_ids)]

    for _, row in sub.iterrows():
        for run in range(1, 5):
            acc = row.get(f"acc_run{run}")
            if pd.isna(acc) or int(acc) != 1:
                continue
            cost = safe_float(row.get(f"total_cost_run{run}"))
            if cost is not None and cost > 0:
                vals.append(cost)

    return float(np.mean(vals)) if vals else np.nan


def compute_prediction_metrics(pred_csv: Path) -> Dict[str, float]:
    """Compute prediction-vs-ground-truth metrics from one prediction CSV."""
    out = {
        "corr_pred_input_gt_input": np.nan,
        "corr_pred_output_gt_output": np.nan,
        "ratio_pred_cost_task_cost": np.nan,
    }
    if not pred_csv.exists():
        return out

    df = pd.read_csv(pred_csv)
    run_corr_in: List[float] = []
    run_corr_out: List[float] = []
    ratios: List[float] = []

    for run in (1, 2, 3):
        pi = pd.to_numeric(df.get(f"predicted_input_tokens_run{run}"), errors="coerce")
        po = pd.to_numeric(df.get(f"predicted_output_tokens_run{run}"), errors="coerce")
        pc = pd.to_numeric(df.get(f"prediction_cost_run{run}"), errors="coerce")
        gi = pd.to_numeric(df.get("gt_input_token_avg"), errors="coerce")
        go = pd.to_numeric(df.get("gt_output_token_avg"), errors="coerce")
        tc = pd.to_numeric(df.get("task_cost_avg"), errors="coerce")

        valid_in = pi.notna() & gi.notna()
        valid_out = po.notna() & go.notna()
        valid_ratio = pc.notna() & tc.notna() & (tc != 0)

        c_in = corr(pi[valid_in].tolist(), gi[valid_in].tolist())
        c_out = corr(po[valid_out].tolist(), go[valid_out].tolist())
        if c_in is not None:
            run_corr_in.append(c_in)
        if c_out is not None:
            run_corr_out.append(c_out)

        if valid_ratio.any():
            ratios.extend((pc[valid_ratio] / tc[valid_ratio]).tolist())

    if run_corr_in:
        out["corr_pred_input_gt_input"] = float(np.mean(run_corr_in))
    if run_corr_out:
        out["corr_pred_output_gt_output"] = float(np.mean(run_corr_out))
    if ratios:
        out["ratio_pred_cost_task_cost"] = float(np.mean(ratios))

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified model comparison table generator")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset"),
        help="Root directory containing per-model token/cost CSVs",
    )
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=Path("self_prediction_analysis"),
        help="Directory containing per-model self-prediction CSVs",
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
        default=Path("analysis/unified_model_comparison_summary.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    model_paths = default_model_paths(args.dataset_root)
    if args.model_csv:
        model_paths.update(parse_model_paths(args.model_csv))

    frames: Dict[str, pd.DataFrame] = {}
    for model, path in sorted(model_paths.items()):
        if path.exists():
            df = pd.read_csv(path)
            if "instance_id" not in df.columns:
                raise ValueError(f"Missing instance_id in {path}")
            frames[model] = df

    if not frames:
        raise FileNotFoundError("No model CSVs found")

    model_names = sorted(frames.keys())
    success_ids = build_success_subset(frames, model_names)

    rows = []
    for model in model_names:
        df = frames[model]
        pred_metrics = compute_prediction_metrics(find_prediction_csv(args.prediction_root, model))

        rows.append(
            {
                "model_id": model,
                "model_display": model,
                "total_tokens_all_mean": mean_total_tokens_all(df),
                "total_cost_all_mean": mean_total_cost_all(df),
                "total_tokens_success_mean": mean_total_tokens_success(df, success_ids),
                "total_cost_success_mean": mean_total_cost_success(df, success_ids),
                "corr_pred_input_gt_input": pred_metrics["corr_pred_input_gt_input"],
                "corr_pred_output_gt_output": pred_metrics["corr_pred_output_gt_output"],
                "ratio_pred_cost_task_cost": pred_metrics["ratio_pred_cost_task_cost"],
            }
        )

    out_df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)

    print(f"Models used: {', '.join(model_names)}")
    print(f"Success subset size: {len(success_ids)}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
