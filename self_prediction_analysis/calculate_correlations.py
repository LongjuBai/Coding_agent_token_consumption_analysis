#!/usr/bin/env python3
"""Compute self-prediction correlation metrics and save summary CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


MODELS: List[Tuple[str, str, str]] = [
    ("GPT-5", "gpt5_predictions.csv", "gpt5_predictions_log.csv"),
    ("GPT-5.2", "gpt5_2_predictions.csv", "gpt5_2_predictions_log.csv"),
    ("Qwen3 Coder", "qwen3_coder_predictions.csv", "qwen3_coder_predictions_log.csv"),
    ("Gemini 3", "gemini3_predictions.csv", "gemini3_predictions_log.csv"),
    ("Kimi K2", "kimi_k2_predictions.csv", "kimi_k2_predictions_log.csv"),
    ("Claude 3.7 Sonnet", "claude3_7_sonnet_predictions.csv", "claude3_7_sonnet_predictions_log.csv"),
    ("Sonnet 4 Base", "sonnet4_base_predictions.csv", "sonnet4_base_predictions_log.csv"),
    ("Sonnet 4.5", "sonnet4_5_predictions.csv", "sonnet4_5_predictions_log.csv"),
]

RESULT_COLUMNS = [
    "model",
    "corr_predInput_gtInput",
    "corr_predOutput_gtOutput",
    "corr_predInput_gtInput_log",
    "corr_predOutput_gtOutput_log",
    "corr_predCost_inputErr",
    "corr_predCost_outputErr",
    "corr_predCost_taskCost",
    "ratio_predCost_taskCost",
]


def set_csv_field_size_limit() -> None:
    """Increase CSV field size limit for large rows."""
    max_size = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_size)
            return
        except OverflowError:
            max_size //= 10


def safe_float(value: object) -> Optional[float]:
    """Best-effort conversion to float."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def pearson(x: List[float], y: List[float]) -> Optional[float]:
    """Compute Pearson correlation with defensive checks."""
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return None
    if len(set(x)) <= 1 or len(set(y)) <= 1:
        return None
    val = float(np.corrcoef(np.array(x), np.array(y))[0, 1])
    if np.isnan(val):
        return None
    return val


def load_task_cost_fallback_map(csv_path: Path) -> Dict[str, float]:
    """Load instance_id -> mean(total_cost_run1..4 positive) fallback map."""
    out: Dict[str, float] = {}
    if not csv_path.exists():
        return out

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_id = row.get("instance_id") or row.get("problem_id")
            if not row_id:
                continue

            vals = []
            for run in (1, 2, 3, 4):
                c = safe_float(row.get(f"total_cost_run{run}"))
                if c is not None and c > 0:
                    vals.append(c)

            if vals:
                out[row_id] = float(np.mean(vals))

    return out


def compute_correlations_for_file(path: Path, task_cost_fallback: Optional[Dict[str, float]] = None) -> Dict[str, Optional[float]]:
    """Compute all requested metrics from one prediction CSV."""
    out: Dict[str, Optional[float]] = {
        "pred_input_gt_input": None,
        "pred_output_gt_output": None,
        "pred_cost_input_err": None,
        "pred_cost_output_err": None,
        "pred_cost_task_cost": None,
        "pred_cost_task_cost_ratio": None,
    }
    if not path.exists():
        return out

    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    run_data = {1: [], 2: [], 3: []}

    for row in rows:
        gt_input = safe_float(row.get("gt_input_token_avg"))
        gt_output = safe_float(row.get("gt_output_token_avg"))
        task_cost = safe_float(row.get("task_cost_avg"))

        row_id = row.get("problem_id") or row.get("instance_id")
        if (task_cost is None or task_cost <= 0) and task_cost_fallback and row_id:
            fallback = task_cost_fallback.get(row_id)
            if fallback is not None and fallback > 0:
                task_cost = fallback

        for run in (1, 2, 3):
            pred_input = safe_float(row.get(f"predicted_input_tokens_run{run}"))
            pred_output = safe_float(row.get(f"predicted_output_tokens_run{run}"))
            pred_cost = safe_float(row.get(f"prediction_cost_run{run}"))

            if (
                gt_input is None
                or gt_output is None
                or task_cost is None
                or pred_input is None
                or pred_output is None
                or pred_cost is None
            ):
                continue

            run_data[run].append(
                {
                    "gt_input": gt_input,
                    "gt_output": gt_output,
                    "task_cost": task_cost,
                    "pred_input": pred_input,
                    "pred_output": pred_output,
                    "pred_cost": pred_cost,
                }
            )

    def mean_run_corr(get_x, get_y) -> Optional[float]:
        corrs = []
        for run in (1, 2, 3):
            x = [get_x(d) for d in run_data[run]]
            y = [get_y(d) for d in run_data[run]]
            c = pearson(x, y)
            if c is not None:
                corrs.append(c)
        return float(np.mean(corrs)) if corrs else None

    out["pred_input_gt_input"] = mean_run_corr(lambda d: d["pred_input"], lambda d: d["gt_input"])
    out["pred_output_gt_output"] = mean_run_corr(lambda d: d["pred_output"], lambda d: d["gt_output"])
    out["pred_cost_input_err"] = mean_run_corr(lambda d: d["pred_cost"], lambda d: abs(d["pred_input"] - d["gt_input"]))
    out["pred_cost_output_err"] = mean_run_corr(lambda d: d["pred_cost"], lambda d: abs(d["pred_output"] - d["gt_output"]))
    out["pred_cost_task_cost"] = mean_run_corr(lambda d: d["pred_cost"], lambda d: d["task_cost"])

    ratios = []
    for run in (1, 2, 3):
        for d in run_data[run]:
            if d["task_cost"] != 0:
                ratios.append(d["pred_cost"] / d["task_cost"])
    out["pred_cost_task_cost_ratio"] = float(np.mean(ratios)) if ratios else None

    return out


def format_metric(value: Optional[float]) -> str:
    """Format metric for CSV."""
    if value is None:
        return ""
    if np.isnan(value):
        return ""
    return f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate self-prediction correlations")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("self_prediction_analysis"),
        help="Directory containing prediction CSV files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("self_prediction_analysis/correlations_results.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--qwen-fallback-csv",
        type=Path,
        default=Path("dataset/qwen/swe_bench_token_cost_aggregated_total_with_accuracy.csv"),
        help="Optional fallback task-cost source for Qwen",
    )
    args = parser.parse_args()

    set_csv_field_size_limit()

    qwen_fallback = load_task_cost_fallback_map(args.qwen_fallback_csv)
    rows_out = []

    for model_name, regular_name, log_name in MODELS:
        fallback = qwen_fallback if model_name == "Qwen3 Coder" else None

        regular = compute_correlations_for_file(args.root / regular_name, task_cost_fallback=fallback)
        log_metrics = compute_correlations_for_file(args.root / log_name, task_cost_fallback=fallback)

        row = {
            "model": model_name,
            "corr_predInput_gtInput": format_metric(regular["pred_input_gt_input"]),
            "corr_predOutput_gtOutput": format_metric(regular["pred_output_gt_output"]),
            "corr_predInput_gtInput_log": format_metric(log_metrics["pred_input_gt_input"]),
            "corr_predOutput_gtOutput_log": format_metric(log_metrics["pred_output_gt_output"]),
            "corr_predCost_inputErr": format_metric(regular["pred_cost_input_err"]),
            "corr_predCost_outputErr": format_metric(regular["pred_cost_output_err"]),
            "corr_predCost_taskCost": format_metric(regular["pred_cost_task_cost"]),
            "ratio_predCost_taskCost": format_metric(regular["pred_cost_task_cost_ratio"]),
        }
        rows_out.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {args.output} ({len(rows_out)} rows)")


if __name__ == "__main__":
    main()
