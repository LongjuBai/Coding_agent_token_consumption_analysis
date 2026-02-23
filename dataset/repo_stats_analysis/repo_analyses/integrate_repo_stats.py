#!/usr/bin/env python3
"""Integrate repository-analysis JSON stats into a tabular CSV dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def load_analysis(path: Path) -> Dict:
    """Load one analysis JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_ext(ext: str) -> str:
    """Normalize extension key for column names."""
    if not ext:
        return "no_ext"
    return ext.replace(".", "_")


def gather_top_file_types(json_files: List[Path], top_k: int) -> List[str]:
    """Find top-k file extensions across analyses."""
    counts: Counter = Counter()
    for path in json_files:
        data = load_analysis(path)
        counts.update(data.get("file_types", {}))
    return [ext for ext, _ in counts.most_common(top_k)]


def build_repo_stats(json_files: List[Path], top_types: List[str]) -> Dict[str, Dict[str, int]]:
    """Build repo -> stats mapping."""
    stats: Dict[str, Dict[str, int]] = {}

    for path in json_files:
        data = load_analysis(path)
        repo_name = str(data.get("repository", "")).strip()
        if not repo_name:
            continue

        row: Dict[str, int] = {
            "total_files": int(data.get("total_files", 0) or 0),
            "total_lines": int(data.get("total_lines", 0) or 0),
        }

        file_types = data.get("file_types", {})
        for ext in top_types:
            col = f"count_{normalize_ext(ext)}"
            row[col] = int(file_types.get(ext, 0) or 0)

        stats[repo_name] = row

    return stats


def repo_keys(repo_name: str) -> Tuple[str, str]:
    """Return (full_name, short_name) keys for matching."""
    full = repo_name.strip()
    short = full.split("/")[-1] if "/" in full else full
    return full.lower(), short.lower()


def match_repo_stats(repo_value: str, stats: Dict[str, Dict[str, int]]) -> Dict[str, int] | None:
    """Match CSV repo value to stats map using full or short repo name."""
    full, short = repo_keys(repo_value)

    for repo_name, row in stats.items():
        repo_full, repo_short = repo_keys(repo_name)
        if full == repo_full or full == repo_short:
            return row
        if short == repo_full or short == repo_short:
            return row

    return None


def integrate(csv_path: Path, stats: Dict[str, Dict[str, int]], top_types: List[str]) -> pd.DataFrame:
    """Add repository stats columns into the source CSV."""
    df = pd.read_csv(csv_path)
    if "repo" not in df.columns:
        raise ValueError(f"Missing 'repo' column in {csv_path}")

    stat_cols = ["total_files", "total_lines"] + [f"count_{normalize_ext(ext)}" for ext in top_types]
    for col in stat_cols:
        if col not in df.columns:
            df[col] = 0

    for idx, repo_value in df["repo"].fillna("").astype(str).items():
        matched = match_repo_stats(repo_value, stats)
        if not matched:
            continue
        for col in stat_cols:
            df.at[idx, col] = int(matched.get(col, 0))

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrate repository statistics into CSV dataset")
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=Path("dataset/repo_stats_analysis/repo_analyses"),
        help="Directory containing *_analysis.json files",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("dataset/repo_stats_analysis/repo_analyses/simplified_with_tool_avgs.csv"),
        help="Input CSV path",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("dataset/repo_stats_analysis/repo_analyses/simplified_with_tool_avgs_with_repo_stats.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--top-k-file-types",
        type=int,
        default=5,
        help="Number of most-common file types to include",
    )
    args = parser.parse_args()

    json_files = sorted(args.json_dir.glob("*_analysis.json"))
    if not json_files:
        raise FileNotFoundError(f"No *_analysis.json files found in {args.json_dir}")
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    top_types = gather_top_file_types(json_files, args.top_k_file_types)
    stats = build_repo_stats(json_files, top_types)
    out_df = integrate(args.input_csv, stats, top_types)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)

    print(f"Analyzed JSON files: {len(json_files)}")
    print(f"Repositories with stats: {len(stats)}")
    print(f"Top file types: {top_types}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
