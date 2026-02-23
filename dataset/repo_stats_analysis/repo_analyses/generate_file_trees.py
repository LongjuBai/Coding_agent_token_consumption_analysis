#!/usr/bin/env python3
"""Generate plain-text file trees from repository analysis JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def parse_file_path(file_path: str) -> List[str]:
    """Split a repository path into components."""
    return [p for p in file_path.split("/") if p]


def build_directory_tree(file_details: Dict[str, List[Dict]]) -> Dict:
    """Build nested directory tree from file_details map."""
    tree = {"files": [], "dirs": {}}

    for file_type, files in file_details.items():
        if not isinstance(files, list):
            continue

        for file_info in files:
            if not isinstance(file_info, dict):
                continue
            path = file_info.get("path", "")
            if not path:
                continue

            parts = parse_file_path(path)
            if not parts:
                continue

            node = tree
            for part in parts[:-1]:
                node = node["dirs"].setdefault(part, {"files": [], "dirs": {}})

            node["files"].append(
                {
                    "name": parts[-1],
                    "size": int(file_info.get("size", 0) or 0),
                    "lines": int(file_info.get("lines", 0) or 0),
                    "type": file_type,
                }
            )

    return tree


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    size_kb = size_bytes / 1024
    if size_kb >= 1024:
        return f"{size_kb / 1024:.1f}MB"
    return f"{size_kb:.1f}KB"


def render_tree(node: Dict, prefix: str = "") -> List[str]:
    """Render directory tree in ASCII format."""
    lines: List[str] = []

    files = sorted(node["files"], key=lambda x: x["name"])
    dirs = sorted(node["dirs"].keys())
    entries_total = len(files) + len(dirs)

    idx = 0
    for file_info in files:
        idx += 1
        is_last = idx == entries_total
        branch = "`-- " if is_last else "|-- "
        label = f"{file_info['name']} ({format_size(file_info['size'])}, {file_info['lines']} lines)"
        lines.append(f"{prefix}{branch}{label}")

    for dirname in dirs:
        idx += 1
        is_last = idx == entries_total
        branch = "`-- " if is_last else "|-- "
        lines.append(f"{prefix}{branch}{dirname}/")

        child_prefix = f"{prefix}{'    ' if is_last else '|   '}"
        lines.extend(render_tree(node["dirs"][dirname], child_prefix))

    return lines


def generate_file_tree(json_path: Path, output_path: Path) -> None:
    """Generate one file-tree text file from one analysis JSON."""
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    repository = data.get("repository", "unknown")
    total_files = int(data.get("total_files", 0) or 0)
    total_lines = int(data.get("total_lines", 0) or 0)
    file_details = data.get("file_details", {})

    tree = build_directory_tree(file_details)

    out_lines = [
        f"Repository: {repository}",
        f"Total files: {total_files}",
        f"Total lines: {total_lines:,}",
        "=" * 60,
        "",
    ]
    out_lines.extend(render_tree(tree))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate file tree text files from repo analysis JSON")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("dataset/repo_stats_analysis/repo_analyses"),
        help="Directory containing *_analysis.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/repo_stats_analysis/repo_analyses/file_trees"),
        help="Directory to write *_file_tree.txt outputs",
    )
    parser.add_argument(
        "--pattern",
        default="*_analysis.json",
        help="Glob pattern for input JSON files",
    )
    args = parser.parse_args()

    json_files = sorted(args.input_dir.glob(args.pattern))
    if not json_files:
        raise FileNotFoundError(f"No files matching {args.pattern} in {args.input_dir}")

    success = 0
    for json_file in json_files:
        repo_stem = json_file.stem.replace("_analysis", "")
        output_path = args.output_dir / f"{repo_stem}_file_tree.txt"
        try:
            generate_file_tree(json_file, output_path)
            success += 1
        except Exception as exc:  # pragma: no cover
            print(f"Failed {json_file}: {exc}")

    print(f"Processed {success}/{len(json_files)} analysis files")
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
