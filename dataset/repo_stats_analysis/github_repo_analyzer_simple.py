#!/usr/bin/env python3
"""Analyze repository file statistics via GitHub API (PyGithub).

Example:
  python dataset/repo_stats_analysis/github_repo_analyzer_simple.py \
      --repo owner/name --output owner_name_analysis.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

from tqdm import tqdm

try:
    from github import Github
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyGithub is required. Install with: pip install PyGithub") from exc


def analyze_github_repo(owner: str, repo_name: str, token: Optional[str] = None, show_progress: bool = True) -> Optional[Dict]:
    """Analyze a GitHub repository and return file-type/line statistics."""
    client = Github(token) if token else Github()

    try:
        repo = client.get_repo(f"{owner}/{repo_name}")
    except Exception as exc:
        print(f"Error accessing repository {owner}/{repo_name}: {exc}")
        return None

    file_stats: Dict[str, List[Dict]] = defaultdict(list)
    file_types: Dict[str, int] = defaultdict(int)
    total_files = 0
    total_lines = 0

    root_contents = repo.get_contents("")

    def count_files(contents_list) -> int:
        count = 0
        for content in contents_list:
            if content.type == "file":
                count += 1
            elif content.type == "dir":
                try:
                    sub = repo.get_contents(content.path)
                    count += count_files(sub)
                except Exception:
                    continue
        return count

    total_target = count_files(root_contents) if show_progress else None
    pbar = tqdm(total=total_target, desc="Processing files", unit="file") if show_progress else None

    def walk(contents_list) -> None:
        nonlocal total_files, total_lines

        for content in contents_list:
            if content.type == "file":
                if pbar:
                    pbar.set_description(f"Processing {content.name}")
                    pbar.update(1)

                try:
                    decoded = content.decoded_content.decode("utf-8", errors="ignore")
                except Exception:
                    decoded = ""

                line_count = len(decoded.splitlines())
                extension = os.path.splitext(content.name)[1].lower()

                info = {
                    "path": content.path,
                    "name": content.name,
                    "size": int(content.size or 0),
                    "lines": line_count,
                    "type": "file",
                }

                file_stats[extension].append(info)
                file_types[extension] += 1
                total_files += 1
                total_lines += line_count

            elif content.type == "dir":
                try:
                    sub = repo.get_contents(content.path)
                    walk(sub)
                except Exception:
                    continue

    try:
        walk(root_contents)
    finally:
        if pbar:
            pbar.close()

    return {
        "repository": f"{owner}/{repo_name}",
        "total_files": total_files,
        "total_lines": total_lines,
        "file_types": dict(file_types),
        "file_details": dict(file_stats),
    }


def print_analysis(analysis: Dict) -> None:
    """Pretty-print analysis summary."""
    print(f"Repository Analysis: {analysis['repository']}")
    print("=" * 60)
    print(f"Total Files: {analysis['total_files']}")
    print(f"Total Lines: {analysis['total_lines']}")
    print("\nFile Types:")

    for ext, count in sorted(analysis["file_types"].items(), key=lambda x: x[1], reverse=True):
        label = ext if ext else "[no extension]"
        print(f"  {label}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a GitHub repository")
    parser.add_argument("--repo", required=True, help="Repository in owner/name format")
    parser.add_argument("--token", default=None, help="GitHub token (optional)")
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    args = parser.parse_args()

    if "/" not in args.repo:
        raise ValueError("--repo must be in owner/name format")

    owner, repo_name = args.repo.split("/", 1)
    analysis = analyze_github_repo(owner, repo_name, token=args.token, show_progress=not args.no_progress)
    if not analysis:
        raise RuntimeError("Failed to analyze repository")

    print_analysis(analysis)

    if args.output:
        out_path = os.path.abspath(args.output)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2)
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
