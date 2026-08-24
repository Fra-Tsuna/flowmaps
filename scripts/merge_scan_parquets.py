#!/usr/bin/env python3
"""Merge scan_*.parquet into scan_merged.parquet for each env under data/."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import polars as pl

SCAN_RE = re.compile(r"^scan_(\d+)\.parquet$")


def find_scans(env_dir: Path) -> List[Path]:
    scans: List[Tuple[int, Path]] = []
    for p in env_dir.iterdir():
        if not p.is_file():
            continue
        match = SCAN_RE.match(p.name)
        if match:
            scans.append((int(match.group(1)), p))
    scans.sort(key=lambda x: x[0])
    return [p for _, p in scans]


def merge_env(env_dir: Path, out_name: str, overwrite: bool) -> bool:
    scans = find_scans(env_dir)
    if not scans:
        return False

    out_path = env_dir / out_name
    if out_path.exists() and not overwrite:
        return False

    lazy_frames = [pl.scan_parquet(str(p)) for p in scans]
    merged = pl.concat(lazy_frames, how="vertical")
    merged.collect(streaming=True).write_parquet(out_path)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Path to data root containing env directories (default: flow-sim/data)",
    )
    parser.add_argument(
        "--out-name",
        default="scan_merged.parquet",
        help="Name of merged parquet file to write inside each env dir",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing merged parquet files",
    )
    args = parser.parse_args()

    data_root = args.data_root
    if not data_root.exists():
        raise SystemExit(f"Data root does not exist: {data_root}")

    # Find all directories that contain at least one scan_N.parquet file,
    # regardless of nesting depth (handles both flat and nested habit layouts).
    seen_dirs: set[Path] = set()
    for scan_file in sorted(data_root.rglob("scan_[0-9]*.parquet")):
        if SCAN_RE.match(scan_file.name):
            seen_dirs.add(scan_file.parent)
    env_dirs = sorted(seen_dirs)

    merged_count = 0
    for env_dir in env_dirs:
        if merge_env(env_dir, args.out_name, args.overwrite):
            merged_count += 1

    print(f"Merged {merged_count} envs under {data_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
