#!/usr/bin/env python3
"""
Build dataset_index.csv for EVCS simulator exports.

Expected folder pattern:
    data/omn_muscat_hybrid_seed42/csv
    data/omn_nizwa_real_seed42/csv
    data/omn_salalah_synthetic_seed42/csv

Output columns:
    city_id,country,data_mode,seed,data_dir

Usage:
    python make_dataset_index.py --root data --output dataset_index.csv --seed 42
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional


COUNTRY_MAP: Dict[str, str] = {
    "omn": "Oman",
    "oman": "Oman",
    "sau": "Saudi Arabia",
    "ksa": "Saudi Arabia",
    "saudi": "Saudi Arabia",
}

MODE_VALUES = {"real", "hybrid", "synthetic"}


def parse_dataset_folder_name(name: str) -> Optional[dict]:
    """Parse names like omn_muscat_hybrid_seed42."""
    pattern = re.compile(
        r"^(?P<country_code>[a-zA-Z]+)_(?P<city>.+?)_(?P<mode>real|hybrid|synthetic)_seed(?P<seed>\d+)$",
        re.IGNORECASE,
    )
    match = pattern.match(name)
    if not match:
        return None

    country_code = match.group("country_code").lower()
    city = match.group("city").lower()
    mode = match.group("mode").lower()
    seed = int(match.group("seed"))

    country = COUNTRY_MAP.get(country_code, country_code.upper())
    city_id = f"{country_code}_{city}"

    return {
        "city_id": city_id,
        "country": country,
        "data_mode": mode,
        "seed": seed,
    }


def is_valid_csv_export_dir(path: Path) -> bool:
    """Check whether a folder appears to contain simulator CSV exports."""
    if not path.is_dir():
        return False

    expected_keywords = ["candidate", "demand", "node", "edge"]
    filenames = [p.name.lower() for p in path.glob("*")]
    return any(any(keyword in name for keyword in expected_keywords) for name in filenames)


def find_dataset_csv_dirs(root: Path) -> List[Path]:
    """
    Find CSV export folders.

    Supports:
        data/omn_muscat_hybrid_seed42/csv
    and:
        data/omn_muscat_hybrid_seed42
    """
    candidates: List[Path] = []

    for path in root.rglob("*"):
        if not path.is_dir():
            continue

        if path.name.lower() == "csv" and parse_dataset_folder_name(path.parent.name):
            if is_valid_csv_export_dir(path):
                candidates.append(path)

        elif parse_dataset_folder_name(path.name):
            csv_child = path / "csv"
            if csv_child.exists() and csv_child.is_dir() and is_valid_csv_export_dir(csv_child):
                candidates.append(csv_child)
            elif is_valid_csv_export_dir(path):
                candidates.append(path)

    return sorted(set(candidates), key=lambda p: str(p))


def build_index(root: Path, fixed_seed: Optional[int] = None) -> List[dict]:
    rows: List[dict] = []

    for csv_dir in find_dataset_csv_dirs(root):
        # Important fix:
        # If the folder is ".../csv", parse the parent dataset folder name,
        # not the literal folder name "csv".
        dataset_folder = csv_dir.parent if csv_dir.name.lower() == "csv" else csv_dir

        parsed = parse_dataset_folder_name(dataset_folder.name)
        if parsed is None:
            continue

        if fixed_seed is not None and parsed["seed"] != fixed_seed:
            continue

        rows.append(
            {
                "city_id": parsed["city_id"],
                "country": parsed["country"],
                "data_mode": parsed["data_mode"],
                "seed": parsed["seed"],
                "data_dir": str(csv_dir).replace("\\", "/"),
            }
        )

    return sorted(rows, key=lambda r: (r["city_id"], r["data_mode"], r["seed"], r["data_dir"]))


def check_mode_balance(rows: List[dict]) -> List[str]:
    warnings: List[str] = []
    by_city: Dict[str, set] = {}

    for row in rows:
        by_city.setdefault(row["city_id"], set()).add(row["data_mode"])

    for city_id, modes in sorted(by_city.items()):
        missing = MODE_VALUES - modes
        if missing:
            warnings.append(f"WARNING: {city_id} is missing mode(s): {', '.join(sorted(missing))}")

    return warnings


def write_csv(rows: List[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["city_id", "country", "data_mode", "seed", "data_dir"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Root folder containing dataset folders.")
    parser.add_argument("--output", type=Path, default=Path("dataset_index.csv"))
    parser.add_argument("--seed", type=int, default=42, help="Keep only this seed. Use -1 to keep all seeds.")
    args = parser.parse_args()

    fixed_seed = None if args.seed == -1 else args.seed

    rows = build_index(args.root, fixed_seed=fixed_seed)
    write_csv(rows, args.output)

    print(f"Wrote {len(rows)} rows to {args.output}")

    for warning in check_mode_balance(rows):
        print(warning)

    if not rows:
        print("No dataset folders found. Expected pattern: data/omn_muscat_hybrid_seed42/csv")


if __name__ == "__main__":
    main()