"""Shared helpers for the experiment scripts."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args(description: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results", help="folder for CSV output")
    p.add_argument("--store", default=None, help="folder for the trace store (records and frames); off by default")
    return p.parse_args()


def write_csv(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {path}")
