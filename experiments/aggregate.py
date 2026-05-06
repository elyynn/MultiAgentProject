"""Aggregator for the minimal experiment suite.

For each arm, reads every per-seed `final_summary.json` and `epoch_metrics.csv`
from `outputs/v2/<arm>/seed_<n>/results/` and emits:
  - outputs/v2/<arm>/aggregate/summary.json
  - outputs/v2/<arm>/aggregate/epoch_curves.npz
  - outputs/v2/<arm>/aggregate/per_seed_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from typing import Dict, List, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
V2_DIR = os.path.join(_ROOT, "outputs", "v2")

ARM_NAMES = [
    "default",
    "no_detection",
    "no_spillover",
    "null_ai",
    "sens_detect_lo",
    "sens_detect_hi",
]


def _seed_dirs(arm_dir: str) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    if not os.path.isdir(arm_dir):
        return out
    for name in os.listdir(arm_dir):
        m = re.match(r"^seed_(-?\d+)$", name)
        if m:
            out.append((int(m.group(1)), os.path.join(arm_dir, name)))
    out.sort(key=lambda t: t[0])
    return out


def _load_final_summary(seed_dir: str) -> Dict | None:
    path = os.path.join(seed_dir, "results", "final_summary.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _load_epoch_metrics(seed_dir: str) -> List[Dict] | None:
    path = os.path.join(seed_dir, "results", "epoch_metrics.csv")
    if not os.path.isfile(path):
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and (
        not isinstance(x, float) or not (math.isnan(x) or math.isinf(x))
    )


def _stat_block(values: List[float], per_seed: Dict[int, float]) -> Dict:
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    if n == 0:
        return {"mean": None, "sem": None, "ci95_low": None, "ci95_high": None,
                "n": 0, "per_seed": per_seed}
    mean = float(np.mean(arr))
    if n > 1:
        sd = float(np.std(arr, ddof=1))
        sem = sd / math.sqrt(n)
    else:
        sem = 0.0
    ci_half = 1.96 * sem
    return {
        "mean": mean,
        "sem": sem,
        "ci95_low": mean - ci_half,
        "ci95_high": mean + ci_half,
        "n": n,
        "per_seed": per_seed,
    }


def aggregate_arm(arm: str) -> Dict:
    arm_dir = os.path.join(V2_DIR, arm)
    out_dir = os.path.join(arm_dir, "aggregate")
    os.makedirs(out_dir, exist_ok=True)

    seed_dirs = _seed_dirs(arm_dir)
    if not seed_dirs:
        print(f"[aggregate/{arm}] no seed dirs found")
        return {"arm": arm, "n_seeds_found": 0}

    # ---- Load final summaries ----
    summaries: List[Tuple[int, Dict]] = []
    for seed, sdir in seed_dirs:
        s = _load_final_summary(sdir)
        if s is None:
            print(f"[aggregate/{arm}] WARN: missing final_summary.json for seed_{seed}")
            continue
        summaries.append((seed, s))

    # Collect all numeric keys across summaries
    keys = []
    for _, s in summaries:
        for k, v in s.items():
            if k not in keys and _is_number(v):
                keys.append(k)

    summary_block: Dict[str, Dict] = {}
    for k in keys:
        vals: List[float] = []
        per_seed: Dict[int, float] = {}
        for seed, s in summaries:
            v = s.get(k)
            if _is_number(v):
                vals.append(float(v))
                per_seed[seed] = float(v)
        summary_block[k] = _stat_block(vals, per_seed)

    summary_path = os.path.join(out_dir, "summary.json")
    summary_doc = {
        "arm": arm,
        "n_seeds": len(summaries),
        "seeds": [s for s, _ in summaries],
        "fields": summary_block,
    }
    with open(summary_path, "w") as f:
        json.dump(summary_doc, f, indent=2)
    print(f"[aggregate/{arm}] wrote {summary_path}")

    # ---- Per-seed summary CSV ----
    csv_path = os.path.join(out_dir, "per_seed_summary.csv")
    if summaries:
        all_keys = []
        for _, s in summaries:
            for k in s.keys():
                if k not in all_keys:
                    all_keys.append(k)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["seed"] + all_keys)
            writer.writeheader()
            for seed, s in summaries:
                row = {"seed": seed}
                row.update(s)
                writer.writerow(row)
        print(f"[aggregate/{arm}] wrote {csv_path}")

    # ---- Epoch curves NPZ ----
    epoch_data: List[Tuple[int, List[Dict]]] = []
    for seed, sdir in seed_dirs:
        rows = _load_epoch_metrics(sdir)
        if rows is None:
            print(f"[aggregate/{arm}] WARN: missing epoch_metrics.csv for seed_{seed}")
            continue
        epoch_data.append((seed, rows))

    if epoch_data:
        # Determine numeric columns from first non-empty file
        first_rows = epoch_data[0][1]
        if first_rows:
            numeric_cols: List[str] = []
            for col in first_rows[0].keys():
                v = first_rows[0][col]
                try:
                    float(v)
                    numeric_cols.append(col)
                except (TypeError, ValueError):
                    pass

            # Determine common length (use min across seeds, expect 100)
            lengths = [len(rows) for _, rows in epoch_data]
            n_epochs = min(lengths) if lengths else 0
            n_seeds = len(epoch_data)

            arrays: Dict[str, np.ndarray] = {}
            for col in numeric_cols:
                a = np.full((n_seeds, n_epochs), np.nan, dtype=float)
                for i, (_, rows) in enumerate(epoch_data):
                    for j in range(n_epochs):
                        try:
                            a[i, j] = float(rows[j][col])
                        except (TypeError, ValueError, KeyError):
                            pass
                arrays[col] = a
            arrays["seeds"] = np.asarray([s for s, _ in epoch_data], dtype=np.int64)

            npz_path = os.path.join(out_dir, "epoch_curves.npz")
            np.savez(npz_path, **arrays)
            print(f"[aggregate/{arm}] wrote {npz_path} "
                  f"(n_seeds={n_seeds}, n_epochs={n_epochs}, n_cols={len(numeric_cols)})")

    return {
        "arm": arm,
        "n_seeds_found": len(seed_dirs),
        "n_summaries": len(summaries),
        "n_epoch_files": len(epoch_data),
    }


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Aggregate per-seed runs into per-arm summaries.")
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument("--arm", default=None, help="Arm name to aggregate.")
    g.add_argument("--all", action="store_true", help="Aggregate all known arms.")
    args = p.parse_args(argv)

    if args.all or args.arm is None:
        arms = ARM_NAMES
    else:
        arms = [args.arm]

    for arm in arms:
        aggregate_arm(arm)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
