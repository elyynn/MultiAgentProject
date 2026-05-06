"""Sweep driver for the minimal experiment suite.

Per (arm, seed):
  1. Build a fresh config from get_default_config().
  2. Apply per-arm mutations.
  3. Set cfg.seed and cfg.output_dir = outputs/v2/<arm>/seed_<n>.
  4. Call run_simulation, then save_results.
  5. Write run_manifest.json.

Failures are caught per (arm, seed) so one crash does not abort the suite.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Callable, Dict, List, Tuple

# Ensure project root is importable regardless of cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy

from config import SimulationConfig, get_default_config
from simulation import run_simulation
from utils import save_results


SHARED_SEEDS: List[int] = [42, 123, 2024, 7, 1337, 99, 314, 271, 8675309, 555]


def _arm_default(cfg: SimulationConfig) -> None:
    pass


def _arm_no_detection(cfg: SimulationConfig) -> None:
    cfg.detection_prob_by_type = {0: 0.0, 1: 0.0, 2: 0.0}


def _arm_no_spillover(cfg: SimulationConfig) -> None:
    cfg.num_spillover_companies = 0


def _arm_null_ai(cfg: SimulationConfig) -> None:
    cfg.ai_signal_boost = 0


def _arm_sens_detect_lo(cfg: SimulationConfig) -> None:
    cfg.detection_prob_by_type = {0: 0.175, 1: 0.10, 2: 0.05}


def _arm_sens_detect_hi(cfg: SimulationConfig) -> None:
    cfg.detection_prob_by_type = {0: 0.70, 1: 0.40, 2: 0.20}


# (arm_name, mutator, seeds)
ARMS: List[Tuple[str, Callable[[SimulationConfig], None], List[int]]] = [
    ("default", _arm_default, SHARED_SEEDS[:10]),
    ("no_detection", _arm_no_detection, SHARED_SEEDS[:5]),
    ("no_spillover", _arm_no_spillover, SHARED_SEEDS[:5]),
    ("null_ai", _arm_null_ai, SHARED_SEEDS[:5]),
    ("sens_detect_lo", _arm_sens_detect_lo, SHARED_SEEDS[:5]),
    ("sens_detect_hi", _arm_sens_detect_hi, SHARED_SEEDS[:5]),
]


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return out.decode("ascii").strip()
    except Exception:
        return "unknown"


def _config_to_dict(cfg: SimulationConfig) -> dict:
    return dataclasses.asdict(cfg)


def _write_manifest(
    out_dir: str,
    *,
    arm: str,
    seed: int,
    cfg: SimulationConfig,
    git_sha: str,
    start_iso: str,
    end_iso: str,
    wall_clock_s: float,
    status: str,
    error: str | None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        "arm": arm,
        "seed": seed,
        "config": _config_to_dict(cfg),
        "git_sha": git_sha,
        "python_version": sys.version,
        "numpy_version": numpy.__version__,
        "start_iso": start_iso,
        "end_iso": end_iso,
        "wall_clock_s": wall_clock_s,
        "hostname": socket.gethostname(),
        "command_line": sys.argv,
        "status": status,
        "error": error,
    }
    path = os.path.join(out_dir, "run_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def run_one(arm_name: str, mutator: Callable[[SimulationConfig], None], seed: int,
            git_sha: str) -> Tuple[bool, float, str | None]:
    cfg = get_default_config()
    mutator(cfg)
    cfg.seed = seed
    out_dir = os.path.join(_ROOT, "outputs", "v2", arm_name, f"seed_{seed}")
    cfg.output_dir = out_dir
    cfg.save_results = True
    os.makedirs(out_dir, exist_ok=True)

    start_iso = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    err_msg = None
    status = "ok"

    try:
        results = run_simulation(cfg)
        save_results(results, cfg)
    except Exception:
        status = "error"
        err_msg = traceback.format_exc()

    wall = time.perf_counter() - t0
    end_iso = datetime.now(timezone.utc).isoformat()

    _write_manifest(
        out_dir,
        arm=arm_name,
        seed=seed,
        cfg=cfg,
        git_sha=git_sha,
        start_iso=start_iso,
        end_iso=end_iso,
        wall_clock_s=wall,
        status=status,
        error=err_msg,
    )

    return (status == "ok"), wall, err_msg


def _select_arms(arm_arg: str) -> List[Tuple[str, Callable, List[int]]]:
    if arm_arg == "all":
        return list(ARMS)
    matched = [a for a in ARMS if a[0] == arm_arg]
    if not matched:
        names = ", ".join(a[0] for a in ARMS)
        raise SystemExit(f"Unknown arm '{arm_arg}'. Available: {names}, all")
    return matched


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the minimal experiment suite.")
    p.add_argument("--arm", default="all", help="Arm name or 'all' (default: all).")
    p.add_argument(
        "--seeds",
        default=None,
        help="Optional comma-separated seed list to override the per-arm defaults.",
    )
    args = p.parse_args(argv)

    arms = _select_arms(args.arm)
    seed_override: List[int] | None = None
    if args.seeds:
        seed_override = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    git_sha = _git_sha()
    print(f"[suite] git_sha={git_sha}  python={sys.version.split()[0]}  numpy={numpy.__version__}")
    print(f"[suite] arms: {[a[0] for a in arms]}")

    suite_t0 = time.perf_counter()
    total_runs = 0
    ok_runs = 0
    fail_runs: List[Tuple[str, int, str]] = []

    for arm_name, mutator, default_seeds in arms:
        seeds = seed_override if seed_override is not None else default_seeds
        for seed in seeds:
            total_runs += 1
            ok, wall, err = run_one(arm_name, mutator, seed, git_sha)
            if ok:
                ok_runs += 1
                print(f"[{arm_name}/seed_{seed}] ok in {wall:.1f}s")
            else:
                fail_runs.append((arm_name, seed, err or ""))
                first_line = (err or "").strip().splitlines()[-1] if err else "unknown"
                print(f"[{arm_name}/seed_{seed}] FAILED: {first_line}")

    suite_wall = time.perf_counter() - suite_t0
    print(f"[suite] done. {ok_runs}/{total_runs} ok in {suite_wall:.1f}s")
    if fail_runs:
        print(f"[suite] {len(fail_runs)} failures:")
        for arm, seed, _ in fail_runs:
            print(f"  - {arm}/seed_{seed}")

    return 0 if ok_runs == total_runs else 1


if __name__ == "__main__":
    raise SystemExit(main())
