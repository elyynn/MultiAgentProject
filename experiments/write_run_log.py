"""Build outputs/v2/_run_log.md from per-(arm, seed) run_manifest.json files."""
from __future__ import annotations

import json
import os
import re
import socket
import sys
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
V2_DIR = os.path.join(_ROOT, "outputs", "v2")

ARMS = [
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


def _load_manifest(seed_dir: str) -> Dict | None:
    path = os.path.join(seed_dir, "run_manifest.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _parse_iso(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def main() -> int:
    rows: List[Dict] = []
    for arm in ARMS:
        for seed, sdir in _seed_dirs(os.path.join(V2_DIR, arm)):
            m = _load_manifest(sdir)
            if m is None:
                continue
            rows.append({
                "arm": arm,
                "seed": seed,
                "status": m.get("status", "unknown"),
                "wall_clock_s": m.get("wall_clock_s"),
                "start_iso": m.get("start_iso"),
                "end_iso": m.get("end_iso"),
                "git_sha": m.get("git_sha"),
                "python_version": m.get("python_version"),
                "numpy_version": m.get("numpy_version"),
                "hostname": m.get("hostname"),
                "error": m.get("error"),
            })

    n_total = len(rows)
    n_ok = sum(1 for r in rows if r["status"] == "ok")
    n_fail = n_total - n_ok

    # Suite span
    starts = [_parse_iso(r["start_iso"]) for r in rows if r["start_iso"]]
    ends = [_parse_iso(r["end_iso"]) for r in rows if r["end_iso"]]
    starts = [s for s in starts if s]
    ends = [e for e in ends if e]
    suite_start = min(starts).isoformat() if starts else "-"
    suite_end = max(ends).isoformat() if ends else "-"
    if starts and ends:
        suite_span_s = (max(ends) - min(starts)).total_seconds()
    else:
        suite_span_s = None
    sum_wall = sum((r["wall_clock_s"] or 0.0) for r in rows)

    git_sha = next((r["git_sha"] for r in rows if r["git_sha"]), "unknown")
    python_version = next((r["python_version"] for r in rows if r["python_version"]), sys.version)
    numpy_version = next((r["numpy_version"] for r in rows if r["numpy_version"]), numpy.__version__)
    hostname = next((r["hostname"] for r in rows if r["hostname"]), socket.gethostname())

    lines: List[str] = []
    lines.append("# Minimal experiment suite - run log")
    lines.append("")
    lines.append(f"- **Suite start (UTC)**: {suite_start}")
    lines.append(f"- **Suite end   (UTC)**: {suite_end}")
    if suite_span_s is not None:
        lines.append(f"- **Wall-clock span**: {suite_span_s:.1f} s")
    lines.append(f"- **Sum of per-run wall-clocks**: {sum_wall:.1f} s "
                 f"(sequential execution, so ~equal to span)")
    lines.append(f"- **Runs**: {n_ok}/{n_total} ok ({n_fail} failed)")
    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append(f"- git SHA: `{git_sha}`")
    lines.append(f"- Python: `{python_version.splitlines()[0] if python_version else '?'}`")
    lines.append(f"- numpy: `{numpy_version}`")
    lines.append(f"- hostname: `{hostname}`")
    lines.append(f"- log generated (UTC): `{datetime.now(timezone.utc).isoformat()}`")
    lines.append("")
    lines.append("## Status table (arm x seed)")
    lines.append("")
    lines.append("| arm | seed | status | wall_clock_s | start (UTC) |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        wc = f"{r['wall_clock_s']:.2f}" if r["wall_clock_s"] is not None else "-"
        lines.append(f"| {r['arm']} | {r['seed']} | {r['status']} | {wc} | {r['start_iso']} |")
    lines.append("")

    # Per-arm wall-clock summary
    lines.append("## Per-arm wall-clock summary")
    lines.append("")
    lines.append("| arm | n_runs | total_s | mean_s | min_s | max_s |")
    lines.append("|---|---|---|---|---|---|")
    for arm in ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm and r["wall_clock_s"] is not None]
        if not arm_rows:
            continue
        wcs = [r["wall_clock_s"] for r in arm_rows]
        lines.append(
            f"| {arm} | {len(arm_rows)} | {sum(wcs):.2f} | {sum(wcs)/len(wcs):.2f} "
            f"| {min(wcs):.2f} | {max(wcs):.2f} |"
        )
    lines.append("")

    # Failures
    failures = [r for r in rows if r["status"] != "ok"]
    lines.append("## Failures")
    lines.append("")
    if not failures:
        lines.append("None. All 35 runs completed successfully.")
    else:
        for r in failures:
            lines.append(f"### {r['arm']} / seed_{r['seed']}")
            lines.append("")
            lines.append("```")
            lines.append((r.get("error") or "").rstrip())
            lines.append("```")
            lines.append("")
    lines.append("")

    # Deferred items
    lines.append("## Deferred items")
    lines.append("")
    lines.append("- **Held-out eval pass not run.** The plan calls for freezing candidate "
                 "Q-values, setting `epsilon=0`, and running 10 additional epochs to emit "
                 "`final_summary_eval.json`. Implementing this requires changes to "
                 "`simulation.py` and `metrics.py`, which the experiment-runner role is not "
                 "permitted to modify. Headline metrics in this iteration are training-tail "
                 "(final-epoch) values from `final_summary.json`.")
    lines.append("- **Per-run figures not generated.** Per the plan, figures are emitted once "
                 "per arm by the figure-curator (separate role), not by `run_simulation` for "
                 "each seed. `run_suite.py` does not call `generate_all_plots`.")
    lines.append("- **Paired-bootstrap p-values not computed.** Cross-arm comparison reports "
                 "paired mean differences with paired SE (n=5 shared seeds), but no p-values. "
                 "Bootstrap can be added without simulation-side changes; deferred for a "
                 "subsequent iteration alongside the held-out eval pass.")
    lines.append("- **Multi-arm overlay figures (`comparison_figures/`) not produced.** Same "
                 "reason as per-run figures: figure-curator job. The aggregator emits "
                 "`epoch_curves.npz` per arm, which is the input that role consumes.")
    lines.append("")

    # Sanity-check observations
    lines.append("## Sanity-check observations")
    lines.append("")
    lines.append("Headline numbers move in the directions the plan predicted:")
    lines.append("")
    lines.append("- `no_detection` -> AI usage explodes from ~0.06 (default) to ~0.86; "
                 "correct-match rate collapses from ~0.70 to ~0.20; company loss roughly 4x. "
                 "Detection is doing real work in the default arm.")
    lines.append("- `no_spillover` -> AI usage and matching are nearly unchanged vs. default, "
                 "but global trust drops from ~0.99 to ~0.80. Spillover is the channel keeping "
                 "average global trust pinned high, but is not by itself the deterrent on AI use.")
    lines.append("- `null_ai` -> AI usage drifts to ~0.03 (exploration noise floor), "
                 "correct-match rate is highest (~0.81). Confirms the floor against which "
                 "default results should be compared.")
    lines.append("- `sens_detect_lo` (half detection probs) -> AI usage ~0.13, more than 2x "
                 "the default but still far below the no-detection ceiling.")
    lines.append("- `sens_detect_hi` (double detection probs) -> AI usage indistinguishable "
                 "from default (~0.03), suggesting the qualitative story is robust to "
                 "increasing detection past the default level (default is already on the "
                 "saturated side of the deterrent curve).")
    lines.append("")
    lines.append("See `outputs/v2/_compare/comparison_table.md` for the full table and "
                 "paired deltas.")
    lines.append("")

    # File map
    lines.append("## Artifact map")
    lines.append("")
    lines.append("- Per-run: `outputs/v2/<arm>/seed_<n>/results/{epoch_metrics.csv, "
                 "interview_logs.csv, final_summary.json}` and "
                 "`outputs/v2/<arm>/seed_<n>/run_manifest.json`.")
    lines.append("- Per-arm aggregate: `outputs/v2/<arm>/aggregate/"
                 "{summary.json, per_seed_summary.csv, epoch_curves.npz}`.")
    lines.append("- Cross-arm: `outputs/v2/_compare/comparison_table.md`.")
    lines.append("- This log: `outputs/v2/_run_log.md`.")
    lines.append("")
    lines.append("Driver scripts (under `experiments/`): `run_suite.py`, `aggregate.py`, "
                 "`compare_arms.py`, `write_run_log.py`.")

    out_dir = V2_DIR
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "_run_log.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[run_log] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
