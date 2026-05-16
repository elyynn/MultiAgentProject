"""Cross-arm comparison: writes outputs/v2/_compare/comparison_table.md.

For each headline metric, one row per arm, mean +/- sem (n=N).
For non-default arms, an indented line with paired difference vs. default
on the seeds shared by both arms (paired SE, no p-value this iteration).
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Dict, List, Tuple

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

HEADLINE_METRICS = [
    "final_average_ai_usage",
    "final_average_global_trust",
    "final_correct_match_rate",
    "final_overoffer_rate",
    "final_average_company_loss",
]


def _load_arm(arm: str) -> Dict | None:
    path = os.path.join(V2_DIR, arm, "aggregate", "summary.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _fmt_mean_sem(field: Dict) -> str:
    if field is None or field.get("n", 0) == 0:
        return "-"
    return f"{field['mean']:.4f} +/- {field['sem']:.4f} (n={field['n']})"


def _paired_diff(arm_field: Dict, default_field: Dict) -> Tuple[float, float, int] | None:
    if arm_field is None or default_field is None:
        return None
    a_per_seed = arm_field.get("per_seed", {}) or {}
    d_per_seed = default_field.get("per_seed", {}) or {}
    # Keys may be strings after JSON round-trip
    def _norm(d: Dict) -> Dict[int, float]:
        out = {}
        for k, v in d.items():
            try:
                out[int(k)] = float(v)
            except (TypeError, ValueError):
                pass
        return out
    a = _norm(a_per_seed)
    d = _norm(d_per_seed)
    shared = sorted(set(a.keys()) & set(d.keys()))
    if not shared:
        return None
    diffs = [a[s] - d[s] for s in shared]
    n = len(diffs)
    mean_d = sum(diffs) / n
    if n > 1:
        var = sum((x - mean_d) ** 2 for x in diffs) / (n - 1)
        sem_d = math.sqrt(var / n)
    else:
        sem_d = 0.0
    return mean_d, sem_d, n


def main() -> int:
    arms_data: Dict[str, Dict | None] = {a: _load_arm(a) for a in ARMS}
    default = arms_data.get("default")

    out_dir = os.path.join(V2_DIR, "_compare")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "comparison_table.md")

    lines: List[str] = []
    lines.append("# Cross-arm comparison")
    lines.append("")
    lines.append(
        "Each cell: `mean +/- sem (n=N)` over per-seed final-epoch (training-tail) values. "
        "For non-default arms, the indented `delta` line is the paired mean difference "
        "vs. `default` on shared seeds (paired SE; no bootstrap p-values this iteration)."
    )
    lines.append("")

    header = ["arm"] + HEADLINE_METRICS
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for arm in ARMS:
        a = arms_data.get(arm)
        if a is None:
            continue
        fields = a.get("fields", {})
        cells = [arm]
        for m in HEADLINE_METRICS:
            cells.append(_fmt_mean_sem(fields.get(m)))
        lines.append("| " + " | ".join(cells) + " |")

        if arm != "default" and default is not None:
            d_fields = default.get("fields", {})
            diff_cells = ["&nbsp;&nbsp;delta vs default"]
            for m in HEADLINE_METRICS:
                diff = _paired_diff(fields.get(m), d_fields.get(m))
                if diff is None:
                    diff_cells.append("-")
                else:
                    md, sd, n = diff
                    diff_cells.append(f"{md:+.4f} +/- {sd:.4f} (n={n})")
            lines.append("| " + " | ".join(diff_cells) + " |")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Headline metrics are training-tail (final epoch). Held-out eval pass was "
                 "deferred this iteration because it requires modifying `simulation.py`/`metrics.py`.")
    lines.append("- All seeds shared across arms 2-6 are the first five from the registered seed list "
                 "`[42, 123, 2024, 7, 1337, 99, 314, 271, 8675309, 555]`; arm `default` uses all ten.")
    lines.append("- Paired delta uses only the seeds present in both arms (typically n=5).")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[compare] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
