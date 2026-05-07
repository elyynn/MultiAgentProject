"""
Aggregate fictitious-play results across arms into comparison tables.
"""

import os
import csv
import json


def aggregate_results(results, compare_dir):
    """
    Build comparison_table.csv and comparison_table.md from arm results.

    results: dict arm_name -> result dict from run_fp_suite.run_arm()
    compare_dir: output directory
    """
    rows = []
    key_metrics = [
        "ai_adoption_overall",
        "ai_adoption_low",
        "ai_adoption_medium",
        "ai_adoption_high",
        "correct_match_rate",
        "overoffer_rate",
        "underoffer_rate",
        "candidate_welfare",
        "firm_welfare",
        "total_welfare",
        "separating_index",
    ]

    for arm_name, res in results.items():
        summary = res["eval_summary"]
        regret = res["regret"]
        convergence = res["convergence"]

        row = {"arm": arm_name}
        row["r_max"] = f"{regret['r_max']:.4f}"
        row["is_converged"] = str(convergence["is_converged"])
        row["max_delta"] = f"{convergence['max_delta']:.4f}"

        for k in key_metrics:
            if k in summary:
                m = summary[k]
                row[k] = f"{m['mean']:.4f}"
                row[f"{k}_se"] = f"{m['se']:.4f}"
                row[f"{k}_ci"] = f"[{m['ci_lo']:.4f}, {m['ci_hi']:.4f}]"
            else:
                row[k] = "N/A"
                row[f"{k}_se"] = "N/A"
                row[f"{k}_ci"] = "N/A"

        # Firm policy distribution
        for m_name in ["LowVerify", "BaseVerify", "HighVerify"]:
            key = f"firm_policy_{m_name.lower()}"
            if key in summary:
                row[f"firm_{m_name}"] = f"{summary[key]['mean']:.4f}"
            else:
                row[f"firm_{m_name}"] = "N/A"

        rows.append(row)

    if not rows:
        return

    # CSV
    all_keys = list(rows[0].keys())
    csv_path = os.path.join(compare_dir, "comparison_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)

    # Markdown — concise table with key metrics only
    md_cols = ["arm", "r_max", "is_converged",
                "ai_adoption_overall", "ai_adoption_low", "ai_adoption_medium", "ai_adoption_high",
                "correct_match_rate", "candidate_welfare", "firm_welfare", "separating_index",
                "firm_LowVerify", "firm_BaseVerify", "firm_HighVerify"]
    md_cols = [c for c in md_cols if c in all_keys]

    md_path = os.path.join(compare_dir, "comparison_table.md")
    with open(md_path, "w") as f:
        f.write("# Fictitious-Play Arm Comparison\n\n")
        header = " | ".join(md_cols)
        sep = " | ".join(["---"] * len(md_cols))
        f.write(f"| {header} |\n")
        f.write(f"| {sep} |\n")
        for row in rows:
            line = " | ".join(str(row.get(c, "N/A")) for c in md_cols)
            f.write(f"| {line} |\n")

    print(f"Comparison table saved to {compare_dir}/")
    return rows
