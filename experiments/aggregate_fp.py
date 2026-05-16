"""
Aggregate FP-style results across arms into comparison tables.

Reads the structure produced by run_fp_suite.run_arm() — including multi-seed
regret/convergence summaries — and emits comparison_table.csv plus a concise
markdown table.
"""

import os
import csv


def aggregate_results(results, compare_dir):
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
        row["r_max_mean"] = f"{regret['r_max']:.4f}"
        row["r_max_max"] = f"{regret['r_max_max']:.4f}"
        row["is_converged"] = str(convergence["is_converged"])
        row["all_seeds_stable"] = str(convergence["all_seeds_strategy_stable"])
        row["max_delta_max"] = f"{convergence['max_delta_max']:.4f}"
        row["flips_in_window_max"] = str(convergence["flips_in_window_max"])

        for k in key_metrics:
            if k in summary:
                m = summary[k]
                row[k] = f"{m['mean']:.4f}"
                row[f"{k}_se"] = f"{m['se']:.4f}" if isinstance(m.get('se'), float) and m['se'] == m['se'] else "nan"
                row[f"{k}_ci"] = f"[{m['ci_lo']:.4f}, {m['ci_hi']:.4f}]"
            else:
                row[k] = "N/A"
                row[f"{k}_se"] = "N/A"
                row[f"{k}_ci"] = "N/A"

        for m_name in ["LowVerify", "BaseVerify", "HighVerify"]:
            key = f"firm_policy_{m_name.lower()}"
            row[f"firm_{m_name}"] = (
                f"{summary[key]['mean']:.4f}" if key in summary else "N/A"
            )

        # Suspicion marginal
        for label in ["firm_suspicion_10", "firm_suspicion_50", "firm_suspicion_90"]:
            row[label] = f"{summary[label]['mean']:.4f}" if label in summary else "N/A"

        rows.append(row)

    if not rows:
        return

    all_keys = list(rows[0].keys())
    csv_path = os.path.join(compare_dir, "comparison_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)

    md_cols = [
        "arm", "r_max_mean", "r_max_max", "is_converged", "flips_in_window_max",
        "ai_adoption_overall", "ai_adoption_low", "ai_adoption_medium", "ai_adoption_high",
        "correct_match_rate", "candidate_welfare", "firm_welfare", "separating_index",
        "firm_LowVerify", "firm_BaseVerify", "firm_HighVerify",
        "firm_suspicion_10", "firm_suspicion_50", "firm_suspicion_90",
    ]
    md_cols = [c for c in md_cols if c in all_keys]

    md_path = os.path.join(compare_dir, "comparison_table.md")
    with open(md_path, "w") as f:
        f.write("# FP-style Arm Comparison\n\n")
        f.write("CIs use Student-t critical value with df = num_training_seeds - 1.\n")
        f.write("`is_converged` requires all seeds strategy-stable AND mean r_max < threshold.\n\n")
        header = " | ".join(md_cols)
        sep = " | ".join(["---"] * len(md_cols))
        f.write(f"| {header} |\n")
        f.write(f"| {sep} |\n")
        for row in rows:
            line = " | ".join(str(row.get(c, "N/A")) for c in md_cols)
            f.write(f"| {line} |\n")

    print(f"Comparison table saved to {compare_dir}/")
    return rows
