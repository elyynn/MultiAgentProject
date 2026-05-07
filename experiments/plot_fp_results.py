"""
Produce all required figures for the fictitious-play experiment.

Figure 1: Candidate strategy trajectories (AI adoption by type) — per arm
Figure 2: Firm strategy trajectories — per arm
Figure 3: Final AI adoption by type across arms (bar chart with CI)
Figure 4: Match efficiency across arms
Figure 5: Best-response regret by arm
Figure 6: Detection calibration
Cross-arm: strategy convergence overlay
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


COLORS = {
    "low": "#e74c3c",
    "medium": "#3498db",
    "high": "#2ecc71",
    "LowVerify": "#f39c12",
    "BaseVerify": "#8e44ad",
    "HighVerify": "#2980b9",
}
ARM_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"
]


def _savefig(fig, path):
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-arm figures (saved into arm's figures/ subdirectory)
# ---------------------------------------------------------------------------

def _plot_candidate_trajectory(arm_name, logs, arm_dir):
    """Figure 1: candidate AI adoption over iterations."""
    iters = [r["iteration"] for r in logs]
    low_ai = [r["sigma_c_low_ai"] for r in logs]
    med_ai = [r["sigma_c_medium_ai"] for r in logs]
    high_ai = [r["sigma_c_high_ai"] for r in logs]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iters, low_ai, color=COLORS["low"], label="Low type", lw=1.5)
    ax.plot(iters, med_ai, color=COLORS["medium"], label="Medium type", lw=1.5)
    ax.plot(iters, high_ai, color=COLORS["high"], label="High type", lw=1.5)
    ax.set_xlabel("Fictitious-play iteration")
    ax.set_ylabel("P(use AI)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Candidate AI Adoption — {arm_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    _savefig(fig, os.path.join(arm_dir, "figures", "candidate_trajectory.png"))


def _plot_firm_trajectory(arm_name, logs, arm_dir):
    """Figure 2: firm policy distribution over iterations."""
    iters = [r["iteration"] for r in logs]
    lv = [r["sigma_f_low_verify"] for r in logs]
    bv = [r["sigma_f_base_verify"] for r in logs]
    hv = [r["sigma_f_high_verify"] for r in logs]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iters, lv, color=COLORS["LowVerify"], label="LowVerify", lw=1.5)
    ax.plot(iters, bv, color=COLORS["BaseVerify"], label="BaseVerify", lw=1.5)
    ax.plot(iters, hv, color=COLORS["HighVerify"], label="HighVerify", lw=1.5)
    ax.set_xlabel("Fictitious-play iteration")
    ax.set_ylabel("Empirical probability")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Firm Policy Distribution — {arm_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    _savefig(fig, os.path.join(arm_dir, "figures", "firm_trajectory.png"))


# ---------------------------------------------------------------------------
# Cross-arm comparison figures
# ---------------------------------------------------------------------------

def _plot_ai_adoption_by_type(results, compare_dir):
    """Figure 3: Final AI adoption by type across arms (bar chart + CI)."""
    arms = list(results.keys())
    types_labels = ["Low", "Medium", "High"]
    type_keys = ["ai_adoption_low", "ai_adoption_medium", "ai_adoption_high"]

    n_arms = len(arms)
    n_types = 3
    x = np.arange(n_arms)
    width = 0.25
    offsets = np.array([-1, 0, 1]) * width

    fig, ax = plt.subplots(figsize=(max(8, n_arms * 1.5), 5))

    for ti, (label, key) in enumerate(zip(types_labels, type_keys)):
        means, ci_lo, ci_hi = [], [], []
        for arm in arms:
            s = results[arm]["eval_summary"].get(key, {})
            means.append(s.get("mean", 0))
            ci_lo.append(s.get("ci_lo", 0))
            ci_hi.append(s.get("ci_hi", 0))

        means = np.array(means)
        err_lo = means - np.array(ci_lo)
        err_hi = np.array(ci_hi) - means
        yerr = np.array([err_lo, err_hi])

        ax.bar(
            x + offsets[ti], means, width,
            label=f"{label} type",
            color=list(COLORS.values())[ti],
            alpha=0.85,
            yerr=yerr,
            capsize=3,
            error_kw={"elinewidth": 1.2},
        )

    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("fp_", "") for a in arms], rotation=25, ha="right")
    ax.set_ylabel("P(use AI)")
    ax.set_ylim(0, 1.1)
    ax.set_title("Final AI Adoption by Type Across Arms")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _savefig(fig, os.path.join(compare_dir, "ai_adoption_by_type.png"))


def _plot_match_efficiency(results, compare_dir):
    """Figure 4: Match efficiency across arms."""
    arms = list(results.keys())
    metric_keys = ["correct_match_rate", "overoffer_rate", "underoffer_rate"]
    labels = ["Correct match", "Overoffer", "Underoffer"]
    bar_colors = ["#2ecc71", "#e74c3c", "#3498db"]

    n_arms = len(arms)
    x = np.arange(n_arms)
    width = 0.25
    offsets = np.array([-1, 0, 1]) * width

    fig, ax = plt.subplots(figsize=(max(8, n_arms * 1.5), 5))

    for ti, (key, label, color) in enumerate(zip(metric_keys, labels, bar_colors)):
        means, ci_lo, ci_hi = [], [], []
        for arm in arms:
            s = results[arm]["eval_summary"].get(key, {})
            means.append(s.get("mean", 0))
            ci_lo.append(s.get("ci_lo", 0))
            ci_hi.append(s.get("ci_hi", 0))

        means = np.array(means)
        err_lo = means - np.array(ci_lo)
        err_hi = np.array(ci_hi) - means

        ax.bar(
            x + offsets[ti], means, width,
            label=label, color=color, alpha=0.85,
            yerr=np.array([err_lo, err_hi]),
            capsize=3, error_kw={"elinewidth": 1.2},
        )

    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("fp_", "") for a in arms], rotation=25, ha="right")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.1)
    ax.set_title("Match Efficiency Across Arms")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _savefig(fig, os.path.join(compare_dir, "match_efficiency_by_arm.png"))


def _plot_regret(results, compare_dir):
    """Figure 5: Best-response regret by arm."""
    arms = list(results.keys())
    r_max = [results[a]["regret"]["r_max"] for a in arms]
    r_firm = [results[a]["regret"]["r_firm"] for a in arms]
    r_low = [results[a]["regret"]["r_candidate_low"] for a in arms]
    r_med = [results[a]["regret"]["r_candidate_medium"] for a in arms]
    r_high = [results[a]["regret"]["r_candidate_high"] for a in arms]

    x = np.arange(len(arms))
    fig, ax = plt.subplots(figsize=(max(8, len(arms) * 1.5), 5))
    ax.bar(x - 0.3, r_low, 0.15, label="Cand. Low", color=COLORS["low"], alpha=0.8)
    ax.bar(x - 0.15, r_med, 0.15, label="Cand. Med.", color=COLORS["medium"], alpha=0.8)
    ax.bar(x, r_high, 0.15, label="Cand. High", color=COLORS["high"], alpha=0.8)
    ax.bar(x + 0.15, r_firm, 0.15, label="Firm", color="#8e44ad", alpha=0.8)
    ax.plot(x, r_max, "k^", ms=7, label="r_max", zorder=5)
    ax.axhline(0.03, color="red", ls="--", lw=1.2, label="0.03 threshold")
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("fp_", "") for a in arms], rotation=25, ha="right")
    ax.set_ylabel("Best-response regret")
    ax.set_title("Best-Response Regret by Arm")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    _savefig(fig, os.path.join(compare_dir, "regret_by_arm.png"))


def _plot_detection_calibration(results, compare_dir):
    """Figure 6: Detection calibration — empirical vs configured rates."""
    arm = "fp_default"
    if arm not in results:
        arm = list(results.keys())[0]

    cfg = results[arm]["cfg"]
    eval_summary = results[arm]["eval_summary"]

    types = ["Low", "Medium", "High"]
    type_keys = [
        "detection_rate_given_ai_low",
        "detection_rate_given_ai_medium",
        "detection_rate_given_ai_high",
    ]
    type_ids = [0, 1, 2]

    # Configured baseline (BaseVerify)
    configured = [
        min(1.0, cfg.detection_multiplier["BaseVerify"] * cfg.base_detection_prob[t])
        for t in type_ids
    ]
    empirical = [eval_summary.get(k, {}).get("mean", float("nan")) for k in type_keys]
    err = [1.96 * eval_summary.get(k, {}).get("se", 0) for k in type_keys]

    x = np.arange(len(types))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - 0.18, configured, 0.35, label="Configured (BaseVerify)", color="#3498db", alpha=0.7)
    ax.bar(x + 0.18, empirical, 0.35, label="Empirical", color="#e74c3c", alpha=0.7,
           yerr=err, capsize=4, error_kw={"elinewidth": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(types)
    ax.set_ylabel("P(detected | AI used)")
    ax.set_ylim(0, 1.1)
    ax.set_title("Detection Calibration")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _savefig(fig, os.path.join(compare_dir, "detection_calibration.png"))


def _plot_welfare(results, compare_dir):
    """Bar chart of candidate/firm/total welfare across arms."""
    arms = list(results.keys())
    x = np.arange(len(arms))
    cand_means = [results[a]["eval_summary"].get("candidate_welfare", {}).get("mean", 0) for a in arms]
    firm_means = [results[a]["eval_summary"].get("firm_welfare", {}).get("mean", 0) for a in arms]
    total_means = [results[a]["eval_summary"].get("total_welfare", {}).get("mean", 0) for a in arms]

    fig, ax = plt.subplots(figsize=(max(8, len(arms) * 1.5), 5))
    ax.bar(x - 0.25, cand_means, 0.24, label="Candidate welfare", color="#2ecc71", alpha=0.85)
    ax.bar(x, firm_means, 0.24, label="Firm welfare", color="#e74c3c", alpha=0.85)
    ax.bar(x + 0.25, total_means, 0.24, label="Total welfare", color="#3498db", alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("fp_", "") for a in arms], rotation=25, ha="right")
    ax.set_ylabel("Expected utility")
    ax.set_title("Welfare by Arm")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _savefig(fig, os.path.join(compare_dir, "welfare_by_arm.png"))


def _plot_strategy_convergence(results, compare_dir):
    """Overlay candidate AI adoption trajectories across arms (one panel per type)."""
    type_keys = ["sigma_c_low_ai", "sigma_c_medium_ai", "sigma_c_high_ai"]
    type_labels = ["Low type", "Medium type", "High type"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ti, (key, label) in enumerate(zip(type_keys, type_labels)):
        ax = axes[ti]
        for ai, (arm_name, res) in enumerate(results.items()):
            logs = res["logs"]
            iters = [r["iteration"] for r in logs]
            vals = [r[key] for r in logs]
            ax.plot(iters, vals,
                    color=ARM_COLORS[ai % len(ARM_COLORS)],
                    label=arm_name.replace("fp_", ""),
                    lw=1.2, alpha=0.85)
        ax.set_title(label)
        ax.set_xlabel("Iteration")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        if ti == 0:
            ax.set_ylabel("P(use AI)")
        if ti == 2:
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Fictitious-Play Candidate Strategy Trajectories Across Arms")
    plt.tight_layout()
    _savefig(fig, os.path.join(compare_dir, "strategy_convergence.png"))


def _plot_firm_policy_distribution(results, compare_dir):
    """Stacked bar chart of final firm policy distribution across arms."""
    arms = list(results.keys())
    policy_keys = ["firm_policy_lowverify", "firm_policy_baseverify", "firm_policy_highverify"]
    policy_labels = ["LowVerify", "BaseVerify", "HighVerify"]
    policy_colors = [COLORS["LowVerify"], COLORS["BaseVerify"], COLORS["HighVerify"]]

    x = np.arange(len(arms))
    bottoms = np.zeros(len(arms))

    fig, ax = plt.subplots(figsize=(max(8, len(arms) * 1.5), 5))
    for key, label, color in zip(policy_keys, policy_labels, policy_colors):
        vals = np.array([
            results[a]["eval_summary"].get(key, {}).get("mean", 0)
            for a in arms
        ])
        ax.bar(x, vals, bottom=bottoms, label=label, color=color, alpha=0.85)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("fp_", "") for a in arms], rotation=25, ha="right")
    ax.set_ylabel("Empirical probability")
    ax.set_ylim(0, 1.05)
    ax.set_title("Firm Policy Distribution Across Arms")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _savefig(fig, os.path.join(compare_dir, "firm_policy_distribution.png"))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def plot_all(results, compare_dir):
    """Produce all figures. Per-arm figures go in arm dirs; cross-arm in compare_dir."""

    # Per-arm
    for arm_name, res in results.items():
        arm_dir = os.path.join(res["cfg"].output_dir, arm_name)
        fig_dir = os.path.join(arm_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        _plot_candidate_trajectory(arm_name, res["logs"], arm_dir)
        _plot_firm_trajectory(arm_name, res["logs"], arm_dir)

    # Cross-arm
    _plot_ai_adoption_by_type(results, compare_dir)
    _plot_match_efficiency(results, compare_dir)
    _plot_regret(results, compare_dir)
    _plot_detection_calibration(results, compare_dir)
    _plot_welfare(results, compare_dir)
    _plot_strategy_convergence(results, compare_dir)
    _plot_firm_policy_distribution(results, compare_dir)

    print(f"Figures saved to {compare_dir}/ and per-arm figures/ subdirectories.")
