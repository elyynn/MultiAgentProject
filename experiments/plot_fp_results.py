"""
Figures for the FP-style experiment.

Per-arm:
  - candidate trajectory (one line per type, multi-seed envelope as faint traces)
  - firm trajectory (verification marginal + suspicion marginal, multi-seed envelope)

Cross-arm:
  - AI adoption by type (mean ± t-CI)
  - match efficiency
  - regret
  - detection calibration
  - welfare
  - strategy-convergence overlays
  - firm verification + suspicion distributions
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "low": "#e74c3c",
    "medium": "#3498db",
    "high": "#2ecc71",
    "LowVerify": "#f39c12",
    "BaseVerify": "#8e44ad",
    "HighVerify": "#2980b9",
    "sus_low": "#16a085",
    "sus_mid": "#d35400",
    "sus_high": "#c0392b",
}
ARM_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2",
]


def _savefig(fig, path):
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-seed (single-trajectory) helpers — used by run_fp_baseline and per-seed dirs
# ---------------------------------------------------------------------------

def _plot_candidate_trajectory(arm_name, logs, out_dir):
    iters = [r["iteration"] for r in logs]
    low_ai = [r["sigma_c_low_ai"] for r in logs]
    med_ai = [r["sigma_c_medium_ai"] for r in logs]
    high_ai = [r["sigma_c_high_ai"] for r in logs]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iters, low_ai, color=COLORS["low"], label="Low type", lw=1.5)
    ax.plot(iters, med_ai, color=COLORS["medium"], label="Medium type", lw=1.5)
    ax.plot(iters, high_ai, color=COLORS["high"], label="High type", lw=1.5)
    ax.set_xlabel("FP iteration")
    ax.set_ylabel("P(use AI)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Candidate AI Adoption — {arm_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    _savefig(fig, os.path.join(fig_dir, "candidate_trajectory.png"))


def _plot_firm_trajectory(arm_name, logs, out_dir):
    iters = [r["iteration"] for r in logs]
    lv = [r["sigma_f_low_verify"] for r in logs]
    bv = [r["sigma_f_base_verify"] for r in logs]
    hv = [r["sigma_f_high_verify"] for r in logs]
    sl = [r["sigma_f_sus_low"] for r in logs]
    sm = [r["sigma_f_sus_mid"] for r in logs]
    sh = [r["sigma_f_sus_high"] for r in logs]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    axes[0].plot(iters, lv, color=COLORS["LowVerify"], label="LowVerify", lw=1.5)
    axes[0].plot(iters, bv, color=COLORS["BaseVerify"], label="BaseVerify", lw=1.5)
    axes[0].plot(iters, hv, color=COLORS["HighVerify"], label="HighVerify", lw=1.5)
    axes[0].set_title("Firm verification marginal")
    axes[0].set_xlabel("FP iteration")
    axes[0].set_ylabel("Empirical probability")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(iters, sl, color=COLORS["sus_low"], label="suspicion=0.10", lw=1.5)
    axes[1].plot(iters, sm, color=COLORS["sus_mid"], label="suspicion=0.50", lw=1.5)
    axes[1].plot(iters, sh, color=COLORS["sus_high"], label="suspicion=0.90", lw=1.5)
    axes[1].set_title("Firm AI-suspicion marginal")
    axes[1].set_xlabel("FP iteration")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"Firm Strategy — {arm_name}")
    plt.tight_layout()
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    _savefig(fig, os.path.join(fig_dir, "firm_trajectory.png"))


# ---------------------------------------------------------------------------
# Multi-seed envelope plots (per-arm)
# ---------------------------------------------------------------------------

def _multi_seed_trajectory(arm_name, all_logs, out_dir):
    """Overlay per-seed candidate AI adoption traces, then the across-seed mean as bold."""
    if not all_logs:
        return
    n_seeds = len(all_logs)
    iters = [r["iteration"] for r in all_logs[0]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    type_keys = ["sigma_c_low_ai", "sigma_c_medium_ai", "sigma_c_high_ai"]
    type_labels = ["Low type", "Medium type", "High type"]
    type_colors = [COLORS["low"], COLORS["medium"], COLORS["high"]]
    for ti, (key, label, color) in enumerate(zip(type_keys, type_labels, type_colors)):
        ax = axes[ti]
        traces = []
        for k, logs in enumerate(all_logs):
            vals = np.array([r[key] for r in logs])
            traces.append(vals)
            ax.plot(iters, vals, color=color, alpha=0.30, lw=1.0)
        traces = np.array(traces)
        mean = traces.mean(axis=0)
        ax.plot(iters, mean, color=color, lw=2.0, label=f"{label} (mean of {n_seeds})")
        ax.set_title(label)
        ax.set_xlabel("FP iteration")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        if ti == 0:
            ax.set_ylabel("P(use AI)")
        ax.legend(fontsize=8)
    fig.suptitle(f"Candidate AI Adoption (per-seed traces + mean) — {arm_name}")
    plt.tight_layout()
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    _savefig(fig, os.path.join(fig_dir, "candidate_trajectory_multiseed.png"))


# ---------------------------------------------------------------------------
# Cross-arm comparison
# ---------------------------------------------------------------------------

def _plot_ai_adoption_by_type(results, compare_dir):
    arms = list(results.keys())
    types_labels = ["Low", "Medium", "High"]
    type_keys = ["ai_adoption_low", "ai_adoption_medium", "ai_adoption_high"]

    n_arms = len(arms)
    x = np.arange(n_arms)
    width = 0.25
    offsets = np.array([-1, 0, 1]) * width

    fig, ax = plt.subplots(figsize=(max(8, n_arms * 1.5), 5))
    type_colors = [COLORS["low"], COLORS["medium"], COLORS["high"]]
    for ti, (label, key) in enumerate(zip(types_labels, type_keys)):
        means, ci_lo, ci_hi = [], [], []
        for arm in arms:
            s = results[arm]["eval_summary"].get(key, {})
            means.append(s.get("mean", 0))
            ci_lo.append(s.get("ci_lo", 0))
            ci_hi.append(s.get("ci_hi", 0))
        means = np.array(means)
        err_lo = np.maximum(0, means - np.array(ci_lo))
        err_hi = np.maximum(0, np.array(ci_hi) - means)
        yerr = np.array([err_lo, err_hi])
        ax.bar(x + offsets[ti], means, width,
               label=f"{label} type", color=type_colors[ti],
               alpha=0.85, yerr=yerr, capsize=3, error_kw={"elinewidth": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("fp_", "") for a in arms], rotation=25, ha="right")
    ax.set_ylabel("P(use AI)")
    ax.set_ylim(0, 1.1)
    ax.set_title("Final AI Adoption by Type Across Arms (mean ± t-CI across training seeds)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _savefig(fig, os.path.join(compare_dir, "ai_adoption_by_type.png"))


def _plot_match_efficiency(results, compare_dir):
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
        err_lo = np.maximum(0, means - np.array(ci_lo))
        err_hi = np.maximum(0, np.array(ci_hi) - means)
        ax.bar(x + offsets[ti], means, width, label=label, color=color, alpha=0.85,
               yerr=np.array([err_lo, err_hi]), capsize=3, error_kw={"elinewidth": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("fp_", "") for a in arms], rotation=25, ha="right")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.1)
    ax.set_title("Match Efficiency Across Arms")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _savefig(fig, os.path.join(compare_dir, "match_efficiency_by_arm.png"))


def _plot_regret(results, compare_dir):
    arms = list(results.keys())
    r_max = [results[a]["regret"]["r_max"] for a in arms]
    r_firm = [results[a]["regret"]["r_firm"] for a in arms]
    r_low = [results[a]["regret"]["r_candidate_low"] for a in arms]
    r_med = [results[a]["regret"]["r_candidate_medium"] for a in arms]
    r_high = [results[a]["regret"]["r_candidate_high"] for a in arms]
    threshold = next(iter(results.values()))["cfg"].fp_regret_threshold

    x = np.arange(len(arms))
    fig, ax = plt.subplots(figsize=(max(8, len(arms) * 1.5), 5))
    ax.bar(x - 0.3, r_low, 0.15, label="Cand. Low", color=COLORS["low"], alpha=0.8)
    ax.bar(x - 0.15, r_med, 0.15, label="Cand. Med.", color=COLORS["medium"], alpha=0.8)
    ax.bar(x, r_high, 0.15, label="Cand. High", color=COLORS["high"], alpha=0.8)
    ax.bar(x + 0.15, r_firm, 0.15, label="Firm", color="#8e44ad", alpha=0.8)
    ax.plot(x, r_max, "k^", ms=7, label="r_max", zorder=5)
    ax.axhline(threshold, color="red", ls="--", lw=1.2, label=f"r_max threshold ({threshold})")
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("fp_", "") for a in arms], rotation=25, ha="right")
    ax.set_ylabel("Mean best-response regret across seeds")
    ax.set_title("Best-Response Regret by Arm")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    _savefig(fig, os.path.join(compare_dir, "regret_by_arm.png"))


def _plot_detection_calibration(results, compare_dir):
    arm = "fp_default" if "fp_default" in results else next(iter(results))
    cfg = results[arm]["cfg"]
    eval_summary = results[arm]["eval_summary"]

    types = ["Low", "Medium", "High"]
    type_keys = [
        "detection_rate_given_ai_low",
        "detection_rate_given_ai_medium",
        "detection_rate_given_ai_high",
    ]
    type_ids = [0, 1, 2]

    configured = [
        min(1.0, cfg.detection_multiplier["BaseVerify"] * cfg.base_detection_prob[t])
        for t in type_ids
    ]
    empirical = [eval_summary.get(k, {}).get("mean", float("nan")) for k in type_keys]
    err = [
        max(0, eval_summary.get(k, {}).get("ci_hi", 0) - eval_summary.get(k, {}).get("mean", 0))
        for k in type_keys
    ]

    x = np.arange(len(types))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - 0.18, configured, 0.35, label="Configured (BaseVerify)", color="#3498db", alpha=0.7)
    ax.bar(x + 0.18, empirical, 0.35, label="Empirical", color="#e74c3c", alpha=0.7,
           yerr=err, capsize=4, error_kw={"elinewidth": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(types)
    ax.set_ylabel("P(detected | AI used)")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Detection Calibration ({arm})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _savefig(fig, os.path.join(compare_dir, "detection_calibration.png"))


def _plot_welfare(results, compare_dir):
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
    ax.set_title("Welfare by Arm (mean across training seeds)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _savefig(fig, os.path.join(compare_dir, "welfare_by_arm.png"))


def _plot_strategy_convergence(results, compare_dir):
    """Overlay across-seed mean trajectories per arm, one panel per type."""
    type_keys = ["sigma_c_low_ai", "sigma_c_medium_ai", "sigma_c_high_ai"]
    type_labels = ["Low type", "Medium type", "High type"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ti, (key, label) in enumerate(zip(type_keys, type_labels)):
        ax = axes[ti]
        for ai, (arm_name, res) in enumerate(results.items()):
            all_logs = res.get("all_logs", [res["logs"]])
            traces = np.array([[r[key] for r in logs] for logs in all_logs])
            iters = [r["iteration"] for r in all_logs[0]]
            mean = traces.mean(axis=0)
            color = ARM_COLORS[ai % len(ARM_COLORS)]
            ax.plot(iters, mean, color=color, lw=1.4, alpha=0.95,
                    label=arm_name.replace("fp_", ""))
            if traces.shape[0] > 1:
                lo = traces.min(axis=0)
                hi = traces.max(axis=0)
                ax.fill_between(iters, lo, hi, color=color, alpha=0.10)
        ax.set_title(label)
        ax.set_xlabel("Iteration")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        if ti == 0:
            ax.set_ylabel("P(use AI)")
        if ti == 2:
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Candidate Strategy Trajectories Across Arms (mean ± per-seed envelope)")
    plt.tight_layout()
    _savefig(fig, os.path.join(compare_dir, "strategy_convergence.png"))


def _plot_firm_policy_distribution(results, compare_dir):
    arms = list(results.keys())
    policy_keys = ["firm_policy_lowverify", "firm_policy_baseverify", "firm_policy_highverify"]
    policy_labels = ["LowVerify", "BaseVerify", "HighVerify"]
    policy_colors = [COLORS["LowVerify"], COLORS["BaseVerify"], COLORS["HighVerify"]]

    x = np.arange(len(arms))
    bottoms = np.zeros(len(arms))

    fig, ax = plt.subplots(figsize=(max(8, len(arms) * 1.5), 5))
    for key, label, color in zip(policy_keys, policy_labels, policy_colors):
        vals = np.array([results[a]["eval_summary"].get(key, {}).get("mean", 0) for a in arms])
        ax.bar(x, vals, bottom=bottoms, label=label, color=color, alpha=0.85)
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("fp_", "") for a in arms], rotation=25, ha="right")
    ax.set_ylabel("Empirical probability (verification marginal)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Firm Verification Policy Distribution Across Arms")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _savefig(fig, os.path.join(compare_dir, "firm_policy_distribution.png"))


def _plot_firm_suspicion_distribution(results, compare_dir):
    arms = list(results.keys())
    sus_keys = ["firm_suspicion_10", "firm_suspicion_50", "firm_suspicion_90"]
    sus_labels = ["suspicion=0.10", "suspicion=0.50", "suspicion=0.90"]
    sus_colors = [COLORS["sus_low"], COLORS["sus_mid"], COLORS["sus_high"]]

    x = np.arange(len(arms))
    bottoms = np.zeros(len(arms))

    fig, ax = plt.subplots(figsize=(max(8, len(arms) * 1.5), 5))
    for key, label, color in zip(sus_keys, sus_labels, sus_colors):
        vals = np.array([results[a]["eval_summary"].get(key, {}).get("mean", 0) for a in arms])
        ax.bar(x, vals, bottom=bottoms, label=label, color=color, alpha=0.85)
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("fp_", "") for a in arms], rotation=25, ha="right")
    ax.set_ylabel("Empirical probability (suspicion marginal)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Firm AI-Suspicion Distribution Across Arms")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _savefig(fig, os.path.join(compare_dir, "firm_suspicion_distribution.png"))


def plot_all(results, compare_dir):
    # Per-arm
    for arm_name, res in results.items():
        cfg = res["cfg"]
        arm_dir = os.path.join(cfg.output_dir, arm_name)
        # Representative single-seed plot from seed 0
        _plot_candidate_trajectory(arm_name, res["logs"], arm_dir)
        _plot_firm_trajectory(arm_name, res["logs"], arm_dir)
        # Multi-seed envelope
        _multi_seed_trajectory(arm_name, res.get("all_logs", [res["logs"]]), arm_dir)

    # Cross-arm
    _plot_ai_adoption_by_type(results, compare_dir)
    _plot_match_efficiency(results, compare_dir)
    _plot_regret(results, compare_dir)
    _plot_detection_calibration(results, compare_dir)
    _plot_welfare(results, compare_dir)
    _plot_strategy_convergence(results, compare_dir)
    _plot_firm_policy_distribution(results, compare_dir)
    _plot_firm_suspicion_distribution(results, compare_dir)

    print(f"Figures saved to {compare_dir}/ and per-arm figures/ subdirectories.")
