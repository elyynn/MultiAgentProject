"""Driver: produce all v2 figures (paper + slides) from outputs/v2/.

Reads only from ``outputs/v2/<arm>/aggregate/*`` and ``outputs/v2/<arm>/seed_*/results/interview_logs.csv``.
Writes to ``outputs/v2/figures/{paper,slides,captions}/``.

CLI
---
    python figures/make_figures.py [--variant paper|slides|both] [--figs F1,F2,...]

Defaults: ``--variant both`` produces every figure in both variants (10 figs x 2 = 20 PNGs + 10 PDFs + 10 captions).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import style  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

V2 = ROOT / "outputs" / "v2"
FIG_OUT = V2 / "figures"
PAPER_DIR = FIG_OUT / "paper"
SLIDES_DIR = FIG_OUT / "slides"
CAP_DIR = FIG_OUT / "captions"
for _d in (PAPER_DIR, SLIDES_DIR, CAP_DIR):
    _d.mkdir(parents=True, exist_ok=True)


ARMS = ["default", "no_detection", "no_spillover", "null_ai",
        "sens_detect_lo", "sens_detect_hi"]

# Detection probabilities by type from cfg (config.py:98-102).
DETECTION_PROB_BY_TYPE = {0: 0.35, 1: 0.20, 2: 0.10}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_curves(arm: str) -> dict:
    p = V2 / arm / "aggregate" / "epoch_curves.npz"
    with np.load(p) as d:
        return {k: d[k].copy() for k in d.files}


def load_summary(arm: str) -> dict:
    with open(V2 / arm / "aggregate" / "summary.json") as f:
        return json.load(f)


def list_seeds(arm: str) -> list[int]:
    """Discover seed directories present for an arm (sorted by int)."""
    seeds = []
    for p in (V2 / arm).glob("seed_*"):
        if p.is_dir():
            try:
                seeds.append(int(p.name.split("_", 1)[1]))
            except ValueError:
                pass
    return sorted(seeds)


def iter_interview_logs(arm: str, seed: int):
    """Yield dict rows from an interview log CSV (parsed scalars)."""
    p = V2 / arm / f"seed_{seed}" / "results" / "interview_logs.csv"
    with open(p, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def load_interview_log_arrays(arm: str, seed: int) -> dict:
    """Load one seed's interview log into numpy arrays (only the cols we need)."""
    epochs, true_types, ai_actions, offers, detected = [], [], [], [], []
    for row in iter_interview_logs(arm, seed):
        epochs.append(int(row["epoch"]))
        true_types.append(int(row["true_type"]))
        ai_actions.append(int(row["ai_action"]))
        offers.append(int(row["offer"]))
        detected.append(int(row["detected"]))
    return {
        "epoch":     np.asarray(epochs, dtype=np.int64),
        "true_type": np.asarray(true_types, dtype=np.int64),
        "ai_action": np.asarray(ai_actions, dtype=np.int64),
        "offer":     np.asarray(offers, dtype=np.int64),
        "detected":  np.asarray(detected, dtype=np.int64),
    }


# ---------------------------------------------------------------------------
# Caption writer
# ---------------------------------------------------------------------------

def write_caption(fig_id: str, short_name: str, body_md: str) -> None:
    path = CAP_DIR / f"{fig_id}_{short_name}.md"
    path.write_text(body_md.strip() + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Figure functions
# ---------------------------------------------------------------------------

def _save(fig, variant: str, fig_id: str, short_name: str) -> None:
    base = f"{fig_id}_{short_name}"
    if variant == "paper":
        fig.savefig(PAPER_DIR / f"{base}.pdf")
        fig.savefig(PAPER_DIR / f"{base}.png")
    elif variant == "slides":
        fig.savefig(SLIDES_DIR / f"{base}.png")
    plt.close(fig)


# ----- F1: AI adoption over time (default arm) -----
def fig_F1(variant: str) -> None:
    style.apply_variant(variant)
    curves = load_curves("default")
    epochs = curves["epoch"][0]  # all rows identical (epoch index)
    n_seeds = curves["seeds"].shape[0]

    fig, ax = plt.subplots(figsize=style.panel_figsize(variant))
    style.add_ci_line(ax, epochs, curves["average_ai_usage"],
                      style.OVERALL_COLOR, label="Overall")
    for t in style.TYPE_ORDER:
        style.add_ci_line(ax, epochs, curves[f"average_ai_usage_{['low','medium','high'][t]}"],
                          style.TYPE_COLORS[t], label=style.TYPE_LABELS[t])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Fraction using AI")
    ax.set_title(f"AI adoption over time (default, N={n_seeds} seeds, mean +/- 95% CI)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    _save(fig, variant, "F1", "ai_adoption_over_time")


def cap_F1():
    write_caption("F1", "ai_adoption_over_time",
        """
# F1 - AI adoption over time

- **Source artifact**: `outputs/v2/default/aggregate/epoch_curves.npz`
  (columns `average_ai_usage`, `average_ai_usage_low/medium/high`)
- **N seeds**: 10 (`default` arm)
- **Config diff**: none (baseline)
- **What it shows**: Fraction of candidates choosing the AI action across the
  100 training epochs, broken down by `true_type` and shown overall. Shaded
  bands are 95% CI across seeds (1.96 * SEM).
- **Claim supported**: AI use settles at a low overall rate (~6%) and is
  carried almost entirely by the Medium-type candidates - High-type rarely
  benefits, Low-type is detected too easily.
""")


# ----- F2: Detection rate over time (default arm) -----
def fig_F2(variant: str) -> None:
    style.apply_variant(variant)
    curves = load_curves("default")
    epochs = curves["epoch"][0]
    n_seeds = curves["seeds"].shape[0]

    fig, ax = plt.subplots(figsize=style.panel_figsize(variant))
    style.add_ci_line(ax, epochs, curves["detection_rate"],
                      style.OVERALL_COLOR, label="Overall")
    for t in style.TYPE_ORDER:
        style.add_ci_line(ax, epochs, curves[f"detection_rate_{['low','medium','high'][t]}"],
                          style.TYPE_COLORS[t], label=style.TYPE_LABELS[t])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Detection rate (per candidate, all rounds)")
    ax.set_title(f"Detection rate over time (default, N={n_seeds}, aggregate)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    _save(fig, variant, "F2", "detection_rate_over_time")


def cap_F2():
    write_caption("F2", "detection_rate_over_time",
        """
# F2 - Detection rate over time (aggregate)

- **Source artifact**: `outputs/v2/default/aggregate/epoch_curves.npz`
  (columns `detection_rate`, `detection_rate_low/medium/high`)
- **N seeds**: 10 (`default` arm)
- **Config diff**: none (baseline)
- **What it shows**: Per-epoch detection rate (detected / all candidates) overall
  and by `true_type`, mean +/- 95% CI across seeds.
- **Important caveat**: This is the *aggregate* rate (denominator = all
  candidates, including non-AI users). When a type's AI-usage is near zero
  the per-type curve is dominated by Monte Carlo noise. The
  AI-conditioned rate (proper validation that the simulator's
  `detection_prob_by_type` parameter is recovered) is in **F8**.
- **Claim supported**: Companion to F8; do not use this figure alone to argue
  that detection works.
""")


# ----- F3: Firm trust over time (two panels, separate y-axes) -----
def fig_F3(variant: str) -> None:
    style.apply_variant(variant)
    curves = load_curves("default")
    epochs = curves["epoch"][0]
    n_seeds = curves["seeds"].shape[0]

    fig, axes = plt.subplots(1, 2, figsize=style.panel_figsize(variant, ncols=2))
    ax1, ax2 = axes
    style.add_ci_line(ax1, epochs, curves["average_individual_trust"],
                      style.ARM_COLORS["default"], label="individual trust")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Average individual trust")
    ax1.set_title("Individual trust (per-candidate posterior)")
    ax1.legend(loc="best", frameon=False)

    style.add_ci_line(ax2, epochs, curves["average_global_trust"],
                      style.ARM_COLORS["no_detection"], label="global trust")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Average global trust")
    ax2.set_title("Global trust (firm-level)")
    ax2.legend(loc="best", frameon=False)

    fig.suptitle(f"Firm trust over time (default, N={n_seeds}, mean +/- 95% CI)",
                 y=1.02)
    fig.tight_layout()
    _save(fig, variant, "F3", "trust_over_time")


def cap_F3():
    write_caption("F3", "trust_over_time",
        """
# F3 - Firm trust over time

- **Source artifact**: `outputs/v2/default/aggregate/epoch_curves.npz`
  (columns `average_individual_trust`, `average_global_trust`)
- **N seeds**: 10 (`default` arm)
- **Config diff**: none (baseline)
- **What it shows**: Two panels, **separate y-axes** because the two trust
  scalars have different semantics (individual is per-candidate posterior;
  global is firm-level scalar). Mean +/- 95% CI across seeds.
- **Claim supported**: Resolves pilot pitfall P4 (sharing a y-axis was
  misleading). Both metrics should be reported, but global trust should
  not be the headline (audit B2).
""")


# ----- F4: Losses over time -----
def fig_F4(variant: str) -> None:
    style.apply_variant(variant)
    curves = load_curves("default")
    epochs = curves["epoch"][0]
    n_seeds = curves["seeds"].shape[0]

    fig, axes = plt.subplots(1, 2, figsize=style.panel_figsize(variant, ncols=2))
    ax1, ax2 = axes

    # Sign-flip: candidate loss is negative; "utility" reads more naturally.
    candidate_utility = -curves["average_candidate_loss"]
    style.add_ci_line(ax1, epochs, candidate_utility,
                      style.ARM_COLORS["null_ai"],
                      label="candidate utility = -loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Average candidate utility (negative loss)")
    ax1.set_title("Candidate utility over time")
    ax1.legend(loc="best", frameon=False)
    ax1.axhline(0, color="black", linewidth=0.6, alpha=0.4)

    style.add_ci_line(ax2, epochs, curves["average_company_loss"],
                      style.ARM_COLORS["no_detection"], label="company loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Average company loss")
    ax2.set_title("Company loss over time")
    ax2.legend(loc="best", frameon=False)

    fig.suptitle(f"Losses over time (default, N={n_seeds}, mean +/- 95% CI)",
                 y=1.02)
    fig.tight_layout()
    _save(fig, variant, "F4", "losses_over_time")


def cap_F4():
    write_caption("F4", "losses_over_time",
        """
# F4 - Losses over time

- **Source artifact**: `outputs/v2/default/aggregate/epoch_curves.npz`
  (columns `average_candidate_loss`, `average_company_loss`)
- **N seeds**: 10 (`default` arm)
- **Config diff**: none (baseline)
- **What it shows**: Left panel: candidate **utility** = -`average_candidate_loss`,
  so positive is good. Right panel: average company loss (positive = bad).
  Bands are 95% CI across seeds.
- **Claim supported**: Closes audit W3/W6 (sign-convention honesty). The
  candidate loss in the raw artifact is negative; relabeling as utility
  prevents the pilot's confusion.
""")


# ----- F5: Market efficiency (correct/over/under) -----
def fig_F5(variant: str) -> None:
    style.apply_variant(variant)
    curves = load_curves("default")
    epochs = curves["epoch"][0]
    n_seeds = curves["seeds"].shape[0]
    null_curves = load_curves("null_ai")
    null_correct_mean = null_curves["correct_match_rate"].mean(axis=0)[-10:].mean()
    null_n = null_curves["seeds"].shape[0]

    fig, ax = plt.subplots(figsize=style.panel_figsize(variant))
    style.add_ci_line(ax, epochs, curves["correct_match_rate"],
                      "#009E73", label="correct match")
    style.add_ci_line(ax, epochs, curves["overoffer_rate"],
                      "#D55E00", label="over-offer")
    style.add_ci_line(ax, epochs, curves["underoffer_rate"],
                      "#0072B2", label="under-offer")

    ax.axhline(null_correct_mean, color="#009E73", linestyle="--",
               linewidth=1.2, alpha=0.7,
               label=f"null_ai correct floor = {null_correct_mean:.3f} (N={null_n})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Rate")
    ax.set_title(f"Market efficiency over time (default, N={n_seeds}, mean +/- 95% CI)")
    ax.set_ylim(0, 1)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    _save(fig, variant, "F5", "market_efficiency_over_time")


def cap_F5():
    write_caption("F5", "market_efficiency_over_time",
        """
# F5 - Market efficiency over time

- **Source artifacts**:
  - `outputs/v2/default/aggregate/epoch_curves.npz`
    (columns `correct_match_rate`, `overoffer_rate`, `underoffer_rate`)
  - `outputs/v2/null_ai/aggregate/epoch_curves.npz`
    (columns `correct_match_rate`, last-10-epoch mean)
- **N seeds**: 10 (default), 5 (null_ai)
- **Config diff**: `null_ai` disables the AI action entirely.
- **What it shows**: Three rates over training; bands are 95% CI across the
  10 default seeds. The dashed horizontal line is the `null_ai` correct-match
  floor: with no AI available, the firm's own classifier achieves a higher
  match rate, so AI in the `default` setting actually *hurts* matching
  efficiency.
- **Claim supported**: Closes audit B2 (correct-match should not be reported
  in isolation - context against the no-AI floor matters).
""")


# ----- F6: AI usage trajectory across all arms -----
def fig_F6(variant: str) -> None:
    style.apply_variant(variant)
    fig, ax = plt.subplots(figsize=style.panel_figsize(variant))
    epochs_ref = None
    n_per_arm = {}
    for arm in style.ARM_ORDER:
        c = load_curves(arm)
        if epochs_ref is None:
            epochs_ref = c["epoch"][0]
        n_per_arm[arm] = c["seeds"].shape[0]
        style.add_ci_line(ax, epochs_ref, c["average_ai_usage"],
                          style.ARM_COLORS[arm],
                          label=f"{style.ARM_LABELS[arm]} (N={n_per_arm[arm]})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Fraction using AI")
    ax.set_title("AI usage trajectory by arm (mean +/- 95% CI)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    _save(fig, variant, "F6", "ai_usage_all_arms")


def cap_F6():
    write_caption("F6", "ai_usage_all_arms",
        """
# F6 - AI usage trajectory across arms (headline)

- **Source artifacts**: `outputs/v2/<arm>/aggregate/epoch_curves.npz`
  for arms: `default`, `null_ai`, `no_detection`, `no_spillover`,
  `sens_detect_lo`, `sens_detect_hi`.
- **N seeds**: 10 for `default`, 5 each for the five ablation arms.
- **Config diff**: each ablation flips one knob in `cfg`
  (`default` baseline; `no_detection` -> detection prob = 0;
  `no_spillover` -> trust spillover off; `null_ai` -> AI action disabled;
  `sens_detect_{lo,hi}` -> detection probabilities scaled).
- **What it shows**: AI adoption rate over training, one mean +/- 95% CI band
  per arm.
- **Claim supported**: This is the headline figure of the suite. It cleanly
  separates regimes: removing detection -> nearly everyone uses AI;
  baseline + sensitivity arms cluster low (<= 15%); removing spillover roughly
  doubles default AI use; `null_ai` is an instrumentation sanity check (~3%
  reflects the small floor introduced by epsilon-exploration before the
  no-AI mask is applied).
""")


# ----- F7: Headline scalars across arms (5-panel bar chart) -----
def fig_F7(variant: str) -> None:
    style.apply_variant(variant)
    fields = [
        ("final_average_ai_usage",      "AI usage"),
        ("final_average_global_trust",  "global trust"),
        ("final_correct_match_rate",    "correct match rate"),
        ("final_overoffer_rate",        "over-offer rate"),
        ("final_average_company_loss",  "company loss"),
    ]
    summaries = {arm: load_summary(arm) for arm in style.ARM_ORDER}
    arms = style.ARM_ORDER

    fig, axes = plt.subplots(
        1, 5,
        figsize=style.panel_figsize(variant, ncols=5, width_scale=0.55),
        sharey=False,
    )
    for ax, (key, title) in zip(axes, fields):
        means = [summaries[a]["fields"][key]["mean"] for a in arms]
        sems  = [summaries[a]["fields"][key]["sem"]  for a in arms]
        colors = [style.ARM_COLORS[a] for a in arms]
        bars = ax.bar(range(len(arms)), means, yerr=sems, color=colors,
                      capsize=3, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([style.ARM_LABELS[a] for a in arms],
                           rotation=45, ha="right")
        ax.set_title(title)
        ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)

    fig.suptitle("Headline scalars across arms (final epoch, mean +/- SEM)",
                 y=1.02)
    fig.tight_layout()
    _save(fig, variant, "F7", "headline_scalars_across_arms")


def cap_F7():
    write_caption("F7", "headline_scalars_across_arms",
        """
# F7 - Headline scalars across arms

- **Source artifact**: `outputs/v2/<arm>/aggregate/summary.json` for all six arms.
- **N seeds**: 10 (`default`), 5 (each ablation).
- **Config diff**: see F6 caption.
- **What it shows**: Five-panel bar chart of the headline final-epoch scalars
  (AI usage, global trust, correct-match rate, over-offer rate, company loss),
  one bar per arm, error bars = SEM across seeds. Color per arm matches F6.
- **Claim supported**: Compact summary of the cross-arm comparison table
  (`outputs/v2/_compare/comparison_table.md`); supports the result that
  removing detection drives the system into a high-AI / high-loss regime,
  while sensitivity arms behave similarly to `default`.
""")


# ----- F8: AI-conditioned detection rate by type -----
def _ai_conditioned_detection(arm: str, last_n_epochs: int = 10):
    """Per-seed empirical Pr(detected | ai_action=1, true_type=t), restricted to
    the last ``last_n_epochs`` epochs.

    Returns dict ``{type: list_of_per_seed_rates}``.
    """
    seeds = list_seeds(arm)
    out = {0: [], 1: [], 2: []}
    for s in seeds:
        log = load_interview_log_arrays(arm, s)
        epoch_max = int(log["epoch"].max())
        cutoff = epoch_max - last_n_epochs + 1
        m_window = log["epoch"] >= cutoff
        m_ai = log["ai_action"] == 1
        for t in (0, 1, 2):
            m = m_window & m_ai & (log["true_type"] == t)
            n = int(m.sum())
            if n == 0:
                out[t].append(np.nan)
            else:
                out[t].append(float(log["detected"][m].mean()))
    return out, seeds


def fig_F8(variant: str) -> None:
    style.apply_variant(variant)
    rates, seeds = _ai_conditioned_detection("default", last_n_epochs=10)

    fig, ax = plt.subplots(figsize=style.panel_figsize(variant))
    types = style.TYPE_ORDER
    means = [np.nanmean(rates[t]) if len(rates[t]) > 0 else np.nan for t in types]
    sems = []
    for t in types:
        arr = np.asarray(rates[t], dtype=float)
        n_valid = int(np.sum(~np.isnan(arr)))
        if n_valid > 1:
            sems.append(np.nanstd(arr, ddof=1) / np.sqrt(n_valid))
        else:
            sems.append(0.0)
    colors = [style.TYPE_COLORS[t] for t in types]
    xpos = np.arange(len(types))
    ax.bar(xpos, means, yerr=sems, color=colors, capsize=4,
           edgecolor="black", linewidth=0.5,
           label="empirical Pr(detected | AI used)")

    # Annotate the configured detection_prob_by_type for each type.
    for t, x in zip(types, xpos):
        target = DETECTION_PROB_BY_TYPE[t]
        ax.hlines(target, x - 0.4, x + 0.4, colors="black",
                  linestyles="dashed", linewidth=1.2)
    ax.plot([], [], color="black", linestyle="dashed",
            label="cfg.detection_prob_by_type (target)")

    ax.set_xticks(xpos)
    ax.set_xticklabels([style.TYPE_LABELS[t] for t in types])
    ax.set_xlabel("Candidate type")
    ax.set_ylabel("Pr(detected | AI used)")
    ax.set_ylim(0, max(0.45, max(DETECTION_PROB_BY_TYPE.values()) + 0.1))
    ax.set_title(
        f"AI-conditioned detection rate by type "
        f"(default, last 10 epochs, N={len(seeds)} seeds)"
    )
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    _save(fig, variant, "F8", "ai_conditioned_detection_by_type")


def cap_F8():
    write_caption("F8", "ai_conditioned_detection_by_type",
        """
# F8 - AI-conditioned detection rate by type (replaces W2)

- **Source artifact**: `outputs/v2/default/seed_*/results/interview_logs.csv`
  (filter rows: `ai_action == 1`, last 10 epochs of each seed).
- **N seeds**: 10 (`default` arm).
- **Config diff**: none (baseline).
- **What it shows**: Empirical detection probability conditioned on the AI
  having actually been used, grouped by `true_type`, averaged across seeds.
  Bars: mean +/- SEM across seeds. Dashed black tick at each bar = the value
  set in `cfg.detection_prob_by_type` (Low=0.35, Medium=0.20, High=0.10).
- **Claim supported**: Closes audit W2 - the simulator's per-type detection
  parameters *are* recovered when the right denominator is used. The pilot's
  "detection rate by type" plot used the all-candidates denominator, where
  rare AI usage made the per-type curves dominated by Monte Carlo noise.
""")


# ----- F9: True offer distribution by type -----
def _offer_distribution(arm: str, last_n_epochs: int = 10):
    """Per-seed empirical fractions of offer in {0,1,2} by true_type, last N epochs.

    Returns dict ``{type: {offer_cat: list_per_seed}}``.
    """
    seeds = list_seeds(arm)
    out = {t: {0: [], 1: [], 2: []} for t in (0, 1, 2)}
    for s in seeds:
        log = load_interview_log_arrays(arm, s)
        epoch_max = int(log["epoch"].max())
        cutoff = epoch_max - last_n_epochs + 1
        m_win = log["epoch"] >= cutoff
        for t in (0, 1, 2):
            m = m_win & (log["true_type"] == t)
            offers = log["offer"][m]
            n = int(offers.size)
            for cat in (0, 1, 2):
                out[t][cat].append(
                    float((offers == cat).mean()) if n > 0 else np.nan
                )
    return out, seeds


def fig_F9(variant: str) -> None:
    style.apply_variant(variant)
    dist, seeds = _offer_distribution("default", last_n_epochs=10)

    types = style.TYPE_ORDER
    cats = [0, 1, 2]
    fig, ax = plt.subplots(figsize=style.panel_figsize(variant))

    means = {t: {c: float(np.nanmean(dist[t][c])) for c in cats} for t in types}
    sems = {}
    for t in types:
        sems[t] = {}
        for c in cats:
            arr = np.asarray(dist[t][c], dtype=float)
            n_valid = int(np.sum(~np.isnan(arr)))
            sems[t][c] = (np.nanstd(arr, ddof=1) / np.sqrt(n_valid)) if n_valid > 1 else 0.0

    xpos = np.arange(len(types))
    width = 0.6
    bottoms = np.zeros(len(types))
    for c in cats:
        heights = np.array([means[t][c] for t in types])
        errs = np.array([sems[t][c] for t in types])
        bars = ax.bar(xpos, heights, width=width, bottom=bottoms,
                      color=style.OFFER_COLORS[c], edgecolor="black",
                      linewidth=0.5, label=style.OFFER_LABELS[c])
        # Place an error bar on the *cumulative top* of each segment so
        # variance per category is visible without overlap.
        ax.errorbar(xpos, bottoms + heights, yerr=errs, fmt="none",
                    ecolor="black", capsize=3, linewidth=0.8)
        bottoms = bottoms + heights

    ax.set_xticks(xpos)
    ax.set_xticklabels([style.TYPE_LABELS[t] for t in types])
    ax.set_xlabel("Candidate true type")
    ax.set_ylabel("Fraction of offers")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"True offer distribution by type "
        f"(default, last 10 epochs, N={len(seeds)} seeds)"
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=3, frameon=False)
    fig.tight_layout()
    _save(fig, variant, "F9", "true_offer_distribution_by_type")


def cap_F9():
    write_caption("F9", "true_offer_distribution_by_type",
        """
# F9 - True offer distribution by type (replaces B1)

- **Source artifact**: `outputs/v2/default/seed_*/results/interview_logs.csv`
  (last 10 epochs of each seed; categorical fraction of `offer in {0,1,2}`
  per `true_type`).
- **N seeds**: 10 (`default` arm).
- **Config diff**: none (baseline).
- **What it shows**: Stacked bars of the actual offer distribution per true
  type. Segment heights = mean fraction across seeds; error bars at each
  segment top = SEM across seeds.
- **Claim supported**: Closes audit B1. The pilot plot
  `outputs/figures/offer_distribution_by_type.png` was a linear *interpolation*
  of mean offers and so misrepresented the categorical distribution. This
  figure shows the truth: e.g., for High-type ~90% of offers are correct
  (offer=2), while Low-type splits roughly into reject and over-offer.
""")


# ----- F10: Per-type AI usage over time (default) -----
def fig_F10(variant: str) -> None:
    style.apply_variant(variant)
    curves = load_curves("default")
    epochs = curves["epoch"][0]
    n_seeds = curves["seeds"].shape[0]

    fig, ax = plt.subplots(figsize=style.panel_figsize(variant))
    for t in style.TYPE_ORDER:
        col = f"average_ai_usage_{['low','medium','high'][t]}"
        style.add_ci_line(ax, epochs, curves[col],
                          style.TYPE_COLORS[t], label=style.TYPE_LABELS[t])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Fraction using AI")
    ax.set_title(f"Per-type AI usage over time (default, N={n_seeds}, mean +/- 95% CI)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="best", frameon=False, title="True type")
    fig.tight_layout()
    _save(fig, variant, "F10", "per_type_ai_usage_over_time")


def cap_F10():
    write_caption("F10", "per_type_ai_usage_over_time",
        """
# F10 - Per-type AI usage over time

- **Source artifact**: `outputs/v2/default/aggregate/epoch_curves.npz`
  (columns `average_ai_usage_low/medium/high`).
- **N seeds**: 10 (`default` arm).
- **Config diff**: none (baseline).
- **What it shows**: AI-usage trajectory broken out per true type, with
  95% CI bands across seeds. Same data as F1 but with the "Overall" line
  removed so the type-level dynamics are easier to read.
- **Claim supported**: Underscores methodology B4 - the High-type ceiling.
  Even with detection turned to baseline rates, High-types essentially
  never adopt AI because their unaided signal already gets the offer they
  want; AI use is carried by Medium-types.
""")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

FIG_REGISTRY: Dict[str, Tuple[Callable[[str], None], Callable[[], None], str]] = {
    "F1":  (fig_F1,  cap_F1,  "ai_adoption_over_time"),
    "F2":  (fig_F2,  cap_F2,  "detection_rate_over_time"),
    "F3":  (fig_F3,  cap_F3,  "trust_over_time"),
    "F4":  (fig_F4,  cap_F4,  "losses_over_time"),
    "F5":  (fig_F5,  cap_F5,  "market_efficiency_over_time"),
    "F6":  (fig_F6,  cap_F6,  "ai_usage_all_arms"),
    "F7":  (fig_F7,  cap_F7,  "headline_scalars_across_arms"),
    "F8":  (fig_F8,  cap_F8,  "ai_conditioned_detection_by_type"),
    "F9":  (fig_F9,  cap_F9,  "true_offer_distribution_by_type"),
    "F10": (fig_F10, cap_F10, "per_type_ai_usage_over_time"),
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="both",
                   choices=["paper", "slides", "both"])
    p.add_argument("--figs", default="all",
                   help="comma-separated list of figure ids (default: all)")
    args = p.parse_args(argv)

    if args.figs == "all":
        ids = list(FIG_REGISTRY.keys())
    else:
        ids = [s.strip() for s in args.figs.split(",") if s.strip()]
        unknown = [i for i in ids if i not in FIG_REGISTRY]
        if unknown:
            print(f"unknown figure ids: {unknown}", file=sys.stderr)
            return 2

    variants = ["paper", "slides"] if args.variant == "both" else [args.variant]

    failures = []
    t0 = time.perf_counter()
    for fid in ids:
        fig_fn, cap_fn, _name = FIG_REGISTRY[fid]
        try:
            cap_fn()
        except Exception as e:
            failures.append((fid, "caption", str(e)))
            print(f"[{fid}] caption failed: {e}", file=sys.stderr)
        for v in variants:
            try:
                fig_fn(v)
                print(f"[{fid}] {v} ok")
            except Exception as e:
                failures.append((fid, v, str(e)))
                print(f"[{fid}] {v} FAILED: {e}", file=sys.stderr)
                # close any half-open figure
                plt.close("all")
    dt = time.perf_counter() - t0
    print(f"\nDone in {dt:.1f}s. Figures: {len(ids)}; variants: {variants}; "
          f"failures: {len(failures)}")
    if failures:
        for f in failures:
            print(f"  FAIL {f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
