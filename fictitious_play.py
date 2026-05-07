"""
Best-response dynamics for the Bayesian hiring game with FP-on-(verification, suspicion).

Per the methodology audit (B3-FP), this is *not* canonical fictitious play in the
Brown/Robinson sense — but with the redesigned firm action space `(verification,
ai_suspicion)`, the stage game is now stationary across iterations, which is the
key requirement for FP convergence theorems on finite games.

Both populations share an iteration-level CRN seed; within an iteration, paired
shocks are reused across the actions being compared so the argmax decision sees
matched MC noise (audit I3-FP).
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from typing import Optional, Tuple

import numpy as np
from tqdm import tqdm

from payoffs import (
    candidate_expected_utility,
    firm_expected_utility,
    _draw_candidate_noise,
    _draw_firm_noise,
    _make_crn_rng,
)


def _tqdm_disabled() -> bool:
    """Disable tqdm bars in non-TTY runs (M8-FP)."""
    return os.environ.get("FP_DISABLE_TQDM") == "1" or not sys.stdout.isatty()


def normalize_rows(counts: np.ndarray) -> np.ndarray:
    row_sums = counts.sum(axis=1, keepdims=True)
    return counts / row_sums


def _sigma_c_dict(counts_candidate: np.ndarray, cfg):
    sigma = normalize_rows(counts_candidate)
    return {
        (theta, action): float(sigma[i, j])
        for i, theta in enumerate(cfg.types)
        for j, action in enumerate(cfg.candidate_actions)
    }


def _conservative_argmax(values, prefer_lower: bool, tol: float) -> int:
    """Return argmax with deterministic tie-break."""
    values = np.asarray(values)
    best = values.max()
    candidates = np.where(np.abs(values - best) < tol)[0]
    return int(candidates[0]) if prefer_lower else int(candidates[-1])


def _cost_conservative_argmax(values, costs, tol: float) -> int:
    values = np.asarray(values)
    costs = np.asarray(costs)
    best = values.max()
    tied = np.where(np.abs(values - best) < tol)[0]
    return int(tied[np.argmin(costs[tied])])


# ---------------------------------------------------------------------------
# Joint firm action helpers
# ---------------------------------------------------------------------------

def _joint_firm_action_list(cfg):
    """List of (verification, suspicion_idx) tuples in canonical order."""
    return list(cfg.firm_joint_actions())


def _firm_action_costs(cfg):
    """Verification cost for each joint firm action, used as tie-break key."""
    actions = _joint_firm_action_list(cfg)
    return np.array([cfg.verification_cost[m] for (m, _k) in actions])


# ---------------------------------------------------------------------------
# Regret diagnostic
# ---------------------------------------------------------------------------

def compute_best_response_regret(counts_candidate, counts_firm, cfg, rng):
    """
    Empirical best-response regret at the final empirical strategies.

    counts_firm: shape (num_firm_joint_actions,), ordered as cfg.firm_joint_actions().
    """
    sigma_c = _sigma_c_dict(counts_candidate, cfg)
    sigma_f = counts_firm / counts_firm.sum()
    actions = _joint_firm_action_list(cfg)

    # Candidate regrets (one per type)
    candidate_regrets = []
    for i, theta in enumerate(cfg.types):
        action_utils = []
        for cand_action in cfg.candidate_actions:
            u = 0.0
            for j, (verification, suspicion_idx) in enumerate(actions):
                u += sigma_f[j] * candidate_expected_utility(
                    theta, cand_action, verification, suspicion_idx, cfg, rng
                )
            action_utils.append(u)
        action_utils = np.array(action_utils)
        sigma_row = normalize_rows(counts_candidate)[i]
        candidate_regrets.append(float(action_utils.max() - (sigma_row * action_utils).sum()))

    # Firm regret (over the joint action space)
    firm_utils = np.array([
        firm_expected_utility(verification, suspicion_idx, sigma_c, cfg, rng)
        for (verification, suspicion_idx) in actions
    ])
    firm_regret = float(firm_utils.max() - (sigma_f * firm_utils).sum())

    r_max = max(max(candidate_regrets), firm_regret)
    return {
        "r_max": float(r_max),
        "r_firm": firm_regret,
        "r_candidate_low": candidate_regrets[0],
        "r_candidate_medium": candidate_regrets[1],
        "r_candidate_high": candidate_regrets[2],
    }


# ---------------------------------------------------------------------------
# Main FP loop
# ---------------------------------------------------------------------------

def run_fictitious_play(cfg, fixed_firm: Optional[Tuple[str, int]] = None,
                         seed: Optional[int] = None, desc: str = "FP"):
    """
    Run best-response dynamics for cfg.num_fp_iterations iterations.

    Parameters
    ----------
    cfg : FPConfig
    fixed_firm : optional (verification, suspicion_idx). If supplied, the firm's joint
        action is held fixed for all iterations and only the candidate population
        runs FP. (Folds the old `_run_fixed_firm` duplicate — audit M2-FP.)
    seed : optional override for cfg.seed (used by multi-seed loops).
    desc : label for the tqdm bar.

    Returns
    -------
    counts_candidate : (num_types, num_candidate_actions)
    counts_firm      : (num_firm_joint_actions,)  ordered as cfg.firm_joint_actions()
    logs             : list of per-iteration dicts
    """
    if seed is None:
        seed = cfg.seed
    rng = np.random.default_rng(seed)

    n_t = len(cfg.types)
    n_a = len(cfg.candidate_actions)
    actions = _joint_firm_action_list(cfg)
    n_f = len(actions)
    firm_costs = _firm_action_costs(cfg)
    tie_tol = cfg.effective_tie_tol()

    counts_candidate = np.ones((n_t, n_a))

    # Smoothed initial counts for firm. If fixed_firm, lock all weight there.
    if fixed_firm is not None:
        verification_fixed, suspicion_fixed = fixed_firm
        fixed_idx = actions.index((verification_fixed, suspicion_fixed))
        counts_firm = np.zeros(n_f)
        counts_firm[fixed_idx] = cfg.num_fp_iterations + n_f
    else:
        counts_firm = np.ones(n_f)

    # BR-flip tracking
    last_candidate_br = [None] * n_t
    last_firm_br = None
    candidate_flips_total = [0] * n_t
    firm_flips_total = 0

    logs = []

    for t in tqdm(range(cfg.num_fp_iterations), desc=desc, unit="iter",
                  dynamic_ncols=True, disable=_tqdm_disabled()):
        sigma_c_arr = normalize_rows(counts_candidate)
        sigma_c = _sigma_c_dict(counts_candidate, cfg)
        sigma_f = counts_firm / counts_firm.sum()

        crn_rng = _make_crn_rng(seed=seed * 1_000_003 + t) if cfg.common_random_numbers else rng

        # ---- Candidate best response, with paired CRN across actions per (theta, m, k) ----
        candidate_br = np.empty(n_t, dtype=int)
        candidate_action_utils = np.zeros((n_t, n_a))
        for i, theta in enumerate(cfg.types):
            for j, (verification, suspicion_idx) in enumerate(actions):
                shared = _draw_candidate_noise(crn_rng, cfg.num_payoff_samples)
                for k, cand_action in enumerate(cfg.candidate_actions):
                    u = candidate_expected_utility(
                        theta, cand_action, verification, suspicion_idx, cfg,
                        rng=crn_rng, shared_noise=shared,
                    )
                    candidate_action_utils[i, k] += sigma_f[j] * u
            candidate_br[i] = _conservative_argmax(
                candidate_action_utils[i], prefer_lower=cfg.tie_break_prefer_lower, tol=tie_tol,
            )

        # ---- Firm best response (only if not fixed), shared CRN across (m, k) ----
        if fixed_firm is None:
            shared_firm = _draw_firm_noise(crn_rng, cfg.num_payoff_samples, sigma_c_arr, cfg)
            firm_utils = np.array([
                firm_expected_utility(
                    verification, suspicion_idx, sigma_c_arr, cfg,
                    rng=crn_rng, shared_noise=shared_firm,
                )
                for (verification, suspicion_idx) in actions
            ])
            firm_br_idx = _cost_conservative_argmax(firm_utils, firm_costs, tol=tie_tol)
        else:
            firm_br_idx = fixed_idx

        # Track flips
        for i in range(n_t):
            if last_candidate_br[i] is not None and candidate_br[i] != last_candidate_br[i]:
                candidate_flips_total[i] += 1
            last_candidate_br[i] = int(candidate_br[i])
        if last_firm_br is not None and firm_br_idx != last_firm_br:
            firm_flips_total += 1
        last_firm_br = firm_br_idx

        # Update empirical counts
        for i in range(n_t):
            counts_candidate[i, candidate_br[i]] += 1
        if fixed_firm is None:
            counts_firm[firm_br_idx] += 1

        # Log
        sigma_c_now = normalize_rows(counts_candidate)
        sigma_f_now = counts_firm / counts_firm.sum()
        # Marginalise sigma_f over verification and suspicion separately for plotting
        n_m = len(cfg.firm_actions)
        n_s = len(cfg.firm_ai_suspicion_levels)
        sigma_f_grid = sigma_f_now.reshape(n_m, n_s)
        marg_verification = sigma_f_grid.sum(axis=1)  # (3,)
        marg_suspicion = sigma_f_grid.sum(axis=0)     # (3,)

        verification_br, suspicion_br = actions[firm_br_idx]
        log_entry = {
            "iteration": t,
            "sigma_c_low_ai": float(sigma_c_now[0, 1]),
            "sigma_c_medium_ai": float(sigma_c_now[1, 1]),
            "sigma_c_high_ai": float(sigma_c_now[2, 1]),
            "sigma_f_low_verify": float(marg_verification[0]),
            "sigma_f_base_verify": float(marg_verification[1]),
            "sigma_f_high_verify": float(marg_verification[2]),
            "sigma_f_sus_low": float(marg_suspicion[0]),
            "sigma_f_sus_mid": float(marg_suspicion[1]),
            "sigma_f_sus_high": float(marg_suspicion[2]),
            "candidate_br_low": int(candidate_br[0]),
            "candidate_br_medium": int(candidate_br[1]),
            "candidate_br_high": int(candidate_br[2]),
            "firm_br_verification": verification_br,
            "firm_br_suspicion_idx": int(suspicion_br),
        }
        logs.append(log_entry)

    flips = {
        "candidate_low": candidate_flips_total[0],
        "candidate_medium": candidate_flips_total[1],
        "candidate_high": candidate_flips_total[2],
        "firm": firm_flips_total,
    }

    return counts_candidate, counts_firm, logs, flips


def check_convergence(logs, flips, cfg):
    """
    Stricter convergence check (audit I5/I6-FP):
      stable_mixture := max single-step change in marginal sigmas over last W iters
                        is below cfg.fp_stability_max_delta
      no_late_flips  := zero BR flips in the last W iterations
      converged      := stable_mixture AND no_late_flips

    Note: r_max threshold is checked separately, in run_arm, after held-out evaluation
    has the regret diagnostic.
    """
    w = cfg.fp_stability_window
    if len(logs) < w + 1:
        return {
            "is_strategy_stable": False,
            "max_delta": float("inf"),
            "flips_in_window": int(_flips_in_window(logs, w)),
            "br_flips_total": flips,
        }

    keys = [
        "sigma_c_low_ai", "sigma_c_medium_ai", "sigma_c_high_ai",
        "sigma_f_low_verify", "sigma_f_base_verify", "sigma_f_high_verify",
        "sigma_f_sus_low", "sigma_f_sus_mid", "sigma_f_sus_high",
    ]
    recent = logs[-w:]
    deltas = []
    for i in range(1, len(recent)):
        for k in keys:
            deltas.append(abs(recent[i][k] - recent[i - 1][k]))
    max_delta = max(deltas) if deltas else 0.0

    flips_window = _flips_in_window(logs, w)

    return {
        "is_strategy_stable": (max_delta < cfg.fp_stability_max_delta) and (flips_window == 0),
        "max_delta": float(max_delta),
        "flips_in_window": int(flips_window),
        "br_flips_total": flips,
    }


def _flips_in_window(logs, w):
    if len(logs) < 2:
        return 0
    recent = logs[-w:] if len(logs) >= w else logs
    flips = 0
    keys_to_track = [
        "candidate_br_low", "candidate_br_medium", "candidate_br_high",
        "firm_br_verification", "firm_br_suspicion_idx",
    ]
    for i in range(1, len(recent)):
        for k in keys_to_track:
            if recent[i][k] != recent[i - 1][k]:
                flips += 1
    return flips
