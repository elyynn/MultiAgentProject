"""
Fictitious-play learning for the Bayesian hiring signaling game.

Both candidate types and firms best-respond to empirical opponent frequencies,
updating counts after each iteration.
"""

import numpy as np
from tqdm import tqdm
from payoffs import (
    candidate_expected_utility,
    firm_expected_utility,
    _make_crn_rng,
)


def normalize_rows(counts):
    """Row-normalize a (num_types x num_actions) count array."""
    row_sums = counts.sum(axis=1, keepdims=True)
    return counts / row_sums


def _sigma_c_dict(counts_candidate, cfg):
    """Convert count array to dict sigma_c[theta, action] -> float."""
    sigma = normalize_rows(counts_candidate)
    return {
        (theta, action): sigma[i, j]
        for i, theta in enumerate(cfg.types)
        for j, action in enumerate(cfg.candidate_actions)
    }


def _conservative_argmax(values, prefer_lower=True, tol=1e-8):
    """
    Return index of max value; break ties in favour of lower index.
    prefer_lower=True: in ties pick the lower-index action (no-AI=0).
    """
    values = np.array(values)
    best = values.max()
    candidates = np.where(np.abs(values - best) < tol)[0]
    return int(candidates[0]) if prefer_lower else int(candidates[-1])


def _cost_conservative_argmax(utilities, costs, tol=1e-8):
    """
    Pick firm policy with highest utility; break ties by lowest cost.
    """
    utilities = np.array(utilities)
    costs = np.array(costs)
    best = utilities.max()
    tied = np.where(np.abs(utilities - best) < tol)[0]
    return int(tied[np.argmin(costs[tied])])


def compute_best_response_regret(counts_candidate, counts_firm, cfg, rng):
    """
    Compute max best-response regret across candidate types and firm.

    r_C_theta = max_a U_C(theta,a,m_bar) - sum_a sigma_C[theta,a] * U_C(theta,a,m_bar)
    r_F       = max_m U_F(m; sigma_C)    - sum_m sigma_F[m] * U_F(m; sigma_C)
    """
    sigma_c = _sigma_c_dict(counts_candidate, cfg)
    sigma_f = counts_firm / counts_firm.sum()

    # Candidate regrets
    candidate_regrets = []
    for i, theta in enumerate(cfg.types):
        action_utils = []
        for action in cfg.candidate_actions:
            u = sum(
                sigma_f[j] * candidate_expected_utility(theta, action, m, sigma_c, cfg, rng)
                for j, m in enumerate(cfg.firm_actions)
            )
            action_utils.append(u)
        action_utils = np.array(action_utils)
        sigma_row = normalize_rows(counts_candidate)[i]
        max_u = action_utils.max()
        expected_u = (sigma_row * action_utils).sum()
        candidate_regrets.append(max_u - expected_u)

    # Firm regret
    firm_utils = np.array([
        firm_expected_utility(m, sigma_c, cfg, rng)
        for m in cfg.firm_actions
    ])
    max_firm_u = firm_utils.max()
    expected_firm_u = (sigma_f * firm_utils).sum()
    firm_regret = max_firm_u - expected_firm_u

    r_max = max(max(candidate_regrets), firm_regret)
    return {
        "r_max": r_max,
        "r_firm": firm_regret,
        "r_candidate_low": candidate_regrets[0],
        "r_candidate_medium": candidate_regrets[1],
        "r_candidate_high": candidate_regrets[2],
    }


def run_fictitious_play(cfg):
    """
    Run fictitious play for cfg.num_fp_iterations iterations.

    Returns:
        counts_candidate: np.ndarray shape (3, 2)
        counts_firm: np.ndarray shape (len(firm_actions),)
        logs: list of dicts, one per iteration
    """
    rng = np.random.default_rng(cfg.seed)

    # Laplace-smoothed initial counts
    counts_candidate = np.ones((len(cfg.types), len(cfg.candidate_actions)))
    counts_firm = np.ones(len(cfg.firm_actions))

    logs = []

    for t in tqdm(range(cfg.num_fp_iterations), desc="FP", unit="iter", dynamic_ncols=True):
        sigma_c = _sigma_c_dict(counts_candidate, cfg)
        sigma_f = counts_firm / counts_firm.sum()

        # Use common random numbers for payoff stability
        crn_rng = _make_crn_rng(seed=t) if cfg.common_random_numbers else rng

        # Candidate best responses (one per type)
        candidate_br = {}
        for i, theta in enumerate(cfg.types):
            action_utils = [
                sum(
                    sigma_f[j] * candidate_expected_utility(
                        theta, action, m, sigma_c, cfg, crn_rng
                    )
                    for j, m in enumerate(cfg.firm_actions)
                )
                for action in cfg.candidate_actions
            ]
            candidate_br[theta] = cfg.candidate_actions[
                _conservative_argmax(action_utils, prefer_lower=True, tol=cfg.tie_tol)
            ]

        # Firm best response
        firm_utils = [
            firm_expected_utility(m, sigma_c, cfg, crn_rng)
            for m in cfg.firm_actions
        ]
        costs = [cfg.verification_cost[m] for m in cfg.firm_actions]
        firm_br_idx = _cost_conservative_argmax(firm_utils, costs, tol=cfg.tie_tol)
        firm_br = cfg.firm_actions[firm_br_idx]

        # Update counts
        for i, theta in enumerate(cfg.types):
            a_star = candidate_br[theta]
            j = cfg.candidate_actions.index(a_star)
            counts_candidate[i, j] += 1

        counts_firm[firm_br_idx] += 1

        # Log current empirical strategies
        sigma_c_now = normalize_rows(counts_candidate)
        sigma_f_now = counts_firm / counts_firm.sum()
        logs.append({
            "iteration": t,
            "sigma_c_low_ai": sigma_c_now[0, 1],
            "sigma_c_medium_ai": sigma_c_now[1, 1],
            "sigma_c_high_ai": sigma_c_now[2, 1],
            "sigma_f_low_verify": sigma_f_now[0],
            "sigma_f_base_verify": sigma_f_now[1],
            "sigma_f_high_verify": sigma_f_now[2],
            "candidate_br_low": candidate_br[0],
            "candidate_br_medium": candidate_br[1],
            "candidate_br_high": candidate_br[2],
            "firm_br": firm_br,
        })

    return counts_candidate, counts_firm, logs


def check_convergence(logs, cfg):
    """
    Check empirical strategy stability over the last fp_stability_window iterations.

    Returns dict with is_converged bool and max_delta.
    """
    w = cfg.fp_stability_window
    if len(logs) < w + 1:
        return {"is_converged": False, "max_delta": float("inf")}

    keys = [
        "sigma_c_low_ai", "sigma_c_medium_ai", "sigma_c_high_ai",
        "sigma_f_low_verify", "sigma_f_base_verify", "sigma_f_high_verify",
    ]
    recent = logs[-w:]
    deltas = []
    for i in range(1, len(recent)):
        for k in keys:
            deltas.append(abs(recent[i][k] - recent[i - 1][k]))

    max_delta = max(deltas)
    return {
        "is_converged": max_delta < cfg.fp_stability_tol,
        "max_delta": max_delta,
    }
