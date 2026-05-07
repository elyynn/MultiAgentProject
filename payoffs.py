"""
Payoff computation for the FP-style hiring game.

Joint firm action: (verification_level, ai_suspicion_idx).

The firm's offer rule is Bayes-optimal under its *committed* ai_suspicion belief
(P(a=1) = suspicion_p, uniform across types) — NOT under the real sigma_c. This is
what makes the stage game stationary and the dynamics canonical FP-on-finite-actions.

All hot paths are vectorised over Monte Carlo samples. The candidate and firm utility
functions accept optional pre-drawn noise (`shared_noise=...`) so that callers can
reuse the same standard-normal / uniform shocks across compared actions (paired CRN
within an iteration), which materially reduces the variance of the argmax decision.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


# Module-level counter for degenerate posterior events (M10-FP).
DEGENERATE_POSTERIOR_COUNT: int = 0


def _reset_degenerate_counter():
    global DEGENERATE_POSTERIOR_COUNT
    DEGENERATE_POSTERIOR_COUNT = 0


def get_degenerate_posterior_count() -> int:
    return DEGENERATE_POSTERIOR_COUNT


# ---------------------------------------------------------------------------
# Scalar helpers (kept for diagnostics)
# ---------------------------------------------------------------------------

def signal_density(s, theta, action, cfg):
    mu = theta + (cfg.ai_signal_boost[theta] if action == 1 else 0.0)
    return norm.pdf(s, loc=mu, scale=cfg.signal_sigma)


def detection_probability(theta, action, firm_policy, cfg):
    if action == 0:
        return cfg.false_positive_rate[firm_policy]
    return min(1.0, cfg.detection_multiplier[firm_policy] * cfg.base_detection_prob[theta])


# ---------------------------------------------------------------------------
# Vectorised core
# ---------------------------------------------------------------------------

def _precompute_mu(cfg):
    """mu[theta_idx, action_idx] — shape (3, 2)."""
    return np.array([
        [theta + (cfg.ai_signal_boost[theta] if a == 1 else 0.0)
         for a in cfg.candidate_actions]
        for theta in cfg.types
    ])


def _precompute_p_det(firm_policy, cfg):
    """p_det[theta_idx, action_idx] — shape (3, 2)."""
    return np.array([
        [detection_probability(theta, a, firm_policy, cfg) for a in cfg.candidate_actions]
        for theta in cfg.types
    ])


def _belief_under_suspicion(suspicion_p: float, cfg) -> np.ndarray:
    """
    P(a | theta) under the firm's committed suspicion. Uniform across types.
    Returns shape (num_types, num_actions).
    """
    n_t = len(cfg.types)
    n_a = len(cfg.candidate_actions)
    if n_a != 2:
        raise ValueError(f"Expected binary candidate actions, got {n_a}")
    belief = np.empty((n_t, n_a))
    belief[:, 0] = 1.0 - suspicion_p
    belief[:, 1] = suspicion_p
    return belief


def _bayes_optimal_offer_under_belief(s_batch, d_batch, verification, suspicion_p, cfg,
                                       mu_arr=None, p_det_arr=None):
    """
    Vectorised offer rule under the firm's committed belief.

    The firm's posterior over (theta, a) uses prior(theta) * P(a|theta; suspicion_p) —
    NOT the empirical sigma_c. The stage game is therefore stationary across FP iterations.

    Returns: (n,) int offers in {0, 1, 2}
    """
    global DEGENERATE_POSTERIOR_COUNT

    if mu_arr is None:
        mu_arr = _precompute_mu(cfg)
    if p_det_arr is None:
        p_det_arr = _precompute_p_det(verification, cfg)

    prior = np.array([cfg.type_prior[t] for t in cfg.types])  # (3,)
    thetas_arr = np.array(cfg.types, dtype=float)              # (3,)
    belief = _belief_under_suspicion(suspicion_p, cfg)          # (3, 2)

    # f_s: (n, 3, 2)
    f_s = norm.pdf(
        s_batch[:, None, None],
        loc=mu_arr[None, :, :],
        scale=cfg.signal_sigma,
    )

    # p_d_obs: (n, 3, 2)
    p_d_obs = np.where(
        d_batch[:, None, None] == 1,
        p_det_arr[None, :, :],
        1.0 - p_det_arr[None, :, :],
    )

    # joint(theta, a | s, d) ∝ prior(theta) * belief(a|theta) * f(s|theta,a) * p(d|theta,a)
    joint = prior[None, :, None] * belief[None, :, :] * f_s * p_d_obs  # (n, 3, 2)

    total = joint.sum(axis=(1, 2), keepdims=True)
    degen_mask = total < 1e-300
    if np.any(degen_mask):
        DEGENERATE_POSTERIOR_COUNT += int(degen_mask.sum())
    total = np.where(degen_mask, 1.0, total)
    posterior = joint / total                                   # (n, 3, 2)

    # marginal P(theta | s, d): (n, 3)
    p_theta = posterior.sum(axis=2)

    # E[(o - theta)^2] for offers in {0,1,2}: (n, 3 offers)
    offers_range = np.array([0.0, 1.0, 2.0])
    loss = (
        p_theta[:, :, None]
        * (offers_range[None, None, :] - thetas_arr[None, :, None]) ** 2
    ).sum(axis=1)

    return loss.argmin(axis=1).astype(int)


def bayes_optimal_offer(s, d, verification, suspicion_p, cfg):
    """Scalar wrapper, used only by evaluate_fp diagnostics."""
    offers = _bayes_optimal_offer_under_belief(
        np.array([s], dtype=float), np.array([d], dtype=int),
        verification, suspicion_p, cfg,
    )
    return int(offers[0])


# ---------------------------------------------------------------------------
# Expected utilities — accept optional shared noise for paired CRN
# ---------------------------------------------------------------------------

def _draw_candidate_noise(rng, n: int):
    """Pre-draw paired (Z, U_d) for use across compared candidate actions."""
    return rng.standard_normal(n), rng.uniform(size=n)


def candidate_expected_utility(theta, action, verification, suspicion_idx, cfg,
                               rng=None, shared_noise=None):
    """
    E[U_C(theta, action, (verification, suspicion_idx))] via vectorised MC.

    `shared_noise`: optional (Z, U_d) tuple with shape (n,). If provided, signals are
    `mu + sigma * Z` and detection outcomes are `U_d < p_det`. Reusing the same shocks
    across `action` values (paired sampling) reduces argmax variance — see audit I3-FP.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    n = cfg.num_payoff_samples
    suspicion_p = cfg.firm_ai_suspicion_levels[suspicion_idx]

    mu = theta + (cfg.ai_signal_boost[theta] if action == 1 else 0.0)
    p_det = detection_probability(theta, action, verification, cfg)
    R_m = cfg.reputation_damage()

    if shared_noise is None:
        Z, U_d = _draw_candidate_noise(rng, n)
    else:
        Z, U_d = shared_noise

    s_samples = mu + cfg.signal_sigma * Z
    d_samples = (U_d < p_det).astype(int)

    mu_arr = _precompute_mu(cfg)
    p_det_arr = _precompute_p_det(verification, cfg)

    offers = _bayes_optimal_offer_under_belief(
        s_samples, d_samples, verification, suspicion_p, cfg, mu_arr, p_det_arr
    )
    offer_vals = np.array([cfg.offer_value[o] for o in offers])

    effort = cfg.ai_effort_benefit[theta] * action
    det_cost = p_det * action * (cfg.detection_penalty + cfg.reputation_penalty_weight * R_m)

    return float(offer_vals.mean() + effort - det_cost)


def _draw_firm_noise(rng, n: int, sigma_c_arr, cfg):
    """
    Pre-draw (theta_indices, action_indices, Z_signal, U_det) shared across firm actions.
    Reused so all (m, k) firm actions are evaluated on the same population sample.
    """
    type_probs = np.array([cfg.type_prior[t] for t in cfg.types])
    theta_indices = rng.choice(len(cfg.types), size=n, p=type_probs)

    cum = sigma_c_arr[theta_indices].cumsum(axis=1)  # (n, 2)
    u_a = rng.uniform(size=n)
    if len(cfg.candidate_actions) != 2:
        raise ValueError("Firm sampling assumes binary candidate actions")
    action_indices = (u_a[:, None] >= cum).sum(axis=1).clip(0, 1)

    Z_signal = rng.standard_normal(n)
    U_det = rng.uniform(size=n)
    return theta_indices, action_indices, Z_signal, U_det


def firm_expected_utility(verification, suspicion_idx, sigma_c, cfg,
                          rng=None, shared_noise=None):
    """
    E[U_F((verification, suspicion_idx); sigma_c)] = -E[(o-theta)^2] - c_m.

    sigma_c may be a dict (theta, a) -> p, or an ndarray (3, 2).
    `shared_noise`: optional (theta_idx, action_idx, Z_signal, U_det) for paired CRN.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    sigma_c_arr = _to_arr(sigma_c, cfg)
    n = cfg.num_payoff_samples
    suspicion_p = cfg.firm_ai_suspicion_levels[suspicion_idx]

    if shared_noise is None:
        theta_indices, action_indices, Z_signal, U_det = _draw_firm_noise(rng, n, sigma_c_arr, cfg)
    else:
        theta_indices, action_indices, Z_signal, U_det = shared_noise

    thetas = np.array(cfg.types)[theta_indices]

    mu_arr = _precompute_mu(cfg)
    p_det_arr = _precompute_p_det(verification, cfg)

    mu_samples = mu_arr[theta_indices, action_indices]
    s_samples = mu_samples + cfg.signal_sigma * Z_signal
    p_det_samples = p_det_arr[theta_indices, action_indices]
    d_samples = (U_det < p_det_samples).astype(int)

    offers = _bayes_optimal_offer_under_belief(
        s_samples, d_samples, verification, suspicion_p, cfg, mu_arr, p_det_arr
    )
    mismatch = (offers - thetas) ** 2

    return float(-mismatch.mean() - cfg.verification_cost[verification])


def _to_arr(sigma_c, cfg) -> np.ndarray:
    if isinstance(sigma_c, np.ndarray):
        return sigma_c
    return np.array([
        [sigma_c[(theta, a)] for a in cfg.candidate_actions]
        for theta in cfg.types
    ])


def _make_crn_rng(seed):
    return np.random.default_rng(seed)
