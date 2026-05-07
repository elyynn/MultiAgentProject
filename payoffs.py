"""
Payoff computation for fictitious-play hiring game.

All hot-path functions are vectorised over Monte Carlo samples using numpy —
no per-sample Python loops. The key function is _bayes_optimal_offer_batch,
which computes all n offers in a single set of matrix operations.
"""

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Scalar helpers (kept for diagnostics / evaluate_fp single-sample calls)
# ---------------------------------------------------------------------------

def signal_density(s, theta, action, cfg):
    mu = theta + (cfg.ai_signal_boost[theta] if action == 1 else 0.0)
    return norm.pdf(s, loc=mu, scale=cfg.signal_sigma)


def detection_probability(theta, action, firm_policy, cfg):
    if action == 0:
        return cfg.false_positive_rate[firm_policy]
    return min(1.0, cfg.detection_multiplier[firm_policy] * cfg.base_detection_prob[theta])


def posterior_type_action(s, d, firm_policy, sigma_c, cfg):
    """Scalar posterior — used only in evaluate_fp diagnostics."""
    sigma_c_arr = _to_arr(sigma_c, cfg)
    joint = {}
    for i, theta in enumerate(cfg.types):
        prior = cfg.type_prior[theta]
        for j, a in enumerate(cfg.candidate_actions):
            mu = theta + (cfg.ai_signal_boost[theta] if a == 1 else 0.0)
            f_s = norm.pdf(s, loc=mu, scale=cfg.signal_sigma)
            p_d = detection_probability(theta, a, firm_policy, cfg)
            p_d_obs = p_d if d == 1 else (1.0 - p_d)
            joint[(theta, a)] = prior * sigma_c_arr[i, j] * f_s * p_d_obs
    total = sum(joint.values())
    if total < 1e-300:
        n = len(joint)
        return {k: 1.0 / n for k in joint}
    return {k: v / total for k, v in joint.items()}


def bayes_optimal_offer(s, d, firm_policy, sigma_c, cfg):
    """Scalar offer — wraps the batch version for single samples."""
    sigma_c_arr = _to_arr(sigma_c, cfg)
    offers = _bayes_optimal_offer_batch(
        np.array([s]), np.array([d], dtype=int),
        firm_policy, sigma_c_arr, cfg
    )
    return int(offers[0])


# ---------------------------------------------------------------------------
# Vectorised core
# ---------------------------------------------------------------------------

def _to_arr(sigma_c, cfg):
    """Convert sigma_c dict or array to ndarray shape (3, 2)."""
    if isinstance(sigma_c, np.ndarray):
        return sigma_c
    return np.array([
        [sigma_c[(theta, a)] for a in cfg.candidate_actions]
        for theta in cfg.types
    ])


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


def _bayes_optimal_offer_batch(s_batch, d_batch, firm_policy, sigma_c_arr, cfg,
                                mu_arr=None, p_det_arr=None):
    """
    Vectorised Bayes-optimal offer for a batch of (s, d) pairs.

    All n offers are computed in parallel via numpy broadcasting — no Python loop
    over samples.

    s_batch:      (n,) signal observations
    d_batch:      (n,) detection outcomes in {0, 1}
    sigma_c_arr:  (3, 2) empirical candidate strategy, row-normalised
    Returns:      (n,) int offers in {0, 1, 2}
    """
    if mu_arr is None:
        mu_arr = _precompute_mu(cfg)
    if p_det_arr is None:
        p_det_arr = _precompute_p_det(firm_policy, cfg)

    prior = np.array([cfg.type_prior[t] for t in cfg.types])  # (3,)
    thetas_arr = np.array(cfg.types, dtype=float)              # (3,)

    # f_s: (n, 3, 2) — normal pdf under each (theta, action)
    f_s = norm.pdf(
        s_batch[:, None, None],       # (n, 1, 1)
        loc=mu_arr[None, :, :],       # (1, 3, 2)
        scale=cfg.signal_sigma,
    )

    # p_d_obs: (n, 3, 2) — likelihood of observed d under each (theta, action)
    p_d_obs = np.where(
        d_batch[:, None, None] == 1,
        p_det_arr[None, :, :],
        1.0 - p_det_arr[None, :, :],
    )

    # joint: (n, 3, 2) — unnormalised posterior weight
    joint = prior[None, :, None] * sigma_c_arr[None, :, :] * f_s * p_d_obs

    # normalise to get posterior P(theta, a | s, d)
    total = joint.sum(axis=(1, 2), keepdims=True)   # (n, 1, 1)
    total = np.where(total < 1e-300, 1.0, total)
    posterior = joint / total                        # (n, 3, 2)

    # marginal P(theta | s, d): (n, 3)
    p_theta = posterior.sum(axis=2)

    # E[(o - theta)^2] for each offer o in {0, 1, 2}: result shape (n, 3)
    offers_range = np.array([0.0, 1.0, 2.0])
    # (n, 3 types, 1 offer) * squared diff -> sum over types -> (n, 3 offers)
    loss = (
        p_theta[:, :, None]
        * (offers_range[None, None, :] - thetas_arr[None, :, None]) ** 2
    ).sum(axis=1)

    return loss.argmin(axis=1).astype(int)  # (n,)


# ---------------------------------------------------------------------------
# Expected utilities — vectorised, no per-sample loops
# ---------------------------------------------------------------------------

def candidate_expected_utility(theta, action, firm_policy, sigma_c, cfg, rng=None):
    """
    E[U_C(theta, action, firm_policy)] via vectorised Monte Carlo.

    U_C = E[V(o*(s,d,m))] + E_theta*a - P(det|theta,a,m)*(lambda_D + lambda_R*R_m)
    """
    if rng is None:
        rng = np.random.default_rng(0)

    sigma_c_arr = _to_arr(sigma_c, cfg)
    n = cfg.num_payoff_samples

    mu = theta + (cfg.ai_signal_boost[theta] if action == 1 else 0.0)
    p_det = detection_probability(theta, action, firm_policy, cfg)
    R_m = cfg.reputation_damage()

    s_samples = rng.normal(loc=mu, scale=cfg.signal_sigma, size=n)
    d_samples = (rng.uniform(size=n) < p_det).astype(int)

    mu_arr = _precompute_mu(cfg)
    p_det_arr = _precompute_p_det(firm_policy, cfg)

    offers = _bayes_optimal_offer_batch(
        s_samples, d_samples, firm_policy, sigma_c_arr, cfg, mu_arr, p_det_arr
    )
    offer_vals = np.array([cfg.offer_value[o] for o in offers])

    effort = cfg.ai_effort_benefit[theta] * action
    det_cost = p_det * action * (cfg.detection_penalty + cfg.reputation_penalty_weight * R_m)

    return float(offer_vals.mean() + effort - det_cost)


def firm_expected_utility(firm_policy, sigma_c, cfg, rng=None):
    """
    E[U_F(m; sigma_c)] via vectorised Monte Carlo.

    U_F = -E[(o*(s,d,m) - theta)^2] - c_m
    """
    if rng is None:
        rng = np.random.default_rng(0)

    sigma_c_arr = _to_arr(sigma_c, cfg)
    n = cfg.num_payoff_samples

    # Draw types ~ prior
    type_probs = np.array([cfg.type_prior[t] for t in cfg.types])
    theta_indices = rng.choice(len(cfg.types), size=n, p=type_probs)
    thetas = np.array(cfg.types)[theta_indices]  # (n,)

    # Draw actions from sigma_c[theta] — vectorised via cumsum trick
    cum = sigma_c_arr[theta_indices].cumsum(axis=1)  # (n, 2)
    u_a = rng.uniform(size=n)
    actions = (u_a[:, None] >= cum).sum(axis=1).clip(0, 1)  # (n,) in {0,1}

    mu_arr = _precompute_mu(cfg)
    p_det_arr = _precompute_p_det(firm_policy, cfg)

    # Draw signals and detection in one shot
    mu_samples = mu_arr[theta_indices, actions]              # (n,)
    s_samples = rng.normal(loc=mu_samples, scale=cfg.signal_sigma)
    p_det_samples = p_det_arr[theta_indices, actions]        # (n,)
    d_samples = (rng.uniform(size=n) < p_det_samples).astype(int)

    offers = _bayes_optimal_offer_batch(
        s_samples, d_samples, firm_policy, sigma_c_arr, cfg, mu_arr, p_det_arr
    )
    mismatch = (offers - thetas) ** 2

    return float(-mismatch.mean() - cfg.verification_cost[firm_policy])


def _make_crn_rng(seed):
    return np.random.default_rng(seed)
