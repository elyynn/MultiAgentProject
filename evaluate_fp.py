"""
Held-out Monte Carlo evaluation of frozen fictitious-play empirical strategies.

Procedure:
  1. Freeze final empirical candidate and firm strategies.
  2. Simulate num_eval_samples candidate-firm interactions per seed.
  3. Repeat for num_eval_seeds seeds.
  4. Report mean, SE, and 95% CI for all metrics.
"""

import numpy as np
from payoffs import (
    signal_density,
    detection_probability,
    bayes_optimal_offer,
)


def _simulate_interactions(sigma_c_arr, sigma_f_arr, cfg, rng):
    """
    Simulate one batch of cfg.num_eval_samples interactions.

    sigma_c_arr: shape (3, 2), row-normalised
    sigma_f_arr: shape (3,), normalised

    Returns dict of per-interaction arrays.
    """
    n = cfg.num_eval_samples
    firm_actions = cfg.firm_actions
    types = cfg.types

    # Draw types
    type_probs = np.array([cfg.type_prior[theta] for theta in types])
    theta_indices = rng.choice(len(types), size=n, p=type_probs)
    thetas = np.array(types)[theta_indices]

    # Draw candidate actions from empirical strategy
    actions = np.array([
        rng.choice(cfg.candidate_actions, p=sigma_c_arr[ti])
        for ti in theta_indices
    ])

    # Draw firm policies from empirical firm strategy
    firm_indices = rng.choice(len(firm_actions), size=n, p=sigma_f_arr)
    firms = [firm_actions[fi] for fi in firm_indices]

    # Simulate signals, detection, offers
    signals = np.array([
        rng.normal(
            loc=theta + (cfg.ai_signal_boost[theta] if a == 1 else 0.0),
            scale=cfg.signal_sigma,
        )
        for theta, a in zip(thetas, actions)
    ])

    det_probs = np.array([
        detection_probability(theta, a, m, cfg)
        for theta, a, m in zip(thetas, actions, firms)
    ])
    detected = (rng.uniform(size=n) < det_probs).astype(int)

    # Build sigma_c dict for bayes_optimal_offer
    sigma_c_dict = {
        (theta, action): sigma_c_arr[i, j]
        for i, theta in enumerate(types)
        for j, action in enumerate(cfg.candidate_actions)
    }

    offers = np.array([
        bayes_optimal_offer(s, d, m, sigma_c_dict, cfg)
        for s, d, m in zip(signals, detected, firms)
    ])

    return {
        "thetas": thetas,
        "actions": actions,
        "firms": firms,
        "firm_indices": firm_indices,
        "signals": signals,
        "detected": detected,
        "det_probs": det_probs,
        "offers": offers,
    }


def _compute_metrics(sim, cfg):
    """Compute scalar metrics from a simulated batch."""
    thetas = sim["thetas"]
    actions = sim["actions"]
    offers = sim["offers"]
    detected = sim["detected"]
    det_probs = sim["det_probs"]
    firm_indices = sim["firm_indices"]

    ai_mask = actions == 1
    R_m = cfg.reputation_damage()

    # Candidate welfare
    offer_vals = np.array([cfg.offer_value[o] for o in offers])
    effort_benefits = np.array([cfg.ai_effort_benefit[theta] * a for theta, a in zip(thetas, actions)])
    det_costs = det_probs * ai_mask * (cfg.detection_penalty + cfg.reputation_penalty_weight * R_m)
    candidate_welfare = (offer_vals + effort_benefits - det_costs).mean()

    # Firm welfare per interaction
    mismatch = (offers - thetas) ** 2
    costs = np.array([cfg.verification_cost[cfg.firm_actions[fi]] for fi in firm_indices])
    firm_welfare = (-mismatch - costs).mean()

    # Match metrics
    correct = (offers == thetas)
    overoffer = (offers > thetas)
    underoffer = (offers < thetas)

    # Detection conditional on AI use
    det_given_ai = detected[ai_mask].mean() if ai_mask.any() else float("nan")
    det_given_ai_by_type = {}
    for theta in cfg.types:
        mask = (thetas == theta) & ai_mask
        det_given_ai_by_type[theta] = detected[mask].mean() if mask.any() else float("nan")

    # AI adoption by type
    ai_by_type = {}
    for theta in cfg.types:
        mask = thetas == theta
        ai_by_type[theta] = actions[mask].mean() if mask.any() else float("nan")

    # Firm policy distribution
    firm_dist = np.bincount(sim["firm_indices"], minlength=len(cfg.firm_actions)) / len(thetas)

    metrics = {
        "ai_adoption_overall": ai_mask.mean(),
        "ai_adoption_low": ai_by_type[0],
        "ai_adoption_medium": ai_by_type[1],
        "ai_adoption_high": ai_by_type[2],
        "correct_match_rate": correct.mean(),
        "overoffer_rate": overoffer.mean(),
        "underoffer_rate": underoffer.mean(),
        "detection_rate_given_ai": det_given_ai,
        "detection_rate_given_ai_low": det_given_ai_by_type[0],
        "detection_rate_given_ai_medium": det_given_ai_by_type[1],
        "detection_rate_given_ai_high": det_given_ai_by_type[2],
        "candidate_welfare": candidate_welfare,
        "firm_welfare": firm_welfare,
        "total_welfare": candidate_welfare + firm_welfare,
    }
    for j, m in enumerate(cfg.firm_actions):
        metrics[f"firm_policy_{m.lower()}"] = firm_dist[j]

    # Separating index: std of AI adoption across types
    ai_rates = np.array([ai_by_type[theta] for theta in cfg.types])
    metrics["separating_index"] = float(np.nanstd(ai_rates))

    return metrics


def evaluate_frozen_strategies(counts_candidate, counts_firm, cfg):
    """
    Evaluate frozen empirical strategies over cfg.num_eval_seeds seeds.

    Returns:
        summary: dict with per-metric mean, se, ci_lo, ci_hi
        per_seed: list of per-seed metric dicts
    """
    from fictitious_play import normalize_rows

    sigma_c_arr = normalize_rows(counts_candidate)
    sigma_f_arr = counts_firm / counts_firm.sum()

    per_seed = []
    base_rng = np.random.default_rng(cfg.seed + 9999)
    seed_sequence = base_rng.integers(0, 2**31, size=cfg.num_eval_seeds)

    for s_idx, seed in enumerate(seed_sequence):
        rng = np.random.default_rng(int(seed))
        sim = _simulate_interactions(sigma_c_arr, sigma_f_arr, cfg, rng)
        metrics = _compute_metrics(sim, cfg)
        metrics["eval_seed_idx"] = s_idx
        per_seed.append(metrics)

    # Aggregate across seeds
    all_keys = [k for k in per_seed[0] if k != "eval_seed_idx"]
    summary = {}
    for k in all_keys:
        vals = np.array([s[k] for s in per_seed if not np.isnan(s[k])])
        if len(vals) == 0:
            summary[k] = {"mean": float("nan"), "se": float("nan"),
                          "ci_lo": float("nan"), "ci_hi": float("nan")}
            continue
        mean = vals.mean()
        se = vals.std(ddof=1) / np.sqrt(len(vals))
        summary[k] = {
            "mean": float(mean),
            "se": float(se),
            "ci_lo": float(mean - 1.96 * se),
            "ci_hi": float(mean + 1.96 * se),
        }

    return summary, per_seed
