"""
Held-out Monte Carlo evaluation of frozen FP empirical strategies.

Per training seed, run one fully-vectorised evaluation pass with
`num_eval_samples` interactions. Aggregation across training seeds (mean,
t-distribution CI with df=n-1) lives in run_fp_suite.aggregate_arm.
"""

from __future__ import annotations

import numpy as np

from payoffs import (
    detection_probability,
    _precompute_mu,
    _precompute_p_det,
    _bayes_optimal_offer_under_belief,
)


def _simulate_interactions_vectorised(sigma_c_arr, sigma_f_arr, cfg, rng):
    """
    Fully vectorised simulation — no per-sample Python loops (audit M3-FP).

    sigma_f_arr is over the joint firm action space (verification × suspicion).
    """
    n = cfg.num_eval_samples
    n_v = len(cfg.firm_actions)
    n_t = len(cfg.types)

    # ---- Types ----
    type_probs = np.array([cfg.type_prior[t] for t in cfg.types])
    theta_indices = rng.choice(n_t, size=n, p=type_probs)
    thetas = np.array(cfg.types)[theta_indices]

    # ---- Candidate actions from empirical sigma_c ----
    cum = sigma_c_arr[theta_indices].cumsum(axis=1)
    u_a = rng.uniform(size=n)
    if len(cfg.candidate_actions) != 2:
        raise ValueError("Vectorised evaluation assumes binary candidate actions")
    actions = (u_a[:, None] >= cum).sum(axis=1).clip(0, 1)

    # ---- Firm joint action ----
    joint_actions = list(cfg.firm_joint_actions())
    n_f = len(joint_actions)
    firm_indices = rng.choice(n_f, size=n, p=sigma_f_arr)
    verif_idx_per_sample = np.array([cfg.firm_actions.index(joint_actions[fi][0])
                                      for fi in firm_indices])

    # ---- Signals (firm-independent) ----
    mu_arr = _precompute_mu(cfg)
    mu_samples = mu_arr[theta_indices, actions]
    s_samples = mu_samples + cfg.signal_sigma * rng.standard_normal(n)

    # ---- Detection — depends on verification ----
    p_det_full = np.array([_precompute_p_det(v, cfg) for v in cfg.firm_actions])  # (n_v, n_t, n_a)
    p_det_samples = p_det_full[verif_idx_per_sample, theta_indices, actions]
    U_det = rng.uniform(size=n)
    detected = (U_det < p_det_samples).astype(int)

    # ---- Offers, grouped by joint firm action ----
    offers = np.zeros(n, dtype=int)
    for fa_idx, (verification, suspicion_idx) in enumerate(joint_actions):
        mask = firm_indices == fa_idx
        if not mask.any():
            continue
        suspicion_p = cfg.firm_ai_suspicion_levels[suspicion_idx]
        offers[mask] = _bayes_optimal_offer_under_belief(
            s_samples[mask], detected[mask], verification, suspicion_p, cfg,
        )

    return {
        "thetas": thetas,
        "actions": actions,
        "firm_indices": firm_indices,
        "verif_idx_per_sample": verif_idx_per_sample,
        "joint_actions": joint_actions,
        "signals": s_samples,
        "detected": detected,
        "det_probs": p_det_samples,
        "offers": offers,
    }


def _compute_metrics(sim, cfg):
    thetas = sim["thetas"]
    actions = sim["actions"]
    offers = sim["offers"]
    detected = sim["detected"]
    det_probs = sim["det_probs"]
    firm_indices = sim["firm_indices"]
    verif_idx = sim["verif_idx_per_sample"]
    joint_actions = sim["joint_actions"]

    ai_mask = actions == 1
    R_m = cfg.reputation_damage()

    # Candidate welfare per interaction
    offer_vals = np.array([cfg.offer_value[o] for o in offers])
    effort_benefits = np.array([cfg.ai_effort_benefit[t] * a for t, a in zip(thetas, actions)])
    det_costs = det_probs * ai_mask * (cfg.detection_penalty + cfg.reputation_penalty_weight * R_m)
    candidate_welfare = float((offer_vals + effort_benefits - det_costs).mean())

    # Firm welfare (cost depends on verification only)
    mismatch = (offers - thetas) ** 2
    verification_costs = np.array([cfg.verification_cost[m] for m in cfg.firm_actions])
    costs_per_sample = verification_costs[verif_idx]
    firm_welfare = float((-mismatch - costs_per_sample).mean())

    correct = (offers == thetas)
    overoffer = (offers > thetas)
    underoffer = (offers < thetas)

    det_given_ai = float(detected[ai_mask].mean()) if ai_mask.any() else float("nan")
    det_given_ai_by_type = {}
    for theta in cfg.types:
        mask = (thetas == theta) & ai_mask
        det_given_ai_by_type[theta] = float(detected[mask].mean()) if mask.any() else float("nan")

    ai_by_type = {}
    for theta in cfg.types:
        mask = thetas == theta
        ai_by_type[theta] = float(actions[mask].mean()) if mask.any() else float("nan")

    # Marginal firm verification distribution and suspicion distribution
    n_s = len(cfg.firm_ai_suspicion_levels)
    n_v = len(cfg.firm_actions)
    firm_dist_joint = np.bincount(firm_indices, minlength=len(joint_actions)) / len(thetas)
    firm_dist_grid = firm_dist_joint.reshape(n_v, n_s)
    marg_verification = firm_dist_grid.sum(axis=1)
    marg_suspicion = firm_dist_grid.sum(axis=0)

    metrics = {
        "ai_adoption_overall": float(ai_mask.mean()),
        "ai_adoption_low": ai_by_type[0],
        "ai_adoption_medium": ai_by_type[1],
        "ai_adoption_high": ai_by_type[2],
        "correct_match_rate": float(correct.mean()),
        "overoffer_rate": float(overoffer.mean()),
        "underoffer_rate": float(underoffer.mean()),
        "detection_rate_given_ai": det_given_ai,
        "detection_rate_given_ai_low": det_given_ai_by_type[0],
        "detection_rate_given_ai_medium": det_given_ai_by_type[1],
        "detection_rate_given_ai_high": det_given_ai_by_type[2],
        "candidate_welfare": candidate_welfare,
        "firm_welfare": firm_welfare,
        "total_welfare": candidate_welfare + firm_welfare,
    }
    for j, m in enumerate(cfg.firm_actions):
        metrics[f"firm_policy_{m.lower()}"] = float(marg_verification[j])
    for k, p in enumerate(cfg.firm_ai_suspicion_levels):
        metrics[f"firm_suspicion_{int(p*100):02d}"] = float(marg_suspicion[k])

    # Separating index — std of AI adoption across types (with footnote on what this measures).
    # See audit M6-FP.
    ai_rates = np.array([ai_by_type[theta] for theta in cfg.types])
    metrics["separating_index"] = float(np.nanstd(ai_rates))

    return metrics


def evaluate_one_seed(counts_candidate, counts_firm, cfg, eval_seed: int):
    """
    Evaluate a single frozen (counts_candidate, counts_firm) pair with one eval RNG.
    Used inside the multi-training-seed loop.
    """
    from fictitious_play import normalize_rows

    sigma_c_arr = normalize_rows(counts_candidate)
    sigma_f_arr = counts_firm / counts_firm.sum()

    rng = np.random.default_rng(int(eval_seed))
    sim = _simulate_interactions_vectorised(sigma_c_arr, sigma_f_arr, cfg, rng)
    metrics = _compute_metrics(sim, cfg)
    metrics["eval_seed"] = int(eval_seed)
    return metrics


def aggregate_per_seed_metrics(per_seed_metrics, cfg):
    """
    Aggregate a list of per-(training-seed) metric dicts using t-distribution CIs.

    With cfg.num_training_seeds < 30, normal-approx CIs underestimate uncertainty;
    we use the Student-t critical value with df = n-1 (audit Q15).
    """
    from scipy.stats import t as student_t

    if not per_seed_metrics:
        return {}

    # Numeric metric keys (skip seed labels)
    skip = {"eval_seed", "training_seed", "training_seed_idx"}
    keys = [k for k in per_seed_metrics[0] if k not in skip]

    n = len(per_seed_metrics)
    if n >= 2:
        t_crit = float(student_t.ppf(0.975, df=n - 1))
    else:
        t_crit = float("nan")

    summary = {}
    for k in keys:
        vals = np.array([m[k] for m in per_seed_metrics if not np.isnan(m[k])], dtype=float)
        if len(vals) == 0:
            summary[k] = {"mean": float("nan"), "se": float("nan"),
                          "ci_lo": float("nan"), "ci_hi": float("nan"), "n": 0,
                          "t_crit": t_crit}
            continue
        mean = float(vals.mean())
        if len(vals) >= 2:
            se = float(vals.std(ddof=1) / np.sqrt(len(vals)))
            half = t_crit * se if not np.isnan(t_crit) else float("nan")
        else:
            se = float("nan")
            half = float("nan")
        summary[k] = {
            "mean": mean,
            "se": se,
            "ci_lo": mean - half if not np.isnan(half) else float("nan"),
            "ci_hi": mean + half if not np.isnan(half) else float("nan"),
            "n": len(vals),
            "t_crit": t_crit,
        }
    return summary
