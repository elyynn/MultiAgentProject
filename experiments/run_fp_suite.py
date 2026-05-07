"""
Run the full fictitious-play experiment suite.

Usage:
    cd llm_hiring_sim
    python experiments/run_fp_suite.py

Produces one output directory per arm under outputs_fp/<arm>/.
"""

import sys
import os
import json
import copy
import datetime
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config_fp import FPConfig
from fictitious_play import (
    run_fictitious_play,
    check_convergence,
    compute_best_response_regret,
    normalize_rows,
    _make_crn_rng,
)
from evaluate_fp import evaluate_frozen_strategies


# ---------------------------------------------------------------------------
# Arm definitions
# ---------------------------------------------------------------------------

def make_arm_configs():
    arms = {}

    # fp_default: baseline
    arms["fp_default"] = FPConfig()

    # fp_no_detection: pi_theta=0, phi_m=0
    cfg = FPConfig()
    cfg.base_detection_prob = {0: 0.0, 1: 0.0, 2: 0.0}
    cfg.false_positive_rate = {"LowVerify": 0.0, "BaseVerify": 0.0, "HighVerify": 0.0}
    arms["fp_no_detection"] = cfg

    # fp_no_reputation: lambda_R=0
    cfg = FPConfig()
    cfg.reputation_penalty_weight = 0.0
    arms["fp_no_reputation"] = cfg

    # fp_null_ai: Delta_theta=0, E_theta=0 (no AI signal/effort benefit)
    cfg = FPConfig()
    cfg.ai_signal_boost = {0: 0.0, 1: 0.0, 2: 0.0}
    cfg.ai_effort_benefit = {0: 0.0, 1: 0.0, 2: 0.0}
    arms["fp_null_ai"] = cfg

    # fp_fixed_firm: firm policy fixed to BaseVerify
    cfg = FPConfig()
    cfg._fixed_firm = "BaseVerify"
    arms["fp_fixed_firm"] = cfg

    # fp_high_verification_cost: double all c_m
    cfg = FPConfig()
    cfg.verification_cost = {m: 2.0 * v for m, v in FPConfig().verification_cost.items()}
    arms["fp_high_verification_cost"] = cfg

    return arms


# ---------------------------------------------------------------------------
# Run a single arm
# ---------------------------------------------------------------------------

def run_arm(arm_name, cfg):
    print(f"\n[{arm_name}] Starting fictitious play ({cfg.num_fp_iterations} iterations)...")

    fixed_firm = getattr(cfg, "_fixed_firm", None)

    if fixed_firm is not None:
        counts_candidate, counts_firm, logs = _run_fixed_firm(cfg, fixed_firm)
    else:
        counts_candidate, counts_firm, logs = run_fictitious_play(cfg)

    convergence = check_convergence(logs, cfg)
    print(f"[{arm_name}] Convergence check: {convergence}")

    # Regret
    regret_rng = np.random.default_rng(cfg.seed + 1)
    regret = compute_best_response_regret(counts_candidate, counts_firm, cfg, regret_rng)
    print(f"[{arm_name}] Best-response regret r_max={regret['r_max']:.4f}")

    # Held-out evaluation
    print(f"[{arm_name}] Running held-out evaluation ({cfg.num_eval_seeds} seeds)...")
    eval_summary, eval_per_seed = evaluate_frozen_strategies(counts_candidate, counts_firm, cfg)
    print(f"[{arm_name}] Evaluation complete.")

    # Save outputs
    arm_dir = os.path.join(cfg.output_dir, arm_name)
    os.makedirs(os.path.join(arm_dir, "figures"), exist_ok=True)

    _save_trajectory(logs, arm_dir)
    _save_final_strategies(counts_candidate, counts_firm, cfg, arm_dir)
    _save_regret(regret, convergence, arm_dir)
    _save_eval_summary(eval_summary, arm_dir)
    _save_eval_per_seed(eval_per_seed, arm_dir)
    _save_manifest(arm_name, cfg, arm_dir)

    print(f"[{arm_name}] Saved to {arm_dir}/")
    return {
        "arm": arm_name,
        "counts_candidate": counts_candidate,
        "counts_firm": counts_firm,
        "logs": logs,
        "convergence": convergence,
        "regret": regret,
        "eval_summary": eval_summary,
        "eval_per_seed": eval_per_seed,
        "cfg": cfg,
    }


def _run_fixed_firm(cfg, fixed_firm):
    """Run fictitious play with firm policy fixed to fixed_firm."""
    import numpy as np
    from payoffs import candidate_expected_utility, _make_crn_rng
    from fictitious_play import normalize_rows, _sigma_c_dict, _conservative_argmax

    fixed_firm_idx = cfg.firm_actions.index(fixed_firm)
    rng = np.random.default_rng(cfg.seed)

    counts_candidate = np.ones((len(cfg.types), len(cfg.candidate_actions)))
    counts_firm = np.zeros(len(cfg.firm_actions))
    counts_firm[fixed_firm_idx] = cfg.num_fp_iterations + len(cfg.firm_actions)

    logs = []

    for t in range(cfg.num_fp_iterations):
        sigma_c = _sigma_c_dict(counts_candidate, cfg)
        sigma_f = counts_firm / counts_firm.sum()
        crn_rng = _make_crn_rng(seed=t) if cfg.common_random_numbers else rng

        candidate_br = {}
        for i, theta in enumerate(cfg.types):
            action_utils = [
                candidate_expected_utility(theta, action, fixed_firm, sigma_c, cfg, crn_rng)
                for action in cfg.candidate_actions
            ]
            candidate_br[theta] = cfg.candidate_actions[
                _conservative_argmax(action_utils, prefer_lower=True, tol=cfg.tie_tol)
            ]

        for i, theta in enumerate(cfg.types):
            a_star = candidate_br[theta]
            j = cfg.candidate_actions.index(a_star)
            counts_candidate[i, j] += 1

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
            "firm_br": fixed_firm,
        })

    return counts_candidate, counts_firm, logs


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_trajectory(logs, arm_dir):
    path = os.path.join(arm_dir, "fp_trajectory.csv")
    if not logs:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(logs[0].keys()))
        writer.writeheader()
        writer.writerows(logs)


def _save_final_strategies(counts_candidate, counts_firm, cfg, arm_dir):
    sigma_c = normalize_rows(counts_candidate)
    sigma_f = counts_firm / counts_firm.sum()
    data = {
        "sigma_candidate": {
            str(theta): {
                str(action): float(sigma_c[i, j])
                for j, action in enumerate(cfg.candidate_actions)
            }
            for i, theta in enumerate(cfg.types)
        },
        "sigma_firm": {
            m: float(sigma_f[j])
            for j, m in enumerate(cfg.firm_actions)
        },
        "counts_candidate": counts_candidate.tolist(),
        "counts_firm": counts_firm.tolist(),
    }
    with open(os.path.join(arm_dir, "final_empirical_strategies.json"), "w") as f:
        json.dump(data, f, indent=2)


def _np_clean(obj):
    """Recursively convert numpy scalars to Python natives for JSON."""
    if isinstance(obj, dict):
        return {k: _np_clean(v) for k, v in obj.items()}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, np.bool_)):
        return float(obj) if isinstance(obj, np.floating) else bool(obj)
    return obj


def _save_regret(regret, convergence, arm_dir):
    data = _np_clean({"regret": regret, "convergence": convergence})
    with open(os.path.join(arm_dir, "fp_regret.json"), "w") as f:
        json.dump(data, f, indent=2)


def _save_eval_summary(eval_summary, arm_dir):
    with open(os.path.join(arm_dir, "eval_summary.json"), "w") as f:
        json.dump(eval_summary, f, indent=2)


def _save_eval_per_seed(eval_per_seed, arm_dir):
    path = os.path.join(arm_dir, "eval_per_seed.csv")
    if not eval_per_seed:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(eval_per_seed[0].keys()))
        writer.writeheader()
        writer.writerows(eval_per_seed)


def _save_manifest(arm_name, cfg, arm_dir):
    manifest = {
        "arm": arm_name,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "num_fp_iterations": cfg.num_fp_iterations,
        "num_eval_samples": cfg.num_eval_samples,
        "num_eval_seeds": cfg.num_eval_seeds,
        "seed": cfg.seed,
        "fixed_firm": getattr(cfg, "_fixed_firm", None),
    }
    with open(os.path.join(arm_dir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    arms = make_arm_configs()
    results = {}
    for arm_name, cfg in arms.items():
        results[arm_name] = run_arm(arm_name, cfg)

    print("\nAll arms complete. Running aggregation and plotting...")

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    from aggregate_fp import aggregate_results
    from plot_fp_results import plot_all

    compare_dir = os.path.join("outputs_fp", "_compare")
    os.makedirs(compare_dir, exist_ok=True)
    aggregate_results(results, compare_dir)
    plot_all(results, compare_dir)

    print(f"\nDone. Results in outputs_fp/")
