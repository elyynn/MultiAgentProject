"""
Run the full FP-style experiment suite, with multi-training-seed support.

Usage:
    cd llm_hiring_sim
    python experiments/run_fp_suite.py

Per arm: cfg.num_training_seeds independent FP training runs (different seeds), each
followed by a held-out evaluation pass. Cross-seed aggregation uses Student-t
critical values (df = num_training_seeds - 1) — see audit Q15 / B1-FP.

Produces one directory per arm under outputs_fp/<arm>/ with:
    seeds/seed_<k>/   per-seed trajectories, regrets, eval metrics
    eval_summary.json (aggregated, with t-CI)
    fp_regret.json    (per-seed and aggregated)
    fp_convergence.json
    run_manifest.json
"""

from __future__ import annotations

import sys
import os
import json
import csv
import datetime
import dataclasses

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config_fp import FPConfig
from fictitious_play import (
    run_fictitious_play,
    check_convergence,
    compute_best_response_regret,
    normalize_rows,
)
from evaluate_fp import evaluate_one_seed, aggregate_per_seed_metrics
from payoffs import _reset_degenerate_counter, get_degenerate_posterior_count


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

    # fp_null_ai: Delta_theta=0, E_theta=0
    cfg = FPConfig()
    cfg.ai_signal_boost = {0: 0.0, 1: 0.0, 2: 0.0}
    cfg.ai_effort_benefit = {0: 0.0, 1: 0.0, 2: 0.0}
    arms["fp_null_ai"] = cfg

    # fp_fixed_firm: firm joint action fixed to (BaseVerify, suspicion=0.50).
    # Reports cycling, if any, in the candidate population — see audit I6-FP / Q8.
    cfg = FPConfig()
    cfg.fixed_firm = ("BaseVerify", 1)  # suspicion_idx=1 → 0.50
    arms["fp_fixed_firm"] = cfg

    # fp_high_verification_cost: doubled c_m. NULL ABLATION (audit B2-FP / Q3).
    # Kept here for documentation — at the baseline equilibrium LowVerify already
    # strictly dominates, so doubling the cost merely changes firm_welfare by a
    # constant offset and yields *byte-identical* counts. We do not replace this
    # arm; we explain it as a null result in the writeup.
    cfg = FPConfig()
    cfg.verification_cost = {m: 2.0 * v for m, v in FPConfig().verification_cost.items()}
    arms["fp_high_verification_cost"] = cfg

    # fp_tie_high: candidate breaks ties to AI=1 instead of No-AI=0 (audit I2-FP / Q4b).
    # Sanity probe for tie-break path-dependence in knife-edge BR regions.
    cfg = FPConfig()
    cfg.tie_break_prefer_lower = False
    arms["fp_tie_high"] = cfg

    return arms


# ---------------------------------------------------------------------------
# Run a single arm (multi-seed)
# ---------------------------------------------------------------------------

def run_arm(arm_name, cfg):
    print(f"\n[{arm_name}] training_seeds={cfg.num_training_seeds} "
          f"iterations={cfg.num_fp_iterations}")

    arm_dir = os.path.join(cfg.output_dir, arm_name)
    os.makedirs(os.path.join(arm_dir, "figures"), exist_ok=True)

    per_seed = []  # list of dicts holding the artefacts of each training seed
    eval_metrics_per_seed = []

    base_seed = cfg.seed

    for k in range(cfg.num_training_seeds):
        seed_k = base_seed + k
        seed_dir = os.path.join(arm_dir, "seeds", f"seed_{k}")
        os.makedirs(seed_dir, exist_ok=True)

        _reset_degenerate_counter()

        counts_c, counts_f, logs, flips = run_fictitious_play(
            cfg, fixed_firm=cfg.fixed_firm, seed=seed_k,
            desc=f"{arm_name} seed_{k}",
        )
        degen_train = get_degenerate_posterior_count()

        convergence = check_convergence(logs, flips, cfg)

        regret_rng = np.random.default_rng(seed_k + 99991)
        regret = compute_best_response_regret(counts_c, counts_f, cfg, regret_rng)

        eval_seed = seed_k + 9999
        _reset_degenerate_counter()
        metrics = evaluate_one_seed(counts_c, counts_f, cfg, eval_seed)
        degen_eval = get_degenerate_posterior_count()
        metrics["training_seed"] = seed_k
        metrics["training_seed_idx"] = k
        eval_metrics_per_seed.append(metrics)

        # Save per-seed artefacts
        _save_trajectory(logs, seed_dir)
        _save_final_strategies(counts_c, counts_f, cfg, seed_dir)
        _save_regret(regret, convergence, seed_dir, extra={
            "training_seed": seed_k,
            "degenerate_posterior_train": degen_train,
            "degenerate_posterior_eval": degen_eval,
        })
        _save_eval_metrics(metrics, seed_dir)

        per_seed.append({
            "training_seed": seed_k,
            "training_seed_idx": k,
            "counts_candidate": counts_c.tolist(),
            "counts_firm": counts_f.tolist(),
            "logs": logs,
            "flips": flips,
            "convergence": convergence,
            "regret": regret,
            "metrics": metrics,
        })

        print(f"  seed_{k}: r_max={regret['r_max']:.4f}  "
              f"strategy_stable={convergence['is_strategy_stable']}  "
              f"flips_window={convergence['flips_in_window']}  "
              f"flips_total={flips}")

    # ------------------------------------------------------------------
    # Aggregate across training seeds
    # ------------------------------------------------------------------
    eval_summary = aggregate_per_seed_metrics(eval_metrics_per_seed, cfg)
    regret_summary = _aggregate_regret(per_seed)
    convergence_summary = _aggregate_convergence(per_seed, cfg, regret_summary)

    _save_aggregated_strategies(per_seed, cfg, arm_dir)
    _save_eval_summary(eval_summary, arm_dir)
    _save_eval_per_seed(eval_metrics_per_seed, arm_dir)
    _save_regret(regret_summary, convergence_summary, arm_dir,
                 extra={"per_seed": [{"seed": p["training_seed"], "regret": p["regret"]}
                                      for p in per_seed]})
    _save_convergence(convergence_summary, per_seed, arm_dir)
    _save_manifest(arm_name, cfg, arm_dir)

    print(f"[{arm_name}] aggregated: r_max_mean={regret_summary['r_max']:.4f}  "
          f"converged={convergence_summary['is_converged']}")
    return {
        "arm": arm_name,
        "per_seed": per_seed,
        "eval_summary": eval_summary,
        "regret": regret_summary,
        "convergence": convergence_summary,
        "cfg": cfg,
        # Convenience fields used by aggregate/plot:
        "logs": per_seed[0]["logs"],  # representative trajectory (seed 0)
        "all_logs": [p["logs"] for p in per_seed],
        "counts_candidate": np.array(per_seed[0]["counts_candidate"]),
        "counts_firm": np.array(per_seed[0]["counts_firm"]),
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _aggregate_regret(per_seed):
    keys = ["r_max", "r_firm", "r_candidate_low", "r_candidate_medium", "r_candidate_high"]
    out = {}
    for k in keys:
        vals = np.array([p["regret"][k] for p in per_seed], dtype=float)
        out[k] = float(vals.mean())
        out[f"{k}_se"] = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) >= 2 else float("nan")
        out[f"{k}_max"] = float(vals.max())
    return out


def _aggregate_convergence(per_seed, cfg, regret_summary):
    """A run is 'converged' if every seed shows a stable strategy AND mean r_max < threshold."""
    all_stable = all(p["convergence"]["is_strategy_stable"] for p in per_seed)
    regret_ok = regret_summary["r_max"] < cfg.fp_regret_threshold
    return {
        "is_converged": bool(all_stable and regret_ok),
        "all_seeds_strategy_stable": all_stable,
        "r_max_mean_below_threshold": bool(regret_ok),
        "r_max_threshold": cfg.fp_regret_threshold,
        "max_delta_max": max(p["convergence"]["max_delta"] for p in per_seed),
        "flips_in_window_max": max(p["convergence"]["flips_in_window"] for p in per_seed),
        "flips_total_per_seed": [p["flips"] for p in per_seed],
    }


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_trajectory(logs, seed_dir):
    path = os.path.join(seed_dir, "fp_trajectory.csv")
    if not logs:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(logs[0].keys()))
        writer.writeheader()
        writer.writerows(logs)


def _save_final_strategies(counts_candidate, counts_firm, cfg, seed_dir):
    sigma_c = normalize_rows(counts_candidate)
    sigma_f = counts_firm / counts_firm.sum()
    joint_actions = list(cfg.firm_joint_actions())
    data = {
        "sigma_candidate": {
            str(theta): {
                str(action): float(sigma_c[i, j])
                for j, action in enumerate(cfg.candidate_actions)
            }
            for i, theta in enumerate(cfg.types)
        },
        "sigma_firm_joint": {
            cfg.firm_action_label(m, k): float(sigma_f[idx])
            for idx, (m, k) in enumerate(joint_actions)
        },
        "counts_candidate": counts_candidate.tolist(),
        "counts_firm": counts_firm.tolist(),
        "firm_joint_action_labels": [cfg.firm_action_label(m, k) for (m, k) in joint_actions],
    }
    with open(os.path.join(seed_dir, "final_empirical_strategies.json"), "w") as f:
        json.dump(data, f, indent=2)


def _save_aggregated_strategies(per_seed, cfg, arm_dir):
    counts_c = np.mean([np.array(p["counts_candidate"]) for p in per_seed], axis=0)
    counts_f = np.mean([np.array(p["counts_firm"]) for p in per_seed], axis=0)
    sigma_c = normalize_rows(counts_c)
    sigma_f = counts_f / counts_f.sum()
    joint_actions = list(cfg.firm_joint_actions())
    data = {
        "note": "Mean of per-seed normalised counts. Use seeds/seed_<k>/ for individual runs.",
        "sigma_candidate_mean": {
            str(theta): {
                str(action): float(sigma_c[i, j])
                for j, action in enumerate(cfg.candidate_actions)
            }
            for i, theta in enumerate(cfg.types)
        },
        "sigma_firm_joint_mean": {
            cfg.firm_action_label(m, k): float(sigma_f[idx])
            for idx, (m, k) in enumerate(joint_actions)
        },
        "per_seed_sigma_candidate": [
            normalize_rows(np.array(p["counts_candidate"])).tolist()
            for p in per_seed
        ],
        "per_seed_sigma_firm_joint": [
            (np.array(p["counts_firm"]) / np.array(p["counts_firm"]).sum()).tolist()
            for p in per_seed
        ],
        "firm_joint_action_labels": [cfg.firm_action_label(m, k) for (m, k) in joint_actions],
    }
    with open(os.path.join(arm_dir, "final_empirical_strategies.json"), "w") as f:
        json.dump(data, f, indent=2)


def _np_clean(obj):
    if isinstance(obj, dict):
        return {k: _np_clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_np_clean(v) for v in obj]
    if isinstance(obj, tuple):
        return [_np_clean(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _save_regret(regret, convergence, out_dir, extra=None):
    payload = {"regret": regret, "convergence": convergence}
    if extra is not None:
        payload.update(extra)
    with open(os.path.join(out_dir, "fp_regret.json"), "w") as f:
        json.dump(_np_clean(payload), f, indent=2)


def _save_convergence(convergence, per_seed, arm_dir):
    payload = {
        "summary": convergence,
        "per_seed": [
            {"training_seed": p["training_seed"], **p["convergence"], "flips_total": p["flips"]}
            for p in per_seed
        ],
    }
    with open(os.path.join(arm_dir, "fp_convergence.json"), "w") as f:
        json.dump(_np_clean(payload), f, indent=2)


def _save_eval_metrics(metrics, seed_dir):
    with open(os.path.join(seed_dir, "eval_metrics.json"), "w") as f:
        json.dump(_np_clean(metrics), f, indent=2)


def _save_eval_summary(eval_summary, arm_dir):
    with open(os.path.join(arm_dir, "eval_summary.json"), "w") as f:
        json.dump(eval_summary, f, indent=2)


def _save_eval_per_seed(metrics_list, arm_dir):
    if not metrics_list:
        return
    path = os.path.join(arm_dir, "eval_per_seed.csv")
    fieldnames = list(metrics_list[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_list)


def _save_manifest(arm_name, cfg, arm_dir):
    """Full dataclasses.asdict(cfg) dump per audit M1-FP."""
    cfg_dict = dataclasses.asdict(cfg)
    # Type_prior keys are ints — JSON requires str keys
    cfg_dict = _np_clean(cfg_dict)
    cfg_dict = _stringify_int_keys(cfg_dict)
    manifest = {
        "arm": arm_name,
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "config": cfg_dict,
    }
    with open(os.path.join(arm_dir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def _stringify_int_keys(obj):
    if isinstance(obj, dict):
        return {str(k) if isinstance(k, (int, np.integer)) else k: _stringify_int_keys(v)
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify_int_keys(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    arms = make_arm_configs()
    results = {}
    for arm_name, cfg in arms.items():
        results[arm_name] = run_arm(arm_name, cfg)

    print("\nAll arms complete. Aggregating and plotting...")

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    from aggregate_fp import aggregate_results
    from plot_fp_results import plot_all

    compare_dir = os.path.join("outputs_fp", "_compare")
    os.makedirs(compare_dir, exist_ok=True)
    aggregate_results(results, compare_dir)
    plot_all(results, compare_dir)

    print(f"\nDone. Results in outputs_fp/")
