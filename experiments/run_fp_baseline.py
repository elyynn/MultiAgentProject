"""
Run only the fp_default (baseline) arm of the fictitious-play suite.

Usage:
    cd llm_hiring_sim
    python experiments/run_fp_baseline.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_fp import FPConfig
from fictitious_play import (
    run_fictitious_play,
    check_convergence,
    compute_best_response_regret,
)
from evaluate_fp import evaluate_frozen_strategies
from run_fp_suite import (
    _save_trajectory,
    _save_final_strategies,
    _save_regret,
    _save_eval_summary,
    _save_eval_per_seed,
    _save_manifest,
)
from plot_fp_results import _plot_candidate_trajectory, _plot_firm_trajectory

import numpy as np

ARM_NAME = "fp_default"

cfg = FPConfig()

arm_dir = os.path.join(cfg.output_dir, ARM_NAME)
os.makedirs(os.path.join(arm_dir, "figures"), exist_ok=True)

print(f"[{ARM_NAME}] Starting fictitious play ({cfg.num_fp_iterations} iterations)...")
counts_candidate, counts_firm, logs = run_fictitious_play(cfg)

convergence = check_convergence(logs, cfg)
print(f"[{ARM_NAME}] Convergence: {convergence}")

regret_rng = np.random.default_rng(cfg.seed + 1)
regret = compute_best_response_regret(counts_candidate, counts_firm, cfg, regret_rng)
print(f"[{ARM_NAME}] r_max = {regret['r_max']:.4f}")

print(f"[{ARM_NAME}] Running held-out evaluation ({cfg.num_eval_seeds} seeds × {cfg.num_eval_samples:,} samples)...")
eval_summary, eval_per_seed = evaluate_frozen_strategies(counts_candidate, counts_firm, cfg)

_save_trajectory(logs, arm_dir)
_save_final_strategies(counts_candidate, counts_firm, cfg, arm_dir)
_save_regret(regret, convergence, arm_dir)
_save_eval_summary(eval_summary, arm_dir)
_save_eval_per_seed(eval_per_seed, arm_dir)
_save_manifest(ARM_NAME, cfg, arm_dir)
_plot_candidate_trajectory(ARM_NAME, logs, arm_dir)
_plot_firm_trajectory(ARM_NAME, logs, arm_dir)

print(f"\nDone. Results in {arm_dir}/")
print(f"  AI adoption overall : {eval_summary['ai_adoption_overall']['mean']:.3f}")
print(f"  AI adoption by type : low={eval_summary['ai_adoption_low']['mean']:.3f}"
      f"  med={eval_summary['ai_adoption_medium']['mean']:.3f}"
      f"  high={eval_summary['ai_adoption_high']['mean']:.3f}")
print(f"  Correct match rate  : {eval_summary['correct_match_rate']['mean']:.3f}")
print(f"  Candidate welfare   : {eval_summary['candidate_welfare']['mean']:.3f}")
print(f"  Firm welfare        : {eval_summary['firm_welfare']['mean']:.3f}")
