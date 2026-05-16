"""
Run only the fp_default (baseline) arm of the FP-style suite.

Usage:
    cd llm_hiring_sim
    python experiments/run_fp_baseline.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_fp import FPConfig
from run_fp_suite import run_arm
from plot_fp_results import _plot_candidate_trajectory, _plot_firm_trajectory


ARM_NAME = "fp_default"

cfg = FPConfig()
result = run_arm(ARM_NAME, cfg)

# Per-arm trajectory plots — one per training seed.
arm_dir = os.path.join(cfg.output_dir, ARM_NAME)
for k, per_seed_logs in enumerate(result["all_logs"]):
    seed_dir = os.path.join(arm_dir, "seeds", f"seed_{k}")
    os.makedirs(os.path.join(seed_dir, "figures"), exist_ok=True)
    _plot_candidate_trajectory(f"{ARM_NAME} seed_{k}", per_seed_logs, seed_dir)
    _plot_firm_trajectory(f"{ARM_NAME} seed_{k}", per_seed_logs, seed_dir)

s = result["eval_summary"]
print(f"\nDone. Results in {arm_dir}/")
print(f"  Held-out evaluation: {cfg.num_training_seeds} training seeds × "
      f"{cfg.num_eval_samples:,} samples each (1 eval pass per training seed)")
print(f"  AI adoption overall : {s['ai_adoption_overall']['mean']:.3f} "
      f"± {s['ai_adoption_overall']['se']:.3f} (t-CI)")
print(f"  AI adoption by type : low={s['ai_adoption_low']['mean']:.3f}"
      f"  med={s['ai_adoption_medium']['mean']:.3f}"
      f"  high={s['ai_adoption_high']['mean']:.3f}")
print(f"  Correct match rate  : {s['correct_match_rate']['mean']:.3f}")
print(f"  Candidate welfare   : {s['candidate_welfare']['mean']:.3f}")
print(f"  Firm welfare        : {s['firm_welfare']['mean']:.3f}")
