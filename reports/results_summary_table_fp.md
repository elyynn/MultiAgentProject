# Results Summary Table — FP Version

One row per artifact × arm combination. "Internally consistent?" covers
r_max arithmetic, CI bracketing, and counts-sum checks. "Cross-arm
plausible?" covers directional monotonicity against theoretical expectations.
"CI valid?" covers whether bounds bracket the mean at 1.96×SE and whether
the CI measures the right quantity (evaluation-MC vs training-seed).

| Artifact | Exists? | Internally consistent? | Cross-arm plausible? | CI valid? | Verdict |
|---|---|---|---|---|---|
| `outputs_fp/fp_default/run_manifest.json` | YES | YES — arm, seed=42, num_fp_iterations=2000, num_eval_seeds=30 recorded | n/a (single arm) | n/a | WARN — arm-specific config overrides absent; no git SHA |
| `outputs_fp/fp_default/final_empirical_strategies.json` | YES | YES — counts sum to 2002 (candidate) and 2003 (firm); sigma values match counts/sum | YES — LowVerify dominates (0.9990); Low-type adoption 0.0707 consistent with knife-edge regret | n/a | PASS |
| `outputs_fp/fp_default/fp_regret.json` | YES | YES — r_max=0.000942 equals max(r_firm=0.000121, r_c_low=0.000942, r_c_med=0.000580, r_c_high=0.000833) ✓ | YES — lowest r_max among arms with free firm | n/a | PASS |
| `outputs_fp/fp_default/eval_summary.json` | YES | YES — CI bounds bracket means at 1.96×SE; firm_policy sums ≈1.0 | YES — welfare and adoption numbers directionally coherent | WARN — CIs are evaluation-MC only (30 MC seeds on 1 training seed); misleadingly narrow (B1-FP-R) | WARN |
| `outputs_fp/fp_default/eval_per_seed.csv` | YES | YES — 30 data rows; per-row values within plausible range of eval_summary means | YES | n/a — no CI in this file | PASS |
| `outputs_fp/fp_default/fp_trajectory.csv` | YES | YES — iteration 0→2000; sigma values evolve plausibly | YES | n/a | PASS |
| `outputs_fp/fp_default/figures/candidate_trajectory.png` | YES | Not numerically verified (image) | n/a | n/a | PASS |
| `outputs_fp/fp_default/figures/firm_trajectory.png` | YES | Not numerically verified (image) | n/a | n/a | PASS |
| `outputs_fp/fp_no_detection/run_manifest.json` | YES | YES — seed=42, fixed_firm=null | n/a | n/a | WARN — arm config (base_detection_prob=0) not recorded |
| `outputs_fp/fp_no_detection/final_empirical_strategies.json` | YES | YES — counts_candidate all [1,2001]; sigma_candidate ≈0.9995 AI for all types; firm strategy identical to fp_default | YES — universal AI adoption expected when detection=0 | n/a | PASS |
| `outputs_fp/fp_no_detection/fp_regret.json` | YES | YES — r_max=0.000338 = r_candidate_medium ✓ | YES — low regret consistent with dominant strategy | n/a | PASS |
| `outputs_fp/fp_no_detection/eval_summary.json` | YES | YES — detection_rate_given_ai=0.0 exactly with SE=0, CI=[0,0] ✓; CI bounds bracket means at 1.96×SE | YES — AI adoption 0.9995 vs default 0.7186; correct_match 0.9085 < 0.9422 ✓ | WARN — eval-MC CI only; firm_policy_* values byte-identical to fp_default (W1-FP-R) | WARN |
| `outputs_fp/fp_no_detection/eval_per_seed.csv` | YES | YES — 30 rows | YES | n/a | PASS |
| `outputs_fp/fp_no_detection/figures/*.png` | YES (2) | n/a | n/a | n/a | PASS |
| `outputs_fp/fp_no_reputation/run_manifest.json` | YES | YES | n/a | n/a | WARN — arm config (reputation_penalty_weight=0) not recorded |
| `outputs_fp/fp_no_reputation/final_empirical_strategies.json` | YES | YES — counts sum to 2002 / 2003; Low-type: 293/1709; sigma_firm identical to fp_default | YES — Low-type AI rises from 7% to 85% when reputation removed; Medium/High unchanged | n/a | PASS |
| `outputs_fp/fp_no_reputation/fp_regret.json` | YES | YES — r_max=0.001411 = r_candidate_low ✓ | YES — highest r_max of free-firm arms consistent with Low-type indifference zone | n/a | PASS |
| `outputs_fp/fp_no_reputation/eval_summary.json` | YES | YES — CI bounds valid at 1.96×SE; firm_policy values identical to fp_default (W1-FP-R) | YES — adoption 0.9557 > 0.7186; correct_match 0.9125 < 0.9422; candidate_welfare 0.8665 > 0.8075 ✓ | WARN — eval-MC CI only | WARN |
| `outputs_fp/fp_no_reputation/eval_per_seed.csv` | YES | YES — 30 rows | YES | n/a | PASS |
| `outputs_fp/fp_no_reputation/figures/*.png` | YES (2) | n/a | n/a | n/a | PASS |
| `outputs_fp/fp_null_ai/run_manifest.json` | YES | YES | n/a | n/a | WARN — arm config not recorded |
| `outputs_fp/fp_null_ai/final_empirical_strategies.json` | YES | YES — counts_candidate all [2001,1] (no AI); firm strategy identical to fp_default | YES — no AI expected when ai_signal_boost=0 | n/a | PASS |
| `outputs_fp/fp_null_ai/fp_regret.json` | YES | YES — r_max=0.0001964 = r_candidate_low ✓; smallest r_max of all arms | YES — near-optimal no-AI play | n/a | PASS |
| `outputs_fp/fp_null_ai/eval_summary.json` | YES | YES — CI bounds valid; detection_rate_given_ai SE=0.00879 (wide, low-event conditioning, W3-FP-R) | YES — correct_match 0.9683 highest; ai_adoption ~0.0005 ✓; firm_welfare -0.0518 least negative ✓ | WARN — eval-MC CI only; detection CIs degenerate (W3-FP-R) | WARN |
| `outputs_fp/fp_null_ai/eval_per_seed.csv` | YES | YES — 30 rows | YES | n/a | PASS |
| `outputs_fp/fp_null_ai/figures/*.png` | YES (2) | n/a | n/a | n/a | PASS |
| `outputs_fp/fp_fixed_firm/run_manifest.json` | YES | YES — fixed_firm=BaseVerify recorded | n/a | n/a | WARN — full config not recorded |
| `outputs_fp/fp_fixed_firm/final_empirical_strategies.json` | YES | WARN — counts_firm=[0,2003,0] has LowVerify and HighVerify at 0 (not Laplace-1 init); total 2003 ✓ but init discrepancy (W5-FP-R) | YES — sigma_firm=(0,1,0) correct for BaseVerify fixed; Low/Med candidate adoption ~3% | n/a | WARN |
| `outputs_fp/fp_fixed_firm/fp_regret.json` | YES | YES — r_max=0.0595 = r_firm=0.0595 ✓ | YES — r_max is 63x other converged arms | FAIL — is_converged=True is false in the equilibrium sense; r_max=0.0595 far exceeds any reasonable threshold (B3-FP-R) | FAIL |
| `outputs_fp/fp_fixed_firm/eval_summary.json` | YES | YES — CI bounds bracket means at 1.96×SE; detection_rate_given_ai_high SE=0.01515 (wide, W3-FP-R) | YES — AI adoption 0.0239 lowest free-candidate arm; firm_welfare -0.1230 most negative ✓ | WARN — eval-MC CI only; not an equilibrium strategy (B3-FP-R) | WARN |
| `outputs_fp/fp_fixed_firm/eval_per_seed.csv` | YES | YES — 30 rows | YES | n/a | PASS |
| `outputs_fp/fp_fixed_firm/figures/*.png` | YES (2) | n/a | n/a | n/a | PASS |
| `outputs_fp/fp_high_verification_cost/run_manifest.json` | YES | YES | n/a | n/a | WARN — arm config (doubled verification_cost) not recorded |
| `outputs_fp/fp_high_verification_cost/final_empirical_strategies.json` | YES | FAIL — byte-identical to fp_default (counts_candidate: [[1860,142],[7,1995],[9,1993]], counts_firm: [2001,1,1]); null ablation (B2-FP-R) | FAIL — no equilibrium change despite different cost (B2-FP-R) | n/a | FAIL |
| `outputs_fp/fp_high_verification_cost/fp_regret.json` | YES | FAIL — r_max=0.000942, r_candidate_* values identical to fp_default; null ablation (B2-FP-R) | FAIL — null ablation | n/a | FAIL |
| `outputs_fp/fp_high_verification_cost/eval_summary.json` | YES | YES — CI bounds valid; firm_welfare -0.09807 differs from fp_default -0.07795 by exactly the cost increment at eval | FAIL — strategy and adoption numbers identical to fp_default; only firm_welfare differs (B2-FP-R) | WARN — eval-MC CI only | FAIL |
| `outputs_fp/fp_high_verification_cost/eval_per_seed.csv` | YES | YES — 30 rows | FAIL — values effectively identical to fp_default rows | n/a | WARN |
| `outputs_fp/fp_high_verification_cost/figures/*.png` | YES (2) | n/a — visually identical to fp_default figures | n/a | n/a | WARN |
| `outputs_fp/_compare/comparison_table.md` | YES | YES — numbers match source eval_summary.json to displayed precision for all arms | YES — directional ordering correct for distinct arms | WARN — is_converged=True for fp_fixed_firm contradicted by r_max=0.0595 (B3-FP-R); no CI on most columns | WARN |
| `outputs_fp/_compare/comparison_table.csv` | YES | YES — matches .md; r_max and sigma values consistent | YES | WARN — CI stored as unparseable string columns (M4-FP-R); eval-MC CIs only | WARN |
| `outputs_fp/_compare/ai_adoption_by_type.png` | YES | n/a (image) | n/a | n/a | PASS |
| `outputs_fp/_compare/detection_calibration.png` | YES | n/a (image) | n/a | n/a | PASS |
| `outputs_fp/_compare/firm_policy_distribution.png` | YES | n/a (image) — trivially uniform bars for all non-fixed arms (M5-FP-R) | n/a | n/a | WARN |
| `outputs_fp/_compare/match_efficiency_by_arm.png` | YES | n/a (image) | n/a | n/a | PASS |
| `outputs_fp/_compare/regret_by_arm.png` | YES | n/a — 0.03 threshold line visually exposes fp_fixed_firm non-convergence (B3-FP-R) | n/a | n/a | PASS |
| `outputs_fp/_compare/strategy_convergence.png` | YES | n/a (image) | n/a | n/a | PASS |
| `outputs_fp/_compare/welfare_by_arm.png` | YES | n/a (image) | n/a | n/a | PASS |
| Training-seed sweep (multi-seed FP run) | MISSING | MISSING | MISSING | MISSING | MISSING |
| Per-arm config dump (`config_dump.json`) | MISSING | MISSING | MISSING | MISSING | MISSING |
| FP figure references in `paper/main.tex` | MISSING | MISSING | MISSING | MISSING | MISSING |
| BR-flip log per arm | MISSING | MISSING | MISSING | MISSING | MISSING |
