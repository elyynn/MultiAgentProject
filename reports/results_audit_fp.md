# Results Audit — FP Version

## Executive Summary

The FP experiment suite produces six arms of internally consistent JSON/CSV
artifacts with correctly computed r_max values, bracket-valid CIs, and
plausible cross-arm monotonicity for the four genuinely distinct arms.
However, two structural problems dominate: (1) all reported CIs are
evaluation-only Monte-Carlo intervals on a single frozen training run (seed=42),
making every effect-size estimate in `comparison_table.md` statistically
unsupported with respect to training variability; and (2) `fp_high_verification_cost`
is a confirmed null ablation — its candidate strategies, firm strategy, and
regret values are byte-identical to `fp_default`, so the suite effectively
has five arms, not six. Additionally, `fp_fixed_firm` is labelled
`is_converged=True` despite an r_max of 0.0595 — 63x the next-highest
converged arm — indicating a label/definition mismatch that would mislead
any reader scanning the comparison table.

---

## Blocking Findings

### B1-FP-R — CIs are evaluation-sample intervals, not training-seed intervals; every effect size is statistically unsupported

**What it is.** Every `eval_summary.json` across all six arms reports
`ci_lo`/`ci_hi` computed from 30 Monte-Carlo evaluation seeds run against a
single frozen strategy (`seed=42`). The manifest confirms this:

```
outputs_fp/fp_default/run_manifest.json:4-8
{
  "num_fp_iterations": 2000,
  "num_eval_samples": 100000,
  "num_eval_seeds": 30,
  "seed": 42
}
```

The CIs are thus CIs on the *sample mean of 30 × 100 000-sample draws from
one fixed distribution* — they shrink as sqrt(num_eval_samples) and tell
the reader nothing about how the equilibrium itself varies with the random
training path. Because FP draws fresh Monte-Carlo payoff samples each
iteration (`fictitious_play.py:122`), the converged strategy is a function
of the training seed; this variability is entirely uncharted.

**Where.** All six `eval_summary.json` files and the derived
`outputs_fp/_compare/comparison_table.md:5-10` and
`outputs_fp/_compare/comparison_table.csv:2-7`.

**Why it blocks.** Every quantitative claim derivable from the comparison
table (e.g., "fp_no_detection raises AI adoption from 0.7186 to 0.9995")
carries a CI width of ~0.001 that implies 3-decimal-place precision about the
equilibrium. The actual equilibrium-level uncertainty (across training seeds)
is unquantified and could be substantially larger, particularly for arms
where the Low-type BR is knife-edge (`r_candidate_low` dominates `r_max` in
fp_default and fp_no_reputation). The CI presentation actively misleads.

**What to do.** Add a top-level training-seed loop (e.g., 20 seeds) wrapping
`run_fictitious_play`, aggregate strategies across training seeds, and report
mean ± SE across training seeds. Until then, relabel all CIs in every output
file and the comparison table as "evaluation Monte-Carlo CI on one training
seed" and remove any language suggesting significance or robustness.

---

### B2-FP-R — `fp_high_verification_cost` is a confirmed null ablation: identical strategies and regret to `fp_default`

**What it is.** The candidate strategies, firm strategies, and regret
components of `fp_high_verification_cost` are byte-identical to `fp_default`.

Confirmed from raw JSONs:

`outputs_fp/fp_default/final_empirical_strategies.json:21-39`
```
counts_candidate: [[1860, 142], [7, 1995], [9, 1993]]
counts_firm:      [2001, 1, 1]
```

`outputs_fp/fp_high_verification_cost/final_empirical_strategies.json:21-39`
```
counts_candidate: [[1860, 142], [7, 1995], [9, 1993]]
counts_firm:      [2001, 1, 1]
```

Regret values are identical:
- fp_default:                r_max=0.000942, r_candidate_low=0.000942
  (`outputs_fp/fp_default/fp_regret.json:3,5`)
- fp_high_verification_cost: r_max=0.000942, r_candidate_low=0.000942
  (`outputs_fp/fp_high_verification_cost/fp_regret.json:3,5`)

The only number that differs is `firm_welfare`: -0.07795 (default) vs -0.09807
(high-vc) (`outputs_fp/fp_default/eval_summary.json:75` and
`outputs_fp/fp_high_verification_cost/eval_summary.json:75`). This difference
(-0.02012) is purely a deterministic accounting of the doubled verification
cost applied at evaluation time — it does not reflect any equilibrium
response, because `LowVerify` is still strictly dominant after the cost
increase. The comparison table (`outputs_fp/_compare/comparison_table.md:10`)
presents this arm alongside the other ablations without flagging that it
produces no equilibrium change.

**Where.** `outputs_fp/fp_high_verification_cost/` (all files);
`outputs_fp/_compare/comparison_table.md:10`.

**Why it blocks.** The arm is presented as a sensitivity probe but provides
zero equilibrium information. Any claim like "AI adoption and match quality
are robust to doubling verification cost" is trivially true by construction
in this parametrisation (the firm's dominant strategy is unchanged), not an
empirical finding.

**What to do.** Replace this arm with a verification-cost level that bridges
the gap to `BaseVerify` becoming preferred, or scan a range of costs. Label
the arm in the comparison table as "null ablation (no equilibrium change)" if
retained for completeness.

---

### B3-FP-R — `fp_fixed_firm` is labelled `is_converged=True` despite r_max=0.0595, which is 63x the equilibrium threshold; the convergence label is false in the relevant sense

**What it is.** `outputs_fp/fp_fixed_firm/fp_regret.json:3,4`:
```json
"r_max": 0.0595,
"r_firm": 0.0595
```
`outputs_fp/fp_fixed_firm/fp_regret.json:8-9`:
```json
"is_converged": true,
"max_delta": 0.0005314
```

The `is_converged` flag is triggered by `max_delta < 0.002` (a step-size
criterion that is trivially satisfied at late iterations regardless of
equilibrium status), not by `r_max`. At 2000 iterations, every arm's
step size is bounded by 1/2003 ≈ 0.0005 < 0.002, so every arm will be
marked converged by construction. But `r_max=0.0595` means the firm is
leaving ~6% of available utility on the table — the `is_converged=True`
label directly contradicts this.

`outputs_fp/fp_fixed_firm/final_empirical_strategies.json:21-33` shows the
candidate counts: Low [1943, 59], Medium [1942, 60], High [2001, 1]. The
~3% AI adoption by Low and Medium types (59/2002 = 0.0295, 60/2002 = 0.0300)
with ~97% non-AI is consistent with the FP path oscillating between BRs
rather than settling. The fixed firm's 6% regret is the firm's regret from
being forced to play BaseVerify when LowVerify would be preferred.

The comparison table (`outputs_fp/_compare/comparison_table.md:9`) reports
`is_converged: True` and `r_max: 0.0595` in adjacent columns, which is
self-contradictory to any reader who uses `r_max` as an equilibrium
criterion (as `plot_fp_results.py:193` implies with its 0.03 threshold line).

**Where.** `outputs_fp/fp_fixed_firm/fp_regret.json:3,8-9`;
`outputs_fp/_compare/comparison_table.md:9`.

**Why it blocks.** Any claim that all six arms have "converged" to
near-equilibrium strategies is false for this arm. The candidate's 6%
utility loss relative to best-response means the reported candidate welfare
(0.8945) and AI adoption (0.0239) for this arm are not equilibrium values
but depend on the specific 2000-iteration path and the `prefer_lower=True`
tie-break convention.

**What to do.** Define `is_converged` as `r_max < threshold` (e.g., 0.01)
rather than as a step-size criterion. Report `fp_fixed_firm` as
"not converged" and flag it as potential FP cycling evidence (a substantive
finding in its own right). Do not present its strategy mixture as a
representative equilibrium.

---

## Weak Findings

### W1-FP-R — Firm strategy is identical across all five non-fixed arms; the firm side provides no differential signal

**What it is.** Across fp_default, fp_no_detection, fp_no_reputation,
fp_null_ai, and fp_high_verification_cost, the firm strategy is byte-identical:

```
sigma_firm: {LowVerify: 0.99900149775337, BaseVerify: 0.0004992511233150275,
             HighVerify: 0.0004992511233150275}
counts_firm: [2001, 1, 1]
```

(Verified in all five `final_empirical_strategies.json` files, `sigma_firm`
block.) This means that the FP game as configured has a strictly dominant
firm strategy (LowVerify), and the firm never adapts across any of the
experimental conditions that vary candidate behavior.

**Why it matters.** The suite cannot support any claim about how firms
adjust their verification policy in response to different candidate
equilibria — the firm strategy is invariant. This also means the
`firm_policy_*` columns in `comparison_table.md` are uninformative for
all but `fp_fixed_firm`. Any language suggesting "firms respond to AI
adoption" is not supported.

**What to do.** Acknowledge that LowVerify is strictly dominant in the
current parametrisation. To make firm strategy endogenous, the cost
structure must be modified so that verification policies are not trivially
ordered.

---

### W2-FP-R — fp_no_reputation and fp_no_detection have identical firm strategies; ablations only affect candidate strategy

**What it is.** Removing reputation (`fp_no_reputation`) raises Low-type AI
adoption from 0.0707 to 0.8536 (`eval_summary.json:9` in each arm) but the
firm strategy is unchanged (LowVerify dominant). Removing detection
(`fp_no_detection`) drives all-type AI adoption to ~0.9995 but again firm
strategy is unchanged.

The comparison table (`comparison_table.md:6,7`) shows:
- fp_no_detection: candidate_welfare=0.9751, firm_welfare=-0.1116
- fp_no_reputation: candidate_welfare=0.8665, firm_welfare=-0.1076

These welfare differences flow entirely from the candidate side. The
interpretation that "detection reduces firm costs" is confounded with
"detection changes candidate strategy, which changes the mismatch cost
distribution" — the firm's chosen policy is the same in both cases.

**What to do.** Note explicitly in any writeup that firm welfare changes
are driven by changes in candidate equilibrium, not by any firm strategy
adaptation. The current setup cannot identify "firm response to AI adoption."

---

### W3-FP-R — CI widths in eval_summary are suspiciously tight for fp_fixed_firm's detection_rate_given_ai_high, suggesting a near-empty conditioning event

**What it is.** In `fp_fixed_firm/eval_summary.json:62-66`:
```json
"detection_rate_given_ai_high": {
  "mean": 0.11626,
  "se": 0.01515,
  "ci_lo": 0.08657,
  "ci_hi": 0.14595
}
```
SE = 0.01515, CI width = 0.0594. This is ~54x wider than the corresponding
metric in fp_default (se=0.000306, width=0.001). The reason is that
`fp_fixed_firm` has almost no High-type AI use (ai_adoption_high=0.000580,
`eval_summary.json:22`), so `detection_rate_given_ai_high` is conditioning
on a near-zero event (~58 events per 100 000 samples). The CI is valid but
communicates that this particular cell is effectively unidentified.

Similarly for `fp_null_ai/eval_summary.json:45-49`: detection_rate_given_ai
mean=0.1238, se=0.00879, CI=[0.1066, 0.1411] — again conditioning on ~0.05%
of samples.

**Why it matters.** Any comparison of detection rates across arms that
includes fp_fixed_firm's High-type or fp_null_ai's overall detection rate
is comparing well-identified estimates to near-unidentified ones. Figures
that display these together without flagging the conditioning-event size
are visually misleading.

**What to do.** Add a sample-size column to comparison outputs (n_events
= mean_ai_adoption × num_eval_samples) and suppress or asterisk cells
where n_events < 100.

---

### W4-FP-R — The paper (`paper/main.tex`) contains no FP results; all quantitative claims cite legacy bandit outputs

**What it is.** The current paper draft (`paper/main.tex:11`) sets
`\graphicspath{{../outputs/v2/figures/paper/}}` and all cited numbers
refer to `outputs/v2/`, not `outputs_fp/`. The FP suite has no corresponding
paper section.

**Why it matters.** No FP result is traceable to a paper claim, and
no paper claim is traceable to a FP artifact. If the FP outputs are
intended to replace or supplement the bandit results in the paper, that
link is currently missing.

**What to do.** Decide which experiment suite the paper will cite, and
update the `\graphicspath`, figure references, and numerical claims
accordingly.

---

### W5-FP-R — `fp_fixed_firm` counts_firm sums to 2003 under a fixed-firm regime, inconsistent with the Laplace-1 initialisation for non-selected actions

**What it is.** `outputs_fp/fp_fixed_firm/final_empirical_strategies.json:36-40`:
```json
"counts_firm": [0.0, 2003.0, 0.0]
```
With 2000 FP iterations and Laplace init of 1 per action (3 actions), the
free-running arms yield counts_firm summing to 2003. If the firm is fixed at
BaseVerify, LowVerify and HighVerify should each retain their Laplace init
of 1 — but the output shows 0 and 0. This implies `_run_fixed_firm`
initialises firm counts at 0 (not 1) for the non-selected actions, diverging
from the Laplace convention used in `run_fictitious_play`.

**What to do.** Verify whether `_run_fixed_firm` uses the same initialisation
as `run_fictitious_play`. Document the discrepancy or unify the initialisation.

---

## Minor Findings

### M1-FP-R — run_manifest.json does not record arm-specific config overrides

All six `run_manifest.json` files have identical keys: `arm`, `timestamp`,
`num_fp_iterations`, `num_eval_samples`, `num_eval_seeds`, `seed`,
`fixed_firm`. Arm-specific overrides (e.g., `base_detection_prob=0` for
fp_no_detection) are not recorded. Reproducing any arm requires reading
`experiments/run_fp_suite.py` source in addition to the manifest.

---

### M2-FP-R — No git SHA or software version recorded in any output artifact

No `run_manifest.json`, `eval_summary.json`, or `comparison_table.md`
contains a git SHA, Python version, or NumPy version. Timestamps are present
but insufficient for reproducibility without a version anchor.

---

### M3-FP-R — eval_per_seed.csv does not record the actual integer seed used per row

`eval_per_seed.csv` (all arms) has an `eval_seed_idx` column (0–29) but not
the actual integer seed drawn from the base RNG. Individual rows cannot be
independently reproduced without re-running the full evaluation pass.

---

### M4-FP-R — comparison_table.csv encodes CI as a string column `"[lo, hi]"` rather than two numeric columns

CI columns are formatted as string literals (e.g., `"[0.7181, 0.7192]"`)
rather than separate `ci_lo`/`ci_hi` numeric columns. Any downstream numerical
processing must parse the string. The `eval_summary.json` files have proper
numeric CI fields; the CSV is a format inconsistency.

---

### M5-FP-R — `firm_policy_distribution.png` conveys no cross-arm variation

`firm_policy_distribution.png` will appear as a trivially uniform stacked bar
(LowVerify ≈ 1.0 for all non-fixed arms). This is technically correct but
could mislead readers into thinking the figure represents meaningful variation.

---

### M6-FP-R — `fp_null_ai` detection CIs are ~100x wider than fp_default's; this disparity is hidden in the comparison table

`outputs_fp/fp_null_ai/eval_summary.json:45-48`:
```json
"detection_rate_given_ai": {"mean": 0.1238, "se": 0.00879, "ci_lo": 0.1066, "ci_hi": 0.1411}
```
CI width 0.0345 vs fp_default width ~0.0008 (100x). The comparison table
does not show detection-rate CIs, hiding this disparity.

---

## Missing Artifacts

1. **Training-seed sweep results** — No multi-training-seed output exists
   anywhere in `outputs_fp/`. Primary missing artifact (B1-FP-R).
2. **Per-arm config dump** — A `config_dump.json` (full `dataclasses.asdict`
   of the arm config) is absent from every arm directory.
3. **Paper figures referencing FP outputs** — `paper/main.tex` references
   no FP figure paths; no paper-variant PDFs in `outputs_fp/`.
4. **BR-flip log per arm** — No file records the iteration at which each
   candidate type last changed its best-response action; required to diagnose
   FP cycling in `fp_fixed_firm`.

---

## What the Outputs DO Support (Positive Findings)

1. **r_max correctly equals the maximum of per-type regrets in all six arms.**
   Verified by direct arithmetic for all arms (e.g., fp_default:
   max(0.000121, 0.000942, 0.000580, 0.000833) = 0.000942 = r_max ✓).

2. **All CI bounds correctly bracket their means at 1.96×SE.** Spot-checked
   for fp_default ai_adoption_overall: mean=0.718629, SE=0.000280,
   expected [0.718081, 0.719177]; actual [0.718081, 0.719177] ✓.

3. **counts_candidate row sums are consistent with 2000 iterations + 2-action
   Laplace init (= 2002) for all non-fixed arms.** fp_default type 0:
   1860+142=2002 ✓; type 1: 7+1995=2002 ✓; type 2: 9+1993=2002 ✓.

4. **Cross-arm monotonicity for the four distinct arms is directionally
   correct.**
   - fp_no_detection vs fp_default: AI adoption 0.9995 > 0.7186 ✓;
     correct_match_rate 0.9085 < 0.9422 ✓.
   - fp_no_reputation vs fp_default: Low-type adoption 0.8536 >> 0.0707 ✓.
   - fp_null_ai: correct_match highest among non-fixed arms 0.9683 ✓.
   - fp_fixed_firm: firm_welfare most negative (-0.1230) consistent with
     forced sub-optimal policy ✓.

5. **eval_summary.json numbers match comparison_table.md to displayed
   precision for all arms.** No rounding error detected.

6. **eval_per_seed.csv row counts are 30 for all six arms** (consistent with
   `num_eval_seeds=30` in manifests).

7. **fp_no_detection detection_rate_given_ai = 0.0 exactly** with SE=0,
   CI=[0,0], consistent with arm design ✓.

8. **fp_fixed_firm firm_policy_baseverify = 1.0 exactly** with SE=0,
   CI=[1,1], consistent with `run_manifest.json:8` `"fixed_firm": "BaseVerify"` ✓.

9. **All 19 figures expected by `plot_fp_results.py` are present:** 6 arm ×
   2 per-arm figures = 12 in arm subdirs ✓; 7 cross-arm figures in
   `_compare/` ✓.
