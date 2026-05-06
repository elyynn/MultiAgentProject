# Minimal Experiment Plan — LLM-Assisted Hiring Market Simulation

*Anchored to `reports/methodology_audit.md` and `reports/results_audit.md`. Goal: the smallest set of additional runs that turns "an interesting pilot" into "a defensible paper." Plan only — nothing is run.*

---

## Goal & scope

**Goal**: clear the four audit findings that block ANY publishable claim:
- Methodology B1 / Results B3 — single seed.
- Results B4 — training-tail metrics, no held-out evaluation.
- Methodology I1 — no baselines or ablations isolating which mechanism drives the dynamics.
- Methodology I7 — hyperparameters undocumented and untested.

**Out of scope** (deliberately deferred — these are *code* changes, not new experiments):
- Methodology B2 (omniscient candidate via spillover-in-reward leak): requires changing `losses.py`. Defer; flag in paper.
- Methodology B3 (bandit-vs-Q-learning naming): paper-text fix.
- Methodology B4 (high-type AI ceiling): config-only fix possible but changes the model; out of scope here.
- Methodology B5 (loss double-count): requires changing `losses.py`. Defer; flag in paper.
- Methodology B6 (firms don't learn): a research-direction change, not an experiment.
- Results B1 (`offer_distribution_by_type.png` is a linear interpolation): figure-curator job, separate from runs.

The minimal suite below targets the *experimental* gaps; the *modeling* gaps are explicit limitations to acknowledge.

---

## Required infrastructure (pre-run, ~half a day of work)

These do not exist today (per `reports/project_map.md`) and must be built before the suite runs:

| Item | What | Where it goes | Owner |
|---|---|---|---|
| **Sweep wrapper** | A small script that loops over (arm, seed), mutates a fresh `SimulationConfig`, calls `run_simulation`, dumps results to a per-(arm, seed) subfolder. ~50 LOC. Avoids editing `config.py` between runs. | `experiments/run_suite.py` (new) | experiment-runner |
| **Per-run manifest** | Save `run_manifest.json` next to each run's CSVs: seed, full config dict, git SHA (`git rev-parse HEAD`), command line, start/end timestamp, hostname, `numpy.__version__`. Addresses Results "Missing artifact" #1. | extend `utils.py:save_results` (or wrap in `run_suite.py`) | experiment-runner |
| **Held-out eval pass** | After training, with frozen Q-tables and `epsilon=0`, run K=10 additional epochs, log them separately, and emit `final_summary_eval.json`. Addresses Results B4. ~20 LOC change in `simulation.py` + `metrics.py`. | `simulation.py`, `metrics.py` | small code change |
| **Aggregator** | Reads all per-seed `final_summary*.json` for one arm, emits `aggregate_summary.json` with mean, SE, 95% CI for every scalar; emits per-epoch mean ± SE arrays for figures. | `experiments/aggregate.py` (new) | experiment-runner |
| **Output convention** | `outputs/v2/<arm>/seed_<n>/results/...` per run; `outputs/v2/<arm>/aggregate/` per arm. Leave existing `outputs/` (the pilot) untouched. | n/a — convention | experiment-runner |

Without these, the runs below are not aggregatable and the paper still cannot quote "mean ± SE."

---

## The minimum suite — 30 runs, 5 arms

| # | Arm name | What changes vs. default | Seeds | Runs | Addresses | What it answers |
|---|---|---|---|---|---|---|
| 1 | `default` | nothing (current `get_default_config()`) | 10 | 10 | M-B1, R-B3, R-W5 | Statistical sufficiency for the headline numbers and figures: every scalar gets mean ± SE, every figure gets a CI band. |
| 2 | `no_detection` | `detection_prob_by_type = {0: 0, 1: 0, 2: 0}` | 5 | 5 | M-I1 | Counterfactual: what if firms cannot detect? Isolates detection as the deterrent mechanism. AI use should rise to ~100% if learning works. |
| 3 | `no_spillover` | `num_spillover_companies = 0` | 5 | 5 | M-I1, M-I5, R-B2 | Counterfactual: removes the multi-firm reputation channel. Isolates whether spillover (vs. direct trust drop alone) is necessary for the observed dynamics. |
| 4 | `null_ai` | `ai_signal_boost = 0` (AI does nothing observable) | 5 | 5 | M-I1, R-W6 | Null arm: any candidate-side dynamics here are pure exploration noise. Establishes the floor against which `default` results must rise. Also gives a baseline `correct_match_rate` for R-W6. |
| 5 | `sens_detect_lo` | `detection_prob_by_type` × 0.5 (i.e. `{0.175, 0.10, 0.05}`) | 5 | 5 | M-I7 | Sensitivity: half-strength detection. Combined with arm 6, brackets the default. |
| 6 | `sens_detect_hi` | `detection_prob_by_type` × 2 (i.e. `{0.70, 0.40, 0.20}`) | 5 | 5 | M-I7 | Sensitivity: double-strength detection. Combined with arm 5, shows whether the qualitative story survives a 4× swing in the most consequential parameter. |
| | **Total** | | | **35** | | |

Correction: the table sums to 10+5+5+5+5+5 = 35 runs. The plan name says "30" — I am committing to 35. The extra 5 (the second sensitivity arm) is the cheapest insurance against M-I7 being raised in review.

**Seed list**: `[42, 123, 2024, 7, 1337, 99, 314, 271, 8675309, 555]`. Same first five used across all arms; arm 1 gets all ten. Sharing seeds across arms enables paired-difference analysis (lower variance) when comparing arm i vs. default.

---

## Per-run protocol

Every run, regardless of arm:

1. Fresh `SimulationConfig` from `get_default_config()`, mutate the arm-specific fields.
2. Train: 100 epochs as today.
3. **Eval pass** (new): freeze candidate Q-values, set `epsilon=0`, run 10 additional epochs. Emit `final_summary_eval.json` alongside `final_summary.json`. This is the number that goes in the paper, not the training-tail.
4. Save: `epoch_metrics.csv`, `interview_logs.csv`, `final_summary.json` (training tail), `final_summary_eval.json` (held-out), `run_manifest.json`.
5. Emit no figures per run — figures are produced once per arm by the aggregator.

---

## Aggregation deliverables (post-run)

For each arm, the aggregator produces:

- `outputs/v2/<arm>/aggregate/summary.json` — mean, SE, 95% CI, and N for every field in `final_summary_eval.json`.
- `outputs/v2/<arm>/aggregate/epoch_curves.npz` — per-epoch mean ± SE arrays for every column in `epoch_metrics.csv`. Used by the figure-curator to redraw all 6 figures with shaded CI bands (addresses R-W5).
- `outputs/v2/<arm>/aggregate/per_seed_summary.csv` — one row per seed, all final-eval scalars, for diagnosing seed-driven outliers and cherry-picking concerns.

Cross-arm deliverables (one set, in `outputs/v2/_compare/`):
- `comparison_table.md` — for the headline scalars (`final_average_ai_usage`, `final_average_global_trust`, `final_correct_match_rate`, `final_overoffer_rate`, `final_average_company_loss`), one row per arm with mean ± SE, and a paired-difference vs. `default` (mean Δ ± SE, p-value via paired bootstrap on shared seeds).
- `comparison_figures/` — for the 2-3 most informative metrics, multi-arm overlay figures with CI bands.

These artifacts are the inputs the paper-writer and figure-curator will work from.

---

## Anti-cherry-picking discipline

- Pre-register the seed list above and the eval protocol *before* running. No swapping seeds after looking at results.
- Keep every per-seed run on disk, even bad ones. The aggregator must report `N=5/5 succeeded`, not silently drop crashes.
- Report training-tail AND eval numbers side-by-side; if they disagree by >2 SE, that itself is a finding to discuss.
- Compute paired-difference statistics (arm minus default on shared seeds) rather than independent two-sample tests — paired comparison is the honest analysis given the shared-seed design.

---

## What this suite does NOT cover (explicit limitations to state in the paper)

- **Modeling-level confounds remain**: the candidate's reward still uses unobservable spillover trust (M-B2); company loss still uses ground-truth `ai_action` (M-B5); firms still don't learn (M-B6). Acknowledge these as scope limits, not as quantitative claims that this suite refutes.
- **High-type AI is still strictly dominated** (M-B4). The "high-types stop using AI" finding, if reported, is a property of the signal model, not an emergent result. The suite doesn't fix this — it only shows the dynamics are robust to other-parameter variation.
- **Hyperparameter sensitivity is one-knob, two-level only.** A second iteration should sweep `direct_trust_penalty`, `spillover_trust_penalty`, and `num_spillover_companies`. Out of scope here for budget reasons.
- **No firm-side ablation** (e.g. varying offer thresholds): firms don't learn anyway, so this is low-value until M-B6 is addressed.
- **No graph-structured spillover** (M-I5): the paper must call the mechanism "stochastic reputation diffusion," not "reputation network."

---

## Estimated cost

- **Wall-clock per run**: unmeasured. The pilot run produced ~7 MB of `interview_logs.csv` and ran to completion at some point during May 5; measure once on first invocation. Pure-Python on a single core; assume O(minutes) per run as a placeholder.
- **Total runs**: 35.
- **Parallelism**: trivially seed-parallel — run all 35 as independent processes if cores allow. With 4 cores, expect ≤10× single-run wall-clock for the whole suite.
- **Storage**: ~7 MB × 35 = ~250 MB of interview logs. Add ~200 KB of summaries. Well under any reasonable disk budget.
- **Dev time before runs**: ~half a day for the wrapper + manifest + eval pass + aggregator (per "Required infrastructure" above).

---

## Success criteria for the suite

After running and aggregating, the paper can credibly claim **only what survives all four checks**:

1. The effect appears in `default` mean across 10 seeds with SE ≤ 1/3 of the effect size.
2. The effect collapses (or qualitatively changes) in `no_detection` and/or `no_spillover` — i.e. the mechanism is *causally* implicated, not just correlated.
3. The effect persists in held-out eval (epsilon=0, frozen Q), not just training tail.
4. The effect persists across `sens_detect_lo` and `sens_detect_hi` — i.e. is not knife-edge in the most consequential hyperparameter.

Anything that fails any of (1)-(4) is downgraded to "exploratory observation" in the paper, not a claim.
