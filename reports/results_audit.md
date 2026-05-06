# Results Audit — LLM-Assisted Hiring Market Simulation

*Performed by the results-auditor role (read-only). No code or artifacts were modified. Scope: outputs only.*

---

## Summary

The on-disk results comprise a single trajectory of 100 epochs from one seed (`seed=42`, `config.py:8`), with no run manifest, no per-seed subfolders, no error bands, and no train/eval split. The scalar summary in `outputs/results/final_summary.json` reproduces the last row of `outputs/results/epoch_metrics.csv` faithfully (all 9 verifiable scalars match), so internal data plumbing is correct. However, the headline metric — that `final_average_global_trust = 0.990` (`final_summary.json:6`) — sits next to a contradicting `final_detection_rate = 0.006` and `final_average_ai_usage = 0.039` (`final_summary.json:5,8`), and the `converged: false / convergence_epoch: null` reported (`final_summary.json:3-4`) is consistent with my recomputation: only 1 of the 3 series the criterion checks (`average_global_trust`) actually settles within tol=0.01 over the last 10 epochs. The most misleading artifact is `outputs/figures/offer_distribution_by_type.png`: per `visualization.py:65-79` it is a *linear interpolation* of per-type mean offers, not the actual offer distribution its title implies. Single-seed, no-CI, no-eval-split is the dominant blocking issue across every claim.

---

## Unsupported or wrong claims (blocking)

### B1. The `offer_distribution_by_type.png` figure is not what its title says
- **Claim**: `visualization.py:92` titles the figure "Offer Distribution by Candidate Type (Last 10 Epochs)" — implying empirical fractions of {reject, low, high} offers per type from `interview_logs.csv`.
- **Artifact**: `outputs/figures/offer_distribution_by_type.png`.
- **Evidence of mismatch**: `visualization.py:65-79` ignores the dead loop body at `visualization.py:64-68` (commented-out per-row aggregation), then computes `avg = mean(average_offer_<type>)` and constructs `r = max(0, 1-avg)`, `h = max(0, avg-1)`, `l = max(0, 1-r-h)`. This is a piecewise-linear deterministic map from a *mean offer* to a 3-bin "fraction" — it cannot represent any distribution where, e.g., medium-types receive a mix of rejects and high offers that average to 1.0 (the figure would show 100% low offer; the truth could be 50/50 reject/high). The actual `interview_logs.csv` has the per-row data needed to compute the real distribution; it is unused.
- **Verdict**: **MISLEADING**. Either rename the figure ("Mean Offer (Linear-Interpolated Composition)") or recompute from `interview_logs.csv`.

### B2. `final_average_global_trust = 0.990` is presented without context that contradicts the natural reading
- **Claim**: `final_summary.json:6` reports `final_average_global_trust: 0.990150000000001`. A reader will infer "the market has restored trust."
- **Artifact**: `outputs/results/epoch_metrics.csv` row for epoch 99 has `average_global_trust=0.990150000000001` — match. But the same row has `average_ai_usage=0.039` and `detection_rate=0.006` (epoch_metrics.csv epoch 99). Trust is high precisely because there is almost no AI use to detect, not because firms learned to discriminate.
- **Methodology context (not duplicated here, see `reports/methodology_audit.md` B6)**: firms have no policy update; `global_trust` is a heuristic that drifts toward 1.0 when `detected_rate` falls. The reported 0.990 is therefore a *passive* result of low AI use, not a signal that the trust mechanism works.
- **Verdict**: **UNSUPPORTED** as a "trust restoration" claim. The number itself is correctly logged.

### B3. No multi-seed → no statistical claim is supportable
- **Inventoried**: `outputs/` contains exactly one set of artifacts (no `seed_*/` or `run_*/` subdirectories — confirmed by `ls outputs/results/` showing only the 3 named files; `ls outputs/figures/` showing only the 6 named figures).
- **Code source**: `simulation.py:20` consumes `cfg.seed` once; `main.py` has no `--num_seeds`.
- **Effect on every numeric claim**: each scalar in `final_summary.json:2-13` is a point estimate from N=1 trajectory of a stochastic process. No claim of the form "AI use declines to X" or "trust stabilises at Y" can be defended without ≥1 additional seed.
- **Verdict**: **UNSUPPORTED** for any claim that compares, ranks, or asserts a level "of the market." Only "this run produced X" is technically defensible.

### B4. Reported numbers are *training* metrics, not held-out evaluation
- **Code source**: `main.py:35-46` calls `run_simulation(cfg)` and saves the returned aggregates directly. No second pass with frozen Q-values, no validation epoch, no test set. The `final_*` fields are computed from the last training epoch.
- **Effect**: `final_average_ai_usage=0.039` (`final_summary.json:5`) reflects what happens *while* the bandit is still exploring with `epsilon ≈ 0.061` (verified: last-epoch epsilon in `epoch_metrics.csv` row 99 = 0.06057704364907282). A held-out epoch with `epsilon=0` and frozen Q would give a different (and more interpretable) number.
- **Verdict**: **UNSUPPORTED** as an "equilibrium" or "final policy" claim; it is an in-sample dynamics readout.

---

## Weak claims (important)

### W1. `converged: false` is correct for this run, but the convergence machinery is fragile
- **Claim**: `final_summary.json:3-4` reports `converged: false, convergence_epoch: null`.
- **Verification of the criterion** (`metrics.py:110-125`, tol=0.01, window=10):
  - `average_ai_usage` over epochs 90-99: min=0.028, max=0.082, range=**0.054** → fails tol.
  - `average_global_trust` over epochs 90-99: min=0.9820, max=0.9902, range=**0.0081** → passes tol.
  - `average_offer` over epochs 90-99: min=1.051, max=1.073, range=**0.022** → fails tol.
- The `false` flag is data-faithful. But the criterion's brittleness (per `methodology_audit.md` I4) means a future run with slightly less noise could trigger `convergence_epoch = last_epoch` (per `metrics.py:131-134`, see also methodology I3) — which is uninterpretable.
- **Verdict**: **WEAK** but not currently wrong.

### W2. Per-type detection-rate ordering does not match `cfg.detection_prob_by_type` in late epochs
- **Claim implicit in figure** `outputs/figures/detection_rate_over_time.png` and in the design (`config.py:98-102`): `low > medium > high` for detection rate.
- **Artifact**: at epoch 99, `detection_rate_low=0.0034, detection_rate_medium=0.0088, detection_rate_high=0.0000` (`epoch_metrics.csv` row 99) — ordering **violated** (medium > low). 4 of the last 10 epochs violate the ordering. Epochs 0 and 50 satisfy it.
- **Why**: detection rate per epoch is conditional on AI use, which collapses to ≈3-5% in late epochs; per-type AI-user counts shrink to single digits per epoch and the empirical detection rate becomes Monte-Carlo noise. The *parameter* `detection_prob_by_type` is correct; the *aggregate metric* averaged over all interviews (including non-AI) is dominated by who happened to use AI.
- **Verdict**: **WEAK**. The plot will mislead a reader into thinking the model violates its own assumptions. Either condition the plotted rate on `ai_action=1`, or annotate.

### W3. `final_average_candidate_loss = -1.057` reported without explanation that negative loss = candidate gain
- **Claim**: `final_summary.json:12` reports `-1.05726`.
- **Source**: Last row of `epoch_metrics.csv` `average_candidate_loss = -1.05726` — match.
- **Issue**: per `methodology_audit.md` M9, loss is negative because the no-detection trust recovery (+0.01 per interview) is signed-flipped into a candidate "gain" inside `reputation_loss`. A reader sees "-1.057" and may interpret it as the candidate being penalised by 1.057 units. It is the opposite. The summary field name should be `average_candidate_utility` (or sign-flipped) for honesty.
- **Verdict**: **WEAK** presentation honesty, accurate logging.

### W4. `num_epochs_run = 100` is exactly the configured cap — no information about adequacy
- **Claim**: `final_summary.json:2` reports `num_epochs_run: 100`.
- **Source**: `cfg.num_epochs = 100` (`config.py:17`); `cfg.early_stop = False` (`config.py:82`); the loop in `simulation.py:30` runs the full 100. The "100" is a configuration restatement, not evidence the system has equilibrated.
- **Combined with W1**: the criterion fails for 2 of 3 series, so 100 epochs is *not* enough.
- **Verdict**: **WEAK**. Either run longer or report the trajectory-tail variance alongside.

### W5. All figures are single-trajectory; no error bands
- **Source**: `visualization.py:22-141` plots each metric as a single `plt.plot(...)` line; no `fill_between`, no `errorbar`, no shaded CI. With one seed, this is the only option.
- **Affected figures**: all 6 in `outputs/figures/`. A reader interpreting any wiggle (e.g. the dip in `ai_adoption_over_time.png` near epoch 40 or the bumps in `losses_over_time.png`) is reading sampling noise as signal.
- **Verdict**: **WEAK** universally — fixable only by rerunning with seeds.

### W6. `final_correct_match_rate = 0.734` is a misleading "headline"
- **Claim**: `main.py:41` prints "Final correct match rate: 0.734".
- **Source**: `epoch_metrics.csv` row 99 `correct_match_rate = 0.734` — match.
- **Issue**: 0.734 is *not* a model-skill metric — it is dominated by the 50% medium-type prior (`config.py:94`) which receives the low offer when `effective_trust * observed_signal ≈ 1.0`, plus low-types getting rejected once they stop using AI. A trivial baseline ("offer 'low' to everyone") would score `P(true_type=1) = 0.50` correct; "offer 'low' to everyone except observed signal=0 → reject" likely scores in the 0.7-0.8 range without any learning. There is no baseline in `outputs/` to compare against.
- **Verdict**: **WEAK**. Number is correct; framing as "market efficiency" is unsupported absent a baseline arm (per methodology I1).

---

## Presentation issues (minor)

### P1. `final_summary.json` reports floats with full IEEE precision
- e.g. `0.990150000000001` (`final_summary.json:6`), `-1.05726` (`final_summary.json:12`). Cosmetic — round to 4 sig figs in the summary, keep raw precision in the CSV.

### P2. Epsilon never reaches `min_epsilon`
- `epoch_metrics.csv` row 99 `epsilon = 0.06057704364907282`; `cfg.min_epsilon = 0.02` (`config.py:54`). Confirmed monotone non-increasing across all 100 epochs. Worth stating in any "the policy converged" prose.

### P3. `convergence_epoch` is `null` in this run, so the off-by-window bug noted in `methodology_audit.md` I3 has not yet manifested in artifacts. Worth fixing before any run trips it.

### P4. Figure `firm_trust_over_time.png` plots `average_individual_trust` (mean over the full 100×10 trust matrix) and `average_global_trust` (mean over 10 firms) on the same y-axis. Both are in [0,1] but their semantics differ; the figure does not annotate this.

### P5. `interview_logs.csv` (~7 MB, 100 000 rows) is the largest artifact and contains the data needed to fix B1 (true offer distribution), W2 (AI-conditioned detection rate), and to compute per-seed CIs once seeds exist. It is currently used only to write itself out — no figure or summary derives from it.

---

## Missing artifacts the user should provide

1. **Run manifest** — `utils.py:13-77` writes only `epoch_metrics.csv`, `interview_logs.csv`, `final_summary.json`. Nothing records: seed used (recoverable from `config.py` only because no override), git SHA at run time, full config snapshot, command line, wall-clock timestamp, hostname, Python/numpy versions. Without these the run is not reproducible by a third party. Recommend `outputs/results/run_manifest.json` containing all of the above.
2. **Multi-seed runs** — at minimum 5, ideally ≥30, with per-seed subdirectories (e.g. `outputs/seed_42/results/...`) and an aggregator that emits mean ± SE for every field in `final_summary.json`.
3. **Held-out evaluation epoch(s)** — after training, freeze Q-tables, set `epsilon=0`, run K eval epochs, and emit `final_summary_eval.json` separately from training-tail metrics.
4. **Baseline arms** — at least one of: `initial_ai_rate=0`, `detection_prob_by_type` all-zero, fixed-action candidates. Per methodology I1, no causal claim is supportable without these.
5. **True per-type offer distribution figure** — derived from `interview_logs.csv` (counts of `offer ∈ {0,1,2}` grouped by `true_type`), to replace the linearly-interpolated `offer_distribution_by_type.png`.
6. **AI-conditioned detection rate** — figure or column showing detection rate among interviews where `ai_action=1`, broken out by type. Without this conditioning, W2's apparent ordering violation will keep recurring.

---

## Number-by-number sanity check on `final_summary.json`

| Field | Value | Source (epoch 99 of `epoch_metrics.csv`) | Match? |
|---|---|---|---|
| `num_epochs_run` | 100 | (loop length, `cfg.num_epochs=100`) | OK — restatement of config, not measurement |
| `converged` | false | recomputed: 2 of 3 series exceed tol=0.01 over last 10 | OK |
| `convergence_epoch` | null | consistent with `converged=false` | OK |
| `final_average_ai_usage` | 0.039 | `average_ai_usage` = 0.039 | exact match |
| `final_average_global_trust` | 0.990150000000001 | `average_global_trust` = 0.990150000000001 | exact match |
| `final_average_offer` | 1.058 | `average_offer` = 1.058 | exact match |
| `final_detection_rate` | 0.006 | `detection_rate` = 0.006 | exact match |
| `final_correct_match_rate` | 0.734 | `correct_match_rate` = 0.734 | exact match |
| `final_overoffer_rate` | 0.237 | `overoffer_rate` = 0.237 | exact match |
| `final_underoffer_rate` | 0.029 | `underoffer_rate` = 0.029 | exact match |
| `final_average_candidate_loss` | -1.05726 | `average_candidate_loss` = -1.05726 | exact match (sign meaning: see W3) |
| `final_average_company_loss` | 0.307 | `average_company_loss` = 0.307 | exact match |

Internal-consistency spot checks (`epoch_metrics.csv`):
- Epoch 0: `reject+low+high = 0.229+0.457+0.314 = 1.000`; `correct+over+under = 0.765+0.235+0.000 = 1.000`. OK.
- Epoch 50: `reject+low+high = 0.034+0.877+0.089 = 1.000`; `correct+over+under = 0.681+0.262+0.057 = 1.000`. OK.
- Epoch 99: `reject+low+high = 0.058+0.826+0.116 = 1.000`; `correct+over+under = 0.734+0.237+0.029 = 1.000`. OK.
- All `average_ai_usage_{low,medium,high}` values in sampled epochs lie in [0,1]. OK.
- `interview_logs.csv` row count: 100 000 = 100 epochs × 1 000 interviews/epoch (100 candidates × 10 rounds). OK — matches design.
- `epsilon` strictly monotone non-increasing across 100 epochs (verified).
- `detection_rate_low ≥ medium ≥ high` ordering: holds at epochs 0 and 50; violated 4 of 10 times in the last 10 epochs because absolute counts are tiny (see W2).
