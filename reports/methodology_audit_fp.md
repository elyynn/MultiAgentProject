# Methodology Audit — Fictitious-Play Hiring Game (FP version)

*Read-only audit. Sister document to `reports/methodology_audit.md` (bandit version). Scope: `config_fp.py`, `fictitious_play.py`, `payoffs.py`, `evaluate_fp.py`, `experiments/run_fp_suite.py`, `experiments/run_fp_baseline.py`, `experiments/aggregate_fp.py`, `experiments/plot_fp_results.py`, and the `outputs_fp/` artifacts. Per the user's clarification, the legacy bandit files (`agents.py`, `learning.py`, `environment.py`, `losses.py`, `simulation.py`, `paper/main.tex`) are not in scope here — the FP suite is a self-contained reimplementation.*

---

## Executive summary

The FP suite is a clean, faithful, vectorised implementation of two-population fictitious play on a Bayesian hiring signaling game with three types, two candidate actions, and three firm verification levels. As a piece of code it is much tighter than the bandit version: there is a real expected-utility computation, an empirical-frequency belief, a best-response operator, a regret diagnostic, and a held-out evaluation pass. However, the methodology has three blocking problems that materially weaken any claim derived from `outputs_fp/`: (1) results come from a single FP training seed with `seed=42` — what looks like 30-seed CIs are *evaluation-sample* CIs on one frozen strategy, not training-seed CIs; (2) `fp_high_verification_cost` produces byte-identical strategies to `fp_default`, demonstrating that the headline ablation has no effect on the equilibrium and almost certainly no real ablation pressure exists in three of the six arms; (3) the FP iteration mixes a *Bayes-optimal* firm offer rule (computed against `sigma_c`) with an *empirical-frequency* firm verification policy, so the "firm" the candidate is fictitiously playing against is a hybrid object whose convergence guarantees do not follow from any standard FP theorem. There are also several important issues around the dropping of the per-firm trust matrix and spillover network (the entire reputation mechanism is now a constant), the meaning of "1 iteration" in this fully-mixed-population game, and the absence of any tie-breaking sensitivity check despite the candidate BR being knife-edge for type Low.

---

## Blocking findings

### B1-FP. Single training seed; reported CIs are evaluation-only

- `config_fp.py:7` sets `seed: int = 42`. `fictitious_play.py:109` consumes it once: `rng = np.random.default_rng(cfg.seed)`. The training-seed pipeline in `run_fictitious_play` is invoked exactly once per arm in `experiments/run_fp_suite.py:84` and `experiments/run_fp_baseline.py:42`.
- `evaluate_fp.py:170-178` then loops over `cfg.num_eval_seeds = 30` (`config_fp.py:91`) — but each of those seeds only reseeds the *Monte-Carlo evaluator* on the *frozen* `(counts_candidate, counts_firm)` produced by the single training run. The "ci_lo/ci_hi" reported in `outputs_fp/fp_default/eval_summary.json` and surfaced in `outputs_fp/_compare/comparison_table.md` are therefore CIs around the mean of 30 i.i.d. samples from the same distribution — they shrink to zero as `num_eval_samples` grows and tell you nothing about how robust the *equilibrium itself* is to FP randomness.
- Since the FP iteration draws fresh CRN payoff samples each iteration (`fictitious_play.py:122`, `payoffs.py:163-167,201-211`), the converged empirical strategy *is* a function of the seed; this is unaudited variability.
- **What to do**: Add a top-level loop over `num_training_seeds` (e.g. 20-30) wrapping `run_fictitious_play` and aggregate `(counts_candidate, counts_firm)` across them. The held-out evaluator can then either average-of-averages or, better, evaluate per training seed and report mean ± SE across training seeds. Until then, every effect size in `comparison_table.md` is reported with misleadingly tight uncertainty.

### B2-FP. `fp_high_verification_cost` is a null ablation: identical strategies to baseline

- `experiments/run_fp_suite.py:65-67` defines the high-cost arm as doubled `verification_cost` for all three policies. Because verification cost only enters the firm utility (`payoffs.py:218`: `-mismatch.mean() - cfg.verification_cost[firm_policy]`), and because `LowVerify` already strictly dominates in the baseline equilibrium (`outputs_fp/fp_default/final_empirical_strategies.json:16-20` shows firm puts mass 0.999 on `LowVerify`), doubling cost simply increases the *gap* without changing the ordering. The output confirms: `outputs_fp/fp_high_verification_cost/final_empirical_strategies.json` has identical `counts_candidate` (1860/142, 7/1995, 9/1993) and `counts_firm` (2001/1/1) as `outputs_fp/fp_default/final_empirical_strategies.json`.
- `comparison_table.md:5,10` — the only number that differs between `fp_default` and `fp_high_verification_cost` is `firm_welfare` (-0.0780 vs -0.0981), which is a deterministic function of the cost increment and does not reflect any equilibrium response.
- This is blocking because the arm is presented in the comparison table as a sensitivity probe, but it is not informative about anything — it is structurally guaranteed to produce the same equilibrium until the cost is large enough to flip the firm's verification ordering.
- **What to do**: replace with a cost level that bridges the gap to `BaseVerify` or `HighVerify` becoming preferred (or scan a range), or pair it with an arm where `c_LowVerify` is *increased* relative to the others. The way it is written, this arm is a no-op.

### B3-FP. The "firm" the candidate fictitiously plays is a non-standard hybrid

- In `fictitious_play.py:127-138`, the candidate's expected utility against firm action `m` is `candidate_expected_utility(theta, action, m, sigma_c, cfg, ...)`, weighted by the empirical firm frequency `sigma_f[j]`. Inside `payoffs.py:149-179`, the *offer* is computed by `_bayes_optimal_offer_batch(... , sigma_c_arr, ...)` — i.e. the firm immediately Bayes-updates on the candidate empirical action distribution and offers optimally given posterior over types.
- In other words, the firm has *two* layers of decision: a verification policy over `{LowVerify, BaseVerify, HighVerify}` chosen by FP empirical best response, and an *offer* in `{0,1,2}` chosen by Bayes-optimality against `sigma_c`. The first is FP; the second is fully rational and instantaneous.
- This is not standard fictitious play — in textbook FP both players' strategy spaces are fixed, and beliefs over opponents are formed from observed *plays*. Here the "opponent" the candidate sees is partly a moving frequency (verification choice) and partly a function of the candidate's own current belief about itself (the offer rule). This invalidates appeals to FP convergence theorems for two-player zero-sum / potential / 2x2 games — the game being played per iteration is not constant, it depends on `sigma_c` through the offer mapping, so the empirical play of the firm is not a sufficient statistic for what the candidate faces.
- The model is internally consistent and arguably reasonable as "Bayesian Nash dynamics with FP-on-verification", but it should not be sold as canonical fictitious play. The paper-side claim "fits the LEMAS theme as canonical FP" is undermined.
- **What to do**: Either (a) flatten the firm action space to include the offer (so the firm has e.g. `{m} × {threshold-rule parameter}` or just `{m}` × full Bayes-offer at fixed prior), and run pure FP on the resulting joint, or (b) keep the current setup but rename the algorithm "best-response dynamics with Bayes-optimal offers" or similar, and acknowledge that standard FP convergence results do not apply.

---

## Important findings

### I1-FP. The reputation mechanism (per-pair trust matrix, spillover, global trust) has been deleted, not ported

- The bandit code had `trust_matrix` (`environment.py:59-64`), `direct_trust_penalty`, `spillover_trust_penalty`, `num_spillover_companies`, and `update_trust_after_interview` (`environment.py:41-79`) — the entire "reputation network" mechanism that the previous paper draft (`paper/main.tex:34-689`) builds its narrative on.
- The FP version collapses this to a single scalar damage: `config_fp.reputation_damage()` (`config_fp.py:96-100`) returns `direct + expected_spillover_count * spillover`, applied uniformly per detection event in `payoffs.py:163,177`. There is no firm-specific trust state, no across-time accumulation, no graph, no spillover targeting. Trust is just a constant cost coefficient.
- The `fp_no_reputation` arm (`run_fp_suite.py:49-51`) sets `reputation_penalty_weight = 0`, so the comparison is "constant penalty present vs absent" — not "reputation network present vs absent." This is fine *as a sensitivity test on the cost magnitude*, but it cannot support any claim about reputation-as-a-network or about spillover dynamics.
- The bandit-era audit's I5 (uniform-random spillover is not a real network) is now formally moot in the FP code — but if the paper still wants to make spillover claims, the FP results cannot ground them.
- **What to do**: be explicit in any FP-version write-up that "reputation" here is a static cost, not a stateful network mechanism. If a network claim is wanted, FP needs an extension (e.g. firm-specific verification levels with cross-firm belief sharing).

### I2-FP. Tie-breaking is asymmetric in a way that affects the headline result

- `fictitious_play.py:33-41` (`_conservative_argmax`) breaks ties for the candidate in favour of `prefer_lower=True`, i.e. action 0 (No-AI). Called from line 137 with `prefer_lower=True`.
- `fictitious_play.py:44-52` (`_cost_conservative_argmax`) breaks firm ties in favour of lowest verification cost. Called from line 146.
- For the Low type at the converged firm strategy (`sigma_f` ≈ 1.0 on LowVerify), the candidate is approximately indifferent: `outputs_fp/fp_default/final_empirical_strategies.json:22-25` shows the converged Low-type mix is 0.929 / 0.071 No-AI / AI, but `outputs_fp/fp_default/fp_regret.json:5` shows `r_candidate_low = 9.4e-4`. That is the *largest* component of `r_max` and very close to numerical zero.
- Combined with `tie_tol = 1e-8` (`config_fp.py:13`), this means the Low-type asymptote is sensitive to: (i) the Monte-Carlo noise floor of `num_payoff_samples = 2000` (`config_fp.py:86`), which has SE on the order of `1/sqrt(2000) ≈ 0.022` — *four orders of magnitude larger than `tie_tol`* — and (ii) the choice of `prefer_lower=True`.
- A symmetric tie-break, or a tie tolerance tuned to MC-SE, could plausibly flip Low-type AI adoption between ~0.07 and ~0.93. None of the six arms tests this.
- **What to do**: scale `tie_tol` to MC-SE, or run with `prefer_lower=False` as a sanity arm, or replace deterministic FP with smooth FP (logit best-response) so that tie zones are smoothed over.

### I3-FP. `_make_crn_rng(seed=t)` resets the payoff RNG every iteration to a deterministic function of the iteration count

- `fictitious_play.py:122`: `crn_rng = _make_crn_rng(seed=t)` when `cfg.common_random_numbers=True` (the default). This means every FP iteration uses the same MC samples for *that t* across arms, regardless of arm seed — but more importantly, the same iteration-`t` RNG is used for *all six candidate-action utility evaluations* and the *three firm-action evaluations* at iteration `t`. They are not independent draws — they share the same `np.random.default_rng(t)` sequence.
- This is not necessarily wrong (CRN reduces variance of relative comparisons, which is the whole point), but `payoffs.py:165,166` and `:209,211` call `rng.normal(...)` and `rng.uniform(...)` *consecutively* on the same RNG inside each utility call. That means the MC samples used for `candidate_expected_utility(theta=0, a=0, m=LowVerify, ...)` and `candidate_expected_utility(theta=0, a=1, m=LowVerify, ...)` are *different non-overlapping draws from the same stream* — not the same draws. So CRN is happening *between iterations* but not *across actions within an iteration*, which somewhat defeats its purpose for choosing the argmax.
- **What to do**: use antithetic / paired sampling within an iteration (same s,d shocks across actions), or document that "CRN" here means "iteration-level seeding for cross-arm reproducibility" rather than "variance reduction across compared actions."

### I4-FP. The candidate's BR conditions on `sigma_c` (its own belief), not just on `sigma_f`

- `fictitious_play.py:129-133`: `candidate_expected_utility(theta, action, m, sigma_c, ...)`. Inside `payoffs.py:171-172`, the firm's offer is `_bayes_optimal_offer_batch(s_samples, d_samples, firm_policy, sigma_c_arr, ...)` — so the candidate's expected payoff at iteration `t` depends on `sigma_c^t` (the candidate population's *own* current empirical strategy).
- Operationally this is correct: at iteration `t`, all candidates of all types share the same belief about the firm's offer rule, which is computed from the population's empirical strategy. But it means a single Low-type's "best response" assumes the rest of the population is going to play `sigma_c^t` — and once *all* Lows flip to `a=1`, the assumed `sigma_c` is wrong.
- This is the standard population-FP reading (everyone treats the empirical distribution as if it were a fixed mixed strategy). It is the right thing to do under FP, but it is *not* what a single-agent best-responder would compute. The write-up should clarify that the candidate is a representative of a continuum of types, not an individual decision-maker. Otherwise readers will conflate "AI adoption rate of 0.07 for Lows" with "each Low uses AI 7% of the time" rather than "7% of Lows use AI deterministically."

### I5-FP. The "epoch" / convergence concept is mismatched to the algorithm

- `fictitious_play.py:117` runs `cfg.num_fp_iterations = 2000` iterations. Each iteration adds a +1 count to the BR cell (deterministic FP). With smoothed initial counts of 1 (`fictitious_play.py:112-113`), after 2000 iterations the empirical share of an action played at every iteration is ~2001/2003 ≈ 0.999. The exact final shares of 0.929 / 0.997 / 0.996 are then reflective of the iteration *at which* each type's BR last flipped.
- `check_convergence` (`fictitious_play.py:177-201`) requires `max_delta < 0.002` over the last 200 iterations. Since each step changes `sigma_c[i,j]` by at most `1 / (count + 1)`, by iteration 1800 step size is bounded by ~5e-4. So the convergence check is trivially satisfied as long as no BR flip happens in the last 200 iterations — it does *not* test that the BRs have stopped flipping (which is the actual fixed-point condition for FP).
- **What to do**: report (i) iteration of last BR flip per row, (ii) `r_max` (already done) as the actual stationarity diagnostic, and (iii) drop or rename `is_converged` so it doesn't suggest equilibrium has been verified. Currently `comparison_table.md:5-10` reports `is_converged=True` for *every* arm including `fp_fixed_firm` whose `r_max=0.0595` (60x the others) — this is contradictory.

### I6-FP. `fp_fixed_firm` has substantially higher candidate regret than baseline but is still labelled "converged"

- `outputs_fp/fp_fixed_firm/fp_regret.json`: would show `r_max ≈ 0.0595` per `comparison_table.md:9`. This is well above the implicit "equilibrium" threshold (0.03 line drawn in `experiments/plot_fp_results.py:193`).
- The arm fixes firm at `BaseVerify` (`run_fp_suite.py:60-62`, executed in `_run_fixed_firm` `:124-175`). Even with no firm adaptation the candidate side should converge to a stable BR — yet `r_max=0.0595` says some candidate type is leaving 6% of utility on the table.
- Inspection of `outputs_fp/fp_fixed_firm/final_empirical_strategies.json:21-33`: counts are 1943/59, 1942/60, 2001/1 — the Low and Medium types have ~3% AI use, suggesting they oscillated between BRs (the +1 count at each iteration cannot accumulate to 3% in 2000 iterations from a single late flip; it implies ~60 BR flips). This is the FP cycle phenomenon and should be flagged: in this game class, FP is *not* guaranteed to converge, and arm `fp_fixed_firm` is empirical evidence of cycling.
- **What to do**: report BR-flip count per type per arm; explicitly diagnose cycles; consider averaging over a window of late iterations rather than reporting last-iteration empirical mixture as "the equilibrium."

### I7-FP. `prefer_lower=True` plus warm prior of 1 each interacts unfavourably with the Low-type knife-edge

- `fictitious_play.py:112` initialises `counts_candidate = np.ones(...)` — Laplace smoothing of strength 1. Combined with `prefer_lower=True` tie-breaking and the fact that early iterations have very noisy MC estimates of expected utility, the first few BRs disproportionately drive subsequent iteration counts (an "FP path-dependence"). For Low types where utilities are within MC-noise, the first ~50 iterations effectively decide the equilibrium.
- This is observable in the trajectories at `outputs_fp/fp_default/figures/candidate_trajectory.png` (not read here but referenced); a sweep over `seed` would expose this. Currently every arm uses `seed=42` so the path-dependence is invisible.
- **What to do**: report variance over training seeds (per B1-FP) and consider increasing `num_payoff_samples` until MC-SE < typical utility gap.

### I8-FP. Re-check of bandit-version blocking findings under FP

| Bandit B# | Description | Status under FP |
|---|---|---|
| B1 | Single seed | **Repeats as B1-FP**: still single training seed; eval CIs are misleading. |
| B2 | Reward uses unobservable global trust info (omniscient candidate) | **Resolved structurally** — there is no per-firm trust matrix in FP. The cost is a constant, so no information leakage. |
| B3 | "Q-learning" was actually a stateless bandit | **Resolved by replacement** — FP is a clearly named, well-defined learning rule. The implementation is recognisably FP (with caveats in B3-FP). |
| B4 | High-type AI signal ceiling makes AI strictly dominated for Highs | **Partially resolved**. FP uses continuous Gaussian signals (`payoffs.py:18-19`, `signal_sigma=0.25`) — no ceiling. But: `ai_signal_boost` for High is 0.20 (`config_fp.py:56`), `ai_effort_benefit` for High is 0.05 (`config_fp.py:76`), and detection prob is 0.10 (`config_fp.py:63`). Yet `outputs_fp/fp_default/final_empirical_strategies.json:11-14` shows Highs adopt AI at 99.6% — opposite of bandit. This is because: at LowVerify equilibrium, detection cost is `0.5 * 0.10 * (1.0 + 1.0 * (0.60 + 2*0.30)) = 0.11`, while effort benefit alone is 0.05 *plus* signal-boost benefit of 0.2 standard deviations on `sigma_s = 0.25` (large gain in offer probability). The FP framing doesn't reproduce the bandit finding; whether this is "now correct" or "now broken in a different direction" depends on intended modelling. **Worth a careful sensitivity check before claiming FP is "more honest."**
| B5 | `mismatch_loss` and `ai_deception_loss` double-count | **Resolved** — FP firm utility is just `-mismatch - cost` (`payoffs.py:218`). No deception term. |
| B6 | Firms don't learn | **Resolved** — firm now best-responds via `firm_expected_utility` (`payoffs.py:182-218`) and FP empirical update at `fictitious_play.py:155`. With caveats from B3-FP. |

So the FP rewrite resolves B2, B3, B5, B6 cleanly; transforms B4 into "different equilibrium, possibly correct"; and does *not* resolve B1.

### I9-FP. Bandit-version Important findings — status under FP

- **I1 (no baselines)**: addressed via 6 arms in `make_arm_configs` (`run_fp_suite.py:36-69`). Better than bandit. But with B2-FP, only 5 arms are informative.
- **I2 (cold-start)**: not applicable — no temporal trust dynamics.
- **I3 (convergence_epoch reporting)**: replaced by stricter regret diagnostic (good), but I5-FP / I6-FP show `is_converged` is still misreported.
- **I4 (convergence criterion not robust)**: replaced by `fp_stability_tol = 0.002`. At late iterations step size is bounded by `1/(t+3) < 5e-4`, so the criterion trips trivially. Same problem in different clothing.
- **I5 (uniform spillover)**: dissolved (no spillover mechanism in FP). See I1-FP.
- **I6 (multiplicative trust)**: dissolved.
- **I7 (hand-picked hyperparameters)**: still applies. `config_fp.py:33-83` has many free parameters with no provenance. No sensitivity sweep beyond the 6-arm ablation.
- **I8 (zero detection cost weight)**: dissolved (firm cost weighting is now explicit `verification_cost`, used in firm utility).
- **I9 (linear-shrinkage estimator)**: replaced by *actual* Bayesian posterior over `(theta, a)` (`payoffs.py:124-129`). Strict improvement.
- **I10 (group/round assignment)**: not applicable.
- **I11 (epsilon decay)**: not applicable; FP is deterministic BR, no exploration. (See B3-FP — this is a feature concern: no exploration means identifiability of off-equilibrium payoffs is zero.)

---

## Minor findings

### M1-FP. Manifest under-records arm-specific config

- `_save_manifest` (`run_fp_suite.py:246-257`) writes only `num_fp_iterations`, `num_eval_samples`, `num_eval_seeds`, `seed`, `fixed_firm`. The actual arm-specific overrides (e.g. `cfg.base_detection_prob = {0:0,...}` for `fp_no_detection`, or `cfg.verification_cost` doubling for `fp_high_verification_cost`) are *not* persisted. Reproducing an arm requires reading `run_fp_suite.py` source. Add a full `dataclasses.asdict(cfg)` dump.

### M2-FP. `_run_fixed_firm` duplicates ~50 lines of `run_fictitious_play`

- `run_fp_suite.py:124-175` re-implements the FP loop with a hard-coded firm. Drift risk (e.g. if `run_fictitious_play` adds new logging fields, they won't show up in `fp_fixed_firm` trajectories). Refactor: pass an optional `fixed_firm` arg to `run_fictitious_play`.

### M3-FP. `evaluate_fp.py` inner loops are not vectorised

- `evaluate_fp.py:38-54` does Python-level `for ti in theta_indices`, `for theta, a in zip(...)`, and `bayes_optimal_offer` (`payoffs.py:47-54`) is called *per-sample* in a Python loop. With `num_eval_samples = 100_000` × `num_eval_seeds = 30` × 6 arms ≈ 18M scalar calls, this is slow. The vectorised `_bayes_optimal_offer_batch` already exists; use it. Performance, not correctness.

### M4-FP. `experiments/run_fp_baseline.py:51` print statement is misleading

- `f"Running held-out evaluation ({cfg.num_eval_seeds} seeds × {cfg.num_eval_samples:,} samples)"` strongly implies 30 *training* seeds. It is 30 *evaluation* seeds against one frozen training run. Sharpens the framing problem in B1-FP.

### M5-FP. `fp_no_detection` zeros `false_positive_rate` too

- `run_fp_suite.py:45`. Conceptually "no detection" should perhaps zero only the true-positive `base_detection_prob`. Zeroing `false_positive_rate` as well (`fp_*` from 0.005-0.020 to 0) means the firm cannot even mistakenly flag honest candidates — this changes the Bayesian posterior `posterior_type_action` (`payoffs.py:38-39`). Probably intended; document.

### M6-FP. `separating_index = std(ai_rate by type)` is a non-standard metric

- `evaluate_fp.py:150-151`. Fine as a summary, but separation in signaling games is more conventionally measured by a likelihood-ratio or by whether *every* type plays a distinct action. With 3 types and 2 actions, full separation is impossible — at least one pair of types pools. Worth a footnote.

### M7-FP. `comparison_table.csv` and `eval_per_seed.csv` don't include `seed` column

- The eval-seed loop (`evaluate_fp.py:171-178`) generates seeds via `base_rng.integers`. The actual integer seed used per row is not saved — only `eval_seed_idx`. Adding `seed = int(seed)` to the metrics dict would make any evaluation row independently reproducible.

### M8-FP. `tqdm` import (`fictitious_play.py:9`) is unconditional

- Will spam progress bars in non-interactive runs / CI. Wrap with a `disable=not sys.stdout.isatty()` flag or a `cfg.verbose`.

### M9-FP. `firm_expected_utility` draws actions via cumsum-trick (`payoffs.py:200-202`) but the comment says "in {0,1}" for general candidate_actions

- The `.clip(0, 1)` only works because `len(cfg.candidate_actions) == 2`. Brittle if action space ever grows. Use `searchsorted` or assert `len(cfg.candidate_actions) == 2`.

### M10-FP. `bayes_optimal_offer` has a degenerate `total < 1e-300` fallback to *uniform over `(theta,a)` joint*

- `payoffs.py:128-129`: when `total < 1e-300`, posterior is set to 1.0 to avoid divide-by-zero. This means the `argmin` then operates on a *uniform-weighted* loss. For the offer set `{0,1,2}` and types `{0,1,2}`, this consistently picks `o=1` (median). Not wrong, but should be logged if it ever fires.

---

## Open questions for the user

1. **Single training seed (B1-FP)**: do you intend to run the FP suite over multiple training seeds, or is the position "FP is deterministic given seed and CRN, so seed variance is uninteresting"? If the latter, the eval CIs in `comparison_table.md` need to be relabelled as "evaluation Monte-Carlo CI on a single deterministic equilibrium" and any phrase like "robust across seeds" cannot be used.

2. **Hybrid firm (B3-FP)**: is the intent that the firm only chooses verification *intensity* via FP and offers are always Bayes-optimal? Or should the offer rule itself be inside the FP action space (e.g. as a parameterised threshold that is best-responded to)? This determines whether you can cite Robinson / Monderer-Shapley FP convergence results.

3. **High-type AI adoption flip relative to bandit (I8-FP, B4 row)**: the FP setup says Highs use AI ~99% of the time; the bandit said ~0%. Both are equilibria of plausible models. Which does the project intend to claim? If FP, the new claim is "AI use is universal except for Lows", which is a very different headline than the old paper.

4. **Reputation-as-network (I1-FP)**: the FP version has no per-firm trust state and no network; the previous paper's spillover narrative cannot be supported by FP results. Will the paper be rewritten to be a pure signaling-game story, or will spillover dynamics be re-added (which would require leaving FP and going back to a stateful model)?

5. **Fixed-firm cycling (I6-FP)**: should the `fp_fixed_firm` arm's `r_max=0.06` and oscillation be reported as a finding ("FP fails to converge against a fixed adversary in this game class") or treated as a bug to be averaged-out?

6. **Tie-breaking convention (I2-FP)**: does the bias toward `prefer_lower=True` (No-AI) reflect a modeling choice (status-quo bias in candidates), or is it a coincidence of implementation? If the latter, an arm with `prefer_lower=False` should be added.

7. **LEMAS framing**: would you like the report to take a position on whether this is "fictitious play" in the textbook sense (B3-FP says: not quite — it's Bayesian best-response dynamics with an FP layer over verification), or is it OK to call it FP for course-purpose framing?

---

## Files referenced

- `config_fp.py`
- `fictitious_play.py`
- `payoffs.py`
- `evaluate_fp.py`
- `experiments/run_fp_suite.py`
- `experiments/run_fp_baseline.py`
- `experiments/aggregate_fp.py`
- `experiments/plot_fp_results.py`
- `outputs_fp/_compare/comparison_table.md`
- `outputs_fp/fp_default/eval_summary.json`
- `outputs_fp/fp_default/final_empirical_strategies.json`
- `outputs_fp/fp_default/fp_regret.json`
- `outputs_fp/fp_high_verification_cost/final_empirical_strategies.json`
- `outputs_fp/fp_no_detection/final_empirical_strategies.json`
- `outputs_fp/fp_fixed_firm/final_empirical_strategies.json`
