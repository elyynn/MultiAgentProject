# Methodology Audit — LLM-Assisted Hiring Market Simulation

*Performed by the methodology-auditor role (read-only). No code was modified.*
*Scope: `agents.py`, `config.py`, `environment.py`, `learning.py`, `losses.py`, `main.py`, `metrics.py`, `simulation.py`.*

---

## Summary

The codebase is a clean, readable single-run simulation, but the experimental setup has several methodology issues that would invalidate or seriously weaken any quantitative claims. The most damaging are: (a) only one seed is ever run, (b) the candidate "Q-learning" is actually a stateless action-value bandit while the candidate's *reward* depends on hidden global state (information leakage from other firms' trust), (c) the offer-signal model has a hard ceiling that makes high-type AI use mathematically pointless, and (d) there are no baselines or ablations whatsoever, so the "central claim" has no comparator.

---

## Blocking findings

### B1. Single-seed results are not statistically defensible
- `config.py:8` sets `seed: int = 42`. `simulation.py:20` consumes it once. Nothing in `main.py:19-46` loops over seeds. No `--num_seeds` flag exists in `main.py:9-16`.
- All "final" metrics in `metrics.py:127-151` and the convergence diagnosis come from a single trajectory. Trust dynamics, candidate Q-init (`agents.py:42-46`), spillover targets (`environment.py:62-66`), and detection draws (`environment.py:11-15`) are all stochastic. With one seed you cannot distinguish a structural effect from sampling noise.
- **Fix scope**: this is one for-loop in `main.py` plus aggregation in `metrics.py`. Until it exists, no comparative claim ("AI use declines when…") can be supported.

### B2. Reward signal uses information the candidate cannot observe (omniscient candidate)
- `simulation.py:62-73` computes `trust_before / trust_after` on the *full row* `trust_matrix[candidate_id, :]` — i.e. across all 10 firms — and feeds the resulting `c_loss` straight into `update_candidate_q_value`.
- `losses.py:36` then defines `reputation_loss = sum(trust_before - trust_after)`, which includes the spillover penalties applied to two random firms the candidate never visited (`environment.py:61-72`).
- A real candidate cannot observe other firms' trust scores during the same interview round. The agent is therefore learning from a reward that is partly unobservable. This biases the Q-update toward AI-aversion in a way no real candidate could rationally derive, and it is a confound between the "spillover network" mechanic and the learned policy.
- **Why blocking**: any finding about how "candidates respond to detection risk" is contaminated by counterfactual information about firms the candidate never encountered.

### B3. Stateless action-value learning, mislabeled as Q-learning
- `agents.py:14` initializes `q_values: Dict[int, float] = {0: 0.0, 1: 0.0}` — exactly two scalars per candidate, indexed by action only.
- `learning.py:22-31` performs `new_q = old_q + alpha * (utility - old_q)` — a single-step incremental average of past rewards, with no state, no successor value, no discount, no bootstrap.
- This is a 2-armed contextual-free **bandit**, not Q-learning. Per problem framing this matters because (i) the environment is *non-stationary* (firm trust evolves), and a stateless bandit cannot represent that; (ii) the candidate's optimal action genuinely depends on state (own `true_type`, current trust against next firm, current epsilon regime), but none of that is in the Q-table.
- A defensible setup is either (a) call it a bandit and acknowledge the non-stationarity (constant `alpha=0.1` is then OK; in fact required), or (b) actually implement Q-learning with a state representation `(true_type, trust bucket, ...)`. As-is, the paper / write-up should not use the phrase "Q-learning." Methodologically this is blocking because the result interpretation depends on what the agent is.

### B4. Observation ceiling makes high-type AI use a free ride / dominant strategy depending on which side of the loss you read
- `environment.py:7-8`: `min(true_type + ai_action * cfg.ai_signal_boost, cfg.max_ability_level)` clips at 2.
- For `true_type=2`: `observed_signal` is 2 whether `ai_action` is 0 or 1. The firm cannot distinguish them, so AI use by high-types has zero upside on the offer side.
- But on the *cost* side: `losses.py:34` charges a detection penalty (`detect_ai_use` still fires for true_type=2 with `p=0.10`, `environment.py:11-15`), `losses.py:36` charges a reputation penalty if detected, and the candidate receives no benefit. Therefore high-type AI use is **strictly dominated** by non-use in expected loss.
- Yet `agents.py:32-33,42-46` initialise high-types to use AI 30% of the time, and the bandit will rationally drive that to ~0. If the paper reports "high-types stop using AI" as an interesting finding, it is a tautology of the signal model, not an emergent result. Confirm whether this is intentional; if not, the ceiling should be removed or the boost should be type-specific.

### B5. `mismatch_loss` and `ai_deception_loss` double-count overoffer
- `losses.py:64`: `mismatch_loss = (offer - true_type)^2`.
- `losses.py:66-70`: `ai_deception_loss = ai_action * I(offer > true_type)`.
- When AI was used and the firm overoffered, both terms fire. With the default weights all = 1.0 (`config.py:66-68`) and `(offer - true_type)^2 ≥ 1` whenever `offer > true_type`, the firm pays at least `1 + 1 = 2` for an AI-induced overoffer vs. `1` for a non-AI overoffer of the same magnitude. The firm "knows" it was AI-fooled inside its own loss — but the firm has *not* observed `ai_action` (only `detected`, which is stochastic). This is a similar leakage to B2: the firm's loss uses ground truth it does not have at inference time.
- Either (i) gate `ai_deception_loss` on `detected` (then it's a pure detection cost and overlaps with `detection_cost_loss` at `losses.py:77-81`), (ii) drop `ai_deception_loss` and rely on `mismatch_loss`, or (iii) acknowledge this as an analyst-side decomposition that is *not* a learning signal (it isn't fed into anything that learns — see B6).

### B6. Firms do not learn — there is no firm policy update
- `agents.py:19-25` defines `CompanyAgent` with `global_trust`. The only "update" is `environment.update_company_global_trust` at `environment.py:82-100`, which moves a scalar based on detected_rate. That is a heuristic, not an optimization.
- `losses.py:44-85` computes a `company_loss` value, but **nowhere is `e_loss` used to update any firm parameter**. `simulation.py:71` computes `e_loss`; `simulation.py:91` only logs it. The firm's offer policy in `environment.py:32-38` is a fixed threshold rule from `config.py:73-74`.
- This is a one-sided learning game presented as multi-agent RL. The framing should either become "candidates learn against fixed firm policies" (an evaluation of detection-as-deterrent given a fixed firm rule), or actually implement firm learning (e.g. learning the threshold, or learning `effective_trust` weighting, or learning detection investment).
- Until then, "trust dynamics" are mechanical bookkeeping, not strategic firm behaviour.

---

## Important findings

### I1. No baselines or ablations
- The codebase has one configuration. There is no:
  - "no-detection" arm (`detection_prob_by_type` all zero) to isolate the deterrent effect,
  - "no-spillover" arm (`num_spillover_companies = 0`) to isolate the network mechanism,
  - "no-trust" arm (`effective_trust = 1` always) to isolate trust as an information channel,
  - "fixed AI use" arm (skip `update_candidate_q_value`) to isolate learning from initial conditions,
  - "no AI" arm (initial_ai_rate = 0).
- Without ≥1 of these, no causal statement about *which mechanism* drives the dynamics is supportable. This is itself a blocking-class issue for any causal claim, but I list it as Important because experiments — not code — are missing; the harness can produce them.

### I2. Cold-start dynamics give every candidate a default high offer on day 1
- With `prior_mean_ability = 1.0` (`config.py:77`) and `initial_trust = 1.0` everywhere (`config.py:33`, `agents.py:54`, `agents.py:60-64`), `effective_trust = 1.0` on the first interview, so `estimated_ability = observed_signal`.
- Low-type without AI: signal=0 → estimate 0 → reject (threshold 0.70).
- Medium-type without AI: signal=1 → estimate 1.0 → low offer (`1.0 < 1.50`).
- High-type without AI: signal=2 → estimate 2.0 → high offer.
- Low-type *with* AI: signal=1 → low offer. **Strictly better than truth-telling for low-types**, until detection is cumulatively learned.
- This is consistent with the design but means the initial advantage to AI use is entirely concentrated on Low and Medium types; combined with B4 it means the observed dynamics will be "Low and Medium learn to use AI when detection is low and stop when penalty exceeds expected gain; High does nothing interesting." The paper should pre-state this rather than report it as discovery.

### I3. `convergence_epoch` is reported as the last epoch, not the first
- `metrics.py:131-134`: when `converged=True`, `convergence_epoch = last epoch`. The window check at `metrics.py:117-125` only knows that the *most recent* `convergence_window` epochs are stable.
- Effect: any reader of `final_summary.json` inferring "the system stabilised at epoch N" is being told the wrong N — they're being told the run length. The correct value is `len(epoch_logs) - convergence_window` (the first epoch of the stable window) or, better, a search backward for the first time the criterion was satisfied.
- Not blocking because `early_stop=False` by default (`config.py:82`), so the run goes to 100 epochs regardless and `convergence_epoch == 99` always when it triggers — but it is reported and likely to be cited.

### I4. Convergence criterion is not robust
- `metrics.py:117-125` requires *max - min* of three series within `tol=0.01` over a 10-epoch window. Issues:
  - `average_offer` is on a 0-2 scale; `average_global_trust` is on 0-1; `average_ai_usage` is on 0-1. A single shared `tol=0.01` is unequal in relative terms (0.5% of offer range vs 1% of trust range).
  - With ~1000 interviews per epoch (`100 candidates × 10 rounds`), epoch-to-epoch noise from epsilon-greedy alone (`learning.py:7-9`) at min_epsilon=0.02 is ~`sqrt(0.02 * 0.98 / 1000) ≈ 0.0044` per series — close to tol. Convergence will trip on noise floors, not on policy stability.
  - Conjunction over three series with tight tol may also *never* trip; either way, "converged" is not a meaningful flag here.

### I5. Spillover targets are uniformly random with no network structure
- `environment.py:61-66`: `rng.choice(other_companies, size=2, replace=False)` — uniform over the 9 non-interviewing firms, redrawn each detection event.
- The project framing implies "reputation networks." A uniform-random spillover is not a network model — it is a noisy global broadcast with stochastic targeting. There is no graph, no homophily, no industry clustering, no persistence (the same pair of firms aren't preferentially informed).
- Either (i) adopt a fixed firm-firm graph in `agents.py`, or (ii) reframe the mechanism as "stochastic reputation diffusion" and stop calling it a network. As-is, with `num_spillover_companies=2` out of 9, the per-detection expected aggregate trust loss is `0.60 + 2 * 0.30 = 1.20` (capped by min_trust=0), 2x the direct penalty — a very strong indirect signal driven entirely by an unjustified parameter.

### I6. Firm `global_trust` is firm-wide, but firms also use per-candidate individual trust — and the two compose multiplicatively
- `environment.py:24`: `effective_trust = individual_trust * global_trust`. Both initialise to 1.0.
- After a single detection, `individual_trust` for that (cand, firm) pair drops by 0.60 (`config.py:38`). After a single epoch with detection_rate>0, `global_trust` drops by `0.05 * detected_rate` (`environment.py:96`). These are very different time-scales and very different semantics (individual = my prior on this person; global = my prior on the world's trustworthiness). Multiplying them means a firm with `global_trust=0.5` discounts *truthful* signals from candidates with `individual_trust=1.0` by 50% — punishing honest, never-detected candidates for the population's behaviour.
- That may be the intended thesis. If so, it should be defended; if not, an additive or max-based combination would not punish never-detected candidates. Worth flagging because this composition rule is the central informational mechanism of the paper.

### I7. Hyperparameters appear hand-picked; no provenance
- Detection probs `0.35 / 0.20 / 0.10` (`config.py:98-102`), trust penalties `0.60 / 0.30` (`config.py:38-39`), spillover count `2` (`config.py:40`), recovery rate `0.01` (`config.py:43`), candidate `lr=0.10`, `epsilon=0.10`, `decay=0.995`, `min=0.02` (`config.py:51-54`), `firm_global_trust_lr=0.05` (`config.py:46`), thresholds `0.70` and `1.50` (`config.py:73-74`).
- Nothing in the comments or `reports/project_map.md` cites a source, calibration target, or sensitivity analysis. No sweep harness exists. Recommend a sensitivity analysis over at least `direct_trust_penalty`, `spillover_trust_penalty`, `num_spillover_companies`, and `detection_prob_by_type` before any quantitative claim is published.

### I8. `company_detection_cost_weight = 0.0` makes detection free for the firm
- `config.py:69-70`: both `company_detection_cost_weight` and `company_detection_cost` default to 0. The corresponding term at `losses.py:77-81` is identically zero in every reported run.
- Since firms don't learn (B6) this has no policy effect, but it is misleading: the loss decomposition reported in `metrics.py:99` (`average_company_loss`) excludes the very cost that, in any real policy choice over detection effort, would create the trade-off the paper claims to study. If the project narrative is "firms balance detection cost against welfare", `0.0` weight contradicts the narrative.

### I9. Bayesian-coherence claim for ability estimate is weak
- `environment.py:18-29`: `effective_trust * observed_signal + (1-effective_trust) * prior_mean`.
- This is a *linear shrinkage* estimator, not a Bayesian posterior over discrete types `{0,1,2}`. A Bayesian update would compute `P(true_type | observed_signal, p_AI_use, p_detect)` and produce a posterior mean over types. The current rule is fine as a heuristic but the wording "estimate" / "trust as posterior weight" should be hedged. In particular:
  - It can produce non-integer estimates that are then thresholded — which is fine, but the thresholds `0.70` and `1.50` are not derived from anything; they sit awkwardly between integer ability levels.
  - With `effective_trust=0`, every estimate equals `prior_mean=1.0` regardless of signal → every candidate is offered the low offer. A fully untrusted firm makes maximally uniform uninformative offers — defensible, but worth stating.

### I10. Round/group assignment is deterministic given group_id and round_id; not all (group, company) pairs are sampled equally
- `simulation.py:39`: `company_id = (group_id + round_id) % num_companies`.
- With 10 groups × 10 rounds × 10 companies, this is a Latin-square-style rotation. **Each group meets each company exactly once per epoch.** Good — exposure across (group, company) is uniform within an epoch.
- However, **groups are reshuffled every epoch** (`simulation.py:31-35`), so individual *candidates* meet the same set of firms each round (within their group), but the within-group composition rotates. The result is that every candidate eventually meets every firm with high probability, but the *order* in which a given candidate sees firms is non-uniform across epochs. This interacts with cold-start trust dynamics: a candidate who happens to be detected by firm 3 in round 0 will have low individual_trust at firm 3 forever, but the *spillover* targets are re-randomized each event. Worth noting; not necessarily a bug but a subtle source of variance that single-seed runs will not surface.

### I11. Epsilon decay is global, not per-candidate
- `simulation.py:99` applies `decay_epsilon` once per epoch to a single shared `epsilon` variable. Applied to 100 candidates equally.
- Each candidate sees ~10 interviews per epoch (one per round). At decay=0.995 over 100 epochs, terminal epsilon = `max(0.02, 0.10 * 0.995^100) ≈ max(0.02, 0.0606) ≈ 0.061`. So epsilon never reaches min_epsilon and the explore floor stays around 6%.
- For a 2-armed bandit with ~1000 plays per arm over the run, this is fine. But "exploration is bounded below by ~6%" should be stated, and the convergence criterion (I4) interacts with this floor.

---

## Minor findings

### M1. Q-value warm start is asymmetric and biases initial choice
- `agents.py:42-46`: candidates initialised with `+0.1` on their initially preferred action. This is a small bias but combined with the deterministic tie-break in `learning.py:14-17` it ensures a candidate's first non-exploratory action equals their `uses_ai_initially` flag. That makes `initial_ai_rate` more sticky than it otherwise would be. Either remove the warm start or document its effect.

### M2. `min_trust` cap and `max_trust` cap interact with detection signal
- `environment.py:56-58, 69-72`: trust is clipped to `[0, 1]`. Once `individual_trust = 0`, no further detection can lower it, so the candidate's reward stops responding to new detections at that firm — the marginal Q-update at high-detection candidates becomes truncated. This dampens learning precisely for the type the model wants to study (low-types using AI). Consider unbounded log-trust, or document the floor effect.

### M3. `firm_global_trust_learning_rate` and `trust_recovery_rate` are conflated
- `environment.py:91`: when `detected_rate == 0`, recovery uses `cfg.trust_recovery_rate` (the per-interview individual recovery, 0.01). When detection > 0, decrement uses `cfg.firm_global_trust_learning_rate` (0.05). These are conceptually different timescales for the same scalar — the firm's "recovery" is at individual-trust speed but "decay" is at firm-trust speed. Likely a copy-paste; consider an explicit `firm_global_trust_recovery_rate`.

### M4. `num_companies` and `group_size` coupling is implicit
- `simulation.py:39` requires `num_companies` to evenly divide the rotation, which here is true (10/10/10). If a future config breaks this — e.g. `num_companies=8` — exposure will be non-uniform silently. Add an assertion in `run_simulation` start.

### M5. `detected_count`/`interview_count` reset each epoch loses long-run base-rate signal
- `environment.py:99-100`: counters zero out after each global trust update. Firms cannot accumulate "memory" beyond one epoch. This means firm `global_trust` is essentially driven by the most recent epoch's empirical detection rate. Worth stating; perhaps replace with EMA.

### M6. Unused dataclass `field` import in `config.py`
- `config.py:1` imports `field` but does not use it. (`agents.py` does use it correctly at line 14.) Cosmetic.

### M7. `TYPE_NAMES` declared but never used
- `metrics.py:8`. Cosmetic; either use it for column naming consistency or drop.

### M8. `metrics.to_results` does not include any per-type final aggregates
- `metrics.py:127-151`. The epoch logs include type-stratified rates (`metrics.py:84-95`) but `to_results()` only surfaces averages. Any plot or paper claim about "low-types vs high-types" must scrape `epoch_logs` directly. Not a bug, but easy to overlook.

### M9. `reputation_loss` can be negative (counts as a candidate *gain*)
- `losses.py:36`: when no detection occurs, `trust_after - trust_before = +trust_recovery_rate` for the interviewing firm, so `trust_before - trust_after = -0.01` and `reputation_penalty = -0.01`. The candidate is *rewarded* for an interview that didn't trigger detection. This is small (0.01) but applies every interview without detection, so it accumulates: ~0.10 per epoch at zero detection, comparable to one full offer of value. This may incidentally bias candidates toward AI use (because non-AI also gets the recovery, but AI-without-detection gets recovery + offer). Sign is consistent; magnitude is not negligible.

---

## Open questions for the user

1. **Intended agent class for candidates**: should this be a true Q-learner with state (e.g. own type + own past detection count), or is the bandit framing intended? The paper text's wording will determine whether B3 is a fix-the-code or fix-the-prose issue.

2. **Information assumption on candidate side (B2)**: do you intend candidates to know the spillover trust drops at the moment of decision? If yes, justify (rumor mill, Glassdoor); if no, the reputation term must be either (a) discounted by future-encounter probability, (b) revealed only when the candidate next visits a spillover firm, or (c) removed from the learning signal and used only as an analyst-side observable.

3. **High-type AI ceiling (B4)**: is it intentional that high-type AI use has no upside? If you want a "rational AI use" finding, you may want to remove the ceiling or differentiate boost by type.

4. **Spillover network (I5)**: do you want a real graph (firms connected via industry / referral), or is the uniform broadcast deliberate? The latter is hard to defend as "reputation network."

5. **Firm learning (B6)**: is this paper's thesis "candidates learn to game a fixed firm policy" or "co-evolution of detection and use"? Current code is firmly the former; the project_map framing leans toward the latter.

6. **Calibration source**: do detection probabilities, trust penalties, and threshold values come from any literature or empirical anchor? If not, a sensitivity sweep is essential before the central numbers go in a paper.

7. **Multi-seed plan (B1)**: how many seeds, what aggregation (mean ± SE? CI? distributional plots?), and what's the budget? With 100 epochs × ~10k interviews × 100 candidates the run is cheap; ≥30 seeds is feasible.

8. **Convergence reporting (I3, I4)**: do you actually need a convergence flag in the paper? If not, drop the plumbing; if yes, redefine it as "first epoch of a stable window" with a per-series tolerance.
