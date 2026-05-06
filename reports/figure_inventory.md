# v2 figure inventory

This inventory covers the paper- and slide-ready figures produced from the
35-run multi-seed suite under `outputs/v2/`. All figures are computed from
the per-arm aggregate artifacts (`outputs/v2/<arm>/aggregate/`) or, where
the audit demanded sub-aggregate granularity (B1, W2), from the per-(arm, seed)
`interview_logs.csv` files.

## How this set closes outstanding audit findings

| Audit finding | Closure |
|---|---|
| **B1** offer distribution misrepresented as a linear interpolation | **F9** plots the true categorical offer distribution per `true_type` from `interview_logs.csv` (last 10 epochs, all 10 default seeds). |
| **W2** detection rate per type dominated by Monte-Carlo noise | **F8** computes the AI-conditioned detection rate (rows with `ai_action == 1`) from `interview_logs.csv` and overlays the configured `cfg.detection_prob_by_type` targets. The empirical rate matches the parameter to within SEM. |
| **W3 / W6** sign-convention dishonesty for candidate loss | **F4** plots `-average_candidate_loss` and labels it explicitly as utility. |
| **W5** single-trajectory plots no longer acceptable | Every time-series figure (F1-F6, F10) is mean +/- 95% CI across seeds (1.96 * SEM); cross-arm bar chart F7 shows mean +/- SEM. |
| **B2** correct match rate without context; global trust as headline | **F5** annotates the `null_ai` correct-match floor as a dashed horizontal line, exposing that AI in `default` lowers efficiency below the no-AI baseline. **F3** splits global vs. individual trust onto separate panels (no shared y-axis), demoting global trust from headline status. |

## Figure table

| ID | Title | File (paper) | File (slides) | Source data | N seeds | Audit findings closed | Caption file |
|---|---|---|---|---|---|---|---|
| F1 | AI adoption over time (default, overall + per type) | `outputs/v2/figures/paper/F1_ai_adoption_over_time.{pdf,png}` | `outputs/v2/figures/slides/F1_ai_adoption_over_time.png` | `outputs/v2/default/aggregate/epoch_curves.npz` (`average_ai_usage`, `average_ai_usage_low/medium/high`) | 10 | W5 | `outputs/v2/figures/captions/F1_ai_adoption_over_time.md` |
| F2 | Detection rate over time (aggregate, default) | `outputs/v2/figures/paper/F2_detection_rate_over_time.{pdf,png}` | `outputs/v2/figures/slides/F2_detection_rate_over_time.png` | `outputs/v2/default/aggregate/epoch_curves.npz` (`detection_rate`, `detection_rate_low/medium/high`) | 10 | W5 (and links to W2 via caption pointing at F8) | `outputs/v2/figures/captions/F2_detection_rate_over_time.md` |
| F3 | Firm trust over time (individual vs global, separate panels) | `outputs/v2/figures/paper/F3_trust_over_time.{pdf,png}` | `outputs/v2/figures/slides/F3_trust_over_time.png` | `outputs/v2/default/aggregate/epoch_curves.npz` (`average_individual_trust`, `average_global_trust`) | 10 | W5, B2 (separates panels), pilot P4 | `outputs/v2/figures/captions/F3_trust_over_time.md` |
| F4 | Losses over time (candidate utility, company loss) | `outputs/v2/figures/paper/F4_losses_over_time.{pdf,png}` | `outputs/v2/figures/slides/F4_losses_over_time.png` | `outputs/v2/default/aggregate/epoch_curves.npz` (`average_candidate_loss`, `average_company_loss`) | 10 | W3, W5, W6 | `outputs/v2/figures/captions/F4_losses_over_time.md` |
| F5 | Market efficiency over time (correct/over/under, with null_ai floor) | `outputs/v2/figures/paper/F5_market_efficiency_over_time.{pdf,png}` | `outputs/v2/figures/slides/F5_market_efficiency_over_time.png` | `outputs/v2/{default,null_ai}/aggregate/epoch_curves.npz` (`correct_match_rate`, `overoffer_rate`, `underoffer_rate`) | 10 (default), 5 (null_ai) | B2, W5 | `outputs/v2/figures/captions/F5_market_efficiency_over_time.md` |
| F6 | AI usage trajectory across all arms (headline) | `outputs/v2/figures/paper/F6_ai_usage_all_arms.{pdf,png}` | `outputs/v2/figures/slides/F6_ai_usage_all_arms.png` | `outputs/v2/<arm>/aggregate/epoch_curves.npz` (`average_ai_usage`) for all 6 arms | 10 (default), 5 (each ablation) | W5 | `outputs/v2/figures/captions/F6_ai_usage_all_arms.md` |
| F7 | Headline scalars across arms (5-panel bar chart) | `outputs/v2/figures/paper/F7_headline_scalars_across_arms.{pdf,png}` | `outputs/v2/figures/slides/F7_headline_scalars_across_arms.png` | `outputs/v2/<arm>/aggregate/summary.json` for all 6 arms | 10 (default), 5 (each ablation) | W5 (SEM bars), B2 (correct match rate is one of the panels alongside null_ai) | `outputs/v2/figures/captions/F7_headline_scalars_across_arms.md` |
| F8 | AI-conditioned detection rate by type (replaces W2 plot) | `outputs/v2/figures/paper/F8_ai_conditioned_detection_by_type.{pdf,png}` | `outputs/v2/figures/slides/F8_ai_conditioned_detection_by_type.png` | `outputs/v2/default/seed_*/results/interview_logs.csv` (rows: `ai_action == 1`, last 10 epochs) | 10 | **W2** | `outputs/v2/figures/captions/F8_ai_conditioned_detection_by_type.md` |
| F9 | True offer distribution by type (replaces B1 plot) | `outputs/v2/figures/paper/F9_true_offer_distribution_by_type.{pdf,png}` | `outputs/v2/figures/slides/F9_true_offer_distribution_by_type.png` | `outputs/v2/default/seed_*/results/interview_logs.csv` (last 10 epochs, categorical `offer in {0,1,2}` per `true_type`) | 10 | **B1** | `outputs/v2/figures/captions/F9_true_offer_distribution_by_type.md` |
| F10 | Per-type AI usage over time (default) | `outputs/v2/figures/paper/F10_per_type_ai_usage_over_time.{pdf,png}` | `outputs/v2/figures/slides/F10_per_type_ai_usage_over_time.png` | `outputs/v2/default/aggregate/epoch_curves.npz` (`average_ai_usage_low/medium/high`) | 10 | W5; underscores methodology B4 (high-type ceiling) | `outputs/v2/figures/captions/F10_per_type_ai_usage_over_time.md` |

## Style notes

- **Palette (arms)**: Okabe-Ito color-blind safe.
  - `default` = neutral dark gray `#333333` (reads as the baseline).
  - `null_ai` = green `#009E73` (the no-AI floor).
  - `no_detection` = vermillion `#D55E00` (large-effect arm).
  - `no_spillover` = blue `#0072B2`.
  - `sens_detect_lo` = orange `#E69F00`.
  - `sens_detect_hi` = sky blue `#56B4E9`.
  - Same color and label everywhere (F6 + F7).
- **Palette (candidate types)**: Low = blue `#0072B2`, Medium = orange `#E69F00`, High = reddish purple `#CC79A7`. Same color and label everywhere (F1, F2, F8, F9, F10).
- **Palette (offer categories)**: reject = light gray, low offer = sky blue, high offer = vermillion (F9 only).
- **Confidence bands**: 95% CI = 1.96 * SEM across seeds, `fill_between(..., alpha=0.2)`.
- **Paper variant**: `figsize=(5.5, 3.5)` per panel, font size 9-10, line width 1.5; saved as `.pdf` (vector, embedded fonts via `pdf.fonttype=42`) and `.png` at 300 dpi.
- **Slides variant**: `figsize=(10, 6)` per panel, font size 14-16, line width 2.5; saved as `.png` at 200 dpi.
- **Common rcParams**: top/right spines off; light grid (alpha 0.3); `bbox='tight'`.
- **Captions**: one Markdown file per figure under `outputs/v2/figures/captions/`; each lists source artifact, N seeds, config diff, what the figure shows, and what claim it supports.

## Open issues / what this inventory does NOT cover

- **Cross-arm trajectories for trust and losses**. Only AI usage (F6) is shown across all arms this iteration. The cross-arm comparison for trust, market efficiency and losses lives only in the bar chart F7 and the table at `outputs/v2/_compare/comparison_table.md`. A future iteration should add per-arm overlays for `average_global_trust`, `correct_match_rate`, and `average_company_loss`.
- **Held-out evaluation pass not implemented**. Every figure is training-tail (last 10 epochs of the 100-epoch training run). Per the run-log, a clean held-out eval pass is deferred because it would require modifying `simulation.py` / `metrics.py`, which is outside the figure-curator's scope.
- **Per-arm `interview_logs.csv` analyses (F8, F9) only cover `default`**. The same plots could be regenerated per ablation arm, but this iteration scopes them to the baseline because that's where the audit findings B1 and W2 originated. F8/F9 functions in `figures/make_figures.py` are parameterised on `arm` and could be lifted into a per-arm loop later.
- **Statistical tests omitted**. No bootstrap p-values, no formal tests of arm differences. Consistent with `comparison_table.md` ("no bootstrap p-values this iteration"); error bars + CI bands are the only uncertainty signal in these figures.
- **Visualisation spec for individual seed trajectories**. The data is there (per-seed rows in `epoch_curves.npz`) but the only thing each figure shows from per-seed structure is the band. A "spaghetti" backup plot could be added if a reviewer questions whether the mean hides bimodal behaviour - cursory inspection of the bands in F1 and F6 suggests the seeds are tightly clustered.

## Surprises / things to flag from the data

- **F6**: The `no_spillover` arm rises to ~50% AI usage early and then *crashes* by epoch ~40 to roughly the `default` level (~10%). This non-monotone trajectory is masked by the final-epoch column in `comparison_table.md` - worth noting in any narrative that uses F6.
- **F5**: AI in the `default` arm produces a *lower* correct-match rate (~0.70) than the `null_ai` floor (~0.81). Over-offering at ~27% is the main inefficiency; under-offer is small. This is the headline qualitative result hidden by the pilot's per-seed plot.
- **F8**: The AI-conditioned detection rate matches `cfg.detection_prob_by_type` very tightly (Low 0.33 vs target 0.35; Medium 0.21 vs 0.20; High 0.10 vs 0.10), so the simulator's detection mechanism is verifiably calibrated - the previous "noisy" appearance was purely an aggregation artifact.
- **F9**: For `true_type == Medium`, ~98% of offers are in the "low offer" category, which is the *correct* match for Medium (encoding: type 0=Low, 1=Medium, 2=High; offer 0=reject, 1=low, 2=high). For `true_type == Low`, ~78% receive a low offer (over-offer) and ~22% are rejected (correct). This makes Low the dominant source of over-offer error.

## How to regenerate

From the project root (`C:/Users/Computer/Desktop/UC Berkeley/EE 290/MultiAgentProject`):

```bash
# All 10 figures, both variants (paper + slides) - default behaviour.
python figures/make_figures.py

# Equivalent explicit form.
python figures/make_figures.py --variant both --figs all

# Subset (paper variant only, just F8 and F9).
python figures/make_figures.py --variant paper --figs F8,F9
```

The script is purely reads from `outputs/v2/` and writes to `outputs/v2/figures/`; it does not touch any simulation code, the pilot artifacts under `outputs/figures/` or `outputs/results/`, or any per-seed result files.

Wall-clock for the full run: **~22 s** on the host the suite was generated on.
