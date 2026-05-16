# Results Summary Table — LLM-Assisted Hiring Market Simulation

*One row per scalar in `outputs/results/final_summary.json`, plus one row per figure in `outputs/figures/`, plus reproducibility checklist. Verdict legend: OK / WEAK / UNSUPPORTED / MISLEADING / MISSING.*

| Claim | Source | Value | Verified against | Verdict | Notes |
|---|---|---|---|---|---|
| `num_epochs_run` | `final_summary.json:2` | 100 | `cfg.num_epochs=100` (`config.py:17`) | OK | Restates config; not a measurement. |
| `converged` | `final_summary.json:3` | false | recomputed `metrics.py:110-125` over `epoch_metrics.csv` epochs 90-99 | OK | 2 of 3 series exceed tol=0.01. |
| `convergence_epoch` | `final_summary.json:4` | null | consistent with `converged=false` | OK | Off-by-window bug latent (methodology I3). |
| `final_average_ai_usage` | `final_summary.json:5` | 0.039 | `epoch_metrics.csv` row 99 col `average_ai_usage` | OK | Single-seed point estimate; epsilon still ~0.061. |
| `final_average_global_trust` | `final_summary.json:6` | 0.99015 | `epoch_metrics.csv` row 99 col `average_global_trust` | UNSUPPORTED | High because AI use vanished, not because trust mechanism works. |
| `final_average_offer` | `final_summary.json:7` | 1.058 | `epoch_metrics.csv` row 99 col `average_offer` | OK | Single trajectory; range 1.051-1.073 over last 10. |
| `final_detection_rate` | `final_summary.json:8` | 0.006 | `epoch_metrics.csv` row 99 col `detection_rate` | OK | Tiny absolute count; per-type ordering noisy. |
| `final_correct_match_rate` | `final_summary.json:9` | 0.734 | `epoch_metrics.csv` row 99 col `correct_match_rate` | WEAK | No baseline; trivial heuristic likely scores similarly. |
| `final_overoffer_rate` | `final_summary.json:10` | 0.237 | `epoch_metrics.csv` row 99 col `overoffer_rate` | OK | Sums with under+correct to 1.000. |
| `final_underoffer_rate` | `final_summary.json:11` | 0.029 | `epoch_metrics.csv` row 99 col `underoffer_rate` | OK | Sums with over+correct to 1.000. |
| `final_average_candidate_loss` | `final_summary.json:12` | -1.05726 | `epoch_metrics.csv` row 99 col `average_candidate_loss` | MISLEADING | Negative = candidate gain (sign convention, see W3). |
| `final_average_company_loss` | `final_summary.json:13` | 0.307 | `epoch_metrics.csv` row 99 col `average_company_loss` | OK | Excludes detection cost (weight=0, `config.py:69-70`). |
| Figure: AI adoption over time | `outputs/figures/ai_adoption_over_time.png` | curves | `epoch_metrics.csv` cols `average_ai_usage*` | WEAK | Single-trajectory, no CI. |
| Figure: Detection rate over time | `outputs/figures/detection_rate_over_time.png` | curves | `epoch_metrics.csv` cols `detection_rate*` | WEAK | Per-type ordering violates cfg in late epochs (W2). |
| Figure: Firm trust over time | `outputs/figures/firm_trust_over_time.png` | curves | `epoch_metrics.csv` cols `average_individual_trust`, `average_global_trust` | WEAK | Two different semantics on shared y-axis; single seed. |
| Figure: Losses over time | `outputs/figures/losses_over_time.png` | curves | `epoch_metrics.csv` cols `average_candidate_loss`, `average_company_loss` | WEAK | Candidate sign convention not annotated; single seed. |
| Figure: Market efficiency over time | `outputs/figures/market_efficiency_over_time.png` | curves | `epoch_metrics.csv` cols `correct_match_rate`, `overoffer_rate`, `underoffer_rate` | WEAK | No baseline comparator; single seed. |
| Figure: Offer distribution by type | `outputs/figures/offer_distribution_by_type.png` | bars | `visualization.py:65-79` (linear interp of `average_offer_<type>`) | MISLEADING | Title implies empirical distribution; figure is interpolation. |
| Reproducibility: seed log | none | absent | searched `outputs/`; no manifest file | MISSING | Seed only inferable from `config.py:8`. |
| Reproducibility: config snapshot | none | absent | searched `outputs/` | MISSING | No copy of run config saved. |
| Reproducibility: git SHA | none | absent | searched `outputs/` | MISSING | No version pin saved. |
| Reproducibility: command line / timestamp | none | absent | searched `outputs/` | MISSING | No invocation log. |
| Multi-seed runs | inventory | N=1 | only one set of files in `outputs/results/` and `outputs/figures/` | MISSING | Required for any statistical claim. |
| Error bars / CIs | `visualization.py` plots | none | no `fill_between` / `errorbar` calls | MISSING | Impossible with one seed. |
| Train/eval split | `main.py:35-46`, `simulation.py:19-114` | none | `final_*` fields = last training epoch | MISSING | No held-out eval pass. |
| Baseline arms | inventory | none | no comparator artifacts in `outputs/` | MISSING | Methodology I1; required for causal claims. |
