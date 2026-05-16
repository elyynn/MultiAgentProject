# F2 - Detection rate over time (aggregate)

- **Source artifact**: `outputs/v2/default/aggregate/epoch_curves.npz`
  (columns `detection_rate`, `detection_rate_low/medium/high`)
- **N seeds**: 10 (`default` arm)
- **Config diff**: none (baseline)
- **What it shows**: Per-epoch detection rate (detected / all candidates) overall
  and by `true_type`, mean +/- 95% CI across seeds.
- **Important caveat**: This is the *aggregate* rate (denominator = all
  candidates, including non-AI users). When a type's AI-usage is near zero
  the per-type curve is dominated by Monte Carlo noise. The
  AI-conditioned rate (proper validation that the simulator's
  `detection_prob_by_type` parameter is recovered) is in **F8**.
- **Claim supported**: Companion to F8; do not use this figure alone to argue
  that detection works.
