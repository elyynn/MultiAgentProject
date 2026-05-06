# F9 - True offer distribution by type (replaces B1)

- **Source artifact**: `outputs/v2/default/seed_*/results/interview_logs.csv`
  (last 10 epochs of each seed; categorical fraction of `offer in {0,1,2}`
  per `true_type`).
- **N seeds**: 10 (`default` arm).
- **Config diff**: none (baseline).
- **What it shows**: Stacked bars of the actual offer distribution per true
  type. Segment heights = mean fraction across seeds; error bars at each
  segment top = SEM across seeds.
- **Claim supported**: Closes audit B1. The pilot plot
  `outputs/figures/offer_distribution_by_type.png` was a linear *interpolation*
  of mean offers and so misrepresented the categorical distribution. This
  figure shows the truth: e.g., for High-type ~90% of offers are correct
  (offer=2), while Low-type splits roughly into reject and over-offer.
