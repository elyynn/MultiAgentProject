# F4 - Losses over time

- **Source artifact**: `outputs/v2/default/aggregate/epoch_curves.npz`
  (columns `average_candidate_loss`, `average_company_loss`)
- **N seeds**: 10 (`default` arm)
- **Config diff**: none (baseline)
- **What it shows**: Left panel: candidate **utility** = -`average_candidate_loss`,
  so positive is good. Right panel: average company loss (positive = bad).
  Bands are 95% CI across seeds.
- **Claim supported**: Closes audit W3/W6 (sign-convention honesty). The
  candidate loss in the raw artifact is negative; relabeling as utility
  prevents the pilot's confusion.
