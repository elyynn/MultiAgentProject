# F5 - Market efficiency over time

- **Source artifacts**:
  - `outputs/v2/default/aggregate/epoch_curves.npz`
    (columns `correct_match_rate`, `overoffer_rate`, `underoffer_rate`)
  - `outputs/v2/null_ai/aggregate/epoch_curves.npz`
    (columns `correct_match_rate`, last-10-epoch mean)
- **N seeds**: 10 (default), 5 (null_ai)
- **Config diff**: `null_ai` disables the AI action entirely.
- **What it shows**: Three rates over training; bands are 95% CI across the
  10 default seeds. The dashed horizontal line is the `null_ai` correct-match
  floor: with no AI available, the firm's own classifier achieves a higher
  match rate, so AI in the `default` setting actually *hurts* matching
  efficiency.
- **Claim supported**: Closes audit B2 (correct-match should not be reported
  in isolation - context against the no-AI floor matters).
