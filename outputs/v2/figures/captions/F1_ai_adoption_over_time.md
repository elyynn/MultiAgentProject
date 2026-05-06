# F1 - AI adoption over time

- **Source artifact**: `outputs/v2/default/aggregate/epoch_curves.npz`
  (columns `average_ai_usage`, `average_ai_usage_low/medium/high`)
- **N seeds**: 10 (`default` arm)
- **Config diff**: none (baseline)
- **What it shows**: Fraction of candidates choosing the AI action across the
  100 training epochs, broken down by `true_type` and shown overall. Shaded
  bands are 95% CI across seeds (1.96 * SEM).
- **Claim supported**: AI use settles at a low overall rate (~6%) and is
  carried almost entirely by the Medium-type candidates - High-type rarely
  benefits, Low-type is detected too easily.
