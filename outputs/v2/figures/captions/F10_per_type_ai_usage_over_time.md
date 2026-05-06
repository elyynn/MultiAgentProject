# F10 - Per-type AI usage over time

- **Source artifact**: `outputs/v2/default/aggregate/epoch_curves.npz`
  (columns `average_ai_usage_low/medium/high`).
- **N seeds**: 10 (`default` arm).
- **Config diff**: none (baseline).
- **What it shows**: AI-usage trajectory broken out per true type, with
  95% CI bands across seeds. Same data as F1 but with the "Overall" line
  removed so the type-level dynamics are easier to read.
- **Claim supported**: Underscores methodology B4 - the High-type ceiling.
  Even with detection turned to baseline rates, High-types essentially
  never adopt AI because their unaided signal already gets the offer they
  want; AI use is carried by Medium-types.
