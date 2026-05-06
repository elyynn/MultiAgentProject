# F3 - Firm trust over time

- **Source artifact**: `outputs/v2/default/aggregate/epoch_curves.npz`
  (columns `average_individual_trust`, `average_global_trust`)
- **N seeds**: 10 (`default` arm)
- **Config diff**: none (baseline)
- **What it shows**: Two panels, **separate y-axes** because the two trust
  scalars have different semantics (individual is per-candidate posterior;
  global is firm-level scalar). Mean +/- 95% CI across seeds.
- **Claim supported**: Resolves pilot pitfall P4 (sharing a y-axis was
  misleading). Both metrics should be reported, but global trust should
  not be the headline (audit B2).
