# F7 - Headline scalars across arms

- **Source artifact**: `outputs/v2/<arm>/aggregate/summary.json` for all six arms.
- **N seeds**: 10 (`default`), 5 (each ablation).
- **Config diff**: see F6 caption.
- **What it shows**: Five-panel bar chart of the headline final-epoch scalars
  (AI usage, global trust, correct-match rate, over-offer rate, company loss),
  one bar per arm, error bars = SEM across seeds. Color per arm matches F6.
- **Claim supported**: Compact summary of the cross-arm comparison table
  (`outputs/v2/_compare/comparison_table.md`); supports the result that
  removing detection drives the system into a high-AI / high-loss regime,
  while sensitivity arms behave similarly to `default`.
