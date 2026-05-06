# F8 - AI-conditioned detection rate by type (replaces W2)

- **Source artifact**: `outputs/v2/default/seed_*/results/interview_logs.csv`
  (filter rows: `ai_action == 1`, last 10 epochs of each seed).
- **N seeds**: 10 (`default` arm).
- **Config diff**: none (baseline).
- **What it shows**: Empirical detection probability conditioned on the AI
  having actually been used, grouped by `true_type`, averaged across seeds.
  Bars: mean +/- SEM across seeds. Dashed black tick at each bar = the value
  set in `cfg.detection_prob_by_type` (Low=0.35, Medium=0.20, High=0.10).
- **Claim supported**: Closes audit W2 - the simulator's per-type detection
  parameters *are* recovered when the right denominator is used. The pilot's
  "detection rate by type" plot used the all-candidates denominator, where
  rare AI usage made the per-type curves dominated by Monte Carlo noise.
