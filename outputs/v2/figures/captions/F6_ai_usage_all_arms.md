# F6 - AI usage trajectory across arms (headline)

- **Source artifacts**: `outputs/v2/<arm>/aggregate/epoch_curves.npz`
  for arms: `default`, `null_ai`, `no_detection`, `no_spillover`,
  `sens_detect_lo`, `sens_detect_hi`.
- **N seeds**: 10 for `default`, 5 each for the five ablation arms.
- **Config diff**: each ablation flips one knob in `cfg`
  (`default` baseline; `no_detection` -> detection prob = 0;
  `no_spillover` -> trust spillover off; `null_ai` -> AI action disabled;
  `sens_detect_{lo,hi}` -> detection probabilities scaled).
- **What it shows**: AI adoption rate over training, one mean +/- 95% CI band
  per arm.
- **Claim supported**: This is the headline figure of the suite. It cleanly
  separates regimes: removing detection -> nearly everyone uses AI;
  baseline + sensitivity arms cluster low (<= 15%); removing spillover roughly
  doubles default AI use; `null_ai` is an instrumentation sanity check (~3%
  reflects the small floor introduced by epsilon-exploration before the
  no-AI mask is applied).
