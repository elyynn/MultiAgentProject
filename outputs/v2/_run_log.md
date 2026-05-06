# Minimal experiment suite - run log

- **Suite start (UTC)**: 2026-05-05T23:37:35.691677+00:00
- **Suite end   (UTC)**: 2026-05-05T23:39:20.257045+00:00
- **Wall-clock span**: 104.6 s
- **Sum of per-run wall-clocks**: 103.9 s (sequential execution, so ~equal to span)
- **Runs**: 35/35 ok (0 failed)

## Reproducibility

- git SHA: `b0ef95c1ec5ec9107d1efd468416acaabc5c8728`
- Python: `3.13.2 | packaged by Anaconda, Inc. | (main, Feb  6 2025, 18:49:14) [MSC v.1929 64 bit (AMD64)]`
- numpy: `2.4.3`
- hostname: `DESKTOP-A543KCH`
- log generated (UTC): `2026-05-05T23:41:30.149671+00:00`

## Status table (arm x seed)

| arm | seed | status | wall_clock_s | start (UTC) |
|---|---|---|---|---|
| default | 7 | ok | 2.70 | 2026-05-05T23:37:42.142615+00:00 |
| default | 42 | ok | 2.03 | 2026-05-05T23:37:35.691677+00:00 |
| default | 99 | ok | 3.59 | 2026-05-05T23:37:48.270415+00:00 |
| default | 123 | ok | 2.23 | 2026-05-05T23:37:37.736634+00:00 |
| default | 271 | ok | 3.43 | 2026-05-05T23:37:55.369731+00:00 |
| default | 314 | ok | 3.47 | 2026-05-05T23:37:51.883498+00:00 |
| default | 555 | ok | 3.43 | 2026-05-05T23:38:02.000657+00:00 |
| default | 1337 | ok | 3.37 | 2026-05-05T23:37:44.872262+00:00 |
| default | 2024 | ok | 2.15 | 2026-05-05T23:37:39.976574+00:00 |
| default | 8675309 | ok | 3.16 | 2026-05-05T23:37:58.823849+00:00 |
| no_detection | 7 | ok | 3.00 | 2026-05-05T23:38:15.137797+00:00 |
| no_detection | 42 | ok | 3.16 | 2026-05-05T23:38:05.451915+00:00 |
| no_detection | 123 | ok | 3.48 | 2026-05-05T23:38:08.631070+00:00 |
| no_detection | 1337 | ok | 3.29 | 2026-05-05T23:38:18.154826+00:00 |
| no_detection | 2024 | ok | 2.99 | 2026-05-05T23:38:12.137895+00:00 |
| no_spillover | 7 | ok | 3.30 | 2026-05-05T23:38:32.030143+00:00 |
| no_spillover | 42 | ok | 3.25 | 2026-05-05T23:38:21.457773+00:00 |
| no_spillover | 123 | ok | 3.56 | 2026-05-05T23:38:24.723460+00:00 |
| no_spillover | 1337 | ok | 3.26 | 2026-05-05T23:38:35.348775+00:00 |
| no_spillover | 2024 | ok | 3.70 | 2026-05-05T23:38:28.303858+00:00 |
| null_ai | 7 | ok | 3.08 | 2026-05-05T23:38:48.423193+00:00 |
| null_ai | 42 | ok | 3.34 | 2026-05-05T23:38:38.629165+00:00 |
| null_ai | 123 | ok | 3.04 | 2026-05-05T23:38:41.980881+00:00 |
| null_ai | 1337 | ok | 3.51 | 2026-05-05T23:38:51.531959+00:00 |
| null_ai | 2024 | ok | 3.37 | 2026-05-05T23:38:45.032451+00:00 |
| sens_detect_lo | 7 | ok | 3.14 | 2026-05-05T23:39:04.979214+00:00 |
| sens_detect_lo | 42 | ok | 3.12 | 2026-05-05T23:38:55.058113+00:00 |
| sens_detect_lo | 123 | ok | 3.47 | 2026-05-05T23:38:58.195059+00:00 |
| sens_detect_lo | 1337 | ok | 2.38 | 2026-05-05T23:39:08.139317+00:00 |
| sens_detect_lo | 2024 | ok | 3.27 | 2026-05-05T23:39:01.691704+00:00 |
| sens_detect_hi | 7 | ok | 1.93 | 2026-05-05T23:39:16.294947+00:00 |
| sens_detect_hi | 42 | ok | 1.89 | 2026-05-05T23:39:10.534519+00:00 |
| sens_detect_hi | 123 | ok | 1.90 | 2026-05-05T23:39:12.436088+00:00 |
| sens_detect_hi | 1337 | ok | 2.02 | 2026-05-05T23:39:18.236288+00:00 |
| sens_detect_hi | 2024 | ok | 1.93 | 2026-05-05T23:39:14.353390+00:00 |

## Per-arm wall-clock summary

| arm | n_runs | total_s | mean_s | min_s | max_s |
|---|---|---|---|---|---|
| default | 10 | 29.56 | 2.96 | 2.03 | 3.59 |
| no_detection | 5 | 15.92 | 3.18 | 2.99 | 3.48 |
| no_spillover | 5 | 17.07 | 3.41 | 3.25 | 3.70 |
| null_ai | 5 | 16.33 | 3.27 | 3.04 | 3.51 |
| sens_detect_lo | 5 | 15.37 | 3.07 | 2.38 | 3.47 |
| sens_detect_hi | 5 | 9.67 | 1.93 | 1.89 | 2.02 |

## Failures

None. All 35 runs completed successfully.

## Deferred items

- **Held-out eval pass not run.** The plan calls for freezing candidate Q-values, setting `epsilon=0`, and running 10 additional epochs to emit `final_summary_eval.json`. Implementing this requires changes to `simulation.py` and `metrics.py`, which the experiment-runner role is not permitted to modify. Headline metrics in this iteration are training-tail (final-epoch) values from `final_summary.json`.
- **Per-run figures not generated.** Per the plan, figures are emitted once per arm by the figure-curator (separate role), not by `run_simulation` for each seed. `run_suite.py` does not call `generate_all_plots`.
- **Paired-bootstrap p-values not computed.** Cross-arm comparison reports paired mean differences with paired SE (n=5 shared seeds), but no p-values. Bootstrap can be added without simulation-side changes; deferred for a subsequent iteration alongside the held-out eval pass.
- **Multi-arm overlay figures (`comparison_figures/`) not produced.** Same reason as per-run figures: figure-curator job. The aggregator emits `epoch_curves.npz` per arm, which is the input that role consumes.

## Sanity-check observations

Headline numbers move in the directions the plan predicted:

- `no_detection` -> AI usage explodes from ~0.06 (default) to ~0.86; correct-match rate collapses from ~0.70 to ~0.20; company loss roughly 4x. Detection is doing real work in the default arm.
- `no_spillover` -> AI usage and matching are nearly unchanged vs. default, but global trust drops from ~0.99 to ~0.80. Spillover is the channel keeping average global trust pinned high, but is not by itself the deterrent on AI use.
- `null_ai` -> AI usage drifts to ~0.03 (exploration noise floor), correct-match rate is highest (~0.81). Confirms the floor against which default results should be compared.
- `sens_detect_lo` (half detection probs) -> AI usage ~0.13, more than 2x the default but still far below the no-detection ceiling.
- `sens_detect_hi` (double detection probs) -> AI usage indistinguishable from default (~0.03), suggesting the qualitative story is robust to increasing detection past the default level (default is already on the saturated side of the deterrent curve).

See `outputs/v2/_compare/comparison_table.md` for the full table and paired deltas.

## Artifact map

- Per-run: `outputs/v2/<arm>/seed_<n>/results/{epoch_metrics.csv, interview_logs.csv, final_summary.json}` and `outputs/v2/<arm>/seed_<n>/run_manifest.json`.
- Per-arm aggregate: `outputs/v2/<arm>/aggregate/{summary.json, per_seed_summary.csv, epoch_curves.npz}`.
- Cross-arm: `outputs/v2/_compare/comparison_table.md`.
- This log: `outputs/v2/_run_log.md`.

Driver scripts (under `experiments/`): `run_suite.py`, `aggregate.py`, `compare_arms.py`, `write_run_log.py`.
