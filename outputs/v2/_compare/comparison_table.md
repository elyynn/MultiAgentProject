# Cross-arm comparison

Each cell: `mean +/- sem (n=N)` over per-seed final-epoch (training-tail) values. For non-default arms, the indented `delta` line is the paired mean difference vs. `default` on shared seeds (paired SE; no bootstrap p-values this iteration).

| arm | final_average_ai_usage | final_average_global_trust | final_correct_match_rate | final_overoffer_rate | final_average_company_loss |
|---|---|---|---|---|---|
| default | 0.0559 +/- 0.0050 (n=10) | 0.9916 +/- 0.0011 (n=10) | 0.6961 +/- 0.0128 (n=10) | 0.2664 +/- 0.0131 (n=10) | 0.3714 +/- 0.0156 (n=10) |
| no_detection | 0.8564 +/- 0.0183 (n=5) | 1.0000 +/- 0.0000 (n=5) | 0.1998 +/- 0.0181 (n=5) | 0.8002 +/- 0.0181 (n=5) | 1.6004 +/- 0.0363 (n=5) |
| &nbsp;&nbsp;delta vs default | +0.7962 +/- 0.0209 (n=5) | +0.0078 +/- 0.0015 (n=5) | -0.4872 +/- 0.0248 (n=5) | +0.5216 +/- 0.0222 (n=5) | +1.2208 +/- 0.0495 (n=5) |
| no_spillover | 0.1096 +/- 0.0106 (n=5) | 0.8028 +/- 0.0108 (n=5) | 0.6696 +/- 0.0201 (n=5) | 0.2950 +/- 0.0144 (n=5) | 0.4284 +/- 0.0214 (n=5) |
| &nbsp;&nbsp;delta vs default | +0.0494 +/- 0.0152 (n=5) | -0.1894 +/- 0.0108 (n=5) | -0.0174 +/- 0.0083 (n=5) | +0.0164 +/- 0.0059 (n=5) | +0.0488 +/- 0.0129 (n=5) |
| null_ai | 0.0310 +/- 0.0033 (n=5) | 0.9992 +/- 0.0001 (n=5) | 0.8126 +/- 0.0043 (n=5) | 0.1496 +/- 0.0064 (n=5) | 0.2302 +/- 0.0039 (n=5) |
| &nbsp;&nbsp;delta vs default | -0.0292 +/- 0.0068 (n=5) | +0.0070 +/- 0.0015 (n=5) | +0.1256 +/- 0.0138 (n=5) | -0.1290 +/- 0.0126 (n=5) | -0.1494 +/- 0.0216 (n=5) |
| sens_detect_lo | 0.1348 +/- 0.0074 (n=5) | 0.9791 +/- 0.0029 (n=5) | 0.6562 +/- 0.0162 (n=5) | 0.3262 +/- 0.0159 (n=5) | 0.4354 +/- 0.0213 (n=5) |
| &nbsp;&nbsp;delta vs default | +0.0746 +/- 0.0117 (n=5) | -0.0131 +/- 0.0034 (n=5) | -0.0308 +/- 0.0038 (n=5) | +0.0476 +/- 0.0025 (n=5) | +0.0558 +/- 0.0147 (n=5) |
| sens_detect_hi | 0.0322 +/- 0.0021 (n=5) | 0.9825 +/- 0.0025 (n=5) | 0.6362 +/- 0.0191 (n=5) | 0.2952 +/- 0.0166 (n=5) | 0.4454 +/- 0.0257 (n=5) |
| &nbsp;&nbsp;delta vs default | -0.0280 +/- 0.0079 (n=5) | -0.0097 +/- 0.0021 (n=5) | -0.0508 +/- 0.0047 (n=5) | +0.0166 +/- 0.0029 (n=5) | +0.0658 +/- 0.0053 (n=5) |

## Notes

- Headline metrics are training-tail (final epoch). Held-out eval pass was deferred this iteration because it requires modifying `simulation.py`/`metrics.py`.
- All seeds shared across arms 2-6 are the first five from the registered seed list `[42, 123, 2024, 7, 1337, 99, 314, 271, 8675309, 555]`; arm `default` uses all ten.
- Paired delta uses only the seeds present in both arms (typically n=5).
