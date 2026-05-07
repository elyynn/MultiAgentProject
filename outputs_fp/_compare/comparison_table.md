# FP-style Arm Comparison

CIs use Student-t critical value with df = num_training_seeds - 1.
`is_converged` requires all seeds strategy-stable AND mean r_max < threshold.

| arm | r_max_mean | r_max_max | is_converged | flips_in_window_max | ai_adoption_overall | ai_adoption_low | ai_adoption_medium | ai_adoption_high | correct_match_rate | candidate_welfare | firm_welfare | separating_index | firm_LowVerify | firm_BaseVerify | firm_HighVerify | firm_suspicion_10 | firm_suspicion_50 | firm_suspicion_90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fp_default | 0.0066 | 0.0087 | False | 40 | 0.7074 | 0.0714 | 0.9860 | 0.9632 | 0.9261 | 0.8134 | -0.0948 | 0.4259 | 0.9921 | 0.0040 | 0.0040 | 0.0248 | 0.2769 | 0.6983 |
| fp_no_detection | 0.0027 | 0.0028 | True | 0 | 0.9987 | 0.9987 | 0.9987 | 0.9985 | 0.9041 | 0.9976 | -0.1169 | 0.0002 | 0.9921 | 0.0040 | 0.0040 | 0.0041 | 0.0065 | 0.9894 |
| fp_no_reputation | 0.0023 | 0.0024 | False | 2 | 0.9983 | 0.9973 | 0.9987 | 0.9985 | 0.9048 | 0.8833 | -0.1162 | 0.0007 | 0.9921 | 0.0040 | 0.0040 | 0.0041 | 0.0071 | 0.9888 |
| fp_null_ai | 0.0018 | 0.0049 | True | 0 | 0.0014 | 0.0014 | 0.0013 | 0.0013 | 0.9683 | 0.8998 | -0.0527 | 0.0002 | 0.9921 | 0.0040 | 0.0040 | 0.9921 | 0.0039 | 0.0039 |
| fp_fixed_firm | 0.1108 | 0.1160 | False | 2 | 0.0015 | 0.0014 | 0.0013 | 0.0018 | 0.9075 | 0.8112 | -0.1725 | 0.0003 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| fp_high_verification_cost | 0.0068 | 0.0087 | False | 40 | 0.7074 | 0.0714 | 0.9860 | 0.9632 | 0.9261 | 0.8134 | -0.1158 | 0.4259 | 0.9921 | 0.0040 | 0.0040 | 0.0248 | 0.2769 | 0.6983 |
| fp_tie_high | 0.0066 | 0.0105 | False | 44 | 0.7086 | 0.0679 | 0.9875 | 0.9709 | 0.9271 | 0.8130 | -0.0938 | 0.4297 | 0.9921 | 0.0040 | 0.0040 | 0.0225 | 0.2685 | 0.7090 |
