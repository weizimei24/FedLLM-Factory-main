# FedIT round-1 support normalization: one-shot evaluation

All metrics are macro means across 8 clients × 500 held-out rephrased questions. EM/F1/EOS/max-token rates and EM differences are percentage points.

| Condition | EM | F1 | NLL | EOS % | max-token % | mean tokens | ΔEM vs average [paired 95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| average | 4.00 | 10.15 | 2.300 | 100.0 | 0.0 | 4.08 | +0.00 [+0.00, +0.00] |
| global | 2.73 | 7.87 | 2.564 | 99.8 | 0.2 | 4.41 | -1.27 [-1.78, -0.80] |
| direct_sum | 0.12 | 1.47 | 5.207 | 23.0 | 77.0 | 25.76 | -3.88 [-4.45, -3.30] |
| support_A | 4.12 | 10.28 | 2.292 | 100.0 | 0.0 | 4.08 | +0.12 [+0.03, +0.25] |
| support_B_0.1 | 4.05 | 10.19 | 2.299 | 100.0 | 0.0 | 4.09 | +0.05 [+0.00, +0.15] |
| support_B_0.2 | 4.08 | 10.22 | 2.297 | 100.0 | 0.0 | 4.09 | +0.08 [+0.00, +0.18] |
| support_B_0.3 | 4.08 | 10.22 | 2.295 | 100.0 | 0.0 | 4.08 | +0.08 [+0.00, +0.18] |
| scaled_average_1.5 | 4.97 | 11.24 | 2.243 | 100.0 | 0.0 | 4.01 | +0.97 [+0.60, +1.35] |
| scaled_average_2 | 5.60 | 12.18 | 2.225 | 100.0 | 0.0 | 3.94 | +1.60 [+1.10, +2.10] |
| scaled_average_3 | 6.28 | 12.80 | 2.287 | 100.0 | 0.0 | 3.93 | +2.28 [+1.65, +2.93] |
| scaled_average_4 | 5.97 | 12.47 | 2.485 | 99.8 | 0.2 | 4.06 | +1.97 [+1.25, +2.67] |
| norm_matched_average | 4.12 | 10.27 | 2.294 | 100.0 | 0.0 | 4.07 | +0.12 [+0.03, +0.25] |
| shuffled_support_42 | 5.60 | 11.91 | 2.239 | 100.0 | 0.0 | 3.92 | +1.60 [+1.07, +2.10] |
| shuffled_support_43 | 5.42 | 12.16 | 2.266 | 100.0 | 0.0 | 3.93 | +1.42 [+0.85, +2.02] |
| shuffled_support_44 | 5.67 | 12.06 | 2.216 | 100.0 | 0.0 | 4.01 | +1.67 [+1.18, +2.15] |
| ties_merging | 5.42 | 11.98 | 2.213 | 100.0 | 0.0 | 4.03 | +1.42 [+0.95, +1.90] |
| pair_average_0_4 | 3.62 | 9.88 | 2.357 | 100.0 | 0.0 | 4.15 | -0.38 [-0.75, +0.00] |
| pair_sum_0_4 | 4.28 | 10.78 | 2.294 | 100.0 | 0.0 | 4.11 | +0.28 [-0.20, +0.73] |
| pair_supportA_0_4 | 3.62 | 9.88 | 2.356 | 100.0 | 0.0 | 4.15 | -0.38 [-0.75, +0.03] |
| pair_average_3_6 | 5.92 | 11.99 | 2.292 | 100.0 | 0.1 | 3.89 | +1.92 [+1.38, +2.50] |
| pair_sum_3_6 | 5.95 | 12.00 | 2.296 | 100.0 | 0.0 | 3.66 | +1.95 [+1.27, +2.68] |
| pair_supportA_3_6 | 5.90 | 11.94 | 2.291 | 100.0 | 0.1 | 3.89 | +1.90 [+1.38, +2.45] |
| rank8_average | 3.98 | 10.11 | 2.302 | 100.0 | 0.0 | 4.08 | -0.02 [-0.12, +0.05] |
| rank8_support_A | 4.12 | 10.26 | 2.295 | 100.0 | 0.0 | 4.07 | +0.12 [+0.03, +0.25] |
| rank8_scaled_average_3 | 6.25 | 12.80 | 2.282 | 100.0 | 0.0 | 3.92 | +2.25 [+1.62, +2.90] |

Best scaled_average η by macro EM: 3 (selected on these same test examples; comparison is exploratory).

## Per-layer support-A geometry

| Layer | rank | σ² energy N<2 | σ² energy 2≤N≤6 | σ² energy N>6 | ‖support_A‖/‖average‖ |
|---|---:|---:|---:|---:|---:|
| model.layers.0.self_attn.q_proj | 72 | 0.000 | 0.002 | 0.998 | 1.06 |
| model.layers.0.self_attn.v_proj | 72 | 0.000 | 0.002 | 0.998 | 1.05 |
| model.layers.1.self_attn.q_proj | 72 | 0.000 | 0.003 | 0.997 | 1.06 |
| model.layers.1.self_attn.v_proj | 72 | 0.000 | 0.004 | 0.996 | 1.08 |
| model.layers.10.self_attn.q_proj | 72 | 0.000 | 0.002 | 0.998 | 1.02 |
| model.layers.10.self_attn.v_proj | 72 | 0.000 | 0.002 | 0.998 | 1.02 |
| model.layers.11.self_attn.q_proj | 72 | 0.000 | 0.002 | 0.998 | 1.04 |
| model.layers.11.self_attn.v_proj | 72 | 0.000 | 0.002 | 0.998 | 1.03 |
| model.layers.12.self_attn.q_proj | 72 | 0.000 | 0.004 | 0.996 | 1.06 |
| model.layers.12.self_attn.v_proj | 72 | 0.000 | 0.001 | 0.999 | 1.02 |
| model.layers.13.self_attn.q_proj | 72 | 0.000 | 0.002 | 0.998 | 1.05 |
| model.layers.13.self_attn.v_proj | 72 | 0.000 | 0.001 | 0.999 | 1.03 |
| model.layers.14.self_attn.q_proj | 72 | 0.000 | 0.002 | 0.998 | 1.12 |
| model.layers.14.self_attn.v_proj | 72 | 0.000 | 0.001 | 0.998 | 1.02 |
| model.layers.15.self_attn.q_proj | 72 | 0.000 | 0.002 | 0.998 | 1.10 |
| model.layers.15.self_attn.v_proj | 72 | 0.000 | 0.001 | 0.999 | 1.02 |
| model.layers.16.self_attn.q_proj | 72 | 0.001 | 0.002 | 0.997 | 1.09 |
| model.layers.16.self_attn.v_proj | 72 | 0.000 | 0.001 | 0.999 | 1.02 |
| model.layers.17.self_attn.q_proj | 72 | 0.000 | 0.002 | 0.998 | 1.07 |
| model.layers.17.self_attn.v_proj | 72 | 0.000 | 0.001 | 0.999 | 1.03 |
| model.layers.18.self_attn.q_proj | 72 | 0.000 | 0.002 | 0.998 | 1.06 |
| model.layers.18.self_attn.v_proj | 72 | 0.000 | 0.002 | 0.998 | 1.02 |
| model.layers.19.self_attn.q_proj | 72 | 0.000 | 0.004 | 0.996 | 1.07 |
| model.layers.19.self_attn.v_proj | 72 | 0.000 | 0.002 | 0.998 | 1.02 |
| model.layers.2.self_attn.q_proj | 72 | 0.000 | 0.003 | 0.997 | 1.07 |
| model.layers.2.self_attn.v_proj | 72 | 0.000 | 0.005 | 0.995 | 1.05 |
| model.layers.20.self_attn.q_proj | 72 | 0.000 | 0.007 | 0.993 | 1.07 |
| model.layers.20.self_attn.v_proj | 72 | 0.000 | 0.002 | 0.998 | 1.04 |
| model.layers.21.self_attn.q_proj | 72 | 0.000 | 0.003 | 0.997 | 1.05 |
| model.layers.21.self_attn.v_proj | 72 | 0.000 | 0.003 | 0.997 | 1.03 |
| model.layers.22.self_attn.q_proj | 72 | 0.000 | 0.003 | 0.997 | 1.09 |
| model.layers.22.self_attn.v_proj | 72 | 0.000 | 0.004 | 0.996 | 1.06 |
| model.layers.23.self_attn.q_proj | 72 | 0.000 | 0.003 | 0.997 | 1.11 |
| model.layers.23.self_attn.v_proj | 72 | 0.000 | 0.046 | 0.954 | 1.06 |
| model.layers.24.self_attn.q_proj | 72 | 0.000 | 0.078 | 0.922 | 1.18 |
| model.layers.24.self_attn.v_proj | 72 | 0.000 | 0.008 | 0.992 | 1.04 |
| model.layers.25.self_attn.q_proj | 72 | 0.000 | 0.005 | 0.995 | 1.08 |
| model.layers.25.self_attn.v_proj | 72 | 0.000 | 0.005 | 0.995 | 1.04 |
| model.layers.26.self_attn.q_proj | 72 | 0.000 | 0.103 | 0.897 | 1.11 |
| model.layers.26.self_attn.v_proj | 72 | 0.000 | 0.004 | 0.996 | 1.03 |
| model.layers.27.self_attn.q_proj | 72 | 0.000 | 0.003 | 0.997 | 1.11 |
| model.layers.27.self_attn.v_proj | 72 | 0.000 | 0.005 | 0.994 | 1.03 |
| model.layers.3.self_attn.q_proj | 72 | 0.000 | 0.003 | 0.997 | 1.06 |
| model.layers.3.self_attn.v_proj | 72 | 0.000 | 0.003 | 0.997 | 1.09 |
| model.layers.4.self_attn.q_proj | 72 | 0.000 | 0.002 | 0.998 | 1.07 |
| model.layers.4.self_attn.v_proj | 72 | 0.000 | 0.004 | 0.996 | 1.04 |
| model.layers.5.self_attn.q_proj | 72 | 0.000 | 0.005 | 0.995 | 1.07 |
| model.layers.5.self_attn.v_proj | 72 | 0.000 | 0.003 | 0.997 | 1.04 |
| model.layers.6.self_attn.q_proj | 72 | 0.000 | 0.002 | 0.998 | 1.08 |
| model.layers.6.self_attn.v_proj | 72 | 0.000 | 0.003 | 0.997 | 1.05 |
| model.layers.7.self_attn.q_proj | 72 | 0.000 | 0.001 | 0.999 | 1.07 |
| model.layers.7.self_attn.v_proj | 72 | 0.000 | 0.002 | 0.998 | 1.03 |
| model.layers.8.self_attn.q_proj | 72 | 0.000 | 0.003 | 0.997 | 1.04 |
| model.layers.8.self_attn.v_proj | 72 | 0.000 | 0.001 | 0.999 | 1.03 |
| model.layers.9.self_attn.q_proj | 72 | 0.000 | 0.002 | 0.998 | 1.06 |
| model.layers.9.self_attn.v_proj | 72 | 0.000 | 0.001 | 0.999 | 1.03 |

## Paired combinations (all eight clients evaluated)

| Pair | mode | EM | F1 | NLL | EOS % |
|---|---|---:|---:|---:|---:|
| 0+4 | average | 3.62 | 9.88 | 2.357 | 100.0 |
| 0+4 | sum | 4.28 | 10.78 | 2.294 | 100.0 |
| 0+4 | supportA | 3.62 | 9.88 | 2.356 | 100.0 |
| 3+6 | average | 5.92 | 11.99 | 2.292 | 100.0 |
| 3+6 | sum | 5.95 | 12.00 | 2.296 | 100.0 |
| 3+6 | supportA | 5.90 | 11.94 | 2.291 | 100.0 |

## Support variants against amplification and shuffled directions

| Support | comparator | ΔEM | paired 95% CI |
|---|---|---:|---:|
| support_A | scaled_average_3 | -2.15 | [-2.80, -1.50] |
| support_A | norm_matched_average | +0.00 | [+0.00, +0.00] |
| support_A | shuffled_support_42 | -1.47 | [-1.95, -0.97] |
| support_A | shuffled_support_43 | -1.30 | [-1.90, -0.73] |
| support_A | shuffled_support_44 | -1.55 | [-2.02, -1.07] |
| support_B_0.1 | scaled_average_3 | -2.23 | [-2.90, -1.57] |
| support_B_0.1 | norm_matched_average | -0.08 | [-0.18, +0.00] |
| support_B_0.1 | shuffled_support_42 | -1.55 | [-2.05, -1.05] |
| support_B_0.1 | shuffled_support_43 | -1.38 | [-1.98, -0.83] |
| support_B_0.1 | shuffled_support_44 | -1.62 | [-2.10, -1.15] |
| support_B_0.2 | scaled_average_3 | -2.20 | [-2.88, -1.57] |
| support_B_0.2 | norm_matched_average | -0.05 | [-0.12, +0.00] |
| support_B_0.2 | shuffled_support_42 | -1.52 | [-2.02, -1.03] |
| support_B_0.2 | shuffled_support_43 | -1.35 | [-1.95, -0.78] |
| support_B_0.2 | shuffled_support_44 | -1.60 | [-2.08, -1.12] |
| support_B_0.3 | scaled_average_3 | -2.20 | [-2.88, -1.57] |
| support_B_0.3 | norm_matched_average | -0.05 | [-0.12, +0.00] |
| support_B_0.3 | shuffled_support_42 | -1.52 | [-2.02, -1.03] |
| support_B_0.3 | shuffled_support_43 | -1.35 | [-1.95, -0.78] |
| support_B_0.3 | shuffled_support_44 | -1.60 | [-2.08, -1.12] |

Bootstrap resamples individual test questions jointly across conditions (1000 resamples of 4000). Conditions and thresholds were fixed before evaluation. These are diagnostic comparisons on the same test set, not a separate tuning/confirmation split.
