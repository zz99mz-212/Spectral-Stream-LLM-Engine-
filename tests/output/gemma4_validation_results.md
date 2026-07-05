# Gemma 4 E2B — Compression Validation Report
**Generated:** 2026-07-01T15:17:39.055568
**Model:** Gemma 4 E2B

## Summary
| Metric | Value |
|--------|-------|
| Total tests | 110 |
| Successful | 110 |
| Failed | 0 |
| Avg compression ratio | 26001.3x
| Avg relative error | 74.95%
| Avg SNR | 57.0 dB

## Method Ranking (Top 20 by Harmonic Mean)
| Rank | Method | Avg Harmonic Score |
|------|--------|-------------------|
| 1 | additive_codebook_quant | 1.98 |
| 2 | adaptive_scalar_quant | 1.61 |
| 3 | adaptive_sparsity | 0.34 |
| 4 | adaptive_group_quant | 0.04 |
| 5 | adaptive_arithmetic | 0.02 |

## Best Method Per Tensor Type
| Type | Lowest Error | Highest Ratio | Best Harmonic |
| attention_k | additive_codebook_quant (0.000%) | adaptive_arithmetic (60000.0x) | additive_codebook_quant |
| attention_out | additive_codebook_quant (0.000%) | adaptive_arithmetic (60000.0x) | additive_codebook_quant |
| attention_q | additive_codebook_quant (0.000%) | adaptive_arithmetic (60000.0x) | additive_codebook_quant |
| attention_v | additive_codebook_quant (0.000%) | adaptive_arithmetic (60000.0x) | additive_codebook_quant |
| embedding | additive_codebook_quant (0.000%) | adaptive_arithmetic (60000.0x) | additive_codebook_quant |
| ffn_down | additive_codebook_quant (0.000%) | adaptive_arithmetic (60000.0x) | additive_codebook_quant |
| ffn_gate | additive_codebook_quant (0.000%) | adaptive_arithmetic (60000.0x) | additive_codebook_quant |
| ffn_up | additive_codebook_quant (0.000%) | adaptive_arithmetic (60000.0x) | additive_codebook_quant |
| per_layer_proj | additive_codebook_quant (0.000%) | adaptive_arithmetic (60000.0x) | additive_codebook_quant |

## All Results

| Method | Tensor | Type | Ratio | Error(%) | SNR(dB) | PSNR(dB) | CosineSim | Time(ms) |
|--------|--------|------|-------|----------|---------|----------|-----------|----------|
| adaptive_arithmetic | aud_attn_q_0 | attention_q | 60000.0x | 100.00 | -0.0 | 15.1 | 0.0000 | 73.4 |
| adaptive_arithmetic | aud_ffn_0 | ffn_gate | 60000.0x | 100.00 | -0.3 | 13.1 | 0.0000 | 43.5 |
| adaptive_arithmetic | embed_audio | embedding | 60000.0x | 100.00 | -0.0 | 19.9 | 0.0000 | 44.1 |
| adaptive_arithmetic | embed_tokens | embedding | 60000.0x | 100.00 | -0.0 | 29.3 | 0.0000 | 35.2 |
| adaptive_arithmetic | embed_vision | embedding | 60000.0x | 100.00 | -0.0 | 25.4 | 0.0000 | 44.8 |
| adaptive_arithmetic | lm_attn_k_0 | attention_k | 60000.0x | 100.00 | -0.0 | 21.6 | 0.0000 | 46.8 |
| adaptive_arithmetic | lm_attn_o_0 | attention_out | 60000.0x | 100.00 | -0.0 | 23.7 | 0.0000 | 43.5 |
| adaptive_arithmetic | lm_attn_o_4_full | attention_out | 60000.0x | 100.00 | -0.0 | 23.6 | 0.0000 | 52.4 |
| adaptive_arithmetic | lm_attn_q_0 | attention_q | 60000.0x | 100.00 | -0.0 | 24.7 | 0.0000 | 44.3 |
| adaptive_arithmetic | lm_attn_q_4_full | attention_q | 60000.0x | 100.00 | -0.0 | 27.2 | 0.0000 | 44.5 |
| adaptive_arithmetic | lm_attn_v_0 | attention_v | 60000.0x | 100.00 | -0.0 | 21.6 | 0.0000 | 43.3 |
| adaptive_arithmetic | lm_ffn_down_0 | ffn_down | 60000.0x | 100.00 | -0.0 | 23.4 | 0.0000 | 59.4 |
| adaptive_arithmetic | lm_ffn_down_15_wide | ffn_down | 60000.0x | 100.00 | -0.0 | 20.3 | 0.0000 | 46.6 |
| adaptive_arithmetic | lm_ffn_gate_0 | ffn_gate | 60000.0x | 100.00 | -0.0 | 20.8 | 0.0000 | 52.0 |
| adaptive_arithmetic | lm_ffn_gate_15_wide | ffn_gate | 60000.0x | 100.00 | -0.0 | 21.6 | 0.0000 | 74.1 |
| adaptive_arithmetic | lm_ffn_up_0 | ffn_up | 60000.0x | 100.00 | -0.0 | 20.8 | 0.0000 | 74.3 |
| adaptive_arithmetic | lm_ffn_up_15_wide | ffn_up | 60000.0x | 100.00 | -0.0 | 21.2 | 0.0000 | 47.5 |
| adaptive_arithmetic | lm_per_layer_0 | per_layer_proj | 60000.0x | 100.00 | -0.0 | 24.4 | 0.0000 | 48.2 |
| adaptive_arithmetic | lm_per_layer_global | per_layer_proj | 60000.0x | 100.00 | -0.0 | 26.1 | 0.0000 | 47.4 |
| adaptive_arithmetic | vis_attn_o_0 | attention_out | 60000.0x | 100.00 | -0.0 | 27.6 | 0.0000 | 53.6 |
| adaptive_arithmetic | vis_attn_q_0 | attention_q | 60000.0x | 100.00 | -0.0 | 29.6 | 0.0000 | 48.1 |
| adaptive_arithmetic | vis_ffn_gate_0 | ffn_gate | 60000.0x | 100.00 | -0.0 | 23.5 | 0.0000 | 43.6 |
| adaptive_group_quant | aud_attn_q_0 | attention_q | 4.0x | 90.19 | 0.9 | 16.0 | 0.5986 | 1.9 |
| adaptive_group_quant | aud_ffn_0 | ffn_gate | 4.0x | 83.43 | 1.2 | 14.7 | 0.6212 | 1.7 |
| adaptive_group_quant | embed_audio | embedding | 4.0x | 139.61 | -2.9 | 17.0 | 0.4454 | 1.9 |
| adaptive_group_quant | embed_tokens | embedding | 4.0x | 391.14 | -11.8 | 17.4 | 0.2835 | 2.1 |
| adaptive_group_quant | embed_vision | embedding | 4.0x | 282.96 | -9.0 | 16.3 | 0.3210 | 1.9 |
| adaptive_group_quant | lm_attn_k_0 | attention_k | 4.0x | 182.90 | -5.2 | 16.4 | 0.4116 | 1.9 |
| adaptive_group_quant | lm_attn_o_0 | attention_out | 4.0x | 243.75 | -7.7 | 15.9 | 0.3381 | 1.9 |
| adaptive_group_quant | lm_attn_o_4_full | attention_out | 4.0x | 228.10 | -7.2 | 16.4 | 0.3460 | 2.0 |
| adaptive_group_quant | lm_attn_q_0 | attention_q | 4.0x | 223.18 | -7.0 | 17.8 | 0.3542 | 2.0 |
| adaptive_group_quant | lm_attn_q_4_full | attention_q | 4.0x | 301.05 | -9.6 | 17.6 | 0.2965 | 2.0 |
| adaptive_group_quant | lm_attn_v_0 | attention_v | 4.0x | 178.37 | -5.0 | 16.6 | 0.4139 | 1.8 |
| adaptive_group_quant | lm_ffn_down_0 | ffn_down | 4.0x | 217.72 | -6.8 | 16.7 | 0.3664 | 2.0 |
| adaptive_group_quant | lm_ffn_down_15_wide | ffn_down | 4.0x | 144.34 | -3.2 | 17.1 | 0.4497 | 1.9 |
| adaptive_group_quant | lm_ffn_gate_0 | ffn_gate | 4.0x | 151.05 | -3.6 | 17.3 | 0.4430 | 1.9 |
| adaptive_group_quant | lm_ffn_gate_15_wide | ffn_gate | 4.0x | 179.93 | -5.1 | 16.5 | 0.3773 | 1.9 |
| adaptive_group_quant | lm_ffn_up_0 | ffn_up | 4.0x | 170.33 | -4.6 | 16.2 | 0.4202 | 1.9 |
| adaptive_group_quant | lm_ffn_up_15_wide | ffn_up | 4.0x | 158.48 | -4.0 | 17.2 | 0.4257 | 1.8 |
| adaptive_group_quant | lm_per_layer_0 | per_layer_proj | 4.0x | 253.03 | -8.1 | 16.4 | 0.3134 | 1.9 |
| adaptive_group_quant | lm_per_layer_global | per_layer_proj | 4.0x | 264.11 | -8.4 | 17.7 | 0.3304 | 1.9 |
| adaptive_group_quant | vis_attn_o_0 | attention_out | 4.0x | 318.98 | -10.1 | 17.5 | 0.2536 | 1.9 |
| adaptive_group_quant | vis_attn_q_0 | attention_q | 4.0x | 405.01 | -12.1 | 17.4 | 0.2649 | 1.9 |
| adaptive_group_quant | vis_ffn_gate_0 | ffn_gate | 4.0x | 216.53 | -6.7 | 16.8 | 0.3511 | 1.9 |
| adaptive_scalar_quant | aud_attn_q_0 | attention_q | 30000.0x | 16.51 | 15.6 | 30.7 | 0.9870 | 144.9 |
| adaptive_scalar_quant | aud_ffn_0 | ffn_gate | 30000.0x | 13.15 | 17.3 | 30.7 | 0.9919 | 141.0 |
| adaptive_scalar_quant | embed_audio | embedding | 30000.0x | 18.09 | 14.8 | 34.8 | 0.9860 | 343.6 |
| adaptive_scalar_quant | embed_tokens | embedding | 30000.0x | 21.59 | 13.3 | 42.6 | 0.9786 | 379.3 |
| adaptive_scalar_quant | embed_vision | embedding | 30000.0x | 19.01 | 14.4 | 39.8 | 0.9841 | 343.7 |
| adaptive_scalar_quant | lm_attn_k_0 | attention_k | 30000.0x | 18.20 | 14.8 | 36.4 | 0.9858 | 340.7 |
| adaptive_scalar_quant | lm_attn_o_0 | attention_out | 30000.0x | 18.58 | 14.6 | 38.3 | 0.9853 | 352.0 |
| adaptive_scalar_quant | lm_attn_o_4_full | attention_out | 30000.0x | 18.37 | 14.7 | 38.3 | 0.9854 | 350.2 |
| adaptive_scalar_quant | lm_attn_q_0 | attention_q | 30000.0x | 19.33 | 14.3 | 39.0 | 0.9837 | 339.3 |
| adaptive_scalar_quant | lm_attn_q_4_full | attention_q | 30000.0x | 19.57 | 14.2 | 41.4 | 0.9827 | 387.0 |
| adaptive_scalar_quant | lm_attn_v_0 | attention_v | 30000.0x | 18.25 | 14.8 | 36.4 | 0.9859 | 357.0 |
| adaptive_scalar_quant | lm_ffn_down_0 | ffn_down | 30000.0x | 18.42 | 14.7 | 38.1 | 0.9853 | 335.1 |
| adaptive_scalar_quant | lm_ffn_down_15_wide | ffn_down | 30000.0x | 18.06 | 14.9 | 35.2 | 0.9861 | 339.1 |
| adaptive_scalar_quant | lm_ffn_gate_0 | ffn_gate | 30000.0x | 18.11 | 14.8 | 35.7 | 0.9861 | 332.7 |
| adaptive_scalar_quant | lm_ffn_gate_15_wide | ffn_gate | 30000.0x | 18.61 | 14.6 | 36.2 | 0.9856 | 348.2 |
| adaptive_scalar_quant | lm_ffn_up_0 | ffn_up | 30000.0x | 18.37 | 14.7 | 35.6 | 0.9858 | 334.2 |
| adaptive_scalar_quant | lm_ffn_up_15_wide | ffn_up | 30000.0x | 18.43 | 14.7 | 35.8 | 0.9858 | 346.9 |
| adaptive_scalar_quant | lm_per_layer_0 | per_layer_proj | 30000.0x | 19.14 | 14.4 | 38.8 | 0.9841 | 342.4 |
| adaptive_scalar_quant | lm_per_layer_global | per_layer_proj | 30000.0x | 22.45 | 13.0 | 39.1 | 0.9768 | 358.2 |
| adaptive_scalar_quant | vis_attn_o_0 | attention_out | 30000.0x | 24.42 | 12.2 | 39.8 | 0.9704 | 273.6 |
| adaptive_scalar_quant | vis_attn_q_0 | attention_q | 30000.0x | 22.45 | 13.0 | 42.6 | 0.9757 | 293.1 |
| adaptive_scalar_quant | vis_ffn_gate_0 | ffn_gate | 30000.0x | 18.16 | 14.8 | 38.3 | 0.9852 | 278.2 |
| adaptive_sparsity | aud_attn_q_0 | attention_q | 2.4x | 49.83 | 6.0 | 21.1 | 0.8670 | 4.8 |
| adaptive_sparsity | aud_ffn_0 | ffn_gate | 2.7x | 53.09 | 5.2 | 18.6 | 0.8474 | 4.5 |
| adaptive_sparsity | embed_audio | embedding | 2.6x | 38.43 | 8.3 | 28.2 | 0.9232 | 4.5 |
| adaptive_sparsity | embed_tokens | embedding | 2.2x | 29.49 | 10.6 | 39.9 | 0.9555 | 4.5 |
| adaptive_sparsity | embed_vision | embedding | 2.3x | 33.24 | 9.6 | 34.9 | 0.9432 | 4.5 |
| adaptive_sparsity | lm_attn_k_0 | attention_k | 2.5x | 40.94 | 7.8 | 29.4 | 0.9124 | 4.5 |
| adaptive_sparsity | lm_attn_o_0 | attention_out | 2.3x | 36.39 | 8.8 | 32.5 | 0.9314 | 4.5 |
| adaptive_sparsity | lm_attn_o_4_full | attention_out | 2.3x | 36.75 | 8.7 | 32.3 | 0.9300 | 4.5 |
| adaptive_sparsity | lm_attn_q_0 | attention_q | 2.3x | 36.12 | 8.8 | 33.6 | 0.9325 | 4.5 |
| adaptive_sparsity | lm_attn_q_4_full | attention_q | 2.1x | 29.64 | 10.6 | 37.7 | 0.9551 | 4.5 |
| adaptive_sparsity | lm_attn_v_0 | attention_v | 2.6x | 41.02 | 7.7 | 29.3 | 0.9120 | 4.5 |
| adaptive_sparsity | lm_ffn_down_0 | ffn_down | 2.4x | 36.56 | 8.7 | 32.1 | 0.9308 | 4.5 |
| adaptive_sparsity | lm_ffn_down_15_wide | ffn_down | 2.6x | 38.70 | 8.2 | 28.6 | 0.9221 | 4.5 |
| adaptive_sparsity | lm_ffn_gate_0 | ffn_gate | 2.5x | 39.14 | 8.1 | 29.0 | 0.9202 | 4.5 |
| adaptive_sparsity | lm_ffn_gate_15_wide | ffn_gate | 2.3x | 36.58 | 8.7 | 30.3 | 0.9307 | 4.5 |
| adaptive_sparsity | lm_ffn_up_0 | ffn_up | 2.6x | 40.30 | 7.9 | 28.7 | 0.9152 | 4.5 |
| adaptive_sparsity | lm_ffn_up_15_wide | ffn_up | 2.3x | 35.46 | 9.0 | 30.2 | 0.9350 | 4.5 |
| adaptive_sparsity | lm_per_layer_0 | per_layer_proj | 2.1x | 29.63 | 10.6 | 35.0 | 0.9551 | 4.6 |
| adaptive_sparsity | lm_per_layer_global | per_layer_proj | 2.3x | 33.59 | 9.5 | 35.6 | 0.9419 | 4.6 |
| adaptive_sparsity | vis_attn_o_0 | attention_out | 2.0x | 27.49 | 11.2 | 38.8 | 0.9615 | 4.6 |
| adaptive_sparsity | vis_attn_q_0 | attention_q | 2.1x | 26.49 | 11.5 | 41.1 | 0.9643 | 4.6 |
| adaptive_sparsity | vis_ffn_gate_0 | ffn_gate | 2.4x | 34.66 | 9.2 | 32.7 | 0.9380 | 4.5 |
| additive_codebook_quant | aud_attn_q_0 | attention_q | 40000.0x | 0.00 | 268.9 | 284.1 | 1.0000 | 3.3 |
| additive_codebook_quant | aud_ffn_0 | ffn_gate | 40000.0x | 0.00 | 269.0 | 282.4 | 1.0000 | 3.3 |
| additive_codebook_quant | embed_audio | embedding | 40000.0x | 0.00 | 268.0 | 287.9 | 1.0000 | 3.4 |
| additive_codebook_quant | embed_tokens | embedding | 40000.0x | 0.00 | 268.1 | 297.3 | 1.0000 | 3.4 |
| additive_codebook_quant | embed_vision | embedding | 40000.0x | 0.00 | 271.1 | 296.4 | 1.0000 | 3.3 |
| additive_codebook_quant | lm_attn_k_0 | attention_k | 40000.0x | 0.00 | 267.7 | 289.4 | 1.0000 | 3.4 |
| additive_codebook_quant | lm_attn_o_0 | attention_out | 40000.0x | 0.00 | 266.7 | 290.4 | 1.0000 | 3.3 |
| additive_codebook_quant | lm_attn_o_4_full | attention_out | 40000.0x | 0.00 | 263.6 | 287.2 | 1.0000 | 3.5 |
| additive_codebook_quant | lm_attn_q_0 | attention_q | 40000.0x | 0.00 | 267.4 | 292.1 | 1.0000 | 3.4 |
| additive_codebook_quant | lm_attn_q_4_full | attention_q | 40000.0x | 0.00 | 266.5 | 293.7 | 1.0000 | 3.3 |
| additive_codebook_quant | lm_attn_v_0 | attention_v | 40000.0x | 0.00 | 268.3 | 289.8 | 1.0000 | 3.4 |
| additive_codebook_quant | lm_ffn_down_0 | ffn_down | 40000.0x | 0.00 | 262.4 | 285.8 | 1.0000 | 6.5 |
| additive_codebook_quant | lm_ffn_down_15_wide | ffn_down | 40000.0x | 0.00 | 258.7 | 279.0 | 1.0000 | 3.4 |
| additive_codebook_quant | lm_ffn_gate_0 | ffn_gate | 40000.0x | 0.00 | 268.0 | 288.8 | 1.0000 | 3.5 |
| additive_codebook_quant | lm_ffn_gate_15_wide | ffn_gate | 40000.0x | 0.00 | 269.1 | 290.7 | 1.0000 | 6.0 |
| additive_codebook_quant | lm_ffn_up_0 | ffn_up | 40000.0x | 0.00 | 267.7 | 288.6 | 1.0000 | 5.6 |
| additive_codebook_quant | lm_ffn_up_15_wide | ffn_up | 40000.0x | 0.00 | 268.4 | 289.6 | 1.0000 | 3.4 |
| additive_codebook_quant | lm_per_layer_0 | per_layer_proj | 40000.0x | 0.00 | 276.3 | 300.7 | 1.0000 | 3.3 |
| additive_codebook_quant | lm_per_layer_global | per_layer_proj | 40000.0x | 0.00 | 268.0 | 294.2 | 1.0000 | 3.3 |
| additive_codebook_quant | vis_attn_o_0 | attention_out | 40000.0x | 0.00 | 269.6 | 297.2 | 1.0000 | 3.3 |
| additive_codebook_quant | vis_attn_q_0 | attention_q | 40000.0x | 0.00 | 271.2 | 300.8 | 1.0000 | 3.3 |
| additive_codebook_quant | vis_ffn_gate_0 | ffn_gate | 40000.0x | 0.00 | 271.1 | 294.6 | 1.0000 | 3.3 |