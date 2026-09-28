# BeaverTails Frozen-Adapter Extension

Completed 2026-09-28. This closes the missing external-transfer cell in the Jev
study: all six final Qwen3.5 adapters were scored on the exact frozen 2,000-row
BeaverTails sample used by the prior seven-system comparison. No training,
prompt selection, calibration, threshold selection, or model selection used a
BeaverTails outcome. Existing Jev and untouched-model predictions were reused,
so this extension made no new Jev API calls.

## Bottom line

Every adapter improved on Jev's point estimates for AUROC, accuracy, log loss,
and Brier score. The 9B hybrid had the highest AUROC at 0.8985. The 4B students
were stronger all-around: 4B hard had the best accuracy, balanced accuracy, and
Brier score; 4B hybrid had the best log loss; and 4B Jev-target had the highest
4B AUROC.

The paired uncertainty is important. All three 4B adapters had resolved gains
over Jev in accuracy, log loss, and Brier score. The 4B Jev-target adapter also
had a resolved AUROC gain; the 4B hard and hybrid AUROC intervals crossed zero.
All three 4B adapters had resolved AUROC advantages over untouched Nemotron,
but their proper-score differences from Nemotron were mostly unresolved.

## Frozen 13-system matrix

| System | Accuracy | Balanced accuracy | AUROC | Log loss ↓ | Brier ↓ | ECE-15 ↓ | Unsafe recall | Safe specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen 4B + hard | **0.8260** | **0.8216** | 0.8946 | 0.4486 | **0.1328** | 0.0588 | **0.8510** | 0.7923 |
| Qwen 4B + hybrid | 0.8205 | 0.8211 | 0.8948 | **0.4254** | 0.1330 | 0.0707 | 0.8171 | 0.8251 |
| Qwen 4B + Jev | 0.8150 | 0.8214 | 0.8970 | 0.4306 | 0.1366 | 0.0885 | 0.7779 | 0.8650 |
| Qwen 9B + hard | 0.7990 | 0.8121 | 0.8922 | 0.5690 | 0.1620 | 0.1407 | 0.7239 | 0.9002 |
| Qwen 9B + Jev | 0.7825 | 0.8004 | 0.8932 | 0.5283 | 0.1648 | 0.1645 | 0.6794 | 0.9214 |
| Qwen 9B + hybrid | 0.7745 | 0.7945 | **0.8985** | 0.5490 | 0.1738 | 0.1923 | 0.6594 | 0.9296 |
| Jev 1.13.0 | 0.7675 | 0.7867 | 0.8868 | 0.5927 | 0.1798 | 0.1656 | 0.6568 | 0.9167 |
| Qwen 9B base | 0.7640 | 0.7847 | 0.8816 | 0.6973 | 0.1905 | 0.1780 | 0.6446 | 0.9249 |
| Nemotron-3-Nano-4B base | 0.8050 | 0.8096 | 0.8800 | 0.4362 | 0.1408 | **0.0544** | 0.7787 | 0.8404 |
| Qwen 4B base | 0.7580 | 0.7806 | 0.8792 | 0.6179 | 0.1880 | 0.1907 | 0.6280 | 0.9331 |
| Spark-X2.5-4B base | 0.7965 | 0.8050 | 0.8698 | 0.9321 | 0.1787 | 0.1646 | 0.7474 | 0.8627 |
| Nexa-AI-4B base | 0.7515 | 0.7681 | 0.8387 | 1.5212 | 0.2276 | 0.2184 | 0.6559 | 0.8803 |
| Qwen 0.8B base | 0.4260 | 0.5000 | 0.6448 | 0.8711 | 0.3296 | 0.3085 | 0.0000 | 1.0000 |

## Paired uncertainty against Jev

Intervals use 2,000 seeded bootstrap draws over the 1,768 exact-prompt clusters.
Positive values below favor the student.

| Student | AUROC gain (95% interval) | Accuracy gain | Log-loss reduction | Brier reduction |
|---|---:|---:|---:|---:|
| Qwen 4B + Jev | **0.0102 (0.0015, 0.0193)** | **0.0475 (0.0314, 0.0640)** | **0.1621 (0.1320, 0.1934)** | **0.0433 (0.0346, 0.0521)** |
| Qwen 4B + hard | 0.0078 (-0.0021, 0.0178) | **0.0585 (0.0402, 0.0765)** | **0.1441 (0.1094, 0.1827)** | **0.0471 (0.0362, 0.0586)** |
| Qwen 4B + hybrid | 0.0080 (-0.0015, 0.0169) | **0.0530 (0.0347, 0.0718)** | **0.1673 (0.1357, 0.1996)** | **0.0469 (0.0370, 0.0571)** |
| Qwen 9B + Jev | **0.0064 (0.0006, 0.0125)** | **0.0150 (0.0040, 0.0271)** | **0.0644 (0.0475, 0.0831)** | **0.0150 (0.0099, 0.0204)** |
| Qwen 9B + hard | 0.0053 (-0.0022, 0.0131) | **0.0315 (0.0177, 0.0464)** | **0.0237 (0.0014, 0.0479)** | **0.0178 (0.0109, 0.0251)** |
| Qwen 9B + hybrid | **0.0117 (0.0051, 0.0186)** | 0.0070 (-0.0065, 0.0191) | **0.0437 (0.0243, 0.0631)** | 0.0061 (-0.0002, 0.0121) |

## What changed relative to the untouched comparison

The earlier result showed that untouched Nemotron beat Jev on accuracy and
proper scores while Jev retained a small unresolved AUROC lead. Training moved
the Qwen systems materially:

- Every adapter beat its same-size base on AUROC, log loss, and Brier with
  prompt-cluster intervals excluding zero.
- All six adapters beat Nemotron on AUROC with intervals excluding zero.
- Qwen 4B + hard beat Nemotron accuracy by 2.10 points, interval 0.50 to 3.76
  points. Qwen 4B + hybrid improved Brier by 0.0078, interval 0.0002 to 0.0154.
- The larger adapters ranked well but remained worse than Nemotron on proper
  scores. Model size did not solve calibration; the 4B students were better.

No single training target dominated. Jev targets produced the best 4B AUROC;
hard labels produced the best thresholded behavior and Brier; the hybrid
produced the best log loss. This repeats the C-SafeQA conclusion that the
teacher signal is useful but not uniquely privileged supervision.

## Protocol, assumptions, and caveats

- Frozen data: exact 2,000-row sample, seed `20260926`, SHA-256
  `f9033f57ba83837853c6c0e6fd98a4b57fa8305b10025e9536a8605cc9aa9cc2`.
- Frozen students: final Qwen3.5-4B and 9B Jev/hard/hybrid rank-8 QLoRA
  adapters trained only on 3,777 RLCDAlignBench hill-climb rows.
- Readout and prompt: unchanged forced-choice next-token `Yes`/`No`
  probability with the previously registered BeaverTails safety question.
- All six adapter runs produced exactly 2,000 predictions. No failures or
  sample exclusions occurred.
- BeaverTails is public English data from 2023; pretraining or post-training
  contamination is plausible. The final adapters use one optimization seed.
- Model-native chat templates are semantically matched but not byte-identical.
- This tests generic binary response-safety detection, not Jev's Choice/Score
  schemas, multi-question sampler, architecture, or serving guarantees.

The initial 4B launch pointed at the stale experiment-specific Hugging Face
cache and failed before model load. Those zero-prediction preflight artifacts
are preserved under `failed-preflight-cache-path/`. The verified shared cache
was then used for the successful runs; neither data nor model configuration
changed.

## Artifacts

- Complete local run:
  `experiments/jev-calibration/results/beavertails-distilled-20260928-01/`
- Analysis SHA-256:
  `a2817468731e80e6bcbf8c2270fcbb4b4f8b3f81c19c47da1a05d292d539ec09`
- Analysis script: `scripts/analyze_beavertails_distilled.py`
- Two-GPU launcher: `scripts/launch_beavertails_distilled.sh`
- Server run: `<run-root>/beavertails-distilled-20260928-01/`

Server and local hashes match for the analysis and all six prediction files.
