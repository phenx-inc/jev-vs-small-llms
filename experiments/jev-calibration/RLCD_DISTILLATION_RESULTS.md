# RLCDAlignBench Jev Distillation Results

Completed 2026-09-27. All 15 registered training jobs and eight one-shot held-out
evaluations finished successfully. No live Jev API calls were made.

## Bottom line

Yes, the released Jev behavior is substantially distillable into ordinary Qwen3.5
QLoRA adapters on this binary detector task. Jev-target training reduced teacher KL
by 83% for 4B and 80% for 9B relative to their frozen bases. The 9B Jev student
reached Pearson correlation 0.913, Spearman 0.902, and 91.5% threshold agreement
with Jev on 1,174 untouched teacher-paired rows.

The more important result is that Jev imitation was not the best route to official
benchmark performance. Hard-label or hybrid training produced the strongest task
scores. On pooled teacher-paired rows, 9B hard-label QLoRA reached AUROC 0.8814,
log loss 0.4116, and Brier 0.1328, versus Jev's 0.8672, 0.4398, and 0.1351.
The 4B hybrid reached AUROC 0.8760, log loss 0.4225, and Brier 0.1348.

Those apparent advantages are not conclusive across domains: only six held-out
benchmarks exist. The 9B hard model beat Jev on AUROC in 5/6 benchmarks, with a
median paired delta of +0.0123, but the benchmark-bootstrap 95% interval
[-0.0346, +0.0785] crosses zero. The defensible conclusion is therefore:

> Jev's released soft probabilities carry a strong, learnable cross-benchmark
> signal, but ordinary supervised QLoRA explains most or all of the task-quality
> advantage on this benchmark family. The experiment does not show that Jev's
> hidden architecture or general typed decision behavior has been reproduced.

## Frozen evaluation

- Training: 3,777 cached-teacher rows from 25 official hill-climb benchmarks.
- Final official test: all 1,180 rows from six official held-out benchmarks.
- Teacher-paired test: 1,174 rows; the six missing Jev outputs were not imputed.
- Models: pinned Qwen3.5-4B and Qwen3.5-9B.
- Arms: Jev soft target, official hard label, and a frozen 50/50 hybrid.
- Readout: sigmoid of the next-token `Yes` minus `No` logit.
- Training: one epoch, NF4 QLoRA rank 8, effective batch 8, learning rate 2e-4.
- Internal validation: three benchmark-grouped folds for 4B; hyperparameters were
  frozen before CV and transferred unchanged to 9B.
- Final held-out predictions were produced once, after both training lanes finished.

## Held-out results

The table uses the identical 1,174-row teacher-paired slice for every system.
Teacher KL, MAE, and correlation compare the student directly with Jev; lower KL
and MAE are better.

| System | Accuracy | AUROC | Log loss ↓ | Brier ↓ | ECE-15 ↓ | Teacher KL ↓ | Teacher MAE ↓ | Pearson with Jev |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Jev reference | **0.8135** | 0.8672 | 0.4398 | 0.1351 | **0.0536** | 0 | 0 | 1.000 |
| Qwen 4B base | 0.6218 | 0.5568 | 0.7949 | 0.2610 | 0.2080 | 0.3230 | 0.2560 | 0.267 |
| Qwen 4B + Jev | 0.7828 | 0.8483 | 0.4632 | 0.1491 | 0.0682 | **0.0546** | **0.1102** | **0.886** |
| Qwen 4B + hard | 0.7726 | 0.8505 | 0.4583 | 0.1530 | 0.0703 | 0.1527 | 0.1693 | 0.754 |
| Qwen 4B + hybrid | 0.7990 | 0.8760 | 0.4225 | 0.1348 | 0.0577 | 0.0622 | 0.1119 | 0.858 |
| Qwen 9B base | 0.6414 | 0.7668 | 0.6105 | 0.2126 | 0.1749 | 0.2060 | 0.2283 | 0.650 |
| Qwen 9B + Jev | 0.8049 | 0.8524 | 0.4510 | 0.1427 | 0.0614 | **0.0418** | **0.0924** | **0.913** |
| Qwen 9B + hard | 0.8024 | **0.8814** | **0.4116** | **0.1328** | 0.0548 | 0.1458 | 0.1418 | 0.837 |
| Qwen 9B + hybrid | 0.7836 | 0.8548 | 0.4464 | 0.1454 | 0.0571 | 0.0588 | 0.1067 | 0.868 |

Against all 1,180 official rows, the ranking is unchanged: 9B hard has the best
AUROC/log loss/Brier point estimates (0.8823/0.4111/0.1325), while Jev is unavailable
for the six rows without released outputs.

## Benchmark-grouped CV

Mean pooled validation values across the three 4B folds:

| Arm | Official log loss ↓ | Official Brier ↓ | Teacher KL ↓ | Teacher MSE ↓ |
|---|---:|---:|---:|---:|
| Jev | 0.5644 | 0.1935 | **0.0867** | **0.0361** |
| Hard | 0.6138 | 0.1960 | 0.2603 | 0.0757 |
| Hybrid | **0.5471** | **0.1858** | 0.1077 | 0.0445 |

CV anticipated the qualitative 4B result: Jev targets best imitate the teacher,
while the hybrid gives the best official-label proper scores.

## Paired benchmark uncertainty

The post-hoc uncertainty analysis resamples the six benchmark-level paired metric
deltas. Positive AUROC deltas favor the student; positive log-loss deltas mean the
student has lower loss than Jev.

| System | Metric | Median paired delta | 95% benchmark-bootstrap interval | Benchmark wins |
|---|---|---:|---:|---:|
| 4B hybrid | AUROC | -0.0049 | [-0.0787, +0.0393] | 2/6 |
| 4B hybrid | Log-loss improvement | +0.0199 | [-0.0762, +0.1260] | 5/6 |
| 9B Jev | AUROC | -0.0085 | [-0.1344, +0.0081] | 2/6 |
| 9B Jev | Log-loss improvement | -0.0269 | [-0.1571, +0.0545] | 2/6 |
| 9B hard | AUROC | +0.0123 | [-0.0346, +0.0785] | 5/6 |
| 9B hard | Log-loss improvement | +0.0437 | [-0.1473, +0.2838] | 4/6 |

The intervals are deliberately conservative and coarse: six benchmark clusters are
not enough for a strong population-level superiority claim.

## Decisions, assumptions, and limitations

- Released Jev probabilities are treated as teacher behavior, not ground truth.
- Hard and hybrid arms use the same 3,777 paired rows as the Jev arm; the 111
  hill-climb rows missing Jev output were not added selectively to hard training.
- The final run used one deterministic seed. The three CV folds are domain splits,
  not independent optimization seeds. The earlier proposed three-seed success gate
  therefore cannot be claimed.
- CV was run only for 4B and transferred to 9B to control compute. No held-out
  outcome was used to tune hyperparameters.
- The post-hoc correlations and benchmark bootstrap were computed only after the
  one-shot predictions were frozen; they caused no retraining or model selection.
- Public-benchmark contamination remains possible, and the six held-out domains
  provide limited uncertainty resolution.
- This reproduces only the generic binary `misbehaves_noul` behavior. It does not
  reproduce Jev's arbitrary Choice/Score schemas, multi-question efficiency,
  architecture, synthetic training corpus, or undisclosed RLCD method.
- BeaverTails was untouched during model selection and this original run. The
  subsequent frozen evaluation is complete and reported in
  `BEAVERTAILS_DISTILLED_RESULTS.md`; no BeaverTails outcome caused retraining.

## Artifacts

- Local state-free run:
  `experiments/jev-calibration/results/rlcd-distillation-20260927-01/`
- Primary result SHA-256:
  `bfc0e6f2f5cb67d78099d372c21ae84716c2540dbd8d4a2660231f95710bbcd8`
- Post-hoc analysis: `posthoc.json` in the local run directory.
- Server run and adapters:
  `<run-root>/rlcd-distillation-20260927-01/`
- Protocol and implementation:
  `experiments/jev-calibration/RLCD_DISTILLATION.md` and
  `experiments/jev-calibration/scripts/*rlcd_distillation*.py`.

Server and local hashes match for the primary result and all eight held-out
prediction files. Large adapters remain on the GPU server; gated source text was not
copied into the local repository.
