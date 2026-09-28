# RLCDAlignBench Jev Distillation

Status: completed on 2026-09-27. Results are summarized in
`RLCD_DISTILLATION_RESULTS.md`; the authoritative run and adapters are at
`<run-root>/rlcd-distillation-20260927-01` on the GPU server.

## Question

Can a conventional open model learn the behavior represented by Jev's released
probabilities, and does that behavior differ from training on the benchmark's hard labels?

## Frozen data

- Source: `sumleo/RLCDAlignBench` revision
  `b5f2ea11201fc08274ef5a90bd991a8e71e3eb48`.
- Source comparison SHA-256:
  `627f328d23371e4e3134af86584dcb04c1e71c55902bd9778ba6a35577c517ed`.
- Training: 3,777 rows from all 25 official `hill-climb` benchmarks for which a
  released per-row Jev probability exists.
- Internal validation: three deterministic folds grouped by whole benchmark. No
  benchmark appears in both sides of a fold.
- Final test: all 1,180 rows from the six official `held-out` benchmarks.
  Official-label metrics use all 1,180; teacher-fidelity metrics use the 1,174 rows
  with released Jev probabilities.
- The 111 hill-climb rows without cached Jev output are excluded from every arm,
  including the hard-label arm, so training rows stay paired.
- The six missing held-out Jev outputs are never imputed.
- BeaverTails is not read during training, CV, or the official held-out evaluation.

The prepared artifacts are server-only because the states include gated harmful text:
`<run-root>/datasets/rlcd-distillation-b5f2ea1-20260927`.
Its manifest records hashes, counts, fold assignments, and the zero-overlap audit.

## Models and arms

Pinned bases:

- `Qwen/Qwen3.5-4B` at `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- `Qwen/Qwen3.5-9B` at `cc5442c03a5c0bff0bd4c6888d9a40029c637733`

For each prompt, the student probability is the sigmoid of the next-token logit
difference `logit(Yes) - logit(No)`. Training minimizes binary cross-entropy on
that same difference:

- `jev`: released Jev probability.
- `hard`: official binary benchmark label.
- `hybrid`: `0.5 * Jev probability + 0.5 * official label`.

All arms use one epoch of NF4 double-quantized QLoRA, rank 8, alpha 16, dropout
0.05, effective batch size 8, learning rate 2e-4, 3% warmup, and gradient
checkpointing. The cap is 8,192 tokens with a hard failure rather than truncation;
the observed maxima are 4,671 train and 2,642 held-out.

## Evaluation discipline

The 4B model runs three-fold benchmark-grouped CV for all three arms. This CV is a
diagnostic estimate rather than a hyperparameter search: all hyperparameters were
frozen before it started and transfer unchanged to 9B. This is the main cost-saving
trade-off; repeating all nine CV fits at 9B was skipped.

After CV, final adapters train on all 3,777 rows. The persistent manager launches
the final held-out evaluator only if both model-training lanes exit successfully.
It evaluates base, Jev, hard, and hybrid systems for both model sizes exactly once.
Primary outputs are:

- teacher KL, soft cross-entropy, MSE, and MAE on 1,174 paired rows;
- accuracy, balanced accuracy, F1, AUROC, log loss, Brier, and ECE against official
  labels, reported per benchmark and pooled with an explicit pooling warning;
- a Jev reference comparison on the identical 1,174-row paired subset.

Teacher fidelity measures imitation, not correctness. Official-label metrics answer
whether imitation helps or hurts the benchmark task.

## Launch and monitoring

The manager PID is recorded at
`<run-root>/rlcd-distillation-20260927-01.manager.pid` and its console at
`<run-root>/rlcd-distillation-20260927-01.manager.log`. Live state is in
the run directory's `status.json`; per-lane logs are under `logs/`.

The Qwen3.5 optimized linear-attention extensions are not installed, so Transformers
uses its Torch fallback. This is scientifically equivalent for the frozen protocol
but slower; the launch-time completion estimate is roughly 14–16 hours. Installing
new fused kernels mid-run was rejected because it would change the runtime stack and
add build risk after both end-to-end QLoRA smoke tests had passed.

No live Jev API calls are made by this experiment.
