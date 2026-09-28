# C-SafeQA external-transfer evaluation

Date: 2026-09-27

## Scope

This is a frozen, test-only comparison of Jev, six untouched open models, and
the six final Qwen3.5 adapters trained in the RLCDAlignBench distillation study.
No C-SafeQA outcome was used for training, model selection, prompt tuning,
calibration, threshold selection, or additional fine-tuning.

The final matrix contains:

- Jev `jev-1.13.0` through the `jevtypesafe.org` decide endpoint;
- untouched Qwen3.5-0.8B, Qwen3.5-4B, Qwen3.5-9B, Nexa-AI-4B-Instruct,
  NVIDIA Nemotron-3-Nano-4B-BF16, and Spark-X2.5-4B;
- Qwen3.5-4B and Qwen3.5-9B adapters trained with Jev soft labels, official
  hard labels, or a frozen 50/50 hybrid objective.

## Frozen data and leakage controls

C-SafeQA is pinned to commit
`5de32382c9f736f882cd4d0f5cc10477ce6147f2` under CC BY-NC 4.0. The upstream
verifier passed 37,660 rows. Sampling excluded 465 `Disputed` labels and drew
2,000 rows from 37,195 eligible `Safe`/`Unsafe` rows with seed `20260926`,
Hamilton allocation over `split × judge_label × method`, and seeded SHA-256
ranking.

The first unscored draw exposed three exact normalized prompt-response
duplicates. Protocol v2 kept the original seed and stratum quotas, retained the
globally lowest-ranked copy, and deterministically backfilled the affected
strata. The final sample contains 2,000 unique normalized prompt-response pairs
and 1,838 normalized prompt clusters.

- Sample SHA-256:
  `fb336077b7394cb9192ac78011ff8d722fa30e87c655f043fa5969681da8a1c8`
- Prepared comparison SHA-256:
  `2c2988b297c417c805d363f584b303f1fd2cb1251e4758d9bd9105678b390639`
- RLCDAlignBench train overlap: zero exact normalized and zero registered
  high-similarity lexical matches against 3,777 training rows.

The overlap check uses normalized word 3-grams and exact/substring rules. It
does not prove the absence of semantically equivalent paraphrases. Raw gated
text remains only in protected server storage.

## Coverage and scoring

All six model tokenizers fit all 2,000 rows without truncation; the largest
prompt was 3,881 tokens under a 32,768-token cap. Jev's documented 4,000-state-
character limit admits 1,931 rows and excludes 69; excluded rows were not
truncated or replaced. Local-only metrics use all 2,000 rows. The primary
all-system comparison uses the exact 1,931-row Jev intersection.

Open-model probabilities are forced-choice next-token `Yes` versus `No`
probabilities under one frozen semantic prompt and each model's native chat
template. Confidence intervals use 2,000 paired bootstrap draws over normalized
source-prompt clusters rather than individual transformed responses.

## Results

The common slice has 1,583 safe and 348 unsafe responses in 1,775 normalized
prompt clusters. Lower log loss, Brier, and ECE are better.

| System | Accuracy | Balanced acc. | AUROC | Log loss | Brier | ECE | Unsafe recall | Safe specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen 4B + hybrid | **0.8928** | 0.8505 | 0.9437 | 0.2558 | **0.0769** | 0.0564 | 0.7845 | 0.9166 |
| Qwen 9B + hard | 0.8902 | 0.7414 | 0.9428 | 0.2523 | 0.0794 | 0.0536 | 0.5086 | 0.9741 |
| Qwen 4B + Jev | 0.8897 | 0.8072 | 0.9383 | 0.2716 | 0.0799 | 0.0642 | 0.6782 | 0.9362 |
| Qwen 4B + hard | 0.8778 | **0.8649** | **0.9450** | **0.2504** | 0.0817 | 0.0561 | **0.8448** | 0.8850 |
| Qwen 9B + Jev | 0.8731 | 0.6939 | 0.9121 | 0.2845 | 0.0881 | 0.0433 | 0.4138 | 0.9741 |
| Qwen 9B + hybrid | 0.8721 | 0.6843 | 0.9098 | 0.2873 | 0.0888 | **0.0416** | 0.3908 | 0.9779 |
| **Jev** | 0.8705 | 0.6621 | 0.8428 | 0.3442 | 0.0988 | 0.0670 | 0.3362 | **0.9880** |
| Qwen 4B base | 0.8622 | 0.6750 | 0.8619 | 0.3385 | 0.1012 | 0.0523 | 0.3822 | 0.9678 |
| Qwen 9B base | 0.8622 | 0.6772 | 0.8882 | 0.3583 | 0.1032 | 0.0683 | 0.3879 | 0.9665 |
| Spark 4B base | 0.8358 | 0.6667 | 0.7445 | 0.7454 | 0.1472 | 0.1370 | 0.4023 | 0.9311 |
| Qwen 0.8B base | 0.8193 | 0.4997 | 0.5814 | 0.4767 | 0.1508 | 0.0542 | 0.0000 | 0.9994 |
| Nemotron 4B base | 0.7856 | 0.6036 | 0.5953 | 0.5391 | 0.1637 | 0.1367 | 0.3190 | 0.8882 |
| Nexa 4B base | 0.7483 | 0.6077 | 0.6074 | 1.5569 | 0.2253 | 0.2202 | 0.3879 | 0.8275 |

### Main finding

RLCDAlignBench fine-tuning transfers strongly. Every one of the six adapters
beats live Jev on AUROC, log loss, and Brier with prompt-cluster bootstrap 95%
intervals excluding zero. For example:

| Left − right | Accuracy delta | AUROC delta | Log-loss gain | Brier gain |
|---|---:|---:|---:|---:|
| Qwen 4B hybrid − Jev | +0.0223 `[+0.0057,+0.0390]` | +0.1009 `[+0.0754,+0.1260]` | +0.0883 `[+0.0563,+0.1193]` | +0.0219 `[+0.0121,+0.0315]` |
| Qwen 4B Jev-soft − Jev | +0.0192 `[+0.0031,+0.0343]` | +0.0954 `[+0.0711,+0.1202]` | +0.0726 `[+0.0426,+0.1028]` | +0.0189 `[+0.0103,+0.0269]` |
| Qwen 9B hard − Jev | +0.0197 `[+0.0072,+0.0325]` | +0.0999 `[+0.0763,+0.1250]` | +0.0919 `[+0.0715,+0.1120]` | +0.0194 `[+0.0133,+0.0256]` |

The 4B hybrid arm has the highest raw accuracy and lowest Brier score; the 4B
hard-label arm has the highest AUROC, balanced accuracy, unsafe recall, and
lowest log loss. Their AUROC and log-loss differences are unresolved, while
hybrid has significantly higher accuracy by 1.50 points and better Brier by
0.0048. At 9B, hard-label training decisively beats both Jev-soft and hybrid on
accuracy, AUROC, log loss, and Brier. Jev-soft and hybrid 9B are statistically
indistinguishable on those direct contrasts.

The result is not a simple scale story. Qwen 4B and 9B hard-label adapters are
broadly tied. The 4B hybrid significantly beats 9B hybrid, and the 4B Jev-soft
adapter beats 9B Jev-soft on accuracy, AUROC, and Brier. The most effective
student here is therefore a 4B adapter, not the larger model.

### Untouched-model comparison

Jev decisively beats untouched Nemotron, Nexa, Qwen-0.8B, and Spark on every
primary metric in this C-SafeQA slice. This reverses the BeaverTails result in
which Nemotron had stronger proper scores, demonstrating substantial dataset
dependence. Jev and Qwen-4B base are statistically unresolved across accuracy,
AUROC, log loss, and Brier. Qwen-9B base has significantly higher AUROC than
Jev, while their accuracy and proper-score differences are unresolved.

Jev's 87.1% accuracy is also prevalence-sensitive: it detects only 33.6% of
unsafe responses while correctly accepting 98.8% of safe responses. The best
4B adapters materially improve the safety trade-off: hard-label recall is
84.5% at 88.5% specificity, and hybrid recall is 78.4% at 91.7% specificity.

The adapter advantage is not confined to one transformed source. The 4B hard
adapter's AUROC is 1.000 on base and 0.913–0.964 on each transformed split;
Jev ranges from 0.720 to 0.869 on transformed splits.

Final analysis SHA-256:
`68aa5dfb08fa39f8929ea7826c8d1914a14949e737f6fc02817bb8be193f1697`.

## Jev usage and cost

Jev returned 1,931/1,931 valid responses with zero failed attempts:

- model: `jev-1.13.0`;
- input tokens: 1,754,316;
- cost at $0.42 per million input tokens: **$0.73681272**;
- active wall time: 1,037.0 seconds;
- end-to-end latency p50/p95: 2.729/3.068 seconds;
- raw-response SHA-256:
  `fac89d44cd86199c575db8378e532075d6559b460747fbf8da29ad4b163a674b`.

## Deviations and caveats

- Nemotron's remote implementation materializes full-sequence float logits. A
  batch-size-4 attempt ran out of memory after 96 saved rows on the 24 GiB GPU.
  The run resumed from those exact rows at batch size 1 and completed all 2,000.
  This is a resource-only inference change; per-row probabilities are unchanged.
- The all-system estimand excludes 69 long rows because Jev cannot accept them.
- C-SafeQA is access-controlled and non-commercial; raw examples are not copied
  into this repository.
- This is one deterministic sample and one inference prompt per system.
- Native model post-training data and native chat templates differ.
