# Full Jev model-matrix result

## Outcome

The comparison is blog-worthy. On the losslessly rendered short-state slice of
RLCDAlignBench, community-hosted Jev produced better ranking and better proper
probability scores than every tested open model from 0.8B through 9B.

The primary estimand is the median across the 26 benchmarks containing both
labels. Pooled metrics are secondary because the suite mixes tasks and base
rates.

| System | Parameters | Median benchmark AUROC | Pooled AUROC | Pooled log loss ↓ | Pooled Brier ↓ | Pooled ECE-15 ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Live Jev 1.13.0 | undisclosed | **0.8873** | **0.8416** | **0.4937** | **0.1605** | 0.0654 |
| Qwen3.5-9B | 9B | 0.7239 | 0.7451 | 0.5841 | 0.2012 | **0.0597** |
| Spark-X2.5-4B | 4.11B | 0.6669 | 0.6334 | 1.2648 | 0.3330 | 0.2911 |
| Nexa-AI-4B-Instruct | 4.02B | 0.6012 | 0.5771 | 1.9054 | 0.3709 | 0.3579 |
| Qwen3.5-4B | 4B | 0.5943 | 0.6014 | 0.7015 | 0.2501 | 0.1584 |
| Nemotron-3-Nano-4B | 3.97B | 0.5716 | 0.5964 | 0.7149 | 0.2561 | 0.1330 |
| Qwen3.5-0.8B | 0.8B | 0.5237 | 0.5593 | 0.8622 | 0.2969 | 0.2452 |

The ECE exception matters: Qwen3.5-9B has slightly lower pooled ECE than Jev.
Jev's result is not “best on every scalar.” Its stronger evidence is the joint
improvement in discrimination and the strictly proper log-loss and Brier
scores. Median benchmark ECE still favors Jev, 0.1759 versus 0.1898 for
Qwen3.5-9B.

Against the strongest open baseline, Qwen3.5-9B:

- median paired AUROC advantage: +0.1128, with benchmark-bootstrap 95%
  interval +0.0177 to +0.2313;
- AUROC wins/ties/losses: 19/1/6;
- median paired log-loss improvement: 0.1153, interval 0.0485 to 0.1630;
- median paired Brier improvement: 0.0502, interval 0.0185 to 0.0732.

Against the incumbent Qwen3.5-4B, Jev's median AUROC advantage is +0.1948
(interval +0.1025 to +0.3418), with 23 wins and three losses. The AUROC,
log-loss and Brier bootstrap intervals exclude zero against every tested open
baseline.

## The useful scale result

The same-family Qwen scale curve makes the result easier to interpret:

| Qwen size | Median benchmark AUROC | Pooled AUROC | Pooled log loss ↓ |
|---:|---:|---:|---:|
| 0.8B | 0.5237 | 0.5593 | 0.8622 |
| 4B | 0.5943 | 0.6014 | 0.7015 |
| 9B | 0.7239 | 0.7451 | 0.5841 |
| Jev 1.13.0 | **0.8873** | **0.8416** | **0.4937** |

Scale clearly helps, so the 4B comparison was not a straw man. It also does
not close the gap by 9B. This is consistent with specialized decision training
being useful, but it does not identify Jev's hidden architecture or prove that
its method is fundamentally different from ordinary supervised learning.

## Held-out and temperature-scaling control

There are only six eligible upstream held-out benchmarks / 1,075 rows. On
their pooled rows, raw Jev has AUROC 0.8661, log loss 0.4436 and Brier 0.1355.
Raw Qwen3.5-9B has 0.7579, 0.5517 and 0.1842.

A positive scalar temperature was fitted separately for each system on the
2,912 upstream hill-climb rows, then frozen for the held-out rows. Qwen3.5-9B's
held-out log loss improves to 0.5406 and Brier to 0.1824; AUROC is unchanged by
monotone scaling. Jev remains better without needing the fitted temperature.
Applying its fitted temperature slightly worsens its held-out proper scores.

This control weakens the explanation that the result is only a globally wrong
confidence scale. It is not a powered generalization claim: across the six
held-out benchmarks Jev and Qwen3.5-9B each win three AUROCs, and the median
paired AUROC-difference bootstrap interval is -0.0171 to +0.2614.

## What Jev returned

The `/v1/decide` response is a typed JSON decision, not prose. Each preserved
state-free response contains only `model`, `answers`, `usage`, and `meta`.
The requested `misbehaves_noul` answer is a numeric probability. This run
reported model `jev-1.13.0`, provider `workers-ai`, and 99 distinct probability
values across 3,987 answers.

The evaluator question and its choice criteria are supplied by the caller.
Labels such as department, billing, bug, or account are therefore configured
in the request rather than hard-coded outputs that Jev always returns.

## Live/cached agreement

On 3,870 rows with released cached Jev values:

- probability Pearson correlation: 0.9888;
- median absolute change: 0.0100;
- mean absolute change: 0.0273;
- maximum absolute change: 0.3900;
- decision disagreement at 0.5: 2.92%;
- live/cached pooled AUROC: 0.8418/0.8414.

This supports serving-lineage continuity, but the requests are not byte
identical. The release used object state, while the unofficial community host
accepts only a rendered string.

Retrospective decision: this agreement check and the operational measurements
below did not require a full 3,987-row paid replay. The released cache had
already answered the central quality comparison, and the live run reused the
same public benchmark rather than supplying independent validation. The user
therefore considers the full replay wasted effort. For future work, cached
row-level outputs are the default; a paid replay requires a preregistered claim
that cached artifacts cannot answer, preferably on fresh data, plus an explicit
cost-benefit justification.

## Cost, latency, and reliability

- Eligible items: 3,987; final unique successes: 3,987.
- Input tokens: 2,503,309; output tokens: 0.
- Cost at the community pack rate of $0.42/M input tokens: **$1.05139**.
- Active wall time: 5,631.05 seconds, or 93.85 minutes, at four workers and a
  conservative 50 requests/minute limit.
- End-to-end request latency: 2.722 seconds p50 and 3.440 seconds p95.
- HTTP attempts: 4,586 for 3,987 completed items, or 1.15 attempts/item.
- The original invocation completed 3,959 items. Twenty-eight items exhausted
  six retries with HTTP 502; a resume completed all 28.
- No response-schema failure occurred. Exact raw-response hashes, prediction
  links, probabilities and usage counts all verify.

This operational result belongs to `jevtypesafe.org`, which describes itself
as an unofficial community host. It must not be attributed to TypeSafe AI's
direct production service.

## Protocol and model pins

All systems score the same 3,987 benchmark/item keys with selection SHA-256
`f974388d9a652b3f58d1a21a50ee8f5546390c07d2c87bf96e68f65b30aff35e`.
Open models receive the same semantic system/user prompt through their native
chat templates. Their probability is the exact next-token distribution
renormalized over the single-token answers `Yes` and `No`; they generate no
text and receive no evaluation-time calibration unless explicitly stated in
the separate split-calibration control.

Pinned models:

- `Qwen/Qwen3.5-0.8B` at
  `c6046cd1f7e2763bf1abf5a5aef6ad7878e10ccb`;
- `Qwen/Qwen3.5-4B` at
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`;
- `Qwen/Qwen3.5-9B` at
  `cc5442c03a5c0bff0bd4c6888d9a40029c637733`;
- `XHToken/Spark-X2.5-4B` at
  `0bcb35678590218655dff3765b9e61c83b35e9c4`;
- `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` at
  `dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f`;
- `Neura-Tech-AI/Nexa-AI-4B-Instruct` at
  `a336431670e6ff71462ce32b84e5a6a047e2cb01`.

Qwen and Nexa used Torch 2.11.0 / Transformers 5.14.1. Spark and Nemotron used
Torch 2.11.0 with an isolated Transformers 4.57.6 compatibility overlay.
Spark and Nemotron's pinned remote-code modules were inspected before loading;
no filesystem, network, subprocess, `eval`, or `exec` behavior was found.

Model-card claims are background, not measured evidence. Primary sources are
the [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B),
[Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B),
[Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B),
[Spark-X2.5-4B](https://huggingface.co/XHToken/Spark-X2.5-4B),
[Nemotron-3-Nano-4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16),
and [Nexa-AI-4B](https://huggingface.co/Neura-Tech-AI/Nexa-AI-4B-Instruct)
cards.

## Integrity record

- Live prediction records: 4,015 = 3,987 successes + 28 retained HTTP-502
  attempt records.
- Raw successful response envelopes: 3,987.
- Live predictions SHA-256:
  `8674a12862c7818f8be7f85d2ba0564456a9c43e126ff46406f21bd5d89f5881`.
- Raw responses SHA-256:
  `729105872e99bf8400461d44f4d5ff3f4dee4c921b299156956e0006360365cb`.
- Seven-system matrix SHA-256:
  `398a56b517aaa25c655284b6860dd340d629e75f28f7d5a67bcd398210353f96`.
- Gated state/question text is absent from all copied prediction, response and
  comparison artifacts.
- Thirty-three local protocol/metric tests pass.
- Final matrix: `results/model-matrix-20260926-02/`.
- Live artifacts: `results/live-jev-full-eligible-20260926-01/`.

## Assumptions, decisions, shortcuts, and failures

Assumptions:

- The community host accurately reports its model, provider, usage and price;
  there is no direct upstream attestation.
- Lossless canonical rendering preserves the information available to the
  model on rows within the string limit.
- Benchmark-level bootstrap intervals are descriptive for this fixed suite,
  not population guarantees.

Decisions and trade-offs:

- Rows over 4,000 rendered characters are excluded, never truncated. This
  removes 1,081/5,068 rows, touches 22 benchmarks and entirely removes four.
- Per-benchmark medians are primary; pooled metrics are reported for context.
- Qwen3.5-0.8B and 9B provide informative scale endpoints. A redundant 2B run
  was skipped.
- Fara and Goose were excluded because they are specialized computer-use/agent
  models, not like-for-like general decision baselines.
- Model-native chat templates are retained. Prompt meaning is matched, but
  token sequences and template defaults are not byte-identical across models.

Limitations:

- This is a public benchmark and may be contaminated for any tested model.
- Jev is evaluated through an unofficial wrapper with an undisclosed internal
  architecture; the experiment tests observable output quality, not mechanism.
- The open baselines are zero-shot forced-choice LMs, not trained
  discriminative classifiers. A strong supervised classifier remains the most
  important omitted comparator.
- Only six eligible held-out benchmarks remain, so the generalization claim is
  much weaker than the full fixed-suite result.
- Local GPU forward/scoring times exclude different components than remote API
  latency and must not be used as an end-to-end speed comparison.

Engineering failures and resolutions:

- Spark required Transformers 4.57.6 because its remote code is incompatible
  with the shared 5.14 tied-weight API. A wrong cache variable and a log-file
  output-guard collision produced zero-prediction attempts first.
- Nexa's tokenizer failed under the 4.57.6 overlay and succeeded under shared
  Transformers 5.14.1.
- Nemotron initially lacked `mamba_ssm`; a causal-convolution build first
  failed to find CUDA headers. An isolated overlay with explicitly located
  CUDA headers, `causal-conv1d` 1.7.0, `mamba-ssm` 2.2.6.post3 and `einops`
  completed successfully. A cached CPU-only Mamba wheel was rejected because
  it lacked `selective_scan_cuda`.
- The shared model environment initially lacked `pip`; `ensurepip` added it.
  All model-specific compatibility packages remained in the isolated overlay.
- The first Qwen3.5-0.8B smoke used `HF_HOME` for a cache created directly with
  `cache_dir`; switching to `HF_HUB_CACHE` fixed it. Zero predictions were
  produced by the failed smoke.
- The first full Jev invocation retained 28 HTTP-502 records. Resume semantics
  correctly replayed only missing keys. The matrix analyzer was then fixed to
  preserve those failed attempts for audit while scoring only unique successes.
- Resume originally overwrote active-wall metadata. The original 5,595.98-s
  status was preserved in the console log, combined with the 35.07-s resume,
  and the runner was patched so future resumes accumulate timing.
