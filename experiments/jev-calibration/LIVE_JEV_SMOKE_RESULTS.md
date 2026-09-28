# Community-hosted Jev live smoke

## Result

The bounded smoke is complete. Run `live-jev-smoke-20260926-03` obtained 100
valid responses from model `jev-1.13.0`, with zero API or schema failures. On
this deterministic, benchmark-stratified sample, live Jev was substantially
better than the frozen Qwen3.5-4B baseline on pooled discrimination and proper
scores:

| Metric | Live Jev 1.13.0 | Qwen3.5-4B | Live Jev advantage |
|---|---:|---:|---:|
| Accuracy at 0.5 | 0.7300 | 0.5500 | +0.1800 |
| Balanced accuracy | 0.7125 | 0.5042 | +0.2083 |
| AUROC | 0.7965 | 0.5944 | +0.2021 |
| Log loss | 0.5602 | 0.7219 | 0.1617 lower |
| Brier score | 0.1869 | 0.2565 | 0.0697 lower |
| ECE, 15 bins | 0.1408 | 0.1639 | 0.0231 lower |

This is an encouraging engineering smoke, not a confirmatory benchmark result.
Only three or four rows were sampled from most benchmarks. Among the 26
benchmarks with both classes in the smoke, the median per-benchmark AUROC
difference was 0.0, with 11 live-Jev wins, 11 ties and four Qwen wins. The
pooled result mixes domains and base rates and should not replace a full
per-benchmark analysis.

## Current-model drift

Ninety-nine smoke rows have a released cached Jev probability. The live and
cached predictions are very close:

- Pearson probability correlation: 0.9928.
- Median absolute probability change: 0.0100.
- Mean absolute probability change: 0.0222.
- Maximum absolute probability change: 0.2000.
- Threshold-decision disagreement at 0.5: 1/99 (1.01%).
- Live versus cached AUROC on the same 99 rows: 0.7983 versus 0.7929.
- Live versus cached log loss: 0.5566 versus 0.5654.
- Live versus cached Brier: 0.1851 versus 0.1883.

The live request is not a byte-identical replay of the cached request: the
release used structured object state, while the community API accepts only a
string. The runner uses the upstream canonical object-to-text rendering and
never truncates. The close output agreement is therefore useful evidence, but
not proof, that the hosted model is the same serving lineage as the cached Jev
release.

## Scope imposed by the hosted API

The supplied endpoint is `https://jevtypesafe.org/api/v1/decide`. Its own
[documentation](https://jevtypesafe.org/docs/api/) says it is an unofficial
community host, not TypeSafe AI, and limits `state` to a 4,000-character
string. Of the 5,068 canonical rows:

- 3,987 rows can be rendered losslessly within the limit.
- 1,081 rows across 22 benchmarks exceed the limit and are excluded, never
  truncated.
- Four benchmarks have no eligible row: `bias_refletter`,
  `mask_continuation`, `mask_disinformation`, and `mask_factual`.
- The 100-row smoke therefore covers 27 benchmarks, selecting both labels per
  benchmark when available and filling the remaining slots by a seeded hash.

This restriction creates selection bias, particularly for long-context
benchmarks such as `faith_mt_claimcheck` and `faith_mt_grounded`. A full run on
this host could only support a claim about the 3,987-row short-state subset,
not the complete RLCDAlignBench suite.

## Reliability, latency and cost

- Successful responses: 100/100; output/schema failures: 0.
- Model reported: `jev-1.13.0`; provider metadata: `workers-ai`.
- Input tokens: 68,876; output tokens: 0.
- Smoke cost at this host's $0.42/M-token pack rate: $0.02893.
- Remaining prepaid balance after the smoke and two 322-token diagnostic
  calls: 11,840,480 tokens.
- End-to-end request latency: 2.741 s median, 3.007 s p95.
- Server-reported latency: 1.373 s median, 1.506 s p95.
- Wall time at the deliberately conservative 50 requests/minute rate: 127.3 s.

The community host's [pricing page](https://jevtypesafe.org/pricing/) lists
$0.42/M input tokens for the $5 pack and free output tokens. This is ten times
the $0.042/M direct-API rate advertised by TypeSafe, so the two services must
not be conflated in cost comparisons.

Scaling the observed token-to-rendered-character ratio to all 3,987 eligible
rows projected roughly 2.38 million input tokens, about $1.00 at this pack
rate. At the time of this smoke, no full run was authorized. The later
authorized replay used 2.503 million tokens and cost $1.05139; see
[`LIVE_JEV_FULL_RESULTS.md`](LIVE_JEV_FULL_RESULTS.md).

## Integrity record

- Selection SHA-256:
  `847c1bcfd2a084e3e9600230adb85418221542577708a42f4ff5aa375477dcc5`.
- Prediction SHA-256:
  `6e82693bc323deb03ad2fac97431419fbe0ba4219e47b2af7b6ce8173944f3e2`.
- Artifacts contain 100 unique benchmark/item keys, finite probabilities in
  [0,1], and no `state` or `question` fields.
- Twenty-five local tests pass.
- State-free machine-readable artifacts are in
  `results/live-jev-smoke-20260926-03/`.

## Assumptions, decisions, shortcuts and failed attempts

Assumptions:

- The community host's reported `model`, usage and provider metadata are
  accurate; there is no direct cryptographic attestation of the upstream model.
- The canonical text renderer preserves the information in eligible object
  states even though the serving representation differs from the cached run.
- The site's displayed prepaid rate is the applicable marginal cost for this
  key.

Decisions:

- Treat `jevtypesafe.org` as a distinct community-hosted service in every
  result and cost claim, never as TypeSafe's direct production API.
- Exclude over-limit rows rather than truncate or summarize them.
- Stop after the explicitly authorized 100-row smoke; do not infer permission
  for the eligible full replay.

Shortcuts and caveats:

- The smoke is too small for reliable per-benchmark estimates or a definitive
  blog headline.
- The sample is stratified rather than proportional, so pooled metrics describe
  these 100 rows only.
- Qwen and live Jev see the same rendered state content and generic evaluator
  question, but their serving wrappers are not identical.
- Public-benchmark contamination remains possible for both systems.

Attempted but failed or superseded:

- The first preflight used TypeSafe's direct `api.typesafe.ai` endpoint and
  returned 401 because the supplied `jv_live_` key belongs to the community
  host. No benchmark request or token usage occurred.
- A second attempt sent object state to the corrected host. Eighty requests
  returned HTTP 400 with zero token use before the run was stopped. The docs
  revealed the string-only, 4,000-character contract.
- The corrected launch was paused until the user explicitly authorized the
  unofficial third-party destination. The accepted `-03` run is the only live
  benchmark smoke used in this report.
