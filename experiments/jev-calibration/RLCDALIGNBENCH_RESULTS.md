# RLCDAlignBench result: cached Jev versus Qwen3.5-4B

## Outcome

The no-Jev-key pilot is promising. The registered editorial gate returned
`clear_promise`.

On the 28 benchmarks whose cached Jev files completely cover the canonical
labeled rows (4,424 examples), Jev beat the zero-shot Qwen3.5-4B logit baseline
on 24/28 benchmark AUROCs, 21/28 log losses and 22/28 Brier scores. The primary
per-benchmark medians were:

| Metric | Cached Jev | Qwen3.5-4B | Paired Jev advantage |
|---|---:|---:|---:|
| AUROC | 0.8653 | 0.6168 | +0.1642 (benchmark bootstrap 95% interval +0.0905 to +0.2813) |
| Log loss, lower is better | 0.4535 | 0.6596 | 0.2146 lower |
| Brier, lower is better | 0.1468 | 0.2299 | 0.0810 lower |
| ECE-15, lower is better | 0.1814 | 0.2071 | descriptive medians only |
| Adaptive ECE-15, lower is better | 0.1807 | 0.2105 | descriptive medians only |

This is not merely Jev sounding more confident. It ranks examples much better
and assigns materially better probabilities under two strictly proper scoring
rules. Its probabilities are still not perfectly calibrated: median ECE is
about 0.18, so the result supports “better than this 4B zero-shot baseline,” not
“probabilities can be used as guarantees.”

## Generalization check

Most of the suite was used in the upstream study's hill-climb track. The more
important held-out slice has five complete caches / 1,136 examples:

| Held-out metric | Cached Jev median | Qwen median | Paired Jev advantage |
|---|---:|---:|---:|
| AUROC | 0.9256 | 0.8408 | +0.0952 |
| Log loss | 0.3897 | 0.6252 | 0.1336 lower |
| Brier | 0.1147 | 0.2193 | 0.0454 lower |

The held-out benchmark-bootstrap interval for the median AUROC difference is
-0.0061 to +0.4765. The direction is encouraging, but five benchmarks are too
few for a strong universal claim.

Pooled results point the same way—Jev AUROC 0.8354 versus 0.6275 and log loss
0.4980 versus 0.6861—but pooled metrics are secondary because the benchmarks
mix domains, base rates and definitions.

## Release integrity finding

The downloadable release cannot reproduce all of its own headline aggregates:

- 5,068 canonical labeled rows exist, but only 4,951 have raw cached Jev
  probabilities.
- `tensor_trust_extract` is missing 1 response, `tensor_trust_hijack` 110, and
  `bias_scene_heldout` 6, although their bundled metrics report full coverage.
- Three complete caches also disagree with their bundled metrics. Raw versus
  released AUROC is 0.6988 versus 0.7551 for `llm_aggrefact_A`, 0.7793 versus
  0.8861 for `llm_aggrefact_B`, and 0.7511 versus 0.8586 for `summedits`.

For that reason, the primary comparison recomputes both systems from the 28
complete per-example caches. It does not impute missing responses or substitute
the bundled metric when raw data disagree. A secondary all-31 view using the
release-reported Jev AUROCs is stronger (+0.1962 median AUROC difference), but
it is explicitly less auditable and is not the headline result.

## What this says about Jev

This is real evidence that Jev is more than a formatting wrapper around a small
chat model. On this alignment-detection suite, its specialized training and
decision-model interface produce substantially better ranking, proper scores
and selective behavior than a serious exact-logit 4B baseline. The result does
not reveal Jev's hidden architecture or prove a fundamentally new probability
theory; ordinary log loss is already a proper scoring rule. It shows that the
trained system's observable probabilities are useful on these tasks.

A live-key follow-up was not required to answer the core model-quality
comparison: the released row-level Jev probabilities were already sufficient.
In retrospect, replaying the same public examples through the paid API added
only operational measurements (current-version drift, latency, cost and schema
reliability), not the fresh-data validation that would have justified the
expense. Future paid replay requires a specific unanswered claim that cached
artifacts cannot address and a documented cost-benefit decision beforehand.

The first bounded live follow-up is now complete. A 100-row smoke on the
unofficial `jevtypesafe.org` host produced 100 valid `jev-1.13.0` responses,
pooled AUROC 0.7965 versus 0.5944 for Qwen, and 0.9928 live-versus-cached
probability correlation. Its wrapper accepts only string state up to 4,000
characters, however, so only 3,987 canonical rows and 27 benchmarks are
eligible. This is operational validation, not stronger scientific evidence
than the cached study's complete-cache estimand. Full details are in
[`LIVE_JEV_SMOKE_RESULTS.md`](LIVE_JEV_SMOKE_RESULTS.md).

## Audit record

- Accepted run: `rlcdalignbench-qwen35-4b-20260925-01`.
- Dataset revision: `b5f2ea11201fc08274ef5a90bd991a8e71e3eb48`.
- Upstream code: `473b72f290234a49484c4d89c03aababcd74f758`.
- Model: `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Score: exact renormalized next-token probabilities for single-token `Yes`
  (9175) and `No` (2665); no generated confidence and no calibration fit.
- Predictions: 5,068 unique benchmark/item keys, 117 null Jev values, all Qwen
  probabilities finite and within [0,1].
- Maximum prompt: 4,671 tokens under the frozen 16,384-token limit.
- Prediction SHA-256:
  `4dc8d13a31df5c2f36ef639b14e14d8facaacc990292d79a655fad0229a4b704`.
- GPU forward allocation: 387.38 seconds, excluding model load, tokenization,
  disk I/O and network; not a Jev latency comparison.
- Both GPUs were idle after completion. Eighteen local/server tests passed.
- Machine-readable artifacts:
  `results/rlcdalignbench-qwen35-4b-20260925-01/`.

## Assumptions, decisions and limitations

Assumptions:

- The cached `misbehaves_noul.noul` value is the observable Jev probability for
  the generic question.
- Canonical variant paths in `variants.csv` and embedded response sample hashes
  define the correct input set.
- A forced Yes/No next-token distribution is a strong zero-generation 4B
  analogue, though it is not Jev's undisclosed serving implementation.

Decisions:

- Complete raw caches are primary; release aggregates are secondary.
- Metrics are benchmark-level medians, with pooled results only descriptive.
- No threshold, prompt or temperature was tuned on the scored rows.
- Gated state text stays on the GPU server; local predictions contain only IDs,
  labels, probabilities and hashes.

Shortcuts and caveats:

- Only one open 4B model and one strict prompt were tested. A calibrated 4B,
  cross-encoder or supervised classifier may narrow the gap.
- Twenty-three of the 28 primary benchmarks are in the upstream hill-climb
  split; only five complete held-out benchmarks remain.
- Public-benchmark contamination is possible for both systems.
- Cached results cannot measure the live API's current version, operational
  reliability, latency, cost or parallel-question advantage.
- The benchmark bootstrap treats benchmarks as the resampling units and is
  descriptive, not a population-level confidence guarantee.

Attempted but failed or superseded:

- Initial Hugging Face access returned 403 before the user accepted the gated
  terms; it produced no predictions.
- The first launcher stopped before model loading because its own console file
  triggered the non-empty-output guard; zero predictions were produced and the
  guard was fixed.
- A first full score used legacy convenience sample files selected by name.
  The integrity check caught three AUROC failures, revealing that canonical
  paths must come from `variants.csv`. That attempt is preserved server-side as
  invalid and is excluded. The accepted run rescored all 5,068 canonical rows.

## Full live follow-up

The subsequently authorized full eligible replay is complete. On 3,987
losslessly rendered rows, live Jev beat Qwen3.5-9B—the strongest tested open
baseline—by +0.1128 median benchmark AUROC (95% benchmark-bootstrap interval
+0.0177 to +0.2313), while also improving paired log loss and Brier score. A
Qwen 0.8B/4B/9B scale curve shows steady gains with size but does not close the
gap. Full results and operational caveats are in
[`LIVE_JEV_FULL_RESULTS.md`](LIVE_JEV_FULL_RESULTS.md).
