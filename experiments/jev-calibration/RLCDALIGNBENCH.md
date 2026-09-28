# RLCDAlignBench cached-Jev comparison

## Current status

The experiment is complete. The accepted run
`rlcdalignbench-qwen35-4b-20260925-01` scored all 5,068 canonical rows and
returned the registered `clear_promise` verdict. Full interpretation and audit
details are in [`RLCDALIGNBENCH_RESULTS.md`](RLCDALIGNBENCH_RESULTS.md).

No Jev or TypeSafe API key was used. The pinned release audit found 5,068
canonical rows and 4,951
raw cached Jev probabilities. Twenty-eight benchmark caches are complete. Three
are incomplete despite their released aggregate metrics declaring full
coverage: `tensor_trust_extract` is missing 1 response,
`tensor_trust_hijack` 110, and `bias_scene_heldout` 6.

The scientific rule was fixed before the accepted Qwen scoring run:
use the 28 complete raw caches (4,424 rows) for the primary same-row AUROC and
probability comparison. Use the release-reported AUROC across all 31 benchmarks
only as secondary context. Missing cached probabilities are retained as null and
never imputed. The release inconsistency is a reported caveat, not silently
repaired.

## Frozen comparison

- Dataset revision: `b5f2ea11201fc08274ef5a90bd991a8e71e3eb48`.
- Upstream code revision: `473b72f290234a49484c4d89c03aababcd74f758`.
- Scope: the paper's 31 canonical, nondegenerate benchmarks with a reported
  `naive_noul_auroc`; 5,068 examples total.
- Jev arm: cached `misbehaves_noul.noul` probability from the release.
- Open arm: pinned Qwen3.5-4B, exact renormalized next-token `Yes`/`No` logits,
  BF16, non-thinking chat template.
- Input match: the released state plus the exact upstream
  `misbehaves_noul.instructions` text reconstructed from the response's recorded
  battery module.
- No probability fitting or threshold selection is used in the primary
  zero-shot comparison.

Primary metrics are computed per benchmark and then compared across the 28
benchmarks whose raw Jev cache covers every canonical labeled row. AUROC, log
loss, Brier score, ECE, adaptive ECE, high-confidence error and selective
accuracy are recomputed for both systems on identical examples. Pooled metrics
are secondary because benchmark base rates and failure definitions differ.
Hill-climb and held-out aggregates are separate. Release-reported AUROC versus
Qwen on all 31 benchmarks is a secondary, explicitly less-auditable view.

For complete caches, Jev AUROC and all probability metrics are recomputed from
raw rows. Every released metric file must match the public table, and any raw
cache disagreement with that metric is recorded rather than silently replaced.
The release contains three such fully covered disagreements
(`llm_aggrefact_A`, `llm_aggrefact_B`, and `summedits`), which is why raw-cache
results—not the bundled summary—are primary.
Some cached response files also retain officially excluded rows; preparation
selects the unique response file with maximal canonical-ID overlap, joins only
canonical IDs, and records response-only extras rather than treating them as
scored examples.
If a canonical item has no cached generic answer, the full-scope AUROC comparison
still uses the released aggregate for Jev and Qwen's score on all canonical rows.
Raw probability comparisons exclude the entire incomplete benchmark rather than
selecting a potentially biased subset. Missing counts and ID-set hashes are
recorded, and no value is imputed.

## Editorial promise gate

This gate decides whether paying for a later live Jev API study is warranted;
it is not a statistical significance claim.

- `clear_promise`: Jev leads median same-row AUROC across complete caches by at least 0.02; or it is no
  worse than 0.01 in AUROC while improving median log loss or Brier by at least
  0.02.
- `clear_no_promise`: Qwen leads that median AUROC by more than 0.02 and Jev
  wins neither probability metric.
- `mixed`: everything else; inspect held-out and failure-type slices before
  deciding on a live API tranche.

The report also includes a bootstrap interval over benchmarks for the median
AUROC difference. The gate thresholds were chosen for editorial usefulness,
not derived from the test outcomes.

## Reproduction

After Hugging Face access is active on the server:

```bash
cd <repository-root>
bash scripts/launch_rlcdalignbench.sh rlcdalignbench-qwen35-4b-20260925-01
```

The launcher downloads only 97 required files: four metadata files, 31
canonical variant files named by `variants.csv`, 31 cached object-response files and 31 aggregate
metric files. Gated source
text and the prepared joined data remain under `<run-root>/datasets`
on the GPU server. Result predictions contain IDs, labels and probabilities,
but not raw states.

Local and server validation:

```bash
python3 -m unittest discover -s tests -v
```

Eighteen tests currently pass. They cover metric behavior, exact score-tie
handling, prompt protocol utilities, minimal-release selection, cached-response
joining, battery identification and the promise gate.

The accepted artifact audit found 5,068 unique prediction keys, 117 null Jev
values, finite Qwen probabilities, no gated state text in predictions and a
maximum prompt length of 4,671 tokens. Both GPUs were idle after completion.

## Assumptions, decisions, shortcuts and blockers

Assumptions:

- The released cached `misbehaves_noul` probability is the correct observable
  Jev comparator for the paper's headline generic-question result.
- A forced `Yes`/`No` logit distribution is the most serious zero-generation
  4B analogue of Jev Noul.
- The upstream repository and pinned Hugging Face revision are mutually
  consistent; hashes and published AUROCs are checked before accepting results.

Decisions:

- Use only the 31 nondegenerate benchmarks included in the paper's median,
  rather than inflating the sample with single-class or tiny-minority datasets.
- Compare raw zero-shot probabilities first. Any local temperature or threshold
  fitting would be a separately labelled analysis.
- Keep gated state text on the server and sync only non-text predictions and
  aggregate metrics.

Shortcuts/trade-offs:

- The open baseline asks one generic question per state and therefore does not
  test Jev's many-questions-per-call latency advantage.
- Qwen receives a small strict-binary wrapper around the exact upstream question;
  Jev's hidden serving prompt is unavailable, so prompt plumbing cannot be
  perfectly identical.
- Public benchmark contamination remains possible for both systems.

Resolved blocker:

- The initial unauthenticated download failed with `GatedRepoError`. The user
  accepted the dataset terms and authenticated the server; access now succeeds.
  The token is not stored in experiment artifacts.
