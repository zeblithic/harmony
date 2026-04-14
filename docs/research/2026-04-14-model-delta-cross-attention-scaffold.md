# Model delta scaffold — cross-attention to memory (ZEB-117)

**Date:** 2026-04-14
**Status:** scaffold (training/code only; runs deferred pending gamma 20k results)
**Parent:** `2026-04-14-engram-injection-mechanism-findings.md`

## Purpose

Scaffold the Model delta implementation so it can be kicked off immediately
if Model gamma clears the ZEB-117 top decision gate
(gamma val_loss < beta val_loss - 0.05 nats at step 20000). This document
records the specific design choices made during scaffolding and notes
where they diverge from published cross-attention-to-memory architectures.

## Architecture summary

Per-position cross-attention block attached at
`config.engram_injection_layer`:

```text
h_in  --[W_q]--> Q         (hidden_dim -> hidden_dim)
                   \
retrieved top-k ---[W_k]--> K    (engram_dim -> hidden_dim, k neighbors)
          |       \
          +------[W_v]--> V    (engram_dim -> hidden_dim, k neighbors)
                            \
    scores = Q.K / sqrt(d) + alpha * topk_sims    (k-dim softmax)
    out    = attn . V
    h_out  = h_in + W_o(out)                       (W_o zero-init)
```

Key decisions:

1. **Top-k retrieval, not k=1 softmax.** gamma uses softmax-attended
   k=1 to keep retrieval differentiable; delta takes the literal top-k
   rows and lets the cross-attention softmax pick among them. k=8
   default (configurable via `--engram-xattn-k-retrieved`).

2. **Differentiable retrieval-similarity bias.** Top-k gather is
   non-differentiable, so `retrieval_query_proj` would receive no
   gradient if scores depended only on `Q.K`. We add the raw top-k
   cosine similarities as an additive bias to the attention logits.
   The bias weight is configurable (default 1.0) and can be set to 0
   to recover the pure Memorizing-Transformer behavior (frozen
   retrieval projection). `test_retrieval_query_grad_vanishes_without_bias`
   guards the semantics.

3. **W_o zero-init as anti-collapse.** At step 0 the cross-attention
   output is identically 0, so the residual add is a no-op and the
   model trains as if delta were unattached. As `W_o` learns a
   nontrivial direction, injection ramps in smoothly. This replaces
   gamma's gate-clamp + entropy-regularization stack with a single
   structural choice — no warmup schedule, no entropy scaler, no
   side-channel logging. The regression guard is
   `test_forward_at_step_zero_produces_zero_residual`.

4. **Per-head RMSNorm on Q and K** (not V). Mirrors the main attention
   block's QK-norm pattern. Stabilizes the dot product if retrieval
   magnitudes drift from hidden-state magnitudes during training.

5. **Shared corpus table with gamma.** `EngramCrossAttention.load_corpus_table`
   is an alias for `EngramANNInjection.load_corpus_table`, and the table
   buffers are non-persistent (not in state_dict) in both modules —
   callers re-load from the original safetensors file on resume. One
   table file services both bake-off arms.

6. **Mutually exclusive with gamma at the config level.** Both
   `use_ann_engram` and `use_xattn_engram` cannot be True on the same
   `HarmonyModelConfig`; `__post_init__` raises. The `attach_engram_*`
   methods also reject if the other research module is already attached.

## Parameter count vs beta

Computed on `HarmonyModelConfig.tiny()`:

| Model | Total params | Overhead over alpha |
|-------|-------------:|--------------------:|
| alpha | 39,593,472 | — |
| beta  | 40,379,904 | +786,432 |
| delta | 40,314,752 | +721,280 |

delta is **65,152 params lighter than beta** (approximately 8% below the
params-matched control). The original beta design assumed W_k, W_v, W_o
would each be hidden_dim x hidden_dim (3 * 512 * 512 = 786,432), but
delta projects K/V directly from engram_dim (128) rather than promoting
retrieved embeddings to hidden_dim first, saving 2 * (512 - 128) * 512 =
393,216 params on W_k and W_v combined while spending 65,536 on
`retrieval_query_proj` (hidden_dim -> engram_dim for the retrieval query).

**This bias is conservative for the bake-off**: beta is a slightly
over-provisioned control, so any delta advantage over beta at step 20000
understates the true retrieval contribution. If delta matches or beats
beta, the result is stronger evidence of signal than a parameter-matched
comparison would give.

Alternative design (rejected): promote retrieved to hidden_dim via a
shared `retrieval_to_hidden: Linear(engram_dim, hidden_dim)` and let
W_k, W_v both be hidden_dim x hidden_dim. This would match beta's 786K
overhead more precisely but adds a projection layer without obvious
benefit; kept the simpler direct-from-engram_dim version.

## Divergences from Memorizing-Transformer / RETRO

Memorizing-Transformer (Wu et al. 2022) and RETRO (Borgeaud et al. 2022)
both use cross-attention to a retrieved-neighbor cache, but differ in
specifics:

- **Memorizing-Transformer**: frozen external memory, k ~ 32 neighbors,
  single-head cross-attention, retrieval query is derived from the
  main attention Q projection (shared weights). Our delta uses a
  separate `retrieval_query_proj` because retrieval operates in
  engram_dim (128) while main attention operates in hidden_dim (512).
- **RETRO**: chunked cross-attention (retrieval is per-chunk, not
  per-position), neighbors fed through a separate encoder before the
  main decoder attends to them. Our delta is per-position and skips
  the neighbor encoder to keep the scaffold minimal.

We are NOT trying to reproduce either architecture faithfully — the
goal is an *isolated* test of whether cross-attention can transmit
corpus-retrieval signal at the 40M / 20k scale, given identical
corpus-table inputs to gamma. Architectural sophistication (chunked
xattn, learned neighbor encoders) can follow if the minimal version
clears the decision gates.

## Known limitations / future work

- **No KV cache for retrieved neighbors.** Each forward re-runs top-k
  retrieval and re-projects K/V. Fine for 40M / 20k training; would
  need amortization for long-context inference.
- **Full-table softmax only at retrieval.** The differentiable-bias path
  uses the full `sims` tensor before topk, so memory scales
  `B * L * total_entries` at the retrieval step (same as gamma).
  Chunked/top-k-before-softmax retrieval is the obvious scaling follow-up
  if we later grow the corpus table.
- **Retrieval bias weight is fixed, not learned.** Setting it to a
  learnable scalar is a 2-line change; deferred until we see whether
  the fixed default suffices.
- **No auxiliary reconstruction loss.** gamma's optional reconstruction
  loss (s3.3.2 of the parent report) isn't scaffolded for delta; the
  zero-init anti-collapse mechanism is expected to be sufficient on
  its own. Can be added later if delta shows collapse-like dynamics.

## Test coverage

23 new tests in `tests/test_engram.py::TestEngramCrossAttention`:

- Construction validation: dim mismatch, wrong shape, zero/too-large k,
  indivisible hidden_dim, non-finite bias weight, zero/negative
  temperature
- Buffer semantics: table registered as non-parameter buffer; table +
  table_normalized excluded from state_dict
- Retrieval: top-k shape, exact-table-row gather (regression guard
  against any blend/approximation)
- Forward: shape; zero-output at step 0 (o_proj zero-init regression
  guard)
- Gradient flow: grads present on q_proj, k_proj, v_proj, o_proj,
  retrieval_query_proj (differentiable-bias path); grad vanishes on
  retrieval_query_proj when `retrieval_bias_weight=0` (sanity
  check of the bias semantics)
- Integration: `HarmonyModel.attach_engram_xattn` + forward; attach
  rejects when `use_xattn_engram=False`; attach rejects if gamma is
  already attached; config-level mutual-exclusion validation
- Factory: `HarmonyModelConfig.tiny_engram_xattn()` correctness

All 23 pass alongside the 21 existing gamma tests (44 total in these
two classes, no regressions elsewhere).

## Kickoff readiness

If the ZEB-117 20k bake-off clears the top decision gate for gamma,
kick off delta with:

```text
python -m ct87.train --config tiny_engram_xattn \
    --engram-xattn-table <shared_corpus_table> \
    --engram-xattn-k-retrieved 8 \
    --engram-xattn-retrieval-bias-weight 1.0 \
    --steps 20000 --batch-size 128 --seq-len 512 \
    --train-data /data/fineweb-edu-poc/train \
    --val-data /data/fineweb-edu-poc/val \
    --log-every 100 --val-every 1000
```

No pre-flight tuning needed: the architecture has no gate,
anti-collapse schedule, or dynamic regularizer that could phase-interact.
If the first 500–1000 steps show val_loss trajectory divergent from
alpha (either direction) we'll have a cleaner signal than gamma's
pre-flight iteration gave us.
