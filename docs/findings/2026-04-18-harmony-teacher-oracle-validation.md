# Harmony-teacher oracle extraction — plumbing validation

**Date:** 2026-04-18
**Branch:** `zeblith/harmony-teacher-oracle-validation`
**Linear:** ZEB-138 prep (parent)
**Spec:** `AVALON-handoff-ZEB-138-oracle-validation.md` (on `zeblith/zeb-138-same-arch-teacher`)
**Runs on:** AVALON (RTX 5080, 16GB VRAM)

## Status

**Works end-to-end. Code changes PR'd to main.** Three concrete integration
gaps required code changes (no model-core surgery), all in
`training/ct87/generate_oracle_table.py`. Once this PR lands, ZEB-138
cell A can start at T+0s after KRILE ships the Harmony-474M checkpoint —
the only flag delta is the `--teacher harmony:<path>` value.

## Code changes (single file: `generate_oracle_table.py`)

`load_and_validate_teacher` now dispatches based on a `harmony:` URI
prefix on `--teacher`. Three companion functions added at module scope:

1. `_load_harmony_teacher(...)` — loads a HarmonyModel from a torch-pickled
   checkpoint payload (`{"step", "model_state_dict", "config", ...}` —
   the format `train.py` writes when `--checkpoint-interval > 0`).
2. `_load_harmony_compatible_tokenizer(expected_vocab_size)` — tries
   `mistralai/Mistral-7B-v0.1` (gated, may fail HF auth) then
   `TinyLlama/TinyLlama_v1.1` (open, already cached on AVALON from
   ZEB-136). Both share the Mistral v0.1 SentencePiece 32K vocab so
   either suffices for the corpus's tokenizer-parity invariant.
3. `_HarmonyTeacherAdapter` + `_HarmonyTeacherOutput` +
   `_HookedHiddenStatesView` — adapter wrapping `HarmonyModel` to expose
   the HF-style `outputs.hidden_states[resolved_layer]` accessor that
   `process_batch` already uses. Implementation: register one
   `register_forward_hook` on the target module (embeddings for
   `resolved_layer=0`, `model.layers[resolved_layer-1]` otherwise),
   capture the layer's output tensor inline, return it via a one-element
   view that raises `IndexError` if any other layer is requested.

Total diff: ~250 LOC including docstrings + 7 new unit tests in
`tests/test_generate_oracle_table.py`. Zero changes outside
`generate_oracle_table.py` and its tests — `model.py` was inspected but
not modified.

### Three concrete integration gaps that drove the changes

| Gap | Symptom on first attempt | Fix |
|---|---|---|
| **Misuse-guard refusal** | `HarmonyModel.forward()` raises if `config.engram_inject_layers` is non-empty but `engram_injections` was never attached — designed to prevent silent training no-ops. A capgap teacher checkpoint always has `engram_inject_layers=(2,5)` declared. | Clone the loaded config via `dataclasses.replace(config, engram_inject_layers=(), engram_vcontrast_enabled=False, engram_qdiv_enabled=False, use_ann_engram=False, use_xattn_engram=False)` before instantiating `HarmonyModel`. The state_dict still loads cleanly because `engram_residual` is always built; the engram path stays inert because `forward()` never receives `engram_embeddings`. |
| **`engram_injections.*` unexpected keys** | `model.load_state_dict(strict=False)` reports `engram_injections.{2,5}.*` as unexpected, since stock `HarmonyModel(config)` doesn't have them. | Filter the `unexpected` list against a `_TEACHER_IRRELEVANT_PREFIXES` allowlist (`engram_injections.`, `engram_skip_router.`, `engram_xattn.`, `engram_ann.`). Real shape mismatches still raise. |
| **`lm_head.weight` missing when `tie_embeddings=True`** | `embed_tokens.weight` and `lm_head.weight` share storage via assignment after construction, so the state_dict round-trip drops `lm_head.weight` from `missing` only if the load handler is aware. | Filter `lm_head.weight` from `missing` when `config.tie_embeddings` is True. |

### Layer-index semantics — confirmed compatible across HF and Harmony paths

The handoff flagged this as an open question. Verified: both paths use
the same convention. `--layer 0` = embedding output; `--layer i` for
`i ∈ [1, num_layers]` = output of transformer block `i-1`. Negative
indices resolve via `num_layers + 1 + i`, so `--layer -2` = `hidden_states[num_layers - 1]`
= output of the second-to-last block. This matches `--layer -2` in the
TinyLlama (22 blocks → block 20 output) and Mistral (32 blocks → block 30
output) extractions on ZEB-136 / ZEB-134.

## Smoke test output

Ran with the existing ZEB-136 router-off-real checkpoint as a 40M
plumbing-only teacher (the `zeta_ctrl_2048` checkpoint predates the
config-persistence fix and lacks the `config` key, so it can't be used
directly — would need a sidecar config to support it).

```bash
python -m ct87.generate_oracle_table \
  --teacher harmony:/home/zebli/work/LOCAL/zeb136/checkpoints/zeb136_router_off_real/checkpoint.pt \
  --dataset /home/zebli/work/LOCAL/zeb136/data/fineweb-edu-poc/train \
  --layer -2 \
  --entries 1000 \
  --engram-dim 128 \
  --max-sequences 16 \
  --batch-size 4 \
  --seq-len 2048 \
  --dtype bfloat16 \
  --output /tmp/harmony_smoke.safetensors
```

Output (last 7 lines):
```
Loading Harmony teacher checkpoint '/home/zebli/work/LOCAL/zeb136/checkpoints/zeb136_router_off_real/checkpoint.pt' (device=cuda, dtype=bfloat16)...
Harmony teacher has 8 layers; extracting hidden_states[7] (arg: -2).
Loading tokenized dataset from '/home/zebli/work/LOCAL/zeb136/data/fineweb-edu-poc/train'...
Processing 16 sequences at batch_size=4, seq_len=2048
Teacher pass done: 32,768 tokens in 0.0 min (14,534 tok/s avg)
Rows populated: 1000/1000 (100.0%)
PCA done in 0.0s: explained_variance_ratio_total=0.9473
Saved oracle table to /tmp/harmony_smoke.safetensors ((1000, 128), dtype=float32)
```

Output file: `(1000, 128)` float32 safetensors with `engram.weight`
tensor — schema-compatible with `EngramANNInjection.load_corpus_table()` /
`EngramCrossAttention.load_corpus_table()`.

A larger run (1000 sequences × 10K entries × 16 batch size) also
completed cleanly — 2.05M tokens in 2.2 minutes, 15.4K tok/s sustained,
PCA fit 0.951.

## Health metrics

Comparison against the existing teacher baselines from ZEB-134 (Mistral)
and ZEB-136 (TinyLlama):

| Metric | Mistral-7B (ZEB-134) | TinyLlama-1.1B (ZEB-136) | **Harmony-40M smoke** | **Harmony-40M scale** |
|---|---|---|---|---|
| Entries | 10000 | 10000 | 1000 | 10000 |
| Tokens consumed | ~99M | 99M | 32K | 2.05M |
| `populated_fraction` | 1.0 | 1.0 | **1.0** | **1.0** |
| `pca_explained_variance_ratio_total` | 0.879 | 0.934 | **0.947** | **0.951** |
| median pairwise \|cos\| | 0.174 | 0.152 | **0.136** | **0.160** |
| mean pairwise \|cos\| | 0.200 | 0.181 | 0.160 | 0.185 |
| Teacher forward throughput | n/a | 5,017 tok/s | 14,534 tok/s | 15,427 tok/s |
| **Spec thresholds met** | n/a (baseline) | n/a (baseline) | **all (PCA > 0.5, \|cos\| < 0.5, populated 1.0)** | **all** |

The 40M Harmony stand-in actually outperforms TinyLlama / Mistral on
PCA + |cos|. Per the handoff, this is **expected to look "somewhat
healthy but not ZEB-136-TinyLlama-healthy"** because the 40M teacher
has less expressive hidden states than 1.1B. The fact that it matches
or slightly beats them is most likely an **artifact of the n-gram-row
mean being inherently low-rank-friendly when computed over the same
corpus the model was trained on** — the 40M backbone has memorized the
statistical structure of FineWeb-Edu-POC, so per-row Welford means align
naturally with a low-dim subspace. None of this matters for the
plumbing question; it just means the stand-in oracle wouldn't itself
make a useful teacher (which we knew).

Throughput insight: 15.4K tok/s on Harmony-40M vs 5K tok/s on
TinyLlama-1.1B — about 3× faster, scaling roughly with parameter count.
Predicts ~2K tok/s on Harmony-474M (~8× larger), or ~14h for a full
99M-token corpus pass at the spec's `--batch-size 16 --seq-len 2048`.
That's longer than ZEB-136's TinyLlama extraction (5.5h) but still
tractable on AVALON over a quiet day.

## ZEB-138 Harmony-474M readiness — paste-ready command

When KRILE delivers the step-7800 Harmony-474M checkpoint to AVALON
(USB SSD presumably mirrored at e.g. `/home/zebli/work/LOCAL/zeb138/`),
this single command produces the cell-A oracle:

```bash
cd ~/work/zeblithic/harmony/training
.venv/bin/python -m ct87.generate_oracle_table \
  --teacher harmony:/home/zebli/work/LOCAL/zeb138/checkpoints/harmony_474m_step7800/checkpoint.pt \
  --dataset /home/zebli/work/LOCAL/zeb138/data/fineweb-edu-poc/train \
  --layer -2 \
  --entries 10000 \
  --engram-dim 128 \
  --hash-seeds 42,99,137,251 \
  --batch-size 16 \
  --seq-len 2048 \
  --dtype bfloat16 \
  --output ~/work/LOCAL/zeb138/artifacts/oracle_harmony474m_10k.safetensors
```

Pre-flight checklist before kicking that off:
- [ ] Verify checkpoint payload via
      `python -c "import torch; ckpt=torch.load('<path>', weights_only=False); print(sorted(ckpt.keys()))"`
      → must contain both `model_state_dict` and `config`. If the
      checkpoint format on KRILE differs (e.g., bare safetensors), add
      a sidecar `config.json` loader to `_load_harmony_teacher` first.
- [ ] Verify `config.vocab_size == 32000` and that the dataset's pre-
      tokenized arrow files use the same vocab.
- [ ] Verify `config.num_layers >= 2` so `--layer -2` resolves
      sensibly.
- [ ] Make sure the Mistral-compatible tokenizer is in cache —
      `huggingface-cli download TinyLlama/TinyLlama_v1.1` if not.
- [ ] Confirm peak VRAM < 14 GB during the first 1-2 batches (drop
      `--batch-size` if it OOMs; expected ~10-12 GB for a 474M bf16
      model + B=16 L=2048 activations).

Smoke before the full run:
```bash
.venv/bin/python -m ct87.generate_oracle_table \
  --teacher harmony:<...>/checkpoint.pt \
  --dataset <...>/train \
  --layer -2 --entries 1000 --batch-size 4 --seq-len 2048 \
  --max-sequences 16 --dtype bfloat16 --output /tmp/harmony474_smoke.safetensors
```

If smoke produces shape `(1000, 128)` and `populated_fraction > 0.95`,
greenlight the full run.

## Tests

Added `TestHarmonyTeacherURI` (7 cases) to
`training/tests/test_generate_oracle_table.py`. Total module test count:
**42 passed, 1 skipped** (was 35 passed, 1 skipped on origin/main).
Tests cover:

- URI prefix dispatches into `_load_harmony_teacher` with stripped path
- Loader returns the same 5-tuple shape as the HF path
- Adapter forward + hooked-layer access return tensor of correct shape
- Asking for the wrong layer index raises `IndexError`
- Capgap-config misuse guard does NOT fire after the loader's config-clear
- `vocab_size` mismatch raises `ValueError` with actionable message
- Missing `config` key in payload raises `ValueError` with actionable message
- `--layer 0` correctly hooks `embed_tokens` (not `layers[0]`)

## Out-of-scope notes (kept as recommendations, not done in this PR)

- **Stale checkpoint format support** — `zeta_ctrl_2048` (and any other
  pre-config-persistence checkpoint) currently can't be used as a
  Harmony teacher because `_load_harmony_teacher` requires `payload["config"]`.
  If ZEB-138 ever needs an older checkpoint, add a `--harmony-config <preset>`
  flag (e.g., `--harmony-config tiny`) that constructs the config from the
  preset name when the payload lacks it. Trivial follow-up.
- **Optional `harmony:<path>:<config-preset>` URI extension** — would let
  the user override the config preset inline (e.g.,
  `harmony:/path/to/ckpt.safetensors:target` for the 474M preset).
  Useful when KRILE ships a bare safetensors file. Defer until needed.
- **Multi-layer extraction in one forward** — `process_batch` only ever
  asks for one `resolved_layer`, so one hook suffices. If a future ZEB
  experiment wants multiple layers per forward, replace the
  `_HookedHiddenStatesView` with a dict of `{layer_idx: tensor}` and
  attach hooks to all relevant blocks. Out of scope for ZEB-138.

## Reproducibility

- All artifacts at `/tmp/harmony_smoke.safetensors` and
  `/tmp/harmony_scale.safetensors` (ephemeral, regenerable via the
  paste-ready commands above)
- ZEB-136 stand-in teacher checkpoint at
  `/home/zebli/work/LOCAL/zeb136/checkpoints/zeb136_router_off_real/checkpoint.pt`
  (still present from 2026-04-17 ZEB-136 work; see
  `docs/research/2026-04-17-zeb136-findings.md` for provenance)
- Test suite: `cd training && .venv/bin/python -m pytest tests/test_generate_oracle_table.py -q`
