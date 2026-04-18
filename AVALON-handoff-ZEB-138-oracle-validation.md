# AVALON Handoff — Harmony-Teacher Oracle Extraction Validation

**For:** AVALON's Claude Code session
**Issue:** [ZEB-138](https://linear.app/zeblith/issue/ZEB-138/same-architecture-teacher-engram-experiment-scale-teacher-matrix) prep work
**Est. time:** 2-4 hours for end-to-end validation; +2-4 hours if integration gaps require code changes
**Priority:** Medium — strictly speculative until KRILE produces a real Harmony-474M checkpoint in ~8 days, but validating the plumbing now means ZEB-138 cell A can start at T+0s after the checkpoint lands

---

## Goal

Validate that `training/ct87/generate_oracle_table.py` can produce an oracle table from a **Harmony-architecture teacher checkpoint** (not a HuggingFace model). We'll use the existing 40M Harmony checkpoint as a **plumbing-only stand-in teacher** — it won't produce a useful oracle (40M teaching 40M is nearly identity), but it exercises every integration path ZEB-138 will need once KRILE ships the real 474M checkpoint.

Output of this work: either (a) "works end-to-end, here's the output + health metrics" with a PR to main if code changes were needed, or (b) "here's exactly what integration gaps need fixing, here's my recommended patch."

---

## Context — the integration gap

`generate_oracle_table.py` currently loads teachers via the HuggingFace path:

```python
# training/ct87/generate_oracle_table.py:420-450 (abbreviated)
from transformers import AutoModel, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(teacher_model_id)
model = AutoModel.from_pretrained(teacher_model_id, torch_dtype=..., output_hidden_states=True)
```

This works for Mistral-7B (ZEB-134) and TinyLlama-1.1B (ZEB-136). It **will not work** for Harmony checkpoints because:

1. **No HuggingFace config:** Harmony uses `ct87.model.HarmonyModel` with a custom `HarmonyModelConfig`. `AutoModel.from_pretrained()` expects HF-format `config.json`.
2. **Checkpoint format is bare safetensors:** `model_step_N.safetensors` without the HF wrapper files.
3. **No local tokenizer:** Harmony training uses the Mistral v0.1 tokenizer externally; `AutoTokenizer.from_pretrained(<harmony-path>)` will fail. We need to load the Mistral tokenizer regardless of teacher.
4. **Hidden-state API mismatch:** HF models expose `output_hidden_states=True` → tuple of per-layer states. `HarmonyModel.forward(return_hidden_states=True)` returns only the **last layer's pre-norm hidden state** (`training/ct87/model.py:770-872`). For arbitrary-layer extraction, we need a hook on a specific transformer block.

### Good news on the tokenizer parity invariant
Harmony-474M uses the **same Mistral v0.1 SentencePiece tokenizer** (32000 vocab) as the corpus — so the script's hard-fail on `vocab_size != 32000` does NOT need adjustment, we just need to load the tokenizer from `mistralai/Mistral-7B-v0.1` (or a local cache) regardless of which teacher model we're using.

---

## Concrete tasks

### Task 1 — branch + environment
```bash
# On AVALON
cd ~/work/harmony   # or wherever the Harmony repo lives
git fetch origin
git worktree add .worktrees/harmony-teacher-oracle-validation -b zeblith/harmony-teacher-oracle-validation origin/main
cd .worktrees/harmony-teacher-oracle-validation
# Standard Python env setup — should inherit from existing ZEB-136 work
```

### Task 2 — locate a Harmony checkpoint for testing
AVALON should have the 40M Harmony checkpoint used for ZEB-136's student. If not, KRILE can ship it via USB SSD. Required files:
- `model_step_<N>.safetensors` (or `checkpoint.pt` rolling checkpoint)
- The `ct87.model.HarmonyModelConfig` params it was trained with (probably `tiny` or the project's 40M preset)

Note path for the handoff report.

### Task 3 — design the Harmony-teacher branch in `load_and_validate_teacher`
Recommended approach: detect Harmony by a `harmony:` URI prefix or by a `--teacher-type harmony` CLI flag. I'd go with the URI prefix for cleanness.

```python
# Pseudocode sketch — AVALON to refine
def load_and_validate_teacher(teacher_model_id, device, dtype, expected_vocab_size, layer_index):
    if teacher_model_id.startswith("harmony:"):
        return _load_harmony_teacher(
            ckpt_path=teacher_model_id[len("harmony:"):],
            device=device,
            dtype=dtype,
            expected_vocab_size=expected_vocab_size,
            layer_index=layer_index,
        )
    # existing HF path unchanged...
```

`_load_harmony_teacher` needs to:
1. Load Mistral v0.1 tokenizer (hard-coded or via a separate flag) — this preserves vocab parity invariant.
2. Instantiate `ct87.model.HarmonyModel` with the checkpoint's config (read from the safetensors metadata, or require a sidecar `config.json`).
3. `load_state_dict` from the safetensors file.
4. Register a forward hook on block `layer_index` (0-indexed from 0 to num_layers-1, matching Harmony's convention — NOT HF's num_layers+1 convention, see §3.3 of the spec).
5. Return `(model, tokenizer, resolved_layer, teacher_dim, torch)` matching the existing function signature so downstream code doesn't branch.

**Ambiguity to resolve:** Harmony's `HarmonyModel` has 24 transformer blocks at the `target` config. Hooks get us the **post-block output** (which in HF's convention is what hidden_states[layer_idx+1] gives). Confirm layer-index semantics match between paths so `--layer -2` means the same thing for both.

### Task 4 — smoke test
```bash
# Tiny test: 1000 rows, small corpus subset
python3 -m ct87.generate_oracle_table \
    --teacher harmony:/path/to/40m-harmony-ckpt.safetensors \
    --corpus /home/zebli/work/LOCAL/zeb136/corpus \
    --total-entries 1000 \
    --batch-size 4 \
    --layer -2 \
    --output /tmp/harmony40m-stand-in-oracle.safetensors \
    --dtype bfloat16
```

Expected: script runs to completion, output file loads with `safetensors.torch.load_file()` and has `engram.weight` with shape `[1000, 128]`.

### Task 5 — health check
Run the existing health-check script (whatever AVALON used for ZEB-136's TinyLlama table) against the output. Expect the numbers to look SOMEWHAT healthy but not ZEB-136-TinyLlama-healthy (40M teacher has much less diverse hidden states than 1.1B TinyLlama).

Thresholds that MUST hold:
- `populated_fraction` = 1.0 (or very close)
- `pca_explained_variance_ratio_total` > 0.5 (will be lower than TinyLlama's 0.93 because 40M is less expressive, that's fine)
- Median pairwise `|cos|` < 0.5 (looser than ZEB-138 spec's 0.25 because this is plumbing-only)

If these hold, the plumbing works.

### Task 6 — report back
Write a short findings doc at `docs/findings/2026-04-18-harmony-teacher-oracle-validation.md` in the branch with:

1. **Status:** works / needs fixes / blocked
2. **Code changes:** diff or PR link, if any
3. **Smoke test output:** full command + stdout tail + output file metadata
4. **Health metrics:** table of the numbers, compared against ZEB-136 TinyLlama baseline
5. **Harmony-474M readiness checklist:** once KRILE ships the 474M checkpoint, exactly what command/flags run the real extraction? Include one proposed command line ready to paste.

Push the branch + open a PR to main if code changes were made. If changes were needed, ZEB-138 will consume them as a prereq.

---

## What NOT to do

- **Don't train anything.** This is pure plumbing validation — no routers, no student models, no val-loss runs.
- **Don't use the Harmony checkpoint to produce a "real" oracle for ZEB-138.** The 40M-as-teacher table is a stand-in; it will not be used for any actual experiments.
- **Don't branch off any existing ZEB-136/137/138 worktree.** Start from clean origin/main.
- **Don't modify anything outside `generate_oracle_table.py` and potentially a small helper in `ct87/model.py`** (e.g., exposing an easier way to hook intermediate-layer output if needed).

---

## Open question to surface back

If Task 3 requires touching `ct87/model.py` to expose intermediate-layer hidden states more cleanly (e.g., adding a `return_all_hidden_states` flag), that's a model-core change. Propose the minimal patch, flag it, and wait for confirmation before merging — it's a bigger change than "teacher loading" and worth a review.

---

## Success criterion — one sentence

*"`python3 -m ct87.generate_oracle_table --teacher harmony:<path> ...` runs to completion, produces a valid safetensors oracle, and the health check gives plausible numbers — with any needed code changes PR'd to main and a paste-ready command for the real 474M extraction in the findings doc."*
