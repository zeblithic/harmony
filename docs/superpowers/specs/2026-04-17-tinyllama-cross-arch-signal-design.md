# Experiment ZEB-136: TinyLlama cross-architecture cheap-signal (ZEB-102 child)

**Date:** 2026-04-17
**Issue:** ZEB-136 (create in Linear as child of ZEB-102)
**Parent:** ZEB-102 (Engram table quality)
**Depends on:** ZEB-133 / ZEB-134 / ZEB-135 (Δ-diff ≈ 0 results, 2026-04-17)
**Status:** Design approved, awaiting implementation + execution on AVALON
**Runs on:** AVALON (RTX 5080, 16GB VRAM) — frees KRILE to start ZEB-137 pretraining in parallel

## Motivation

After the 2026-04-17 batch of probes (ZEB-133 decay, ZEB-134 skip-to-logit, ZEB-135 EMA baseline-subtract), all three interventions against the ι₂ (Q-div + V-contrast) setup landed at Δ-diff ≈ 0. The joint read: content reaches LM-head input (88% survival, vocab alignment 3.5σ above floor), baseline is not the bottleneck (ratio 0.11), and a direct decoder bypass produces a router that learns to emit uniform logits (`engram_logit_entropy = log(32000)` exactly). At 40M frozen, with Mistral-7B hiddens as the oracle, the optimizer finds no useful LM-improvement gradient from content routing.

This is consistent with the scale-capacity steelman — but with one ambiguity. The experiments so far hold two variables conjoint: **student capacity (40M)** and **cross-architecture gap (Mistral-7B, 175× params, different arch family)**. A failure could be either or both. Before committing to a multi-week larger-Harmony pretraining effort (ZEB-137), a cheap experiment can disambiguate these:

- If a **smaller-gap cross-arch teacher** produces Δ-diff > 0, the Mistral-7B gap specifically was the bottleneck. Cross-architecture retrieval IS viable at 40M; it was just the 175× gap that killed it. This strengthens the scale-up narrative.
- If a smaller-gap cross-arch teacher still collapses to Δ-diff ≈ 0, cross-architecture at 40M is fundamentally hopeless regardless of gap. Teacher-architecture-match (ZEB-138) becomes the decisive test rather than a parallel probe.

## Strategy

Swap the oracle teacher to **TinyLlama-1.1B**, rerun the ι₂ + skip-router 4-cell matrix exactly as ZEB-134 ran it, compare to ZEB-134's Mistral-oracle baseline. One variable changes. Everything else identical.

TinyLlama specifically because:

- **Tokenizer parity.** TinyLlama uses the Llama-2 SentencePiece tokenizer (vocab 32000) — bit-for-bit compatible with the FineWeb-Edu-POC corpus and the 40M student. No re-tokenization required. Tokenizer parity is a load-bearing invariant of the `generate_oracle_table.py` pipeline; garbage IDs produce noise hidden states that Welford happily averages into nonsense. This invariant is automatically satisfied here.
- **Capacity gap 28× (vs Mistral's 175×).** Meaningfully smaller than Mistral-7B but still a genuine cross-architecture gap, not a trivial same-size comparison. Enough to test the gap-sensitivity hypothesis.
- **Pretrained, free, local.** ~2.2GB HuggingFace download (`TinyLlama/TinyLlama_v1.1`). No training required.
- **Architecturally similar to Mistral (Llama-family).** Same positional encoding (RoPE), same tokenizer lineage, similar training data distribution. If cross-arch retrieval works at all, it should work here.

Specific teacher: **`TinyLlama/TinyLlama_v1.1`** — the final base release, NOT the Chat variant. We want raw LM hidden states, not RLHF-adjusted ones.

## Implementation surface

**Zero code changes.** `generate_oracle_table.py` is already parameterized on `--teacher` (arbitrary HuggingFace model ID); the 4-cell training infrastructure (ι₂ + skip-router) already exists in `train.py` / `engram.py` from ZEB-134.

The entire experiment is a procedure over existing tooling.

## Oracle generation

### Preconditions on AVALON

1. Latest `main` synced.
2. FineWeb-Edu-POC tokenized dataset available locally (copied from KRILE via USB SSD or regenerated via `ct87.prepare_data`).
3. 40M base backbone checkpoint (β-baseline equivalent) copied from KRILE to AVALON.
4. `transformers`, `safetensors`, `datasets` Python dependencies installed.
5. HuggingFace network access (TinyLlama-1.1B is public, no auth).

### Oracle extraction command

```bash
python -m ct87.generate_oracle_table \
  --teacher TinyLlama/TinyLlama_v1.1 \
  --dataset <AVALON-path>/fineweb_edu_poc_tokenized \
  --layer -2 \
  --entries 10000 \
  --engram-dim 128 \
  --hash-seeds 42,99,137,251 \
  --batch-size 16 \
  --seq-len 2048 \
  --dtype bfloat16 \
  --output artifacts/oracle_tinyllama_10k.safetensors
```

**Layer choice: `-2` (penultimate block output).** Mirrors Mistral oracle's choice (`--layer -2`, layer 31 of 32). For TinyLlama's 22-layer architecture, `-2` resolves to layer 21. Same semantics (last transformer block before the final RMSNorm), same diagnostic surface.

**Hash seeds: `42,99,137,251`.** Default `DEFAULT_HASH_SEEDS` in `generate_oracle_table.py`. MUST match the student's training seeds — if the student and oracle use different hash seeds, every retrieved row contains content from a different n-gram than what the student is querying about, and the oracle is garbage from the student's perspective. The 40M student's existing ι₂ config uses these seeds (match with Mistral oracle).

**Batch size: 16** (vs Mistral oracle's 8). TinyLlama-1.1B at bf16 is ~2.2GB resident vs Mistral-7B's ~14GB — AVALON's 14GB usable VRAM fits larger batches. Adjust down if activations push VRAM close to limit.

**Entries: 10000, engram_dim: 128.** Match Mistral oracle's shape so the existing 40M student loads this oracle without config changes.

### Oracle health check (CRITICAL — distinguishes teacher-weakness from gap)

Before interpreting any Δ-diff as "smaller gap works/doesn't," verify the TinyLlama oracle isn't obviously pathological. Compare the auto-generated `.stats.json` sidecar against Mistral's:

| Metric | Mistral oracle | TinyLlama target | If dramatically worse |
|---|---|---|---|
| `populated_fraction` | ~0.95-1.0 | ≥ 0.9 | n-gram coverage issue — independent of teacher, suspect dataset alignment |
| `pca_explained_variance_ratio_total` | ~0.5-0.7 | ≥ 0.4 | TinyLlama's 2048-d hidden space compresses less well into 128 dims — flag but continue |
| `populated_rows` | ~10000 | match Mistral | same as populated_fraction |

If TinyLlama's PCA variance is < 0.25 (implausibly low), the teacher's hidden states lack usable structure at 128 dims. That's a teacher-weakness confound; results downstream should be interpreted with caution. Log the ratio in the summary regardless.

For additional diagnostic color: compute the mean pairwise cosine among populated rows (via a small helper script or one-liner). Oracle rows should NOT all be near-parallel — if the median pairwise |cos| > 0.5, the teacher produces content-collapsed hidden states and retrieval is conceptually broken.

## Experiment matrix

Four training runs, ι₂ config + skip-router, 2000 steps each. Match ZEB-134's setup exactly so the comparison is apples-to-apples.

| Run | Router | Oracle | Purpose |
|---|---|---|---|
| 1 | off | `oracle_tinyllama_10k.safetensors` (real) | Baseline: does smaller gap unlock Δ-diff via normal injection path? |
| 2 | off | `oracle_tinyllama_10k_shuffled_seedN.safetensors` | Shuffle control for run 1 |
| 3 | on | `oracle_tinyllama_10k.safetensors` (real) | Does the decoder-bypass router contribute now that gap is smaller? |
| 4 | on | `oracle_tinyllama_10k_shuffled_seedN.safetensors` | Shuffle control for run 3 |

### Shuffle oracle preparation

Generate the shuffled oracle by row-permuting the real oracle with a fixed seed that matches KRILE's ι₂ shuffle seed (for comparability). Use whatever seed the existing KRILE ι₂ shuffle uses; if uncertain, pick `seed=0` and document.

```bash
python -c "
import torch
from safetensors.torch import load_file, save_file
t = load_file('artifacts/oracle_tinyllama_10k.safetensors')['engram.weight']
g = torch.Generator().manual_seed(0)
perm = torch.randperm(t.shape[0], generator=g)
save_file({'engram.weight': t[perm]}, 'artifacts/oracle_tinyllama_10k_shuffled_seed0.safetensors')
"
```

### Training commands

Substitute the actual KRILE-side preset/CLI-flag names if they differ from this template. The exact flag set must match ZEB-134's runs bit-for-bit EXCEPT for the `--engram-xattn-table` path.

```bash
# Run 1: router-off, real oracle
python -m ct87.train \
  --config tiny_engram_xattn_capgap_iota2 \
  --init-from <AVALON-path>/checkpoints/beta_baseline/ \
  --engram-xattn-table artifacts/oracle_tinyllama_10k.safetensors \
  --freeze-backbone \
  --max-steps 2000 \
  --save-dir checkpoints/zeb136_tinyllama_router_off_real/

# Run 2: router-off, shuffled oracle
python -m ct87.train \
  --config tiny_engram_xattn_capgap_iota2 \
  --init-from <AVALON-path>/checkpoints/beta_baseline/ \
  --engram-xattn-table artifacts/oracle_tinyllama_10k_shuffled_seed0.safetensors \
  --freeze-backbone \
  --max-steps 2000 \
  --save-dir checkpoints/zeb136_tinyllama_router_off_shuf/

# Run 3: router-on, real oracle
python -m ct87.train \
  --config tiny_engram_xattn_capgap_iota2 \
  --init-from <AVALON-path>/checkpoints/beta_baseline/ \
  --engram-xattn-table artifacts/oracle_tinyllama_10k.safetensors \
  --engram-skip-to-logit \
  --engram-skip-alpha-init 0.1 \
  --freeze-backbone \
  --max-steps 2000 \
  --save-dir checkpoints/zeb136_tinyllama_router_on_real/

# Run 4: router-on, shuffled oracle
python -m ct87.train \
  --config tiny_engram_xattn_capgap_iota2 \
  --init-from <AVALON-path>/checkpoints/beta_baseline/ \
  --engram-xattn-table artifacts/oracle_tinyllama_10k_shuffled_seed0.safetensors \
  --engram-skip-to-logit \
  --engram-skip-alpha-init 0.1 \
  --freeze-backbone \
  --max-steps 2000 \
  --save-dir checkpoints/zeb136_tinyllama_router_on_shuf/
```

### Optimizer, LR, steps

Match ι₂ exactly (inherited from η-B + θ + ι₂ convention):
- Muon optimizer (backbone frozen, only engram params train)
- Step count: 2000
- LR / schedule: inherit from the `tiny_engram_xattn_capgap_iota2` preset
- Shuffle seed: match KRILE's ι₂ shuffle seed if known; else `seed=0` with explicit note

## Validation protocol

### Forensic probes per run

Run the existing forensic script (extended in PRs #249/#251/#253 with the full `(W)/(A)/(X)/(Q-overlap)/(V-rank)` probe set) against each of the 4 checkpoints:

```bash
python -m training.scripts.forensic_eta_b_capgap \
  --real-checkpoint checkpoints/zeb136_tinyllama_router_off_real/ \
  --shuf-checkpoint checkpoints/zeb136_tinyllama_router_off_shuf/ \
  --real-table artifacts/oracle_tinyllama_10k.safetensors \
  --shuf-table artifacts/oracle_tinyllama_10k_shuffled_seed0.safetensors \
  --alt-shuffle-seed 99 \
  --layers 2,5
```

**Important: use `--alt-shuffle-seed` at a value DIFFERENT from the training-time shuffle seed.** The (X) probe samples a held-out alternative table to avoid the row-permutation tautology that PR #251 fixed — reusing the training-time shuffle seed would collapse the probe to identity.

### If router runs produce nonzero `log_alpha` growth

For router-on runs (runs 3 and 4), extract the router's final parameters and compare to ZEB-134's:

```python
import torch
ckpt = torch.load('checkpoints/zeb136_tinyllama_router_on_real/final.pt', weights_only=True)
sd = ckpt['model_state_dict']
log_alpha = sd['engram_skip_router.log_alpha'].item()
W_align_fro = sd['engram_skip_router.W_align.weight'].norm().item()
print(f"log_alpha={log_alpha:.4f}, alpha={math.exp(log_alpha):.4f}, ||W_align||_F={W_align_fro:.4f}")
```

ZEB-134 Mistral-oracle final: `log_alpha=-1.77, alpha=0.17, ||W_align||_F=1.96`. Material deviation upward from these values means the router is contributing more under the smaller-gap teacher.

## Success criteria / verdict matrix

Primary metric: `Δ-diff = val_loss(real) − val_loss(shuf)` at step 2000, for router-off and router-on cells independently.

Decision thresholds for next-step action:

| Outcome | Interpretation | Action |
|---|---|---|
| **Run1−Run2 Δ-diff ≥ +0.002** | Smaller-gap cross-arch unlocks the normal injection path. Scale-up narrative strongly supported. | Proceed with ZEB-137 (Harmony-350M) as planned; consider running ZEB-138 teacher-match as a confirmatory rather than decisive test |
| **Run1−Run2 Δ-diff < +0.001 BUT Run3−Run4 Δ-diff ≥ +0.002** | Smaller gap requires the decoder-bypass router to surface; normal injection still blind. | ZEB-137 still valuable; ZEB-138 experiment design should include the router |
| **Both Δ-diffs < +0.001** | Cross-arch at 40M frozen is fundamentally hard regardless of gap. Smaller gap didn't help. | ZEB-138 (teacher-match via same-arch Harmony teacher) becomes the decisive test. ZEB-137 pretraining remains on the critical path because ZEB-138 depends on its output. |
| **Oracle-health check fails (PCA variance < 0.25)** | Teacher-weakness confound; TinyLlama hiddens too weak to decode into 128-d | Flag in report; consider Pythia-1.4B as a stronger alternative (same tokenizer-compatibility check required) |
| **Run3 `log_alpha` grows substantially (> 0 reached)** | Router is contributing meaningfully (vs ZEB-134's −1.77) — bypass path works at smaller gap | Document as ZEB-138 input: Harmony-teacher experiments should retain the router |

## Risks and failure modes

### Risk A — Teacher-weakness confound

TinyLlama-1.1B is undertrained relative to a modern 1B-scale checkpoint. Its hidden states may encode less semantic content per dim than Mistral-7B's, making it a weaker oracle independent of the gap question.

**Detection:** oracle-health PCA variance < 0.25 or median pairwise |cos| > 0.5 signals genuine teacher-weakness. A middle-ground result (PCA ~0.3, Δ-diff ~0) is ambiguous.

**Mitigation:** report oracle stats alongside Δ-diff; qualify conclusions accordingly. If TinyLlama fails ambiguously, a stronger intermediate-gap teacher (Pythia-1.4B with GPT-NeoX tokenizer would require corpus re-tokenization; Llama-2-7B itself would be 7× smaller than Mistral-7B but still ~175× gap) is a follow-up.

### Risk B — Shuffle seed mismatch vs KRILE ι₂

If AVALON's shuffle seed differs from KRILE's ι₂ shuffle seed, the TinyLlama shuf control is not directly comparable to the Mistral baseline shuf control — small differences could masquerade as gap effects.

**Detection:** confirm seed match with KRILE before running.

**Mitigation:** USB-SSD transfer the KRILE shuf oracle's generation seed explicitly; or rerun the Mistral shuf oracle generation on AVALON against KRILE's original seed as a cross-check.

### Risk C — Dataset distribution drift

If AVALON's FineWeb-Edu-POC tokenized dataset somehow differs from KRILE's (partial copy, different tokenization pass), the student sees different n-grams during training, breaking comparability.

**Detection:** after USB-SSD transfer, `sha256sum` the tokenized dataset shard files on both ends. Must match byte-for-byte.

**Mitigation:** transfer via USB SSD with checksums; OR regenerate via `ct87.prepare_data` on AVALON and verify identical token stats (vocabulary size, sequence count, total tokens) against KRILE.

### Risk D — AVALON VRAM ceiling during oracle extraction

TinyLlama-1.1B at bf16 = ~2.2GB resident. Peak activation during a B=16, L=2048 forward pass on a Llama-family model is another 2-6GB. Total ~4-8GB, well under AVALON's 14GB usable.

**Detection:** OOM at batch start.

**Mitigation:** drop `--batch-size` from 16 to 8 or 4. Walltime impact modest.

## Data transfer from KRILE to AVALON

USB-SSD transfer required (AVALON cannot directly read KRILE disk). Pack these:

| From KRILE path | What | Approx size |
|---|---|---|
| `checkpoints/beta_baseline/` (or whatever KRILE calls the 40M base used for ι₂ `--init-from`) | 40M base backbone checkpoint; prerequisite for engram training | ~160MB bf16 |
| `<path>/fineweb_edu_poc_tokenized/` (the HuggingFace-disk dataset dir used by KRILE's ι₂ runs) | Tokenized training corpus | ~1-3GB (depends on tokens + metadata) |
| `artifacts/oracle_mistral7b_10k.safetensors` (optional) | For cross-teacher forensic comparison; if AVALON is going to run same-script forensics against a Mistral baseline too | ~5MB |
| KRILE's ι₂ shuffle seed (if custom; note in README) | Ensures shuffle-control comparability | negligible |

After transfer, verify:

```bash
sha256sum <AVALON-path>/fineweb_edu_poc_tokenized/**/*.arrow  # should match KRILE
ls -la <AVALON-path>/checkpoints/beta_baseline/  # directory contents should include model weights + tokenizer config
```

## AVALON → KRILE return transfer

After ZEB-136 completes on AVALON:

| From AVALON | What | Approx size |
|---|---|---|
| `artifacts/oracle_tinyllama_10k.safetensors` + `.stats.json` | Generated oracle + provenance | ~5MB |
| `checkpoints/zeb136_tinyllama_*/final.pt` (all 4 runs) | Engram-trained checkpoints for forensic replay or follow-up analysis | ~200MB × 4 = ~800MB |
| `docs/research/2026-04-17-zeb136-findings.md` (created by AVALON Claude Code at end) | Forensic readings + interpretation + verdict matrix outcome | few KB |

Spec also lives in the branch `zeblith/zeb-136-tinyllama-cross-arch`; push from AVALON when complete so KRILE sees the results.

## Cost estimate

AVALON wall time:

- TinyLlama-1.1B download: ~5min (one-time)
- Oracle extraction (TinyLlama forward over ~800M tokens): ~1-2h
- Shuffled oracle generation (CPU): seconds
- 4 engram training runs × ~10min each (5080 is ~1.5× slower than 4090 per step on a 40M model): ~40min
- Forensic probes (existing script): ~5-10min per checkpoint × 4 = ~30min
- Oracle-health sanity check: ~5min

**Total: ~3-4 hours AVALON wall time**, assuming no debugging iterations.

Parallel: KRILE can begin ZEB-137 Harmony-350M pretraining during this window (separate spec/branch).

## Out of scope

- Modifications to `generate_oracle_table.py` (current implementation already handles TinyLlama via `--teacher` parameter)
- Modifications to `train.py` or `engram.py` (ZEB-134's ι₂ + skip-router infrastructure is the test target; reproducing it is the point)
- End-to-end (full backbone unfreeze) training regimes
- Teacher selection beyond TinyLlama-1.1B (Pythia, Llama-2-7B, others are Risk-A follow-ups if TinyLlama is ambiguous)
- Harmony-teacher extraction (that's ZEB-138, contingent on ZEB-137 producing a checkpoint)
- Any form of engram module retraining or redesign

## Testing expectations

No new code → no new unit tests. Existing tests in `training/tests/test_generate_oracle_table.py` already exercise the teacher-swap path (the test uses a tiny dummy model, not Mistral specifically).

**Soft test before committing to the ~1-2h oracle extraction:** smoke-run with `--max-sequences 4` to verify the pipeline works end-to-end with TinyLlama specifically (tokenizer parity check, layer index resolution, PCA fit). Expected: ~30 seconds wall time, produces a valid-shape oracle safetensors with a minimal populated-rows count. If this fails, debug before committing to the full run.

```bash
python -m ct87.generate_oracle_table \
  --teacher TinyLlama/TinyLlama_v1.1 \
  --dataset <AVALON-path>/fineweb_edu_poc_tokenized \
  --layer -2 \
  --entries 10000 \
  --engram-dim 128 \
  --max-sequences 4 \
  --batch-size 2 \
  --output /tmp/smoke_tinyllama.safetensors
```

## Reporting template

At end of experiment, AVALON's Claude Code should produce `docs/research/2026-04-17-zeb136-findings.md` on the branch with this structure:

```markdown
# ZEB-136 TinyLlama cross-arch findings

**Date:** <YYYY-MM-DD>
**Runs on:** AVALON
**Teacher:** TinyLlama/TinyLlama_v1.1

## Oracle health
- populated_fraction: <X.XX>
- pca_explained_variance_ratio_total: <X.XX>
- median pairwise |cos| across populated rows: <X.XX>
- Comparison to Mistral oracle: <same / worse / pathological>

## Training results
| Run | Router | Oracle | val_loss (step 2000) |
|---|---|---|---|
| 1 | off | real | <X.XXXX> |
| 2 | off | shuf | <X.XXXX> |
| 3 | on | real | <X.XXXX> |
| 4 | on | shuf | <X.XXXX> |

Δ-diff (router-off, real − shuf): <+X.XXXX>
Δ-diff (router-on, real − shuf): <+X.XXXX>

## Router diagnostics (router-on runs)
- log_alpha (final): <X.XXXX>
- alpha (= exp(log_alpha)): <X.XXXX>
- ||W_align||_F: <X.XXXX>
- engram_logit_entropy (final): <X.XXXX> nats (log(32000) = 10.3735)
- cross_run_cos on engram_logits: <+X.XX>

## Forensic matrix (full table per layer, L2 and L5)
[Reproduce the η-B / θ / ι₁ / ι₂ table format with ZEB-136 row added]

## Verdict
<Which row of the decision matrix matched; one-sentence narrative conclusion>

## Open questions / surprises
<Any unexpected findings>
```

## References

- ZEB-133 decay probe (2026-04-17): inconclusive; 88% signal survival, no clean H-decay vs H-blindness verdict
- ZEB-134 skip-to-logit (2026-04-17): `log_alpha = -1.77`, `engram_logit_entropy = log(32000)` = max entropy — steelman-confirmed for 40M × Mistral-7B pairing
- ZEB-135 EMA baseline-subtract (2026-04-17): `baseline_ratio = 0.109`, Q-broadening 6.4×, Δ-diff still ~0 — baseline wasn't the bottleneck
- ZEB-134 checkpoint and results on branch `zeblith/zeb-134-skip-to-logit` (KRILE, not yet in main)
- Oracle generator: `training/ct87/generate_oracle_table.py`
- Forensic script: `training/scripts/forensic_eta_b_capgap.py` (post-PR #253 with Q-overlap + V-rank)
- Gemini Deep Research (2026-04-17): `docs/research/2026-04-17-engram-lm-blindness-cross-layer-research-prompt.md`
- ι₂ baseline config: `tiny_engram_xattn_capgap_iota2` (verify exact preset name in `training/ct87/model.py` on AVALON before running)

## Next step after ZEB-136 completes

Results feed directly into ZEB-138 (scale-up + teacher-match 2×2 matrix) design decisions. ZEB-137 (Harmony-350M pretraining) proceeds on KRILE in parallel regardless — its output is the prerequisite for ZEB-138 either way.
