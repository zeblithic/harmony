# Harmony-350M Target Pretraining (ZEB-137, ZEB-102 child)

**Date:** 2026-04-17
**Issue:** ZEB-137 (create in Linear as child of ZEB-102)
**Parent:** ZEB-102 (Engram table quality)
**Depends on:** None (independent of ZEB-136)
**Blocks:** ZEB-138 (2×2 scale + teacher-match matrix, both cells require a non-40M Harmony checkpoint as substrate)
**Status:** Design approved, awaiting implementation + execution
**Runs on:** KRILE (RTX 4090, 22GB usable) primary; cloud A100 conditional fallback

## Motivation

The 2026-04-17 batch (ZEB-133 / ZEB-134 / ZEB-135) established the scale-capacity steelman for the 40M × Mistral-7B × FineWeb-Edu regime: three orthogonal probes all landed at Δ-diff ≈ 0. Two axes remain untested to discriminate "capacity-ceiling at 40M" from "cross-architecture gap too big" from "frozen-backbone dooms the setup":

1. **Scale-up axis:** same ι₂ + skip-router against a larger backbone. If a 350M student decodes Mistral hiddens where 40M can't, the failure was pure capacity.
2. **Teacher-architecture-match axis:** same 40M student against a same-architecture teacher. If a Harmony teacher decodes where Mistral doesn't, the failure was cross-architecture modality gap.

Both axes require a **larger Harmony checkpoint as substrate**. No such checkpoint currently exists in the zeblithic repo — the 40M `tiny()` is the largest shipped model. ZEB-137 produces the canonical `target()` (24L/1280H/~350M params) checkpoint that enables both ZEB-138 cells.

The `target()` config is already defined in `training/ct87/model.py` and already dispatched by `train.py --config target` at line 847. Infrastructure is ready; this spec defines the training protocol.

## Commodity-hardware research axis

One motivation behind targeting 350M specifically is the self-imposed constraint "what can be built on consumer/commodity gaming hardware?" — GPUs in the 8-24 GB VRAM range (RTX 3070 through RTX 4090), optionally spilling to moderate cloud rental when wall-time bottlenecks warrant it. This shapes:

- Target size: 350M is at the knee of the "trainable on a single 4090 in reasonable wall time" curve. Larger (1B+) exits the commodity envelope and demands multi-GPU or cloud-only.
- Inference fit: PolarQuant-ized 350M fits Opal (RTX 3070, 8GB) with ample KV-cache headroom. A 1B+ model becomes a harder inference target on commodity hardware.
- Cloud budget: hard cap of ~$1000 total across the research program. Cloud rental is opt-in when KRILE wall time exceeds ~1 week AND the experiment's research signal justifies the spend — NOT a default.

## Strategy: Option B training target (~2B tokens)

Chinchilla-optimal for 350M is ~7B tokens (`20× params`). Matched-compute-with-40M (same ~800M tokens seen) is meaningfully undertrained. Option B sits at ~35% of Chinchilla with 2B tokens — the knee of the curve where per-param richness is materially better than 40M's while the training budget stays inside 1 week on KRILE.

**Why not Chinchilla-optimal (~7B tokens, Option C):** at ~9× per-token compute vs 40M, 7B tokens runs ~18-25 days on KRILE alone. Consumes most of the ZEB-137/138 timeline without obvious additional research value — the "is 350M materially better than 40M as a teacher?" question is answered by Option B just as well as by Option C. Option C remains a conditional follow-up if Option B results on ZEB-138 are strongly positive and further capacity improvement becomes interesting.

**Why not matched-compute (~800M tokens, Option A):** ~1-2 days faster but produces a materially undertrained 350M where the teacher-quality improvement over 40M may not be large enough to see the teacher-match signal. Not worth saving the days.

### Compute derivation

- Effective batch: `--batch-size 32 --grad-accum-steps 4 --seq-len 2048` → 262,144 tokens/optimizer-step (256k, standard for this scale)
- Target tokens: 2 × 10⁹
- Optimizer steps: 2e9 / 262k ≈ **7,800 steps**
- KRILE per-step time estimate: ~60-80s at optimized throughput (350M fwd+bwd on B=32)
- Wall time estimate: 7800 × 70s ≈ 152 hours ≈ **6.3 days**

Cloud A100 wall time (same effective batch, ~3× throughput): ~2 days at ~$50-60.

### Checkpoint drops

`--save-every 1500` produces intermediate checkpoints at steps 1500 / 3000 / 4500 / 6000 / 7500 + final (7800) = **6 checkpoints**. Each is a candidate artifact for ZEB-138 — they cover the capacity-curve from ~20% to 100% of the training budget, so ZEB-138 can sweep "teacher quality vs engram decodability" continuously rather than as a single-point test.

`--checkpoint-interval 300` writes resumable optimizer+RNG snapshots every ~2-3 hours. Cheap crash-recovery insurance; retains last 2 snapshots per the existing CLI contract.

## Implementation surface

**Zero code changes.**

- `train.py --config target` already constructs `HarmonyModelConfig.target()` (`train.py:847`)
- `HarmonyModelConfig.target()` specifies the full 24L / 1280H / ~350M architecture (`model.py:192-209`)
- `prepare_data.py --max-tokens N` is parameterized on arbitrary dataset sizes
- `--save-every` + `--checkpoint-interval` handle both artifact drops and resumability
- `--gradient-checkpoint` is available if VRAM becomes tight (not expected at B=32)

**The only risk is a latent bug at 350M batch sizes that didn't surface at 40M scale.** Mitigation: run a 200-step smoke test before committing to the full 7,800-step run (covered below).

## Step 0 — Dataset prep (4-6 hours)

The FineWeb-Edu-POC dataset (~800M tokens) is insufficient for a 2B-token training target without epoching. `prepare_data.py` streams the HuggingFace `HuggingFaceFW/fineweb-edu-score-2` subset and tokenizes with Mistral v0.1 tokenizer (Llama-2 SentencePiece, vocab 32000). Same pipeline as POC, larger `--max-tokens`.

### Preparation command

```bash
python3 -m ct87.prepare_data \
  --output data/fineweb-edu-3b \
  --seq-len 2048 \
  --max-tokens 3_000_000_000 \
  --val-fraction 0.01
```

3B tokens (1B margin above the 2B training target). Keeps Option C within reach if early results are promising and we want to extend — we'd re-prep to 8-10B (another ~10-15 hours of prep) rather than regenerate from scratch.

### Verification

```bash
python3 -c "
from datasets import load_from_disk
d = load_from_disk('data/fineweb-edu-3b/train')
v = load_from_disk('data/fineweb-edu-3b/val')
print(f'train chunks: {len(d):,} ({len(d) * 2048:,} tokens)')
print(f'val chunks:   {len(v):,} ({len(v) * 2048:,} tokens)')
"
```

Expected: train ≥ 1,460,000 chunks (2,990M tokens at seq_len=2048), val ≥ 14,000 chunks (30M tokens). If either falls significantly short, `--max-tokens` under-delivered (network interruption, HF schema shift); re-run or resume.

### Hardware

CPU + network bound. No GPU required. Runs on KRILE while GPU is idle, or on AVALON in parallel with ZEB-136. Expected 2-4 hours on a reasonable internet connection + modern CPU. Storage: ~12-15GB on disk as int32 arrow tensors.

## Step 1 — Smoke test (30 minutes)

Before committing to a 6-day run, verify the exact training command works end-to-end at 350M scale. A 200-step run catches:

- OOM at `batch_size=32` (unlikely but possible on specific driver / activation-memory interaction at 350M)
- Bugs triggered only at the `target` config path that didn't surface with `tiny` (e.g., config-field defaults that silently break larger-scale runs)
- Dataloader configuration issues with the fresh `fineweb-edu-3b` dataset

### Smoke command

```bash
python -m ct87.train \
  --config target \
  --data data/fineweb-edu-3b/train \
  --val-data data/fineweb-edu-3b/val \
  --seq-len 2048 \
  --batch-size 32 \
  --grad-accum-steps 4 \
  --lr 3e-4 \
  --warmup 50 \
  --steps 200 \
  --save-every 100 \
  --checkpoint-interval 0 \
  --dtype bfloat16 \
  --output-dir training/checkpoints/harmony_350m_smoke
```

### Smoke acceptance criteria

- Completes 200 steps without OOM or NaN
- Training loss drops from near-initial (~10.4 at step 0 for vocab 32k) toward ~7-8 by step 200 (rough — LM loss at this point is dominated by embedding alignment, not model quality)
- Step time is in the 60-100s range — materially outside that means the throughput estimate is wrong and the 6-day wall-time projection needs adjustment
- VRAM peak below ~18GB (leaves headroom for gradient accumulation variance)

If any of these fail, flag before launching the full run. If VRAM peak exceeds 18GB, drop to `--batch-size 16 --grad-accum-steps 8` (halves peak, same effective batch, modestly slower throughput).

## Step 2 — Full pretraining run

### Training command

```bash
python -m ct87.train \
  --config target \
  --data data/fineweb-edu-3b/train \
  --val-data data/fineweb-edu-3b/val \
  --seq-len 2048 \
  --batch-size 32 \
  --grad-accum-steps 4 \
  --lr 3e-4 \
  --warmup 800 \
  --steps 7800 \
  --save-every 1500 \
  --checkpoint-interval 300 \
  --dtype bfloat16 \
  --output-dir training/checkpoints/harmony_350m_v1 \
  --log-file training/checkpoints/harmony_350m_v1/train.csv
```

### Hyperparameter choices

| Parameter | Value | Rationale |
|---|---|---|
| `--config target` | target | The 24L/1280H/~350M preset already defined in `model.py` |
| `--batch-size 32` | 32 | GPU-saturating micro-batch on 4090 at 350M; drop to 16 if VRAM is tight |
| `--grad-accum-steps 4` | 4 | Yields 256k-token effective batch; standard LM pretraining scale |
| `--lr 3e-4` | 3e-4 | Matches 40M's successful LR; Muon's spectral-norm scaling is fairly scale-robust. Fallback: 2e-4 if loss diverges |
| `--warmup 800` | 800 | ~10% of training steps (standard); prevents early-loss spikes from frozen LN states meeting live-data grads |
| `--steps 7800` | 7800 | 2B tokens at 256k effective batch |
| `--save-every 1500` | 1500 | 5 intermediate + 1 final = 6 checkpoints spanning 20%-100% of training |
| `--checkpoint-interval 300` | 300 | Resumable snapshots every ~2-3 hours wall time; cheap crash insurance |
| `--dtype bfloat16` | bfloat16 | Modern GPU default; no memory-vs-precision concerns at this scale |

### Optimizer

Muon (existing default in `train.py`). Muon's spectral-norm weight updates are designed to be robust to scale — the same effective LR works across 40M, 350M, and larger. No per-scale LR sweep planned for the initial run.

### Monitoring during training

- **train_loss curve** via CSV log; inspect after smoke test and at each checkpoint drop
- **val_loss** (computed every `--save-every` step) — primary quality signal
- **step time** — drift upward signals a problem (GC thrashing, disk I/O stalls, thermal throttling)
- **GPU utilization** — should sit at 90%+ during steady-state; sub-70% signals dataloader bottleneck

### Per-checkpoint validation target

At step 1500 (first drop): val_loss should be materially below random (log(32000) = 10.37 nats) and materially below a 40M-comparable checkpoint at the same token budget. Rough expectation: val_loss ≤ 5.0 at step 1500 based on typical LM loss curves at ~260M tokens seen. If still near random at this point, hard stop and debug — something is fundamentally broken (data alignment, config issue, optimizer instability).

At step 7800 (final): val_loss ≥ ~3.0-3.5 is the rough "functional 350M at undertrained budget" target. Lower is better; higher flags under-training vs expectation.

## Step 3 — Checkpoint post-processing

Each of the 6 checkpoints from the full run is a candidate oracle-source for ZEB-138. Artifacts to preserve and version:

```
training/checkpoints/harmony_350m_v1/
├── ckpt_1500.pt     # ~20% budget — early undertrained teacher candidate
├── ckpt_3000.pt     # ~40% budget
├── ckpt_4500.pt     # ~60% budget
├── ckpt_6000.pt     # ~80% budget
├── ckpt_7500.pt     # ~95% budget
├── ckpt_7800.pt     # final (100%) — canonical checkpoint
├── train.csv        # training log
└── config.json      # config metadata (auto-generated by train.py)
```

For ZEB-138 we likely use only a subset (e.g., the final + one early + one mid) as oracle sources to keep the 2×2 matrix tractable. The full set gives us a capacity-curve scan if warranted.

## Cloud path (opt-in)

KRILE wall time for Option B is ~6-7 days. That's inside the "tolerable for a single experiment" window. Cloud rental is justified if:

1. KRILE wall time exceeds 1 week AND there's an external deadline pressure, OR
2. KRILE has other experiments (ZEB-136 round 2, future ablations) that can fit in the same window, making cloud the critical-path freer, OR
3. Early training curves (first 1-2 checkpoints) strongly outperform 40M expectations and we want to accelerate to completion.

### Cloud execution

Same commands as above. Targets: Lambda Labs A100 40GB ($1.10/hr on-demand) or vast.ai equivalent. Expected wall time on A100: ~2 days (~3× throughput vs 4090 at this model size). Total cost estimate: $55-75 for Option B end-to-end.

### Budget accounting

Hard cap for the full ZEB-137/138 program: $1000 cloud spend. Option B at cloud ≤ $75; Option C at cloud ≤ $250. Combined budget for training + ablations is well within the cap.

## Risks

### Risk A — OOM at `batch_size=32`

4090's 22GB usable VRAM should comfortably fit 350M at B=32, but activation patterns vary by specific ops and PyTorch version.

**Detection:** smoke test's first step OOMs, or peak VRAM logged by PyTorch exceeds ~20GB during grad accum.

**Mitigation:** `--batch-size 16 --grad-accum-steps 8` halves peak VRAM, same effective batch, ~10-15% slower throughput.

### Risk B — Muon instability at 350M

Muon has been validated up to moderate scale (~1B) in published settings, but specific configs can still hit instability (spectral norm divergence early in training, NaN losses).

**Detection:** train_loss NaN within first ~200 steps, or train_loss exceeds initial warmup-end value by 2× in any 50-step window during warmup.

**Mitigation:** drop `--lr 3e-4` → `2e-4`. If that still diverges, Muon may be genuinely unstable at 350M-scale on this setup — pause and investigate via short targeted runs rather than adding an AdamW fallback on the critical path. Adding `--optimizer adamw` to `train.py` is a non-trivial code change (the current Muon call at `train.py:1478` hardcodes both Muon+embedded-AdamW partitions via `optim.partition_params`), and doing that blind under schedule pressure would defeat the purpose of catching the issue early. Flag and decide the path out of band.

### Risk C — Dataloader bottleneck

At B=32 × seq_len=2048 = 65k tokens/micro-batch, dataloader throughput needs to sustain ~10k tokens/sec to keep GPU fed. HuggingFace Arrow should handle this comfortably but network-mounted storage or slow disk could cap throughput.

**Detection:** GPU utilization < 70% during steady-state (post-warmup), step time materially above 70s.

**Mitigation:** enable dataloader prefetching, increase worker count, move dataset to local SSD if not already. Standard LM-pretraining plumbing.

### Risk D — Checkpoint size + disk

6 checkpoints × ~700MB (model weights) + optimizer+RNG snapshots (~2× per resumable) ≈ 10-15GB total. Manageable but worth pre-checking disk free space before Step 2.

**Detection:** ensure ≥ 30GB free on `training/checkpoints/` disk partition before launch.

**Mitigation:** redirect `--output-dir` to a larger partition if needed.

### Risk E — Val loss never separates from baseline

An unexpected data preparation issue or tokenization mismatch could leave val_loss high despite many training steps. We see this by step 1500's val_loss not dropping below ~5.0.

**Detection:** monitor each `--save-every` checkpoint's reported val_loss.

**Mitigation:** stop, inspect a few val samples manually (decode via tokenizer), compare to 40M's known-good val_loss trajectory at similar token counts. If token stream or dataset is corrupted, re-run Step 0 with stricter verification.

## Cost estimate

KRILE path (primary):

- Step 0 (dataset prep): 2-4 hours CPU+network, free, can parallelize with ZEB-136 on AVALON
- Step 1 (smoke): 30 min GPU
- Step 2 (full train): ~5-7 days GPU continuous
- Step 3 (post-processing): minutes

Total wall time: 5-7 days KRILE-GPU + 2-4 hours CPU.

Cloud A100 fallback path:

- Step 0: same (local)
- Steps 1-2: ~2-2.5 days on A100
- Cost: $55-75

## Out of scope

- Engram modules (ZEB-138 territory; ZEB-137 produces the backbone only)
- 125M or 1B intermediate sizes (350M is the decided target; others are follow-ups if signal warrants)
- Chinchilla-optimal (7B tokens, Option C) — conditional future
- Non-FineWeb-Edu datasets
- Capability evals (MMLU, HellaSwag, etc.) — ZEB-137's deliverable is val_loss curves; formal benchmarks are a separate issue if warranted
- Model-family expansion (e.g., 1.3B, 3B) — outside the commodity-hardware axis
- Quantization / inference export — covered by existing `export_gguf.py` infrastructure, not gated on ZEB-137

## Testing expectations

No new code → no new unit tests.

**Integration test expectation:** before launching Step 2, the smoke test (200 steps) serves as the de facto integration test. It exercises the `--config target` path end-to-end with the new 3B-token dataset, which no other test currently covers.

**Pre-flight checklist before Step 2:**

- [ ] Step 0 complete, val set populated (≥ 14k chunks)
- [ ] Smoke test passes (200 steps, sensible loss drop, no OOM, step time ~60-100s)
- [ ] Checkpoint output directory has ≥ 30GB free
- [ ] `nvidia-smi` shows 4090 at idle / low temp before launch
- [ ] `wandb` disabled or configured if used (not currently in train.py — CSV only)

## Post-completion handoff to ZEB-138

After ZEB-137 completes, the deliverables for ZEB-138 are:

1. **6 checkpoints** in `training/checkpoints/harmony_350m_v1/` on disk
2. **Training log** (`train.csv`) — val_loss curve, step times, any warnings
3. **Pretraining summary** in `docs/research/2026-04-XX-harmony-350m-pretraining-findings.md` on branch `zeblith/zeb-137-harmony-350m-pretraining`, covering:
   - Final val_loss achieved
   - Val_loss at each checkpoint (capacity-curve anchor for ZEB-138)
   - Any anomalies / risks realized
   - Wall time consumed and any cloud spend
4. **Branch pushed** and ready for cross-host checkout

ZEB-138 will then generate a Harmony-350M oracle from one or more of these checkpoints (reusing `generate_oracle_table.py` with a new local-Harmony-teacher path — small code addition) and run the 2×2 matrix of engram experiments.

## References

- Parent issue: ZEB-102 — Engram table quality
- Companion issue: ZEB-136 — TinyLlama cross-arch cheap-signal (AVALON, runs in parallel)
- Downstream: ZEB-138 — 2×2 scale + teacher-match matrix (filed after ZEB-137 completes)
- Model config: `training/ct87/model.py::HarmonyModelConfig.target()` (lines 192-209)
- Training entrypoint: `training/ct87/train.py::main()` (line 396; `target` dispatch at line 847)
- Dataset prep: `training/ct87/prepare_data.py` (FineWeb-Edu-score-2, Mistral v0.1 tokenizer)
- 40M context: `project_krile_engram_training.md` + `project_zeb130_shuffle_kill.md` in `.claude/projects/-Users-zeblith-work/memory/`
