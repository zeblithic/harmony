# ZEB-139 — KL-Retrofit Objective-Axis Experiment Findings

**Date:** 2026-04-19
**Linear:** [ZEB-139](https://linear.app/zeblith/issue/ZEB-139/kl-retrofit-experiment-objective-axis-diagnostic-for-engram-attractor)
**Spec:** `docs/superpowers/specs/2026-04-18-zeb-139-kl-retrofit-design.md`
**Foundation:** [PR #257](https://github.com/zeblithic/harmony/pull/257) (ZEB-134 revival + ZEB-139 KL+CE) and [PR #255](https://github.com/zeblithic/harmony/pull/255) (`--save-teacher-logits` sidecar producer)
**Prior art:** [ZEB-134](https://linear.app/zeblith/issue/ZEB-134) (Skip-to-Logit, CE-only, attractor observed), [ZEB-136](https://linear.app/zeblith/issue/ZEB-136) (TinyLlama cross-arch, attractor observed)

---

## TL;DR

**Adding a Memory-Decoder-style `KL(P_router || P_teacher)` term at λ=0.5 did NOT escape the maximum-entropy attractor on the cross-arch TinyLlama setup.** Both the real-oracle and shuffled-oracle KL+router cells converged to essentially identical val_loss (4.5636 vs 4.5637, Δ-diff = -0.0001 nats) and produced a router output with `cross_run_cos = +0.9999` between the two cells — the smoking gun for the cheap-win confound. KL forced both routers to the same content-independent average distribution rather than learning per-position content routing.

**Per spec §11 outer matrix this is the "KL-retrofit attractor HOLDS" outcome.** Combined with whatever ZEB-138 produces on the orthogonal teacher-architecture axis, it points toward either teacher-arch dominance (if ZEB-138 breaks) or a structural ceiling at 40M (if ZEB-138 also holds — Gemini §7 steelman).

---

## Setup

### Code prereqs (both merged to main 2026-04-19)

- **PR #255**: `--save-teacher-logits` flag added to `generate_oracle_table.py`. Welford-means the teacher's full LM-head outputs (vocab=32000) keyed by the same xxhash row indices as the existing oracle. Sidecar is `[10K, 32K] bf16` ≈ 640MB. Throughput recovered from a 24× regression via a GPU-side `index_add_` accumulator (CPU `np.add.at` on a `[10K, 32K] f64` master is fundamentally bandwidth-bound at ~10 GB/s).
- **PR #257**: ZEB-134's `SkipToLogitEngramRouter` revived (W_align d_model→d_model + log_alpha scalar + frozen LM-head reuse), and ZEB-139's KL+CE wired into `train.py`: `--kl-lambda` + `--oracle-teacher-logits` flags, per-token-normalized `F.kl_div(log_p_teacher, log_p_router, log_target=True).sum(-1).mean()` (the `log_target=True` form gets the spec'd FORWARD KL `KL(P_router || P_teacher)` direction; PyTorch's `F.kl_div(input, target)` computes `KL(target || input)` per its `target * (log target − input)` formula).

### Teacher-logits sidecar extraction

Re-ran `generate_oracle_table.py --save-teacher-logits` on the same TinyLlama-1.1B teacher + 99M-token FineWeb-Edu-POC corpus that ZEB-136 used. Wall time **5.8 hours** at 4,771 tok/s sustained on the 5080 (~14% slower than ZEB-136's no-sidecar 5,017 tok/s baseline — the GPU-resident `[10K, 32K] f64` sum table + bf16→fp32 cast per chunk accounts for the gap).

Sanity check: `pca_explained_variance_ratio_total = 0.9338690864205668`, **bit-identical to ZEB-136's stored value**. Confirms the GPU-side `SumAccumulatorTable`/`GpuSumAccumulatorTable` math is equivalent to the original CPU `WelfordTable` for the hidden-state path (proves the perf-optimization didn't change numerics).

Shuffled artifacts produced by a single `torch.randperm(seed=0)` applied to BOTH the oracle (`engram.weight`) and the sidecar (`teacher_logits.weight`) — same permutation across both files so cell-4's per-position teacher target is independently scrambled in both the engram-emb path AND the KL target path.

### 4-cell matrix configuration

Identical to ZEB-136's `run_4cell_matrix.sh` for cells 1+2 (router-off baselines), with `--engram-skip-to-logit --engram-skip-alpha-init 0.1 --kl-lambda 0.5 --oracle-teacher-logits …` added to cells 3+4.

| Cell | Router | Oracle | Teacher-logits sidecar | KL term |
| --- | --- | --- | --- | --- |
| 1 | off | real | — | off |
| 2 | off | shuf (seed=0) | — | off |
| 3 | on | real | real | λ=0.5 |
| 4 | on | shuf (seed=0) | shuf (seed=0, same perm) | λ=0.5 |

Each cell init's from `zeta_ctrl_2048/checkpoint.pt` (the same backbone-frozen baseline ZEB-136 used) with `--allow-partial-init` so the new `engram_skip_router.W_align` (zero) and `engram_skip_router.log_alpha` (= log(0.1)) start from their constructor's safe-init values. 2000 steps each, batch=4, seq=2048, bf16 mixed precision, `--engram-vcontrast` + `--engram-qdiv` aux losses still active per spec §4.2.

---

## Results

### val_loss matrix (final, step 2000)

|  | Real oracle | Shuffled oracle | Δ-diff (real − shuf) |
| --- | --- | --- | --- |
| **Router off, KL off** (cells 1, 2) | 4.5546 | 4.5546 | **+0.0000** |
| **Router on, KL on** (cells 3, 4) | 4.5636 | 4.5637 | **−0.0001** |
| Δ vs no-router baseline | +0.0090 | +0.0091 | — |

**Two observations from the matrix alone**:

1. The router-off baseline reproduces ZEB-136's cells 1+2 to 4 decimal places (4.5546 vs ZEB-136's 4.5546 / 4.5544). Sanity check passes — the data path and frozen-backbone init are unchanged.
2. The router-on KL+CE cells got val_loss ~0.009 nats *worse* than the no-router baseline, with cell 3 vs cell 4 essentially identical. This is the inverse of what would constitute a positive ZEB-139 result.

### Cell 3 vs Cell 4 fingerprint (the discriminator)

Per spec §11's intra-experiment discriminator table, cell 3 vs cell 4 separates "KL signal is content-dependent" (real teacher info actually helping, the clean positive result) from "KL signal is content-independent" (KL forcing sharp output regardless of input — the cheap-win confound).

Pulled from `forensics/router_on_kl.txt` (probe: `scripts.probe_skip_to_logit`):

```text
real: log_alpha=-1.7360  alpha=exp=0.1762  ||W_align||_F=1.3461
shuf: log_alpha=-1.7342  alpha=exp=0.1765  ||W_align||_F=1.3471

cross_run_cos engram_logits  =  +0.9999
max LM-head row |cos|        =  0.7779
engram_logit_entropy (nats)  =  10.3467  (log(vocab) = 10.3735)
```

| Fingerprint metric | Spec §7 threshold (broken if…) | ZEB-136 (no KL) | ZEB-139 (KL=0.5, real) | Verdict |
| --- | --- | --- | --- | --- |
| `engram_logit_entropy` | < log(V) − 0.1 = 10.27 | 10.3735 (= log V) | **10.3467** | **HOLDS** (Δ from log V = 0.027, well above the 0.1 threshold for "broken") |
| `α` | outside [0.14, 0.20] | 0.1644 | **0.1762** | **HOLDS** (still inside attractor band) |
| Cross-run cosine (real vs shuf, router) | < 0.7 | +0.7979 | **+0.9999** | **HOLDS** + WORSE — KL drove the two routers to converge |
| Δ-diff (real − shuf val_loss) | ≥ +0.001 nats | +0.0002 | **−0.0001** | **HOLDS** + slight reverse |
| `W_align` Frobenius drift | > 2× init (init = 0) | 1.91 | **1.35** | **HOLDS** + smaller — KL kept W_align contained |

**All five thresholds say the attractor HOLDS.** And `cross_run_cos = +0.9999` is the dispositive result for the cheap-win discriminator: when two router models trained on completely different per-position teacher targets (real vs shuffled-via-permutation) end up producing essentially the same output distribution to 4-decimal cosine, the model is matching SOMETHING content-independent — almost certainly the corpus-wide token-frequency average that the Welford-mean teacher logits encode after enough position averaging.

That same `cross_run_cos` jumping from 0.80 (no KL) to 1.00 (with KL=0.5) is the mechanism: KL pressure pulls both routers to the SAME target distribution. The "real" and "shuf" sidecars contain the same set of per-row teacher distributions just at different row indices — the KL signal therefore averages out to "match the corpus distribution somehow", which is identical regardless of how rows are permuted.

### KL trajectory

From the per-step CSV logs (`run3_router_on_real_kl.csv`):

```text
step    0  loss=2.9147  kl_loss=1.2697  alpha=0.10  W_align=0  (init)
step  300  loss=3.1555  kl_loss=1.2705  (essentially flat — W_align still ~0, gradient through alpha is zero by construction)
step  600  loss=2.9575  kl_loss=1.2705  (alpha=0.1, W_align starting to grow under small alpha gradient)
step  900  loss=2.9254  kl_loss=1.2678  (KL begins moving)
step 1200  loss=2.9449  kl_loss=1.2587
step 1500  loss=2.9853  kl_loss=1.2453
step 1800  loss=2.9885  kl_loss=1.2212
final     loss=4.5636 (val)  KL trajectory: 1.27 → 1.22 nats over 2000 steps (Δ = −0.05 nats)
```

The KL did decrease monotonically — the router IS learning to better match the teacher distribution. But the magnitude is small (~4% relative drop) and the destination is content-independent: cell 4 (shuffled sidecar) shows the IDENTICAL trajectory (`kl_loss` 1.2709 → 1.2209 in the same number of steps). Both runs are converging toward the same "average TinyLlama distribution" target, which is not what the experiment hoped to find.

### Pair A baseline forensic (router-off cells)

Reproduces ZEB-136's standard η-B capgap battery on cells 1+2 (full output in `forensics/router_off_no_kl.txt`). All ten ZEB-130 probes (D/P/E/M/C/W/A/X/Q-overlap/V-rank) within noise of ZEB-136's prior values; cross-run cos at L2 = +0.87 / L5 = +0.80 (matching ZEB-136's known content-poor baseline). Confirms the no-router data path didn't drift between ZEB-136 and ZEB-139.

---

## Verdict matrix (this experiment × ZEB-138)

Per spec §11:

| This (KL-retrofit) | ZEB-138 (same-arch teacher, CE-only) | Combined interpretation |
| --- | --- | --- |
| Holds | Holds | **Structural ceiling confirmed** at 40M (Gemini §7 steelman) — neither objective shift nor teacher-arch shift escapes the attractor; the 40M frozen-backbone linear pipeline is the binding constraint. Multi-layer non-linear `W_align` OR end-to-end retraining without freezing is the recommended next axis. |
| Holds | Breaks | **Teacher-arch dominates, objective insufficient** — ZEB-138's same-arch decode break is the load-bearing axis; pursue same-arch teacher + capgap as the substrate, deprioritize KL+CE. |

ZEB-138's verdict is pending KRILE's Harmony-474M handoff and the corresponding 4-cell run on AVALON. **ZEB-139's contribution to the matrix is now locked in as "Holds".**

---

## Open questions and next-step recommendations

### 1. λ-sweep (spec §12 question 1)

The spec says "If no break at 0.5, try 0.9 once before concluding." This is worth doing for completeness, but the `cross_run_cos = +0.9999` result strongly suggests a higher λ would just intensify the convergence to the average distribution — it cranks up the same lever that's already saturating. **Recommendation: run a single λ=0.9 cell-3 + cell-4 pair (~30 min on AVALON) to nail down the λ-sensitivity signal, then close the door on the KL-only axis at 40M.**

### 2. Same-arch teacher + KL (spec §10 follow-up)

The 2×2's fourth cell (same-arch teacher AND KL term) is contingent on either ZEB-139 or ZEB-138 yielding signal. Since ZEB-139 didn't, and ZEB-138 is pending, this remains "wait for ZEB-138." If ZEB-138 also holds, the 2×2 is closed (structural ceiling) and same-arch+KL becomes redundant. If ZEB-138 breaks, same-arch+KL becomes the natural follow-up to test whether KL adds to the same-arch signal.

### 3. The Gemini §7 steelman — multi-layer non-linear `W_align`

If ZEB-138 also holds, the Gemini Deep Research findings (§7) recommend abandoning the single-layer-linear `W_align` in favor of either a multi-layer non-linear projection (more capacity in the alignment path) or unfreezing the backbone (resolves the "frozen 40M can't decode high-dim teacher features" steelman). Both are substantially more invasive than ZEB-139 was. Multi-layer `W_align` is probably the cheaper try-first.

### 4. Diagnostic bonus: KL trajectory IS learning, just not usefully

Worth flagging that KL did monotonically decrease (1.27 → 1.22) and `max LM-head row cos` jumped from 0.22 (ZEB-136) to 0.78 (ZEB-139). The router IS aligning with vocab directions — it's just aligning ALL positions with the SAME average direction (cross_run_cos = 1.0). A future variant could try a temperature on the router-side softmax (spec §12 question 2) or a per-token KL mask that down-weights frequent-token positions — both might force the router to pay attention to per-position content rather than averaging it out. These are speculative; the cleaner next move is the λ-sweep + ZEB-138 result.

---

## Artifacts

All under `/home/zebli/work/LOCAL/zeb139/`:

- **Oracle**: `artifacts/oracle_tinyllama_10k.safetensors` (4.9 MB, [10K, 128] f32) and `_shuffled_seed0.safetensors`
- **Teacher-logits sidecar**: `artifacts/oracle_tinyllama_10k_teacher_logits.safetensors` (611 MB, [10K, 32K] bf16) and `_shuffled_seed0_teacher_logits.safetensors`
- **Stats**: `artifacts/oracle_tinyllama_10k.safetensors.stats.json` (PCA explained variance, populated rows, hash seeds)
- **Per-cell training logs (CSV)**: `logs/run{1..4}_*.csv` (200 rows × 36 columns each, including the new `kl_loss` column)
- **Per-cell checkpoints**: `checkpoints/zeb139_router_{off,on}_{real,shuf}{,_kl}/checkpoint.pt`
- **Forensic outputs**: `forensics/router_off_no_kl.txt` (full 10-probe battery, pair A) and `forensics/router_on_kl.txt` (skip-to-logit diagnostics, pair B)
- **Scripts**: `scripts/shuffle_oracle_and_sidecar.py`, `scripts/run_4cell_matrix.sh` (cells 1-4), `scripts/run_cells_3_and_4.sh` (re-run after stale-checkout fix), `scripts/run_forensics.sh`

ZEB-136's prior forensics (`/home/zebli/work/LOCAL/zeb136/forensics/router_on.txt`) are the direct comparison point for the ZEB-139 (KL+CE) vs ZEB-136 (CE-only) contrast.

---

## Operational notes

- The 4-cell matrix's first attempt failed at cell 3 because the local main repo dir was checked out on `zeblith/zeb-138-same-arch-teacher` (stale, predates PR #257). The venv's `ct87` editable install therefore imported a `train.py` without the `--engram-skip-to-logit` / `--kl-lambda` flags. Cells 1+2 succeeded incidentally (no-router code path is identical across branches). Resolved by `git checkout main && git pull` and re-running cells 3+4 only (each cell init's independently from `zeta_ctrl_2048`, so no chaining was lost).
- Total wall time for the experiment: ~6h oracle extraction + ~30 min cells 1+2 + ~30 min cells 3+4 + a few min for forensics. The spec §8 estimate of "4-6h end-to-end" was off by ~3× on the oracle extraction (the new logits-Welford accumulator is the dominant cost); the matrix + forensics matched spec.
