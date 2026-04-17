# Experiment ι-Q-diversity: MoE load-balancing auxiliary loss for engram retrieval (ZEB-130)

**Date:** 2026-04-17
**Issue:** ZEB-130 (engram capacity-gap investigation)
**Parent:** ZEB-102 (Engram table quality)
**Depends on:** PR #247 (η-B capgap infrastructure), PR #250 (θ-V-contrast + aux-sink pattern), PR #249 (forensic probes), PR #251 (corrected (X) probe — in flight)
**Status:** Design approved, awaiting implementation plan

## Motivation

PR #250 landed θ-V-contrast: an auxiliary loss that penalizes cosine alignment between injection outputs computed from real vs value-shuffled retrieval. The hypothesis was that the η-B + ζ-ctrl content-invariance pathology at 40M (oracle contains information but training metrics are invariant to content shuffles) was V-side: V learned to produce similar injections regardless of what Q retrieved, so contrastive pressure on V would force content-routing.

KRILE ran θ at 40M on 2026-04-17 matched to η-B hyperparameters. **θ failed to content-route:**

| Metric | η-B (baseline) | θ (V-contrast) | Target | Verdict |
|---|---|---|---|---|
| Cross-run cos (L2) | +0.34 | +0.72 | < 0.15 | Worse |
| Cross-run cos (L5) | +0.63 | +0.84 | < 0.15 | Worse |
| (D) unique rows (L2) | 5.45% | 0.626% | — | Collapsed 8× |
| (D) unique rows (L5) | 6.01% | 0.258% | — | Collapsed 23× |
| Δ-removal diff (real − shuffled) | +0.0001 | +0.000183 | > 0.002 | Missed |
| vcontrast aux final | ~0.5 expected | 0.000736 | ~0.1 | Collapsed 100× |

KRILE's weight inspection after θ:

| Module | η real Fro | θ real Fro | Effective rank | σ_min |
|---|---|---|---|---|
| L2 v_proj | 14.32 | 16.23 | 123.4/128 | 0.74 |
| L5 v_proj | 14.28 | 17.04 | 123.4/128 | 0.77 |
| L2 o_proj | 22.65 | 24.91 | 416.7/512 | 6e-4 |
| L5 o_proj | 22.85 | 26.08 | 419.5/512 | 7e-4 |

**V is healthy.** θ's `v_proj` Frobenius norm is actually larger than η's (16-17 vs 14); rank is full; no collapse to bias or low-rank subspace. The H1 (bias-path) and H2 (V-rank-collapse) hypotheses are ruled out.

**H3 (Q-collapse) is the working hypothesis:** Q queries a narrow signature-separable row neighborhood (θ retrieves ~170 rows at L5 across 64k query positions); V orthogonalizes within that tiny subset (which is what actually drove the vcontrast aux collapse); CE loss is weakly satisfied because the joint optimum `(Q→narrow, V→orthogonal-within-narrow)` is a local attractor. V-contrast pressure *accelerated* Q-collapse rather than preventing V-shortcut: the easiest way to make real-vs-shuffled outputs orthogonal is to have V operate on a narrow set of rows where orthogonalization is tractable.

**ι-Q-diversity** adds an MoE-style load-balancing auxiliary loss on Q's row-usage marginal distribution. Penalizes both hard-selection concentration (`f[i] = fraction of (B·L·k) selections on row i`) and attention-weight concentration (`P[i] = fraction of total attention mass on row i`) simultaneously. Independent of V-contrast — can run alone (ι₁) or together with V-contrast (ι₂) to test whether closing both shortcut paths jointly unblocks content-routing at 40M.

## Design decisions summary

| Decision | Choice | Rationale |
|---|---|---|
| Experiment scope | Run ι₁ (Q-div alone on η-B) AND ι₂ (Q-div + V-contrast on θ setup) | 4-cell matrix with existing η/θ gives causal attribution; each run ~3.5 min on KRILE |
| Loss formulation | MoE-style `L = N · Σᵢ fᵢ · Pᵢ` | Penalizes the exact observed failure mode (hard-selection concentration); literature-proven (Shazeer '17, Fedus '21) |
| Usage aggregation | Per-batch, per-injection-layer, joint over heads | Mirrors V-contrast's pattern; per-head would over-constrain |
| Target distribution | Uniform over all N rows | Gives MoE's natural floor of 1.0; λ_Q controls how hard to push |
| λ_Q default | 0.01 | MoE-standard starting point; order of magnitude smaller than V-contrast's 0.1 |
| Warmup schedule | Linear 0 → λ_Q over first 200 steps | Same as V-contrast; avoids fighting initialization |
| Architecture | Composable aux-loss sinks on unified `GatedEngramInjection` | Independent feature toggles (vcontrast + qdiv) compose naturally; removes subclass proliferation |
| Numerical stability | No epsilon needed; all quantities non-negative; bf16 training safe | `bincount` + `scatter_add_` are stable |
| Checkpoint state | No new RNG state (Q-div is deterministic given forward) | V-contrast's `shuffle_generator` handling unchanged |

## Section 1: Architecture

Single unified `GatedEngramInjection` class with optional `vcontrast_sink` and `qdiv_sink` kwargs. Both auxiliary losses are side-channels: the forward pass appends per-layer scalars to caller-provided lists; the training loop drains, scales by λ × warmup, and adds to the total loss.

**Files touched:**

| File | Change |
|---|---|
| `training/ct87/engram.py` | Refactor: unify `GatedEngramInjection`, remove `ContrastiveGatedEngramInjection`, add `return_attn` kwarg to `_attention_block`, add module-level `compute_qdiv_aux` helper |
| `training/ct87/model.py` | Add `engram_qdiv_{enabled,lambda,warmup_steps}` to `HarmonyModelConfig`; add `_qdiv_aux_losses` list to `HarmonyModel`; add two new presets (`tiny_engram_xattn_capgap_qdiv`, `tiny_engram_xattn_capgap_vcontrast_qdiv`); validation in `__post_init__` |
| `training/ct87/train.py` | Add Q-div aux drain + λ warmup + total-loss contribution; add `qdiv_*` CSV columns (conditional on `engram_qdiv_enabled`, per-layer); add `--engram-qdiv{,-lambda,-warmup-steps}` CLI flags with symmetric flag/preset rejection |
| `training/tests/test_iota_q_diversity.py` | NEW — ~17 tests: unit tests for `compute_qdiv_aux`, sink-configuration behavior matrix, API tests for `_attention_block`, CSV/CLI subprocess smokes |

## Section 2: Loss math

### Formula

Per injection-layer per-batch:

```
f[i]   = count(topk_idx == i) / (B · L · k)                   # hard frequency, detached
P[i]   = Σ_(b,l,h,j: topk_idx[b,l,j]=i) attn[b,l,h,j] / (B·L·H)   # soft mass, differentiable
L_qdiv = N · Σᵢ f[i] · P[i]                                   # scalar, ≥ 1.0 under uniform
```

Only `P` carries gradient. `f` is a non-differentiable frequency weight (no `.detach()` needed — `bincount` has no gradient path — but explicit detach is a safety net).

### Behavior at key distributions

| Distribution | Loss |
|---|---|
| Uniform over all N | 1.0 (floor) |
| Uniform over S < N rows | N/S |
| Concentrated on single row | N |
| θ observed (S≈169 at L5, N=10000) | ~60 |

### Implementation

Module-level helper in `engram.py`:

```python
def compute_qdiv_aux(
    topk_idx: torch.Tensor,   # [B, L, k] int64, values in [0, N)
    attn: torch.Tensor,       # [B, L, H, k] float, softmaxed over last dim
    table_size: int,          # N
) -> torch.Tensor:            # scalar loss
    """MoE-style load-balancing auxiliary loss over retrieval row usage.

    Minimized when Q spreads retrieval uniformly over table rows.
    Only P (soft attention) carries gradient — f (hard count) is
    non-differentiable and serves as a frequency weight.
    """
    B, L, k = topk_idx.shape
    H = attn.shape[2]

    f = torch.bincount(
        topk_idx.reshape(-1), minlength=table_size,
    ).to(attn.dtype) / (B * L * k)

    idx = topk_idx.unsqueeze(2).expand(B, L, H, k).reshape(-1)
    P = torch.zeros(table_size, device=attn.device, dtype=attn.dtype)
    P.scatter_add_(0, idx, attn.reshape(-1))
    P = P / (B * L * H)

    return table_size * (f * P).sum()
```

### Gradient path

`scatter_add_` → `attn` → `F.softmax(scores)` → `scores = q·k/√D + retrieval_bias_weight · topk_sims`. Three weight tensors receive gradient:

- `EngramCrossAttention.q_proj.weight` (through `q`)
- `EngramCrossAttention.k_proj.weight` (through `k_tensor`)
- `EngramCrossAttention.retrieval_query_proj.weight` (through `topk_sims`, which is differentiable w.r.t. `q_norm = F.normalize(retrieval_query_proj(hidden))`)

The third path is the most load-bearing — it's how Q's retrieval preference itself gets pushed toward diversity.

## Section 3: Module refactor (`engram.py`)

### 3a. `_attention_block` — add `return_attn` kwarg

```python
def _attention_block(
    self,
    hidden_state: torch.Tensor,
    retrieved: torch.Tensor,
    topk_sims: torch.Tensor,
    return_attn: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    B, L, _ = hidden_state.shape
    H, D, k = self.num_heads, self.head_dim, self.k_retrieved

    retrieved = self.retrieval_norm(retrieved)
    q = self.q_norm(self.q_proj(hidden_state).view(B, L, H, D))
    k_tensor = self.k_norm(self.k_proj(retrieved).view(B, L, k, H, D))
    v_tensor = self.v_proj(retrieved).view(B, L, k, H, D)

    scores = torch.einsum("blhd,blkhd->blhk", q, k_tensor) / (D ** 0.5)
    scores = scores + self.retrieval_bias_weight * topk_sims.unsqueeze(2)
    attn = F.softmax(scores, dim=-1)

    out = torch.einsum("blhk,blkhd->blhd", attn, v_tensor)
    if self.use_head_gates:
        out = out * torch.sigmoid(self.head_gates).view(1, 1, H, 1)
    out = self.o_proj(out.reshape(B, L, H * D))

    if return_attn:
        return out, attn
    return out
```

Bit-identical behavior when `return_attn=False` (covered by regression test).

### 3b. `GatedEngramInjection` — unify with composable aux sinks

Remove `ContrastiveGatedEngramInjection`. Absorb its logic into `GatedEngramInjection` as a conditional branch gated on `self._vcontrast_sink is not None`.

```python
class GatedEngramInjection(nn.Module):
    """Gated engram cross-attention injection with optional training-only
    auxiliary losses.

    Aux losses attach via caller-supplied sink lists:
    - vcontrast_sink: V-contrastive aux (penalizes content-blindness at V).
    - qdiv_sink: Q-div aux (MoE load balancing on retrieval marginal).

    Sinks are drained per optimizer step by the training loop. Passing None
    disables the corresponding aux path entirely (no extra compute).
    """

    def __init__(
        self,
        engram_xattn: EngramCrossAttention,
        alpha_init: float = 0.0,
        *,
        vcontrast_sink: list[torch.Tensor] | None = None,
        qdiv_sink: list[torch.Tensor] | None = None,
        shuffle_generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        self.engram_xattn = engram_xattn
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self._vcontrast_sink = vcontrast_sink
        self._qdiv_sink = qdiv_sink
        self._shuffle_generator = shuffle_generator

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        xattn = self.engram_xattn
        need_vcontrast = self.training and self._vcontrast_sink is not None
        need_qdiv = self.training and self._qdiv_sink is not None
        need_idx = need_vcontrast or need_qdiv
        need_attn = need_qdiv

        # Single retrieval covers both aux paths when both are enabled.
        if need_idx:
            retrieved, topk_sims, topk_idx = xattn.retrieve_topk(
                hidden_state, return_indices=True,
            )
        else:
            retrieved, topk_sims = xattn.retrieve_topk(hidden_state)

        # Main injection forward — conditionally capture attention.
        if need_attn:
            inj_real, attn_weights = xattn._attention_block(
                hidden_state, retrieved, topk_sims, return_attn=True,
            )
            self._qdiv_sink.append(
                compute_qdiv_aux(topk_idx, attn_weights, xattn.table.shape[0])
            )
        else:
            inj_real = xattn._attention_block(hidden_state, retrieved, topk_sims)

        # V-contrast aux — shuffled-value branch.
        if need_vcontrast:
            N = xattn.table.shape[0]
            gen = self._shuffle_generator
            if gen is not None:
                perm = torch.randperm(N, generator=gen, device=gen.device)
                if perm.device != xattn.table.device:
                    perm = perm.to(xattn.table.device)
            else:
                perm = torch.randperm(N, device=xattn.table.device)
            shuf_idx = perm[topk_idx]
            retrieved_shuf = xattn.table[shuf_idx]
            inj_shuf = xattn._attention_block(hidden_state, retrieved_shuf, topk_sims)
            cos = F.cosine_similarity(inj_real, inj_shuf, dim=-1)
            self._vcontrast_sink.append((cos ** 2).mean())

        return hidden_state + torch.tanh(self.alpha) * inj_real
```

### 3c. Delete `ContrastiveGatedEngramInjection`

Remove the class. Update callsites in `HarmonyModel` to instantiate `GatedEngramInjection` directly with the appropriate sink kwargs. V-contrast tests from PR 250 test observable behavior (loss values, determinism, bit-identity under seeding) and survive the import-path update unchanged.

### Regression invariants

- `vcontrast_sink=None, qdiv_sink=None` → forward bit-identical to η-B baseline pre-refactor.
- `vcontrast_sink=list, qdiv_sink=None` → aux loss values match PR 250's seeded-determinism reference.
- `vcontrast_sink=None, qdiv_sink=list` → no second `_attention_block` call per forward.
- Both sinks set → each fires once per forward; values are independent.

## Section 4: Training-loop integration (`train.py`)

### 4a. Per-step drain

```python
# After loss = criterion(...), before loss.backward():
aux_vcontrast_total = None
aux_qdiv_total = None
lam_vcontrast = 0.0
lam_qdiv = 0.0

if config.engram_vcontrast_enabled:
    per_layer_vc = list(model._contrastive_aux_losses)
    model._contrastive_aux_losses.clear()
    if per_layer_vc:
        aux_vcontrast_total = torch.stack(per_layer_vc).sum()
        lam_vcontrast = lambda_schedule(
            step, config.engram_vcontrast_lambda, config.engram_vcontrast_warmup_steps,
        )

if config.engram_qdiv_enabled:
    per_layer_qd = list(model._qdiv_aux_losses)
    model._qdiv_aux_losses.clear()
    if per_layer_qd:
        aux_qdiv_total = torch.stack(per_layer_qd).sum()
        lam_qdiv = lambda_schedule(
            step, config.engram_qdiv_lambda, config.engram_qdiv_warmup_steps,
        )

total_loss = loss
if aux_vcontrast_total is not None:
    total_loss = total_loss + lam_vcontrast * aux_vcontrast_total
if aux_qdiv_total is not None:
    total_loss = total_loss + lam_qdiv * aux_qdiv_total

total_loss.backward()
```

### 4b. CSV columns (conditional, per-layer, sorted)

```python
qdiv_cols: list[str] = []
if config.engram_qdiv_enabled:
    qdiv_cols = [
        "qdiv_aux_loss",
        *(f"qdiv_aux_L{i}" for i in sorted(config.engram_inject_layers or ())),
        "qdiv_lambda",
    ]
expected_header = [
    # ... 16 base columns ...
    *vcontrast_cols,
    *qdiv_cols,
    # ... hg_slot_N columns ...
]
n_hg_slots = len(expected_header) - (16 + n_vcontrast_cols + n_qdiv_cols)
```

Per-layer accumulator uses `zip(..., strict=True)` with `sorted(config.engram_inject_layers)`, matching PR 250's pattern.

### 4c. Console output

```python
if config.engram_qdiv_enabled and aux_qdiv_total is not None:
    print(
        f"  [qdiv] total={aux_qdiv_total.item():.4f} "
        f"λ={lam_qdiv:.4f} "
        f"per-layer={[f'{x.item():.4f}' for x in per_layer_qd]}"
    )
```

End-of-run summary block for Q-div mirrors V-contrast's final block.

### 4d. CLI flags

```python
p.add_argument("--engram-qdiv", action="store_true",
    help="Enable Q-side load-balancing aux loss. Must match the selected preset.")
p.add_argument("--engram-qdiv-lambda", type=float, default=None,
    help="λ for Q-div aux loss. Requires --engram-qdiv + qdiv-enabled preset.")
p.add_argument("--engram-qdiv-warmup-steps", type=int, default=None,
    help="Warmup steps for Q-div λ. Requires --engram-qdiv + preset.")
```

Symmetric validation:

```python
if args.engram_qdiv != config.engram_qdiv_enabled:
    print("Error: --engram-qdiv must match the selected preset.", file=sys.stderr)
    sys.exit(1)
if not config.engram_qdiv_enabled and (
    args.engram_qdiv_lambda is not None
    or args.engram_qdiv_warmup_steps is not None
):
    print("Error: --engram-qdiv-{lambda,warmup-steps} require a Q-div preset "
          "+ --engram-qdiv.", file=sys.stderr)
    sys.exit(1)
```

### 4e. Checkpoint / resume

No changes. Q-div has no per-step RNG state. V-contrast's `capgap_shuffle_gen` capture/restore is untouched.

## Section 5: `HarmonyModelConfig` + presets (`model.py`)

### 5a. New fields

```python
@dataclass
class HarmonyModelConfig:
    # ... existing fields ...
    engram_qdiv_enabled: bool = False
    engram_qdiv_lambda: float = 0.01
    engram_qdiv_warmup_steps: int = 200
```

### 5b. `__post_init__` validation

```python
if self.engram_qdiv_enabled and not self.engram_xattn_enabled:
    raise ValueError(
        "engram_qdiv_enabled requires engram_xattn_enabled=True; "
        "Q-div operates on retrieval softmax which only exists in the "
        "cross-attention engram path."
    )
if self.engram_qdiv_enabled and not self.engram_inject_layers:
    raise ValueError(
        "engram_qdiv_enabled requires at least one injection layer; "
        "configure engram_inject_layers before enabling Q-div."
    )
if self.engram_qdiv_lambda < 0:
    raise ValueError(
        f"engram_qdiv_lambda must be ≥ 0, got {self.engram_qdiv_lambda}"
    )
if self.engram_qdiv_warmup_steps < 0:
    raise ValueError(
        f"engram_qdiv_warmup_steps must be ≥ 0, got "
        f"{self.engram_qdiv_warmup_steps}"
    )
```

### 5c. Two new presets

```python
@staticmethod
def tiny_engram_xattn_capgap_qdiv() -> "HarmonyModelConfig":
    """ι₁: capgap baseline + Q-div only (no V-contrast).

    Ablation test: does load-balancing Q's retrieval distribution alone
    unstick content-invariance, or is V-side pressure also needed?
    """
    config = HarmonyModelConfig.tiny_engram_xattn_capgap()
    config.engram_qdiv_enabled = True
    config.__post_init__()
    return config


@staticmethod
def tiny_engram_xattn_capgap_vcontrast_qdiv() -> "HarmonyModelConfig":
    """ι₂: capgap + V-contrast + Q-div together.

    Combined shortcut-closure test: does pressuring V toward content-
    sensitivity AND Q toward diversity jointly content-route at 40M?
    """
    config = HarmonyModelConfig.tiny_engram_xattn_capgap_vcontrast()
    config.engram_qdiv_enabled = True
    config.__post_init__()
    return config
```

### 5d. Example KRILE invocations

```bash
# ι₁: Q-div only
python scripts/train.py \
    --preset tiny_engram_xattn_capgap_qdiv \
    --engram-qdiv \
    --seed 42 --seq-len 2048 --batch-size 4 --lr 9e-4 \
    --xattn-top-k 8 --max-steps 2000 \
    --init-from checkpoints/zeta_ctrl_2048/checkpoint.pt \
    --allow-partial-init \
    # ... other standard capgap flags ...

# ι₂: V-contrast + Q-div
python scripts/train.py \
    --preset tiny_engram_xattn_capgap_vcontrast_qdiv \
    --engram-vcontrast --engram-qdiv \
    --seed 42 --seq-len 2048 --batch-size 4 --lr 9e-4 \
    --xattn-top-k 8 --max-steps 2000 \
    --init-from checkpoints/zeta_ctrl_2048/checkpoint.pt \
    --allow-partial-init \
    # ... other standard capgap flags ...
```

## Section 6: Testing strategy (`training/tests/test_iota_q_diversity.py`)

### 6a. Unit tests for `compute_qdiv_aux` (5 tests)

| Test | Verifies |
|---|---|
| `test_uniform_selection_gives_unit_loss` | Uniform `topk_idx` + uniform `attn` → `L ≈ 1.0` (±1e-4, N=100, k=4) |
| `test_full_concentration_gives_n_loss` | All selections on row 0, all attention on slot 0 → `L == N` |
| `test_partial_concentration_intermediate` | Uniform over S<N rows → `L ≈ N/S` within sampling noise |
| `test_gradient_flows_to_attn_only` | `torch.autograd.gradcheck` on `L` w.r.t. `attn`; verify `f`'s bincount output has no `requires_grad` |
| `test_deterministic_same_input_same_loss` | Two identical calls → `torch.equal` |

### 6b. `GatedEngramInjection` sink-configuration matrix (4 tests)

| Test | Configuration | Verifies |
|---|---|---|
| `test_no_sinks_matches_pre_refactor_baseline` | No sinks | Output bit-identical to η-B; no extra `_attention_block` calls |
| `test_only_vcontrast_sink_matches_pr250` | vcontrast_sink only | Seeded values match PR 250 reference; no qdiv entries |
| `test_only_qdiv_sink_appends_one_per_forward` | qdiv_sink only | One scalar per forward; no shuffled-value branch runs |
| `test_both_sinks_fire_independently` | Both set | Both receive one tensor per forward; swapping one doesn't affect the other |

### 6c. `_attention_block` API tests (2 tests)

| Test | Verifies |
|---|---|
| `test_return_attn_shape` | Returned `attn` is `[B, L, H, k]`, sums to 1 along last dim |
| `test_return_attn_false_matches_default` | `return_attn=False` output `torch.equal` to `return_attn=True` tuple[0] |

### 6d. Multi-layer + training-loop integration (4 tests)

| Test | Verifies |
|---|---|
| `test_sinks_receive_ascending_layer_order` | Declared `engram_inject_layers=(5, 2)` still sinks in L2→L5 order |
| `test_csv_qdiv_cols_present_when_enabled` | Subprocess with ι₁ preset → CSV has `qdiv_aux_loss`, `qdiv_aux_L2`, `qdiv_aux_L5`, `qdiv_lambda` columns |
| `test_csv_qdiv_cols_absent_when_disabled` | Subprocess with η-B preset → CSV header has no `qdiv_*` columns |
| `test_cli_flag_preset_mismatch_rejected` | Subprocess `--preset tiny_engram_xattn_capgap --engram-qdiv` → exit non-zero |

### 6e. One-step preset smoke tests (2 tests)

| Test | Preset | Verifies |
|---|---|---|
| `test_iota_1_one_step_smoke` | `tiny_engram_xattn_capgap_qdiv` | One step completes; CSV written; qdiv aux > 0 |
| `test_iota_2_one_step_smoke` | `tiny_engram_xattn_capgap_vcontrast_qdiv` | One step completes; both vcontrast and qdiv columns populated |

### Test count + runtime

~17 tests. Unit + wrapper tests run < 1s on CPU. Subprocess tests use `tiny_*` configs, ~5-15s each. Full suite comfortably in CI.

### Explicit non-goals

- No mocking of `torch.bincount` / `scatter_add` (trusted ops).
- No gradient-magnitude sanity tests (gradcheck covers correctness; magnitude is training-time observation via CSV).
- No end-to-end "does it train" test (out of scope for CI; that's KRILE's job).

## Section 7: Success criteria + experiment plan

### 7a. Quantitative success criteria

| Metric | θ observed | ι target | Interpretation |
|---|---|---|---|
| (D) unique-row fraction L5 | 0.258% | **≥ 3%** | Q-collapse fixed |
| (D) unique-row fraction L2 | 0.626% | **≥ 5%** | Q-collapse fixed |
| Δ-removal diff (real − shuffled) | +0.000183 | **≥ +0.002** | Content-routing achieved |
| Cross-run cos L2 | +0.72 | **≤ 0.30** | No shared-amplifier direction |
| Cross-run cos L5 | +0.84 | **≤ 0.30** | No shared-amplifier direction |
| (X) corrected \|cos\| (post-PR #251) | TBD | **ideally ≤ 0.30** | V content-sensitivity confirmed |
| qdiv_aux_loss final | n/a | **≤ 3.0** | Q uses ≥ 3000 of 10000 rows (S ≥ 3000) |
| LM val loss | 4.5379 | **≤ 4.55** | Aux doesn't cripple main objective |

Success = (D) + (Δ-removal diff) + (cross-run cos) all met. (X) corrected informs interpretation, doesn't gate success.

### 7b. Partial-success rubric

| Observation | Interpretation |
|---|---|
| ι₁ succeeds, ι₂ ≈ ι₁ | Q-collapse was the sole bottleneck; V-contrast redundant |
| ι₁ fails, ι₂ succeeds | Both shortcut paths were active; joint closure needed |
| ι₁ partial, ι₂ succeeds | ι₂ is production config |
| Both fail | 40M cannot content-route through this architecture; pivot to hard-retrieval-gate or 90M scale |
| ι₁ succeeds, ι₂ fails | V-contrast hurts when Q is spread; revisit λ_V or drop V-contrast |

### 7c. KRILE runbook

| Step | Action | Owner | Prereq | Runtime |
|---|---|---|---|---|
| 1 | Re-run θ + η forensic with corrected (X) probe | KRILE | PR #251 merged | ~10s/ckpt |
| 2 | Share (X) reading; decide if ι₂ needs λ_V adjustment | Koya + KRILE | Step 1 | ~5 min discuss |
| 3 | Merge ι PR | Koya | ι PR approved | n/a |
| 4 | Train ι₁ (2000 steps, seed=42, matched η-B hparams) | KRILE | Step 3 | ~3.5 min |
| 5 | Train ι₂ (2000 steps, seed=42, matched θ hparams) | KRILE | Step 3 | ~3.5 min |
| 6 | Forensic on ι₁ checkpoint (all probes) | KRILE | Step 4 | ~10s |
| 7 | Forensic on ι₂ checkpoint (all probes) | KRILE | Step 5 | ~10s |
| 8 | Shuffle-control runs: ι₁-shuf, ι₂-shuf | KRILE | Steps 4-5 | ~7 min total |
| 9 | Compute Δ-removal (real − shuffled) for ι₁ and ι₂ | Koya | Step 8 | ~2 min |
| 10 | Results report, memory update, next-step decision via 7b | Koya | Step 9 | ~30 min |

Total KRILE GPU time ~20 min. Total Koya analysis ~1 hour.

### 7d. Live-monitoring signals during training

- **qdiv_aux trajectory**: should start ~30-60 (reflecting θ's ~S=169 initial state if init from ζ-ctrl-2048), descend toward 1-3 as Q spreads. Plateau at 1-3 = healthy.
- **qdiv_aux_L2 vs qdiv_aux_L5 divergence**: flags per-layer Q-quality asymmetry; informs possible per-layer λ in ι₁.₁ follow-up.
- **LM loss (val)**: should stay within 2% of η-B. If climbs materially, λ_Q too high; abort + retry with 0.003.
- **V-contrast aux (ι₂ only)**: if stays large (> 0.05) rather than collapsing as in θ, that's a good sign — means Q-spread isn't immediately satisfying V-contrast trivially.
- **tanh(α) magnitude both layers**: ι targets ±0.18 on both layers, matching η-B. θ's L5 dropped to −0.08 (half η-B) — a warning sign.

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| λ_Q=0.01 too aggressive, LM loss climbs | Medium | Abort criterion: LM loss > 1.02× η-B at step 500; retry with 0.003 |
| λ_Q=0.01 too weak, D stays < 3% | Medium | Secondary criterion: if qdiv_aux stays > 10 at step 500, the loss isn't biting; retry with 0.03 |
| Q-div forces diversity but hurts per-query relevance (sim_gap drops) | Low-medium | Forensic (P) sim-gap reading after training; compare to η-B's +0.07-0.10 baseline |
| Removing `ContrastiveGatedEngramInjection` breaks downstream code we didn't audit | Low | `grep -r ContrastiveGatedEngramInjection` pre-PR; any hits updated in same PR |
| Q-div + V-contrast joint optimization oscillates | Low | CSV lets us watch both aux trajectories; if either climbs while other descends, we can kill one and re-run |
| bf16 numerical drift in `scatter_add_` at large N | Very low | All quantities non-negative; loss floor 1.0 is well above bf16 resolution |

## Out-of-scope (future work)

- **ι₃ λ_Q sweep**: deferred to a follow-up if ι₁/ι₂ both show partial progress.
- **ι₄ retrieval-noise**: replace a fraction of retrieved rows with Gaussian noise during training. Third shortcut-kill axis; queued if ι₁+ι₂ both fail.
- **Per-layer λ_Q**: separate λ for L2 and L5 if one layer converges and the other doesn't.
- **EMA over batches**: running estimate of row usage. Variance reduction; deferred unless per-batch turns out too noisy.
- **Hard retrieval-relevance gate**: different redesign axis; reserved if ι entirely fails.
- **AVALON parallel training**: infrastructure work; single-KRILE pipeline is fast enough for ι.
- **90M scale test**: final fallback if ι and hard-gate both fail; tests whether 40M backbone capacity is the bottleneck.
