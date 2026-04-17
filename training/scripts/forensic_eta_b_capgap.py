"""Forensic analysis for ZEB-130 η-B shuffle-oracle control.

Probes the injection mechanism on saved η-B capgap checkpoints to localize
the content-invariant utilization reported by the 2026-04-16 shuffle-oracle
control (Δ-removal identical to 4 decimals, g2 sign flip with preserved
magnitude).

Given two η-B capgap checkpoints (real-oracle + shuffled-oracle runs), loads
both models, runs them on a fixed val batch, and reports:

  (D) Retrieval diversity:  how many distinct top-k rows the retrieval
                            selects across all (batch, token) positions
  (P) Retrieval peakedness: gap between top-1 and top-k cosine similarities
  (E) Attention entropy:    per-head cross-attention softmax entropy
                            normalized by log(k)
  (M) Injection magnitude:  L2 norm of gate × xattn output, relative to the
                            hidden state residual stream norm
  (C) Cross-run alignment:  cosine similarity between real-run and
                            shuffled-run injection outputs at matched
                            (batch, token) positions
  (W) Within-run concentration: per-run resultant length (Fisher-von Mises
                            R) of per-token injection unit vectors. 1 = all
                            tokens emit the same direction (fixed amplifier);
                            0 = uniform distribution. Distinguishes "V is a
                            fixed projection" from "V is token-sensitive
                            but cross-run response distributions align."
  (A) Injection-vs-hidden:  cos between injection and the residual stream
                            it's added to. |A| ≈ 1 = residual-axis
                            amplifier; 0 = injection is orthogonal to the
                            current hidden direction
  (Q-overlap) Per-token     Monte-Carlo random-pair Jaccard of top-k row
              retrieval     indices. 1 ⇒ every token retrieves the same
              fixity        subset (content-blind Q despite possibly large
                            |S|); ~k/table_size ⇒ per-token retrieval is
                            content-dispersive. Distinguishes "Q has
                            broad marginal usage but fixed per-token
                            retrieval" from "Q varies per-token."
  (Q-occupancy) Uniformity  Coefficient of variation of row-occupancy
              of row usage  counts among USED rows. 0 ⇒ every used row is
                            retrieved equally often; high ⇒ a few rows
                            dominate. Orthogonal to |S| alone.
  (V-rank)    Effective     rank(V(E[S])) vs rank(E[S]) for the retrieved
              output rank   subset S. Reports hard rank (SVD threshold)
              of V          and effective rank (entropy of σ²). V losing
                            rank relative to E suggests V-output
                            compression — informs whether Q breadth alone
                            can unlock content-routing or whether V has a
                            deeper capacity limit.

These drive a verdict:

  (R) Retrieval-broken   — Q projection converged to content-insensitive
                           direction (low D, high E, small P on both runs)
  (I) Injection-broken   — retrieval healthy but injection path is content-
                           free (C ≈ 0, or M ≈ 0)
  (I*) Injection-broken, magnitude substantial — the closest fingerprint of
                           the shuffle-kill: material M but orthogonal C,
                           meaning the model learns an arbitrary per-run
                           auxiliary residual as long as some nonzero one
                           is available
  (B) Both-broken        — low D AND small M
  (H) Healthy            — peaked retrieval + material magnitude + high C

Key observation: `EngramCrossAttention.retrieve_topk()` is pure cosine top-k
over the full table, so row-permuting the table is mathematically a no-op
on the retrieved CONTENT given identical Q projections (the same rows are
selected, just at different j-labels). Any difference in behavior between
the two runs localizes entirely to training-trajectory divergence of the
learned projections. The (C) cross-run comparison is the definitive probe
for whether the two trajectories found the same useful direction or two
arbitrary ones.

The (X) cross-table within-run probe handles this correctly by sampling
the alt table from a random Gaussian with matched per-dim statistics — a
genuine CONTENT swap, not a row relabeling — so top-k on the alt table
returns truly different vectors from top-k on the primary table.

Usage (on KRILE):

    cd training
    python scripts/forensic_eta_b_capgap.py \\
        --real-ckpt checkpoints/eta_b_capgap/checkpoint.pt \\
        --shuffled-ckpt checkpoints/eta_b_capgap_shuffled/checkpoint.pt \\
        --real-table artifacts/oracle_mistral7b_10k.safetensors \\
        --shuffled-table artifacts/oracle_mistral7b_10k_shuffled_seed0.safetensors \\
        --val-data /data/fineweb-edu-poc/val \\
        --num-batches 2 --batch-size 4 --seq-len 2048 \\
        --k-retrieved 8

No training, no gradients. ~1 min on a 4090 to load both checkpoints and
forward a handful of val batches.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

_REPO_TRAINING = Path(__file__).resolve().parent.parent
if str(_REPO_TRAINING) not in sys.path:
    sys.path.insert(0, str(_REPO_TRAINING))

from ct87.engram import EngramCrossAttention, GatedEngramInjection  # noqa: E402
from ct87.model import HarmonyModel, HarmonyModelConfig  # noqa: E402
from ct87.train import make_hf_dataloader  # noqa: E402

# train.py saves numpy RNG state alongside the torch state_dict for run
# reproducibility. That state pickles through numpy's _reconstruct factory,
# which is not on torch's default weights_only=True allow-list. Enumerate
# the minimum set of numpy globals needed to round-trip the saved RNG so we
# can load real capgap checkpoints without falling back to weights_only=False
# (which would re-open the arbitrary-code-execution surface that round 2 of
# PR 248 review closed). _reconstruct moved from numpy.core to numpy._core
# in numpy 2.0; import path try-order covers both.
try:
    from numpy._core.multiarray import _reconstruct as _NUMPY_RECONSTRUCT  # noqa: E402
except ImportError:  # numpy 1.x
    from numpy.core.multiarray import _reconstruct as _NUMPY_RECONSTRUCT  # noqa: E402

# numpy 2.x introduced concrete scalar-dtype classes (numpy.dtypes.UInt32DType
# etc.) that appear in the pickle stream of any ndarray/RNG-state payload.
# Without allow-listing them, weights_only=True refuses with errors like
# "got <class 'numpy.dtypes.UInt32DType'>". Enumerate common numeric dtypes;
# getattr guards so this still works on numpy 1.x (no numpy.dtypes module).
_NUMPY_DTYPE_CLASS_NAMES: tuple[str, ...] = (
    "BoolDType",
    "Int8DType", "Int16DType", "Int32DType", "Int64DType",
    "UInt8DType", "UInt16DType", "UInt32DType", "UInt64DType",
    "Float16DType", "Float32DType", "Float64DType",
)
_numpy_dtype_classes: list = []
_numpy_dtypes_mod = getattr(np, "dtypes", None)
if _numpy_dtypes_mod is not None:
    for _name in _NUMPY_DTYPE_CLASS_NAMES:
        _cls = getattr(_numpy_dtypes_mod, _name, None)
        if isinstance(_cls, type):
            _numpy_dtype_classes.append(_cls)

_TORCH_LOAD_SAFE_GLOBALS: list = [
    HarmonyModelConfig,
    _NUMPY_RECONSTRUCT,
    np.ndarray,
    np.dtype,
    *_numpy_dtype_classes,
]


def _positive_int(value: str) -> int:
    """argparse type that rejects <= 0 (protects downstream divide-by-zero / topk)."""
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {ivalue}")
    return ivalue


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Forensic analysis for ZEB-130 η-B shuffle-oracle control.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--real-ckpt", required=True, type=Path,
                   help="η-B checkpoint trained with the real oracle table.")
    p.add_argument("--shuffled-ckpt", required=True, type=Path,
                   help="η-B checkpoint trained with the row-shuffled oracle table.")
    p.add_argument("--real-table", required=True, type=Path,
                   help="Safetensors oracle table the real checkpoint was trained with.")
    p.add_argument("--shuffled-table", required=True, type=Path,
                   help="Safetensors oracle table the shuffled checkpoint was trained with.")
    p.add_argument("--val-data", required=True, type=str,
                   help="Validation HF dataset path (same format as --val-data in train.py).")
    p.add_argument("--batch-size", type=_positive_int, default=4)
    p.add_argument("--seq-len", type=_positive_int, default=2048)
    p.add_argument("--num-batches", type=_positive_int, default=2,
                   help="Number of val batches to forensic over. 2 is usually enough "
                        "since each batch is B*L = thousands of tokens.")
    p.add_argument("--k-retrieved", type=_positive_int, default=8,
                   help="Top-k retrieval parameter. Must match the training run's "
                        "--xattn-top-k value for the forensic to reflect the trained model.")
    # retrieval_bias_weight and retrieval_temperature are constructor-only
    # floats on EngramCrossAttention — plain Python attributes, not
    # nn.Parameters or persistent buffers. load_state_dict cannot restore
    # them, so they must be supplied at construction time to match the
    # training run. Defaults mirror train.py and the module constructor:
    # bias_weight=1.0, temperature=None → 1/sqrt(engram_dim).
    p.add_argument("--retrieval-bias-weight", type=float, default=1.0,
                   help="Mirrors train.py --engram-xattn-retrieval-bias-weight. "
                        "Must match the value the checkpoints were trained with.")
    p.add_argument("--retrieval-temperature", type=float, default=None,
                   help="Softmax temperature for retrieve_softmax_weighted. Not "
                        "used in the forensic's inline forward (we reproduce "
                        "retrieve_topk), but passed to the constructor for parity "
                        "with the trained module. None = 1/sqrt(engram_dim).")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for val-batch selection. Identical across both models so they "
                        "see the same tokens — required for (C) cross-run comparison.")
    p.add_argument(
        "--alt-table-seed", type=int, default=42,
        help=(
            "Seed for the random-gaussian alt table used by the (X) cross-"
            "table within-run probe. Default 42. The probe forwards a single "
            "trained model against its training-primary table AND against a "
            "freshly-sampled random-gaussian table with matched per-dim mean "
            "and std (seeded by this argument). Comparing injection outputs "
            "is the direct test of V content-sensitivity: a content-blind V "
            "produces similar outputs for both tables; a content-sensitive V "
            "produces near-orthogonal outputs. Row-permuting the primary "
            "table is a no-op on top-k retrieved content (same rows, "
            "relabeled), so this probe instead samples genuinely different "
            "content at matched first and second moments."
        ),
    )
    return p.parse_args()


def load_capgap_model(
    ckpt_path: Path,
    table_path: Path,
    k_retrieved: int,
    retrieval_bias_weight: float,
    retrieval_temperature: float | None,
    device: str,
) -> tuple[HarmonyModel, HarmonyModelConfig, list[int]]:
    """Load an η-B capgap checkpoint and attach its engram injections.

    The table buffers (`table`, `table_normalized`) are non-persistent in
    EngramCrossAttention — they're initialized from the passed `table_path`
    and recomputed from it after `load_state_dict` via the module's
    `_load_from_state_dict` hook. Passing the correct table path for each
    run is what makes the forensic meaningful.
    """
    # weights_only=True refuses to deserialize arbitrary Python objects.
    # _TORCH_LOAD_SAFE_GLOBALS enumerates the custom types present in capgap
    # checkpoints: HarmonyModelConfig (the model config dataclass) and the
    # numpy globals needed to reconstruct saved RNG state. See the comment
    # near the module-level definition for the numpy rationale.
    with torch.serialization.safe_globals(_TORCH_LOAD_SAFE_GLOBALS):
        payload: dict[str, Any] = torch.load(
            str(ckpt_path), map_location="cpu", weights_only=True,
        )
    if "config" not in payload:
        raise KeyError(
            f"{ckpt_path} has no 'config' key — predates config persistence. "
            "Forensic requires a checkpoint saved after the config-persistence fix "
            "(commit 42eb4ff or later).",
        )
    config: HarmonyModelConfig = payload["config"]
    inject_layers = list(config.engram_inject_layers)
    if not inject_layers:
        raise ValueError(
            f"{ckpt_path}: config.engram_inject_layers is empty — not an η-B capgap "
            "checkpoint. Forensic expects the multi-layer gated-injection architecture.",
        )

    model = HarmonyModel(config)
    table = EngramCrossAttention.load_corpus_table(str(table_path))

    injections: dict[int, GatedEngramInjection] = {}
    for layer_idx in inject_layers:
        xattn = EngramCrossAttention(
            config, table,
            num_heads=config.num_query_heads,
            k_retrieved=k_retrieved,
            retrieval_bias_weight=retrieval_bias_weight,
            retrieval_temperature=retrieval_temperature,
        )
        injections[layer_idx] = GatedEngramInjection(
            xattn, alpha_init=config.engram_gate_init,
        )
    model.attach_gated_engram_injections(injections)

    # `.table` and `.table_normalized` on every EngramCrossAttention are
    # non-persistent buffers, so they're never in a checkpoint's state_dict
    # and will always appear in `missing`. Filter them out of both sides —
    # anything else missing or unexpected means the checkpoint is wrong for
    # this architecture, and forensic results would be run on partially-
    # initialized weights. Fail fast rather than print a warning nobody reads.
    missing, unexpected = model.load_state_dict(
        payload["model_state_dict"], strict=False,
    )
    real_missing = [k for k in missing if ".table" not in k]
    real_unexpected = [k for k in unexpected if ".table" not in k]
    if real_missing or real_unexpected:
        def _summary(keys: list[str]) -> str:
            head = keys[:5]
            tail = f" (+{len(keys) - 5} more)" if len(keys) > 5 else ""
            return f"{head}{tail}"
        raise RuntimeError(
            f"{ckpt_path.name}: incompatible state_dict load — forensic would "
            f"run on partially-initialized weights and produce invalid verdicts. "
            f"missing={_summary(real_missing)} unexpected={_summary(real_unexpected)}. "
            f"Check that --real-ckpt / --shuffled-ckpt point at η-B capgap "
            f"checkpoints matching the current architecture.",
        )

    model.eval().to(device)
    return model, config, inject_layers


class InjectionPreHook:
    """Forward pre-hook that captures the input hidden state to an injection.

    Reset `captured` to None between batches via `reset()` so a hook that
    fails to fire on a subsequent batch surfaces as a None-check failure
    instead of silently analyzing stale data from the prior batch.
    """

    def __init__(self) -> None:
        self.captured: torch.Tensor | None = None

    def __call__(
        self, _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...],
    ) -> None:
        self.captured = inputs[0].detach()

    def reset(self) -> None:
        self.captured = None


def _q_overlap_stats(
    topk_idx: torch.Tensor,
    num_pairs: int = 500,
    seed: int = 0,
) -> dict[str, float]:
    """Per-token retrieval fixity via Monte-Carlo random-pair Jaccard.

    Tests H3a: does Q retrieve ~the same rows for every token (content-blind
    per-token retrieval), or do different tokens retrieve genuinely different
    neighborhoods? If marginal |S| is large but random-pair Jaccard is near
    1, retrieval is fixed per-token — a pathology that the MoE-style marginal
    Q-diversity aux alone would NOT fix.

    Full O(N²) Jaccard is 256MB+ at N~8k; we instead sample `num_pairs`
    random index pairs and compute intersection via the broadcast
    `[M,k,1] == [M,1,k] → .any(-1).sum(-1)` identity (exploits that top-k
    returns distinct rows, so each pair_a value matches pair_b at most once).
    500 pairs gives SE ≈ 0.02 on mean Jaccard — plenty for a forensic
    verdict.

    Also reports the coefficient of variation (std/mean) of row-occupancy
    counts among USED rows — a per-row-usage uniformity measure that is
    orthogonal to |S|. Together, mean Jaccard + occupancy CV locate where
    Q's pathology lives: marginal vs per-token, usage vs set.

    Args:
        topk_idx: Retrieved row indices, shape [B, L, k]. Must be distinct
            per row (standard topk output).
        num_pairs: Monte-Carlo sample size. 500 is a good default.
        seed: RNG seed for pair sampling. Fixed default for reproducibility
            across a single forensic run; we resample with different seeds
            if inter-batch variance matters.
    """
    B, L, k = topk_idx.shape
    N = B * L
    flat = topk_idx.reshape(N, k)

    gen = torch.Generator(device=flat.device).manual_seed(seed)
    idx_a = torch.randint(0, N, (num_pairs,), generator=gen, device=flat.device)
    if N > 1:
        # Exclude self-pairs without a rejection loop: draw an offset in
        # [1, N) uniformly, then idx_b = (idx_a + offset) mod N. Guarantees
        # idx_b != idx_a and keeps the distribution uniform over the
        # off-diagonal pairs. Self-pairs would force Jaccard=1 and bias
        # the mean upward on small-N forensic runs (N=B*L can be as small
        # as ~100 in quick smoke tests).
        offset = torch.randint(1, N, (num_pairs,), generator=gen, device=flat.device)
        idx_b = (idx_a + offset) % N
    else:
        # N=1 is pathological (single token) — return zero Jaccard rather
        # than forcing self-pair comparisons.
        idx_b = idx_a
    pair_a = flat[idx_a]
    pair_b = flat[idx_b]

    match = (pair_a.unsqueeze(-1) == pair_b.unsqueeze(-2)).any(dim=-1)
    intersection = match.sum(dim=-1).to(torch.float32)
    union = (2 * k) - intersection
    jaccard = intersection / union.clamp_min(1.0)

    # Sparse count: allocates an output sized by the number of DISTINCT used
    # rows, not max(topk_idx)+1. torch.bincount would allocate a dense vector
    # sized by max_index+1 — wasteful for 10k+ row tables.
    _, counts = flat.reshape(-1).unique(return_counts=True)
    used = counts.to(torch.float32)
    if used.numel() > 1:
        occupancy_cv = (used.std(unbiased=False) / used.mean().clamp_min(1.0)).item()
    else:
        occupancy_cv = 0.0

    return {
        "q_overlap_random_pair_jaccard_mean": jaccard.mean().item(),
        "q_overlap_random_pair_jaccard_p50": jaccard.median().item(),
        "q_occupancy_cv_used": occupancy_cv,
    }


def _rank_stats_from_svdvals(
    s: torch.Tensor, m: int, n: int,
) -> tuple[int, float]:
    """Derive both hard rank and entropy-based effective rank from a
    pre-computed singular-value spectrum.

    Hard-rank tolerance matches torch.linalg.matrix_rank's default:
    ``max(M, N) * eps * max(s)``. Precomputing the SVD once and deriving
    both ranks avoids the double decomposition the naive path does
    (matrix_rank + svdvals each run their own SVD).

    Returns ``(hard_rank, effective_rank)``.
    """
    if s.numel() == 0:
        return 0, 0.0
    eps = torch.finfo(s.dtype).eps
    tol = max(m, n) * eps * s.max().item()
    hard_rank = int((s > tol).sum().item())
    s_sq = s * s
    total = s_sq.sum().clamp_min(1e-12)
    p = s_sq / total
    p_safe = p.clamp_min(1e-12)
    entropy = -(p * p_safe.log()).sum()
    effective_rank = float(torch.exp(entropy).item())
    return hard_rank, effective_rank


def _effective_rank(matrix: torch.Tensor) -> float:
    """Entropy-based effective rank: exp(H(σ² / ||σ²||)).

    Smoother than a hard SVD-threshold rank (no cliff at the cutoff).
    exp(H) equals the hard rank when σ is uniform-supported on k dimensions
    and decays smoothly as the singular-value spectrum concentrates.

    Thin wrapper over ``_rank_stats_from_svdvals`` — callers that already
    have the SVD should use the shared helper directly to avoid a second
    decomposition.
    """
    matrix_fp32 = matrix.to(torch.float32)
    s = torch.linalg.svdvals(matrix_fp32)
    _, effective_rank = _rank_stats_from_svdvals(s, *matrix_fp32.shape)
    return effective_rank


def _v_rank_stats(
    xattn: EngramCrossAttention,
    topk_idx: torch.Tensor,
) -> dict[str, float]:
    """V-output rank on the retrieved subset S.

    Tests H3c: does V compress the retrieved-row span into a narrower
    output subspace? θ L5's forensic reported |S|=198 rank-74 — V losing
    ~20 dims vs the retrieved input. Reproduces that comparison on any
    checkpoint: rank of `V(retrieval_norm(E[S]))` vs rank of
    `retrieval_norm(E[S])` itself.

    Reports both hard rank (tolerance mirroring
    ``torch.linalg.matrix_rank``'s default) and effective rank (entropy
    of normalized σ²). Effective rank is less threshold-sensitive and
    usually the more informative number for "is V preserving or
    compressing information?".

    Runs SVD exactly once per matrix — hard and effective rank are both
    derived from the same singular-value spectrum.
    """
    unique_rows = topk_idx.unique()
    e_s = xattn.table[unique_rows]
    # Cast to the layer-norm's parameter dtype so the forward runs under whatever
    # precision the loaded checkpoint uses. Fall back to the table's own dtype
    # when retrieval_norm has no parameters (e.g. Identity in unit tests).
    ln_dtype = next(
        (p.dtype for p in xattn.retrieval_norm.parameters()),
        xattn.table.dtype,
    )
    e_s_normed = xattn.retrieval_norm(e_s.to(ln_dtype))
    v_e_s = xattn.v_proj(e_s_normed)

    e_s_fp32 = e_s_normed.to(torch.float32)
    v_e_s_fp32 = v_e_s.to(torch.float32)
    e_svd = torch.linalg.svdvals(e_s_fp32)
    v_svd = torch.linalg.svdvals(v_e_s_fp32)
    e_rank, e_eff = _rank_stats_from_svdvals(e_svd, *e_s_fp32.shape)
    v_rank, v_eff = _rank_stats_from_svdvals(v_svd, *v_e_s_fp32.shape)

    return {
        "subset_size_S": int(unique_rows.numel()),
        "e_rank_hard": float(e_rank),
        "v_rank_hard": float(v_rank),
        "e_rank_effective": e_eff,
        "v_rank_effective": v_eff,
        "v_rank_ratio": v_rank / max(e_rank, 1),
    }


@torch.no_grad()
def analyze_injection(
    wrapper: GatedEngramInjection,
    hidden_state: torch.Tensor,
    inject_mult: float = 1.0,
) -> tuple[dict[str, float | int], torch.Tensor]:
    """Re-run the injection forward manually so we can capture intermediates.

    The public `EngramCrossAttention.forward` returns only the final residual,
    swallowing the retrieval distribution and attention weights. We reproduce
    the forward inline so the retrieval and attention tensors are available
    for diagnostics. Logic is byte-for-byte the same as
    `EngramCrossAttention.forward` as of commit b2c3a96.

    `inject_mult` mirrors `HarmonyModel.engram_inject_mult` — the runtime-only
    scalar the model multiplies the wrapper's output by before adding to the
    residual stream (default 1.0; `--zero-injection-eval` sets it to 0). The
    forensic's injection tensor reflects the actual residual added, not just
    the wrapper output, so (M) and cross-run cosine are comparable to what
    the forward path produces at inference time.

    Returns `(scalar_stats, injection)`:
      - `scalar_stats` has no tensors so callers can accumulate it cheaply
        across batches.
      - `injection` is the `inject_mult × gate × xattn_out` tensor
        ([B, L, hidden_dim]) for cross-run comparison; callers should
        consume it immediately and drop the reference.
    """
    xattn = wrapper.engram_xattn
    B, L, _ = hidden_state.shape
    H, D, k = xattn.num_heads, xattn.head_dim, xattn.k_retrieved

    q_proj_out = xattn.retrieval_query_proj(hidden_state)
    q_norm_ret = F.normalize(q_proj_out, dim=-1, eps=1e-8)
    sims_full = torch.einsum("ble,te->blt", q_norm_ret, xattn.table_normalized)
    topk_sims, topk_idx = sims_full.topk(k, dim=-1)
    retrieved = xattn.table[topk_idx]
    retrieved_normed = xattn.retrieval_norm(retrieved)

    q_attn = xattn.q_norm(xattn.q_proj(hidden_state).view(B, L, H, D))
    k_attn = xattn.k_norm(xattn.k_proj(retrieved_normed).view(B, L, k, H, D))
    v_attn = xattn.v_proj(retrieved_normed).view(B, L, k, H, D)

    scores = torch.einsum("blhd,blkhd->blhk", q_attn, k_attn) / (D ** 0.5)
    scores = scores + xattn.retrieval_bias_weight * topk_sims.unsqueeze(2)
    attn = F.softmax(scores, dim=-1)

    out = torch.einsum("blhk,blkhd->blhd", attn, v_attn)
    if xattn.use_head_gates:
        gate_heads = torch.sigmoid(xattn.head_gates).view(1, 1, H, 1)
        out = out * gate_heads
    xattn_out = xattn.o_proj(out.reshape(B, L, H * D))

    # Mirror the full residual add from HarmonyModel.forward:
    #   h = h + engram_inject_mult * wrapper(h)
    # where wrapper(h) = gate * xattn_out. Keeping the gate as a tensor (cast
    # to xattn_out's dtype) avoids a gratuitous GPU→CPU sync and matches
    # GatedEngramInjection.forward's dtype contract. The mult is folded in
    # here so `injection` reflects the actual residual added, not just the
    # wrapper's intermediate output — (M) and cross-run cos then correspond
    # to what the forward path produces at eval time.
    gate = torch.tanh(wrapper.alpha).to(dtype=xattn_out.dtype)
    injection = inject_mult * gate * xattn_out

    # (D) retrieval diversity — unique table rows touched across B*L*k slots.
    unique_rows = int(topk_idx.unique().numel())
    n_slots = B * L * k

    # (P) retrieval peakedness — how much bigger is top-1 than top-k?
    top1_sim_mean = topk_sims[..., 0].mean().item()
    topk_sim_mean = topk_sims[..., -1].mean().item()

    # (E) attention entropy — per-head softmax entropy averaged over tokens/heads.
    attn_safe = attn.clamp_min(1e-12)
    entropy_per_token_head = -(attn_safe * attn_safe.log()).sum(dim=-1)
    mean_entropy = entropy_per_token_head.mean().item()
    max_entropy = math.log(k)

    # (M) injection magnitude — relative to the hidden state it's added to.
    injection_norm = injection.norm(dim=-1)
    hidden_norm = hidden_state.norm(dim=-1).clamp_min(1e-9)
    magnitude_ratio = (injection_norm / hidden_norm).mean().item()

    # Unit-normalize once and reuse for both (W) and (A) — same per-token
    # direction vector feeds both probes.
    injection_flat = injection.reshape(-1, injection.shape[-1])
    injection_unit = F.normalize(injection_flat, dim=-1, eps=1e-8)

    # (W) within-run direction concentration. Two complementary statistics:
    #
    #   R  — Fisher-von Mises resultant length ||mean(injection_unit)||.
    #        R = 1 iff all per-token directions are identical (signed);
    #        R = 0 for uniform OR ±-symmetric distributions. Signed-direction
    #        concentration.
    #
    #   RMS_cos — sqrt(mean_ij (u_i · u_j)²). Computed without materializing
    #             the N×N Gram matrix: sum_ij (u_i·u_j)² = ||U^T U||_F² via
    #             the trace trick. RMS_cos = 1 iff all u_i collinear (any
    #             sign — captures the ±-symmetric case R misses); RMS_cos
    #             = 1/√D for uniform distribution on the sphere.
    #
    # L2 of η-B showed R ≈ 0 but likely RMS_cos ≈ 1 (gate sign flip + same
    # magnitude means ± the same axis, which R folds to 0). RMS_cos catches
    # that; R alone does not. Both reported so the joint reading is visible.
    mean_unit_raw = injection_unit.mean(dim=0)
    within_run_concentration = mean_unit_raw.norm().item()
    n_tokens = injection_unit.shape[0]
    if n_tokens > 0:
        # U^T U is [hidden_dim, hidden_dim] — never materializes the full N×N.
        gram_small = injection_unit.transpose(0, 1) @ injection_unit
        sum_sq_cos = (gram_small * gram_small).sum().item()
        mean_sq_cos = sum_sq_cos / (n_tokens * n_tokens)
        within_run_axis_rms = math.sqrt(max(mean_sq_cos, 0.0))
    else:
        within_run_axis_rms = 0.0

    # (A) injection-vs-hidden alignment. Does injection live on the residual-
    # stream axis it's added to? |cos| ≈ 1 means injection is literally ±
    # amplifying the hidden direction (strongest "amplifier not router"
    # reading). Low |cos| means injection writes into directions orthogonal
    # to the current residual — still could be content-blind, but not the
    # specific "residual amplifier" story.
    hidden_flat = hidden_state.reshape(-1, hidden_state.shape[-1])
    hidden_unit = F.normalize(hidden_flat, dim=-1, eps=1e-8)
    per_token_vs_hidden = (injection_unit * hidden_unit).sum(dim=-1)
    injection_vs_hidden_signed = per_token_vs_hidden.mean().item()
    injection_vs_hidden_abs = per_token_vs_hidden.abs().mean().item()

    q_overlap = _q_overlap_stats(topk_idx)
    v_rank = _v_rank_stats(xattn, topk_idx)

    scalar_stats: dict[str, float | int] = {
        "unique_rows": unique_rows,
        "n_slots": n_slots,
        "unique_rows_frac": unique_rows / max(n_slots, 1),
        "top1_sim": top1_sim_mean,
        "topk_sim": topk_sim_mean,
        "sim_gap": top1_sim_mean - topk_sim_mean,
        "mean_entropy": mean_entropy,
        "entropy_frac_of_max": mean_entropy / max_entropy if max_entropy > 0 else 0.0,
        "injection_vs_hidden_norm": magnitude_ratio,
        "gate": float(gate.detach().float().item()),
        "within_run_concentration": within_run_concentration,
        "within_run_axis_rms": within_run_axis_rms,
        "injection_vs_hidden_signed": injection_vs_hidden_signed,
        "injection_vs_hidden_abs": injection_vs_hidden_abs,
        **q_overlap,
        **v_rank,
    }
    return scalar_stats, injection


@torch.no_grad()
def analyze_cross_table(
    model: HarmonyModel,
    primary_table: torch.Tensor,
    alt_table: torch.Tensor,
    val_batches: list[torch.Tensor],
    layers: list[int],
) -> dict[int, dict[str, float]]:
    """Cross-table within-run probe: same model, same tokens, different table.

    Per batch, runs one full model forward to capture each injection layer's
    input hidden state via a forward pre-hook. For each layer, then computes
    TWO injection outputs from that single captured hidden state: one using
    the primary table's retrieval, one using the alt table's retrieval.
    Compares matched-position cosines. (The model itself is NOT run twice —
    doing so would let earlier-layer injections diverge between primary and
    alt, confounding the per-layer V content-sensitivity reading. Sharing
    the hidden state isolates the V projection under test.)

    A content-sensitive V produces different outputs when the retrieved
    vectors actually differ (cross-table |cos| near 0); a content-blind V
    produces near-identical outputs regardless of what gets retrieved (|cos|
    near 1). The secondary "random-baseline" probe (|cos| between two random
    tokens' V outputs within a single forward) gives a dispersion floor — if
    V's within-run token-to-token variation is already high, a low cross-
    table |cos| is less surprising; if V is token-concentrated, a low cross-
    table |cos| is a strong content-sensitivity signal.

    `alt_table` must be a GENUINE content swap, not a row permutation:
    row-permuting `primary_table` is mathematically a no-op on the
    retrieved content under cosine top-k (same rows get selected, just
    at different j-labels), so the probe would always report |cos| = 1.
    Callers should pass a random-gaussian `alt_table` with matched per-
    dim mean and std (see the main-function construction).

    Args:
        model: Trained HarmonyModel with `engram_injections` attached.
        primary_table: The table the model was trained against. Each
            model should be probed against ITS OWN training table —
            probing a shuffled-oracle model against the real oracle
            tests OOD behavior, not V content-sensitivity.
        alt_table: A content-different alt table of matching shape
            (typically random-gaussian with matched primary-table
            statistics; see caller in `run_forensic`).
        val_batches: Validation token batches (each [batch, seq_len]).
        layers: Injection-layer indices (typically [2, 5]).

    Returns:
        Per-layer dict with keys:
            'cross_table_cos_signed', 'cross_table_cos_abs',
            'within_run_random_pair_cos_abs' (the dispersion floor).
    """
    if primary_table.shape != alt_table.shape:
        raise ValueError(
            f"primary_table {primary_table.shape} vs alt_table "
            f"{alt_table.shape} shape mismatch"
        )
    device = next(model.parameters()).device
    primary_table = primary_table.to(device)
    alt_table = alt_table.to(device)
    primary_table_normalized = F.normalize(primary_table, dim=-1, eps=1e-8)
    alt_table_normalized = F.normalize(alt_table, dim=-1, eps=1e-8)

    # Hook to capture the input hidden state at each injection layer.
    probes = {layer: InjectionPreHook() for layer in layers}
    handles = []
    for layer_idx in layers:
        h = model.engram_injections[str(layer_idx)].register_forward_pre_hook(
            probes[layer_idx],
        )
        handles.append(h)

    per_layer: dict[int, list[dict[str, float]]] = {i: [] for i in layers}

    try:
        for batch in val_batches:
            for probe in probes.values():
                probe.reset()

            input_ids = batch.to(device)
            _ = model(input_ids)

            for layer_idx in layers:
                hidden_state = probes[layer_idx].captured
                if hidden_state is None:
                    raise RuntimeError(
                        f"Pre-hook at layer {layer_idx} did not fire — check "
                        f"engram_injections are attached and in the forward path."
                    )
                wrapper = model.engram_injections[str(layer_idx)]
                xattn = wrapper.engram_xattn

                # Primary-table injection (using the in-place buffers as
                # the trained model would see them).
                primary_inj = _injection_with_table(
                    xattn, hidden_state, primary_table, primary_table_normalized,
                )
                alt_inj = _injection_with_table(
                    xattn, hidden_state, alt_table, alt_table_normalized,
                )

                cos = F.cosine_similarity(primary_inj, alt_inj, dim=-1)
                signed = cos.mean().item()
                abs_mean = cos.abs().mean().item()

                # Random-baseline within-run pair: shuffle the primary
                # injection along the (B*L) flattened axis, then compute
                # |cos| against the unshuffled — this is what V-output
                # alignment would look like under "random in high-dim".
                flat = primary_inj.reshape(-1, primary_inj.shape[-1])
                perm = torch.randperm(flat.shape[0], device=flat.device)
                random_pair_cos = F.cosine_similarity(
                    flat, flat[perm], dim=-1,
                ).abs().mean().item()

                per_layer[layer_idx].append({
                    "cross_table_cos_signed": signed,
                    "cross_table_cos_abs": abs_mean,
                    "within_run_random_pair_cos_abs": random_pair_cos,
                })
    finally:
        for h in handles:
            h.remove()

    # Average per-layer scalars across batches.
    out: dict[int, dict[str, float]] = {}
    for layer_idx in layers:
        rows = per_layer[layer_idx]
        n = max(len(rows), 1)
        out[layer_idx] = {
            "cross_table_cos_signed": sum(r["cross_table_cos_signed"] for r in rows) / n,
            "cross_table_cos_abs": sum(r["cross_table_cos_abs"] for r in rows) / n,
            "within_run_random_pair_cos_abs": sum(
                r["within_run_random_pair_cos_abs"] for r in rows
            ) / n,
        }
    return out


def _sample_matched_gaussian_alt(
    primary_table: torch.Tensor, seed: int,
) -> torch.Tensor:
    """Sample a random-gaussian alt table at matched per-dim mean and std.

    Used by the (X) cross-table probe as a genuine content-swap
    counterfactual (row-permuting the primary would be a no-op on top-k
    cosine retrieval). Matched statistics keep retrieval similarities
    in a comparable range so the probe measures V's sensitivity rather
    than its response to OOD magnitudes.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    table_mean = primary_table.mean(dim=0, keepdim=True)
    table_std = primary_table.std(dim=0, keepdim=True).clamp(min=1e-8)
    return (
        torch.randn(primary_table.shape, generator=gen, dtype=primary_table.dtype)
        * table_std
        + table_mean
    )


def _injection_with_table(
    xattn: "EngramCrossAttention",
    hidden_state: torch.Tensor,
    table: torch.Tensor,
    table_normalized: torch.Tensor,
) -> torch.Tensor:
    """Compute injection output using a caller-supplied table (no in-place buffer mutation).

    Mirrors `EngramCrossAttention.forward` but uses the supplied table rather
    than `xattn.table` / `xattn.table_normalized`. We do NOT swap the buffers
    in-place because that would race with concurrent forwards if any exist
    and would also leave the model in a wrong state if an exception fires.
    """
    q = xattn.retrieval_query_proj(hidden_state)
    q_norm = F.normalize(q, dim=-1, eps=1e-8)
    sims = torch.einsum("ble,te->blt", q_norm, table_normalized)
    topk_sims, topk_idx = sims.topk(xattn.k_retrieved, dim=-1)
    retrieved = table[topk_idx]
    # `_attention_block` was added in Task 4 and is guaranteed present in
    # this bundled PR (no cross-PR version skew concern).
    return xattn._attention_block(hidden_state, retrieved, topk_sims)


def _mean(per_batch_stats: list[dict[str, Any]], key: str) -> float:
    return sum(s[key] for s in per_batch_stats) / len(per_batch_stats)


def print_per_run_diagnostics(
    per_layer_real: dict[int, list[dict[str, Any]]],
    per_layer_shuf: dict[int, list[dict[str, Any]]],
    layers: list[int],
    k: int,
) -> None:
    log_k = math.log(k)
    print("=" * 80)
    print("PER-LAYER PER-RUN DIAGNOSTICS")
    print("=" * 80)
    for layer_idx in layers:
        real = per_layer_real[layer_idx]
        shuf = per_layer_shuf[layer_idx]
        print(f"\nLayer {layer_idx}   (k_retrieved = {k}, log(k) = {log_k:.3f})")
        print("                                            real           shuffled")
        print(f"  (D) unique rows selected              {_mean(real, 'unique_rows'):10.0f}     {_mean(shuf, 'unique_rows'):10.0f}")
        print(f"      unique / (B*L*k) positions         {100*_mean(real, 'unique_rows_frac'):8.3f}%    {100*_mean(shuf, 'unique_rows_frac'):8.3f}%")
        print(f"  (P) top-1 sim                           {_mean(real, 'top1_sim'):+8.4f}     {_mean(shuf, 'top1_sim'):+8.4f}")
        print(f"      top-k sim                           {_mean(real, 'topk_sim'):+8.4f}     {_mean(shuf, 'topk_sim'):+8.4f}")
        print(f"      gap (top-1 − top-k)                 {_mean(real, 'sim_gap'):+8.4f}     {_mean(shuf, 'sim_gap'):+8.4f}")
        print(f"  (E) attn entropy / log(k)               {_mean(real, 'entropy_frac_of_max'):8.4f}     {_mean(shuf, 'entropy_frac_of_max'):8.4f}")
        print(f"  (M) ||inj|| / ||hidden||                {_mean(real, 'injection_vs_hidden_norm'):8.4f}     {_mean(shuf, 'injection_vs_hidden_norm'):8.4f}")
        print(f"      gate = tanh(alpha)                  {_mean(real, 'gate'):+8.4f}     {_mean(shuf, 'gate'):+8.4f}")
        print(f"  (W) within-run concentration R          {_mean(real, 'within_run_concentration'):8.4f}     {_mean(shuf, 'within_run_concentration'):8.4f}")
        print(f"      within-run axis RMS cos             {_mean(real, 'within_run_axis_rms'):8.4f}     {_mean(shuf, 'within_run_axis_rms'):8.4f}")
        print(f"  (A) cos(inj, hidden)  signed            {_mean(real, 'injection_vs_hidden_signed'):+8.4f}     {_mean(shuf, 'injection_vs_hidden_signed'):+8.4f}")
        print(f"      cos(inj, hidden)  |abs|             {_mean(real, 'injection_vs_hidden_abs'):8.4f}     {_mean(shuf, 'injection_vs_hidden_abs'):8.4f}")
        print(f"  (Q-overlap) random-pair Jaccard mean    {_mean(real, 'q_overlap_random_pair_jaccard_mean'):8.4f}     {_mean(shuf, 'q_overlap_random_pair_jaccard_mean'):8.4f}")
        print(f"              random-pair Jaccard p50     {_mean(real, 'q_overlap_random_pair_jaccard_p50'):8.4f}     {_mean(shuf, 'q_overlap_random_pair_jaccard_p50'):8.4f}")
        print(f"  (Q-occupancy) CV (used rows)            {_mean(real, 'q_occupancy_cv_used'):8.4f}     {_mean(shuf, 'q_occupancy_cv_used'):8.4f}")
        print(f"  (V-rank) |S|                          {_mean(real, 'subset_size_S'):10.0f}     {_mean(shuf, 'subset_size_S'):10.0f}")
        print(f"           rank(E[S])        hard          {_mean(real, 'e_rank_hard'):8.2f}     {_mean(shuf, 'e_rank_hard'):8.2f}")
        print(f"           rank(V(E[S]))     hard          {_mean(real, 'v_rank_hard'):8.2f}     {_mean(shuf, 'v_rank_hard'):8.2f}")
        print(f"           rank(E[S])        effective     {_mean(real, 'e_rank_effective'):8.2f}     {_mean(shuf, 'e_rank_effective'):8.2f}")
        print(f"           rank(V(E[S]))     effective     {_mean(real, 'v_rank_effective'):8.2f}     {_mean(shuf, 'v_rank_effective'):8.2f}")
        print(f"           v / e rank ratio                {_mean(real, 'v_rank_ratio'):8.4f}     {_mean(shuf, 'v_rank_ratio'):8.4f}")


def print_cross_run(
    cross_run_per_layer: dict[int, list[tuple[float, float]]],
    layers: list[int],
) -> None:
    """Print cross-run cosine from pre-aggregated scalar stats.

    `cross_run_per_layer[layer_idx]` is a list of `(signed_cos_mean,
    abs_cos_mean)` tuples, one per batch. Aggregated at print time rather
    than at compute time so the per-batch numbers remain inspectable if we
    want to add them to the output later.
    """
    print("\n" + "=" * 80)
    print("CROSS-RUN COMPARISON  (real vs shuffled on matched tokens)")
    print("=" * 80)
    print("Signed cos ≈ 0 AND |cos| ≈ 0 → the two runs picked arbitrary")
    print("orthogonal directions (fingerprint of content-free utilization).\n")
    for layer_idx in layers:
        batches = cross_run_per_layer[layer_idx]
        n = len(batches)
        mean_signed = sum(s for s, _ in batches) / n
        mean_abs = sum(a for _, a in batches) / n
        print(f"  Layer {layer_idx}:  signed cos = {mean_signed:+.4f}    |cos| = {mean_abs:.4f}")


def print_verdict_criteria(k: int) -> None:
    print("\n" + "=" * 80)
    print("VERDICT CRITERIA")
    print("=" * 80)
    print(f"""
(R) RETRIEVAL-BROKEN   if, on BOTH runs:
    (D) unique / (B*L*k) ≲ 5%                 # few rows ever selected
    (E) attn entropy / log(k) ≳ 0.9           # softmax near-uniform over top-{k}
    (P) sim-gap ≲ 0.05                        # top-1 barely beats top-{k}
    → Q projection converged to content-insensitive direction.

(I) INJECTION-BROKEN   if retrieval looks reasonable but:
    (M) ||inj|| / ||hidden|| ≲ 0.01            # tiny residual contribution
    AND cross-run |cos| ≲ 0.1                  # orthogonal directions across runs
    → injection path doesn't carry content; any learned direction works.

(I*) INJECTION-BROKEN, MAGNITUDE SUBSTANTIAL:
    (M) material (≥ 0.05) BUT cross-run |cos| ≲ 0.1
    → model learns an arbitrary per-run auxiliary residual. This is the
      closest mechanistic fingerprint of the shuffle-kill observation
      (same magnitude, arbitrary polarity/direction).

(B) BOTH-BROKEN:  low (D) AND small (M).

(H) HEALTHY (hypothetical):  peaked retrieval + diverse rows + material
    magnitude + cross-run cos ≳ 0.3. The 2026-04-17 forensic landed here,
    which alone did NOT invalidate the shuffle-kill claim (content-invariant
    training metrics stand) — it only ruled out (I*)'s "orthogonal per-run
    directions" prediction.

--- (W) and (A) joint readings (added 2026-04-17) ---

The cross-run cos-only rubric above cannot distinguish "fixed per-run
amplifier with aligned directions" from "token-sensitive V with aligned
response distributions." (W) and (A) refine the verdict:

Pattern matching uses two (W) statistics:
  R       — signed mean-direction resultant (1 = fixed direction & polarity;
            0 = uniform OR ±-symmetric).
  RMS_cos — sign-invariant axis concentration via the Gram-squared trace
            (1 = fixed axis regardless of polarity; ~1/√D for uniform).

(I**) FIXED RESIDUAL-AXIS AMPLIFIER  — most specific amplifier reading:
    RMS_cos ≳ 0.9 on both runs
    AND (A) |cos(inj, hidden)| ≳ 0.5 on both runs
    → V emits a single axis per run, living on the residual-stream axis.
      The injection is literally amplifying (or nulling) the hidden state.
      Scale-up unlikely to help; redesign MUST break V's collapse to a
      fixed axis. R vs RMS_cos split tells us about polarity stability:
      R ≈ RMS_cos → same polarity (L5-like); R ≪ RMS_cos → ±-symmetric
      axis (L2-like, where the axis is learned but the sign is arbitrary).

(I***) FIXED NON-RESIDUAL AMPLIFIER:
    RMS_cos ≳ 0.9 on both runs AND (A) |cos(inj, hidden)| ≲ 0.2
    → V emits a fixed axis per run but orthogonal to the residual.
      Still content-blind; redesign target is same (force V token-
      sensitivity) but the "residual amplifier" framing doesn't fit.

(D*) DISTRIBUTIONAL ALIGNMENT  — harder-to-fix story:
    RMS_cos ≲ 0.5 on both runs BUT cross-run cos ≳ 0.3
    → Per-token injection directions span diverse axes within a run, but
      the DISTRIBUTIONS of those directions align across runs. V is token-
      sensitive; the content-invariance is a trained alignment of response
      patterns, not a fixed projection. Redesign implications differ —
      hard-gating on retrieval relevance may not be sufficient.

Interpretation rubric for 2026-04-16 η-B + 2026-04-17 ζ-ctrl data:

  Observed at 40M η-B:
    - Retrieval healthy: (D) 5-6%, (E) ~0.72, (P) sim gap +0.07 to +0.10.
    - (M) material: 0.04-0.07 × hidden norm.
    - Cross-run signed cos: +0.336 (L2), +0.630 (L5).
    → Rules out (R), (I), (I*). Rubric pointed to (H), but (H) alone
      doesn't explain content-invariance of training metrics.
    → NEXT: (W) and (A) measurement to select between (I**) / (I***) / (D*).

--- (X) cross-table content-sensitivity reading (added 2026-04-17) ---

(X) is the within-run content-swap probe: forward the same trained model
against its training-primary table AND against a random-gaussian alt
table of matched per-dim statistics. Both (P) retrievals are genuine
top-k — the alt table contains different vectors, so the retrieved
content differs (not just relabeled).

(V-CONTENT-BLIND)  (X) |cos| ≳ 0.8 on both runs
  → V emits similar injection outputs regardless of retrieved content.
    V is functionally a projection of (hidden, attention-weight summary)
    with the retrieved vectors contributing little. Redesign must force
    V-side content dependence (e.g., V-contrastive aux, KV tying, no-
    bias V projection).

(V-CONTENT-SENSITIVE)  (X) |cos| ≲ 0.2 on both runs
  → V responds to retrieved content. If the model is ALSO content-
    invariant at training metrics (shuffle-kill pattern), the pathology
    localizes upstream of V: Q queries a narrow row neighborhood where V
    orthogonalizes without the breadth needed for LM-task content
    routing. Redesign target is Q-side (load-balancing / retrieval-
    diversity aux).

(X) |cos| between 0.2 and 0.8 is ambiguous — compare to the within-run
random-pair |cos| floor (printed alongside each (X) reading): if (X) is
near the floor, V is dispersing tokens across axes but not specifically
content-routing; if (X) is materially above the floor, there is partial
content dependence.

--- (Q-overlap), (Q-occupancy), (V-rank) readings (added 2026-04-17) ---

The (D) unique-rows probe answers "how many rows does Q use in aggregate"
but not "is Q content-sensitive at a per-token granularity." η-B's 49%
retrieval + content-invariance paradox suggests a large |S| can coexist
with per-token retrieval fixity. The (Q-overlap) and (Q-occupancy) probes
split (D) into two orthogonal questions, and (V-rank) tests whether V
further compresses the retrieved span.

(Q-overlap) random-pair Jaccard mean J̄:
  J̄ ≳ 0.8  → per-token retrieval is near-fixed; marginal breadth is a
             projection artifact. The MoE f·P aux alone would NOT fix
             this; need a per-token Q-entropy or Q-diversity term.
  J̄ ≈ 0.5  → moderate per-token variation. Shared neighborhood but
             not fully fixed.
  J̄ ≲ 0.2  → per-token retrieval is content-dispersive. The (D) = 5%
             η-pre-ι reading combined with low J̄ would mean each token
             has its own neighborhood; the pathology is elsewhere.

(Q-occupancy) CV (coefficient of variation among used rows):
  CV ≲ 0.3 → row usage is nearly uniform among the |S| selected rows.
             This is what the MoE f·P aux targets — η-B post-ι₁ should
             move toward this.
  CV ≳ 1.0 → a few rows dominate the used set. High CV + high |S| =
             "many rows are used, but only a handful get most traffic."
  CV ≳ 2.0 → severe concentration (effectively retrieval-collapsed even
             if |S| looks large).

(V-rank) rank(V(E[S])) vs rank(E[S]):
  v/e ≈ 1  → V preserves the retrieved subset's rank. V's output
             dimensionality matches its input; no compression.
  v/e ≲ 0.8 → V loses rank on the retrieved subset. If V is otherwise
             content-sensitive ((X) |cos| low), this signals a V-side
             bottleneck orthogonal to Q — even broadening Q might not
             unlock full content-routing.
  v/e ≲ 0.5 → severe V compression. Consider V-side redesign
             (e.g., V-contrast tightening, wider hidden_dim).

Joint readings define the next experimental direction:

  High J̄ (≳ 0.8), low CV  → H3a confirmed; queue ι₃ (per-token
                             Q-entropy aux, orthogonal to marginal f·P).
  Low J̄ (≲ 0.2), high CV  → Per-token retrieval is dispersed but
                             marginal usage is peaked; ι₁'s f·P aux
                             should help directly.
  v/e ≲ 0.8 on both runs   → V-side bottleneck co-exists with Q issues;
                             consider V-rank-preservation aux or V
                             widening.
""")


def run_forensic(args: argparse.Namespace) -> None:
    device = args.device
    print(f"Loading real-oracle model from {args.real_ckpt} ...")
    real_model, real_config, layers_real = load_capgap_model(
        args.real_ckpt, args.real_table, args.k_retrieved,
        args.retrieval_bias_weight, args.retrieval_temperature, device,
    )
    print(f"Loading shuffled-oracle model from {args.shuffled_ckpt} ...")
    shuf_model, shuf_config, layers_shuf = load_capgap_model(
        args.shuffled_ckpt, args.shuffled_table, args.k_retrieved,
        args.retrieval_bias_weight, args.retrieval_temperature, device,
    )

    if layers_real != layers_shuf:
        raise ValueError(
            f"Injection-layer mismatch between checkpoints: "
            f"real={layers_real} shuffled={layers_shuf}. "
            f"Forensic requires matched architectures.",
        )
    if real_config.vocab_size != shuf_config.vocab_size:
        raise ValueError(
            f"Vocab size mismatch: real={real_config.vocab_size} "
            f"shuffled={shuf_config.vocab_size}. Forensic needs matched models.",
        )
    # hidden_dim is the shape contract for the injection tensors fed into
    # F.cosine_similarity below. Without this check, a mismatch would load
    # both checkpoints cleanly and only crash inside the cosine op on the
    # first batch. Fail fast alongside the other config-compat checks.
    if real_config.hidden_dim != shuf_config.hidden_dim:
        raise ValueError(
            f"Hidden dim mismatch: real={real_config.hidden_dim} "
            f"shuffled={shuf_config.hidden_dim}. Cross-run cosine requires "
            f"matched injection output geometry.",
        )
    layers = layers_real

    real_probes: dict[int, InjectionPreHook] = {}
    shuf_probes: dict[int, InjectionPreHook] = {}
    for layer_idx in layers:
        real_probes[layer_idx] = InjectionPreHook()
        shuf_probes[layer_idx] = InjectionPreHook()
        real_model.engram_injections[str(layer_idx)].register_forward_pre_hook(
            real_probes[layer_idx],
        )
        shuf_model.engram_injections[str(layer_idx)].register_forward_pre_hook(
            shuf_probes[layer_idx],
        )

    val_loader = make_hf_dataloader(
        args.val_data, args.seq_len, args.batch_size, args.seed,
    )

    print(
        f"\nRunning forensic on {args.num_batches} val batches "
        f"(batch_size={args.batch_size}, seq_len={args.seq_len})\n",
    )

    # Accumulated scalar stats only — no [B, L, hidden_dim] tensors retained
    # across batches. The large injection tensors are consumed inline to
    # compute cross-run cosine, then dropped.
    per_layer_real: dict[int, list[dict[str, float | int]]] = {i: [] for i in layers}
    per_layer_shuf: dict[int, list[dict[str, float | int]]] = {i: [] for i in layers}
    cross_run_per_layer: dict[int, list[tuple[float, float]]] = {i: [] for i in layers}

    for _batch_idx in range(args.num_batches):
        # Clear prior captures so a hook that fails to fire this batch surfaces
        # as a None-check failure instead of silently reusing the prior batch's
        # hidden state.
        for probe in real_probes.values():
            probe.reset()
        for probe in shuf_probes.values():
            probe.reset()

        batch = next(val_loader).to(device)
        input_ids = batch[:, :-1]

        with torch.no_grad():
            _ = real_model(input_ids)
            _ = shuf_model(input_ids)

        for layer_idx in layers:
            real_h = real_probes[layer_idx].captured
            shuf_h = shuf_probes[layer_idx].captured
            if real_h is None or shuf_h is None:
                raise RuntimeError(
                    f"Pre-hook at layer {layer_idx} did not fire — check that "
                    f"engram_injections are attached and in the forward path.",
                )
            # engram_inject_mult is a runtime-only HarmonyModel attribute
            # (default 1.0, set to 0 under --zero-injection-eval). Pass
            # whatever each model instance has so the reconstructed residual
            # matches what that model's forward path actually produces.
            real_stats, real_inj = analyze_injection(
                real_model.engram_injections[str(layer_idx)], real_h,
                inject_mult=real_model.engram_inject_mult,
            )
            shuf_stats, shuf_inj = analyze_injection(
                shuf_model.engram_injections[str(layer_idx)], shuf_h,
                inject_mult=shuf_model.engram_inject_mult,
            )
            # Compute cross-run cosine inline so the [B, L, hidden_dim]
            # injection tensors can be dropped before the next batch.
            cos = F.cosine_similarity(real_inj, shuf_inj, dim=-1)
            signed = cos.mean().item()
            abs_mean = cos.abs().mean().item()
            cross_run_per_layer[layer_idx].append((signed, abs_mean))
            del real_inj, shuf_inj, cos

            per_layer_real[layer_idx].append(real_stats)
            per_layer_shuf[layer_idx].append(shuf_stats)

    print_per_run_diagnostics(per_layer_real, per_layer_shuf, layers, args.k_retrieved)
    print_cross_run(cross_run_per_layer, layers)

    print("\n--- (X) cross-table within-run probe ---")
    print(f"Alt-table seed: {args.alt_table_seed}")

    # Each model is probed against ITS OWN training table as primary, with
    # an alt table sampled from a random gaussian at matched per-dim mean
    # and std. Probing shuf_model against args.real_table would test OOD
    # behavior (V seeing a table it never trained on) rather than V's
    # content-sensitivity on its actual training distribution.
    primary_table_real = EngramCrossAttention.load_corpus_table(str(args.real_table))
    primary_table_shuf = EngramCrossAttention.load_corpus_table(str(args.shuffled_table))
    alt_table_real = _sample_matched_gaussian_alt(primary_table_real, args.alt_table_seed)
    alt_table_shuf = _sample_matched_gaussian_alt(primary_table_shuf, args.alt_table_seed)

    # Re-collect val batches for the cross-table probe (val_loader is
    # one-shot in some implementations; safest to reuse the same dataloader
    # constructor with the same seed).
    val_loader_xtable = make_hf_dataloader(
        args.val_data, args.seq_len, args.batch_size, args.seed,
    )
    val_batches: list[torch.Tensor] = []
    for _ in range(args.num_batches):
        val_batches.append(next(val_loader_xtable)[:, :-1])

    cross_table_real = analyze_cross_table(
        real_model, primary_table_real, alt_table_real, val_batches, layers,
    )
    print("\n[real-oracle model]")
    for layer_idx in layers:
        stats = cross_table_real[layer_idx]
        print(f"  L{layer_idx}:")
        print(f"    signed (primary vs alt)           {stats['cross_table_cos_signed']:+.4f}")
        print(f"    |abs|                             {stats['cross_table_cos_abs']:.4f}")
        print(f"    random-pair |cos| (orth. floor)   {stats['within_run_random_pair_cos_abs']:.4f}")

    cross_table_shuf = analyze_cross_table(
        shuf_model, primary_table_shuf, alt_table_shuf, val_batches, layers,
    )
    print("\n[shuffled-oracle model]")
    for layer_idx in layers:
        stats = cross_table_shuf[layer_idx]
        print(f"  L{layer_idx}:")
        print(f"    signed (primary vs alt)           {stats['cross_table_cos_signed']:+.4f}")
        print(f"    |abs|                             {stats['cross_table_cos_abs']:.4f}")
        print(f"    random-pair |cos| (orth. floor)   {stats['within_run_random_pair_cos_abs']:.4f}")

    print_verdict_criteria(args.k_retrieved)


def main() -> None:
    run_forensic(parse_args())


if __name__ == "__main__":
    main()
