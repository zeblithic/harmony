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
on WHICH rows get retrieved given identical Q projections. Any difference
in behavior between the two runs localizes entirely to training-trajectory
divergence of the learned projections. The (C) cross-run comparison is the
definitive probe for whether the two trajectories found the same useful
direction or two arbitrary ones.

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

import torch
import torch.nn.functional as F

_REPO_TRAINING = Path(__file__).resolve().parent.parent
if str(_REPO_TRAINING) not in sys.path:
    sys.path.insert(0, str(_REPO_TRAINING))

from ct87.engram import EngramCrossAttention, GatedEngramInjection  # noqa: E402
from ct87.model import HarmonyModel, HarmonyModelConfig  # noqa: E402
from ct87.train import make_hf_dataloader  # noqa: E402


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
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for val-batch selection. Identical across both models so they "
                        "see the same tokens — required for (C) cross-run comparison.")
    return p.parse_args()


def load_capgap_model(
    ckpt_path: Path,
    table_path: Path,
    k_retrieved: int,
    device: str,
) -> tuple[HarmonyModel, HarmonyModelConfig, list[int]]:
    """Load an η-B capgap checkpoint and attach its engram injections.

    The table buffers (`table`, `table_normalized`) are non-persistent in
    EngramCrossAttention — they're initialized from the passed `table_path`
    and recomputed from it after `load_state_dict` via the module's
    `_load_from_state_dict` hook. Passing the correct table path for each
    run is what makes the forensic meaningful.
    """
    payload: dict[str, Any] = torch.load(
        str(ckpt_path), map_location="cpu", weights_only=False,
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


@torch.no_grad()
def analyze_injection(
    wrapper: GatedEngramInjection,
    hidden_state: torch.Tensor,
) -> tuple[dict[str, float | int], torch.Tensor]:
    """Re-run the injection forward manually so we can capture intermediates.

    The public `EngramCrossAttention.forward` returns only the final residual,
    swallowing the retrieval distribution and attention weights. We reproduce
    the forward inline so the retrieval and attention tensors are available
    for diagnostics. Logic is byte-for-byte the same as
    `EngramCrossAttention.forward` as of commit b2c3a96.

    Returns `(scalar_stats, injection)`:
      - `scalar_stats` has no tensors so callers can accumulate it cheaply
        across batches.
      - `injection` is the gate × xattn_out tensor ([B, L, hidden_dim]) for
        cross-run comparison; callers should consume it immediately and
        drop the reference so tensors don't accumulate across batches.
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

    # Mirror GatedEngramInjection.forward: keep the gate as a tensor, cast
    # to xattn_out's dtype before multiplying. Avoids a gratuitous GPU→CPU
    # sync and matches the production dtype contract.
    gate = torch.tanh(wrapper.alpha).to(dtype=xattn_out.dtype)
    injection = gate * xattn_out

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
    }
    return scalar_stats, injection


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
        print(f"                                            real           shuffled")
        print(f"  (D) unique rows selected              {_mean(real, 'unique_rows'):10.0f}     {_mean(shuf, 'unique_rows'):10.0f}")
        print(f"      unique / (B*L*k) positions         {100*_mean(real, 'unique_rows_frac'):8.3f}%    {100*_mean(shuf, 'unique_rows_frac'):8.3f}%")
        print(f"  (P) top-1 sim                           {_mean(real, 'top1_sim'):+8.4f}     {_mean(shuf, 'top1_sim'):+8.4f}")
        print(f"      top-k sim                           {_mean(real, 'topk_sim'):+8.4f}     {_mean(shuf, 'topk_sim'):+8.4f}")
        print(f"      gap (top-1 − top-k)                 {_mean(real, 'sim_gap'):+8.4f}     {_mean(shuf, 'sim_gap'):+8.4f}")
        print(f"  (E) attn entropy / log(k)               {_mean(real, 'entropy_frac_of_max'):8.4f}     {_mean(shuf, 'entropy_frac_of_max'):8.4f}")
        print(f"  (M) ||inj|| / ||hidden||                {_mean(real, 'injection_vs_hidden_norm'):8.4f}     {_mean(shuf, 'injection_vs_hidden_norm'):8.4f}")
        print(f"      gate = tanh(alpha)                  {_mean(real, 'gate'):+8.4f}     {_mean(shuf, 'gate'):+8.4f}")


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
    magnitude + cross-run cos ≳ 0.3 → both runs learned the same direction.
    This would invalidate the shuffle-kill interpretation; it is NOT
    expected given the training metrics, but the forensic can confirm.

Interpretation rubric for common expected outcomes:

  Expected (most likely given shuffle-kill data):
    - Retrieval looks fine: peaked, diverse rows, low entropy.
    - (M) is ~0.02 × hidden norm: small but not negligible.
    - Cross-run |cos| near 0: arbitrary direction per run.
    → Verdict (I*): the 40M backbone+frozen config extracts no content
      from the injection path regardless of what the retrieval feeds in.
      Next move: scale up (90M) or redesign integration mechanism.

  Alternate (weaker):
    - Retrieval is somewhat peaked but rows clump onto a small subset of
      the table on both runs.
    → Verdict (R/I): Q projection is under-trained at this scale.
      Next move: longer capgap training, or scale up to see if Q learns.
""")


def run_forensic(args: argparse.Namespace) -> None:
    device = args.device
    print(f"Loading real-oracle model from {args.real_ckpt} ...")
    real_model, real_config, layers_real = load_capgap_model(
        args.real_ckpt, args.real_table, args.k_retrieved, device,
    )
    print(f"Loading shuffled-oracle model from {args.shuffled_ckpt} ...")
    shuf_model, shuf_config, layers_shuf = load_capgap_model(
        args.shuffled_ckpt, args.shuffled_table, args.k_retrieved, device,
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
            real_stats, real_inj = analyze_injection(
                real_model.engram_injections[str(layer_idx)], real_h,
            )
            shuf_stats, shuf_inj = analyze_injection(
                shuf_model.engram_injections[str(layer_idx)], shuf_h,
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
    print_verdict_criteria(args.k_retrieved)


def main() -> None:
    run_forensic(parse_args())


if __name__ == "__main__":
    main()
