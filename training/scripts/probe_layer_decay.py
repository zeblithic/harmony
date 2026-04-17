"""ZEB-133: layer-wise content-decay probe.

Given an ι₂ checkpoint (or any multi-layer gated-engram checkpoint),
loads two models that share weights but use different oracle tables
(real vs shuffled). Runs identical tokens through both and measures:

  - Per-site ‖h_real − h_shuf‖ / ‖h_real‖ at five points in the network:
      L_pre_inj    — input to the last engram injection site
      L_post_inj   — output of the last injection (= pre + gate * inj_out)
      L{N-2}_out   — transformer layer output one step after injection
      L{N-1}_out   — last transformer layer output (pre final_norm)
      LM_head_in   — final_norm(last_hidden), the tensor fed to lm_head

  - Direction cos(h_real − h_shuf at site, h_real − h_shuf at L_post_inj)
      A value near 1.0 means the delta at this site is the same direction
      as the one injected; near 0 means it has been rotated/filtered away.

  - LM-head row alignment AT LM_head_in: for the delta vector at each
    position, report max |cos| with any LM-head row and mean-top-10 |cos|.
    Low numbers mean the surviving delta does not point along any
    vocabulary direction, so the LM-head cannot translate it into token
    probability differences regardless of its magnitude.

The three numbers together discriminate three failure modes:

  H-decay      ||delta|| shrinks across L6/L7 and/or final_norm.
               Content is injected but downstream layers wash it out.
  H-blindness  ||delta|| survives intact but LM-head row alignment
               stays low — the content survives but the LM can't read
               it.
  H-tiny       ||delta|| at L_post_inj is already microscopic; even
               perfect preservation wouldn't matter.

Usage (on KRILE):

    cd training
    python3 scripts/probe_layer_decay.py \\
        --ckpt checkpoints/iota_2_vcontrast_qdiv/checkpoint.pt \\
        --real-table artifacts/oracle_mistral7b_10k.safetensors \\
        --shuffled-table artifacts/oracle_mistral7b_10k_shuffled_seed0.safetensors \\
        --val-data /data/fineweb-edu-poc/val \\
        --num-samples 128 --batch-size 4 --seq-len 2048

No training, no gradients. ~30 s on a 4090.
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

from ct87.engram import EngramCrossAttention  # noqa: E402
from ct87.model import HarmonyModel  # noqa: E402
from ct87.train import make_hf_dataloader  # noqa: E402
from scripts.forensic_eta_b_capgap import (  # noqa: E402
    _sample_matched_gaussian_alt,
    load_capgap_model,
)

# Generic site keys used by the probe. The probe reports on the LAST
# injection site (not the first) plus every transformer layer strictly
# after it, plus the LM-head input.
LAYER_DECAY_SITES: tuple[str, ...] = (
    "L_pre_inj",
    "L_post_inj",
    "L_post_layer_1",
    "L_post_layer_2",
    "LM_head_in",
)

# Verdicts the probe can emit. Emitted labels are exactly these strings
# so the post-batch synthesizer can grep for them without regex tweaks.
VERDICT_LABELS: tuple[str, ...] = (
    "H-decay",
    "H-blindness",
    "H-tiny",
    "ambiguous",
)


# ---------------------------------------------------------------------------
# Small numerical helpers — pulled out so they are independently testable.
# ---------------------------------------------------------------------------


def _finite_positive_float(value: str) -> float:
    f = float(value)
    if not math.isfinite(f) or f <= 0:
        raise argparse.ArgumentTypeError(f"must be finite and > 0, got {value!r}")
    return f


def _positive_int(value: str) -> int:
    i = int(value)
    if i <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {value!r}")
    return i


def fractional_norm(
    diff: torch.Tensor, ref: torch.Tensor, eps: float = 1e-12,
) -> float:
    """Mean over positions of ‖diff‖ / ‖ref‖.

    Both tensors have shape [..., D]; the reduction is over everything
    except the last dim. The output is a plain Python float.
    """
    if diff.shape != ref.shape:
        raise ValueError(f"diff/ref shape mismatch: {diff.shape} vs {ref.shape}")
    dn = diff.flatten(end_dim=-2).norm(dim=-1)
    rn = ref.flatten(end_dim=-2).norm(dim=-1).clamp(min=eps)
    return (dn / rn).mean().item()


def direction_cosines(
    deltas: dict[str, torch.Tensor], reference_key: str,
) -> dict[str, float]:
    """Mean per-position cosine of each delta tensor against `reference_key`.

    Zero-norm reference positions are masked out. Returns NaN for sites
    where every position has a zero reference (which should not happen
    post-injection but is defensible for pre-injection).
    """
    ref = deltas[reference_key].flatten(end_dim=-2)
    ref_n = ref.norm(dim=-1)
    # Mask zeros so we don't contaminate the mean with 0/0.
    valid = ref_n > 1e-12
    ref_unit = F.normalize(ref, dim=-1, eps=1e-12)
    out: dict[str, float] = {}
    for key, d in deltas.items():
        d_flat = d.flatten(end_dim=-2)
        d_unit = F.normalize(d_flat, dim=-1, eps=1e-12)
        cos = (d_unit * ref_unit).sum(dim=-1)
        if valid.any():
            out[key] = cos[valid].mean().item()
        else:
            out[key] = float("nan")
    return out


def lm_head_row_alignment(
    delta: torch.Tensor, lm_head_weight: torch.Tensor, top_k: int = 10,
) -> dict[str, float]:
    """Per-position cos(delta, lm_head_row) statistics.

    Args:
        delta:          [..., D] the real-shuf diff at LM-head input.
        lm_head_weight: [V, D] LM-head rows.
        top_k:          how many of the highest cosines to average per
                        position (reports that mean as "top10_mean_cos").

    Returns:
        dict with "max_cos" (mean over positions of per-position max)
        and "top10_mean_cos" (mean over positions of the per-position
        mean of the top_k largest cosines).
    """
    d_flat = F.normalize(delta.flatten(end_dim=-2), dim=-1, eps=1e-12)
    w_norm = F.normalize(lm_head_weight, dim=-1, eps=1e-12)
    # cos: [positions, V]
    cos = (d_flat @ w_norm.T).abs()
    k = min(top_k, cos.shape[-1])
    topk = cos.topk(k, dim=-1).values
    return {
        "max_cos": topk[:, 0].mean().item(),
        "top10_mean_cos": topk.mean(dim=-1).mean().item(),
    }


# ---------------------------------------------------------------------------
# Hook-based hidden-state capture.
# ---------------------------------------------------------------------------


class _DecayCapture:
    """Registers hooks needed to capture every site's hidden state.

    Sites captured:
      * pre_last_inj    — forward-pre-hook on engram_injections[str(last)]
      * post_last_inj   — post-hook on layers[last] (actually, the wrapper
                          injection call is inlined; we reconstruct post-inj
                          from pre + the wrapper's return, see NOTE below)
      * post_layer[i]   — post-hook on layers[i] for each i strictly after
                          the last injection layer
      * lm_head_in      — post-hook on final_norm

    NOTE: the model's forward pass reads
        injection_out = engram_injections[key](h)
        h = h + mult * injection_out
    There is no module-level hook point for `h + mult * injection_out`, so
    we compute post-injection inside the pre-hook on the *following*
    layer: its input IS the post-injection hidden state. Robust even if
    block_attnres mixing happens at the boundary — we capture BEFORE the
    mix at the next-layer boundary.

    The cleanest next-layer hook point is the first operation after the
    injection. That's either:
      - layers[next].forward input, if the next index is NOT a block
        boundary, OR
      - block_attnres.block_input call, if it IS a block boundary.

    To avoid special-casing block boundaries, we wrap the injection
    module's forward so the wrapper stashes `h_pre` (its input) and
    `inj_out` (its return). post_inj = h_pre + mult * inj_out is then
    computed outside the hook.
    """

    def __init__(self, model: HarmonyModel, last_inj_layer: int) -> None:
        self.model = model
        self.last_inj_layer = last_inj_layer
        self.inject_mult = float(model.engram_inject_mult)
        self.captured: dict[str, list[torch.Tensor]] = {}
        self._handles: list = []
        # Layers strictly after the last injection (these are where decay
        # happens). We name them L_post_layer_1, L_post_layer_2, ...
        num_layers = len(model.layers)
        self.post_layer_indices = list(range(last_inj_layer + 1, num_layers))

    def __enter__(self) -> "_DecayCapture":
        self._install()
        return self

    def __exit__(self, *exc_info) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _append(self, key: str, tensor: torch.Tensor) -> None:
        self.captured.setdefault(key, []).append(tensor.detach().float().cpu())

    def _install(self) -> None:
        model = self.model
        last = self.last_inj_layer

        # Wrap the injection module's forward so we capture its input
        # (pre_inj) and its output (inj_out) deterministically.
        inj_module = model.engram_injections[str(last)]
        original_forward = inj_module.forward
        cap = self

        def wrapped_forward(h: torch.Tensor, *args, **kwargs):
            cap._append("pre_inj", h)
            inj_out = original_forward(h, *args, **kwargs)
            cap._append("inj_out", inj_out)
            return inj_out

        inj_module.forward = wrapped_forward  # type: ignore[assignment]
        # Record the monkey-patch so __exit__ can restore it.
        self._handles.append(_RestoreAttribute(inj_module, "forward", original_forward))

        # Post-hooks on every transformer layer strictly after the last
        # injection. Hook output is [B, L, D] — store as-is.
        for offset, idx in enumerate(self.post_layer_indices, start=1):
            key = f"post_layer_{offset}"

            def hook(_mod, _inp, output, _k=key):
                cap._append(_k, output)

            self._handles.append(model.layers[idx].register_forward_hook(hook))

        # Post-hook on final_norm — its output is what lm_head receives.
        def final_norm_hook(_mod, _inp, output):
            cap._append("lm_head_in", output)

        self._handles.append(model.final_norm.register_forward_hook(final_norm_hook))


class _RestoreAttribute:
    """Helper that mimics a handle.remove() for monkey-patched attributes.

    nn.Module.register_forward_hook returns a RemovableHandle with a
    .remove() method. To use the same cleanup idiom for a monkey-patched
    forward, we wrap the restore in a tiny object exposing .remove().
    """

    def __init__(self, obj: Any, attr: str, original: Any) -> None:
        self._obj = obj
        self._attr = attr
        self._original = original

    def remove(self) -> None:
        setattr(self._obj, self._attr, self._original)


def _post_injection_from_capture(
    capture: _DecayCapture,
) -> list[torch.Tensor]:
    """Reconstruct post-injection hidden state for each captured batch.

    post_inj[b] = pre_inj[b] + inject_mult * inj_out[b]
    """
    pres = capture.captured.get("pre_inj", [])
    outs = capture.captured.get("inj_out", [])
    if len(pres) != len(outs):
        raise RuntimeError(
            f"Capture mismatch: pre_inj has {len(pres)} batches, inj_out "
            f"has {len(outs)}. Did the hook fire for every forward?"
        )
    return [p + capture.inject_mult * o for p, o in zip(pres, outs, strict=True)]


# ---------------------------------------------------------------------------
# The probe itself.
# ---------------------------------------------------------------------------


def measure_layer_decay(
    model_real: HarmonyModel,
    model_shuf: HarmonyModel,
    batch: torch.Tensor,
) -> dict[str, dict[str, float]]:
    """Forward `batch` through both models, compute per-site decay stats.

    Both models must have the same `engram_inject_layers` and the last
    injection layer must NOT be the final transformer layer (otherwise
    the probe has no "post-injection" transformer layers to measure
    decay in). The probe reports stats at the last-injection site plus
    every layer strictly after it, plus LM-head input.

    Returns:
        dict site_key → {"fraction": float, "direction_cos_vs_L5": float}
    """
    layers_real = list(model_real.config.engram_inject_layers)
    layers_shuf = list(model_shuf.config.engram_inject_layers)
    if layers_real != layers_shuf:
        raise ValueError(
            f"Models disagree on engram_inject_layers: "
            f"real={layers_real}, shuf={layers_shuf}"
        )
    last = layers_real[-1]
    num_layers = len(model_real.layers)
    if last >= num_layers - 1:
        raise ValueError(
            f"Last engram injection is at layer {last} but num_layers={num_layers}. "
            f"The probe needs at least one transformer layer strictly after "
            f"the final injection to measure decay."
        )

    device = next(model_real.parameters()).device
    batch = batch.to(device)

    with torch.no_grad():
        with _DecayCapture(model_real, last) as cap_real, \
             _DecayCapture(model_shuf, last) as cap_shuf:
            _ = model_real(batch)
            _ = model_shuf(batch)

    # Reconstruct post-injection for both.
    post_inj_real = _post_injection_from_capture(cap_real)
    post_inj_shuf = _post_injection_from_capture(cap_shuf)

    # Flatten each site to a single tensor across the single batch we just ran.
    def _cat(capture: dict[str, list[torch.Tensor]], key: str) -> torch.Tensor:
        if key not in capture or not capture[key]:
            raise KeyError(f"site {key} missing from capture")
        return torch.cat(capture[key], dim=0)

    pre_real = _cat(cap_real.captured, "pre_inj")
    pre_shuf = _cat(cap_shuf.captured, "pre_inj")
    post_real = torch.cat(post_inj_real, dim=0)
    post_shuf = torch.cat(post_inj_shuf, dim=0)

    # Always exactly 2 post-injection transformer layers for ι₂ (L6, L7),
    # but generalize so the probe works on any config.
    num_post = len(cap_real.post_layer_indices)
    post_layer_real = [_cat(cap_real.captured, f"post_layer_{i}") for i in range(1, num_post + 1)]
    post_layer_shuf = [_cat(cap_shuf.captured, f"post_layer_{i}") for i in range(1, num_post + 1)]

    lm_head_in_real = _cat(cap_real.captured, "lm_head_in")
    lm_head_in_shuf = _cat(cap_shuf.captured, "lm_head_in")

    # Deltas per site.
    deltas: dict[str, torch.Tensor] = {
        "L_pre_inj": pre_real - pre_shuf,
        "L_post_inj": post_real - post_shuf,
    }
    for i, (rr, ss) in enumerate(zip(post_layer_real, post_layer_shuf, strict=True), start=1):
        deltas[f"L_post_layer_{i}"] = rr - ss
    deltas["LM_head_in"] = lm_head_in_real - lm_head_in_shuf

    # Fractions per site.
    refs: dict[str, torch.Tensor] = {
        "L_pre_inj": pre_real,
        "L_post_inj": post_real,
        "LM_head_in": lm_head_in_real,
    }
    for i, rr in enumerate(post_layer_real, start=1):
        refs[f"L_post_layer_{i}"] = rr

    fractions = {k: fractional_norm(deltas[k], refs[k]) for k in deltas}
    cosines = direction_cosines(deltas, reference_key="L_post_inj")

    return {
        k: {"fraction": fractions[k], "direction_cos_vs_L5": cosines[k]}
        for k in deltas
    }


def verdict_from_stats(
    site_stats: dict[str, dict[str, float]],
    lm_stats: dict[str, float],
    post_decay_threshold: float = 0.25,
    lm_align_threshold: float = 0.05,
    tiny_threshold: float = 5e-3,
) -> str:
    """Turn site/LM stats into one of VERDICT_LABELS.

    Heuristics:
      * If L_post_inj fraction < tiny_threshold, call it H-tiny (the
        signal at the injection site is already too small for anything
        downstream to matter).
      * Otherwise compare LM_head_in fraction against L_post_inj:
          - if it has shrunk by more than post_decay_threshold ratio,
            label H-decay;
          - else if LM-head max_cos < lm_align_threshold, label
            H-blindness;
          - else "ambiguous" (signal survives AND aligns with vocab —
            unexpected; let the user decide).
    """
    post_inj = site_stats["L_post_inj"]["fraction"]
    lm_in = site_stats["LM_head_in"]["fraction"]

    if post_inj < tiny_threshold:
        return "H-tiny"
    # Ratio of surviving magnitude at LM-head input to the original injection.
    survive_ratio = lm_in / max(post_inj, 1e-12)
    if survive_ratio < post_decay_threshold:
        return "H-decay"
    if lm_stats.get("max_cos", 0.0) < lm_align_threshold:
        return "H-blindness"
    return "ambiguous"


# ---------------------------------------------------------------------------
# CLI entry point.
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ckpt", required=True, type=Path,
                   help="Checkpoint to analyse (usually ι₂).")
    p.add_argument("--real-table", required=True, type=Path,
                   help="Oracle table the checkpoint was trained on.")
    p.add_argument("--alt-table-seed", type=int, default=42,
                   help="Seed for the random-gaussian alt table used for the "
                        "real-vs-alt forensic diff. We cannot use the "
                        "row-permuted shuffled oracle as the diff target "
                        "because retrieve_topk is permutation-invariant on "
                        "CONTENT — both tables produce identical retrieved "
                        "vectors, just at different index labels. Instead we "
                        "sample a random-gaussian table at matched per-dim "
                        "first and second moments (same approach as the "
                        "fixed (X) probe in PR 251).")
    p.add_argument("--val-data", required=True, type=str)
    p.add_argument("--num-samples", type=_positive_int, default=128,
                   help="Total number of sequences to forensic over. Will be "
                        "assembled from num-samples/batch-size batches.")
    p.add_argument("--batch-size", type=_positive_int, default=4)
    p.add_argument("--seq-len", type=_positive_int, default=2048)
    p.add_argument("--k-retrieved", type=_positive_int, default=8,
                   help="Must match the training run's --xattn-top-k.")
    p.add_argument("--retrieval-bias-weight", type=_finite_positive_float,
                   default=1.0)
    p.add_argument("--retrieval-temperature", type=float, default=None)
    p.add_argument("--lm-head-top-k", type=_positive_int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _fmt_fraction(x: float) -> str:
    return "nan" if math.isnan(x) else f"{x:.4f}"


def _fmt_cos(x: float) -> str:
    if math.isnan(x):
        return "(n/a)"
    return f"{x:+.3f}"


def _write_matched_gaussian_alt(
    real_table_path: Path, seed: int, out_dir: Path,
) -> Path:
    """Sample a random-gaussian alt table at matched per-dim stats, write it.

    Returns the path to a fresh .safetensors file that `load_capgap_model`
    can consume via its existing `load_corpus_table` contract.
    """
    from safetensors.torch import save_file
    real_table = EngramCrossAttention.load_corpus_table(str(real_table_path))
    alt_table = _sample_matched_gaussian_alt(real_table, seed)
    out_path = out_dir / f"probe_alt_table_seed{seed}.safetensors"
    save_file({"engram.weight": alt_table}, str(out_path))
    return out_path


def run_probe(args: argparse.Namespace) -> None:
    import tempfile
    tmpdir = Path(tempfile.mkdtemp(prefix="zeb133_"))
    print(f"Sampling random-gaussian alt table (seed={args.alt_table_seed}) at {tmpdir} ...")
    alt_table_path = _write_matched_gaussian_alt(
        args.real_table, args.alt_table_seed, tmpdir,
    )

    print(f"Loading real-oracle model from {args.ckpt} with table {args.real_table.name} ...")
    real_model, config, layers = load_capgap_model(
        args.ckpt, args.real_table, args.k_retrieved,
        retrieval_bias_weight=args.retrieval_bias_weight,
        retrieval_temperature=args.retrieval_temperature,
        device=args.device,
    )
    print(f"Loading alt-oracle model from {args.ckpt} with matched-gaussian alt table ...")
    shuf_model, _, _ = load_capgap_model(
        args.ckpt, alt_table_path, args.k_retrieved,
        retrieval_bias_weight=args.retrieval_bias_weight,
        retrieval_temperature=args.retrieval_temperature,
        device=args.device,
    )

    real_model.eval()
    shuf_model.eval()

    num_batches = max(1, args.num_samples // args.batch_size)
    val_loader = make_hf_dataloader(
        args.val_data, args.seq_len, args.batch_size, args.seed,
    )
    batches: list[torch.Tensor] = []
    for _ in range(num_batches):
        b = next(val_loader)[:, :-1]
        batches.append(b.to(args.device))

    last_inj = layers[-1]
    num_total_layers = len(real_model.layers)
    print(
        f"\nProbing decay across {num_batches} batches "
        f"(B={args.batch_size}, L={args.seq_len}). "
        f"Last injection = L{last_inj}; post-injection transformer layers "
        f"= [{', '.join(f'L{i}' for i in range(last_inj + 1, num_total_layers))}].\n"
    )

    # Accumulate per-batch stats and average at the end — memory-safe
    # (never holds all hidden states at once).
    site_fraction_sums: dict[str, float] = {}
    site_cos_sums: dict[str, float] = {}
    lm_max_sum = 0.0
    lm_top10_sum = 0.0
    n = 0

    # Order sites for the final report. Keys are generic (post_layer_1,
    # post_layer_2, ...) but we pretty-print them with absolute layer
    # indices for readability.
    num_post = num_total_layers - last_inj - 1
    ordered_keys: list[str] = ["L_pre_inj", "L_post_inj"]
    for i in range(1, num_post + 1):
        ordered_keys.append(f"L_post_layer_{i}")
    ordered_keys.append("LM_head_in")

    # Print labels for ordered_keys combining the generic site name and
    # the concrete layer index for this run.
    def _pretty(site: str) -> str:
        if site == "L_pre_inj":
            return f"L{last_inj}_pre_inj"
        if site == "L_post_inj":
            return f"L{last_inj}_post_inj"
        if site.startswith("L_post_layer_"):
            offset = int(site.rsplit("_", 1)[-1])
            return f"L{last_inj + offset}_out"
        return site

    lm_head_weight = real_model.lm_head.weight.detach().float().cpu()

    for batch in batches:
        out = measure_layer_decay(real_model, shuf_model, batch)
        for k in ordered_keys:
            site_fraction_sums[k] = site_fraction_sums.get(k, 0.0) + out[k]["fraction"]
            site_cos_sums[k] = site_cos_sums.get(k, 0.0) + out[k]["direction_cos_vs_L5"]

        # LM-head row alignment: recompute the delta at LM-head-in and
        # project. Needs the LM-head weight on CPU (or device) and the
        # captured delta.
        with torch.no_grad(), \
             _DecayCapture(real_model, last_inj) as cr, \
             _DecayCapture(shuf_model, last_inj) as cs:
            _ = real_model(batch)
            _ = shuf_model(batch)
            delta_lm = (
                torch.cat(cr.captured["lm_head_in"], dim=0)
                - torch.cat(cs.captured["lm_head_in"], dim=0)
            )
        stats = lm_head_row_alignment(delta_lm, lm_head_weight, top_k=args.lm_head_top_k)
        lm_max_sum += stats["max_cos"]
        lm_top10_sum += stats["top10_mean_cos"]
        n += 1

    # Average.
    site_stats = {
        k: {
            "fraction": site_fraction_sums[k] / n,
            "direction_cos_vs_L5": site_cos_sums[k] / n,
        }
        for k in ordered_keys
    }
    lm_stats = {
        "max_cos": lm_max_sum / n,
        "top10_mean_cos": lm_top10_sum / n,
    }

    # Report.
    print("=" * 78)
    print("(ZEB-133) Layer-wise content-decay probe")
    print("=" * 78)
    print(f"\nSite            | fraction  | dir_cos_vs_L5")
    print("----------------+-----------+---------------")
    for k in ordered_keys:
        print(
            f"{_pretty(k):<15s} |  {_fmt_fraction(site_stats[k]['fraction']):>7s}  | "
            f"{_fmt_cos(site_stats[k]['direction_cos_vs_L5']):>10s}"
        )
    print("\nLM-head alignment at LM_head_in:")
    print(f"  max row cos:      {lm_stats['max_cos']:.4f}")
    print(f"  mean-top-{args.lm_head_top_k} cos:  {lm_stats['top10_mean_cos']:.4f}")

    verdict = verdict_from_stats(site_stats, lm_stats)
    print(f"\nVerdict: {verdict}")
    _print_verdict_reasoning(verdict, site_stats, lm_stats)


def _print_verdict_reasoning(
    verdict: str,
    site_stats: dict[str, dict[str, float]],
    lm_stats: dict[str, float],
) -> None:
    post_inj = site_stats["L_post_inj"]["fraction"]
    lm_in = site_stats["LM_head_in"]["fraction"]
    ratio = lm_in / max(post_inj, 1e-12)
    print("Reasoning:")
    if verdict == "H-decay":
        print(
            f"  L_post_inj fraction = {post_inj:.4f} but LM_head_in fraction = "
            f"{lm_in:.4f} (survival ratio {ratio:.2f}). Downstream layers or "
            "final_norm are filtering out the injected delta. Fix: route the "
            "engram output closer to logits (ZEB-134 skip-to-logit)."
        )
    elif verdict == "H-blindness":
        print(
            f"  Δ magnitude survives to LM-head ({ratio:.2f}× of L_post_inj) but "
            f"max LM-head row |cos| = {lm_stats['max_cos']:.4f}. The surviving "
            "signal does not align with any vocabulary direction, so the LM-head "
            "cannot read it. Fix: train a tuned-lens-style adapter into logit "
            "space (ZEB-134)."
        )
    elif verdict == "H-tiny":
        print(
            f"  L_post_inj fraction = {post_inj:.4f} is already at floor. Even "
            "perfect preservation downstream would not yield material val-loss "
            "movement. Fix: audit injection magnitude / gate (ZEB-135 EMA-sub "
            "may help if baseline dominates)."
        )
    else:
        print(
            "  Signal survives and aligns with LM-head rows — expected to move "
            "val-loss. If it does not, the bottleneck is elsewhere (e.g. "
            "per-token routing is wrong). Investigate cross-run cos, "
            "per-row Jaccard."
        )


def main() -> None:
    run_probe(parse_args())


if __name__ == "__main__":
    main()
