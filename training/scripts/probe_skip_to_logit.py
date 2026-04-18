"""ZEB-134 Skip-to-Logit router forensic.

Loads a checkpoint trained with --engram-skip-to-logit on both real and
shuffled oracle tables and reports:

  - Final alpha and W_align frobenius norm for each run
  - Cross-run cosine of the router's engram_logits at matched positions
  - Max LM-head row cos of W_align(engram_out) per position
  - Shannon entropy of softmax(engram_logits) alone

No training, no gradients. ~30 s on a 4090.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_TRAINING = Path(__file__).resolve().parent.parent
if str(_REPO_TRAINING) not in sys.path:
    sys.path.insert(0, str(_REPO_TRAINING))

from ct87.engram import EngramCrossAttention, GatedEngramInjection, SkipToLogitEngramRouter  # noqa: E402
from ct87.model import HarmonyModel, HarmonyModelConfig  # noqa: E402
from ct87.train import make_hf_dataloader  # noqa: E402
from scripts.forensic_eta_b_capgap import _TORCH_LOAD_SAFE_GLOBALS  # noqa: E402


def _positive_int(v: str) -> int:
    i = int(v)
    if i <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {v}")
    return i


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--real-ckpt", required=True, type=Path)
    p.add_argument("--shuffled-ckpt", required=True, type=Path)
    p.add_argument("--real-table", required=True, type=Path)
    p.add_argument("--shuffled-table", required=True, type=Path)
    p.add_argument("--val-data", required=True, type=str)
    p.add_argument("--num-batches", type=_positive_int, default=2)
    p.add_argument("--batch-size", type=_positive_int, default=4)
    p.add_argument("--seq-len", type=_positive_int, default=2048)
    p.add_argument("--k-retrieved", type=_positive_int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _load_model_with_skip_router(
    ckpt_path: Path, table_path: Path, k_retrieved: int, device: str,
) -> tuple[HarmonyModel, HarmonyModelConfig, list[int]]:
    """Load a checkpoint that was trained with --engram-skip-to-logit.

    Mirrors load_capgap_model but additionally attaches a
    SkipToLogitEngramRouter so the saved router params load correctly.
    alpha_init here is arbitrary — the state_dict overwrites log_alpha.
    """
    with torch.serialization.safe_globals(_TORCH_LOAD_SAFE_GLOBALS):
        payload: dict[str, Any] = torch.load(
            str(ckpt_path), map_location="cpu", weights_only=True,
        )
    config: HarmonyModelConfig = payload["config"]
    inject_layers = list(config.engram_inject_layers)
    if not inject_layers:
        raise ValueError(f"{ckpt_path}: empty engram_inject_layers")

    model = HarmonyModel(config)
    table = EngramCrossAttention.load_corpus_table(str(table_path))
    injections: dict[int, GatedEngramInjection] = {}
    for layer_idx in inject_layers:
        xattn = EngramCrossAttention(
            config, table, num_heads=config.num_query_heads,
            k_retrieved=k_retrieved, retrieval_bias_weight=1.0,
            retrieval_temperature=None,
        )
        injections[layer_idx] = GatedEngramInjection(
            xattn, alpha_init=config.engram_gate_init,
        )
    model.attach_gated_engram_injections(injections)

    # Detect the skip-router keys and attach a router shell; load will
    # overwrite log_alpha + W_align.
    has_skip = any(
        k.startswith("engram_skip_router.")
        for k in payload["model_state_dict"]
    )
    if has_skip:
        router = SkipToLogitEngramRouter(
            hidden_dim=config.hidden_dim,
            lm_head_weight=model.lm_head.weight,
            alpha_init=0.1,  # overwritten by state_dict
        )
        model.attach_engram_skip_router(router)

    missing, unexpected = model.load_state_dict(
        payload["model_state_dict"], strict=False,
    )
    real_missing = [k for k in missing if ".table" not in k]
    real_unexpected = [k for k in unexpected if ".table" not in k]
    if real_missing or real_unexpected:
        raise RuntimeError(
            f"{ckpt_path.name}: state_dict load mismatch. "
            f"missing={real_missing[:5]} unexpected={real_unexpected[:5]}"
        )
    model.eval().to(device)
    return model, config, inject_layers


@torch.no_grad()
def run_probe(args: argparse.Namespace) -> None:
    print(f"Loading {args.real_ckpt} (real) ...")
    real_model, config, layers = _load_model_with_skip_router(
        args.real_ckpt, args.real_table, args.k_retrieved, args.device,
    )
    print(f"Loading {args.shuffled_ckpt} (shuf) ...")
    shuf_model, _, _ = _load_model_with_skip_router(
        args.shuffled_ckpt, args.shuffled_table, args.k_retrieved, args.device,
    )

    if real_model.engram_skip_router is None or shuf_model.engram_skip_router is None:
        raise RuntimeError(
            "Checkpoints missing engram_skip_router.* keys — rerun training "
            "with --engram-skip-to-logit."
        )

    val = make_hf_dataloader(
        args.val_data, args.seq_len, args.batch_size, args.seed,
    )
    batches = [
        next(val)[:, :-1].to(args.device) for _ in range(args.num_batches)
    ]

    last = layers[-1]

    def _engram_out_and_logits(model: HarmonyModel, batch: torch.Tensor):
        """Return (engram_out at last injection, skip_logits)."""
        # The model's forward stashes the last injection on
        # _last_engram_skip_input. Running forward then reading the
        # attribute gives us exactly that tensor without installing
        # hooks.
        _ = model(batch)
        eo = model._last_engram_skip_input
        if eo is None:
            raise RuntimeError("engram skip input not captured; attach failed?")
        skip_logits = model.engram_skip_router(eo)
        return eo.detach().float(), skip_logits.detach().float()

    cross_run_cos_sum = 0.0
    max_row_cos_sum = 0.0
    entropy_sum = 0.0
    n = 0
    for batch in batches:
        eo_r, lg_r = _engram_out_and_logits(real_model, batch)
        eo_s, lg_s = _engram_out_and_logits(shuf_model, batch)

        # Cross-run cos on engram logits [B, L, V], pooled per position.
        a = F.normalize(lg_r.reshape(-1, lg_r.shape[-1]), dim=-1)
        b = F.normalize(lg_s.reshape(-1, lg_s.shape[-1]), dim=-1)
        cross_run_cos_sum += (a * b).sum(dim=-1).mean().item()

        # Max LM-head row cos: the rotated engram output W_align(eo_r)
        # projected onto each lm_head row.
        lm_w = real_model.lm_head.weight.detach().float().cpu()
        aligned = real_model.engram_skip_router.W_align(
            eo_r.to(real_model.engram_skip_router.W_align.weight.dtype)
        ).float().cpu()
        a_flat = F.normalize(aligned.reshape(-1, aligned.shape[-1]), dim=-1)
        w_n = F.normalize(lm_w, dim=-1)
        # [positions, vocab] cosine magnitudes; take per-position max.
        per_pos_max = (a_flat @ w_n.T).abs().max(dim=-1).values
        max_row_cos_sum += per_pos_max.mean().item()

        # Entropy of engram_logits softmax. Higher = more uniform.
        probs = F.softmax(lg_r.reshape(-1, lg_r.shape[-1]), dim=-1)
        per_pos_entropy = -(probs * probs.clamp(min=1e-12).log()).sum(dim=-1)
        entropy_sum += per_pos_entropy.mean().item()
        n += 1

    print("\n" + "=" * 78)
    print("(ZEB-134) Skip-to-Logit router forensic")
    print("=" * 78)
    for tag, mdl in [("real", real_model), ("shuf", shuf_model)]:
        r = mdl.engram_skip_router
        print(
            f"  {tag}: log_alpha={r.log_alpha.item():+.4f}  "
            f"alpha=exp={r.alpha.item():.4f}  "
            f"||W_align||_F={r.W_align.weight.detach().float().norm().item():.4f}"
        )
    print(f"\n  cross_run_cos engram_logits  =  {cross_run_cos_sum / n:+.4f}")
    print(f"  max LM-head row |cos|        =  {max_row_cos_sum / n:.4f}")
    print(f"  engram_logit_entropy (nats)  =  {entropy_sum / n:.4f}  "
          f"(log(vocab) = {math.log(config.vocab_size):.4f})")


def main() -> None:
    run_probe(parse_args())


if __name__ == "__main__":
    main()
