"""Training loop for ct87 -- CLI entry point.

Usage:
    python -m ct87.train --config tiny --data <path> --steps 200

For testing without a real dataset, use --synthetic to generate random tokens.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from typing import TYPE_CHECKING, Iterator

import torch
import torch.nn.functional as F

from ct87.model import HarmonyModel, HarmonyModelConfig
from ct87.optim import Muon, WSDSchedule, partition_params

if TYPE_CHECKING:
    from ct87.engram import EngramTable


def make_synthetic_dataloader(
    vocab_size: int, seq_len: int, batch_size: int, seed: int = 42,
) -> Iterator[torch.Tensor]:
    """Infinite dataloader of random token sequences for testing.

    Yields batches of shape [batch_size, seq_len + 1] (extra token for targets).
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    while True:
        yield torch.randint(0, vocab_size, (batch_size, seq_len + 1), generator=rng)


def make_hf_dataloader(
    data_path: str, seq_len: int, batch_size: int, seed: int = 42,
) -> Iterator[torch.Tensor]:
    """Infinite dataloader from a pre-tokenized HuggingFace dataset.

    Loads the dataset eagerly, concatenates all token sequences into one
    long stream, then returns an iterator that yields random windows of
    [seq_len + 1] tokens packed into batches.

    Raises ValueError if the dataset has fewer tokens than seq_len + 1
    (the minimum needed for one input/target pair).
    """
    from datasets import load_from_disk

    dataset = load_from_disk(data_path)
    all_tokens: list[int] = []
    for example in dataset:
        all_tokens.extend(example["input_ids"])

    all_tokens_t = torch.tensor(all_tokens, dtype=torch.long)
    total = len(all_tokens_t)
    window = seq_len + 1

    if total < window:
        raise ValueError(
            f"Dataset at {data_path} has {total} tokens, but seq_len+1={window} "
            f"tokens are needed for at least one training window. "
            f"Use a larger dataset or reduce --seq-len."
        )

    def _iter() -> Iterator[torch.Tensor]:
        rng = torch.Generator()
        rng.manual_seed(seed)
        while True:
            starts = torch.randint(0, total - window + 1, (batch_size,), generator=rng)
            batch = torch.stack([all_tokens_t[s : s + window] for s in starts])
            yield batch

    return _iter()


def save_checkpoint(
    model: HarmonyModel,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    output_dir: str,
) -> None:
    """Save model weights (safetensors) and optionally optimizer state."""
    from safetensors.torch import save_model

    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(output_dir, f"model_step_{step}.safetensors")
    save_model(model, weights_path)

    if optimizer is not None:
        opt_path = os.path.join(output_dir, f"optimizer_step_{step}.pt")
        torch.save(optimizer.state_dict(), opt_path)


def _save_thought_norm(
    thought_norm: torch.nn.Module,
    step: int,
    output_dir: str,
) -> None:
    """Save ThoughtNorm weights alongside the model checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"thought_norm_step_{step}.pt")
    torch.save(thought_norm.state_dict(), path)


def _save_uq_head(
    uq_head: torch.nn.Module,
    step: int,
    output_dir: str,
) -> None:
    """Save UQ head weights alongside the model checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"uq_head_step_{step}.pt")
    torch.save(uq_head.state_dict(), path)


def _save_mtp_head(
    mtp_head: torch.nn.Module,
    step: int,
    output_dir: str,
) -> None:
    """Save MTP head weights alongside the model checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"mtp_head_step_{step}.pt")
    torch.save(mtp_head.state_dict(), path)


def _save_latent_projection(
    projection: torch.nn.Module,
    step: int,
    output_dir: str,
) -> None:
    """Save latent projection weights alongside the model checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"latent_projection_step_{step}.pt")
    torch.save(projection.state_dict(), path)


def save_resumable_checkpoint(
    model: HarmonyModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    output_dir: str,
    rng_state: dict | None = None,
    last_val_loss: float | None = None,
    dynamic_entropy_lambda: float | None = None,
) -> None:
    """Save a full resumable checkpoint with atomic rename.

    NOTE: Only saves model + optimizer state. Auxiliary modules
    (thought_norm, uq_head, mtp_head, latent_projection) are NOT
    included — they're rebuilt from scratch on resume. This is fine
    for engram experiments (epsilon/ablations) which don't use those
    modules. Extend when needed.
    """
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "last_val_loss": last_val_loss,
        # Persist the model config so --init-from can run its architecture-
        # compatibility check against checkpoints produced by this script.
        # Without this, the compat check silently skips (because
        # ckpt.get('config') is None) and strict=False would load zero
        # backbone params on an architecture mismatch.
        "config": model.config,
    }
    if rng_state is not None:
        payload["rng_state"] = rng_state
    if dynamic_entropy_lambda is not None:
        payload["dynamic_entropy_lambda"] = dynamic_entropy_lambda

    ckpt_path = os.path.join(output_dir, "checkpoint.pt")
    prev_path = os.path.join(output_dir, "checkpoint_prev.pt")
    tmp_path = os.path.join(output_dir, "checkpoint.tmp")

    torch.save(payload, tmp_path)

    if os.path.exists(ckpt_path):
        if os.path.exists(prev_path):
            os.remove(prev_path)
        os.rename(ckpt_path, prev_path)
    os.rename(tmp_path, ckpt_path)


def capture_rng_state(
    device: torch.device | None = None,
    capgap_shuffle_gen: torch.Generator | None = None,
) -> dict:
    """Capture all RNG states for deterministic resume.

    The optional `capgap_shuffle_gen` is the dedicated V-contrast shuffle
    generator (see `--engram-vcontrast-shuffle-seed`). It advances with each
    training forward, so resuming without restoring its state would replay
    permutations from step 0 and break the reproducibility contract.
    """
    import random
    import numpy as np

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if device is not None and device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state(device)
    elif torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state()
    if capgap_shuffle_gen is not None:
        state["capgap_shuffle_gen"] = capgap_shuffle_gen.get_state()
    return state


def restore_rng_state(
    state: dict,
    device: torch.device | None = None,
    capgap_shuffle_gen: torch.Generator | None = None,
) -> None:
    """Restore all RNG states from a checkpoint."""
    import random
    import numpy as np

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state:
        if device is not None and device.type == "cuda":
            torch.cuda.set_rng_state(state["torch_cuda"], device)
        elif torch.cuda.is_available():
            torch.cuda.set_rng_state(state["torch_cuda"])
    # Restore the V-contrast shuffle generator in-place (the wrappers already
    # hold a reference to it, so mutating its state is enough). Silently skip
    # when either the checkpoint or the current run lacks the generator — one
    # side may be a pre-θ-V-contrast checkpoint, or the current run may not
    # use --engram-vcontrast-shuffle-seed.
    if capgap_shuffle_gen is not None and "capgap_shuffle_gen" in state:
        capgap_shuffle_gen.set_state(state["capgap_shuffle_gen"])


def load_checkpoint(
    model: HarmonyModel,
    output_dir: str,
    step: int,
) -> None:
    """Load model weights from a safetensors checkpoint."""
    from safetensors.torch import load_model

    weights_path = os.path.join(output_dir, f"model_step_{step}.safetensors")
    load_model(model, weights_path)


def set_lr(optimizer: torch.optim.Optimizer, multiplier: float) -> None:
    """Set learning rate for all param groups based on schedule multiplier."""
    for group in optimizer.param_groups:
        if "_base_lr" not in group:
            group["_base_lr"] = group["lr"]
        group["lr"] = group["_base_lr"] * multiplier


def compute_validation_loss(
    model: HarmonyModel,
    val_loader: Iterator[torch.Tensor],
    vocab_size: int,
    device: torch.device,
    num_batches: int = 10,
    amp_dtype: torch.dtype | None = None,
    engram_table: EngramTable | None = None,
    thought_norm: torch.nn.Module | None = None,
    think_token_id: int | None = None,
    num_thoughts: int = 0,
    latent_projection: torch.nn.Module | None = None,
) -> float:
    """Run validation and return average cross-entropy loss.

    When thought_norm and think_token_id are provided with num_thoughts > 0,
    uses COCONUT forward/loss so val_loss matches the training objective.

    When latent_projection is provided, uses projection-generated keys for
    engram lookup instead of xxhash (must also provide engram_table).
    """
    was_training = model.training
    model.train(False)
    use_amp = amp_dtype is not None
    device_type = device.type
    use_coconut = thought_norm is not None and num_thoughts > 0
    if use_coconut:
        from ct87.coconut import coconut_forward, coconut_loss
    try:
        total_loss = 0.0
        with torch.no_grad():
            for _ in range(num_batches):
                batch = next(val_loader).to(device)
                input_ids = batch[:, :-1]
                targets = batch[:, 1:]
                engram_emb = None
                if engram_table is not None and latent_projection is not None:
                    emb = model.embed_tokens(input_ids)
                    engram_emb = engram_table.lookup_batch_projected(
                        input_ids, emb, latent_projection,
                    )
                elif engram_table is not None:
                    engram_emb = engram_table.lookup_batch(input_ids)
                with torch.autocast(device_type, dtype=amp_dtype, enabled=use_amp):
                    if use_coconut:
                        logits, think_mask = coconut_forward(
                            model, thought_norm, input_ids,
                            think_token_id, num_thoughts,
                            engram_embeddings=engram_emb,
                        )
                        think_targets = torch.full(
                            (targets.shape[0], num_thoughts), -100,
                            dtype=targets.dtype, device=targets.device,
                        )
                        aug_targets = torch.cat([think_targets, targets], dim=1)
                        loss = coconut_loss(logits, aug_targets, think_mask)
                    else:
                        logits = model(input_ids, engram_embeddings=engram_emb)
                        loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
                total_loss += loss.item()
    finally:
        model.train(was_training)
    return total_loss / num_batches


def detect_device(requested: str | None) -> torch.device:
    """Auto-detect best available device."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lambda_schedule(step: int, warmup_steps: int, target: float) -> float:
    """Linear warmup from 0 to `target` over `warmup_steps` steps; constant `target` after.

    theta-V-contrast / iota-Q-diversity aux-loss schedule (ZEB-130). When `warmup_steps` is 0
    (or negative), the linear ramp is skipped and `target` is returned for all steps.
    """
    if warmup_steps <= 0:
        return target
    if step >= warmup_steps:
        return target
    return target * step / warmup_steps


def freeze_backbone_for_capgap(model: "HarmonyModel") -> None:
    """Freeze all parameters except engram_injections.* (η-B / ZEB-130).

    Sets requires_grad=False on every parameter whose name does not start
    with "engram_injections." — the optimizer then only sees the
    GatedEngramInjection params (xattn projections + alpha scalars) when
    constructed from a requires_grad filter.

    Raises RuntimeError if the model has no attached engram_injections —
    freezing everything with zero trainable params would leave the optimizer
    with empty param groups and the first optimizer step would crash with
    an obscure PyTorch error.
    """
    if getattr(model, "engram_injections", None) is None:
        raise RuntimeError(
            "freeze_backbone_for_capgap() requires engram_injections to be "
            "attached (use --config tiny_engram_xattn_capgap and construct "
            "GatedEngramInjection modules first). The model has no "
            "engram_injections.* params — freezing everything would leave "
            "the optimizer with zero trainable parameters."
        )
    # Both the multi-layer injection modules and the ZEB-134 skip-to-logit
    # router are research-only adapters that must train under capgap.
    # Trailing dots in the prefixes matter: "engram_injections.0.alpha" is
    # trainable, but a hypothetical future "engram_injections_shared"
    # sibling would NOT be caught by `startswith("engram_injections")`
    # alone and would silently be left trainable.
    trainable_prefixes = ("engram_injections.", "engram_skip_router.")
    frozen_count = 0
    trainable_count = 0
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in trainable_prefixes):
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()
    print(
        f"[capgap] Frozen {frozen_count:,} backbone params; "
        f"{trainable_count:,} engram params remain trainable."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ct87 model")
    parser.add_argument(
        "--config",
        choices=[
            "tiny", "target", "tiny_ffn_expanded",
            "tiny_engram_ann", "tiny_engram_ann_routed",
            "tiny_engram_xattn", "tiny_engram_xattn_routed",
            "tiny_engram_xattn_consol_online",
            "tiny_engram_xattn_consol_phased",
            "tiny_engram_xattn_ctrl",
            "tiny_engram_xattn_capgap",
            "tiny_engram_xattn_capgap_vcontrast",
            "tiny_engram_xattn_capgap_qdiv",
            "tiny_engram_xattn_capgap_vcontrast_qdiv",
        ],
        default="tiny",
        help="Model config. 'tiny_ffn_expanded' is Model beta (params-matched "
             "dense control). 'tiny_engram_ann' enables ZEB-117 Model gamma "
             "(ANN retrieval + gated residual + anti-collapse); requires "
             "--engram-ann-table. 'tiny_engram_xattn' enables ZEB-117 Model "
             "delta (cross-attention to memory + top-k retrieval); requires "
             "--engram-xattn-table.",
    )
    parser.add_argument("--data", type=str, default=None, help="Path to pre-tokenized HF dataset")
    parser.add_argument("--val-data", type=str, default=None, help="Path to validation HF dataset (optional)")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic random data")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument(
        "--checkpoint-interval", type=int, default=0,
        help="Save resumable checkpoint every N steps (0=disabled). "
             "Includes model, optimizer, and global RNG state. "
             "Dataloader position is not checkpointed, so resumed "
             "runs may see different batch ordering. "
             "Retains last 2 checkpoints for crash recovery.",
    )
    parser.add_argument("--output-dir", type=str, default="training/checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-file", type=str, default=None, help="Path to CSV log file")
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Path to safetensors checkpoint to resume/fine-tune from",
    )
    parser.add_argument(
        "--init-from", type=str, default=None,
        help=(
            "Path to a checkpoint .pt file for weight initialization only "
            "(η-B capacity-gap / ZEB-130). Loads model weights via "
            "strict=False (missing keys for new engram injections are "
            "allowed), then starts training fresh (step=0, fresh optimizer, "
            "fresh RNG). Mutually exclusive with --resume-from."
        ),
    )
    parser.add_argument(
        "--allow-partial-init", action="store_true",
        help=(
            "Permit --init-from to proceed when non-engram backbone keys are "
            "missing (otherwise the load fails hard to prevent silently "
            "running with a randomly-initialized backbone). Also permits "
            "--init-from when the source checkpoint has no config metadata "
            "to compare against."
        ),
    )
    parser.add_argument(
        "--freeze-backbone", action="store_true",
        help=(
            "Freeze all non-engram parameters (η-B capacity-gap / ZEB-130). "
            "Only engram_injections.* params receive gradients and are "
            "added to the optimizer. Typically combined with --init-from "
            "and the tiny_engram_xattn_capgap config."
        ),
    )
    parser.add_argument(
        "--xattn-top-k", type=int, default=4,
        help="top-k for cross-attention retrieval (shared by δ and η-B paths).",
    )
    parser.add_argument(
        "--gradient-checkpoint", action="store_true",
        help="Enable gradient checkpointing (recompute activations to save VRAM)",
    )
    parser.add_argument(
        "--engram-table", type=str, default=None,
        help="Path to Engram safetensors table (enables Engram injection during training)",
    )
    parser.add_argument(
        "--engram-seeds", type=str, default="42,99,137,251",
        help="Comma-separated xxhash64 seeds for Engram table lookup (default: 42,99,137,251)",
    )
    parser.add_argument(
        "--latent-projection", type=str, default=None,
        help="Path to frozen latent projection checkpoint (switches engram lookup "
             "from xxhash to projection-generated keys for fine-tuning)",
    )

    # ---- ZEB-117 Model gamma: ANN engram injection + anti-collapse ----
    parser.add_argument(
        "--engram-ann-table", type=str, default=None,
        help="Path to corpus engram safetensors for Model gamma ANN retrieval. "
             "Required when --config=tiny_engram_ann.",
    )
    parser.add_argument(
        "--engram-ann-warmup-steps", type=int, default=800,
        help="Gate hard-clamp duration (steps) for Model gamma (default: 800). "
             "Before this step count the gate is clamped to min "
             "--engram-ann-gate-clamp-min to force gradient flow through "
             "the memory path during the chaotic-alignment phase.",
    )
    parser.add_argument(
        "--engram-ann-gate-clamp-min", type=float, default=0.5,
        help="Minimum gate value during warmup (default: 0.5).",
    )
    parser.add_argument(
        "--engram-ann-entropy-weight", type=float, default=5e-3,
        help="lambda_ent for gate-entropy regularization (default: 0.005). "
             "Recommended range: [1e-3, 1e-2]. See research report s3.3.1.",
    )
    parser.add_argument(
        "--engram-ann-entropy-bounds", type=str, default="0.1,0.4",
        help="Comma-separated (low, high) bounds on moving-average gate "
             "probability; lambda_ent is dynamically scaled to keep the mean "
             "inside this band. Default '0.1,0.4'. Pass 'off' to disable "
             "dynamic scaling.",
    )

    # ---- ZEB-117 Model delta: cross-attention engram injection ----
    parser.add_argument(
        "--engram-xattn-table", type=str, default=None,
        help="Path to corpus engram safetensors for Model delta cross-attention "
             "retrieval. Required when --config=tiny_engram_xattn. Same format "
             "as --engram-ann-table.",
    )
    parser.add_argument(
        "--engram-xattn-k-retrieved", type=int, default=8,
        help="Number of top-k corpus neighbors each position attends to "
             "(default: 8). Memory scales linearly.",
    )
    parser.add_argument(
        "--engram-xattn-retrieval-bias-weight", type=float, default=1.0,
        help="Weight on the retrieval-similarity bias added to cross-attention "
             "logits (default: 1.0). Preserves gradient flow into the "
             "retrieval-query projection (top-k gather is non-differentiable).",
    )
    parser.add_argument(
        "--engram-xattn-num-heads", type=int, default=None,
        help="Override the cross-attention head count (default: inherit "
             "config.num_query_heads). hidden_dim must be divisible by this value.",
    )
    # ---- ZEB-130: θ-V-contrast ----
    parser.add_argument(
        "--engram-vcontrast", action="store_true",
        help=(
            "Enable θ-V-contrast V-contrastive auxiliary loss (ZEB-130). "
            "Requires --config=tiny_engram_xattn_capgap_vcontrast (or another "
            "preset with engram_inject_layers set). Adds a per-layer aux loss "
            "penalizing cosine alignment between real-table and shuffled-table "
            "post-o_proj outputs."
        ),
    )
    parser.add_argument(
        "--engram-vcontrast-lambda", type=float, default=None,
        help=(
            "Override the config's engram_vcontrast_lambda (default 1.0). "
            "Aux loss is added as lambda * sum_layers(aux_loss_l)."
        ),
    )
    parser.add_argument(
        "--engram-vcontrast-warmup-steps", type=int, default=None,
        help=(
            "Override the config's engram_vcontrast_warmup_steps "
            "(default 200). Linear warmup from 0 to lambda."
        ),
    )
    parser.add_argument(
        "--engram-vcontrast-shuffle-seed", type=int, default=None,
        help=(
            "Optional seed for the per-step shuffle generator. Default: "
            "use the global PyTorch RNG (preferred for production runs; "
            "the seed is for reproducibility debugging only)."
        ),
    )
    # ---- ZEB-130: iota-Q-diversity ----
    parser.add_argument(
        "--engram-qdiv",
        action="store_true",
        help=(
            "Enable Q-side load-balancing aux loss. Must match the selected "
            "preset's engram_qdiv_enabled value."
        ),
    )
    parser.add_argument(
        "--engram-qdiv-lambda",
        type=float,
        default=None,
        help=(
            "Override lambda for Q-div aux loss. Requires --engram-qdiv + a "
            "qdiv-enabled preset. None = use preset's value."
        ),
    )
    parser.add_argument(
        "--engram-qdiv-warmup-steps",
        type=int,
        default=None,
        help=(
            "Override warmup steps for Q-div lambda. Requires --engram-qdiv + "
            "a qdiv-enabled preset. None = use preset's value."
        ),
    )
    # ---- ZEB-134: Skip-to-Logit engram router ----
    parser.add_argument(
        "--engram-skip-to-logit",
        action="store_true",
        help=(
            "Attach a SkipToLogitEngramRouter that feeds the engram "
            "output at the last entry of engram_inject_layers directly "
            "into the logit sum, bypassing the frozen decoder layers. "
            "Tests whether the ι₂ forensic-success / LM-blindness "
            "regime is a decoder-bypass problem (ZEB-134). Requires a "
            "multi-layer gated-injection preset (tiny_engram_xattn_capgap*)."
        ),
    )
    parser.add_argument(
        "--engram-skip-alpha-init",
        type=float,
        default=0.1,
        help=(
            "Initial alpha for the Skip-to-Logit router (ZEB-134). "
            "Must be finite and > 0. Stored internally as log_alpha so "
            "alpha stays positive without a hard clamp. 0.1 is small "
            "enough to avoid destabilizing the main logit path on the "
            "first training step while leaving room for alpha to grow."
        ),
    )
    # ---- ZEB-139: KL+CE hybrid loss (Memory-Decoder-style) ----
    parser.add_argument(
        "--kl-lambda",
        type=float,
        default=0.0,
        help=(
            "Mixing weight for the ZEB-139 KL+CE hybrid loss: "
            "L = (1-λ)·CE + λ·KL(softmax(skip_logits) || softmax("
            "teacher_logits[trigram_hash])). Default 0 disables the KL "
            "term. Requires --engram-skip-to-logit (KL targets the "
            "Skip-to-Logit router's output, not the final mixed logits) "
            "and --oracle-teacher-logits (the bf16 sidecar produced by "
            "generate_oracle_table.py --save-teacher-logits)."
        ),
    )
    parser.add_argument(
        "--oracle-teacher-logits",
        type=str,
        default=None,
        help=(
            "Path to the teacher-logits sidecar (a safetensors file with "
            "tensor 'teacher_logits.weight' shape [total_entries, vocab] "
            "dtype bf16, produced by generate_oracle_table.py --save-"
            "teacher-logits). Required when --kl-lambda > 0. Loaded into "
            "CPU bf16 at startup (~640 MB at 10K rows × 32K vocab); "
            "per-batch row gather is the only GPU traffic."
        ),
    )
    parser.add_argument(
        "--latent-intermediate-dim", type=int, default=None,
        help="Intermediate dimension for latent projection MLP",
    )
    parser.add_argument(
        "--latent-dim", type=int, default=None,
        help="Output dimension for latent projection MLP",
    )
    parser.add_argument(
        "--latent-projection-init", action="store_true",
        help="Randomly initialize latent projection (for from-scratch co-training)",
    )
    parser.add_argument(
        "--contrastive-loss", action="store_true",
        help="Enable contrastive co-training (makes projection trainable, uses "
             "projection keys for engram retrieval with InfoNCE auxiliary loss)",
    )
    parser.add_argument(
        "--contrastive-loss-weight", type=float, default=0.1,
        help="Weight for contrastive auxiliary loss (default: 0.1)",
    )
    parser.add_argument(
        "--contrastive-temperature", type=float, default=0.07,
        help="Temperature for InfoNCE contrastive loss (default: 0.07)",
    )
    parser.add_argument(
        "--contrastive-k", type=int, default=4,
        help="Number of top neighbors to preserve in contrastive loss (default: 4)",
    )
    parser.add_argument(
        "--coconut", action="store_true",
        help="Enable COCONUT continuous thought training",
    )
    parser.add_argument(
        "--think-token-id", type=int, default=None,
        help="Token ID for <think> (required with --coconut)",
    )
    parser.add_argument(
        "--ct-max-steps", type=int, default=4,
        help="Max thought steps for COCONUT curriculum (default: 4)",
    )
    parser.add_argument(
        "--uq-head", action="store_true",
        help="Enable UQ head training (auxiliary uncertainty classification task)",
    )
    parser.add_argument(
        "--uq-loss-weight", type=float, default=0.1,
        help="Weight for UQ auxiliary loss (default: 0.1)",
    )
    parser.add_argument(
        "--mtp-head", action="store_true",
        help="Enable multi-token prediction head (shared-weight recursive MTP)",
    )
    parser.add_argument(
        "--mtp-depth", type=int, default=4,
        help="Number of future tokens to predict (default: 4)",
    )
    parser.add_argument(
        "--mtp-loss-weight", type=float, default=1.0,
        help="Weight for MTP auxiliary loss (default: 1.0)",
    )
    parser.add_argument(
        "--qat", action="store_true",
        help="Enable quantization-aware training (q8_0 fake quantization on base model Linear layers)",
    )
    parser.add_argument(
        "--qat-start-pct", type=float, default=0.9,
        help="Fraction of total steps at which to activate QAT (default: 0.9 = last 10%%)",
    )
    # ---- ZEB-128: Engram consolidation ----
    parser.add_argument(
        "--consolidation-mode", choices=["none", "online", "phased"],
        default="none",
        help="Consolidation strategy: 'none' (control), 'online' (MSE loss "
             "throughout), 'phased' (MSE loss after --consolidation-start-step)",
    )
    parser.add_argument(
        "--consolidation-lambda", type=float, default=0.1,
        help="Weight for consolidation MSE loss (default: 0.1)",
    )
    parser.add_argument(
        "--consolidation-start-step", type=int, default=0,
        help="Step at which consolidation MSE loss activates (default: 0 for "
             "online, typically 7000 for phased)",
    )
    parser.add_argument(
        "--consolidation-anneal", action="store_true",
        help="Linearly anneal injection multiplier from 1.0 to 0.0 during "
             "consolidation phase (only meaningful with --consolidation-mode=phased)",
    )
    parser.add_argument(
        "--zero-injection-eval", action="store_true",
        help="Load checkpoint, zero engram injection, run validation, and exit. "
             "Used for post-removal measurement in consolidation experiments.",
    )
    args = parser.parse_args()

    if args.init_from is not None and args.resume_from is not None:
        print(
            "Error: --init-from and --resume-from are mutually exclusive. "
            "Use --resume-from to continue training from a checkpoint; use "
            "--init-from to load weights only and start a fresh run.",
            file=sys.stderr,
        )
        sys.exit(2)
    if args.init_from is not None and not args.init_from.endswith(".pt"):
        print(
            "Error: --init-from must point to a resumable checkpoint .pt "
            "file (weights + config), not a safetensors file.",
            file=sys.stderr,
        )
        sys.exit(2)

    # η-B capacity-gap contract: --freeze-backbone is meant to train ONLY
    # engram_injections.* params. The optimizer-construction path below still
    # adds thought_norm / uq_head / mtp_head / latent_projection / consol_decoder
    # param groups unconditionally, so combining --freeze-backbone with any of
    # the aux-module flags silently breaks the "injections-only" invariant.
    # Fail here instead so the run doesn't complete hours later as a confounded
    # experiment.
    if args.freeze_backbone:
        aux_conflicts = []
        if args.coconut:
            aux_conflicts.append("--coconut")
        if args.uq_head:
            aux_conflicts.append("--uq-head")
        if args.mtp_head:
            aux_conflicts.append("--mtp-head")
        if args.contrastive_loss:
            aux_conflicts.append("--contrastive-loss")
        if args.consolidation_mode != "none":
            aux_conflicts.append(f"--consolidation-mode={args.consolidation_mode}")
        if aux_conflicts:
            print(
                "Error: --freeze-backbone is incompatible with aux-trainable "
                "modules; capgap runs must optimize only engram_injections.*. "
                f"Conflicting flags: {', '.join(aux_conflicts)}",
                file=sys.stderr,
            )
            sys.exit(2)

    # --zero-injection-eval is an eval-only early-exit path, so training data
    # is not required when that flag is set.
    if (
        args.data is None
        and not args.synthetic
        and not args.zero_injection_eval
    ):
        print("Error: must provide --data <path> or --synthetic", file=sys.stderr)
        sys.exit(1)

    if args.grad_accum_steps < 1:
        print("Error: --grad-accum-steps must be >= 1", file=sys.stderr)
        sys.exit(1)

    if args.coconut and args.think_token_id is None:
        print("Error: --think-token-id is required with --coconut", file=sys.stderr)
        sys.exit(1)

    # ZEB-139 KL+CE validation. Both flags must be set together; either alone
    # is a misconfiguration we want to catch upfront so the user doesn't burn
    # a long training run on a silently-disabled KL term.
    if not 0.0 <= args.kl_lambda <= 1.0:
        print(
            f"Error: --kl-lambda must be in [0.0, 1.0]; got "
            f"{args.kl_lambda!r}.",
            file=sys.stderr,
        )
        sys.exit(2)
    if args.kl_lambda > 0:
        if not args.engram_skip_to_logit:
            print(
                "Error: --kl-lambda > 0 requires --engram-skip-to-logit. "
                "The KL target is the Skip-to-Logit router's output "
                "(skip_logits); without the router there is nothing for "
                "the KL term to operate on.",
                file=sys.stderr,
            )
            sys.exit(2)
        if args.oracle_teacher_logits is None:
            print(
                "Error: --kl-lambda > 0 requires --oracle-teacher-logits "
                "(the bf16 sidecar produced by generate_oracle_table.py "
                "--save-teacher-logits).",
                file=sys.stderr,
            )
            sys.exit(2)
    elif args.oracle_teacher_logits is not None:
        # Symmetric guard: the path is only consulted when KL is on, so
        # passing it without --kl-lambda silently has no effect — surface
        # the misconfiguration before any training time is spent.
        print(
            "Error: --oracle-teacher-logits "
            f"({args.oracle_teacher_logits!r}) was set without "
            "--kl-lambda > 0. The sidecar is only consulted when the KL "
            "term is enabled; pass --kl-lambda > 0 or drop "
            "--oracle-teacher-logits.",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.latent_projection is not None:
        if args.latent_intermediate_dim is None or args.latent_dim is None:
            print(
                "Error: --latent-projection requires both "
                "--latent-intermediate-dim and --latent-dim",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.engram_table is None:
            print(
                "Error: --latent-projection requires --engram-table",
                file=sys.stderr,
            )
            sys.exit(1)

    if args.contrastive_loss:
        if args.latent_projection is None and not args.latent_projection_init:
            print(
                "Error: --contrastive-loss requires --latent-projection or "
                "--latent-projection-init",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.latent_intermediate_dim is None or args.latent_dim is None:
            print(
                "Error: --contrastive-loss requires both "
                "--latent-intermediate-dim and --latent-dim",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.engram_table is None:
            print(
                "Error: --contrastive-loss requires --engram-table",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.contrastive_temperature <= 0 or not math.isfinite(args.contrastive_temperature):
            print(
                "Error: --contrastive-temperature must be finite and > 0",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.contrastive_k < 1:
            print("Error: --contrastive-k must be >= 1", file=sys.stderr)
            sys.exit(1)
        if args.contrastive_loss_weight < 0 or not math.isfinite(args.contrastive_loss_weight):
            print(
                "Error: --contrastive-loss-weight must be finite and >= 0",
                file=sys.stderr,
            )
            sys.exit(1)

    if args.latent_projection_init:
        if not args.contrastive_loss:
            print(
                "Error: --latent-projection-init requires --contrastive-loss "
                "(random projection without training is not useful)",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.contrastive_loss_weight <= 0:
            print(
                "Error: --latent-projection-init requires "
                "--contrastive-loss-weight > 0",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.latent_intermediate_dim is None or args.latent_dim is None:
            print(
                "Error: --latent-projection-init requires both "
                "--latent-intermediate-dim and --latent-dim",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.latent_projection is not None:
            print(
                "Error: --latent-projection-init is mutually exclusive with "
                "--latent-projection",
                file=sys.stderr,
            )
            sys.exit(1)

    if args.config == "tiny":
        config = HarmonyModelConfig.tiny()
    elif args.config == "tiny_ffn_expanded":
        config = HarmonyModelConfig.tiny_ffn_expanded()
    elif args.config == "tiny_engram_ann":
        config = HarmonyModelConfig.tiny_engram_ann()
    elif args.config == "tiny_engram_ann_routed":
        config = HarmonyModelConfig.tiny_engram_ann_routed()
    elif args.config == "tiny_engram_xattn":
        config = HarmonyModelConfig.tiny_engram_xattn()
    elif args.config == "tiny_engram_xattn_routed":
        config = HarmonyModelConfig.tiny_engram_xattn_routed()
    elif args.config == "tiny_engram_xattn_consol_online":
        config = HarmonyModelConfig.tiny_engram_xattn_consol_online()
    elif args.config == "tiny_engram_xattn_consol_phased":
        config = HarmonyModelConfig.tiny_engram_xattn_consol_phased()
    elif args.config == "tiny_engram_xattn_ctrl":
        config = HarmonyModelConfig.tiny_engram_xattn_ctrl()
    elif args.config == "tiny_engram_xattn_capgap":
        config = HarmonyModelConfig.tiny_engram_xattn_capgap()
    elif args.config == "tiny_engram_xattn_capgap_vcontrast":
        config = HarmonyModelConfig.tiny_engram_xattn_capgap_vcontrast()
    elif args.config == "tiny_engram_xattn_capgap_qdiv":
        config = HarmonyModelConfig.tiny_engram_xattn_capgap_qdiv()
    elif args.config == "tiny_engram_xattn_capgap_vcontrast_qdiv":
        config = HarmonyModelConfig.tiny_engram_xattn_capgap_vcontrast_qdiv()
    else:
        config = HarmonyModelConfig.target()
    seq_len = args.seq_len or (512 if args.config.startswith("tiny") else 2048)

    # θ-V-contrast (ZEB-130): --engram-vcontrast must exactly match the
    # preset's `engram_vcontrast_enabled`. Preset mutation via flag makes
    # CSV logs ambiguous (the recorded `--config=` no longer describes what
    # actually ran) — better to fail fast than silently change the loss
    # surface under the wrong preset name.
    if args.engram_vcontrast != config.engram_vcontrast_enabled:
        print(
            "Error: --engram-vcontrast must match the selected preset. "
            f"--config={args.config} has "
            f"engram_vcontrast_enabled={config.engram_vcontrast_enabled}; "
            f"--engram-vcontrast={'set' if args.engram_vcontrast else 'unset'}. "
            "Use --config=tiny_engram_xattn_capgap_vcontrast (or another "
            "V-contrast preset) for V-contrast runs; otherwise drop the flag.",
            file=sys.stderr,
        )
        sys.exit(1)
    # Override knobs only make sense when V-contrast is on. Rejecting them
    # otherwise catches typos like forgetting the `_vcontrast` suffix on the
    # preset name while still passing tuning flags.
    if not config.engram_vcontrast_enabled and (
        args.engram_vcontrast_lambda is not None
        or args.engram_vcontrast_warmup_steps is not None
        or args.engram_vcontrast_shuffle_seed is not None
    ):
        print(
            "Error: --engram-vcontrast-{lambda,warmup-steps,shuffle-seed} "
            "require a V-contrast preset + --engram-vcontrast.",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.engram_vcontrast_lambda is not None:
        config.engram_vcontrast_lambda = args.engram_vcontrast_lambda
    if args.engram_vcontrast_warmup_steps is not None:
        config.engram_vcontrast_warmup_steps = args.engram_vcontrast_warmup_steps
    if config.engram_vcontrast_enabled:
        # Re-run validation after CLI mutation, mirroring tiny_ffn_expanded /
        # tiny_engram_xattn_capgap pattern.
        config.__post_init__()

    # iota-Q-diversity (ZEB-130): --engram-qdiv must exactly match the preset's
    # engram_qdiv_enabled. Same reasoning as V-contrast: preset mutation via
    # flag makes CSV logs ambiguous.
    if args.engram_qdiv != config.engram_qdiv_enabled:
        print(
            "Error: --engram-qdiv must match the selected preset's "
            f"engram_qdiv_enabled (preset={config.engram_qdiv_enabled}, "
            f"flag={args.engram_qdiv}).",
            file=sys.stderr,
        )
        sys.exit(1)
    if not config.engram_qdiv_enabled and (
        args.engram_qdiv_lambda is not None
        or args.engram_qdiv_warmup_steps is not None
    ):
        print(
            "Error: --engram-qdiv-{lambda,warmup-steps} require a Q-div "
            "preset + --engram-qdiv.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Apply overrides after validation. Two reasons to validate at the CLI
    # layer before mutating config:
    #   (a) The `< 0` check in __post_init__ misses NaN/Inf because all
    #       comparisons with NaN return False — NaN leaking into the loss
    #       would silently corrupt every subsequent training step.
    #   (b) Fail-fast with a clear message naming the flag, so the user
    #       sees "--engram-qdiv-lambda must be ..." rather than the
    #       generic "engram_qdiv_lambda must be >= 0" from __post_init__.
    if args.engram_qdiv_lambda is not None:
        if (
            not math.isfinite(args.engram_qdiv_lambda)
            or args.engram_qdiv_lambda < 0
        ):
            print(
                "Error: --engram-qdiv-lambda must be finite and >= 0, got "
                f"{args.engram_qdiv_lambda!r}.",
                file=sys.stderr,
            )
            sys.exit(1)
        config.engram_qdiv_lambda = args.engram_qdiv_lambda
    if args.engram_qdiv_warmup_steps is not None:
        if args.engram_qdiv_warmup_steps < 0:
            print(
                "Error: --engram-qdiv-warmup-steps must be >= 0, got "
                f"{args.engram_qdiv_warmup_steps!r}.",
                file=sys.stderr,
            )
            sys.exit(1)
        config.engram_qdiv_warmup_steps = args.engram_qdiv_warmup_steps
    if config.engram_qdiv_enabled:
        config.__post_init__()

    # Model gamma arg validation (ZEB-117)
    if args.config in ("tiny_engram_ann", "tiny_engram_ann_routed"):
        if args.engram_ann_table is None:
            print(
                "Error: --config=tiny_engram_ann requires --engram-ann-table "
                "(path to corpus safetensors table)",
                file=sys.stderr,
            )
            sys.exit(1)
        if (
            not math.isfinite(args.engram_ann_entropy_weight)
            or not (0.0 <= args.engram_ann_entropy_weight <= 1.0)
        ):
            print(
                "Error: --engram-ann-entropy-weight must be finite and in [0.0, 1.0]",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.engram_ann_warmup_steps < 0:
            print("Error: --engram-ann-warmup-steps must be >= 0", file=sys.stderr)
            sys.exit(1)
        if not (0.0 <= args.engram_ann_gate_clamp_min <= 1.0):
            print(
                "Error: --engram-ann-gate-clamp-min must be in [0.0, 1.0]",
                file=sys.stderr,
            )
            sys.exit(1)

    # Model delta arg validation (ZEB-117)
    if args.config in ("tiny_engram_xattn", "tiny_engram_xattn_routed",
                        "tiny_engram_xattn_consol_online",
                        "tiny_engram_xattn_consol_phased",
                        "tiny_engram_xattn_ctrl"):
        if args.engram_xattn_table is None:
            print(
                "Error: --config=tiny_engram_xattn requires --engram-xattn-table "
                "(path to corpus safetensors table)",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.engram_xattn_k_retrieved <= 0:
            print(
                "Error: --engram-xattn-k-retrieved must be > 0",
                file=sys.stderr,
            )
            sys.exit(1)
        if not math.isfinite(args.engram_xattn_retrieval_bias_weight):
            print(
                "Error: --engram-xattn-retrieval-bias-weight must be finite",
                file=sys.stderr,
            )
            sys.exit(1)
        if (
            args.engram_xattn_num_heads is not None
            and args.engram_xattn_num_heads <= 0
        ):
            print(
                "Error: --engram-xattn-num-heads must be > 0 when specified",
                file=sys.stderr,
            )
            sys.exit(1)

    # ZEB-128 consolidation arg validation
    if args.consolidation_mode != "none":
        if not config.use_xattn_engram:
            print(
                "Error: --consolidation-mode requires a cross-attention engram "
                "config (e.g. tiny_engram_xattn_consol_online)",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.consolidation_lambda <= 0 or not math.isfinite(args.consolidation_lambda):
            print(
                "Error: --consolidation-lambda must be finite and > 0",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.consolidation_start_step < 0:
            print(
                "Error: --consolidation-start-step must be >= 0",
                file=sys.stderr,
            )
            sys.exit(1)
    if args.consolidation_anneal and args.consolidation_mode != "phased":
        print(
            "Error: --consolidation-anneal only applies to --consolidation-mode=phased",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.coconut:
        if args.think_token_id < 0:
            print("Error: --think-token-id must be >= 0", file=sys.stderr)
            sys.exit(1)
        if args.ct_max_steps < 0:
            print("Error: --ct-max-steps must be >= 0", file=sys.stderr)
            sys.exit(1)
        if args.think_token_id >= config.vocab_size:
            print(
                f"Error: --think-token-id ({args.think_token_id}) must be "
                f"< vocab_size ({config.vocab_size})",
                file=sys.stderr,
            )
            sys.exit(1)
        if seq_len + args.ct_max_steps > config.max_seq_len:
            print(
                f"Error: seq_len ({seq_len}) + ct_max_steps ({args.ct_max_steps}) "
                f"exceeds max_seq_len ({config.max_seq_len})",
                file=sys.stderr,
            )
            sys.exit(1)

    if args.uq_loss_weight < 0:
        print("Error: --uq-loss-weight must be >= 0", file=sys.stderr)
        sys.exit(1)

    if args.mtp_depth < 1:
        print("Error: --mtp-depth must be >= 1", file=sys.stderr)
        sys.exit(1)

    if args.mtp_loss_weight < 0:
        print("Error: --mtp-loss-weight must be >= 0", file=sys.stderr)
        sys.exit(1)

    if args.qat and (
        not math.isfinite(args.qat_start_pct)
        or args.qat_start_pct < 0
        or args.qat_start_pct >= 1.0
    ):
        print("Error: --qat-start-pct must be finite and in [0.0, 1.0)", file=sys.stderr)
        sys.exit(1)

    if args.qat_start_pct != 0.9 and not args.qat:
        print("Warning: --qat-start-pct has no effect without --qat", file=sys.stderr)

    device = detect_device(args.device)
    torch.manual_seed(args.seed)
    amp_dtype = torch.bfloat16 if args.dtype == "bfloat16" else None
    use_amp = amp_dtype is not None
    device_type = device.type
    print(f"Config: {args.config}, device: {device}, seq_len: {seq_len}, dtype: {args.dtype}")

    model = HarmonyModel(config).to(device)
    if args.gradient_checkpoint:
        model.set_gradient_checkpointing(True)
        print("Gradient checkpointing enabled")

    # ZEB-117 Model gamma: attach ANN injection module if configured.
    engram_ann_entropy_bounds: tuple[float, float] | None = None
    if config.use_ann_engram:
        from ct87.engram import EngramANNInjection

        ann_table = EngramANNInjection.load_corpus_table(args.engram_ann_table)
        ann_module = EngramANNInjection(
            config,
            ann_table,
            clamp_until_step=args.engram_ann_warmup_steps,
            clamp_min=args.engram_ann_gate_clamp_min,
            use_head_gates=config.use_head_gates,
        ).to(device)
        model.attach_engram_ann(ann_module)
        print(
            f"Model gamma ANN engram attached: {ann_module.total_entries:,} table "
            f"entries, engram_dim={ann_module.engram_dim}, "
            f"clamp_until_step={ann_module.clamp_until_step}, "
            f"clamp_min={ann_module.clamp_min}, "
            f"entropy_weight={args.engram_ann_entropy_weight}"
        )
        if args.engram_ann_entropy_bounds.lower() != "off":
            try:
                parts = args.engram_ann_entropy_bounds.split(",")
                if len(parts) != 2:
                    raise ValueError("expected exactly two comma-separated values")
                lo, hi = float(parts[0]), float(parts[1])
            except ValueError as e:
                print(
                    f"Error: --engram-ann-entropy-bounds must be 'low,high' "
                    f"with 0 < low < high < 1 (or 'off'): {e}",
                    file=sys.stderr,
                )
                sys.exit(1)
            if not (0.0 < lo < hi < 1.0):
                print(
                    "Error: --engram-ann-entropy-bounds must be 'low,high' "
                    "with 0 < low < high < 1",
                    file=sys.stderr,
                )
                sys.exit(1)
            engram_ann_entropy_bounds = (lo, hi)

    # ZEB-117 Model delta: attach cross-attention engram injection.
    if config.use_xattn_engram:
        from ct87.engram import EngramCrossAttention

        xattn_table = EngramCrossAttention.load_corpus_table(
            args.engram_xattn_table,
        )
        xattn_module = EngramCrossAttention(
            config,
            xattn_table,
            num_heads=args.engram_xattn_num_heads,
            k_retrieved=args.engram_xattn_k_retrieved,
            retrieval_bias_weight=args.engram_xattn_retrieval_bias_weight,
            use_head_gates=config.use_head_gates,
        ).to(device)
        model.attach_engram_xattn(xattn_module)
        print(
            f"Model delta xattn engram attached: "
            f"{xattn_module.total_entries:,} table entries, "
            f"engram_dim={xattn_module.engram_dim}, "
            f"num_heads={xattn_module.num_heads}, "
            f"k_retrieved={xattn_module.k_retrieved}, "
            f"bias_weight={xattn_module.retrieval_bias_weight}"
        )

    # Hoisted so the capture/restore sites in the training loop (which need
    # to include this generator's state in resumable checkpoints) can
    # reference it unconditionally — set to a real Generator inside the
    # engram-injection block below when V-contrast is enabled + seeded.
    capgap_shuffle_gen: torch.Generator | None = None

    # η-B multi-layer gated injection (ZEB-130).
    # Must happen BEFORE freeze_backbone_for_capgap() which raises if no
    # engram_injections are attached.
    if config.engram_inject_layers:
        from ct87.engram import (
            EngramCrossAttention,
            GatedEngramInjection,
        )

        if args.engram_xattn_table is not None:
            capgap_table = EngramCrossAttention.load_corpus_table(args.engram_xattn_table)
        elif args.synthetic:
            # Synthetic smoke tests can use a random placeholder table sized
            # (vocab_size, engram_dim) so the capgap path is exercised without
            # requiring a real oracle table artifact on disk.
            print(
                "[capgap] --synthetic with no --engram-xattn-table; using random "
                f"({config.vocab_size}, {config.engram_dim}) placeholder table."
            )
            capgap_table = torch.randn(config.vocab_size, config.engram_dim)
        else:
            # Real run with no table would silently attend over random memory
            # and produce meaningless retrieval — fail loudly instead.
            print(
                "Error: --config=tiny_engram_xattn_capgap requires "
                "--engram-xattn-table for non-synthetic runs (a random "
                "placeholder table would invalidate the experiment). Pass "
                "--synthetic for smoke tests.",
                file=sys.stderr,
            )
            sys.exit(2)

        # θ-V-contrast: when enabled, wrappers share a single aux-loss list.
        # The training step pulls aux losses out of that list, sums them, and
        # adds lambda * sum to LM loss.
        capgap_aux_sink: list[torch.Tensor] | None = (
            [] if config.engram_vcontrast_enabled else None
        )
        # Optional dedicated generator for the per-step value-shuffle. All
        # wrappers share a single generator so seeding once makes the sequence
        # of permutations across layers reproducible. CPU device avoids
        # device-placement sensitivity — the permutation tensor is a small
        # index tensor that gets moved to the table's device inside the
        # wrapper if needed. Hoisted above; assigned here when enabled.
        if (
            config.engram_vcontrast_enabled
            and args.engram_vcontrast_shuffle_seed is not None
        ):
            capgap_shuffle_gen = torch.Generator(device="cpu")
            capgap_shuffle_gen.manual_seed(int(args.engram_vcontrast_shuffle_seed))
            print(
                "[capgap] V-contrast shuffle seeded with "
                f"{args.engram_vcontrast_shuffle_seed} (reproducibility mode)"
            )
        # Unified: GatedEngramInjection accepts optional aux-loss sink kwargs.
        # vcontrast_sink is passed only when vcontrast is enabled (the sink is
        # None otherwise, which disables the V-contrast branch in forward()).
        # qdiv_sink is passed only when qdiv is enabled.
        capgap_injections: dict[int, GatedEngramInjection] = {}
        for layer_idx in config.engram_inject_layers:
            xattn_mod = EngramCrossAttention(
                config,
                capgap_table,
                num_heads=config.num_query_heads,
                k_retrieved=args.xattn_top_k,
            )
            capgap_injections[layer_idx] = GatedEngramInjection(
                xattn_mod,
                alpha_init=config.engram_gate_init,
                vcontrast_sink=(
                    capgap_aux_sink if config.engram_vcontrast_enabled else None
                ),
                qdiv_sink=(
                    model._qdiv_aux_losses if config.engram_qdiv_enabled else None
                ),
                shuffle_generator=capgap_shuffle_gen,
            )
        model.attach_gated_engram_injections(capgap_injections)
        if config.engram_vcontrast_enabled:
            # Both the model and the wrappers reference the same list — the
            # model clears it at the start of each training-mode forward and
            # the wrappers append per-layer scalars.
            model._contrastive_aux_losses = capgap_aux_sink
        # Ensure newly-attached submodules are on the correct device (model was
        # already moved to device before this block).
        model.engram_injections.to(device)

        # ZEB-134 Skip-to-Logit router attachment. Construct after
        # injections so the router can reference the model's lm_head
        # weight by identity (keeps the tied-embedding invariant).
        if args.engram_skip_to_logit:
            if (
                not math.isfinite(args.engram_skip_alpha_init)
                or args.engram_skip_alpha_init <= 0
            ):
                print(
                    "Error: --engram-skip-alpha-init must be finite and > 0, "
                    f"got {args.engram_skip_alpha_init!r}.",
                    file=sys.stderr,
                )
                sys.exit(1)
            from ct87.engram import SkipToLogitEngramRouter
            skip_router = SkipToLogitEngramRouter(
                hidden_dim=config.hidden_dim,
                lm_head_weight=model.lm_head.weight,
                alpha_init=args.engram_skip_alpha_init,
            ).to(device)
            model.attach_engram_skip_router(skip_router)

        aux_tags = []
        if config.engram_vcontrast_enabled:
            aux_tags.append("+vcontrast")
        if config.engram_qdiv_enabled:
            aux_tags.append("+qdiv")
        if args.engram_skip_to_logit:
            aux_tags.append("+skip-to-logit")
        aux_suffix = f"({','.join(aux_tags)})" if aux_tags else ""
        setup_parts = [
            f"[capgap] Attached GatedEngramInjection{aux_suffix}"
            f" at layers {list(config.engram_inject_layers)} with alpha_init="
            f"{config.engram_gate_init}"
        ]
        if config.engram_vcontrast_enabled:
            setup_parts.append(
                f"; vcontrast lambda={config.engram_vcontrast_lambda} "
                f"warmup={config.engram_vcontrast_warmup_steps}"
            )
        if config.engram_qdiv_enabled:
            setup_parts.append(
                f"; qdiv lambda={config.engram_qdiv_lambda} "
                f"warmup={config.engram_qdiv_warmup_steps}"
            )
        print("".join(setup_parts))

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # ZEB-128: Consolidation decoder
    consol_decoder = None
    if args.consolidation_mode != "none" and model.engram_xattn is not None:
        from ct87.engram import EngramConsolidationDecoder
        consol_decoder = EngramConsolidationDecoder(config.hidden_dim).to(device)
        print(
            f"Consolidation decoder attached: mode={args.consolidation_mode}, "
            f"lambda={args.consolidation_lambda}, "
            f"start_step={args.consolidation_start_step}"
        )

    # Resume from checkpoint after all modules are attached (including ANN)
    # so their weights are loaded. strict=False handles tied embeddings where
    # lm_head shares embed_tokens weights (missing key is expected).
    start_step = 0
    _pending_optimizer_state = None
    _pending_entropy_lambda = None
    if args.resume_from is not None and args.resume_from.endswith(".pt"):
        print(f"Loading resumable checkpoint from {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt or "step" not in ckpt:
            print(
                f"Error: {args.resume_from} is not a resumable checkpoint "
                "(expected keys: step, model_state_dict). Use a .safetensors "
                "file for weights-only resume.",
                file=sys.stderr,
            )
            sys.exit(1)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        start_step = ckpt["step"] + 1
        _pending_optimizer_state = ckpt.get("optimizer_state_dict")
        if "rng_state" in ckpt:
            restore_rng_state(
                ckpt["rng_state"], device,
                capgap_shuffle_gen=capgap_shuffle_gen,
            )
        if "dynamic_entropy_lambda" in ckpt:
            _pending_entropy_lambda = ckpt["dynamic_entropy_lambda"]
        last_val = ckpt.get("last_val_loss")
        if last_val is None:
            last_val = ckpt.get("best_val_loss")
        if last_val is not None:
            print(f"  Resuming from step {start_step}, last_val_loss={last_val:.4f}")
        else:
            print(f"  Resuming from step {start_step}")
        del ckpt
    elif args.resume_from is not None:
        from safetensors import safe_open
        with safe_open(args.resume_from, framework="pt") as f:
            ckpt_keys = set(f.keys())
        model_keys = set(model.state_dict().keys())
        missing = model_keys - ckpt_keys
        unexpected = ckpt_keys - model_keys
        # lm_head.weight is expected missing when tied to embed_tokens
        expected_missing = {"lm_head.weight"}
        real_missing = missing - expected_missing
        if real_missing:
            print(f"Warning: {len(real_missing)} missing keys in checkpoint: "
                  f"{sorted(real_missing)[:5]}{'...' if len(real_missing) > 5 else ''}",
                  file=sys.stderr)
        if unexpected:
            print(f"Warning: {len(unexpected)} unexpected keys in checkpoint: "
                  f"{sorted(unexpected)[:5]}{'...' if len(unexpected) > 5 else ''}",
                  file=sys.stderr)
        from safetensors.torch import load_model
        load_model(model, args.resume_from, strict=False)
        print(f"Resumed from checkpoint: {args.resume_from}")
    elif args.init_from is not None:
        print(f"Initializing model weights from {args.init_from}")
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
        if "model_state_dict" not in ckpt:
            print(
                f"Error: {args.init_from} is not a resumable checkpoint "
                "(no model_state_dict).",
                file=sys.stderr,
            )
            sys.exit(2)
        # Architecture-compatibility check. strict=False silently skips
        # shape-mismatched keys; without this guard, an --init-from from an
        # incompatible architecture (e.g., different hidden_dim) would load
        # zero backbone params and the run would proceed with a randomly-
        # initialized model masquerading as "loaded from checkpoint".
        src_config = ckpt.get("config")
        if src_config is None and not args.allow_partial_init:
            print(
                f"Error: --init-from source {args.init_from} has no config "
                "metadata to compare against. Without a compat check, "
                "strict=False could silently load zero backbone params on an "
                "architecture mismatch. Pass --allow-partial-init to override "
                "(e.g., for legacy checkpoints).",
                file=sys.stderr,
            )
            sys.exit(2)
        if src_config is not None:
            _shape_critical_fields = (
                "num_layers", "hidden_dim", "num_query_heads", "num_kv_heads",
                "head_dim", "ffn_dim", "vocab_size", "engram_dim",
            )
            mismatches = [
                (f, getattr(src_config, f, None), getattr(config, f, None))
                for f in _shape_critical_fields
                if getattr(src_config, f, None) != getattr(config, f, None)
            ]
            if mismatches:
                print(
                    "Error: --init-from source architecture is incompatible "
                    "with the current config (strict=False would silently skip "
                    "shape-mismatched params). Mismatches:",
                    file=sys.stderr,
                )
                for name, src_val, cur_val in mismatches:
                    print(
                        f"  {name}: source={src_val!r} vs current={cur_val!r}",
                        file=sys.stderr,
                    )
                sys.exit(2)
        missing, unexpected = model.load_state_dict(
            ckpt["model_state_dict"], strict=False
        )
        if unexpected:
            preview = ", ".join(unexpected[:5])
            if len(unexpected) > 5:
                preview += f", ... (+{len(unexpected) - 5} more)"
            print(
                f"Warning: --init-from has {len(unexpected)} unexpected "
                f"keys (will be ignored): {preview}",
                file=sys.stderr,
            )
        if missing:
            non_engram_missing = [
                k for k in missing if not k.startswith("engram_injections.")
            ]
            if non_engram_missing:
                preview = ", ".join(non_engram_missing[:5])
                if len(non_engram_missing) > 5:
                    preview += f", ... (+{len(non_engram_missing) - 5} more)"
                if args.allow_partial_init:
                    print(
                        f"Warning: --init-from missing "
                        f"{len(non_engram_missing)} non-engram keys "
                        f"(--allow-partial-init override): {preview}",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"Error: --init-from missing "
                        f"{len(non_engram_missing)} non-engram backbone keys; "
                        "would run with randomly-initialized parameters. Pass "
                        "--allow-partial-init to override. Missing: "
                        f"{preview}",
                        file=sys.stderr,
                    )
                    sys.exit(2)
            else:
                print(
                    f"--init-from loaded cleanly (missing only "
                    f"{len(missing)} engram_injections keys as expected)."
                )
        # Explicitly DO NOT touch optimizer, step, or RNG: this is a fresh run.
        del ckpt

    # Load Engram table if provided
    engram_table = None
    if args.engram_table is not None:
        from ct87.engram import EngramTable
        seeds = [int(s) for s in args.engram_seeds.split(",")]
        engram_table = EngramTable.from_safetensors(
            args.engram_table, hash_seeds=seeds, device=str(device),
        )
        print(
            f"Engram table loaded: {engram_table.total_entries:,} entries, "
            f"dim={engram_table.engram_dim}, heads={engram_table.num_heads}"
        )

    # ZEB-139: load the teacher-logits sidecar when KL+CE is enabled.
    # Kept on CPU as bf16 (~640 MB at 10K rows × 32K vocab); per-batch
    # `oracle_teacher_logits[row_indices]` returns a small [B, T, vocab]
    # slice that the loss block transfers to GPU + casts to fp32.
    # `oracle_teacher_total_entries` is recorded separately so the
    # canonical-row-index helper can take the modulo against the
    # sidecar's actual row count (matches what generate_oracle_table.py
    # used when producing the file — must agree with --engram-table's
    # total_entries by construction, but enforce here too).
    oracle_teacher_logits: torch.Tensor | None = None
    oracle_teacher_total_entries: int = 0
    if args.kl_lambda > 0:
        from safetensors.torch import load_file as _load_safetensors
        sidecar_tensors = _load_safetensors(args.oracle_teacher_logits)
        if "teacher_logits.weight" not in sidecar_tensors:
            print(
                f"Error: --oracle-teacher-logits "
                f"({args.oracle_teacher_logits!r}) does not contain a "
                f"'teacher_logits.weight' tensor; available keys: "
                f"{list(sidecar_tensors.keys())}.",
                file=sys.stderr,
            )
            sys.exit(1)
        oracle_teacher_logits = sidecar_tensors["teacher_logits.weight"]
        if oracle_teacher_logits.dim() != 2:
            print(
                f"Error: teacher_logits.weight must be 2-D "
                f"[total_entries, vocab]; got shape "
                f"{tuple(oracle_teacher_logits.shape)}.",
                file=sys.stderr,
            )
            sys.exit(1)
        if oracle_teacher_logits.shape[1] != config.vocab_size:
            print(
                f"Error: teacher_logits.weight vocab dim "
                f"({oracle_teacher_logits.shape[1]}) does not match the "
                f"student's config.vocab_size ({config.vocab_size}). The "
                f"sidecar was produced for a different tokenizer.",
                file=sys.stderr,
            )
            sys.exit(1)
        if engram_table is not None and oracle_teacher_logits.shape[0] != engram_table.total_entries:
            print(
                f"Error: teacher_logits.weight has "
                f"{oracle_teacher_logits.shape[0]} rows but the engram "
                f"table has {engram_table.total_entries} — both must use "
                f"the same xxhash row count (otherwise per-position "
                f"hashes don't agree). Re-extract one or the other to "
                f"match.",
                file=sys.stderr,
            )
            sys.exit(1)
        oracle_teacher_total_entries = int(oracle_teacher_logits.shape[0])
        print(
            f"Teacher-logits sidecar loaded: "
            f"{oracle_teacher_total_entries:,} rows × "
            f"{oracle_teacher_logits.shape[1]:,} vocab "
            f"(dtype={oracle_teacher_logits.dtype}, "
            f"~{oracle_teacher_logits.element_size() * oracle_teacher_logits.numel() / 1024 ** 2:.0f} MB CPU-resident)"
        )

    # Latent projection setup - three modes:
    # 1. --latent-projection + --contrastive-loss: load from checkpoint, trainable
    # 2. --latent-projection (no contrastive): load from checkpoint, frozen
    # 3. --latent-projection-init: randomly initialize, trainable
    latent_projection = None
    if args.latent_projection is not None:
        from ct87.latent_projection import LatentProjection

        latent_projection = LatentProjection.from_checkpoint(
            args.latent_projection,
            hidden_dim=config.hidden_dim,
            intermediate_dim=args.latent_intermediate_dim,
            latent_dim=args.latent_dim,
            device=device,
        )
        proj_params = sum(p.numel() for p in latent_projection.parameters())
        if args.contrastive_loss:
            print(
                f"Latent projection loaded (trainable): "
                f"{config.hidden_dim}->{args.latent_intermediate_dim}->{args.latent_dim}, "
                f"{proj_params:,} params"
            )
        else:
            latent_projection.requires_grad_(False)
            print(
                f"Latent projection loaded (frozen): "
                f"{config.hidden_dim}->{args.latent_intermediate_dim}->{args.latent_dim}, "
                f"{proj_params:,} params"
            )
    elif args.latent_projection_init:
        from ct87.latent_projection import LatentProjection

        latent_projection = LatentProjection(
            hidden_dim=config.hidden_dim,
            intermediate_dim=args.latent_intermediate_dim,
            latent_dim=args.latent_dim,
        ).to(device)
        proj_params = sum(p.numel() for p in latent_projection.parameters())
        print(
            f"Latent projection initialized (trainable): "
            f"{config.hidden_dim}->{args.latent_intermediate_dim}->{args.latent_dim}, "
            f"{proj_params:,} params"
        )

    # COCONUT continuous thought setup
    thought_norm = None
    coconut_curriculum = None
    if args.coconut:
        from ct87.coconut import ThoughtNorm, CurriculumSchedule, coconut_forward, coconut_loss

        config.think_token_id = args.think_token_id
        config.ct_max_steps = args.ct_max_steps
        thought_norm = ThoughtNorm(config.hidden_dim, eps=config.rms_norm_eps).to(device)
        coconut_curriculum = CurriculumSchedule(args.ct_max_steps, args.steps)
        print(
            f"COCONUT enabled: think_token_id={args.think_token_id}, "
            f"ct_max_steps={args.ct_max_steps}"
        )

    # UQ head setup
    uq_head = None
    uq_feature_config = None
    if args.uq_head:
        from ct87.uq import (
            UqHead, UqFeatureConfig, LayerNormCollector,
            extract_uq_features, compute_pseudo_labels, compute_uq_loss,
        )
        uq_head = UqHead().to(device)
        uq_feature_config = UqFeatureConfig.for_num_layers(config.num_layers)
        uq_param_count = sum(p.numel() for p in uq_head.parameters())
        print(f"UQ head enabled: {uq_param_count} params, loss_weight={args.uq_loss_weight}")

    # MTP head setup
    mtp_head = None
    if args.mtp_head:
        from ct87.mtp import MtpHead
        mtp_head = MtpHead(config, depth=args.mtp_depth).to(device)
        mtp_param_count = sum(p.numel() for p in mtp_head.parameters())
        print(f"MTP head enabled: depth={args.mtp_depth}, {mtp_param_count:,} params, loss_weight={args.mtp_loss_weight}")

    # η-B capacity-gap freezing: MUST come before optimizer construction so
    # the Muon partitioner only sees trainable params (frozen backbone gets
    # zero optimizer-state overhead).
    if args.freeze_backbone:
        freeze_backbone_for_capgap(model)

    muon_params, adam_params = partition_params(model)
    if thought_norm is not None:
        adam_params.extend(thought_norm.parameters())
    if uq_head is not None:
        adam_params.extend(uq_head.parameters())
    if mtp_head is not None:
        adam_params.extend(mtp_head.parameters())
    if latent_projection is not None and args.contrastive_loss:
        adam_params.extend(latent_projection.parameters())
    optimizer = Muon(muon_params, adam_params, lr=args.lr, adam_lr=args.lr)
    # Add the consolidation decoder's param group BEFORE loading optimizer
    # state. Otherwise resuming a consolidation run hits a param-group-count
    # mismatch in load_state_dict and falls back to a fresh optimizer.
    if consol_decoder is not None:
        optimizer.add_param_group({
            "params": list(consol_decoder.parameters()),
            "lr": args.lr,
        })
    if _pending_optimizer_state is not None:
        try:
            optimizer.load_state_dict(_pending_optimizer_state)
            print(f"  Optimizer state restored for step {start_step}")
        except (ValueError, KeyError) as e:
            print(
                f"  Warning: could not restore optimizer state ({e}); "
                "continuing with fresh optimizer (weights still loaded)",
                file=sys.stderr,
            )
        del _pending_optimizer_state
    schedule = WSDSchedule(warmup_steps=args.warmup, total_steps=args.steps)

    # Skip building the training dataloader for eval-only runs — they never
    # enter the training loop, and --zero-injection-eval is allowed to omit
    # --data / --synthetic.
    dataloader = None
    if not args.zero_injection_eval:
        if args.synthetic:
            dataloader = make_synthetic_dataloader(config.vocab_size, seq_len, args.batch_size, args.seed)
        else:
            dataloader = make_hf_dataloader(args.data, seq_len, args.batch_size, args.seed)

    val_loader = None
    if args.val_data is not None:
        val_loader = make_hf_dataloader(args.val_data, seq_len, args.batch_size, args.seed + 1)
        print(f"Validation data loaded from {args.val_data}")

    # ZEB-128: Zero-injection evaluation mode
    if args.zero_injection_eval:
        if args.resume_from is None:
            print("Error: --zero-injection-eval requires --resume-from", file=sys.stderr)
            sys.exit(1)
        if not args.resume_from.endswith(".pt"):
            print(
                "Error: --zero-injection-eval requires a resumable .pt "
                "checkpoint (safetensors checkpoints do not carry the step "
                "metadata needed for this mode)",
                file=sys.stderr,
            )
            sys.exit(1)
        if val_loader is None and args.synthetic:
            # --synthetic smoke-test compatibility: build a synthetic val loader
            # so --zero-injection-eval works without a real corpus (ZEB-130).
            val_loader = make_synthetic_dataloader(
                config.vocab_size, seq_len, args.batch_size, args.seed + 1
            )
        if val_loader is None:
            print("Error: --zero-injection-eval requires --val-data", file=sys.stderr)
            sys.exit(1)
        ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        required_keys = {"model_state_dict", "step"}
        if not required_keys.issubset(ckpt.keys()):
            print(
                f"Error: checkpoint missing required keys: "
                f"{required_keys - ckpt.keys()}",
                file=sys.stderr,
            )
            sys.exit(1)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        step = ckpt["step"]
        # Materialize a fixed set of validation batches so pre/post
        # measurements compare the same data — otherwise delta_removal is
        # dominated by batch sampling noise rather than the injection toggle.
        val_batches = [next(val_loader) for _ in range(10)]

        def _replay_batches():
            for b in val_batches:
                yield b

        val_pre = compute_validation_loss(
            model, _replay_batches(), config.vocab_size, device,
            num_batches=len(val_batches),
            amp_dtype=amp_dtype, engram_table=engram_table,
            latent_projection=latent_projection,
        )
        model.engram_inject_mult = 0.0
        val_post = compute_validation_loss(
            model, _replay_batches(), config.vocab_size, device,
            num_batches=len(val_batches),
            amp_dtype=amp_dtype, engram_table=engram_table,
            latent_projection=latent_projection,
        )
        delta_removal = val_post - val_pre
        print(f"Step: {step}")
        print(f"Val loss (with injection):    {val_pre:.6f}")
        print(f"Val loss (without injection): {val_post:.6f}")
        print(f"Delta (removal cost):         {delta_removal:+.6f}")
        return

    csv_file = None
    csv_writer = None
    if args.log_file:
        # θ-V-contrast (ZEB-130) CSV columns are only emitted when the
        # feature is enabled — non-vcontrast runs keep their existing CSV
        # shape, and per-layer `vcontrast_aux_L{i}` columns are derived from
        # `engram_inject_layers` so they reflect the active config rather
        # than a fixed (2, 5) preset.
        vcontrast_cols: list[str] = []
        if config.engram_vcontrast_enabled:
            # Sorted to match the forward-pass population order of
            # `_contrastive_aux_losses` (ascending layer index). Keeps CSV
            # columns and the per-layer accumulator aligned regardless of
            # how the preset declared engram_inject_layers.
            vcontrast_cols = [
                "vcontrast_aux_loss",
                *(f"vcontrast_aux_L{i}" for i in sorted(config.engram_inject_layers or ())),
                "vcontrast_lambda",
            ]
        # iota-Q-diversity (ZEB-130): conditional CSV columns, mirroring the
        # V-contrast pattern above. Only emitted when qdiv is enabled.
        qdiv_cols: list[str] = []
        if config.engram_qdiv_enabled:
            qdiv_cols = [
                "qdiv_aux_loss",
                *(
                    f"qdiv_aux_L{i}"
                    for i in sorted(config.engram_inject_layers or ())
                ),
                "qdiv_lambda",
            ]
        expected_header = [
            "step", "loss", "uq_loss", "mtp_loss", "cl_loss",
            "ann_ent_loss", "ann_gate_mean", "ann_lambda_ent",
            "val_loss", "lr", "grad_norm", "num_thoughts", "dt_ms",
            "hg_0", "hg_1", "hg_2", "hg_3", "hg_4", "hg_5", "hg_6", "hg_7",
            "hg_std", "hg_min", "hg_max",
            "mse_loss", "consol_phase", "inject_mult",
            *vcontrast_cols,
            *qdiv_cols,
            # ZEB-139 KL+CE: appended at the end so existing CSVs from
            # pre-ZEB-139 runs are still a "trailing prefix" of this
            # header — caught by the generic migration below and padded
            # with empty trailing cells without losing data.
            "kl_loss",
        ]
        legacy_header_no_cl = ["step", "loss", "uq_loss", "mtp_loss", "val_loss", "lr", "grad_norm", "num_thoughts", "dt_ms"]
        legacy_header_no_ann = ["step", "loss", "uq_loss", "mtp_loss", "cl_loss", "val_loss", "lr", "grad_norm", "num_thoughts", "dt_ms"]
        legacy_header_no_hg = [
            "step", "loss", "uq_loss", "mtp_loss", "cl_loss",
            "ann_ent_loss", "ann_gate_mean", "ann_lambda_ent",
            "val_loss", "lr", "grad_norm", "num_thoughts", "dt_ms",
        ]
        if os.path.exists(args.log_file) and os.path.getsize(args.log_file) > 0:
            with open(args.log_file, newline="") as existing:
                header = next(csv.reader(existing), [])
            migrations_needed = []
            if header == legacy_header_no_cl:
                migrations_needed.append("cl_loss")
            if header in (legacy_header_no_cl, legacy_header_no_ann):
                migrations_needed.append("ann")
            if header == legacy_header_no_hg:
                migrations_needed.append("hg")
            # Generalized trailing-column migration: if the existing header is
            # a prefix of expected_header, pad rows with empty trailing columns.
            # Covers pre-consolidation (hg columns present, no mse/consol) and
            # any future appended-column migrations without needing a named
            # legacy constant per generation.
            is_trailing_prefix = (
                header
                and header != expected_header
                and len(header) < len(expected_header)
                and header == expected_header[: len(header)]
            )
            if migrations_needed or is_trailing_prefix:
                with open(args.log_file, newline="") as f:
                    rows = list(csv.reader(f))
                rows[0] = expected_header
                for i in range(1, len(rows)):
                    if "cl_loss" in migrations_needed:
                        # Insert empty cl_loss column (index 4)
                        rows[i].insert(4, "")
                    if "ann" in migrations_needed:
                        # Insert three empty ann columns after cl_loss
                        for offset in (5, 5, 5):
                            rows[i].insert(offset, "")
                    if "hg" in migrations_needed:
                        rows[i].extend([""] * 11)
                    # Pad any remaining column deficit (covers multi-generation
                    # gaps and the generalized trailing-prefix case).
                    deficit = len(expected_header) - len(rows[i])
                    if deficit > 0:
                        rows[i].extend([""] * deficit)
                with open(args.log_file, "w", newline="") as f:
                    csv.writer(f).writerows(rows)
            elif header and header != expected_header:
                print(
                    f"Error: incompatible CSV header in {args.log_file}: {header}",
                    file=sys.stderr,
                )
                sys.exit(1)
        csv_file = open(args.log_file, "a", newline="")
        csv_writer = csv.writer(csv_file)
        if os.path.getsize(args.log_file) == 0:
            csv_writer.writerow(expected_header)

    qat_enabled = False
    if args.qat and args.steps > 0:
        qat_start_step = min(math.ceil(args.qat_start_pct * args.steps), args.steps - 1)
    else:
        qat_start_step = args.steps + 1

    # Dynamic lambda_ent for Model gamma: adjusted up/down to keep mean gate prob
    # inside engram_ann_entropy_bounds. Scales multiplicatively after each
    # step based on whether the mean drifted out of band.
    dynamic_entropy_lambda = float(args.engram_ann_entropy_weight)
    if start_step > 0 and _pending_entropy_lambda is not None:
        dynamic_entropy_lambda = _pending_entropy_lambda
        print(f"  dynamic_entropy_lambda restored: {dynamic_entropy_lambda:.6f}")
    ENTROPY_LAMBDA_SCALE_UP = 1.10
    ENTROPY_LAMBDA_SCALE_DOWN = 0.95
    ENTROPY_LAMBDA_MIN = float(args.engram_ann_entropy_weight)
    ENTROPY_LAMBDA_MAX = 1.0
    num_thoughts = 0

    try:
        for step in range(start_step, args.steps):
            # Model gamma: propagate current step to the ANN engram so it
            # knows whether to apply the hard gate clamp.
            if model.engram_ann is not None:
                model.engram_ann.set_step(step)
            # Activate QAT at the configured step (base model only -
            # UQ head, ThoughtNorm, MTP head are separate modules)
            if args.qat and not qat_enabled and step >= qat_start_step:
                from ct87.qat import enable_qat
                enable_qat(model)
                qat_enabled = True
                print(f"  -> QAT enabled at step {step} (q8_0 fake quantization)")

            step_start = time.time()
            lr_mult = schedule.get_lr_multiplier(step)
            set_lr(optimizer, lr_mult)
            optimizer.zero_grad()

            accum_loss = 0.0
            accum_uq_loss = 0.0
            accum_mtp_loss = 0.0
            accum_cl_loss = 0.0
            accum_ann_ent_loss = 0.0
            accum_ann_gate_mean = 0.0
            accum_mse_loss = 0.0
            # ZEB-139 KL+CE: accumulator for the KL term, only populated
            # when --kl-lambda > 0. Reset per outer step to match the
            # other aux-loss accumulators' grad_accum_steps averaging.
            accum_kl_loss = 0.0
            # θ-V-contrast (ZEB-130): aux-loss accumulators. Sum is captured
            # as a scalar (no grad), per-layer values are retained for CSV.
            accum_vcontrast_aux = 0.0
            accum_vcontrast_per_layer: dict[str, float] = {}
            # iota-Q-diversity (ZEB-130): parallel aux-loss accumulators.
            accum_qdiv_aux = 0.0
            accum_qdiv_per_layer: dict[str, float] = {}
            num_thoughts = coconut_curriculum.num_thoughts(step) if coconut_curriculum is not None else 0
            need_hidden = mtp_head is not None
            use_coconut = args.coconut and num_thoughts > 0

            # ZEB-128: Update injection multiplier for phased annealing.
            # Denominator uses (args.steps - 1) so inject_mult reaches exactly
            # 0.0 on the final executed step (range(start, args.steps) stops
            # at args.steps - 1).
            inject_mult = 1.0
            if args.consolidation_anneal and step >= args.consolidation_start_step:
                anneal_steps = max(
                    1, (args.steps - 1) - args.consolidation_start_step
                )
                progress = (step - args.consolidation_start_step) / anneal_steps
                inject_mult = max(0.0, 1.0 - progress)
            if model.engram_xattn is not None:
                model.engram_inject_mult = inject_mult

            for micro_step in range(args.grad_accum_steps):
                batch = next(dataloader).to(device)
                input_ids = batch[:, :-1]
                targets = batch[:, 1:]

                # Compute Engram embeddings if table is loaded
                engram_emb = None
                cl_projected = None
                cl_ngram_avgs = None
                if engram_table is not None and latent_projection is not None and args.contrastive_loss:
                    # Co-training: compute keys with grad for contrastive loss,
                    # use them for engram retrieval
                    from ct87.latent_projection import compute_ngram_averages

                    # Embeddings for key generation don't need grad -
                    # everything is detached before projection anyway.
                    # CE gradients flow through the model's own forward pass.
                    with torch.no_grad():
                        emb = model.embed_tokens(input_ids)
                    seq_len_actual = input_ids.shape[1]

                    all_keys: list[bytes] = []
                    all_positions: list[int] = []
                    all_ngram_avgs: list[torch.Tensor] = []
                    all_projected: list[torch.Tensor] = []

                    for b_idx in range(input_ids.shape[0]):
                        ngram_avgs, positions = compute_ngram_averages(
                            emb[b_idx : b_idx + 1], seq_len_actual,
                        )
                        projected = latent_projection(ngram_avgs.detach())
                        binary_keys = latent_projection.to_binary_keys(projected)

                        all_ngram_avgs.append(ngram_avgs.detach())
                        all_projected.append(projected)
                        all_keys.extend(binary_keys)
                        all_positions.extend(positions)

                    cl_ngram_avgs = all_ngram_avgs
                    cl_projected = all_projected

                    # Build engram embeddings from pre-computed keys
                    engram_parts = []
                    offset = 0
                    for b_idx in range(input_ids.shape[0]):
                        n_keys = len(all_ngram_avgs[b_idx])
                        part_keys = all_keys[offset : offset + n_keys]
                        part_pos = all_positions[offset : offset + n_keys]
                        offset += n_keys
                        part = engram_table.lookup_from_keys(
                            part_keys, part_pos,
                            batch_size=1, seq_len=seq_len_actual,
                        )
                        engram_parts.append(part)
                    engram_emb = torch.cat(engram_parts, dim=0)
                elif engram_table is not None and latent_projection is not None:
                    with torch.no_grad():
                        emb = model.embed_tokens(input_ids)
                        engram_emb = engram_table.lookup_batch_projected(
                            input_ids, emb, latent_projection,
                        )
                elif engram_table is not None:
                    with torch.no_grad():
                        engram_emb = engram_table.lookup_batch(input_ids)

                with torch.autocast(device_type, dtype=amp_dtype, enabled=use_amp):
                    # Forward pass (optionally with LayerNormCollector for UQ)
                    think_mask = None
                    hidden = None
                    if uq_head is not None:
                        with LayerNormCollector(model, uq_feature_config.norm_layers) as collector:
                            if use_coconut:
                                coconut_result = coconut_forward(
                                    model, thought_norm, input_ids,
                                    args.think_token_id, num_thoughts,
                                    engram_embeddings=engram_emb,
                                    return_hidden_states=need_hidden,
                                )
                                if need_hidden:
                                    logits, think_mask, hidden = coconut_result
                                else:
                                    logits, think_mask = coconut_result
                            elif need_hidden:
                                logits, hidden = model(
                                    input_ids, engram_embeddings=engram_emb,
                                    return_hidden_states=True,
                                )
                            else:
                                logits = model(input_ids, engram_embeddings=engram_emb)
                        uq_norms = collector.get_norms()
                    elif use_coconut:
                        coconut_result = coconut_forward(
                            model, thought_norm, input_ids,
                            args.think_token_id, num_thoughts,
                            engram_embeddings=engram_emb,
                            return_hidden_states=need_hidden,
                        )
                        if need_hidden:
                            logits, think_mask, hidden = coconut_result
                        else:
                            logits, think_mask = coconut_result
                    elif need_hidden:
                        logits, hidden = model(
                            input_ids, engram_embeddings=engram_emb,
                            return_hidden_states=True,
                        )
                    else:
                        logits = model(input_ids, engram_embeddings=engram_emb)

                    # Compute main LM loss
                    if use_coconut:
                        think_targets = torch.full(
                            (targets.shape[0], num_thoughts), -100,
                            dtype=targets.dtype, device=targets.device,
                        )
                        aug_targets = torch.cat([think_targets, targets], dim=1)
                        loss = coconut_loss(logits, aug_targets, think_mask)
                    else:
                        loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), targets.reshape(-1))

                    # ZEB-139 KL+CE: mix KL(P_router || P_teacher) into
                    # the loss when --kl-lambda > 0. Per-token-normalized
                    # to match CE's reduction="mean" — see spec §4.3 and
                    # PR #255 round-2 review for the rationale (batchmean
                    # would scale KL with seq_len, confounding lambda).
                    if args.kl_lambda > 0:
                        skip_logits = model._last_skip_logits
                        if skip_logits is None:
                            raise RuntimeError(
                                "--kl-lambda > 0 but model._last_skip_logits "
                                "is None after forward — Skip-to-Logit router "
                                "didn't fire. This usually means engram_inject_mult "
                                "is 0 or no engram_injections were attached. "
                                "Check that --engram-skip-to-logit is set with "
                                "a multi-layer gated-injection preset (e.g. "
                                "tiny_engram_xattn_capgap*)."
                            )
                        # Single canonical hash per position (Option C in
                        # the ZEB-139 design): trigram with the first
                        # engram seed. Positions p<2 (no trigram) get -1
                        # and are masked out of the KL average.
                        from ct87.engram import compute_canonical_trigram_row_indices
                        row_indices_cpu = compute_canonical_trigram_row_indices(
                            input_ids,
                            total_entries=oracle_teacher_total_entries,
                            canonical_seed=seeds[0],
                        )  # [B, T] cpu long, -1 for invalid
                        kl_mask = (row_indices_cpu >= 0)  # [B, T] cpu bool
                        # Clamp negatives to 0 for the lookup (masked
                        # contributions get zeroed out by `kl_mask`
                        # below; the index value at those positions is
                        # irrelevant).
                        valid_indices_cpu = row_indices_cpu.clamp(min=0)
                        # CPU gather → [B, T, vocab] bf16, then move to
                        # GPU + cast to fp32 for stable softmax math.
                        teacher_logits_sample = oracle_teacher_logits[
                            valid_indices_cpu
                        ].to(device=skip_logits.device, dtype=torch.float32)
                        log_p_router = F.log_softmax(
                            skip_logits.float(), dim=-1,
                        )
                        p_teacher = F.softmax(teacher_logits_sample, dim=-1)
                        kl_per_token = F.kl_div(
                            log_p_router, p_teacher, reduction="none",
                        ).sum(-1)  # [B, T]
                        kl_mask_dev = kl_mask.to(skip_logits.device)
                        # Masked mean: divide by the number of positions
                        # that have a valid trigram, not by total tokens.
                        # `.clamp(min=1)` guards a degenerate batch with
                        # seq_len < 3 (would otherwise be div-by-zero).
                        denom = kl_mask_dev.sum().clamp(min=1).to(
                            kl_per_token.dtype,
                        )
                        kl_loss_val = (kl_per_token * kl_mask_dev).sum() / denom
                        # Convex blend: when lambda=0 KL contributes zero
                        # and CE is unchanged. When lambda=1 the loss is
                        # pure KL (no CE pressure on the base LM head).
                        loss = (
                            (1.0 - args.kl_lambda) * loss
                            + args.kl_lambda * kl_loss_val
                        )
                        accum_kl_loss += kl_loss_val.item()

                    # UQ auxiliary loss
                    if uq_head is not None:
                        with torch.no_grad():
                            uq_features, uq_probs = extract_uq_features(
                                uq_norms, logits.detach(), uq_feature_config,
                            )
                            # For COCONUT: use 0 as target at think positions
                            if use_coconut:
                                uq_targets = aug_targets.clone()
                                uq_targets[uq_targets == -100] = 0
                            else:
                                uq_targets = targets
                            uq_class_labels, uq_conf_targets = compute_pseudo_labels(
                                logits.detach(), uq_targets, uq_features,
                                top_k=uq_feature_config.top_k,
                                probs=uq_probs,
                            )

                        uq_class_logits, uq_confidence = uq_head(uq_features)
                        uq_loss_val = compute_uq_loss(
                            uq_class_logits, uq_confidence,
                            uq_class_labels, uq_conf_targets,
                            think_mask=think_mask,
                        )
                        loss = loss + args.uq_loss_weight * uq_loss_val
                        accum_uq_loss += uq_loss_val.item()

                    # MTP auxiliary loss
                    if mtp_head is not None and hidden is not None:
                        if use_coconut:
                            # Skip think-token prefix: MTP only on real tokens
                            mtp_hidden = hidden[:, num_thoughts:, :]
                            mtp_targets = targets
                        else:
                            mtp_hidden = hidden
                            mtp_targets = targets
                        mtp_loss_val = mtp_head(
                            mtp_hidden, mtp_targets,
                            model.embed_tokens, model.lm_head,
                        )
                        loss = loss + args.mtp_loss_weight * mtp_loss_val
                        accum_mtp_loss += mtp_loss_val.item()

                    # Contrastive auxiliary loss
                    if args.contrastive_loss and cl_projected:
                        from ct87.latent_projection import contrastive_loss

                        cl_total = sum(
                            contrastive_loss(
                                avg, proj,
                                temperature=args.contrastive_temperature,
                                k=args.contrastive_k,
                            )
                            for avg, proj in zip(cl_ngram_avgs, cl_projected)
                        ) / len(cl_projected)
                        loss = loss + args.contrastive_loss_weight * cl_total
                        accum_cl_loss += cl_total.item()

                    # Model gamma: gate-entropy regularization (ZEB-117).
                    # Reads the gate stored by HarmonyModel during the
                    # forward; penalizes low-entropy (collapsed) gate
                    # distributions to preserve gradient flow through
                    # the memory path. See research report s3.3.1.
                    if model.engram_ann is not None and model._last_ann_gate is not None:
                        from ct87.engram import compute_gate_entropy_loss
                        gate = model._last_ann_gate
                        # Clear the side-channel immediately so the model
                        # doesn't hold a reference to the grad-history
                        # tensor longer than needed. The next forward
                        # would overwrite it anyway; clearing here makes
                        # the lifetime explicit.
                        model._last_ann_gate = None
                        ent_loss = compute_gate_entropy_loss(gate)
                        loss = loss + dynamic_entropy_lambda * ent_loss
                        accum_ann_ent_loss += ent_loss.item()
                        accum_ann_gate_mean += gate.mean().item()

                    # ZEB-128: Consolidation MSE loss
                    if (
                        consol_decoder is not None
                        and model._last_xattn_output is not None
                        and model._last_pre_injection_hidden is not None
                        and step >= args.consolidation_start_step
                    ):
                        consol_target = model._last_xattn_output.detach()
                        consol_input = model._last_pre_injection_hidden
                        consol_pred = consol_decoder(consol_input)
                        mse_loss = F.mse_loss(consol_pred, consol_target)
                        loss = loss + args.consolidation_lambda * mse_loss
                        accum_mse_loss += mse_loss.item()

                    # θ-V-contrast (ZEB-130): aggregate per-layer V-contrastive
                    # aux losses appended by GatedEngramInjection forwards.
                    # The sink is owned by the model and cleared at the start
                    # of every training-mode forward.
                    if (
                        config.engram_vcontrast_enabled
                        and model._contrastive_aux_losses
                    ):
                        aux_per_layer = model._contrastive_aux_losses
                        # Stack to a single tensor so the .sum() is a single
                        # CUDA op rather than a Python-loop over scalars.
                        aux_total = torch.stack(aux_per_layer).sum()
                        lam = lambda_schedule(
                            step,
                            config.engram_vcontrast_warmup_steps,
                            config.engram_vcontrast_lambda,
                        )
                        loss = loss + lam * aux_total
                        # Detach for logging — these are consumed by CSV /
                        # console emit and must not retain the graph.
                        accum_vcontrast_aux += aux_total.detach().item()
                        # Per-layer accumulator: keys are str(layer_idx),
                        # matching the ModuleDict key convention. The model's
                        # forward iterates layer indices in ASCENDING order
                        # and appends one aux-loss per injection hit, so
                        # `aux_per_layer[i]` corresponds to the i-th
                        # smallest entry in `config.engram_inject_layers`.
                        # Sort here so a config like `(5, 2)` maps values to
                        # the correct keys instead of silently swapping them.
                        for layer_key, layer_loss in zip(
                            (str(i) for i in sorted(config.engram_inject_layers)),
                            aux_per_layer,
                            strict=True,
                        ):
                            accum_vcontrast_per_layer[layer_key] = (
                                accum_vcontrast_per_layer.get(layer_key, 0.0)
                                + layer_loss.detach().item()
                            )

                    # iota-Q-diversity (ZEB-130): drain Q-div aux loss sink,
                    # apply lambda warmup, add to total loss before backward.
                    # Accumulators (accum_qdiv_aux, accum_qdiv_per_layer) are
                    # populated here so the CSV / console blocks below can read
                    # them without holding live tensors past backward.
                    if config.engram_qdiv_enabled and model._qdiv_aux_losses:
                        per_layer_qd: list[torch.Tensor] = list(
                            model._qdiv_aux_losses
                        )
                        model._qdiv_aux_losses.clear()
                        aux_qdiv_total = torch.stack(per_layer_qd).sum()
                        lam_qdiv = lambda_schedule(
                            step,
                            config.engram_qdiv_warmup_steps,
                            config.engram_qdiv_lambda,
                        )
                        loss = loss + lam_qdiv * aux_qdiv_total
                        accum_qdiv_aux += aux_qdiv_total.detach().item()
                        for layer_key, layer_loss in zip(
                            (str(i) for i in sorted(config.engram_inject_layers)),
                            per_layer_qd,
                            strict=True,
                        ):
                            accum_qdiv_per_layer[layer_key] = (
                                accum_qdiv_per_layer.get(layer_key, 0.0)
                                + layer_loss.detach().item()
                            )

                accum_loss += loss.item()
                (loss / args.grad_accum_steps).backward()

            grad_norm = None
            if args.max_grad_norm > 0:
                all_params = list(model.parameters())
                if thought_norm is not None:
                    all_params.extend(thought_norm.parameters())
                if uq_head is not None:
                    all_params.extend(uq_head.parameters())
                if mtp_head is not None:
                    all_params.extend(mtp_head.parameters())
                if latent_projection is not None and args.contrastive_loss:
                    all_params.extend(latent_projection.parameters())
                if consol_decoder is not None:
                    all_params.extend(consol_decoder.parameters())
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    all_params, args.max_grad_norm,
                ).item()

            optimizer.step()

            # Model gamma: adapt lambda_ent to keep the mean gate probability
            # inside the configured band. Prevents both gate collapse
            # (mean drifts below low bound → raise lambda) and saturation
            # (mean drifts above high bound → lower lambda, let the gate
            # sparsify naturally).
            # Skip during warmup clamp: the gate is artificially held high,
            # so scaling down would drain lambda before it's ever needed.
            if (
                model.engram_ann is not None
                and engram_ann_entropy_bounds is not None
                and args.grad_accum_steps > 0
                and step >= model.engram_ann.clamp_until_step
            ):
                mean_gate = accum_ann_gate_mean / args.grad_accum_steps
                low, high = engram_ann_entropy_bounds
                if mean_gate < low:
                    dynamic_entropy_lambda = min(
                        ENTROPY_LAMBDA_MAX,
                        dynamic_entropy_lambda * ENTROPY_LAMBDA_SCALE_UP,
                    )
                elif mean_gate > high:
                    dynamic_entropy_lambda = max(
                        ENTROPY_LAMBDA_MIN,
                        dynamic_entropy_lambda * ENTROPY_LAMBDA_SCALE_DOWN,
                    )

            dt_ms = (time.time() - step_start) * 1000
            raw_loss = accum_loss / args.grad_accum_steps
            current_lr = optimizer.param_groups[0]["lr"]

            if step % 10 == 0:
                ct_str = f"  thoughts={num_thoughts}" if args.coconut else ""
                uq_str = ""
                if uq_head is not None:
                    raw_uq = accum_uq_loss / args.grad_accum_steps
                    uq_str = f"  uq_loss={raw_uq:.4f}"
                mtp_str = ""
                if mtp_head is not None:
                    raw_mtp = accum_mtp_loss / args.grad_accum_steps
                    mtp_str = f"  mtp_loss={raw_mtp:.4f}"
                cl_str = ""
                if args.contrastive_loss:
                    raw_cl = accum_cl_loss / args.grad_accum_steps
                    cl_str = f"  cl_loss={raw_cl:.4f}"
                kl_str = ""
                if args.kl_lambda > 0:
                    raw_kl = accum_kl_loss / args.grad_accum_steps
                    kl_str = f"  kl_loss={raw_kl:.4f}  kl_lambda={args.kl_lambda:.2f}"
                ann_str = ""
                if model.engram_ann is not None:
                    raw_ent = accum_ann_ent_loss / args.grad_accum_steps
                    raw_gate = accum_ann_gate_mean / args.grad_accum_steps
                    ann_str = (
                        f"  ann_ent={raw_ent:.4f}"
                        f"  ann_gate={raw_gate:.3f}"
                        f"  lambda_ent={dynamic_entropy_lambda:.4f}"
                    )
                hg_str = ""
                hg_mod = None
                if hasattr(model, "engram_xattn") and model.engram_xattn is not None and hasattr(model.engram_xattn, "head_gates"):
                    hg_mod = model.engram_xattn
                elif hasattr(model, "engram_ann") and model.engram_ann is not None and hasattr(model.engram_ann, "head_gates"):
                    hg_mod = model.engram_ann
                if hg_mod is not None:
                    with torch.no_grad():
                        gates = torch.sigmoid(hg_mod.head_gates)
                    hg_str = (
                        f"  hg_std={gates.std().item():.3f}"
                        f"  hg_min={gates.min().item():.3f}"
                        f"  hg_max={gates.max().item():.3f}"
                    )
                consol_str = ""
                if consol_decoder is not None:
                    raw_mse = accum_mse_loss / args.grad_accum_steps
                    consol_active = 1 if step >= args.consolidation_start_step else 0
                    consol_str = (
                        f"  mse={raw_mse:.4f}"
                        f"  phase={consol_active}"
                        f"  inject={inject_mult:.3f}"
                    )
                # η-B capgap: log tanh(alpha) gate values per injection layer.
                # This is the primary diagnostic for whether the optimizer opens
                # the gates (alpha > 0 tanh -> 0.5+) or keeps them closed.
                capgap_str = ""
                if model.engram_injections is not None:
                    with torch.no_grad():
                        gate_parts = []
                        for layer_key, injection in model.engram_injections.items():
                            gate_val = torch.tanh(injection.alpha).item()
                            gate_parts.append(f"g{layer_key}={gate_val:+.3f}")
                        capgap_str = "  " + " ".join(gate_parts)
                # θ-V-contrast (ZEB-130): aux loss + lambda + per-layer.
                vcontrast_str = ""
                if config.engram_vcontrast_enabled:
                    raw_aux = accum_vcontrast_aux / args.grad_accum_steps
                    raw_lam = lambda_schedule(
                        step,
                        config.engram_vcontrast_warmup_steps,
                        config.engram_vcontrast_lambda,
                    )
                    parts = [f"aux={raw_aux:.4f}"]
                    # Sorted to match CSV column order.
                    for lk in (str(i) for i in sorted(config.engram_inject_layers)):
                        if lk in accum_vcontrast_per_layer:
                            v = accum_vcontrast_per_layer[lk] / args.grad_accum_steps
                            parts.append(f"aux_L{lk}={v:.4f}")
                    parts.append(f"λ={raw_lam:.3f}")
                    vcontrast_str = "  " + "  ".join(parts)
                print(
                    f"step={step:5d}  loss={raw_loss:.4f}  lr={current_lr:.6f}"
                    f"{ct_str}{uq_str}{mtp_str}{cl_str}{kl_str}{ann_str}{hg_str}{consol_str}{vcontrast_str}{capgap_str}"
                )
                # iota-Q-diversity (ZEB-130): separate console line for qdiv
                # aux. Print unconditionally when qdiv is enabled — suppressing
                # the line when accum_qdiv_aux == 0.0 hides both (a) warmup
                # steps where lambda is still 0 and (b) regressions where the
                # sink failed to populate. Both are signals we want visible.
                if config.engram_qdiv_enabled:
                    raw_qdiv_aux = accum_qdiv_aux / args.grad_accum_steps
                    raw_qdiv_lam = lambda_schedule(
                        step,
                        config.engram_qdiv_warmup_steps,
                        config.engram_qdiv_lambda,
                    )
                    qdiv_parts = [f"aux={raw_qdiv_aux:.4f}"]
                    for lk in (str(i) for i in sorted(config.engram_inject_layers)):
                        if lk in accum_qdiv_per_layer:
                            v = accum_qdiv_per_layer[lk] / args.grad_accum_steps
                            qdiv_parts.append(f"aux_L{lk}={v:.4f}")
                    qdiv_parts.append(f"lambda={raw_qdiv_lam:.3f}")
                    print("  [qdiv] " + "  ".join(qdiv_parts))

            val_loss_str = ""
            if args.save_every > 0 and step > 0 and step % args.save_every == 0:
                save_checkpoint(model, optimizer, step, args.output_dir)
                if thought_norm is not None:
                    _save_thought_norm(thought_norm, step, args.output_dir)
                if uq_head is not None:
                    _save_uq_head(uq_head, step, args.output_dir)
                if mtp_head is not None:
                    _save_mtp_head(mtp_head, step, args.output_dir)
                if latent_projection is not None and args.contrastive_loss:
                    _save_latent_projection(latent_projection, step, args.output_dir)
                print(f"  -> checkpoint saved at step {step}")
                if val_loader is not None:
                    val_loss = compute_validation_loss(
                        model, val_loader, config.vocab_size, device,
                        amp_dtype=amp_dtype, engram_table=engram_table,
                        thought_norm=thought_norm,
                        think_token_id=args.think_token_id if args.coconut else None,
                        num_thoughts=num_thoughts,
                        latent_projection=latent_projection,
                    )
                    val_loss_str = f"{val_loss:.6f}"
                    print(f"  -> val_loss={val_loss:.4f}")

            if (
                args.checkpoint_interval > 0
                and step > 0
                and step % args.checkpoint_interval == 0
            ):
                save_resumable_checkpoint(
                    model, optimizer, step, args.output_dir,
                    rng_state=capture_rng_state(
                        device, capgap_shuffle_gen=capgap_shuffle_gen,
                    ),
                    last_val_loss=float(val_loss_str) if val_loss_str else None,
                    dynamic_entropy_lambda=dynamic_entropy_lambda,
                )
                print(f"  -> resumable checkpoint saved at step {step}")

            if step % 10 == 0 and csv_writer is not None:
                uq_loss_str = ""
                if uq_head is not None:
                    uq_loss_str = f"{accum_uq_loss / args.grad_accum_steps:.6f}"
                mtp_loss_str = ""
                if mtp_head is not None:
                    mtp_loss_str = f"{accum_mtp_loss / args.grad_accum_steps:.6f}"
                cl_loss_str = ""
                if args.contrastive_loss:
                    cl_loss_str = f"{accum_cl_loss / args.grad_accum_steps:.6f}"
                ann_ent_str = ""
                ann_gate_str = ""
                ann_lambda_str = ""
                if model.engram_ann is not None:
                    ann_ent_str = f"{accum_ann_ent_loss / args.grad_accum_steps:.6f}"
                    ann_gate_str = f"{accum_ann_gate_mean / args.grad_accum_steps:.6f}"
                    ann_lambda_str = f"{dynamic_entropy_lambda:.6f}"
                # Non-hg columns: 13 base + 3 consol + 1 kl_loss (ZEB-139)
                # + (0 when vcontrast off) or (2 scalars + N per-layer) when
                # on. Subtract from total to size the hg block (hg_0..hg_N,
                # std/min/max). The constant 17 must increase whenever a
                # non-conditional column is appended to expected_header
                # outside the hg / vcontrast / qdiv blocks — bumping it
                # is what keeps `len(hg_cols)` matching the hg slot count
                # that expected_header reserved.
                n_vcontrast_cols = len(vcontrast_cols)
                n_qdiv_cols = len(qdiv_cols)
                n_hg_slots = len(expected_header) - (17 + n_vcontrast_cols + n_qdiv_cols)
                hg_cols = [""] * n_hg_slots
                hg_module = None
                if hasattr(model, "engram_xattn") and model.engram_xattn is not None and hasattr(model.engram_xattn, "head_gates"):
                    hg_module = model.engram_xattn
                elif hasattr(model, "engram_ann") and model.engram_ann is not None and hasattr(model.engram_ann, "head_gates"):
                    hg_module = model.engram_ann
                if hg_module is not None:
                    with torch.no_grad():
                        gates = torch.sigmoid(hg_module.head_gates)
                    n_logged = min(n_hg_slots - 3, gates.numel())
                    for i in range(n_logged):
                        hg_cols[i] = f"{gates[i].item():.6f}"
                    hg_cols[-3] = f"{gates.std().item():.6f}"
                    hg_cols[-2] = f"{gates.min().item():.6f}"
                    hg_cols[-1] = f"{gates.max().item():.6f}"
                mse_loss_str = ""
                consol_phase_str = ""
                inject_mult_str = ""
                if consol_decoder is not None:
                    mse_loss_str = f"{accum_mse_loss / args.grad_accum_steps:.6f}"
                    consol_phase_str = "1" if step >= args.consolidation_start_step else "0"
                    inject_mult_str = f"{inject_mult:.6f}"
                # θ-V-contrast (ZEB-130): emit aux + per-layer + lambda only
                # when enabled, matching the conditional header. Row cell
                # order must track `vcontrast_cols` exactly.
                vcontrast_row_cells: list[str] = []
                if config.engram_vcontrast_enabled:
                    raw_aux = accum_vcontrast_aux / args.grad_accum_steps
                    vcontrast_row_cells.append(f"{raw_aux:.6f}")
                    # Sorted to match header order and the accumulator's zip.
                    for layer_idx in sorted(config.engram_inject_layers or ()):
                        lk = str(layer_idx)
                        if lk in accum_vcontrast_per_layer:
                            v = accum_vcontrast_per_layer[lk] / args.grad_accum_steps
                            vcontrast_row_cells.append(f"{v:.6f}")
                        else:
                            vcontrast_row_cells.append("")
                    current_lam = lambda_schedule(
                        step,
                        config.engram_vcontrast_warmup_steps,
                        config.engram_vcontrast_lambda,
                    )
                    vcontrast_row_cells.append(f"{current_lam:.6f}")
                # iota-Q-diversity (ZEB-130): emit aux + per-layer + lambda only
                # when enabled, matching `qdiv_cols` order exactly.
                qdiv_row_cells: list[str] = []
                if config.engram_qdiv_enabled:
                    raw_qdiv = accum_qdiv_aux / args.grad_accum_steps
                    qdiv_row_cells.append(f"{raw_qdiv:.6f}")
                    for layer_idx in sorted(config.engram_inject_layers or ()):
                        lk = str(layer_idx)
                        if lk in accum_qdiv_per_layer:
                            v = accum_qdiv_per_layer[lk] / args.grad_accum_steps
                            qdiv_row_cells.append(f"{v:.6f}")
                        else:
                            qdiv_row_cells.append("")
                    current_qdiv_lam = lambda_schedule(
                        step,
                        config.engram_qdiv_warmup_steps,
                        config.engram_qdiv_lambda,
                    )
                    qdiv_row_cells.append(f"{current_qdiv_lam:.6f}")
                kl_loss_str = (
                    f"{accum_kl_loss / args.grad_accum_steps:.6f}"
                    if args.kl_lambda > 0
                    else ""
                )
                csv_writer.writerow([
                    step,
                    f"{raw_loss:.6f}",
                    uq_loss_str,
                    mtp_loss_str,
                    cl_loss_str,
                    ann_ent_str,
                    ann_gate_str,
                    ann_lambda_str,
                    val_loss_str,
                    f"{current_lr:.8f}",
                    f"{grad_norm:.6f}" if grad_norm is not None else "",
                    num_thoughts,
                    f"{dt_ms:.1f}",
                    *hg_cols,
                    mse_loss_str,
                    consol_phase_str,
                    inject_mult_str,
                    *vcontrast_row_cells,
                    *qdiv_row_cells,
                    kl_loss_str,
                ])
                csv_file.flush()

        if start_step >= args.steps:
            print(f"Nothing to do: already at step {start_step} (--steps={args.steps})")
        else:
            save_checkpoint(model, optimizer, args.steps, args.output_dir)
            if thought_norm is not None:
                _save_thought_norm(thought_norm, args.steps, args.output_dir)
            if uq_head is not None:
                _save_uq_head(uq_head, args.steps, args.output_dir)
            if mtp_head is not None:
                _save_mtp_head(mtp_head, args.steps, args.output_dir)
            if latent_projection is not None and args.contrastive_loss:
                _save_latent_projection(latent_projection, args.steps, args.output_dir)
            final_val_loss = None
            if val_loader is not None:
                final_val_loss = compute_validation_loss(
                    model, val_loader, config.vocab_size, device,
                    amp_dtype=amp_dtype, engram_table=engram_table,
                    thought_norm=thought_norm,
                    think_token_id=args.think_token_id if args.coconut else None,
                    num_thoughts=num_thoughts,
                    latent_projection=latent_projection,
                )
                print(f"Final val_loss={final_val_loss:.4f}")
            if args.checkpoint_interval > 0:
                save_resumable_checkpoint(
                    model, optimizer, args.steps - 1, args.output_dir,
                    rng_state=capture_rng_state(
                        device, capgap_shuffle_gen=capgap_shuffle_gen,
                    ),
                    last_val_loss=final_val_loss,
                    dynamic_entropy_lambda=dynamic_entropy_lambda,
                )
            print(f"Training complete. Final checkpoint at step {args.steps}")

            # θ-V-contrast (ZEB-130): final summary block.
            if config.engram_vcontrast_enabled:
                with torch.no_grad():
                    print(f"[vcontrast] Final step={args.steps}")
                    if model._contrastive_aux_losses:
                        # The list survives one final forward — but reliably the
                        # most recent training-step accumulator is what we want.
                        print(
                            f"    last_aux_total = {accum_vcontrast_aux / max(args.grad_accum_steps, 1):.6f}"
                        )
                    if final_val_loss is not None:
                        print(f"    val_loss (with inj)    = {final_val_loss:.6f}")
                    if model.engram_injections is not None:
                        for layer_key, injection in model.engram_injections.items():
                            alpha_val = injection.alpha.detach().item()
                            print(
                                f"    alpha_L{layer_key} = {alpha_val:+.6f}, "
                                f"tanh(alpha_L{layer_key}) = "
                                f"{math.tanh(alpha_val):+.6f}"
                            )

            # iota-Q-diversity (ZEB-130): final summary block, mirroring V-contrast.
            if config.engram_qdiv_enabled:
                print(f"[qdiv] Final step={args.steps}")
                print(
                    f"    last_aux_total = {accum_qdiv_aux / max(args.grad_accum_steps, 1):.6f}"
                )
                for lk in (str(i) for i in sorted(config.engram_inject_layers or ())):
                    if lk in accum_qdiv_per_layer:
                        v = accum_qdiv_per_layer[lk] / max(args.grad_accum_steps, 1)
                        print(f"    last_aux_L{lk} = {v:.6f}")
                final_qdiv_lam = lambda_schedule(
                    args.steps - 1,
                    config.engram_qdiv_warmup_steps,
                    config.engram_qdiv_lambda,
                )
                print(f"    final_lambda = {final_qdiv_lam:.6f}")
    finally:
        if csv_file is not None:
            csv_file.close()


if __name__ == "__main__":
    main()
