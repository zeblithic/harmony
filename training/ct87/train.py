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


def capture_rng_state(device: torch.device | None = None) -> dict:
    """Capture all RNG states for deterministic resume."""
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
    return state


def restore_rng_state(state: dict, device: torch.device | None = None) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ct87 model")
    parser.add_argument(
        "--config",
        choices=[
            "tiny", "target", "tiny_ffn_expanded",
            "tiny_engram_ann", "tiny_engram_ann_routed",
            "tiny_engram_xattn", "tiny_engram_xattn_routed",
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
    args = parser.parse_args()

    if args.data is None and not args.synthetic:
        print("Error: must provide --data <path> or --synthetic", file=sys.stderr)
        sys.exit(1)

    if args.grad_accum_steps < 1:
        print("Error: --grad-accum-steps must be >= 1", file=sys.stderr)
        sys.exit(1)

    if args.coconut and args.think_token_id is None:
        print("Error: --think-token-id is required with --coconut", file=sys.stderr)
        sys.exit(1)

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
    else:
        config = HarmonyModelConfig.target()
    seq_len = args.seq_len or (512 if args.config.startswith("tiny") else 2048)

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
    if args.config in ("tiny_engram_xattn", "tiny_engram_xattn_routed"):
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

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

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
            restore_rng_state(ckpt["rng_state"], device)
        if "dynamic_entropy_lambda" in ckpt:
            _pending_entropy_lambda = ckpt["dynamic_entropy_lambda"]
        last_val = ckpt.get("last_val_loss") or ckpt.get("best_val_loss")
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

    if args.synthetic:
        dataloader = make_synthetic_dataloader(config.vocab_size, seq_len, args.batch_size, args.seed)
    else:
        dataloader = make_hf_dataloader(args.data, seq_len, args.batch_size, args.seed)

    val_loader = None
    if args.val_data is not None:
        val_loader = make_hf_dataloader(args.val_data, seq_len, args.batch_size, args.seed + 1)
        print(f"Validation data loaded from {args.val_data}")

    csv_file = None
    csv_writer = None
    if args.log_file:
        expected_header = [
            "step", "loss", "uq_loss", "mtp_loss", "cl_loss",
            "ann_ent_loss", "ann_gate_mean", "ann_lambda_ent",
            "val_loss", "lr", "grad_norm", "num_thoughts", "dt_ms",
            "hg_0", "hg_1", "hg_2", "hg_3", "hg_4", "hg_5", "hg_6", "hg_7",
            "hg_std", "hg_min", "hg_max",
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
            if migrations_needed:
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
                    # Pad any remaining column deficit (covers multi-generation gaps)
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
            num_thoughts = coconut_curriculum.num_thoughts(step) if coconut_curriculum is not None else 0
            need_hidden = mtp_head is not None
            use_coconut = args.coconut and num_thoughts > 0
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
                print(
                    f"step={step:5d}  loss={raw_loss:.4f}  lr={current_lr:.6f}"
                    f"{ct_str}{uq_str}{mtp_str}{cl_str}{ann_str}{hg_str}"
                )

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
                    rng_state=capture_rng_state(device),
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
                n_hg_slots = len(expected_header) - 13  # 13 base columns before hg_*
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
                ])
                csv_file.flush()

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
                model, optimizer, args.steps, args.output_dir,
                rng_state=capture_rng_state(device),
                last_val_loss=final_val_loss,
                dynamic_entropy_lambda=dynamic_entropy_lambda,
            )
        print(f"Training complete. Final checkpoint at step {args.steps}")
    finally:
        if csv_file is not None:
            csv_file.close()


if __name__ == "__main__":
    main()
