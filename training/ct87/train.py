"""Training loop for ct87 -- CLI entry point.

Usage:
    python -m ct87.train --config tiny --data <path> --steps 200

For testing without a real dataset, use --synthetic to generate random tokens.
"""

from __future__ import annotations

import argparse
import csv
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
) -> float:
    """Run validation and return average cross-entropy loss."""
    was_training = model.training
    model.train(False)
    use_amp = amp_dtype is not None
    device_type = device.type
    try:
        total_loss = 0.0
        with torch.no_grad():
            for _ in range(num_batches):
                batch = next(val_loader).to(device)
                input_ids = batch[:, :-1]
                targets = batch[:, 1:]
                engram_emb = None
                if engram_table is not None:
                    engram_emb = engram_table.lookup_batch(input_ids)
                with torch.autocast(device_type, dtype=amp_dtype, enabled=use_amp):
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
    parser.add_argument("--config", choices=["tiny", "target"], default="tiny")
    parser.add_argument("--data", type=str, default=None, help="Path to pre-tokenized HF dataset")
    parser.add_argument("--val-data", type=str, default=None, help="Path to validation HF dataset (optional)")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic random data")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--output-dir", type=str, default="training/checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-file", type=str, default=None, help="Path to CSV log file")
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

    config = HarmonyModelConfig.tiny() if args.config == "tiny" else HarmonyModelConfig.target()
    seq_len = args.seq_len or (512 if args.config == "tiny" else 2048)

    if args.coconut:
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
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

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

    # COCONUT continuous thought setup
    thought_norm = None
    coconut_curriculum = None
    if args.coconut:
        from ct87.coconut import ThoughtNorm, CurriculumSchedule

        config.think_token_id = args.think_token_id
        config.ct_max_steps = args.ct_max_steps
        thought_norm = ThoughtNorm(config.hidden_dim, eps=config.rms_norm_eps).to(device)
        coconut_curriculum = CurriculumSchedule(args.ct_max_steps, args.steps)
        print(
            f"COCONUT enabled: think_token_id={args.think_token_id}, "
            f"ct_max_steps={args.ct_max_steps}"
        )

    muon_params, adam_params = partition_params(model)
    if thought_norm is not None:
        adam_params.extend(thought_norm.parameters())
    optimizer = Muon(muon_params, adam_params, lr=args.lr, adam_lr=args.lr)
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
        csv_file = open(args.log_file, "a", newline="")
        csv_writer = csv.writer(csv_file)
        if os.path.getsize(args.log_file) == 0:
            csv_writer.writerow(["step", "loss", "val_loss", "lr", "grad_norm", "dt_ms"])

    try:
        for step in range(args.steps):
            step_start = time.time()
            lr_mult = schedule.get_lr_multiplier(step)
            set_lr(optimizer, lr_mult)
            optimizer.zero_grad()

            accum_loss = 0.0
            num_thoughts = coconut_curriculum.num_thoughts(step) if coconut_curriculum is not None else 0
            for micro_step in range(args.grad_accum_steps):
                batch = next(dataloader).to(device)
                input_ids = batch[:, :-1]
                targets = batch[:, 1:]

                # Compute Engram embeddings if table is loaded
                engram_emb = None
                if engram_table is not None:
                    with torch.no_grad():
                        engram_emb = engram_table.lookup_batch(input_ids)

                with torch.autocast(device_type, dtype=amp_dtype, enabled=use_amp):
                    if args.coconut and num_thoughts > 0:
                        from ct87.coconut import coconut_forward, coconut_loss
                        logits, think_mask = coconut_forward(
                            model, thought_norm, input_ids,
                            args.think_token_id, num_thoughts,
                            engram_embeddings=engram_emb,
                        )
                        # Prepend ignore targets for think positions
                        think_targets = torch.full(
                            (targets.shape[0], num_thoughts), -100,
                            dtype=targets.dtype, device=targets.device,
                        )
                        aug_targets = torch.cat([think_targets, targets], dim=1)
                        loss = coconut_loss(logits, aug_targets, think_mask)
                    else:
                        logits = model(input_ids, engram_embeddings=engram_emb)
                        loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), targets.reshape(-1))
                accum_loss += loss.item()
                (loss / args.grad_accum_steps).backward()

            grad_norm = None
            if args.max_grad_norm > 0:
                all_params = list(model.parameters())
                if thought_norm is not None:
                    all_params.extend(thought_norm.parameters())
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    all_params, args.max_grad_norm,
                ).item()

            optimizer.step()

            dt_ms = (time.time() - step_start) * 1000
            raw_loss = accum_loss / args.grad_accum_steps
            current_lr = optimizer.param_groups[0]["lr"]

            if step % 10 == 0:
                ct_str = f"  thoughts={num_thoughts}" if args.coconut else ""
                print(f"step={step:5d}  loss={raw_loss:.4f}  lr={current_lr:.6f}{ct_str}")

            val_loss_str = ""
            if args.save_every > 0 and step > 0 and step % args.save_every == 0:
                save_checkpoint(model, optimizer, step, args.output_dir)
                if thought_norm is not None:
                    _save_thought_norm(thought_norm, step, args.output_dir)
                print(f"  -> checkpoint saved at step {step}")
                if val_loader is not None:
                    val_loss = compute_validation_loss(
                        model, val_loader, config.vocab_size, device,
                        amp_dtype=amp_dtype, engram_table=engram_table,
                    )
                    val_loss_str = f"{val_loss:.6f}"
                    print(f"  -> val_loss={val_loss:.4f}")

            if step % 10 == 0 and csv_writer is not None:
                csv_writer.writerow([
                    step,
                    f"{raw_loss:.6f}",
                    val_loss_str,
                    f"{current_lr:.8f}",
                    f"{grad_norm:.6f}" if grad_norm is not None else "",
                    f"{dt_ms:.1f}",
                ])
                csv_file.flush()

        save_checkpoint(model, optimizer, args.steps, args.output_dir)
        if thought_norm is not None:
            _save_thought_norm(thought_norm, args.steps, args.output_dir)
        if val_loader is not None:
            val_loss = compute_validation_loss(
                model, val_loader, config.vocab_size, device,
                amp_dtype=amp_dtype, engram_table=engram_table,
            )
            print(f"Final val_loss={val_loss:.4f}")
        print(f"Training complete. Final checkpoint at step {args.steps}")
    finally:
        if csv_file is not None:
            csv_file.close()


if __name__ == "__main__":
    main()
