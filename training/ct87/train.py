"""Training loop for ct87 -- CLI entry point.

Usage:
    python -m ct87.train --config tiny --data <path> --steps 200

For testing without a real dataset, use --synthetic to generate random tokens.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterator

import torch
import torch.nn.functional as F

from ct87.model import HarmonyModel, HarmonyModelConfig
from ct87.optim import Muon, WSDSchedule, partition_params


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

    Loads the dataset, concatenates all token sequences into one long stream,
    then yields random windows of [seq_len + 1] tokens packed into batches.
    """
    from datasets import load_from_disk

    dataset = load_from_disk(data_path)
    all_tokens: list[int] = []
    for example in dataset:
        all_tokens.extend(example["input_ids"])

    all_tokens_t = torch.tensor(all_tokens, dtype=torch.long)
    total = len(all_tokens_t)
    window = seq_len + 1

    rng = torch.Generator()
    rng.manual_seed(seed)

    while True:
        starts = torch.randint(0, total - window, (batch_size,), generator=rng)
        batch = torch.stack([all_tokens_t[s : s + window] for s in starts])
        yield batch


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


def load_checkpoint(
    model: HarmonyModel,
    output_dir: str,
    step: int,
) -> None:
    """Load model weights from a safetensors checkpoint."""
    from safetensors.torch import load_model

    weights_path = os.path.join(output_dir, f"model_step_{step}.safetensors")
    load_model(model, weights_path, strict=False)


def set_lr(optimizer: torch.optim.Optimizer, multiplier: float, base_lr: float) -> None:
    """Set learning rate for all param groups based on schedule multiplier."""
    for group in optimizer.param_groups:
        if "_base_lr" not in group:
            group["_base_lr"] = group["lr"]
        group["lr"] = group["_base_lr"] * multiplier


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
    args = parser.parse_args()

    if args.data is None and not args.synthetic:
        print("Error: must provide --data <path> or --synthetic", file=sys.stderr)
        sys.exit(1)

    config = HarmonyModelConfig.tiny() if args.config == "tiny" else HarmonyModelConfig.target()
    seq_len = args.seq_len or (512 if args.config == "tiny" else 2048)

    device = detect_device(args.device)
    torch.manual_seed(args.seed)
    print(f"Config: {args.config}, device: {device}, seq_len: {seq_len}")

    model = HarmonyModel(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    muon_params, adam_params = partition_params(model)
    optimizer = Muon(muon_params, adam_params, lr=args.lr, adam_lr=args.lr)
    schedule = WSDSchedule(warmup_steps=args.warmup, total_steps=args.steps)

    if args.synthetic:
        dataloader = make_synthetic_dataloader(config.vocab_size, seq_len, args.batch_size, args.seed)
    else:
        dataloader = make_hf_dataloader(args.data, seq_len, args.batch_size, args.seed)

    for step in range(args.steps):
        lr_mult = schedule.get_lr_multiplier(step)
        set_lr(optimizer, lr_mult, args.lr)

        batch = next(dataloader).to(device)
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]

        logits = model(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), targets.reshape(-1))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 10 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"step={step:5d}  loss={loss.item():.4f}  lr={current_lr:.6f}")

        if args.save_every > 0 and step > 0 and step % args.save_every == 0:
            save_checkpoint(model, optimizer, step, args.output_dir)
            print(f"  -> checkpoint saved at step {step}")

    save_checkpoint(model, optimizer, args.steps, args.output_dir)
    print(f"Training complete. Final checkpoint at step {args.steps}")


if __name__ == "__main__":
    main()
