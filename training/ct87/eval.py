"""Evaluation harness for ct87 -- compute perplexity on held-out data.

Usage:
    python -m ct87.eval --checkpoint model.safetensors --config tiny --data <path>
    python -m ct87.eval --checkpoint model.safetensors --config tiny --synthetic --num-batches 50

Outputs average cross-entropy loss, perplexity, and throughput.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Iterator

import torch
import torch.nn.functional as F

from ct87.model import HarmonyModel, HarmonyModelConfig
from ct87.train import detect_device, make_hf_dataloader, make_synthetic_dataloader


def load_checkpoint_path(model: HarmonyModel, path: str) -> None:
    """Load model weights from a safetensors file at an explicit path."""
    from safetensors.torch import load_model

    load_model(model, path)


def evaluate(
    model: HarmonyModel,
    dataloader: Iterator[torch.Tensor],
    vocab_size: int,
    device: torch.device,
    num_batches: int,
    amp_dtype: torch.dtype | None = None,
    engram_table: object | None = None,
) -> dict[str, float | int]:
    """Run evaluation and return metrics dict.

    Returns:
        Dict with keys: loss, perplexity, total_tokens (int),
        tokens_per_sec, elapsed_sec.
    """
    if num_batches < 1:
        raise ValueError("num_batches must be >= 1")

    was_training = model.training
    model.eval()
    use_amp = amp_dtype is not None
    device_type = device.type

    total_loss = 0.0
    total_tokens = 0
    start_time = time.time()

    try:
        with torch.no_grad():
            for i in range(num_batches):
                try:
                    batch = next(dataloader)
                except StopIteration as exc:
                    raise ValueError(
                        f"dataloader exhausted after {i} batches; "
                        f"requested num_batches={num_batches}"
                    ) from exc
                input_ids = batch[:, :-1]
                targets = batch[:, 1:]

                # Engram lookup on CPU tensors to avoid device-host sync
                # (lookup_batch calls .tolist() internally)
                engram_emb = None
                if engram_table is not None:
                    engram_emb = engram_table.lookup_batch(input_ids)

                input_ids = input_ids.to(device)
                targets = targets.to(device)

                with torch.autocast(device_type, dtype=amp_dtype, enabled=use_amp):
                    logits = model(input_ids, engram_embeddings=engram_emb)
                    loss = F.cross_entropy(
                        logits.reshape(-1, vocab_size),
                        targets.reshape(-1),
                    )

                batch_tokens = targets.numel()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens

                if (i + 1) % 10 == 0:
                    avg = total_loss / total_tokens
                    print(f"  batch {i + 1}/{num_batches}  loss={avg:.4f}  ppl={math.exp(avg):.2f}")
    finally:
        model.train(was_training)

    elapsed = time.time() - start_time
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    tokens_per_sec = total_tokens / elapsed

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "total_tokens": total_tokens,
        "tokens_per_sec": tokens_per_sec,
        "elapsed_sec": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ct87 model perplexity")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to safetensors checkpoint")
    parser.add_argument("--config", choices=["tiny", "target"], required=True)
    parser.add_argument("--data", type=str, default=None, help="Path to pre-tokenized HF dataset")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic random data")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=100)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--engram-table", type=str, default=None,
        help="Path to Engram safetensors table",
    )
    parser.add_argument(
        "--engram-seeds", type=str, default="42,99,137,251",
        help="Comma-separated xxhash64 seeds for Engram table lookup",
    )
    args = parser.parse_args()

    if args.data is None and not args.synthetic:
        print("Error: must provide --data <path> or --synthetic", file=sys.stderr)
        sys.exit(1)

    if args.data is not None and args.synthetic:
        print("Error: --data and --synthetic are mutually exclusive", file=sys.stderr)
        sys.exit(1)

    if args.batch_size < 1:
        print("Error: --batch-size must be >= 1", file=sys.stderr)
        sys.exit(1)

    config = HarmonyModelConfig.tiny() if args.config == "tiny" else HarmonyModelConfig.target()
    seq_len = args.seq_len or (512 if args.config == "tiny" else 2048)

    if seq_len < 1 or seq_len > config.max_seq_len:
        print(
            f"Error: --seq-len must be between 1 and {config.max_seq_len} "
            f"for config '{args.config}'",
            file=sys.stderr,
        )
        sys.exit(1)

    device = detect_device(args.device)
    if args.dtype == "bfloat16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        print("Error: --dtype bfloat16 not supported on this CUDA device", file=sys.stderr)
        sys.exit(1)
    amp_dtype = torch.bfloat16 if args.dtype == "bfloat16" else None
    print(f"Config: {args.config}, device: {device}, seq_len: {seq_len}, dtype: {args.dtype}")

    model = HarmonyModel(config).to(device)
    load_checkpoint_path(model, args.checkpoint)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

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

    if args.synthetic:
        dataloader = make_synthetic_dataloader(config.vocab_size, seq_len, args.batch_size, args.seed)
    else:
        dataloader = make_hf_dataloader(args.data, seq_len, args.batch_size, args.seed)

    print(f"Evaluating {args.num_batches} batches (batch_size={args.batch_size}, seq_len={seq_len})...")
    metrics = evaluate(
        model, dataloader, config.vocab_size, device,
        num_batches=args.num_batches,
        amp_dtype=amp_dtype,
        engram_table=engram_table,
    )

    print("\nResults:")
    print(f"  loss:       {metrics['loss']:.4f}")
    print(f"  perplexity: {metrics['perplexity']:.2f}")
    print(f"  tokens:     {metrics['total_tokens']:,}")
    print(f"  tok/sec:    {metrics['tokens_per_sec']:.0f}")
    print(f"  elapsed:    {metrics['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
