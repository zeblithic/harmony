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
from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from ct87.model import HarmonyModel, HarmonyModelConfig
from ct87.train import detect_device, make_hf_dataloader, make_synthetic_dataloader

if TYPE_CHECKING:
    from ct87.engram import EngramTable


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
    engram_table: EngramTable | None = None,
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


def evaluate_projected(
    model: HarmonyModel,
    dataloader: Iterator[torch.Tensor],
    vocab_size: int,
    device: torch.device,
    num_batches: int,
    engram_table: EngramTable,
    projection: object,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, float | int]:
    """Run perplexity measurement with projection-generated Engram keys.

    Same as evaluate() but uses projection->binarize->xxhash for Engram
    key generation instead of xxhash-on-token-bytes.

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

                # Get embeddings for projection key generation
                embeddings = model.embed_tokens(input_ids.to(device))
                engram_emb = engram_table.lookup_batch_projected(
                    input_ids, embeddings.cpu(), projection,
                )

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


def run_comparison(
    model: HarmonyModel,
    config: HarmonyModelConfig,
    dataloader_fn: object,
    device: torch.device,
    num_batches: int,
    amp_dtype: torch.dtype | None,
    engram_table: EngramTable | None = None,
    projection: object = None,
) -> None:
    """Run three-way perplexity comparison and print results table.

    Measures: (1) baseline (no engram), (2) engram with xxhash keys,
    (3) engram with projection keys.  Each run uses a fresh dataloader
    with the same seed for identical data.
    """
    results: list[tuple[str, dict[str, float | int]]] = []

    # 1. Baseline (no engram)
    print("\n=== Baseline (no engram) ===")
    dl = dataloader_fn()
    m = evaluate(model, dl, config.vocab_size, device, num_batches, amp_dtype)
    results.append(("Baseline", m))

    # 2. Engram with xxhash keys
    if engram_table is not None:
        print("\n=== Engram (xxhash keys) ===")
        dl = dataloader_fn()
        m = evaluate(model, dl, config.vocab_size, device, num_batches, amp_dtype, engram_table)
        results.append(("Engram (xxhash)", m))

    # 3. Engram with projection keys
    if engram_table is not None and projection is not None:
        print("\n=== Engram (projection keys) ===")
        dl = dataloader_fn()
        m = evaluate_projected(model, dl, config.vocab_size, device, num_batches, engram_table, projection, amp_dtype)
        results.append(("Engram (projection)", m))

    # Print comparison table
    baseline_loss = results[0][1]["loss"]
    print("\n" + "=" * 72)
    print("COMPARISON RESULTS")
    print("=" * 72)
    print(f"{'Config':<25} {'Loss':>8} {'PPL':>10} {'vs Baseline':>12} {'tok/s':>8}")
    print("-" * 72)
    for name, m in results:
        delta = ""
        if m["loss"] != baseline_loss:
            pct = (m["loss"] - baseline_loss) / baseline_loss * 100
            delta = f"{pct:+.3f}%"
        print(
            f"{name:<25} {m['loss']:>8.4f} {m['perplexity']:>10.2f} "
            f"{delta:>12} {m['tokens_per_sec']:>8.0f}"
        )
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure ct87 model perplexity")
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
    # Latent projection args
    parser.add_argument(
        "--latent-projection", type=str, default=None,
        help="Path to latent projection checkpoint (enables projection key mode)",
    )
    parser.add_argument(
        "--latent-intermediate-dim", type=int, default=None,
        help="Intermediate dimension for latent projection MLP",
    )
    parser.add_argument(
        "--latent-dim", type=int, default=None,
        help="Output dimension for latent projection MLP",
    )
    # Comparison and analysis modes
    parser.add_argument(
        "--compare", action="store_true",
        help="Run three-way comparison: baseline, xxhash engram, projection engram",
    )
    parser.add_argument(
        "--key-overlap", action="store_true",
        help="Run key overlap analysis: compare xxhash vs projection key distributions",
    )
    parser.add_argument(
        "--key-overlap-batches", type=int, default=10,
        help="Number of batches for key overlap analysis (default: 10)",
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

    if args.num_batches < 1:
        print("Error: --num-batches must be >= 1", file=sys.stderr)
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

    if args.compare and args.engram_table is None:
        print("Error: --compare requires --engram-table", file=sys.stderr)
        sys.exit(1)

    config = HarmonyModelConfig.tiny() if args.config == "tiny" else HarmonyModelConfig.target()
    seq_len = args.seq_len if args.seq_len is not None else (512 if args.config == "tiny" else 2048)

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

        try:
            seeds = [int(s) for s in args.engram_seeds.split(",")]
        except ValueError:
            print(
                f"Error: invalid --engram-seeds value: must be comma-separated "
                f"integers, got '{args.engram_seeds}'",
                file=sys.stderr,
            )
            sys.exit(1)
        engram_table = EngramTable.from_safetensors(
            args.engram_table, hash_seeds=seeds, device=str(device),
        )
        print(
            f"Engram table loaded: {engram_table.total_entries:,} entries, "
            f"dim={engram_table.engram_dim}, heads={engram_table.num_heads}"
        )

    projection = None
    if args.latent_projection is not None:
        from ct87.latent_projection import LatentProjection

        projection = LatentProjection.from_checkpoint(
            args.latent_projection,
            hidden_dim=config.hidden_dim,
            intermediate_dim=args.latent_intermediate_dim,
            latent_dim=args.latent_dim,
        )
        proj_params = sum(p.numel() for p in projection.parameters())
        print(
            f"Latent projection loaded: "
            f"{config.hidden_dim}->{args.latent_intermediate_dim}->{args.latent_dim}, "
            f"{proj_params:,} params"
        )

    # Key overlap analysis (runs before main perplexity measurement)
    if args.key_overlap:
        if projection is None or engram_table is None:
            print(
                "Error: --key-overlap requires --latent-projection and --engram-table",
                file=sys.stderr,
            )
            sys.exit(1)

        from ct87.latent_projection import compute_key_overlap

        print(f"\nKey overlap analysis ({args.key_overlap_batches} batches)...")
        if args.synthetic:
            overlap_dl = make_synthetic_dataloader(config.vocab_size, seq_len, args.batch_size, args.seed)
        else:
            overlap_dl = make_hf_dataloader(args.data, seq_len, args.batch_size, args.seed)

        agg_lookups = 0
        agg_matching = 0
        for _bi in range(args.key_overlap_batches):
            batch = next(overlap_dl)
            input_ids = batch[:, :-1].to(device)
            stats = compute_key_overlap(model, projection, engram_table, input_ids)
            agg_lookups += stats["total_lookups"]
            agg_matching += stats["matching_indices"]

        pct = (agg_matching / agg_lookups * 100) if agg_lookups > 0 else 0.0

        print("\n" + "=" * 50)
        print("KEY OVERLAP ANALYSIS")
        print("=" * 50)
        print(f"  Total (n-gram, head) lookups:  {agg_lookups:,}")
        print(f"  Matching table indices:        {agg_matching:,}")
        print(f"  Overlap:                       {pct:.2f}%")
        print(f"  Divergent:                     {100 - pct:.2f}%")
        print("=" * 50)
        if pct > 90:
            print("  -> Keys are very similar -- projection may not change much.")
        elif pct < 10:
            print("  -> Keys are very different -- watch for train/inference mismatch.")
        else:
            print("  -> Moderate divergence -- projection is reshuffling some lookups.")
        print()

    # Comparison mode: run all three configs
    if args.compare:
        def make_dataloader():
            if args.synthetic:
                return make_synthetic_dataloader(config.vocab_size, seq_len, args.batch_size, args.seed)
            return make_hf_dataloader(args.data, seq_len, args.batch_size, args.seed)

        run_comparison(
            model, config, make_dataloader, device, args.num_batches,
            amp_dtype, engram_table, projection,
        )
        return

    # Standard single-config perplexity measurement
    if args.synthetic:
        dataloader = make_synthetic_dataloader(config.vocab_size, seq_len, args.batch_size, args.seed)
    else:
        dataloader = make_hf_dataloader(args.data, seq_len, args.batch_size, args.seed)

    if args.latent_projection is not None and engram_table is not None:
        print(f"\nMeasuring with projection keys ({args.num_batches} batches)...")
        metrics = evaluate_projected(
            model, dataloader, config.vocab_size, device,
            num_batches=args.num_batches,
            engram_table=engram_table,
            projection=projection,
            amp_dtype=amp_dtype,
        )
    else:
        print(f"\nMeasuring {args.num_batches} batches (batch_size={args.batch_size}, seq_len={seq_len})...")
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
