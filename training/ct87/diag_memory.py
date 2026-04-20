"""Memory-leak diagnostic for ct87.prepare_data.

Runs four scenarios in sequence, each for a bounded number of documents,
and prints RSS + key counters every 10k docs. This tells us whether the
leak lives in:
  A. HF Datasets streaming (text iteration only)
  B. Tokenizer (encode each text)
  C. array.array accumulation (the path prepare_data takes)
  D. Same as C but with malloc_trim(0) every 10k docs

Usage: python3 -m ct87.diag_memory [docs_per_scenario]
"""

from __future__ import annotations

import array
import ctypes
import gc
import os
import sys


def rss_kb() -> int:
    """Current RSS in KB via /proc. This diagnostic targets the tokenizer +
    pyarrow heap-growth regime on Linux/WSL; /proc is the cleanest source of
    *current* (not peak) RSS, and the leak patterns we're investigating
    are glibc-specific anyway. Fail loudly on platforms without /proc rather
    than silently fall back to resource.getrusage.ru_maxrss (which is peak,
    not current — a confusing semantic shift for a memory-growth diagnostic)."""
    status_path = f"/proc/{os.getpid()}/status"
    if not os.path.exists(status_path):
        raise RuntimeError(
            "ct87.diag_memory requires /proc (Linux/WSL). "
            "The leak patterns it reproduces are glibc-specific and the "
            "script has no meaningful fallback on other platforms."
        )
    with open(status_path) as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return -1


# malloc_trim is glibc-specific — fall back to a no-op on macOS/musl so this
# diagnostic can still run there (it just won't exercise the arena-release
# path). pyarrow pool release below is cross-platform.
try:
    _libc = ctypes.CDLL("libc.so.6", use_errno=True)
    _libc.malloc_trim.argtypes = [ctypes.c_int]
    _libc.malloc_trim.restype = ctypes.c_int

    def malloc_trim() -> None:
        _libc.malloc_trim(0)
except (OSError, AttributeError):
    def malloc_trim() -> None:
        pass


try:
    import pyarrow as _pa
    _pa_pool = _pa.default_memory_pool()
except (ImportError, AttributeError):
    _pa_pool = None


def release_all() -> None:
    """Full drain: gc + pyarrow pool + glibc arena trim. Mirrors the path
    prepare_data.py uses during streaming tokenization."""
    gc.collect()
    if _pa_pool is not None:
        _pa_pool.release_unused()
    malloc_trim()


def report(label: str, n: int, tokens: int, t_start: float) -> None:
    import time
    dt = time.time() - t_start
    print(
        f"  {label} n={n:>7,} tokens={tokens:>12,} "
        f"rss={rss_kb()/1024/1024:>6.2f}GB "
        f"rate={int(n/max(dt, 0.01)):>6,}d/s"
    )


def scenario_a_text_only(ds, n_docs: int) -> None:
    import time
    print("\n=== A: iterate text only (no tokenizer, no accumulate) ===")
    print(f"  baseline rss={rss_kb()/1024/1024:.2f}GB")
    t0 = time.time()
    n = 0
    total_chars = 0
    for example in ds:
        # Skip blanks to match the filtering in scenarios B/C/D — otherwise
        # n_docs counts different workloads across scenarios and the RSS
        # comparison drifts with how many empty rows happen to land in the
        # first n_docs of the stream.
        text = example["text"]
        if not text or not text.strip():
            continue
        total_chars += len(text)
        n += 1
        if n % 10_000 == 0:
            report("A", n, total_chars, t0)
        if n >= n_docs:
            break
    print(f"  final rss={rss_kb()/1024/1024:.2f}GB")


def scenario_b_tokenize_only(ds, tokenizer, n_docs: int) -> None:
    import time
    print("\n=== B: tokenize each text, discard result ===")
    print(f"  baseline rss={rss_kb()/1024/1024:.2f}GB")
    t0 = time.time()
    n = 0
    total_tokens = 0
    for example in ds:
        text = example["text"]
        if not text or not text.strip():
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False)
        total_tokens += len(tokens)
        n += 1
        if n % 10_000 == 0:
            report("B", n, total_tokens, t0)
        if n >= n_docs:
            break
    print(f"  final rss={rss_kb()/1024/1024:.2f}GB")


def scenario_c_accumulate(ds, tokenizer, eos, n_docs: int) -> None:
    import time
    print("\n=== C: tokenize + array.array accumulate (the current path) ===")
    print(f"  baseline rss={rss_kb()/1024/1024:.2f}GB")
    t0 = time.time()
    stream = array.array("H")
    n = 0
    total_tokens = 0
    for example in ds:
        text = example["text"]
        if not text or not text.strip():
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False)
        stream.extend(tokens)
        stream.append(eos)
        total_tokens += len(tokens)
        n += 1
        if n % 10_000 == 0:
            report("C", n, total_tokens, t0)
        if n >= n_docs:
            break
    print(f"  stream len={len(stream):,} bytes={len(stream)*2:,} final rss={rss_kb()/1024/1024:.2f}GB")


def scenario_d_accumulate_trim(ds, tokenizer, eos, n_docs: int) -> None:
    """Same as C, but runs the same full drain prepare_data.py uses
    (gc + pyarrow pool + glibc malloc_trim) every 10k docs."""
    import time
    print("\n=== D: same as C, with release_all() every 10k docs ===")
    print(f"  baseline rss={rss_kb()/1024/1024:.2f}GB")
    t0 = time.time()
    stream = array.array("H")
    n = 0
    total_tokens = 0
    for example in ds:
        text = example["text"]
        if not text or not text.strip():
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False)
        stream.extend(tokens)
        stream.append(eos)
        total_tokens += len(tokens)
        n += 1
        if n % 10_000 == 0:
            report("D", n, total_tokens, t0)
            release_all()
            print(f"    after release_all rss={rss_kb()/1024/1024:.2f}GB")
        if n >= n_docs:
            break
    print(f"  stream len={len(stream):,} final rss={rss_kb()/1024/1024:.2f}GB")


def main() -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    n_docs = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000

    print(f"pid={os.getpid()} docs_per_scenario={n_docs}")
    print(f"initial rss={rss_kb()/1024/1024:.2f}GB")

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    eos = tokenizer.eos_token_id
    print(f"after tokenizer load rss={rss_kb()/1024/1024:.2f}GB")

    # Fresh streaming iterator for each scenario
    def fresh():
        return load_dataset(
            "HuggingFaceFW/fineweb-edu-score-2",
            split="train",
            streaming=True,
        )

    scenario_a_text_only(fresh(), n_docs)
    release_all()
    print(f"between scenarios (post release_all) rss={rss_kb()/1024/1024:.2f}GB")

    scenario_b_tokenize_only(fresh(), tokenizer, n_docs)
    release_all()
    print(f"between scenarios (post release_all) rss={rss_kb()/1024/1024:.2f}GB")

    scenario_c_accumulate(fresh(), tokenizer, eos, n_docs)
    release_all()
    print(f"between scenarios (post release_all) rss={rss_kb()/1024/1024:.2f}GB")

    scenario_d_accumulate_trim(fresh(), tokenizer, eos, n_docs)


if __name__ == "__main__":
    main()
