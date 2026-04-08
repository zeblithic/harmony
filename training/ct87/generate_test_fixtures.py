"""Generate GGUF test fixtures for Rust integration tests.

Run from the training/ directory:
    python -m ct87.generate_test_fixtures
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import torch

from ct87.export_gguf import export_gguf
from ct87.model import HarmonyModel, HarmonyModelConfig

# Micro config matching test_config() in harmony_model.rs
MICRO_CONFIG = HarmonyModelConfig(
    num_layers=4,
    hidden_dim=32,
    num_query_heads=4,
    num_kv_heads=2,
    head_dim=8,
    ffn_dim=64,
    vocab_size=128,
    max_seq_len=64,
    rope_theta=10000.0,
    rms_norm_eps=1e-6,
    layers_per_block=2,
    engram_injection_layer=1,
    engram_dim=16,
    tie_embeddings=True,
)

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "crates" / "harmony-inference" / "tests" / "fixtures"


def generate(tie: bool, filename: str, name: str) -> None:
    torch.manual_seed(42)
    config = replace(MICRO_CONFIG, tie_embeddings=tie)
    model = HarmonyModel(config)
    export_gguf(model.state_dict(), config, FIXTURE_DIR / filename, name)
    print(f"Generated {FIXTURE_DIR / filename}")


def main() -> None:
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    generate(tie=True, filename="tiny_harmony.gguf", name="test-tied")
    generate(tie=False, filename="tiny_harmony_untied.gguf", name="test-untied")
    print("Done!")


if __name__ == "__main__":
    main()
