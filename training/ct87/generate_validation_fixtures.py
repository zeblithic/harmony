"""Generate validation fixtures for Phase 0h candle validation loop.

Creates a deterministic tiny model, exports to GGUF, runs a forward pass,
and writes reference logits as JSON. The Rust integration test in
crates/harmony-inference/tests/validate_candle.rs loads these fixtures
and verifies candle produces identical logits.

Run from the training/ directory:
    python3 -m ct87.generate_validation_fixtures
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from ct87.export_gguf import export_gguf
from ct87.model import HarmonyModel, HarmonyModelConfig

# Micro config matching test_config() in harmony_model.rs and MICRO_CONFIG
# in generate_test_fixtures.py.
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

# Hardcoded input tokens -- all within vocab_size=128, chosen for
# inspectability and reproducibility without seed management.
INPUT_TOKENS = [1, 5, 12, 3, 7, 42, 18, 100]

FIXTURE_DIR = (
    Path(__file__).resolve().parents[2]
    / "crates"
    / "harmony-inference"
    / "tests"
    / "fixtures"
)


def main() -> None:
    os.makedirs(FIXTURE_DIR, exist_ok=True)

    # Deterministic model construction
    torch.manual_seed(42)
    model = HarmonyModel(MICRO_CONFIG)

    # Export GGUF
    gguf_path = FIXTURE_DIR / "validation_model.gguf"
    export_gguf(model.state_dict(), MICRO_CONFIG, gguf_path, "validation-micro")
    print(f"Generated {gguf_path}")

    # Forward pass -- extract last-position logits
    model.eval()
    input_ids = torch.tensor([INPUT_TOKENS], dtype=torch.long)  # [1, 8]
    with torch.no_grad():
        logits = model(input_ids)  # [1, 8, 128]
    last_logits = logits[0, -1, :].tolist()  # 128 float values

    # Write reference JSON -- config fields included as a staleness guard
    # so the Rust test can detect if fixtures are out of sync with the model.
    reference = {
        "input_tokens": INPUT_TOKENS,
        "last_logits": last_logits,
        "config": {
            "num_layers": MICRO_CONFIG.num_layers,
            "hidden_dim": MICRO_CONFIG.hidden_dim,
            "vocab_size": MICRO_CONFIG.vocab_size,
            "layers_per_block": MICRO_CONFIG.layers_per_block,
        },
    }
    json_path = FIXTURE_DIR / "validation_reference.json"
    with open(json_path, "w") as f:
        json.dump(reference, f, indent=2)
        f.write("\n")
    print(f"Generated {json_path}")
    print(f"Logit vector length: {len(last_logits)}")
    print("Done!")


if __name__ == "__main__":
    main()
