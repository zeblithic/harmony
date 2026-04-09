"""GGUF exporter for ct87 -- converts safetensors checkpoints to GGUF format.

Usage:
    python -m ct87.export_gguf --checkpoint model.safetensors --config tiny --output model.gguf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from gguf import GGUFWriter
from safetensors.torch import load_file

from ct87.model import HarmonyModelConfig


def build_naming_map(config: HarmonyModelConfig) -> dict[str, str]:
    """Build the PyTorch state_dict key -> GGUF tensor name mapping.

    Returns a dict where keys are PyTorch state_dict keys and values are
    the corresponding GGUF tensor names. When tie_embeddings is True,
    lm_head.weight is excluded (the loader falls back to token_embd.weight).
    """
    mapping: dict[str, str] = {}
    mapping["embed_tokens.weight"] = "token_embd.weight"

    for i in range(config.num_layers):
        mapping[f"layers.{i}.attn_norm.weight"] = f"blk.{i}.attn_norm.weight"
        mapping[f"layers.{i}.attn.q_proj.weight"] = f"blk.{i}.attn_q.weight"
        mapping[f"layers.{i}.attn.k_proj.weight"] = f"blk.{i}.attn_k.weight"
        mapping[f"layers.{i}.attn.v_proj.weight"] = f"blk.{i}.attn_v.weight"
        mapping[f"layers.{i}.attn.o_proj.weight"] = f"blk.{i}.attn_output.weight"
        mapping[f"layers.{i}.attn.q_norm.weight"] = f"blk.{i}.attn_q_norm.weight"
        mapping[f"layers.{i}.attn.k_norm.weight"] = f"blk.{i}.attn_k_norm.weight"
        mapping[f"layers.{i}.ffn_norm.weight"] = f"blk.{i}.ffn_norm.weight"
        mapping[f"layers.{i}.mlp.gate_proj.weight"] = f"blk.{i}.ffn_gate.weight"
        mapping[f"layers.{i}.mlp.up_proj.weight"] = f"blk.{i}.ffn_up.weight"
        mapping[f"layers.{i}.mlp.down_proj.weight"] = f"blk.{i}.ffn_down.weight"

    mapping["final_norm.weight"] = "output_norm.weight"

    if not config.tie_embeddings:
        mapping["lm_head.weight"] = "output.weight"

    for i in range(config.num_blocks - 1):
        mapping[f"block_attnres.queries.{i}"] = (
            f"harmony.block_attnres.query.{i}.weight"
        )

    # Engram gated residual weights
    mapping["engram_residual.key_proj.weight"] = "harmony.engram_residual.key_proj.weight"
    mapping["engram_residual.value_proj.weight"] = "harmony.engram_residual.value_proj.weight"
    mapping["engram_residual.gate_norm.weight"] = "harmony.engram_residual.gate_norm.weight"
    mapping["engram_residual.key_norm.weight"] = "harmony.engram_residual.key_norm.weight"
    mapping["engram_residual.conv1d.weight"] = "harmony.engram_residual.conv1d.weight"

    return mapping


def build_thought_norm_map() -> dict[str, str]:
    """Build naming map for ThoughtNorm weights (COCONUT continuous thought)."""
    return {
        "norm.weight": "harmony.continuous_thought.norm.weight",
        "gate_bias": "harmony.continuous_thought.gate_bias",
    }


def build_uq_head_map() -> dict[str, str]:
    """Build naming map for UQ head weights (uncertainty quantification).

    Weight tensors are transposed during export to match the Rust UqHead's
    ``[in_features, out_features]`` matmul convention (``features.matmul(&weight)``).
    PyTorch stores ``[out_features, in_features]``.
    """
    return {
        "classifier_fc1.weight": "harmony.uq.classifier.fc1.weight",
        "classifier_fc1.bias": "harmony.uq.classifier.fc1.bias",
        "classifier_fc2.weight": "harmony.uq.classifier.fc2.weight",
        "classifier_fc2.bias": "harmony.uq.classifier.fc2.bias",
        "confidence_linear.weight": "harmony.uq.confidence.weight",
        "confidence_linear.bias": "harmony.uq.confidence.bias",
    }


def write_metadata(
    writer: GGUFWriter, config: HarmonyModelConfig, name: str,
) -> None:
    """Write all harmony GGUF metadata keys."""
    writer.add_name(name)
    writer.add_uint32("harmony.block_count", config.num_layers)
    writer.add_uint32("harmony.embedding_length", config.hidden_dim)
    writer.add_uint32("harmony.attention.head_count", config.num_query_heads)
    writer.add_uint32("harmony.attention.head_count_kv", config.num_kv_heads)
    writer.add_uint32("harmony.attention.key_length", config.head_dim)
    writer.add_uint32("harmony.feed_forward_length", config.ffn_dim)
    writer.add_uint32("harmony.context_length", config.max_seq_len)
    # Explicit float() downcast: these f64 config values are stored as f32 in
    # GGUF metadata. The precision loss is negligible (~3e-13 for rms_norm_eps)
    # but means a GGUF-loaded model isn't bit-identical to the original config.
    writer.add_float32("harmony.rope.freq_base", float(config.rope_theta))
    writer.add_float32(
        "harmony.attention.layer_norm_rms_epsilon", float(config.rms_norm_eps),
    )
    writer.add_uint32("harmony.vocab_size", config.vocab_size)
    writer.add_uint32("harmony.layers_per_block", config.layers_per_block)
    writer.add_bool("harmony.tie_embeddings", config.tie_embeddings)
    writer.add_uint32(
        "harmony.engram_injection_layer", config.engram_injection_layer,
    )
    writer.add_uint32("harmony.engram_dim", config.engram_dim)

    # Continuous thought metadata (optional — only written when enabled).
    if config.think_token_id is not None:
        if config.think_token_id >= config.vocab_size:
            raise ValueError(
                f"think_token_id ({config.think_token_id}) must be "
                f"< vocab_size ({config.vocab_size})"
            )
        writer.add_bool("harmony.continuous_thought.enabled", True)
        writer.add_uint32(
            "harmony.continuous_thought.think_token_id", config.think_token_id,
        )
        max_steps = config.ct_max_steps if config.ct_max_steps is not None else 4
        threshold = (
            config.ct_confidence_threshold
            if config.ct_confidence_threshold is not None
            else 0.85
        )
        writer.add_uint32("harmony.continuous_thought.max_steps", max_steps)
        writer.add_float32(
            "harmony.continuous_thought.confidence_threshold", threshold,
        )


def export_gguf(
    state_dict: dict[str, torch.Tensor],
    config: HarmonyModelConfig,
    output_path: str | Path,
    name: str | None = None,
    thought_norm_state: dict[str, torch.Tensor] | None = None,
    uq_head_state: dict[str, torch.Tensor] | None = None,
) -> None:
    """Export a ct87 state_dict to GGUF format.

    Args:
        state_dict: Model state_dict (from safetensors or model.state_dict()).
        config: Model configuration matching the checkpoint.
        output_path: Path to write the GGUF file.
        name: Optional model name for metadata. Defaults to "ct87".
        thought_norm_state: Optional ThoughtNorm state_dict for COCONUT
            continuous thought. Only provided for CT-trained models.
        uq_head_state: Optional UQ head state_dict for uncertainty
            quantification. Only provided for UQ-trained models.

    Raises:
        ValueError: If state_dict keys don't match expected keys for config.
    """
    if name is None:
        name = "ct87"

    naming_map = build_naming_map(config)

    # Validate completeness
    expected_keys = set(naming_map.keys())
    actual_keys = set(state_dict.keys())

    # When tied, lm_head.weight exists in state_dict but isn't in naming_map
    if config.tie_embeddings and "lm_head.weight" in actual_keys:
        actual_keys = actual_keys - {"lm_head.weight"}

    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"Missing keys: {sorted(missing)}")
        if extra:
            parts.append(f"Extra keys: {sorted(extra)}")
        raise ValueError("; ".join(parts))

    writer = GGUFWriter(str(output_path), arch="harmony")
    write_metadata(writer, config, name)

    for pytorch_key, gguf_name in naming_map.items():
        tensor = state_dict[pytorch_key]
        # BlockAttnRes queries: squeeze from [1, 1, hidden] to [hidden]
        if "block_attnres.queries" in pytorch_key:
            tensor = tensor.squeeze()
        arr = tensor.detach().cpu().float().numpy()
        writer.add_tensor(gguf_name, arr)

    # ThoughtNorm weights (COCONUT continuous thought)
    if thought_norm_state is not None:
        tn_map = build_thought_norm_map()
        expected_tn = set(tn_map.keys())
        actual_tn = set(thought_norm_state.keys())
        if expected_tn != actual_tn:
            raise ValueError(
                f"ThoughtNorm state_dict mismatch. "
                f"Expected: {sorted(expected_tn)}, got: {sorted(actual_tn)}"
            )
        for pytorch_key, gguf_name in tn_map.items():
            t = thought_norm_state[pytorch_key].detach().cpu().float()
            # GGUF requires >= 1 dimension; gate_bias is a 0-d scalar
            arr = t.numpy() if t.ndim > 0 else t.unsqueeze(0).numpy()
            writer.add_tensor(gguf_name, arr)

    # UQ head weights (uncertainty quantification)
    if uq_head_state is not None:
        uq_map = build_uq_head_map()
        expected_uq = set(uq_map.keys())
        actual_uq = set(uq_head_state.keys())
        if expected_uq != actual_uq:
            raise ValueError(
                f"UQ head state_dict mismatch. "
                f"Expected: {sorted(expected_uq)}, got: {sorted(actual_uq)}"
            )
        writer.add_bool("harmony.uq.enabled", True)
        writer.add_uint32("harmony.uq.num_features", 8)
        writer.add_uint32("harmony.uq.hidden_dim", 32)
        writer.add_uint32("harmony.uq.num_classes", 4)
        for pytorch_key, gguf_name in uq_map.items():
            t = uq_head_state[pytorch_key].detach().cpu().float()
            # Transpose 2D weights: PyTorch [out, in] -> Rust [in, out]
            if t.ndim == 2:
                t = t.T
            arr = t.numpy()
            writer.add_tensor(gguf_name, arr)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ct87 checkpoint to GGUF")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to safetensors checkpoint",
    )
    parser.add_argument("--config", choices=["tiny", "target"], required=True)
    parser.add_argument("--output", type=str, required=True, help="Output GGUF path")
    parser.add_argument(
        "--name", type=str, default=None, help="Model name for metadata",
    )
    parser.add_argument(
        "--thought-norm", type=str, default=None,
        help="Path to ThoughtNorm checkpoint (thought_norm_step_*.pt) for COCONUT exports",
    )
    parser.add_argument(
        "--think-token-id", type=int, default=None,
        help="Token ID for <think> (required with --thought-norm)",
    )
    parser.add_argument(
        "--ct-max-steps", type=int, default=4,
        help="Max continuous thought steps (default: 4)",
    )
    parser.add_argument(
        "--ct-confidence-threshold", type=float, default=0.85,
        help="Confidence threshold for continuous thought (default: 0.85)",
    )
    parser.add_argument(
        "--uq-head", type=str, default=None,
        help="Path to UQ head checkpoint (uq_head_step_*.pt) for export",
    )
    args = parser.parse_args()

    if args.thought_norm is not None and args.think_token_id is None:
        parser.error("--think-token-id is required when --thought-norm is provided")
    if args.think_token_id is not None and args.thought_norm is None:
        parser.error("--thought-norm is required when --think-token-id is provided")

    config = (
        HarmonyModelConfig.tiny()
        if args.config == "tiny"
        else HarmonyModelConfig.target()
    )

    # Set CT config fields when exporting with continuous thought
    if args.think_token_id is not None:
        config.think_token_id = args.think_token_id
        config.ct_max_steps = args.ct_max_steps
        config.ct_confidence_threshold = args.ct_confidence_threshold

    state_dict = load_file(args.checkpoint)
    name = args.name or f"ct87-{args.config}"

    thought_norm_state = None
    if args.thought_norm is not None:
        thought_norm_state = torch.load(args.thought_norm, map_location="cpu", weights_only=True)

    uq_head_state = None
    if args.uq_head is not None:
        uq_head_state = torch.load(args.uq_head, map_location="cpu", weights_only=True)

    export_gguf(
        state_dict, config, args.output, name,
        thought_norm_state=thought_norm_state,
        uq_head_state=uq_head_state,
    )
    print(f"Exported to {args.output}")


if __name__ == "__main__":
    main()
