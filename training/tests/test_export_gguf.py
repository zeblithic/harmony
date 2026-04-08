"""Tests for the GGUF exporter."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from ct87.export_gguf import build_naming_map, export_gguf
from ct87.model import HarmonyModel, HarmonyModelConfig


def _test_config(tie: bool = True) -> HarmonyModelConfig:
    """Micro config matching Rust test_config() in harmony_model.rs."""
    return HarmonyModelConfig(
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
        tie_embeddings=tie,
    )


# ---- Naming map tests ----


class TestBuildNamingMap:
    def test_tied_excludes_output_weight(self):
        config = _test_config(tie=True)
        nm = build_naming_map(config)
        assert "lm_head.weight" not in nm
        assert "output.weight" not in nm.values()

    def test_untied_includes_output_weight(self):
        config = _test_config(tie=False)
        nm = build_naming_map(config)
        assert "lm_head.weight" in nm
        assert nm["lm_head.weight"] == "output.weight"

    def test_layer_count_matches(self):
        config = _test_config()
        nm = build_naming_map(config)
        layer_keys = [k for k in nm if k.startswith("layers.")]
        # 11 tensor keys per transformer layer
        assert len(layer_keys) == config.num_layers * 11

    def test_block_attnres_query_count(self):
        config = _test_config()
        nm = build_naming_map(config)
        query_keys = [k for k in nm if "block_attnres" in k]
        # 2 blocks -> 1 boundary query
        assert len(query_keys) == config.num_blocks - 1

    def test_embed_and_norm_present(self):
        config = _test_config()
        nm = build_naming_map(config)
        assert nm["embed_tokens.weight"] == "token_embd.weight"
        assert nm["final_norm.weight"] == "output_norm.weight"


# ---- Export tests ----


class TestExportGguf:
    def test_roundtrip_tensor_values(self):
        """Export model to GGUF and read back -- verify tensor values match."""
        from gguf import GGUFReader

        config = _test_config(tie=True)
        torch.manual_seed(42)
        model = HarmonyModel(config)
        state_dict = model.state_dict()

        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = Path(tmpdir) / "test.gguf"
            export_gguf(state_dict, config, gguf_path, name="test")

            reader = GGUFReader(str(gguf_path))
            gguf_tensors = {t.name: t for t in reader.tensors}

            naming_map = build_naming_map(config)
            for pytorch_key, gguf_name in naming_map.items():
                assert gguf_name in gguf_tensors, f"Missing: {gguf_name}"
                original = state_dict[pytorch_key].detach().float().numpy()
                if "block_attnres.queries" in pytorch_key:
                    original = original.squeeze()
                loaded = np.array(gguf_tensors[gguf_name].data)
                np.testing.assert_allclose(
                    loaded.reshape(original.shape),
                    original,
                    rtol=1e-6,
                    err_msg=f"Mismatch for {gguf_name}",
                )

    def test_metadata_present(self):
        """Verify all metadata keys are present."""
        from gguf import GGUFReader

        config = _test_config()
        torch.manual_seed(42)
        model = HarmonyModel(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = Path(tmpdir) / "test.gguf"
            export_gguf(model.state_dict(), config, gguf_path, name="test-model")

            reader = GGUFReader(str(gguf_path))
            fields = reader.fields

            expected_keys = [
                "general.architecture",
                "general.name",
                "harmony.block_count",
                "harmony.embedding_length",
                "harmony.attention.head_count",
                "harmony.attention.head_count_kv",
                "harmony.attention.key_length",
                "harmony.feed_forward_length",
                "harmony.context_length",
                "harmony.rope.freq_base",
                "harmony.attention.layer_norm_rms_epsilon",
                "harmony.vocab_size",
                "harmony.layers_per_block",
                "harmony.tie_embeddings",
                "harmony.engram_injection_layer",
                "harmony.engram_dim",
            ]
            for key in expected_keys:
                assert key in fields, f"Missing metadata key: {key}"

    def test_tied_skips_output_weight(self):
        """When tie_embeddings=True, output.weight absent from GGUF."""
        from gguf import GGUFReader

        config = _test_config(tie=True)
        torch.manual_seed(42)
        model = HarmonyModel(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = Path(tmpdir) / "test.gguf"
            export_gguf(model.state_dict(), config, gguf_path)

            reader = GGUFReader(str(gguf_path))
            names = {t.name for t in reader.tensors}
            assert "output.weight" not in names
            assert "token_embd.weight" in names

    def test_untied_includes_output_weight(self):
        """When tie_embeddings=False, output.weight present in GGUF."""
        from gguf import GGUFReader

        config = _test_config(tie=False)
        torch.manual_seed(42)
        model = HarmonyModel(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = Path(tmpdir) / "test.gguf"
            export_gguf(model.state_dict(), config, gguf_path)

            reader = GGUFReader(str(gguf_path))
            names = {t.name for t in reader.tensors}
            assert "output.weight" in names
            assert "token_embd.weight" in names

    def test_missing_keys_raises(self):
        """Exporter rejects state_dict with missing keys."""
        config = _test_config()
        state_dict = {"embed_tokens.weight": torch.zeros(128, 32)}

        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = Path(tmpdir) / "test.gguf"
            with pytest.raises(ValueError, match="Missing keys"):
                export_gguf(state_dict, config, gguf_path)

    def test_extra_keys_raises(self):
        """Exporter rejects state_dict with extra keys."""
        config = _test_config()
        torch.manual_seed(42)
        model = HarmonyModel(config)
        state_dict = model.state_dict()
        state_dict["spurious.weight"] = torch.zeros(10)

        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = Path(tmpdir) / "test.gguf"
            with pytest.raises(ValueError, match="Extra keys"):
                export_gguf(state_dict, config, gguf_path)

    def test_continuous_thought_metadata_absent_when_disabled(self):
        """No continuous thought metadata when think_token_id is None."""
        from gguf import GGUFReader

        config = _test_config()
        assert config.think_token_id is None
        torch.manual_seed(42)
        model = HarmonyModel(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = Path(tmpdir) / "test.gguf"
            export_gguf(model.state_dict(), config, gguf_path)

            reader = GGUFReader(str(gguf_path))
            fields = reader.fields
            assert "harmony.continuous_thought.enabled" not in fields
            assert "harmony.continuous_thought.think_token_id" not in fields

    def test_continuous_thought_metadata_present_when_enabled(self):
        """Continuous thought metadata written when think_token_id is set."""
        from gguf import GGUFReader

        config = _test_config()
        config.think_token_id = 127  # within vocab_size=128
        torch.manual_seed(42)
        model = HarmonyModel(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = Path(tmpdir) / "test.gguf"
            export_gguf(model.state_dict(), config, gguf_path)

            reader = GGUFReader(str(gguf_path))
            fields = reader.fields

            ct_keys = [
                "harmony.continuous_thought.enabled",
                "harmony.continuous_thought.think_token_id",
                "harmony.continuous_thought.max_steps",
                "harmony.continuous_thought.confidence_threshold",
            ]
            for key in ct_keys:
                assert key in fields, f"Missing continuous thought key: {key}"
