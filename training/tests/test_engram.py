"""Tests for Engram gated residual injection and table lookup."""

import struct
import tempfile
from pathlib import Path

import numpy as np
import torch
import pytest

from ct87.model import HarmonyModel, HarmonyModelConfig
from ct87.engram import EngramGatedResidual, EngramTable, _hash_ngram, _xxhash64


def _tiny_config() -> HarmonyModelConfig:
    """Shared tiny config for Engram tests."""
    return HarmonyModelConfig(
        num_layers=4, hidden_dim=32, num_query_heads=4, num_kv_heads=2,
        head_dim=8, ffn_dim=64, vocab_size=128, max_seq_len=64,
        rope_theta=10000.0, rms_norm_eps=1e-6, layers_per_block=2,
        engram_injection_layer=1, engram_dim=16, tie_embeddings=True,
    )


# ---------------------------------------------------------------------------
# xxhash64 parity tests
# ---------------------------------------------------------------------------


class TestXxhash64:
    def test_parity_with_rust_seed_42(self):
        """Must match Rust: hash_ngram(&[1, 2, 3], 42) = 11547749587308120431."""
        assert _hash_ngram([1, 2, 3], 42) == 11547749587308120431

    def test_parity_with_rust_seed_99(self):
        """Must match Rust: hash_ngram(&[1, 2, 3], 99) = 469971568748895552."""
        assert _hash_ngram([1, 2, 3], 99) == 469971568748895552

    def test_determinism(self):
        h1 = _hash_ngram([10, 20, 30], 42)
        h2 = _hash_ngram([10, 20, 30], 42)
        assert h1 == h2

    def test_different_seeds_differ(self):
        h1 = _hash_ngram([1, 2], 42)
        h2 = _hash_ngram([1, 2], 99)
        assert h1 != h2

    def test_different_tokens_differ(self):
        h1 = _hash_ngram([1, 2], 42)
        h2 = _hash_ngram([3, 4], 42)
        assert h1 != h2

    def test_empty_tokens(self):
        # Must not crash; result is just the seed-based hash of empty input
        h = _hash_ngram([], 42)
        assert isinstance(h, int)


# ---------------------------------------------------------------------------
# EngramGatedResidual tests
# ---------------------------------------------------------------------------


class TestEngramGatedResidual:
    def test_output_shape(self):
        cfg = _tiny_config()
        module = EngramGatedResidual(cfg)
        h = torch.randn(2, 10, cfg.hidden_dim)
        e = torch.randn(2, 10, cfg.engram_dim)
        out = module(h, e)
        assert out.shape == h.shape

    def test_zero_engram_returns_zero_residual(self):
        cfg = _tiny_config()
        module = EngramGatedResidual(cfg)
        h = torch.randn(1, 5, cfg.hidden_dim)
        e = torch.zeros(1, 5, cfg.engram_dim)
        out = module(h, e)
        # Zero engram → zero projections → zero gated value → zero conv → silu(0) = 0
        assert out.abs().max().item() < 1e-6

    def test_single_token(self):
        cfg = _tiny_config()
        module = EngramGatedResidual(cfg)
        h = torch.randn(1, 1, cfg.hidden_dim)
        e = torch.randn(1, 1, cfg.engram_dim)
        out = module(h, e)
        assert out.shape == (1, 1, cfg.hidden_dim)

    def test_causal_conv_no_future_leakage(self):
        """Changing future engram positions must not affect earlier positions."""
        cfg = _tiny_config()
        module = EngramGatedResidual(cfg)
        # Initialize conv1d with non-zero weights for a real test
        torch.nn.init.ones_(module.conv1d.weight)
        module.eval()

        h = torch.randn(1, 6, cfg.hidden_dim)
        e = torch.randn(1, 6, cfg.engram_dim)
        e_modified = e.clone()
        e_modified[0, 4:, :] = torch.randn(2, cfg.engram_dim)

        with torch.no_grad():
            out1 = module(h, e)
            out2 = module(h, e_modified)
        assert torch.allclose(out1[0, :4], out2[0, :4], atol=1e-5)


# ---------------------------------------------------------------------------
# EngramTable tests
# ---------------------------------------------------------------------------


class TestEngramTable:
    def _make_table(self, entries: int = 100, dim: int = 16) -> EngramTable:
        """Create a table with deterministic non-zero values."""
        table = torch.randn(entries, dim)
        return EngramTable(table, hash_seeds=[42, 99])

    def test_lookup_batch_shape(self):
        tbl = self._make_table()
        input_ids = torch.randint(0, 128, (2, 10))
        out = tbl.lookup_batch(input_ids)
        assert out.shape == (2, 10, 16)

    def test_position_zero_is_zero(self):
        """Position 0 has no N-gram coverage → zero embedding."""
        tbl = self._make_table()
        input_ids = torch.randint(0, 128, (1, 5))
        out = tbl.lookup_batch(input_ids)
        assert out[0, 0].abs().max().item() < 1e-6

    def test_later_positions_nonzero(self):
        """Positions 1+ should have non-zero embeddings from N-gram lookups."""
        tbl = self._make_table()
        input_ids = torch.randint(0, 128, (1, 5))
        out = tbl.lookup_batch(input_ids)
        for pos in range(1, 5):
            assert out[0, pos].abs().max().item() > 1e-6

    def test_determinism(self):
        tbl = self._make_table()
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        out1 = tbl.lookup_batch(input_ids)
        out2 = tbl.lookup_batch(input_ids)
        assert torch.allclose(out1, out2)

    def test_different_tokens_differ(self):
        tbl = self._make_table()
        a = torch.tensor([[1, 2, 3]])
        b = torch.tensor([[10, 20, 30]])
        out_a = tbl.lookup_batch(a)
        out_b = tbl.lookup_batch(b)
        # Position 1 (first bigram) should differ
        assert not torch.allclose(out_a[0, 1], out_b[0, 1])

    def test_single_token_all_zero(self):
        tbl = self._make_table()
        input_ids = torch.tensor([[42]])
        out = tbl.lookup_batch(input_ids)
        assert out.abs().max().item() < 1e-6

    def test_from_safetensors(self):
        """Load from a safetensors file."""
        from safetensors.numpy import save_file

        table_data = np.random.randn(50, 8).astype(np.float16)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.safetensors"
            save_file({"engram.weight": table_data}, str(path))
            tbl = EngramTable.from_safetensors(path, hash_seeds=[42])
            assert tbl.total_entries == 50
            assert tbl.engram_dim == 8


# ---------------------------------------------------------------------------
# Model integration tests
# ---------------------------------------------------------------------------


class TestModelWithEngram:
    def test_forward_with_engram_shape(self):
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 5))
        engram = torch.randn(2, 5, cfg.engram_dim)
        logits = model(input_ids, engram_embeddings=engram)
        assert logits.shape == (2, 5, cfg.vocab_size)

    def test_forward_without_engram_shape(self):
        """Backward compatible: None engram_embeddings still works."""
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 5))
        logits = model(input_ids)
        assert logits.shape == (2, 5, cfg.vocab_size)

    def test_engram_injection_changes_output(self):
        """Model with non-zero Engram input should differ from without."""
        cfg = _tiny_config()
        torch.manual_seed(42)
        model = HarmonyModel(cfg)
        # Initialize conv1d non-zero so injection has effect
        torch.nn.init.ones_(model.engram_residual.conv1d.weight)
        model.eval()

        input_ids = torch.randint(0, cfg.vocab_size, (1, 5))
        engram = torch.randn(1, 5, cfg.engram_dim)

        with torch.no_grad():
            logits_no_engram = model(input_ids)
            logits_with_engram = model(input_ids, engram_embeddings=engram)

        assert not torch.allclose(logits_no_engram, logits_with_engram, atol=1e-5)

    def test_zero_engram_matches_no_engram(self):
        """Zero Engram embeddings should produce identical output to no Engram."""
        cfg = _tiny_config()
        torch.manual_seed(42)
        model = HarmonyModel(cfg)
        model.eval()

        input_ids = torch.randint(0, cfg.vocab_size, (1, 5))
        zero_engram = torch.zeros(1, 5, cfg.engram_dim)

        with torch.no_grad():
            logits_none = model(input_ids)
            logits_zero = model(input_ids, engram_embeddings=zero_engram)

        assert torch.allclose(logits_none, logits_zero, atol=1e-6)

    def test_engram_residual_weights_in_state_dict(self):
        """EngramGatedResidual weights should appear in state_dict."""
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        sd = model.state_dict()
        expected_keys = [
            "engram_residual.key_proj.weight",
            "engram_residual.value_proj.weight",
            "engram_residual.gate_norm.weight",
            "engram_residual.key_norm.weight",
            "engram_residual.conv1d.weight",
        ]
        for key in expected_keys:
            assert key in sd, f"Missing state_dict key: {key}"


# ---------------------------------------------------------------------------
# GGUF export tests
# ---------------------------------------------------------------------------


class TestGgufExportWithEngram:
    def test_naming_map_includes_engram_weights(self):
        from ct87.export_gguf import build_naming_map
        cfg = _tiny_config()
        nm = build_naming_map(cfg)
        expected = [
            "engram_residual.key_proj.weight",
            "engram_residual.value_proj.weight",
            "engram_residual.gate_norm.weight",
            "engram_residual.key_norm.weight",
            "engram_residual.conv1d.weight",
        ]
        for key in expected:
            assert key in nm, f"Missing naming map key: {key}"
            assert nm[key].startswith("harmony.engram_residual.")

    def test_export_roundtrip_with_engram(self):
        """Export model with Engram weights and verify they're in the GGUF."""
        from gguf import GGUFReader
        from ct87.export_gguf import export_gguf

        cfg = _tiny_config()
        torch.manual_seed(42)
        model = HarmonyModel(cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gguf"
            export_gguf(model.state_dict(), cfg, path)

            reader = GGUFReader(str(path))
            names = {t.name for t in reader.tensors}
            assert "harmony.engram_residual.key_proj.weight" in names
            assert "harmony.engram_residual.value_proj.weight" in names
            assert "harmony.engram_residual.gate_norm.weight" in names
            assert "harmony.engram_residual.key_norm.weight" in names
            assert "harmony.engram_residual.conv1d.weight" in names
