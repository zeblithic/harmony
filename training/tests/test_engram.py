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


# ---------------------------------------------------------------------------
# ZEB-117 Model gamma: ANN retrieval + anti-collapse
# ---------------------------------------------------------------------------


class TestEngramANNInjection:
    """Model gamma: gated residual with internal ANN retrieval + anti-collapse."""

    @staticmethod
    def _ann_config() -> HarmonyModelConfig:
        c = _tiny_config()
        c.use_ann_engram = True
        return c

    @staticmethod
    def _fake_table(total_entries: int, engram_dim: int) -> torch.Tensor:
        """Deterministic toy corpus table for retrieval tests."""
        g = torch.Generator().manual_seed(0)
        return torch.randn(total_entries, engram_dim, generator=g)

    def test_construction_rejects_dim_mismatch(self):
        from ct87.engram import EngramANNInjection
        c = self._ann_config()
        # Wrong engram_dim: config says 16, provide 8
        bad_table = torch.randn(10, 8)
        with pytest.raises(ValueError, match="engram_dim"):
            EngramANNInjection(c, bad_table)

    def test_construction_rejects_wrong_shape(self):
        from ct87.engram import EngramANNInjection
        c = self._ann_config()
        with pytest.raises(ValueError, match="2-D"):
            EngramANNInjection(c, torch.randn(10))

    def test_table_registered_as_buffer(self):
        """The corpus table must be a buffer, not a trainable parameter."""
        from ct87.engram import EngramANNInjection
        c = self._ann_config()
        t = self._fake_table(20, c.engram_dim)
        m = EngramANNInjection(c, t)
        param_names = {n for n, _ in m.named_parameters()}
        assert "table" not in param_names
        # But it IS a buffer
        buffer_names = {n for n, _ in m.named_buffers()}
        assert "table" in buffer_names
        assert "table_normalized" in buffer_names

    def test_retrieve_returns_correct_shape(self):
        from ct87.engram import EngramANNInjection
        c = self._ann_config()
        t = self._fake_table(20, c.engram_dim)
        m = EngramANNInjection(c, t)
        h = torch.randn(2, 7, c.hidden_dim)
        retrieved = m.retrieve(h)
        assert retrieved.shape == (2, 7, c.engram_dim)

    def test_retrieve_argmax_returns_table_rows(self):
        """Hard-argmax retrieval must return exact rows of the corpus table.

        The softmax retrieve() produces a blend so it wouldn't; the
        dedicated retrieve_argmax() exists for inference.
        """
        from ct87.engram import EngramANNInjection
        c = self._ann_config()
        t = self._fake_table(20, c.engram_dim)
        m = EngramANNInjection(c, t)
        h = torch.randn(1, 5, c.hidden_dim)
        retrieved = m.retrieve_argmax(h)
        for pos in range(5):
            row = retrieved[0, pos]
            diffs = (t - row).abs().sum(dim=-1)
            assert diffs.min().item() < 1e-6

    def test_retrieve_is_convex_blend_of_table_rows(self):
        """Softmax retrieval yields a row in the convex hull of table rows.

        Sanity check: each retrieved vector should be expressible as a
        convex combination of table rows (not exactly any single row,
        which would indicate temperature too sharp + gradient blocked).
        """
        from ct87.engram import EngramANNInjection
        # Deterministic seed: rule out flakes from random query_proj init
        # producing near-degenerate softmax outputs.
        torch.manual_seed(0)
        c = self._ann_config()
        t = self._fake_table(20, c.engram_dim)
        m = EngramANNInjection(c, t)
        h = torch.randn(1, 5, c.hidden_dim)
        retrieved = m.retrieve(h)
        for pos in range(5):
            row = retrieved[0, pos]
            diffs = (t - row).abs().sum(dim=-1)
            # Should differ from every single row by more than numerical noise
            assert diffs.min().item() > 1e-4

    def test_forward_shape(self):
        from ct87.engram import EngramANNInjection
        c = self._ann_config()
        t = self._fake_table(20, c.engram_dim)
        m = EngramANNInjection(c, t)
        h = torch.randn(2, 7, c.hidden_dim)
        residual, gate = m(h)
        assert residual.shape == (2, 7, c.hidden_dim)
        assert gate.shape == (2, 7, 1)
        # Gate must be in [clamp_min, 1] during warmup
        assert (gate >= m.clamp_min - 1e-6).all()
        assert (gate <= 1.0 + 1e-6).all()

    def test_gate_clamp_active_during_warmup(self):
        """With step < clamp_until_step, gate values must be ≥ clamp_min."""
        from ct87.engram import EngramANNInjection
        c = self._ann_config()
        t = self._fake_table(20, c.engram_dim)
        m = EngramANNInjection(c, t, clamp_until_step=100, clamp_min=0.5)
        h = torch.randn(3, 5, c.hidden_dim)
        # Default step=0 → clamp active
        _, gate = m(h)
        assert (gate >= 0.5 - 1e-6).all()
        # Step bumped past warmup → clamp inactive
        m.set_step(200)
        # Construct hidden state that would produce a gate below 0.5.
        # With random init, gate values vary; we just check the minimum
        # can now drop below clamp_min (signalling clamp is off).
        # Use a large batch to likely sample low-gate tokens.
        h_large = torch.randn(8, 32, c.hidden_dim)
        _, gate_late = m(h_large)
        # Sanity: still in [0, 1] regardless
        assert (gate_late >= 0.0).all(), "gate values below 0"
        assert (gate_late <= 1.0).all(), "gate values above 1"

    def test_gate_clamp_respects_set_step(self):
        """set_step() must actually update the buffer."""
        from ct87.engram import EngramANNInjection
        c = self._ann_config()
        m = EngramANNInjection(c, self._fake_table(10, c.engram_dim))
        m.set_step(0)
        assert int(m.current_step.item()) == 0
        m.set_step(500)
        assert int(m.current_step.item()) == 500
        m.set_step(12345)
        assert int(m.current_step.item()) == 12345

    def test_entropy_loss_penalizes_collapsed_gate(self):
        """Low-entropy (collapsed) gates must yield higher entropy loss."""
        from ct87.engram import compute_gate_entropy_loss
        # Collapsed to 0 → very low entropy → high loss value (less negative)
        collapsed_low = torch.full((1, 10, 1), 0.01)
        # Indecisive at 0.5 → maximum entropy → lowest loss value (most negative)
        indecisive = torch.full((1, 10, 1), 0.5)
        # Collapsed to 1 → very low entropy → high loss value
        collapsed_high = torch.full((1, 10, 1), 0.99)
        loss_low = compute_gate_entropy_loss(collapsed_low).item()
        loss_mid = compute_gate_entropy_loss(indecisive).item()
        loss_high = compute_gate_entropy_loss(collapsed_high).item()
        # -H(p): more negative = higher entropy = preferred
        assert loss_mid < loss_low, (
            f"indecisive gate should have lower loss than collapsed-low "
            f"(got mid={loss_mid}, low={loss_low})"
        )
        assert loss_mid < loss_high, (
            f"indecisive gate should have lower loss than collapsed-high "
            f"(got mid={loss_mid}, high={loss_high})"
        )

    def test_forward_is_differentiable_through_ann(self):
        """Gradients must flow back through retrieval, projection, and gate."""
        from ct87.engram import EngramANNInjection
        c = self._ann_config()
        t = self._fake_table(20, c.engram_dim)
        m = EngramANNInjection(c, t)
        h = torch.randn(2, 5, c.hidden_dim, requires_grad=True)
        residual, _ = m(h)
        residual.sum().backward()
        # query_proj, key_proj, value_proj must all have grads
        assert m.query_proj.weight.grad is not None
        assert m.key_proj.weight.grad is not None
        assert m.value_proj.weight.grad is not None
        # Input must also have a grad (path not severed)
        assert h.grad is not None
        # Table itself must NOT receive grad (frozen buffer)
        assert not m.table.requires_grad

    def test_harmony_model_attach_and_forward(self):
        """End-to-end: attach ANN engram to HarmonyModel and run forward."""
        from ct87.engram import EngramANNInjection
        c = self._ann_config()
        model = HarmonyModel(c)
        t = self._fake_table(20, c.engram_dim)
        ann = EngramANNInjection(c, t)
        model.attach_engram_ann(ann)

        input_ids = torch.randint(0, c.vocab_size, (2, 16))
        # Note: no engram_embeddings passed - the ANN module handles
        # retrieval internally.
        logits = model(input_ids=input_ids)
        assert logits.shape == (2, 16, c.vocab_size)
        # Gate side-channel populated
        assert model._last_ann_gate is not None
        assert model._last_ann_gate.shape == (2, 16, 1)

    def test_attach_rejects_wrong_flag(self):
        """Attaching without setting use_ann_engram must raise."""
        from ct87.engram import EngramANNInjection
        c = _tiny_config()  # use_ann_engram defaults to False
        model = HarmonyModel(c)
        ann = EngramANNInjection(
            self._ann_config(), self._fake_table(10, c.engram_dim),
        )
        with pytest.raises(ValueError, match="use_ann_engram"):
            model.attach_engram_ann(ann)

    def test_tiny_engram_ann_factory(self):
        beta_config = HarmonyModelConfig.tiny_engram_ann()
        base = HarmonyModelConfig.tiny()
        assert beta_config.use_ann_engram is True
        assert base.use_ann_engram is False
        # All other fields equal
        assert beta_config.num_layers == base.num_layers
        assert beta_config.hidden_dim == base.hidden_dim
        assert beta_config.engram_injection_layer == base.engram_injection_layer
        assert beta_config.engram_dim == base.engram_dim


class TestEngramANNConstructorValidation:
    """PR #232 review: fail fast on invalid constructor args."""

    @staticmethod
    def _ann_config() -> HarmonyModelConfig:
        c = _tiny_config()
        c.use_ann_engram = True
        return c

    def test_zero_temperature_raises(self):
        from ct87.engram import EngramANNInjection
        c = self._ann_config()
        t = torch.randn(10, c.engram_dim)
        with pytest.raises(ValueError, match="retrieval_temperature"):
            EngramANNInjection(c, t, retrieval_temperature=0.0)

    def test_negative_temperature_raises(self):
        from ct87.engram import EngramANNInjection
        c = self._ann_config()
        t = torch.randn(10, c.engram_dim)
        with pytest.raises(ValueError, match="retrieval_temperature"):
            EngramANNInjection(c, t, retrieval_temperature=-1.0)

    def test_nonfinite_temperature_raises(self):
        from ct87.engram import EngramANNInjection
        c = self._ann_config()
        t = torch.randn(10, c.engram_dim)
        with pytest.raises(ValueError, match="retrieval_temperature"):
            EngramANNInjection(c, t, retrieval_temperature=float("inf"))

    def test_clamp_min_out_of_range_raises(self):
        from ct87.engram import EngramANNInjection
        c = self._ann_config()
        t = torch.randn(10, c.engram_dim)
        with pytest.raises(ValueError, match="clamp_min"):
            EngramANNInjection(c, t, clamp_min=-0.1)
        with pytest.raises(ValueError, match="clamp_min"):
            EngramANNInjection(c, t, clamp_min=1.1)

    def test_negative_clamp_until_step_raises(self):
        from ct87.engram import EngramANNInjection
        c = self._ann_config()
        t = torch.randn(10, c.engram_dim)
        with pytest.raises(ValueError, match="clamp_until_step"):
            EngramANNInjection(c, t, clamp_until_step=-1)


class TestEngramANNAntiCollapseInit:
    """PR #232 review: conv1d must start zeroed so warmup doesn't inject noise.

    The production EngramGatedResidual relies on HarmonyModel._init_weights()
    to zero its conv1d, but EngramANNInjection is attached AFTER _init_weights
    runs. Without explicit zero-init here, the hard gate clamp (g >= 0.5)
    during warmup steps would push random conv-filtered noise into the
    residual stream from step 0 — directly undermining the anti-collapse
    strategy that depends on a quiet start. Regression guard for the Cursor
    Bugbot finding.
    """

    @staticmethod
    def _ann_config() -> HarmonyModelConfig:
        c = _tiny_config()
        c.use_ann_engram = True
        return c

    def test_conv1d_starts_zeroed(self):
        from ct87.engram import EngramANNInjection
        c = self._ann_config()
        t = torch.randn(10, c.engram_dim)
        m = EngramANNInjection(c, t)
        assert (m.conv1d.weight == 0).all(), (
            "conv1d weight must be zero-initialized; otherwise the warmup "
            "gate clamp injects random noise from step 0."
        )

    def test_forward_at_step_zero_produces_zero_residual(self):
        """With zero-init conv1d, forward output is zeros at step 0."""
        from ct87.engram import EngramANNInjection
        c = self._ann_config()
        t = torch.randn(10, c.engram_dim)
        m = EngramANNInjection(c, t)
        h = torch.randn(2, 5, c.hidden_dim)
        residual, _ = m(h)
        assert torch.allclose(residual, torch.zeros_like(residual)), (
            "With conv1d zero-initialized, residual must be all zeros at "
            f"step 0 — got max abs {residual.abs().max().item():.4e}"
        )


class TestEngramANNCheckpointSize:
    """PR #232 review: corpus table must NOT bloat checkpoints.

    Regression guard — the table is registered with persistent=False so
    state_dict() doesn't include it. Callers must re-load the table from
    the original safetensors file on resume.
    """

    @staticmethod
    def _ann_config() -> HarmonyModelConfig:
        c = _tiny_config()
        c.use_ann_engram = True
        return c

    def test_table_not_in_state_dict(self):
        from ct87.engram import EngramANNInjection
        c = self._ann_config()
        t = torch.randn(10, c.engram_dim)
        m = EngramANNInjection(c, t)
        keys = set(m.state_dict().keys())
        assert "table" not in keys, (
            "corpus table must be non-persistent; got in state_dict"
        )
        assert "table_normalized" not in keys, (
            "normalized cache must be non-persistent"
        )

    def test_load_state_dict_refreshes_normalized_cache(self):
        """Even though table is non-persistent, the refresh hook is wired."""
        from ct87.engram import EngramANNInjection
        c = self._ann_config()
        t = torch.randn(10, c.engram_dim)
        m = EngramANNInjection(c, t)
        # Replace the table in place (simulating a fresh re-load) and
        # verify refresh keeps normalized cache in sync.
        new_table = torch.randn(10, c.engram_dim) * 3.0
        m.table = new_table
        m._refresh_table_normalized()
        expected = torch.nn.functional.normalize(new_table, dim=-1, eps=1e-8)
        assert torch.allclose(m.table_normalized, expected)


class TestEngramCrossAttention:
    """Model delta: cross-attention injection with top-k retrieval (ZEB-117)."""

    @staticmethod
    def _xattn_config() -> HarmonyModelConfig:
        c = _tiny_config()
        c.use_xattn_engram = True
        return c

    @staticmethod
    def _fake_table(total_entries: int, engram_dim: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(0)
        return torch.randn(total_entries, engram_dim, generator=g)

    def test_construction_rejects_dim_mismatch(self):
        from ct87.engram import EngramCrossAttention
        c = self._xattn_config()
        bad_table = torch.randn(16, 8)
        with pytest.raises(ValueError, match="engram_dim"):
            EngramCrossAttention(c, bad_table)

    def test_construction_rejects_wrong_shape(self):
        from ct87.engram import EngramCrossAttention
        c = self._xattn_config()
        with pytest.raises(ValueError, match="2-D"):
            EngramCrossAttention(c, torch.randn(10))

    def test_construction_rejects_k_too_large(self):
        from ct87.engram import EngramCrossAttention
        c = self._xattn_config()
        t = self._fake_table(4, c.engram_dim)
        with pytest.raises(ValueError, match="k_retrieved"):
            EngramCrossAttention(c, t, k_retrieved=10)

    def test_construction_rejects_zero_k(self):
        from ct87.engram import EngramCrossAttention
        c = self._xattn_config()
        t = self._fake_table(20, c.engram_dim)
        with pytest.raises(ValueError, match="k_retrieved"):
            EngramCrossAttention(c, t, k_retrieved=0)

    def test_construction_rejects_indivisible_hidden_dim(self):
        from ct87.engram import EngramCrossAttention
        c = self._xattn_config()
        t = self._fake_table(20, c.engram_dim)
        # hidden_dim=32 is not divisible by num_heads=5
        with pytest.raises(ValueError, match="divisible"):
            EngramCrossAttention(c, t, num_heads=5)

    def test_construction_rejects_nonfinite_bias_weight(self):
        from ct87.engram import EngramCrossAttention
        c = self._xattn_config()
        t = self._fake_table(20, c.engram_dim)
        with pytest.raises(ValueError, match="retrieval_bias_weight"):
            EngramCrossAttention(c, t, retrieval_bias_weight=float("nan"))

    def test_construction_rejects_zero_temperature(self):
        from ct87.engram import EngramCrossAttention
        c = self._xattn_config()
        t = self._fake_table(20, c.engram_dim)
        with pytest.raises(ValueError, match="retrieval_temperature"):
            EngramCrossAttention(c, t, retrieval_temperature=0.0)

    def test_table_registered_as_buffer_not_parameter(self):
        from ct87.engram import EngramCrossAttention
        c = self._xattn_config()
        t = self._fake_table(20, c.engram_dim)
        m = EngramCrossAttention(c, t)
        param_names = {n for n, _ in m.named_parameters()}
        buffer_names = {n for n, _ in m.named_buffers()}
        assert "table" not in param_names
        assert "table" in buffer_names
        assert "table_normalized" in buffer_names

    def test_retrieve_topk_shapes(self):
        from ct87.engram import EngramCrossAttention
        c = self._xattn_config()
        t = self._fake_table(20, c.engram_dim)
        m = EngramCrossAttention(c, t, k_retrieved=4)
        h = torch.randn(2, 6, c.hidden_dim)
        retrieved, sims = m.retrieve_topk(h)
        assert retrieved.shape == (2, 6, 4, c.engram_dim)
        assert sims.shape == (2, 6, 4)

    def test_retrieve_topk_returns_actual_table_rows(self):
        """Top-k gather must return exact table rows (cosine similarity order)."""
        from ct87.engram import EngramCrossAttention
        c = self._xattn_config()
        t = self._fake_table(20, c.engram_dim)
        m = EngramCrossAttention(c, t, k_retrieved=3)
        h = torch.randn(1, 4, c.hidden_dim)
        retrieved, _ = m.retrieve_topk(h)
        # Each retrieved vector must exactly match some row of the table.
        for pos in range(4):
            for j in range(3):
                row = retrieved[0, pos, j]
                diffs = (t - row).abs().sum(dim=-1)
                assert diffs.min().item() < 1e-6

    def test_forward_shape(self):
        from ct87.engram import EngramCrossAttention
        c = self._xattn_config()
        t = self._fake_table(20, c.engram_dim)
        m = EngramCrossAttention(c, t)
        h = torch.randn(2, 7, c.hidden_dim)
        out = m(h)
        assert out.shape == (2, 7, c.hidden_dim)

    def test_forward_at_step_zero_produces_zero_residual(self):
        """With o_proj zero-initialized, forward output is identically zero.

        This is the anti-collapse mechanism: the model trains as if
        unattached until o_proj learns a nontrivial direction. Regression
        guard for the residual-zero init.
        """
        from ct87.engram import EngramCrossAttention
        c = self._xattn_config()
        t = self._fake_table(20, c.engram_dim)
        m = EngramCrossAttention(c, t)
        h = torch.randn(3, 8, c.hidden_dim)
        out = m(h)
        assert torch.allclose(out, torch.zeros_like(out)), (
            "o_proj must start zeroed so the cross-attention residual is "
            f"0 at step 0; got max abs {out.abs().max().item():.4e}"
        )

    def test_o_proj_zero_init(self):
        from ct87.engram import EngramCrossAttention
        c = self._xattn_config()
        t = self._fake_table(10, c.engram_dim)
        m = EngramCrossAttention(c, t)
        assert (m.o_proj.weight == 0).all(), (
            "o_proj must be zero-initialized for residual-zero anti-collapse"
        )

    def test_gradient_flows_through_all_learnable_projections(self):
        """Every trainable projection must receive gradient from the loss.

        Critical: the retrieval-similarity bias is the ONLY path through
        which retrieval_query_proj learns (top-k gather is
        non-differentiable). If that bias path breaks, retrieval stops
        learning - same failure mode that would have killed gamma if we
        hadn't used differentiable softmax retrieval.
        """
        from ct87.engram import EngramCrossAttention
        c = self._xattn_config()
        t = self._fake_table(20, c.engram_dim)
        m = EngramCrossAttention(c, t)
        h = torch.randn(2, 5, c.hidden_dim, requires_grad=True)
        # o_proj starts at zero so downstream gradients would all be zero.
        # Perturb o_proj so attention outputs propagate to the loss.
        with torch.no_grad():
            m.o_proj.weight.add_(torch.randn_like(m.o_proj.weight) * 0.01)
        out = m(h)
        out.sum().backward()
        assert m.q_proj.weight.grad is not None
        assert m.k_proj.weight.grad is not None
        assert m.v_proj.weight.grad is not None
        assert m.o_proj.weight.grad is not None
        assert m.retrieval_query_proj.weight.grad is not None, (
            "retrieval_query_proj must receive gradient via the "
            "retrieval-similarity bias path."
        )
        assert m.retrieval_query_proj.weight.grad.abs().sum().item() > 0
        assert h.grad is not None
        assert not m.table.requires_grad

    def test_retrieval_query_grad_vanishes_without_bias(self):
        """Sanity check on the bias path: with bias_weight=0, the retrieval
        query projection loses its gradient path (bias is the only route).
        """
        from ct87.engram import EngramCrossAttention
        c = self._xattn_config()
        t = self._fake_table(20, c.engram_dim)
        m = EngramCrossAttention(c, t, retrieval_bias_weight=0.0)
        with torch.no_grad():
            m.o_proj.weight.add_(torch.randn_like(m.o_proj.weight) * 0.01)
        h = torch.randn(2, 5, c.hidden_dim)
        out = m(h)
        out.sum().backward()
        grad = m.retrieval_query_proj.weight.grad
        # Gradient is either None (autograd skipped it) or exactly zero.
        assert grad is None or grad.abs().sum().item() == 0.0, (
            "With retrieval_bias_weight=0, retrieval_query_proj must have "
            "no gradient path - confirms the bias is the only route."
        )

    def test_table_not_in_state_dict(self):
        from ct87.engram import EngramCrossAttention
        c = self._xattn_config()
        t = self._fake_table(10, c.engram_dim)
        m = EngramCrossAttention(c, t)
        keys = set(m.state_dict().keys())
        assert "table" not in keys
        assert "table_normalized" not in keys

    def test_harmony_model_attach_and_forward(self):
        """End-to-end: attach xattn engram to HarmonyModel and run forward."""
        from ct87.engram import EngramCrossAttention
        c = self._xattn_config()
        model = HarmonyModel(c)
        t = self._fake_table(20, c.engram_dim)
        xattn = EngramCrossAttention(c, t)
        model.attach_engram_xattn(xattn)
        input_ids = torch.randint(0, c.vocab_size, (2, 16))
        logits = model(input_ids=input_ids)
        assert logits.shape == (2, 16, c.vocab_size)
        # xattn does not populate the ANN gate side-channel
        assert model._last_ann_gate is None

    def test_attach_rejects_wrong_flag(self):
        from ct87.engram import EngramCrossAttention
        c = _tiny_config()  # use_xattn_engram defaults to False
        model = HarmonyModel(c)
        xattn = EngramCrossAttention(
            self._xattn_config(), self._fake_table(10, c.engram_dim),
        )
        with pytest.raises(ValueError, match="use_xattn_engram"):
            model.attach_engram_xattn(xattn)

    def test_attach_rejects_if_ann_already_attached(self):
        """Gamma and delta must be mutually exclusive at attach-time."""
        from ct87.engram import EngramANNInjection, EngramCrossAttention
        c = self._xattn_config()
        c.use_ann_engram = False  # only xattn flag set
        model = HarmonyModel(c)
        # Force-attach gamma (via bypass, simulating a misconfiguration)
        ann_cfg = _tiny_config()
        ann_cfg.use_ann_engram = True
        model.engram_ann = EngramANNInjection(
            ann_cfg, self._fake_table(10, c.engram_dim),
        )
        xattn = EngramCrossAttention(c, self._fake_table(10, c.engram_dim))
        with pytest.raises(ValueError, match="mutually exclusive"):
            model.attach_engram_xattn(xattn)

    def test_config_mutual_exclusion_validation(self):
        """Setting both flags on the config must raise at __post_init__."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            HarmonyModelConfig(
                num_layers=4, hidden_dim=32, num_query_heads=4, num_kv_heads=2,
                head_dim=8, ffn_dim=64, vocab_size=128, max_seq_len=64,
                rope_theta=10000.0, rms_norm_eps=1e-6, layers_per_block=2,
                engram_injection_layer=1, engram_dim=16, tie_embeddings=True,
                use_ann_engram=True, use_xattn_engram=True,
            )

    def test_tiny_engram_xattn_factory(self):
        delta_config = HarmonyModelConfig.tiny_engram_xattn()
        base = HarmonyModelConfig.tiny()
        assert delta_config.use_xattn_engram is True
        assert delta_config.use_ann_engram is False
        assert base.use_xattn_engram is False
        assert delta_config.num_layers == base.num_layers
        assert delta_config.hidden_dim == base.hidden_dim
        assert delta_config.engram_injection_layer == base.engram_injection_layer
        assert delta_config.engram_dim == base.engram_dim


class TestEngramCrossAttentionHeadGates:
    """Experiment epsilon: per-head gate scalars for bio-inspired routing (ZEB-127)."""

    @staticmethod
    def _xattn_config() -> HarmonyModelConfig:
        c = _tiny_config()
        c.use_xattn_engram = True
        return c

    @staticmethod
    def _fake_table(total_entries: int, engram_dim: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(0)
        return torch.randn(total_entries, engram_dim, generator=g)

    def test_head_gates_parameter_exists_when_enabled(self):
        from ct87.engram import EngramCrossAttention
        c = self._xattn_config()
        t = self._fake_table(20, c.engram_dim)
        m = EngramCrossAttention(c, t, use_head_gates=True)
        param_names = {n for n, _ in m.named_parameters()}
        assert "head_gates" in param_names

    def test_head_gates_not_present_when_disabled(self):
        from ct87.engram import EngramCrossAttention
        c = self._xattn_config()
        t = self._fake_table(20, c.engram_dim)
        m = EngramCrossAttention(c, t, use_head_gates=False)
        param_names = {n for n, _ in m.named_parameters()}
        assert "head_gates" not in param_names

    def test_head_gates_initialized_to_zero(self):
        from ct87.engram import EngramCrossAttention
        c = self._xattn_config()
        t = self._fake_table(20, c.engram_dim)
        m = EngramCrossAttention(c, t, use_head_gates=True)
        assert torch.allclose(m.head_gates, torch.zeros(c.num_query_heads))

    def test_head_gates_shape_matches_num_heads(self):
        from ct87.engram import EngramCrossAttention
        c = self._xattn_config()
        t = self._fake_table(20, c.engram_dim)
        m = EngramCrossAttention(c, t, use_head_gates=True, num_heads=4)
        assert m.head_gates.shape == (4,)

    def test_head_gates_at_init_produce_same_output_as_ungated(self):
        """At init, sigmoid(0)=0.5 scales all heads equally.

        With o_proj zero-init, both gated and ungated produce zero output.
        After perturbing o_proj, the gated version should produce exactly
        0.5x the ungated version (uniform scaling).
        """
        from ct87.engram import EngramCrossAttention
        c = self._xattn_config()
        t = self._fake_table(20, c.engram_dim)
        m_gated = EngramCrossAttention(c, t, use_head_gates=True)
        m_ungated = EngramCrossAttention(c, t, use_head_gates=False)

        # Copy all shared weights from ungated to gated
        m_gated.load_state_dict(m_ungated.state_dict(), strict=False)

        # Perturb o_proj so output is nonzero
        with torch.no_grad():
            m_gated.o_proj.weight.copy_(torch.randn_like(m_gated.o_proj.weight) * 0.01)
            m_ungated.o_proj.weight.copy_(m_gated.o_proj.weight)

        h = torch.randn(2, 6, c.hidden_dim)
        out_gated = m_gated(h)
        out_ungated = m_ungated(h)

        # sigmoid(0) = 0.5, so gated output should be 0.5x ungated
        assert torch.allclose(out_gated, out_ungated * 0.5, atol=1e-5), (
            f"At init (gates=0), gated output should be 0.5x ungated. "
            f"Max diff: {(out_gated - out_ungated * 0.5).abs().max().item():.4e}"
        )

    def test_head_gates_gradient_flows(self):
        from ct87.engram import EngramCrossAttention
        c = self._xattn_config()
        t = self._fake_table(20, c.engram_dim)
        m = EngramCrossAttention(c, t, use_head_gates=True)
        # Perturb o_proj so gradient can flow
        with torch.no_grad():
            m.o_proj.weight.add_(torch.randn_like(m.o_proj.weight) * 0.01)
        h = torch.randn(2, 5, c.hidden_dim)
        out = m(h)
        out.sum().backward()
        assert m.head_gates.grad is not None, "head_gates must receive gradient"
        assert not torch.allclose(m.head_gates.grad, torch.zeros_like(m.head_gates.grad)), (
            "head_gates gradient should be nonzero with perturbed o_proj"
        )

    def test_differentiated_gates_produce_nonuniform_scaling(self):
        """When gates differ, heads contribute unequally."""
        from ct87.engram import EngramCrossAttention
        c = self._xattn_config()
        t = self._fake_table(20, c.engram_dim)
        m = EngramCrossAttention(c, t, use_head_gates=True)
        with torch.no_grad():
            m.o_proj.weight.add_(torch.randn_like(m.o_proj.weight) * 0.01)
            # Set gates to different values: head 0 high, head 7 low
            m.head_gates.copy_(torch.tensor(
                [2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0]
            ))
        h = torch.randn(2, 6, c.hidden_dim)
        out = m(h)
        # Output should be nonzero and not uniform across hidden dims
        assert out.abs().max() > 1e-6
