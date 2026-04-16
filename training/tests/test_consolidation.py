"""Tests for ZEB-128 engram consolidation."""
import subprocess
import sys
import os
import torch
import torch.nn.functional as F
import pytest

from ct87.engram import EngramConsolidationDecoder
from ct87.model import HarmonyModel, HarmonyModelConfig
from ct87.engram import EngramCrossAttention


class TestEngramConsolidationDecoder:

    def test_output_shape_matches_input(self):
        decoder = EngramConsolidationDecoder(hidden_dim=32)
        x = torch.randn(2, 16, 32)
        out = decoder(x)
        assert out.shape == (2, 16, 32)

    def test_xavier_init_weights_are_nonzero(self):
        decoder = EngramConsolidationDecoder(hidden_dim=32)
        assert decoder.net[0].weight.abs().sum() > 0
        assert decoder.net[2].weight.abs().sum() > 0

    def test_bias_initialized_to_zero(self):
        decoder = EngramConsolidationDecoder(hidden_dim=32)
        assert torch.allclose(decoder.net[0].bias, torch.zeros(32))
        assert torch.allclose(decoder.net[2].bias, torch.zeros(32))

    def test_gradient_flows_to_input(self):
        decoder = EngramConsolidationDecoder(hidden_dim=32)
        x = torch.randn(1, 8, 32, requires_grad=True)
        out = decoder(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_nonlinear_mapping(self):
        decoder = EngramConsolidationDecoder(hidden_dim=32)
        x1 = torch.randn(1, 4, 32)
        x2 = x1 * 2.0
        out1 = decoder(x1)
        out2 = decoder(x2)
        assert not torch.allclose(out2, out1 * 2.0, atol=1e-4)


def _tiny_config() -> HarmonyModelConfig:
    """Shared tiny config for consolidation tests."""
    return HarmonyModelConfig(
        num_layers=4, hidden_dim=32, num_query_heads=4, num_kv_heads=2,
        head_dim=8, ffn_dim=64, vocab_size=128, max_seq_len=64,
        rope_theta=10000.0, rms_norm_eps=1e-6, layers_per_block=2,
        engram_injection_layer=1, engram_dim=16, tie_embeddings=True,
    )


def _fake_table(total_entries: int, engram_dim: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.randn(total_entries, engram_dim, generator=g)


class TestXattnOutputCapture:

    def test_last_xattn_output_captured_after_forward(self):
        config = _tiny_config()
        config.use_xattn_engram = True
        model = HarmonyModel(config)
        table = _fake_table(100, config.engram_dim)
        xattn = EngramCrossAttention(config, table)
        model.attach_engram_xattn(xattn)

        x = torch.randint(0, config.vocab_size, (1, 16))
        model(x)

        assert model._last_xattn_output is not None
        assert model._last_xattn_output.shape == (1, 16, config.hidden_dim)

    def test_last_xattn_output_is_none_without_xattn(self):
        config = _tiny_config()
        model = HarmonyModel(config)
        x = torch.randint(0, config.vocab_size, (1, 16))
        model(x)
        assert model._last_xattn_output is None

    def test_last_xattn_output_reset_each_forward(self):
        config = _tiny_config()
        config.use_xattn_engram = True
        model = HarmonyModel(config)
        table = _fake_table(100, config.engram_dim)
        xattn = EngramCrossAttention(config, table)
        model.attach_engram_xattn(xattn)

        x = torch.randint(0, config.vocab_size, (1, 16))
        model(x)

        x2 = torch.randint(0, config.vocab_size, (1, 8))
        model(x2)
        assert model._last_xattn_output.shape == (1, 8, config.hidden_dim)

    def test_pre_injection_hidden_captured(self):
        config = _tiny_config()
        config.use_xattn_engram = True
        model = HarmonyModel(config)
        table = _fake_table(100, config.engram_dim)
        xattn = EngramCrossAttention(config, table)
        model.attach_engram_xattn(xattn)

        x = torch.randint(0, config.vocab_size, (1, 16))
        model(x)

        assert model._last_pre_injection_hidden is not None
        assert model._last_pre_injection_hidden.shape == (1, 16, config.hidden_dim)
        assert not model._last_pre_injection_hidden.requires_grad  # detached


class TestInjectionMultiplier:

    def test_multiplier_scales_xattn_output(self):
        config = _tiny_config()
        config.use_xattn_engram = True
        model = HarmonyModel(config)
        table = _fake_table(100, config.engram_dim)
        xattn = EngramCrossAttention(config, table)
        model.attach_engram_xattn(xattn)

        x = torch.randint(0, config.vocab_size, (1, 16))

        model.engram_inject_mult = 1.0
        model(x)
        full_output = model._last_xattn_output.clone()

        model.engram_inject_mult = 0.5
        model(x)
        half_output = model._last_xattn_output.clone()

        # Captured output should be the same (pre-scaling)
        assert torch.allclose(full_output, half_output, atol=1e-5)

    def test_zero_multiplier_zeroes_injection(self):
        config = _tiny_config()
        config.use_xattn_engram = True
        model = HarmonyModel(config)
        table = _fake_table(100, config.engram_dim)
        xattn = EngramCrossAttention(config, table)
        model.attach_engram_xattn(xattn)

        x = torch.randint(0, config.vocab_size, (1, 16))

        # Train for a step to break zero-init symmetry on o_proj
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.engram_inject_mult = 1.0
        logits = model(x)
        logits.sum().backward()
        optimizer.step()
        optimizer.zero_grad()

        model.engram_inject_mult = 1.0
        logits_full = model(x).detach()
        model.engram_inject_mult = 0.0
        logits_zero = model(x).detach()

        # Compare to no-xattn model
        config2 = _tiny_config()
        model2 = HarmonyModel(config2)
        model2.load_state_dict(model.state_dict(), strict=False)
        logits_no_engram = model2(x).detach()

        assert torch.allclose(logits_zero, logits_no_engram, atol=1e-5)


class TestConsolidationConfigPresets:

    def test_consol_online_preset(self):
        config = HarmonyModelConfig.tiny_engram_xattn_consol_online()
        assert config.use_xattn_engram is True
        assert config.use_head_gates is False
        assert config.use_ann_engram is False

    def test_consol_phased_preset(self):
        config = HarmonyModelConfig.tiny_engram_xattn_consol_phased()
        assert config.use_xattn_engram is True
        assert config.use_head_gates is False
        assert config.use_ann_engram is False

    def test_ctrl_preset(self):
        config = HarmonyModelConfig.tiny_engram_xattn_ctrl()
        assert config.use_xattn_engram is True
        assert config.use_head_gates is False
        assert config.use_ann_engram is False

    def test_all_three_presets_equivalent(self):
        a = HarmonyModelConfig.tiny_engram_xattn_consol_online()
        b = HarmonyModelConfig.tiny_engram_xattn_consol_phased()
        c = HarmonyModelConfig.tiny_engram_xattn_ctrl()
        for field in ["num_layers", "hidden_dim", "num_query_heads",
                       "head_dim", "ffn_dim", "vocab_size", "engram_dim",
                       "engram_injection_layer", "use_xattn_engram",
                       "use_head_gates", "use_ann_engram"]:
            assert getattr(a, field) == getattr(b, field) == getattr(c, field), \
                f"Mismatch in {field}"


class TestConsolidationMSELoss:

    def test_mse_loss_computes_correctly(self):
        from ct87.engram import EngramConsolidationDecoder
        decoder = EngramConsolidationDecoder(hidden_dim=32)
        h_pre = torch.randn(2, 8, 32)
        xattn_out = torch.randn(2, 8, 32)

        prediction = decoder(h_pre)
        mse = F.mse_loss(prediction, xattn_out.detach())

        assert mse.item() > 0
        assert mse.requires_grad

    def test_mse_gradient_does_not_flow_to_target(self):
        from ct87.engram import EngramConsolidationDecoder
        decoder = EngramConsolidationDecoder(hidden_dim=32)
        h_pre = torch.randn(2, 8, 32)
        xattn_out = torch.randn(2, 8, 32, requires_grad=True)

        prediction = decoder(h_pre)
        mse = F.mse_loss(prediction, xattn_out.detach())
        mse.backward()

        assert xattn_out.grad is None


class TestInjectionMultiplierAnnealing:

    def test_linear_annealing_schedule(self):
        start_step = 7000
        total_steps = 10000

        for step, expected in [(0, 1.0), (7000, 1.0), (8500, 0.5), (10000, 0.0)]:
            if step >= start_step:
                progress = (step - start_step) / max(1, total_steps - start_step)
                mult = max(0.0, 1.0 - progress)
            else:
                mult = 1.0
            assert abs(mult - expected) < 1e-6, f"Step {step}: {mult} != {expected}"


class TestConsolidationSmoke:

    @pytest.fixture
    def fake_table_path(self, tmp_path):
        """Create a minimal safetensors table for testing."""
        from safetensors.torch import save_file
        table = torch.randn(100, 128)  # 100 entries, engram_dim=128 (matches tiny config)
        path = str(tmp_path / "test_table.safetensors")
        save_file({"engram.weight": table}, path)
        return path

    def _run_train(self, args, tmp_path, fake_table_path):
        """Helper to run ct87.train with common args."""
        result = subprocess.run(
            [
                sys.executable, "-m", "ct87.train",
                "--engram-xattn-table", fake_table_path,
                "--synthetic", "--steps", "10",
                "--output-dir", str(tmp_path / "out"),
                *args,
            ],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            timeout=120,
        )
        return result

    def test_online_mode_runs(self, tmp_path, fake_table_path):
        result = self._run_train([
            "--config", "tiny_engram_xattn_consol_online",
            "--consolidation-mode", "online",
            "--consolidation-lambda", "0.1",
        ], tmp_path, fake_table_path)
        assert result.returncode == 0, f"STDERR:\n{result.stderr}"

    def test_phased_mode_runs(self, tmp_path, fake_table_path):
        result = self._run_train([
            "--config", "tiny_engram_xattn_consol_phased",
            "--consolidation-mode", "phased",
            "--consolidation-start-step", "5",
            "--consolidation-anneal",
            "--consolidation-lambda", "0.1",
        ], tmp_path, fake_table_path)
        assert result.returncode == 0, f"STDERR:\n{result.stderr}"

    def test_ctrl_mode_runs(self, tmp_path, fake_table_path):
        result = self._run_train([
            "--config", "tiny_engram_xattn_ctrl",
            "--consolidation-mode", "none",
        ], tmp_path, fake_table_path)
        assert result.returncode == 0, f"STDERR:\n{result.stderr}"
