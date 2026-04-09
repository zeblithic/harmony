"""Tests for the evaluation harness."""

import math
import tempfile

import torch
import pytest

from ct87.model import HarmonyModel, HarmonyModelConfig
from ct87.train import make_synthetic_dataloader, save_checkpoint
from ct87.eval import evaluate, load_checkpoint_path


def _tiny_config() -> HarmonyModelConfig:
    return HarmonyModelConfig(
        num_layers=4, hidden_dim=32, num_query_heads=4, num_kv_heads=2,
        head_dim=8, ffn_dim=64, vocab_size=128, max_seq_len=64,
        rope_theta=10000.0, rms_norm_eps=1e-6, layers_per_block=2,
        engram_injection_layer=1, engram_dim=16, tie_embeddings=True,
    )


class TestEvaluate:
    def test_returns_expected_keys(self):
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        dataloader = make_synthetic_dataloader(cfg.vocab_size, 16, batch_size=2, seed=99)

        metrics = evaluate(
            model, dataloader, cfg.vocab_size, torch.device("cpu"),
            num_batches=5,
        )

        assert set(metrics.keys()) == {"loss", "perplexity", "total_tokens", "tokens_per_sec", "elapsed_sec"}

    def test_loss_is_finite_positive(self):
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        dataloader = make_synthetic_dataloader(cfg.vocab_size, 16, batch_size=2, seed=99)

        metrics = evaluate(
            model, dataloader, cfg.vocab_size, torch.device("cpu"),
            num_batches=5,
        )

        assert metrics["loss"] > 0.0
        assert math.isfinite(metrics["loss"])

    def test_perplexity_is_exp_loss(self):
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        dataloader = make_synthetic_dataloader(cfg.vocab_size, 16, batch_size=2, seed=99)

        metrics = evaluate(
            model, dataloader, cfg.vocab_size, torch.device("cpu"),
            num_batches=5,
        )

        assert metrics["perplexity"] == pytest.approx(math.exp(metrics["loss"]), rel=1e-6)

    def test_total_tokens_correct(self):
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        seq_len = 16
        batch_size = 2
        num_batches = 5
        dataloader = make_synthetic_dataloader(cfg.vocab_size, seq_len, batch_size, seed=99)

        metrics = evaluate(
            model, dataloader, cfg.vocab_size, torch.device("cpu"),
            num_batches=num_batches,
        )

        expected_tokens = batch_size * seq_len * num_batches
        assert metrics["total_tokens"] == expected_tokens

    def test_deterministic_same_seed(self):
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        model.eval()

        dl1 = make_synthetic_dataloader(cfg.vocab_size, 16, batch_size=2, seed=99)
        m1 = evaluate(model, dl1, cfg.vocab_size, torch.device("cpu"), num_batches=5)

        dl2 = make_synthetic_dataloader(cfg.vocab_size, 16, batch_size=2, seed=99)
        m2 = evaluate(model, dl2, cfg.vocab_size, torch.device("cpu"), num_batches=5)

        assert m1["loss"] == pytest.approx(m2["loss"], rel=1e-6)

    def test_with_bf16(self):
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        dataloader = make_synthetic_dataloader(cfg.vocab_size, 16, batch_size=2, seed=99)

        metrics = evaluate(
            model, dataloader, cfg.vocab_size, torch.device("cpu"),
            num_batches=3, amp_dtype=torch.bfloat16,
        )

        assert metrics["loss"] > 0.0
        assert math.isfinite(metrics["loss"])

    def test_no_grad_modification(self):
        """Evaluation must not change model weights."""
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)

        params_before = {n: p.clone() for n, p in model.named_parameters()}

        dataloader = make_synthetic_dataloader(cfg.vocab_size, 16, batch_size=2, seed=99)
        evaluate(model, dataloader, cfg.vocab_size, torch.device("cpu"), num_batches=5)

        for name, param in model.named_parameters():
            assert torch.equal(param, params_before[name]), f"param {name} changed during eval"

    def test_restores_training_mode(self):
        """evaluate() must restore the model's original training state."""
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        model.train()
        assert model.training is True

        dataloader = make_synthetic_dataloader(cfg.vocab_size, 16, batch_size=2, seed=99)
        evaluate(model, dataloader, cfg.vocab_size, torch.device("cpu"), num_batches=3)

        assert model.training is True, "evaluate() did not restore training mode"

    def test_num_batches_zero_raises(self):
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        dataloader = make_synthetic_dataloader(cfg.vocab_size, 16, batch_size=2, seed=99)

        with pytest.raises(ValueError, match="num_batches must be >= 1"):
            evaluate(model, dataloader, cfg.vocab_size, torch.device("cpu"), num_batches=0)


class TestLoadCheckpointPath:
    def test_load_from_explicit_path(self):
        """load_checkpoint_path loads weights and produces matching logits."""
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
        logits_before = model(input_ids).detach()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(model, None, 0, tmpdir)
            ckpt_path = f"{tmpdir}/model_step_0.safetensors"

            model2 = HarmonyModel(cfg)
            load_checkpoint_path(model2, ckpt_path)
            logits_after = model2(input_ids).detach()

        assert torch.allclose(logits_before, logits_after, atol=1e-6)


class TestEvaluateWithEngram:
    def test_eval_with_engram_table(self):
        """Evaluate with Engram table produces finite metrics."""
        from ct87.engram import EngramTable

        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        table = torch.randn(100, cfg.engram_dim)
        engram_table = EngramTable(table, hash_seeds=[42, 99])

        dataloader = make_synthetic_dataloader(cfg.vocab_size, 16, batch_size=2, seed=99)
        metrics = evaluate(
            model, dataloader, cfg.vocab_size, torch.device("cpu"),
            num_batches=3, engram_table=engram_table,
        )

        assert metrics["loss"] > 0.0
        assert math.isfinite(metrics["loss"])
