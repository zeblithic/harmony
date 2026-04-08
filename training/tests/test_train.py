"""Tests for the training loop."""

import os
import tempfile

import torch
import pytest
from ct87.model import HarmonyModel, HarmonyModelConfig
from ct87.optim import Muon, WSDSchedule, partition_params
from ct87.train import save_checkpoint, load_checkpoint, make_synthetic_dataloader


def _tiny_config() -> HarmonyModelConfig:
    return HarmonyModelConfig(
        num_layers=4, hidden_dim=32, num_query_heads=4, num_kv_heads=2,
        head_dim=8, ffn_dim=64, vocab_size=128, max_seq_len=64,
        rope_theta=10000.0, rms_norm_eps=1e-6, layers_per_block=2,
        engram_injection_layer=1, engram_dim=16, tie_embeddings=True,
    )


class TestOverfit:
    def test_loss_decreases_on_tiny_batch(self):
        """Tiny model overfits a small batch -- loss drops below 2.0 in 50 steps."""
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)

        muon_params, adam_params = partition_params(model)
        optimizer = Muon(muon_params, adam_params, lr=1e-3, adam_lr=1e-3)

        input_ids = torch.randint(0, cfg.vocab_size, (2, 17))
        x = input_ids[:, :-1]
        targets = input_ids[:, 1:]

        initial_loss = None
        final_loss = None

        for step in range(50):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, cfg.vocab_size), targets.reshape(-1),
            )
            if step == 0:
                initial_loss = loss.item()
            final_loss = loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        assert final_loss < initial_loss, f"Loss should decrease: {initial_loss} -> {final_loss}"
        assert final_loss < 2.0, f"Loss should drop below 2.0 after 50 steps, got {final_loss}"


class TestCheckpoint:
    def test_save_load_roundtrip(self):
        """Save and load a checkpoint -- logits should match exactly."""
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)

        input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
        logits_before = model(input_ids).detach()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(model, None, 0, tmpdir)

            model2 = HarmonyModel(cfg)
            load_checkpoint(model2, tmpdir, step=0)
            logits_after = model2(input_ids).detach()

        assert torch.allclose(logits_before, logits_after, atol=1e-6)


class TestSyntheticDataloader:
    def test_batch_shape(self):
        dl = make_synthetic_dataloader(vocab_size=128, seq_len=16, batch_size=4, seed=42)
        batch = next(dl)
        assert batch.shape == (4, 17)  # seq_len + 1

    def test_values_in_range(self):
        dl = make_synthetic_dataloader(vocab_size=128, seq_len=16, batch_size=4, seed=42)
        batch = next(dl)
        assert batch.min().item() >= 0
        assert batch.max().item() < 128

    def test_reproducible(self):
        dl1 = make_synthetic_dataloader(vocab_size=128, seq_len=16, batch_size=4, seed=42)
        dl2 = make_synthetic_dataloader(vocab_size=128, seq_len=16, batch_size=4, seed=42)
        assert torch.equal(next(dl1), next(dl2))


class TestHfDataloaderGuard:
    def test_too_few_tokens_raises(self):
        """make_hf_dataloader raises ValueError when dataset is too small."""
        from ct87.train import make_hf_dataloader
        from datasets import Dataset

        # Create a tiny dataset with only 5 tokens — less than seq_len+1=17
        ds = Dataset.from_dict({"input_ids": [[1, 2, 3, 4, 5]]})
        with tempfile.TemporaryDirectory() as tmpdir:
            ds.save_to_disk(tmpdir)
            with pytest.raises(ValueError, match="tokens are needed"):
                make_hf_dataloader(tmpdir, seq_len=16, batch_size=1)


class TestValidation:
    def test_returns_finite_float(self):
        """compute_validation_loss returns a finite float loss value."""
        from ct87.train import compute_validation_loss, make_synthetic_dataloader

        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        device = torch.device("cpu")

        val_loader = make_synthetic_dataloader(cfg.vocab_size, 16, batch_size=2, seed=99)
        val_loss = compute_validation_loss(model, val_loader, cfg.vocab_size, device, num_batches=3)

        assert isinstance(val_loss, float)
        assert val_loss > 0.0
        assert not torch.isnan(torch.tensor(val_loss))
        assert not torch.isinf(torch.tensor(val_loss))

    def test_no_grad_change(self):
        """Validation does not modify model weights."""
        from ct87.train import compute_validation_loss, make_synthetic_dataloader

        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        device = torch.device("cpu")

        params_before = {n: p.clone() for n, p in model.named_parameters()}

        val_loader = make_synthetic_dataloader(cfg.vocab_size, 16, batch_size=2, seed=99)
        compute_validation_loss(model, val_loader, cfg.vocab_size, device, num_batches=3)

        for name, param in model.named_parameters():
            assert torch.equal(param, params_before[name]), f"param {name} changed during validation"
