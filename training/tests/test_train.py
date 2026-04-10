"""Tests for the training loop."""

import csv
import os
import subprocess
import sys
import tempfile

import torch
import pytest
from ct87.model import HarmonyModel, HarmonyModelConfig
from ct87.optim import Muon, WSDSchedule, partition_params
from ct87.train import (
    save_checkpoint, load_checkpoint, make_synthetic_dataloader,
    compute_validation_loss,
)


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


class TestMixedPrecision:
    def test_bf16_autocast_finite_loss(self):
        """Forward + backward under bf16 autocast produces finite loss, weights stay fp32."""
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)

        batch = torch.randint(0, cfg.vocab_size, (2, 17))
        x, targets = batch[:, :-1], batch[:, 1:]

        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, cfg.vocab_size), targets.reshape(-1),
            )

        loss.backward()

        assert torch.isfinite(loss), f"Loss not finite: {loss.item()}"
        for name, p in model.named_parameters():
            assert p.dtype == torch.float32, f"{name} dtype is {p.dtype}"

    def test_validation_with_amp_dtype(self):
        """compute_validation_loss accepts amp_dtype parameter for bf16."""
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        device = torch.device("cpu")
        val_loader = make_synthetic_dataloader(cfg.vocab_size, 16, batch_size=2, seed=99)

        val_loss = compute_validation_loss(
            model, val_loader, cfg.vocab_size, device,
            num_batches=3, amp_dtype=torch.bfloat16,
        )

        assert isinstance(val_loss, float)
        assert val_loss > 0.0
        assert not torch.isnan(torch.tensor(val_loss))


class TestGradientAccumulation:
    def test_loss_decreases_with_accumulation(self):
        """Model learns when training with gradient accumulation."""
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        muon_params, adam_params = partition_params(model)
        optimizer = Muon(muon_params, adam_params, lr=1e-3, adam_lr=1e-3)
        dataloader = make_synthetic_dataloader(cfg.vocab_size, 16, batch_size=2, seed=42)

        grad_accum_steps = 2
        initial_loss = None
        final_loss = None

        for step in range(30):
            optimizer.zero_grad()
            for _ in range(grad_accum_steps):
                batch = next(dataloader)
                x, targets = batch[:, :-1], batch[:, 1:]
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, cfg.vocab_size), targets.reshape(-1),
                )
                (loss / grad_accum_steps).backward()
            optimizer.step()
            if step == 0:
                initial_loss = loss.item()
            final_loss = loss.item()

        assert final_loss < initial_loss, (
            f"Loss should decrease: {initial_loss:.4f} -> {final_loss:.4f}"
        )


class TestGradientClipping:
    def test_clipping_caps_gradient_norm(self):
        """clip_grad_norm_ with very small max_norm clips gradients effectively."""
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)

        batch = torch.randint(0, cfg.vocab_size, (2, 17))
        x, targets = batch[:, :-1], batch[:, 1:]
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, cfg.vocab_size), targets.reshape(-1),
        )
        loss.backward()

        max_norm = 0.01  # Very small — will definitely clip
        pre_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # Pre-clipping norm should have been larger than max_norm
        assert pre_clip_norm.item() > max_norm

        # Post-clipping: all gradients should now be scaled down
        post_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        assert post_clip_norm.item() <= max_norm * 1.01  # tiny float slack


class TestCsvLogging:
    def test_csv_file_structure(self):
        """CSV log file has correct headers and parseable numeric values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "train.csv")
            checkpoint_dir = os.path.join(tmpdir, "ckpt")
            training_dir = os.path.join(os.path.dirname(__file__), "..")

            result = subprocess.run(
                [
                    sys.executable, "-m", "ct87.train",
                    "--synthetic", "--config", "tiny",
                    "--steps", "30", "--save-every", "0",
                    "--log-file", log_path,
                    "--dtype", "float32",
                    "--output-dir", checkpoint_dir,
                ],
                capture_output=True,
                text=True,
                cwd=training_dir,
                timeout=120,
            )
            assert result.returncode == 0, f"Training failed:\n{result.stderr}"

            with open(log_path) as f:
                reader = csv.reader(f)
                rows = list(reader)

            assert rows[0] == ["step", "loss", "uq_loss", "mtp_loss", "val_loss", "lr", "grad_norm", "num_thoughts", "dt_ms"]
            data_rows = rows[1:]
            # Steps 0, 10, 20 -> 3 data rows (print every 10 steps)
            assert len(data_rows) == 3
            for row in data_rows:
                assert len(row) == 9
                int(row[0])    # step is an integer
                float(row[1])  # loss
                assert row[2] == ""  # uq_loss empty (no --uq-head)
                assert row[3] == ""  # mtp_loss empty (no --mtp-head)
                assert row[4] == ""  # val_loss empty (no --val-data)
                float(row[5])  # lr
                float(row[6])  # grad_norm
                int(row[7])    # num_thoughts
                float(row[8])  # dt_ms

    def test_csv_with_uq_head(self):
        """CSV log file has populated uq_loss when --uq-head is enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "train.csv")
            checkpoint_dir = os.path.join(tmpdir, "ckpt")
            training_dir = os.path.join(os.path.dirname(__file__), "..")

            result = subprocess.run(
                [
                    sys.executable, "-m", "ct87.train",
                    "--synthetic", "--config", "tiny",
                    "--steps", "30", "--save-every", "0",
                    "--log-file", log_path,
                    "--dtype", "float32",
                    "--output-dir", checkpoint_dir,
                    "--uq-head",
                ],
                capture_output=True,
                text=True,
                cwd=training_dir,
                timeout=120,
            )
            assert result.returncode == 0, f"Training failed:\n{result.stderr}"

            with open(log_path) as f:
                reader = csv.reader(f)
                rows = list(reader)

            data_rows = rows[1:]
            assert len(data_rows) == 3
            for row in data_rows:
                assert len(row) == 9
                float(row[2])  # uq_loss should be a real number


class TestMtpHead:
    def test_mtp_loss_is_finite(self):
        """MTP head produces a finite positive loss."""
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        from ct87.mtp import MtpHead

        mtp = MtpHead(cfg, depth=4)
        batch = torch.randint(0, cfg.vocab_size, (2, 17))
        x, targets = batch[:, :-1], batch[:, 1:]

        logits, hidden = model(x, return_hidden_states=True)
        mtp_loss = mtp(hidden, targets, model.embed_tokens, model.lm_head)

        assert torch.isfinite(mtp_loss), f"MTP loss not finite: {mtp_loss.item()}"
        assert mtp_loss.item() > 0.0

    def test_mtp_loss_decreases_with_training(self):
        """MTP loss decreases when trained alongside LM loss."""
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        from ct87.mtp import MtpHead

        mtp = MtpHead(cfg, depth=2)

        muon_params, adam_params = partition_params(model)
        adam_params.extend(mtp.parameters())
        optimizer = Muon(muon_params, adam_params, lr=1e-3, adam_lr=1e-3)

        batch = torch.randint(0, cfg.vocab_size, (2, 17))
        x, targets = batch[:, :-1], batch[:, 1:]

        initial_mtp = None
        final_mtp = None

        for step in range(50):
            logits, hidden = model(x, return_hidden_states=True)
            lm_loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, cfg.vocab_size), targets.reshape(-1),
            )
            mtp_loss = mtp(hidden, targets, model.embed_tokens, model.lm_head)
            loss = lm_loss + mtp_loss

            if step == 0:
                initial_mtp = mtp_loss.item()
            final_mtp = mtp_loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        assert final_mtp < initial_mtp, (
            f"MTP loss should decrease: {initial_mtp:.4f} -> {final_mtp:.4f}"
        )

    def test_mtp_depth_validation(self):
        """MTP depth < 1 raises ValueError."""
        from ct87.mtp import MtpHead

        cfg = _tiny_config()
        with pytest.raises(ValueError, match="depth must be >= 1"):
            MtpHead(cfg, depth=0)

    def test_mtp_short_sequence_returns_zero(self):
        """When seq_len <= depth, MTP returns a zero loss (no valid targets)."""
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        from ct87.mtp import MtpHead

        mtp = MtpHead(cfg, depth=4)
        # seq_len=4 tokens, after shift targets has 3 positions, depth=4 → S=3-4<0
        batch = torch.randint(0, cfg.vocab_size, (1, 5))
        x, targets = batch[:, :-1], batch[:, 1:]

        logits, hidden = model(x, return_hidden_states=True)
        mtp_loss = mtp(hidden, targets, model.embed_tokens, model.lm_head)

        assert mtp_loss.item() == 0.0

    def test_mtp_coconut_masking(self):
        """MTP loss with think_mask differs from without it."""
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        from ct87.mtp import MtpHead

        mtp = MtpHead(cfg, depth=2)
        batch = torch.randint(0, cfg.vocab_size, (2, 17))
        x, targets = batch[:, :-1], batch[:, 1:]

        _, hidden = model(x, return_hidden_states=True)

        loss_no_mask = mtp(hidden, targets, model.embed_tokens, model.lm_head)

        # Create a think_mask that masks out some positions
        think_mask = torch.zeros(2, 16, dtype=torch.bool)
        think_mask[:, :4] = True  # first 4 positions are think tokens

        loss_with_mask = mtp(
            hidden, targets, model.embed_tokens, model.lm_head,
            think_mask=think_mask,
        )

        # Losses should differ since masked positions are excluded
        assert not torch.allclose(loss_no_mask, loss_with_mask), (
            "MTP loss should differ with think_mask applied"
        )


class TestMtpCsvLogging:
    def test_csv_with_mtp_head(self):
        """CSV log file has populated mtp_loss when --mtp-head is enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "train.csv")
            checkpoint_dir = os.path.join(tmpdir, "ckpt")
            training_dir = os.path.join(os.path.dirname(__file__), "..")

            result = subprocess.run(
                [
                    sys.executable, "-m", "ct87.train",
                    "--synthetic", "--config", "tiny",
                    "--steps", "30", "--save-every", "0",
                    "--log-file", log_path,
                    "--dtype", "float32",
                    "--output-dir", checkpoint_dir,
                    "--mtp-head", "--mtp-depth", "2",
                ],
                capture_output=True,
                text=True,
                cwd=training_dir,
                timeout=120,
            )
            assert result.returncode == 0, f"Training failed:\n{result.stderr}"

            with open(log_path) as f:
                reader = csv.reader(f)
                rows = list(reader)

            data_rows = rows[1:]
            assert len(data_rows) == 3
            for row in data_rows:
                assert len(row) == 9
                float(row[3])  # mtp_loss should be a real number
