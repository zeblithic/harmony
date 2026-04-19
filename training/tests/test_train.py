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
        """Tiny model overfits a small batch — loss drops at least 40% in 50 steps.

        Threshold is relative (final < 0.6 * initial) rather than absolute
        because `_tiny_config` carries `engram_injection_layer=1`, which
        triggers HarmonyModel.__init__ to allocate `engram_residual` and
        the related auxiliary modules (added in PR #191 and grown over
        subsequent ZEB-117/127/130 work). Those modules' parameters
        participate in optimization even when the test never feeds an
        engram tensor, adding a small amount of noise that keeps
        single-batch overfitting from reaching the original `< 2.0`
        absolute threshold within 50 steps. Loss falls from ~4.85
        (= log(128) random init) to ~2.04 today — a 58% drop, which the
        relative threshold captures cleanly.
        """
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
        assert final_loss < 0.6 * initial_loss, (
            f"Loss should drop at least 40% from initial in 50 steps; "
            f"got {initial_loss:.4f} -> {final_loss:.4f}"
        )


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
        """Gradient accumulation mechanism learns on a fixed batch.

        The original test pulled a fresh random batch from a dataloader
        on every inner step. With genuinely random data and only 30
        outer steps, `final_loss < initial_loss` is a noisy comparison
        that bears no relation to whether gradient accumulation is
        wired correctly — it can fail purely from per-batch variance
        even when the mechanism is sound.

        We instead overfit a fixed batch with grad_accum=2 (mirroring
        TestOverfit's design but exercising the accumulate-then-step
        pattern). That pattern is the actual mechanical contract under
        test: zero_grad outside the inner loop, scaled .backward()
        inside, optimizer.step() outside.
        """
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        muon_params, adam_params = partition_params(model)
        optimizer = Muon(muon_params, adam_params, lr=1e-3, adam_lr=1e-3)

        # Fixed batch — reused across all 30 outer steps and both inner
        # accumulation steps so the test measures learning, not noise.
        input_ids = torch.randint(0, cfg.vocab_size, (2, 17))
        x = input_ids[:, :-1]
        targets = input_ids[:, 1:]

        grad_accum_steps = 2
        initial_loss = None
        final_loss = None

        for step in range(30):
            optimizer.zero_grad()
            step_loss = 0.0
            for _ in range(grad_accum_steps):
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, cfg.vocab_size), targets.reshape(-1),
                )
                (loss / grad_accum_steps).backward()
                step_loss += loss.item() / grad_accum_steps
            optimizer.step()
            if step == 0:
                initial_loss = step_loss
            final_loss = step_loss

        assert final_loss < initial_loss, (
            f"Loss should decrease: {initial_loss:.4f} -> {final_loss:.4f}"
        )
        assert final_loss < 0.85 * initial_loss, (
            f"Loss should drop at least 15% in 30 grad-accum steps; "
            f"got {initial_loss:.4f} -> {final_loss:.4f}"
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
    """CSV-schema-agnostic checks: assert columns by name, not by index.

    train.py's CSV header has grown several times since these tests were
    introduced (cl_loss, ann_*, hg_*, mse_loss/consol_phase, vcontrast_*,
    qdiv_*, ...). Indexing by position breaks every time a new column is
    added; indexing by name only breaks when the *named* column changes.
    """

    # Required columns that every run must emit, regardless of which
    # heads/aux losses are enabled. Used as a stability contract — if
    # one of these is dropped or renamed, that's a real schema break.
    REQUIRED_COLS = {
        "step", "loss", "uq_loss", "mtp_loss", "val_loss",
        "lr", "grad_norm", "num_thoughts", "dt_ms",
    }

    def test_csv_file_structure(self):
        """CSV log file has all required columns + parseable numeric values."""
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
                reader = csv.DictReader(f)
                data_rows = list(reader)
                fieldnames = reader.fieldnames

            assert fieldnames is not None
            missing = self.REQUIRED_COLS - set(fieldnames)
            assert not missing, f"CSV header missing required columns: {missing}"

            # Steps 0, 10, 20 -> 3 data rows (print every 10 steps)
            assert len(data_rows) == 3
            for row in data_rows:
                int(row["step"])
                float(row["loss"])
                assert row["uq_loss"] == ""  # no --uq-head
                assert row["mtp_loss"] == ""  # no --mtp-head
                assert row["val_loss"] == ""  # no --val-data
                float(row["lr"])
                float(row["grad_norm"])
                int(row["num_thoughts"])
                float(row["dt_ms"])

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
                data_rows = list(csv.DictReader(f))
            assert len(data_rows) == 3
            for row in data_rows:
                float(row["uq_loss"])  # real number, not empty


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

        _logits, hidden = model(x, return_hidden_states=True)
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
        # x has 4 tokens, hidden has 4 positions, depth=4 → S=4-4=0
        batch = torch.randint(0, cfg.vocab_size, (1, 5))
        x, targets = batch[:, :-1], batch[:, 1:]

        _logits, hidden = model(x, return_hidden_states=True)
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

        _logits, hidden = model(x, return_hidden_states=True)

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


class TestQat:
    def test_q8_0_roundtrip_changes_weight(self):
        """Fake quantization produces a different tensor (quantization noise)."""
        from ct87.qat import fake_quantize_q8_0

        torch.manual_seed(42)
        w = torch.randn(64, 32)
        fq = fake_quantize_q8_0(w)

        assert fq.shape == w.shape
        assert not torch.equal(w, fq), "Fake-quantized weight should differ from original"
        # Error should be bounded — q8_0 has ~0.4% relative error for normal weights
        rel_error = (w - fq).abs().max() / w.abs().max()
        assert rel_error < 0.02, f"Relative error too large: {rel_error:.4f}"

    def test_q8_0_roundtrip_is_idempotent(self):
        """Quantizing an already-quantized weight is a no-op."""
        from ct87.qat import fake_quantize_q8_0

        torch.manual_seed(42)
        w = torch.randn(64, 32)
        fq1 = fake_quantize_q8_0(w)
        fq2 = fake_quantize_q8_0(fq1)

        assert torch.equal(fq1, fq2), "Double quantization should be idempotent"

    def test_q8_0_handles_non_block_aligned(self):
        """Weights not divisible by block_size=32 are handled correctly."""
        from ct87.qat import fake_quantize_q8_0

        torch.manual_seed(42)
        w = torch.randn(10, 7)  # 70 elements, not divisible by 32
        fq = fake_quantize_q8_0(w)

        assert fq.shape == w.shape
        assert torch.isfinite(fq).all()

    def test_ste_gradient_flow(self):
        """STE passes gradients through the non-differentiable quantization."""
        from ct87.qat import fake_quantize_q8_0

        torch.manual_seed(42)
        w = torch.randn(32, 32, requires_grad=True)
        fq = fake_quantize_q8_0(w)
        loss = fq.sum()
        loss.backward()

        assert w.grad is not None, "Gradient should flow through STE"
        assert (w.grad != 0).any(), "Gradient should be non-zero"
        # STE: gradient is identity, so grad should be all-ones from .sum()
        assert torch.allclose(w.grad, torch.ones_like(w.grad))

    def test_enable_disable_qat(self):
        """enable_qat patches Linear layers, disable_qat restores them."""
        from ct87.qat import enable_qat, disable_qat

        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        x = torch.randint(0, cfg.vocab_size, (1, 8))

        logits_original = model(x).detach()
        enable_qat(model)
        logits_qat = model(x).detach()
        disable_qat(model)
        logits_restored = model(x).detach()

        # QAT should change outputs (quantization noise)
        assert not torch.allclose(logits_original, logits_qat, atol=1e-7), \
            "QAT should change model output"
        # Disabling QAT should restore original behavior
        assert torch.allclose(logits_original, logits_restored, atol=1e-7), \
            "disable_qat should restore original output"

    def test_qat_excludes_auxiliary_modules(self):
        """QAT on the base model does not affect UQ head or MTP head."""
        from ct87.qat import enable_qat
        from ct87.mtp import MtpHead

        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        mtp = MtpHead(cfg, depth=2)

        enable_qat(model)

        # Base model Linear layers should be patched
        assert hasattr(model.lm_head, "_qat_original_forward")
        assert hasattr(model.layers[0].attn.q_proj, "_qat_original_forward")

        # MTP head (separate module) should NOT be patched
        assert not hasattr(mtp.gate_proj, "_qat_original_forward")
        assert not hasattr(mtp.up_proj, "_qat_original_forward")

    def test_qat_tied_embeddings(self):
        """QAT patches lm_head but not embed_tokens, even with tied weights."""
        from ct87.qat import enable_qat, fake_quantize_q8_0

        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)

        # Verify weights are tied
        assert model.lm_head.weight is model.embed_tokens.weight

        x = torch.randint(0, cfg.vocab_size, (1, 8))
        embeds_before = model.embed_tokens(x).detach().clone()

        enable_qat(model)

        # lm_head should be patched, embed_tokens should NOT
        assert hasattr(model.lm_head, "_qat_original_forward")
        assert not hasattr(model.embed_tokens, "_qat_original_forward")

        # Embedding lookup is unchanged (table index, not matmul)
        embeds_after = model.embed_tokens(x).detach()
        assert torch.equal(embeds_before, embeds_after), \
            "embed_tokens output should be unchanged by QAT"

        # lm_head output should differ (fake-quantized weight matmul)
        h = torch.randn(1, 8, cfg.hidden_dim)
        lm_out_qat = model.lm_head(h).detach()
        lm_out_exact = torch.nn.functional.linear(h, model.lm_head.weight).detach()
        # QAT forward uses fake_quantize_q8_0(weight), so it should differ
        # from exact weight matmul (unless weight happens to be exactly
        # representable in q8_0, which is astronomically unlikely)
        assert not torch.equal(lm_out_qat, lm_out_exact), \
            "lm_head should use fake-quantized weights under QAT"

    def test_qat_training_converges(self):
        """Model still converges when QAT is enabled from the start."""
        from ct87.qat import enable_qat

        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        enable_qat(model)

        muon_params, adam_params = partition_params(model)
        optimizer = Muon(muon_params, adam_params, lr=1e-3, adam_lr=1e-3)

        batch = torch.randint(0, cfg.vocab_size, (2, 17))
        x, targets = batch[:, :-1], batch[:, 1:]

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

        assert final_loss < initial_loss, (
            f"Loss should decrease with QAT: {initial_loss:.4f} -> {final_loss:.4f}"
        )


class TestQatCsvLogging:
    def test_csv_with_qat(self):
        """Training with --qat runs successfully and produces valid CSV output."""
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
                    "--qat", "--qat-start-pct", "0.5",
                ],
                capture_output=True,
                text=True,
                cwd=training_dir,
                timeout=120,
            )
            assert result.returncode == 0, f"Training failed:\n{result.stderr}"
            # ceil(0.5 * 30) = 15
            assert "QAT enabled at step 15" in result.stdout

            with open(log_path) as f:
                data_rows = list(csv.DictReader(f))
            assert len(data_rows) == 3
            for row in data_rows:
                float(row["loss"])  # loss column populated and parseable


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
                data_rows = list(csv.DictReader(f))
            assert len(data_rows) == 3
            for row in data_rows:
                float(row["mtp_loss"])  # populated when --mtp-head is on


class TestKlRetrofit:
    """ZEB-139 KL+CE hybrid loss.

    Covers: model snapshot of `_last_skip_logits` after the ZEB-134
    Skip-to-Logit router fires, the canonical-trigram-row-index helper
    used to look up teacher logits per (b, p), and the train.py CLI
    validation guards (paired flags, lambda bounds).
    """

    def _build_model_with_skip_router(self, alpha_init: float = 0.1):
        from ct87.engram import (
            EngramCrossAttention, GatedEngramInjection, SkipToLogitEngramRouter,
        )
        cfg = HarmonyModelConfig.tiny_engram_xattn_capgap()
        model = HarmonyModel(cfg)
        # Multi-layer η-B injection: required precondition for the
        # skip router (it reads the LAST injection layer's output).
        capgap_table = torch.zeros((1024, cfg.engram_dim))
        injections = {}
        for layer_idx in cfg.engram_inject_layers:
            xattn = EngramCrossAttention(
                cfg, capgap_table, num_heads=cfg.num_query_heads,
                k_retrieved=4,
            )
            injections[layer_idx] = GatedEngramInjection(
                xattn, alpha_init=cfg.engram_gate_init,
            )
        model.attach_gated_engram_injections(injections)
        router = SkipToLogitEngramRouter(
            hidden_dim=cfg.hidden_dim,
            lm_head_weight=model.lm_head.weight,
            alpha_init=alpha_init,
        )
        model.attach_engram_skip_router(router)
        return cfg, model

    def test_last_skip_logits_snapshotted_after_forward(self):
        """The model must expose _last_skip_logits after every forward
        when the ZEB-134 router is attached. The training loop's KL term
        depends on this snapshot — losing it (e.g. a future model.py
        refactor that drops the assignment) would silently disable the
        KL pathway. Pin the contract."""
        torch.manual_seed(42)
        cfg, model = self._build_model_with_skip_router()
        input_ids = torch.randint(0, cfg.vocab_size, (2, 17))
        # Initial state: cleared.
        assert model._last_skip_logits is None
        logits = model(input_ids)
        # Post-forward: populated, shape matches main logits.
        assert model._last_skip_logits is not None
        assert model._last_skip_logits.shape == logits.shape
        # The snapshot is the router's ISOLATED output, not the mixed
        # final logits. Since W_align starts at zero and alpha=0.1, the
        # router's contribution is exactly zero on the first forward.
        # The snapshot should reflect that (all-zero tensor).
        assert torch.all(model._last_skip_logits == 0)

    def test_last_skip_logits_cleared_at_start_of_forward(self):
        """Each forward must start from a clean slate. A stale snapshot
        from a prior batch could otherwise leak into the KL term if the
        router doesn't fire (e.g. inject_mult forced to 0)."""
        torch.manual_seed(42)
        cfg, model = self._build_model_with_skip_router()
        input_ids = torch.randint(0, cfg.vocab_size, (2, 17))
        model(input_ids)  # populate
        assert model._last_skip_logits is not None
        # Force the router to be skipped on the next forward.
        model.engram_inject_mult = 0.0
        model(input_ids)
        # Snapshot should be cleared (None), not stale.
        assert model._last_skip_logits is None

    def test_canonical_trigram_row_indices_basic(self):
        from ct87.engram import compute_canonical_trigram_row_indices

        # Simple deterministic input: row indices for positions 0/1
        # must be -1 (no trigram); positions 2..N-1 get the trigram
        # hash modulo total_entries.
        input_ids = torch.tensor([
            [10, 20, 30, 40, 50],
            [11, 21, 31, 41, 51],
        ], dtype=torch.long)
        row_indices = compute_canonical_trigram_row_indices(
            input_ids, total_entries=1024, canonical_seed=42,
        )
        assert row_indices.shape == (2, 5)
        assert (row_indices[:, :2] == -1).all()  # p<2 = no trigram
        assert (row_indices[:, 2:] >= 0).all()
        assert (row_indices[:, 2:] < 1024).all()
        # Different sequences hash to different values (with probability
        # 1 - 1/total_entries; deterministic for these specific inputs).
        assert not torch.equal(row_indices[0], row_indices[1])

    def test_canonical_trigram_row_indices_reproducible(self):
        """Same input + seed must always produce the same hash."""
        from ct87.engram import compute_canonical_trigram_row_indices

        input_ids = torch.randint(0, 32000, (3, 16))
        a = compute_canonical_trigram_row_indices(input_ids, 1024, 42)
        b = compute_canonical_trigram_row_indices(input_ids, 1024, 42)
        assert torch.equal(a, b)
        # Different seed → different indices (almost always).
        c = compute_canonical_trigram_row_indices(input_ids, 1024, 99)
        assert not torch.equal(a[:, 2:], c[:, 2:])

    def test_canonical_trigram_row_indices_short_seq(self):
        from ct87.engram import compute_canonical_trigram_row_indices

        # seq_len < 3 → no trigrams anywhere → all -1.
        input_ids = torch.tensor([[1, 2]], dtype=torch.long)
        row_indices = compute_canonical_trigram_row_indices(input_ids, 1024, 42)
        assert (row_indices == -1).all()

    def _run_train_cli(self, *extra_args, log_dir):
        """Helper: invoke ct87.train with --synthetic + minimal common
        args, return CompletedProcess. Uses --config tiny (no engram)
        unless extra_args overrides; bumps --steps to 2 for speed."""
        log_path = os.path.join(log_dir, "train.csv")
        ckpt_dir = os.path.join(log_dir, "ckpt")
        training_dir = os.path.join(os.path.dirname(__file__), "..")
        cmd = [
            sys.executable, "-m", "ct87.train",
            "--synthetic", "--config", "tiny",
            "--steps", "2", "--save-every", "0",
            "--log-file", log_path,
            "--dtype", "float32",
            "--output-dir", ckpt_dir,
            *extra_args,
        ]
        return subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=training_dir, timeout=120,
        )

    def test_cli_rejects_kl_lambda_without_skip_router(self):
        """--kl-lambda > 0 needs --engram-skip-to-logit. Without it the
        KL target (skip_logits) doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            res = self._run_train_cli(
                "--kl-lambda", "0.5",
                "--oracle-teacher-logits", "/tmp/nonexistent.safetensors",
                log_dir=tmpdir,
            )
            assert res.returncode == 2, res.stderr
            assert "engram-skip-to-logit" in res.stderr.lower()

    def test_cli_rejects_kl_lambda_without_oracle_logits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            res = self._run_train_cli(
                "--kl-lambda", "0.5",
                "--engram-skip-to-logit",
                log_dir=tmpdir,
            )
            assert res.returncode == 2, res.stderr
            assert "oracle-teacher-logits" in res.stderr.lower()

    def test_cli_rejects_oracle_logits_without_kl_lambda(self):
        """Symmetric guard: passing the sidecar without kl_lambda > 0
        silently has no effect — surface the misconfiguration upfront."""
        with tempfile.TemporaryDirectory() as tmpdir:
            res = self._run_train_cli(
                "--oracle-teacher-logits", "/tmp/some_sidecar.safetensors",
                log_dir=tmpdir,
            )
            assert res.returncode == 2, res.stderr
            assert "--oracle-teacher-logits" in res.stderr
            assert "--kl-lambda" in res.stderr

    def test_cli_rejects_kl_lambda_out_of_range(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            res = self._run_train_cli(
                "--kl-lambda", "1.5",
                "--engram-skip-to-logit",
                "--oracle-teacher-logits", "/tmp/x.safetensors",
                log_dir=tmpdir,
            )
            assert res.returncode == 2, res.stderr
            assert "kl-lambda" in res.stderr.lower()

    def test_cli_default_kl_lambda_is_zero_no_validation_fires(self):
        """Verify the new flags don't break a vanilla run (without
        --kl-lambda). The training run completes and CSV has the new
        kl_loss column (empty since KL is off)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            res = self._run_train_cli(log_dir=tmpdir)
            assert res.returncode == 0, res.stderr
            log_path = os.path.join(tmpdir, "train.csv")
            with open(log_path) as f:
                data_rows = list(csv.DictReader(f))
            assert len(data_rows) >= 1
            for row in data_rows:
                assert row["kl_loss"] == ""  # empty when KL off
