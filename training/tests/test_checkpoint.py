"""Tests for resumable checkpoint save/restore (ZEB-127)."""
import os
import random
import tempfile

import numpy as np
import pytest
import torch

from ct87.model import HarmonyModel, HarmonyModelConfig
from ct87.train import (
    save_resumable_checkpoint,
    capture_rng_state,
    restore_rng_state,
)


class TestResumableCheckpoint:

    def test_save_creates_checkpoint_file(self):
        config = HarmonyModelConfig.tiny()
        model = HarmonyModel(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        with tempfile.TemporaryDirectory() as d:
            save_resumable_checkpoint(model, optimizer, 100, d)
            assert os.path.exists(os.path.join(d, "checkpoint.pt"))

    def test_save_retains_previous_checkpoint(self):
        config = HarmonyModelConfig.tiny()
        model = HarmonyModel(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        with tempfile.TemporaryDirectory() as d:
            save_resumable_checkpoint(model, optimizer, 100, d)
            save_resumable_checkpoint(model, optimizer, 200, d)
            assert os.path.exists(os.path.join(d, "checkpoint.pt"))
            assert os.path.exists(os.path.join(d, "checkpoint_prev.pt"))
            ckpt = torch.load(os.path.join(d, "checkpoint.pt"), weights_only=False)
            assert ckpt["step"] == 200
            prev = torch.load(os.path.join(d, "checkpoint_prev.pt"), weights_only=False)
            assert prev["step"] == 100

    def test_save_at_most_two_checkpoints(self):
        config = HarmonyModelConfig.tiny()
        model = HarmonyModel(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        with tempfile.TemporaryDirectory() as d:
            save_resumable_checkpoint(model, optimizer, 100, d)
            save_resumable_checkpoint(model, optimizer, 200, d)
            save_resumable_checkpoint(model, optimizer, 300, d)
            pt_files = [f for f in os.listdir(d) if f.startswith("checkpoint")]
            assert len(pt_files) == 2
            ckpt = torch.load(os.path.join(d, "checkpoint.pt"), weights_only=False)
            assert ckpt["step"] == 300
            prev = torch.load(os.path.join(d, "checkpoint_prev.pt"), weights_only=False)
            assert prev["step"] == 200

    def test_rng_state_round_trip(self):
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)

        state = capture_rng_state()

        # Advance RNG
        torch.randn(100)
        random.random()
        np.random.randn(100)

        # Restore and generate
        restore_rng_state(state)
        after_restore = torch.randn(10)

        # Reset to same seed and generate — should match
        torch.manual_seed(42)
        after_seed = torch.randn(10)

        assert torch.allclose(after_restore, after_seed)

    def test_model_state_round_trip(self):
        config = HarmonyModelConfig.tiny()
        model = HarmonyModel(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Run a fake step to populate optimizer state
        x = torch.randint(0, config.vocab_size, (1, 32))
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as d:
            save_resumable_checkpoint(
                model, optimizer, 50, d,
                rng_state=capture_rng_state(),
                last_val_loss=3.648,
            )

            # Create a fresh model + optimizer and load
            model2 = HarmonyModel(config)
            optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
            ckpt = torch.load(os.path.join(d, "checkpoint.pt"), weights_only=False)
            model2.load_state_dict(ckpt["model_state_dict"], strict=False)
            optimizer2.load_state_dict(ckpt["optimizer_state_dict"])

            assert ckpt["step"] == 50
            assert ckpt["last_val_loss"] == 3.648

            # Model weights should match
            for (n1, p1), (n2, p2) in zip(
                model.named_parameters(), model2.named_parameters()
            ):
                assert n1 == n2
                assert torch.allclose(p1, p2), f"Mismatch in {n1}"
