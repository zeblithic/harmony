"""Tests for COCONUT continuous thought training."""

import torch
import torch.nn.functional as F
import pytest

from ct87.model import HarmonyModel, HarmonyModelConfig
from ct87.coconut import (
    ThoughtNorm,
    CurriculumSchedule,
    insert_think_tokens,
    coconut_forward,
    coconut_loss,
)


def _tiny_config() -> HarmonyModelConfig:
    return HarmonyModelConfig(
        num_layers=4, hidden_dim=32, num_query_heads=4, num_kv_heads=2,
        head_dim=8, ffn_dim=64, vocab_size=128, max_seq_len=64,
        rope_theta=10000.0, rms_norm_eps=1e-6, layers_per_block=2,
        engram_injection_layer=1, engram_dim=16, tie_embeddings=True,
    )


# ---------------------------------------------------------------------------
# Model input_embeds / return_hidden_states
# ---------------------------------------------------------------------------


class TestModelInputEmbeds:
    def test_input_embeds_same_shape(self):
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        embeds = torch.randn(2, 8, cfg.hidden_dim)
        logits = model(input_embeds=embeds)
        assert logits.shape == (2, 8, cfg.vocab_size)

    def test_input_embeds_matches_embed_tokens(self):
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        model.eval()
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))

        logits_ids = model(input_ids=input_ids)
        embeds = model.embed_tokens(input_ids)
        logits_embeds = model(input_embeds=embeds)

        assert torch.allclose(logits_ids, logits_embeds, atol=1e-5)

    def test_rejects_both_ids_and_embeds(self):
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        ids = torch.randint(0, cfg.vocab_size, (1, 4))
        embeds = torch.randn(1, 4, cfg.hidden_dim)
        with pytest.raises(ValueError, match="Cannot provide both"):
            model(input_ids=ids, input_embeds=embeds)

    def test_rejects_neither(self):
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        with pytest.raises(ValueError, match="Must provide either"):
            model()

    def test_return_hidden_states_shape(self):
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))

        result = model(input_ids=input_ids, return_hidden_states=True)
        assert isinstance(result, tuple)
        logits, hidden = result
        assert logits.shape == (2, 8, cfg.vocab_size)
        assert hidden.shape == (2, 8, cfg.hidden_dim)

    def test_hidden_states_are_pre_norm(self):
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))

        _, hidden = model(input_ids=input_ids, return_hidden_states=True)
        normed = model.final_norm(hidden)
        assert not torch.allclose(hidden, normed, atol=1e-6)


# ---------------------------------------------------------------------------
# ThoughtNorm
# ---------------------------------------------------------------------------


class TestThoughtNorm:
    def test_output_shape(self):
        tn = ThoughtNorm(32)
        x = torch.randn(2, 8, 32)
        out = tn(x)
        assert out.shape == (2, 8, 32)

    def test_gate_starts_weak(self):
        tn = ThoughtNorm(32)
        gate = torch.sigmoid(tn.gate_bias).item()
        assert 0.10 < gate < 0.15  # sigmoid(-2) ≈ 0.119

    def test_learnable(self):
        tn = ThoughtNorm(32)
        assert tn.gate_bias.requires_grad


# ---------------------------------------------------------------------------
# CurriculumSchedule
# ---------------------------------------------------------------------------


class TestCurriculumSchedule:
    def test_starts_at_zero(self):
        sched = CurriculumSchedule(max_steps=4, total_train_steps=1000)
        assert sched.num_thoughts(0) == 0

    def test_reaches_max(self):
        sched = CurriculumSchedule(max_steps=4, total_train_steps=1000)
        assert sched.num_thoughts(999) == 4

    def test_monotonic(self):
        sched = CurriculumSchedule(max_steps=4, total_train_steps=1000)
        prev = 0
        for step in range(1000):
            cur = sched.num_thoughts(step)
            assert cur >= prev
            prev = cur

    def test_stage_boundaries(self):
        sched = CurriculumSchedule(max_steps=4, total_train_steps=1000)
        # stage_length = 1000 // 5 = 200
        assert sched.num_thoughts(0) == 0
        assert sched.num_thoughts(199) == 0
        assert sched.num_thoughts(200) == 1
        assert sched.num_thoughts(399) == 1
        assert sched.num_thoughts(400) == 2
        assert sched.num_thoughts(600) == 3
        assert sched.num_thoughts(800) == 4

    def test_short_run_still_ramps(self):
        """Short runs (total_steps < max_steps+1) should still start at 0."""
        sched = CurriculumSchedule(max_steps=4, total_train_steps=3)
        assert sched.num_thoughts(0) == 0
        assert sched.stage_length >= 1


# ---------------------------------------------------------------------------
# insert_think_tokens
# ---------------------------------------------------------------------------


class TestInsertThinkTokens:
    def test_prefix_shape(self):
        ids = torch.randint(0, 100, (2, 10))
        aug, mask = insert_think_tokens(ids, think_token_id=99, num_thoughts=3)
        assert aug.shape == (2, 13)
        assert mask.shape == (2, 13)

    def test_prefix_values(self):
        ids = torch.randint(0, 100, (2, 10))
        aug, _ = insert_think_tokens(ids, think_token_id=99, num_thoughts=3)
        assert (aug[:, :3] == 99).all()
        assert torch.equal(aug[:, 3:], ids)

    def test_think_mask_correct(self):
        ids = torch.randint(0, 100, (2, 10))
        _, mask = insert_think_tokens(ids, think_token_id=99, num_thoughts=3)
        assert mask[:, :3].all()
        assert not mask[:, 3:].any()

    def test_zero_thoughts_passthrough(self):
        ids = torch.randint(0, 100, (2, 10))
        aug, mask = insert_think_tokens(ids, think_token_id=99, num_thoughts=0)
        assert torch.equal(aug, ids)
        assert not mask.any()


# ---------------------------------------------------------------------------
# coconut_forward
# ---------------------------------------------------------------------------


class TestCoconutForward:
    def test_produces_valid_logits(self):
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        tn = ThoughtNorm(cfg.hidden_dim, eps=cfg.rms_norm_eps)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))

        logits, mask = coconut_forward(model, tn, input_ids, think_token_id=127, num_thoughts=2)
        assert logits.shape == (2, 10, cfg.vocab_size)  # 8 + 2 think
        assert mask.shape == (2, 10)

    def test_gradient_flows_through_feedback(self):
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        tn = ThoughtNorm(cfg.hidden_dim, eps=cfg.rms_norm_eps)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))

        logits, mask = coconut_forward(model, tn, input_ids, think_token_id=127, num_thoughts=2)
        targets = torch.randint(0, cfg.vocab_size, (2, 10))
        loss = coconut_loss(logits, targets, mask, cfg.vocab_size)
        loss.backward()

        assert tn.gate_bias.grad is not None
        assert tn.gate_bias.grad.abs() > 0

    def test_zero_thoughts_matches_standard(self):
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        model.eval()
        tn = ThoughtNorm(cfg.hidden_dim, eps=cfg.rms_norm_eps)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))

        logits_coconut, mask = coconut_forward(model, tn, input_ids, think_token_id=127, num_thoughts=0)
        logits_standard = model(input_ids=input_ids)

        assert not mask.any()
        assert torch.allclose(logits_coconut, logits_standard, atol=1e-5)

    def test_with_engram(self):
        from ct87.engram import EngramTable

        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        tn = ThoughtNorm(cfg.hidden_dim, eps=cfg.rms_norm_eps)

        table = torch.randn(100, cfg.engram_dim)
        engram_table = EngramTable(table, hash_seeds=[42, 99])

        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        engram_emb = engram_table.lookup_batch(input_ids)

        logits, _ = coconut_forward(
            model, tn, input_ids, think_token_id=127, num_thoughts=2,
            engram_embeddings=engram_emb,
        )
        assert logits.shape == (2, 10, cfg.vocab_size)


# ---------------------------------------------------------------------------
# coconut_loss
# ---------------------------------------------------------------------------


class TestCoconutLoss:
    def test_excludes_think_positions(self):
        torch.manual_seed(42)
        vocab_size = 10
        logits = torch.randn(2, 6, vocab_size, requires_grad=True)
        targets = torch.randint(0, vocab_size, (2, 6))
        think_mask = torch.zeros(2, 6, dtype=torch.bool)
        think_mask[:, :2] = True  # first 2 positions are think

        loss = coconut_loss(logits, targets, think_mask, vocab_size)
        loss.backward()

        # Gradients should be zero at think positions
        assert (logits.grad[:, :2, :] == 0).all()
        # Gradients should be non-zero at non-think positions
        assert logits.grad[:, 2:, :].abs().sum() > 0

    def test_matches_standard_on_no_thinks(self):
        torch.manual_seed(42)
        vocab_size = 10
        logits = torch.randn(2, 6, vocab_size)
        targets = torch.randint(0, vocab_size, (2, 6))
        think_mask = torch.zeros(2, 6, dtype=torch.bool)

        loss_coconut = coconut_loss(logits, targets, think_mask, vocab_size)
        loss_standard = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))

        assert loss_coconut.item() == pytest.approx(loss_standard.item(), rel=1e-6)

    def test_gradient_zero_at_think_positions(self):
        torch.manual_seed(42)
        vocab_size = 10
        logits = torch.randn(1, 5, vocab_size, requires_grad=True)
        targets = torch.randint(0, vocab_size, (1, 5))
        think_mask = torch.tensor([[True, True, False, False, False]])

        loss = coconut_loss(logits, targets, think_mask, vocab_size)
        loss.backward()

        assert (logits.grad[0, :2, :] == 0).all()
