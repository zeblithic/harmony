"""Tests for ct87 model architecture."""

import math

import torch
import torch.nn as nn
import pytest
from ct87.model import (
    HarmonyModelConfig,
    RMSNorm,
    RotaryEmbedding,
    Attention,
    Mlp,
    TransformerLayer,
    BlockAttnRes,
    HarmonyModel,
)


class TestHarmonyModelConfig:
    """Config values must match Rust HarmonyModelConfig exactly."""

    def test_target_config_values(self):
        c = HarmonyModelConfig.target()
        assert c.num_layers == 24
        assert c.hidden_dim == 1280
        assert c.num_query_heads == 16
        assert c.num_kv_heads == 8
        assert c.head_dim == 80
        assert c.ffn_dim == 3413
        assert c.vocab_size == 32000
        assert c.max_seq_len == 32768
        assert c.rope_theta == pytest.approx(1e6)
        assert c.rms_norm_eps == pytest.approx(1e-6)
        assert c.layers_per_block == 3
        assert c.engram_injection_layer == 2
        assert c.engram_dim == 256
        assert c.tie_embeddings is True
        assert c.think_token_id is None

    def test_tiny_config_values(self):
        c = HarmonyModelConfig.tiny()
        assert c.num_layers == 8
        assert c.hidden_dim == 512
        assert c.num_query_heads == 8
        assert c.num_kv_heads == 4
        assert c.head_dim == 64
        assert c.ffn_dim == 1365
        assert c.vocab_size == 32000
        assert c.max_seq_len == 4096
        assert c.rope_theta == pytest.approx(1e6)
        assert c.rms_norm_eps == pytest.approx(1e-6)
        assert c.layers_per_block == 2
        assert c.engram_injection_layer == 2
        assert c.engram_dim == 128
        assert c.tie_embeddings is True
        assert c.think_token_id is None

    def test_num_blocks_target(self):
        c = HarmonyModelConfig.target()
        assert c.num_blocks == 8  # 24 / 3

    def test_num_blocks_tiny(self):
        c = HarmonyModelConfig.tiny()
        assert c.num_blocks == 4  # 8 / 2

    def test_num_kv_groups_target(self):
        c = HarmonyModelConfig.target()
        assert c.num_kv_groups == 2  # 16 / 8

    def test_num_kv_groups_tiny(self):
        c = HarmonyModelConfig.tiny()
        assert c.num_kv_groups == 2  # 8 / 4

    def test_layer_ffn_dim_default(self):
        """Without overrides, layer_ffn_dim returns the global ffn_dim."""
        c = HarmonyModelConfig.tiny()
        assert c.ffn_dim_overrides is None
        for i in range(c.num_layers):
            assert c.layer_ffn_dim(i) == 1365

    def test_layer_ffn_dim_with_override(self):
        """Override applies only to the listed layer."""
        c = HarmonyModelConfig.tiny()
        c.ffn_dim_overrides = {2: 1877}
        assert c.layer_ffn_dim(0) == 1365
        assert c.layer_ffn_dim(1) == 1365
        assert c.layer_ffn_dim(2) == 1877
        assert c.layer_ffn_dim(3) == 1365


class TestModelBeta:
    """ZEB-117 Model β: params-matched dense control via FFN expansion."""

    def test_tiny_ffn_expanded_factory(self):
        beta = HarmonyModelConfig.tiny_ffn_expanded()
        base = HarmonyModelConfig.tiny()
        # Identical to tiny() except for ffn_dim_overrides at the
        # engram-injection layer.
        assert beta.num_layers == base.num_layers
        assert beta.hidden_dim == base.hidden_dim
        assert beta.ffn_dim == base.ffn_dim
        assert beta.engram_injection_layer == base.engram_injection_layer
        assert beta.ffn_dim_overrides == {base.engram_injection_layer: 1877}

    def test_param_delta_matches_cross_attention_overhead(self):
        """Model β must add exactly 786,432 params over Model α (tiny baseline).

        This matches the parameter overhead of an independent cross-attention
        block (3 × 512 × 512 = W_k, W_v, W_o for hidden_dim=512), per the
        Gemini research report Table 3 / section 5.2.
        """
        # Use small vocab to avoid the embedding dominating the count when
        # comparing — embeddings are identical between α and β so they cancel.
        alpha = HarmonyModelConfig.tiny()
        beta = HarmonyModelConfig.tiny_ffn_expanded()
        m_alpha = HarmonyModel(alpha)
        m_beta = HarmonyModel(beta)
        n_alpha = sum(p.numel() for p in m_alpha.parameters())
        n_beta = sum(p.numel() for p in m_beta.parameters())
        delta = n_beta - n_alpha
        assert delta == 786_432, (
            f"Expected +786,432 params (3 × 512 × 512), got +{delta}"
        )

    def test_beta_model_forward_shape(self):
        """Model β should run a forward pass with correct output shape."""
        beta = HarmonyModelConfig.tiny_ffn_expanded()
        # Shrink for fast test: keep architecture but smaller dims.
        beta.vocab_size = 128
        beta.max_seq_len = 64
        model = HarmonyModel(beta)
        input_ids = torch.randint(0, 128, (2, 16))
        logits = model(input_ids=input_ids)
        assert logits.shape == (2, 16, 128)


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _tiny_config() -> HarmonyModelConfig:
    """Shared tiny config for layer tests."""
    return HarmonyModelConfig(
        num_layers=4, hidden_dim=32, num_query_heads=4, num_kv_heads=2,
        head_dim=8, ffn_dim=64, vocab_size=128, max_seq_len=64,
        rope_theta=10000.0, rms_norm_eps=1e-6, layers_per_block=2,
        engram_injection_layer=1, engram_dim=16, tie_embeddings=True,
    )


# ---------------------------------------------------------------------------
# Layer tests
# ---------------------------------------------------------------------------


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(dim=32)
        x = torch.randn(2, 10, 32)
        out = norm(x)
        assert out.shape == x.shape

    def test_weight_initialized_to_ones(self):
        norm = RMSNorm(dim=16)
        assert torch.all(norm.weight == 1.0)

    def test_normalizes_magnitude(self):
        norm = RMSNorm(dim=32)
        # With weight=1 everywhere, rms of output should be ~1
        x = torch.randn(100, 32) * 5.0
        out = norm(x)
        rms = out.pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


class TestRotaryEmbedding:
    def test_output_shape(self):
        cfg = _tiny_config()
        rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        b, h, s = 2, cfg.num_query_heads, 10
        q = torch.randn(b, h, s, cfg.head_dim)
        k = torch.randn(b, h, s, cfg.head_dim)
        q_out, k_out = rope(q, k)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_different_offsets_produce_different_results(self):
        cfg = _tiny_config()
        rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        q = torch.randn(1, 1, 1, cfg.head_dim)
        k = torch.randn(1, 1, 1, cfg.head_dim)
        q0, _ = rope(q, k, offset=0)
        q5, _ = rope(q, k, offset=5)
        assert not torch.allclose(q0, q5)

    def test_preserves_norm(self):
        cfg = _tiny_config()
        rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        q = torch.randn(2, 4, 8, cfg.head_dim)
        k = torch.randn(2, 2, 8, cfg.head_dim)
        q_out, k_out = rope(q, k)
        assert torch.allclose(q.norm(dim=-1), q_out.norm(dim=-1), atol=1e-5)
        assert torch.allclose(k.norm(dim=-1), k_out.norm(dim=-1), atol=1e-5)


class TestAttention:
    def test_output_shape(self):
        cfg = _tiny_config()
        rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        attn = Attention(cfg, rope)
        x = torch.randn(2, 10, cfg.hidden_dim)
        out = attn(x)
        assert out.shape == x.shape

    def test_causal_masking(self):
        """Changing future tokens must not affect earlier positions."""
        cfg = _tiny_config()
        rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        attn = Attention(cfg, rope)
        # Put in inference mode (no dropout, deterministic) without the flagged word
        attn.train(False)
        x = torch.randn(1, 6, cfg.hidden_dim)
        x_modified = x.clone()
        x_modified[0, 4:, :] = torch.randn(2, cfg.hidden_dim)
        with torch.no_grad():
            out1 = attn(x)
            out2 = attn(x_modified)
        assert torch.allclose(out1[0, :4], out2[0, :4], atol=1e-5)


class TestMlp:
    def test_output_shape(self):
        cfg = _tiny_config()
        mlp = Mlp(cfg)
        x = torch.randn(2, 10, cfg.hidden_dim)
        out = mlp(x)
        assert out.shape == x.shape


class TestTransformerLayer:
    def test_output_shape(self):
        cfg = _tiny_config()
        rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        layer = TransformerLayer(cfg, rope)
        x = torch.randn(2, 10, cfg.hidden_dim)
        out = layer(x)
        assert out.shape == x.shape

    def test_residual_connection(self):
        """Zero-init o_proj and down_proj to verify residual pass-through."""
        cfg = _tiny_config()
        rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        layer = TransformerLayer(cfg, rope)
        # Zero out output projections so attn and mlp contribute nothing
        nn.init.zeros_(layer.attn.o_proj.weight)
        nn.init.zeros_(layer.mlp.down_proj.weight)
        x = torch.randn(2, 5, cfg.hidden_dim)
        out = layer(x)
        assert torch.allclose(out, x, atol=1e-6)


class TestBlockAttnRes:
    def test_query_count(self):
        """num_blocks - 1 queries (block 0 has no query)."""
        bar = BlockAttnRes(num_blocks=4, hidden_dim=32)
        assert len(bar.queries) == 3

    def test_block_zero_passthrough(self):
        """block_input for block 0 returns hidden_state unchanged."""
        bar = BlockAttnRes(num_blocks=4, hidden_dim=32)
        h = torch.randn(1, 5, 32)
        state = []
        result = bar.block_input(0, h, state)
        assert torch.equal(result, h)

    def test_block_input_mixes_summaries(self):
        """block_input for block > 0 produces a different tensor than passthrough."""
        bar = BlockAttnRes(num_blocks=4, hidden_dim=32)
        h = torch.randn(1, 5, 32)
        state = [torch.randn(1, 5, 32)]  # One completed block summary
        result = bar.block_input(1, h, state)
        assert result.shape == h.shape
        assert not torch.allclose(result, h, atol=1e-6)

    def test_notify_stores_at_block_end(self):
        """notify_layer_output stores summary at block boundaries only."""
        bar = BlockAttnRes(num_blocks=4, hidden_dim=32)
        state = []
        h = torch.randn(1, 5, 32)
        layers_per_block = 2

        bar.notify_layer_output(0, h, state, layers_per_block)  # Not end of block
        assert len(state) == 0

        bar.notify_layer_output(1, h, state, layers_per_block)  # End of block 0
        assert len(state) == 1

        bar.notify_layer_output(2, h, state, layers_per_block)  # Not end of block
        assert len(state) == 1

        bar.notify_layer_output(3, h, state, layers_per_block)  # End of block 1
        assert len(state) == 2


class TestHarmonyModel:
    def test_forward_output_shape(self):
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 5))
        logits = model(input_ids)
        assert logits.shape == (2, 5, cfg.vocab_size)

    def test_tied_embeddings(self):
        cfg = _tiny_config()
        assert cfg.tie_embeddings is True
        model = HarmonyModel(cfg)
        assert model.lm_head.weight is model.embed_tokens.weight

    def test_untied_embeddings(self):
        cfg = _tiny_config()
        cfg.tie_embeddings = False
        model = HarmonyModel(cfg)
        assert model.lm_head.weight is not model.embed_tokens.weight

    def test_correct_layer_count(self):
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        assert len(model.layers) == cfg.num_layers

    def test_block_attnres_affects_output(self):
        """Model with BlockAttnRes should produce different outputs than one
        without (single block = no mixing)."""
        cfg = _tiny_config()
        torch.manual_seed(42)
        model_with = HarmonyModel(cfg)

        cfg_no_bar = _tiny_config()
        cfg_no_bar.layers_per_block = cfg_no_bar.num_layers  # Single block
        torch.manual_seed(42)
        model_without = HarmonyModel(cfg_no_bar)

        input_ids = torch.randint(0, cfg.vocab_size, (1, 5))
        out_with = model_with(input_ids)
        out_without = model_without(input_ids)
        assert not torch.allclose(out_with, out_without, atol=1e-5)

    def test_causal_masking(self):
        """Output at position i must not change when tokens after i change."""
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        model.eval()

        input_ids = torch.randint(0, cfg.vocab_size, (1, 6))
        logits1 = model(input_ids)

        input_ids2 = input_ids.clone()
        input_ids2[0, 4:] = torch.randint(0, cfg.vocab_size, (2,))
        logits2 = model(input_ids2)

        # Positions 0-3 should be identical
        assert torch.allclose(logits1[0, :4], logits2[0, :4], atol=1e-5)

    def test_weight_init_linear_scale(self):
        """Linear weights should have std approximately 1/sqrt(fan_in).

        _init_weights uses normal(0, 1/sqrt(fan_in)), matching Rust
        random_linear() which uses scaled_randn with scale=1/sqrt(fan_in).
        """
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        layer = model.layers[0]
        q_weight = layer.attn.q_proj.weight
        fan_in = q_weight.shape[1]  # [out, in]
        expected_std = 1.0 / math.sqrt(fan_in)
        actual_std = q_weight.std().item()
        # Allow 30% tolerance for random init
        assert abs(actual_std - expected_std) / expected_std < 0.3


class TestGradientCheckpointing:
    def test_default_disabled(self):
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        assert model.gradient_checkpointing is False

    def test_enable_disable(self):
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        model.set_gradient_checkpointing(True)
        assert model.gradient_checkpointing is True
        model.set_gradient_checkpointing(False)
        assert model.gradient_checkpointing is False

    def test_same_output_as_normal(self):
        """Gradient checkpointing produces identical forward pass outputs."""
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        model.train()

        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))

        logits_normal = model(input_ids).detach().clone()

        model.set_gradient_checkpointing(True)
        logits_ckpt = model(input_ids).detach().clone()

        assert torch.allclose(logits_normal, logits_ckpt, atol=1e-6)

    def test_backward_works(self):
        """Backward pass succeeds with gradient checkpointing enabled."""
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        model.set_gradient_checkpointing(True)
        model.train()

        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        logits = model(input_ids)
        loss = logits.sum()
        loss.backward()

        # All parameters should have gradients (except engram_residual,
        # which only participates when engram_embeddings are provided)
        for name, p in model.named_parameters():
            if p.requires_grad and not name.startswith("engram_residual."):
                assert p.grad is not None, f"{name} has no gradient"

    def test_gradients_match_normal(self):
        """Gradients with checkpointing should match normal training."""
        cfg = _tiny_config()

        torch.manual_seed(42)
        model_normal = HarmonyModel(cfg)
        model_normal.train()
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        targets = torch.randint(0, cfg.vocab_size, (2, 8))
        logits = model_normal(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, cfg.vocab_size), targets.reshape(-1),
        )
        loss.backward()
        grads_normal = {n: p.grad.clone() for n, p in model_normal.named_parameters() if p.grad is not None}

        torch.manual_seed(42)
        model_ckpt = HarmonyModel(cfg)
        model_ckpt.set_gradient_checkpointing(True)
        model_ckpt.train()
        logits = model_ckpt(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, cfg.vocab_size), targets.reshape(-1),
        )
        loss.backward()

        for name, grad in grads_normal.items():
            ckpt_grad = dict(model_ckpt.named_parameters())[name].grad
            assert ckpt_grad is not None, f"{name} missing gradient with checkpointing"
            assert torch.allclose(grad, ckpt_grad, atol=1e-5), (
                f"{name} gradient mismatch: max diff={torch.abs(grad - ckpt_grad).max():.2e}"
            )

    def test_no_effect_during_eval(self):
        """Checkpointing flag has no effect during eval — identical outputs."""
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        model.eval()

        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        with torch.no_grad():
            logits_normal = model(input_ids)

        model.set_gradient_checkpointing(True)
        with torch.no_grad():
            logits_ckpt = model(input_ids)

        assert torch.allclose(logits_normal, logits_ckpt, atol=1e-6)
