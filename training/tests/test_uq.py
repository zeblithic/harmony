"""Tests for UQ head training."""

import math

import torch
import torch.nn.functional as F
import pytest

from ct87.model import HarmonyModel, HarmonyModelConfig
from ct87.uq import (
    UqHead,
    UqFeatureConfig,
    LayerNormCollector,
    extract_uq_features,
    compute_pseudo_labels,
    compute_uq_loss,
)


def _tiny_config() -> HarmonyModelConfig:
    return HarmonyModelConfig(
        num_layers=4, hidden_dim=32, num_query_heads=4, num_kv_heads=2,
        head_dim=8, ffn_dim=64, vocab_size=128, max_seq_len=64,
        rope_theta=10000.0, rms_norm_eps=1e-6, layers_per_block=2,
        engram_injection_layer=1, engram_dim=16, tie_embeddings=True,
    )


# ---------------------------------------------------------------------------
# UqFeatureConfig
# ---------------------------------------------------------------------------


class TestUqFeatureConfig:
    def test_for_24_layers(self):
        cfg = UqFeatureConfig.for_num_layers(24)
        assert cfg.norm_layers == [0, 8, 16, 23]
        assert cfg.top_k == 10

    def test_for_8_layers(self):
        cfg = UqFeatureConfig.for_num_layers(8)
        assert cfg.norm_layers == [0, 2, 5, 7]

    def test_for_4_layers(self):
        cfg = UqFeatureConfig.for_num_layers(4)
        assert cfg.norm_layers == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# LayerNormCollector
# ---------------------------------------------------------------------------


class TestLayerNormCollector:
    def test_collects_norms_at_specified_layers(self):
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        layer_indices = [0, 1, 2, 3]

        with LayerNormCollector(model, layer_indices) as collector:
            model(input_ids=input_ids)
        norms = collector.get_norms()

        assert set(norms.keys()) == {0, 1, 2, 3}
        for idx in layer_indices:
            assert norms[idx].shape == (2, 8)

    def test_norms_are_positive(self):
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))

        with LayerNormCollector(model, [0, 3]) as collector:
            model(input_ids=input_ids)
        norms = collector.get_norms()

        for t in norms.values():
            assert (t > 0).all()

    def test_hooks_cleaned_up_after_exit(self):
        cfg = _tiny_config()
        model = HarmonyModel(cfg)

        with LayerNormCollector(model, [0, 1]) as collector:
            assert len(collector._hooks) == 2

        assert len(collector._hooks) == 0

    def test_works_with_gradient_checkpointing(self):
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        model.set_gradient_checkpointing(True)
        model.train()
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))

        with LayerNormCollector(model, [0, 3]) as collector:
            logits = model(input_ids=input_ids)
        norms = collector.get_norms()

        # Norms should be valid even with checkpointing
        assert 0 in norms and 3 in norms
        for t in norms.values():
            assert t.shape == (2, 8)
            assert (t > 0).all()

        # Backward should not break anything
        loss = logits.sum()
        loss.backward()

    def test_coconut_gets_pass2_norms(self):
        """With COCONUT 2-pass, collector returns norms from pass 2."""
        from ct87.coconut import ThoughtNorm, coconut_forward

        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        tn = ThoughtNorm(cfg.hidden_dim, eps=cfg.rms_norm_eps)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        num_thoughts = 2

        with LayerNormCollector(model, [0, 1, 2, 3]) as collector:
            _logits, _think_mask = coconut_forward(
                model, tn, input_ids,
                think_token_id=127, num_thoughts=num_thoughts,
            )
        norms = collector.get_norms()

        # Norms should have augmented seq_len (8 + 2 think tokens = 10)
        for t in norms.values():
            assert t.shape == (2, 10)


# ---------------------------------------------------------------------------
# extract_uq_features
# ---------------------------------------------------------------------------


class TestExtractUqFeatures:
    def test_output_shape(self):
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        uq_cfg = UqFeatureConfig.for_num_layers(cfg.num_layers)

        with LayerNormCollector(model, uq_cfg.norm_layers) as collector:
            logits = model(input_ids=input_ids)
        norms = collector.get_norms()

        with torch.no_grad():
            features, _probs = extract_uq_features(norms, logits, uq_cfg)
        assert features.shape == (2, 8, 8)

    def test_slope_formula(self):
        """f5 = (3*(f4-f1) + (f3-f2)) / 10."""
        norms = {
            0: torch.tensor([[1.0, 1.0]]),
            1: torch.tensor([[2.0, 2.0]]),
            2: torch.tensor([[3.0, 3.0]]),
            3: torch.tensor([[4.0, 4.0]]),
        }
        logits = torch.zeros(1, 2, 10)
        cfg = UqFeatureConfig(norm_layers=[0, 1, 2, 3], top_k=5)
        features, _probs = extract_uq_features(norms, logits, cfg)

        f1, f4 = 1.0, 4.0
        f2, f3 = 2.0, 3.0
        expected_slope = (3.0 * (f4 - f1) + (f3 - f2)) / 10.0
        assert features[0, 0, 4].item() == pytest.approx(expected_slope, abs=1e-6)

    def test_entropy_uniform(self):
        """Uniform logits -> entropy ~= ln(vocab_size)."""
        vocab = 100
        logits = torch.zeros(1, 1, vocab)
        norms = {i: torch.ones(1, 1) for i in range(4)}
        cfg = UqFeatureConfig(norm_layers=[0, 1, 2, 3], top_k=10)
        features, _probs = extract_uq_features(norms, logits, cfg)
        expected = math.log(vocab)
        assert features[0, 0, 5].item() == pytest.approx(expected, abs=1e-3)

    def test_entropy_peaked(self):
        """One-hot logits -> entropy ~= 0."""
        logits = torch.zeros(1, 1, 50)
        logits[0, 0, 0] = 1000.0
        norms = {i: torch.ones(1, 1) for i in range(4)}
        cfg = UqFeatureConfig(norm_layers=[0, 1, 2, 3], top_k=10)
        features, _probs = extract_uq_features(norms, logits, cfg)
        assert features[0, 0, 5].item() < 1e-3

    def test_f8_always_zero(self):
        logits = torch.randn(2, 4, 50)
        norms = {i: torch.ones(2, 4) for i in range(4)}
        cfg = UqFeatureConfig(norm_layers=[0, 1, 2, 3], top_k=10)
        features, _probs = extract_uq_features(norms, logits, cfg)
        assert (features[..., 7] == 0).all()


# ---------------------------------------------------------------------------
# compute_pseudo_labels
# ---------------------------------------------------------------------------


class TestComputePseudoLabels:
    def test_output_shapes(self):
        logits = torch.randn(2, 8, 50)
        targets = torch.randint(0, 50, (2, 8))
        features = torch.randn(2, 8, 8)
        labels, conf = compute_pseudo_labels(logits, targets, features)
        assert labels.shape == (2, 8)
        assert labels.dtype == torch.long
        assert conf.shape == (2, 8)
        assert conf.dtype == torch.float

    def test_confident_label(self):
        """When softmax(logits)[target] > 0.5 -> Confident (0)."""
        logits = torch.zeros(1, 1, 10)
        logits[0, 0, 3] = 100.0  # token 3 gets ~100% prob
        targets = torch.tensor([[3]])
        features = torch.ones(1, 1, 8)  # norms above collapse threshold
        labels, _ = compute_pseudo_labels(logits, targets, features)
        assert labels[0, 0].item() == 0

    def test_spectral_collapse_label(self):
        """When all norms < 0.1 -> SpectralCollapse (2), even if model is 'confident'."""
        logits = torch.zeros(1, 1, 10)
        logits[0, 0, 3] = 100.0
        targets = torch.tensor([[3]])
        # Set f1-f4 to very small norms
        features = torch.zeros(1, 1, 8)
        features[0, 0, :4] = 0.01  # all norms tiny
        labels, _ = compute_pseudo_labels(logits, targets, features)
        assert labels[0, 0].item() == 2  # SpectralCollapse overrides Confident

    def test_high_volume_label(self):
        """High entropy + dispersed probability -> HighVolume (1)."""
        # Uniform logits: entropy = ln(50) ~= 3.9 > ln(10) ~= 2.3, top-10 mass = 0.2
        logits = torch.zeros(1, 1, 50)
        targets = torch.tensor([[0]])
        features = torch.ones(1, 1, 8)
        features[0, 0, 5] = 4.0  # high entropy
        features[0, 0, 6] = 0.2  # low top-k mass
        labels, _ = compute_pseudo_labels(logits, targets, features)
        assert labels[0, 0].item() == 1

    def test_uncertain_label(self):
        """Default fallback -> Uncertain (3)."""
        # Moderate entropy, moderate mass, non-collapsed norms, model not confident
        logits = torch.randn(1, 1, 50)
        targets = torch.tensor([[49]])  # unlikely to be argmax
        features = torch.ones(1, 1, 8)
        features[0, 0, 5] = 1.0  # low entropy (below ln(10))
        features[0, 0, 6] = 0.8  # high top-k mass (not HighVolume)
        labels, _ = compute_pseudo_labels(logits, targets, features)
        assert labels[0, 0].item() == 3

    def test_confidence_target_correct_prediction(self):
        logits = torch.zeros(1, 1, 10)
        logits[0, 0, 5] = 100.0  # argmax = 5
        targets = torch.tensor([[5]])
        features = torch.ones(1, 1, 8)
        _, conf = compute_pseudo_labels(logits, targets, features)
        assert conf[0, 0].item() == 1.0

    def test_confidence_target_wrong_prediction(self):
        logits = torch.zeros(1, 1, 10)
        logits[0, 0, 5] = 100.0  # argmax = 5
        targets = torch.tensor([[3]])  # target != argmax
        features = torch.ones(1, 1, 8)
        _, conf = compute_pseudo_labels(logits, targets, features)
        assert conf[0, 0].item() == 0.0


# ---------------------------------------------------------------------------
# UqHead
# ---------------------------------------------------------------------------


class TestUqHead:
    def test_param_count(self):
        head = UqHead()
        total = sum(p.numel() for p in head.parameters())
        # fc1: 8*32+32=288, fc2: 32*4+4=132, conf: 8*1+1=9, total=429
        assert total == 429

    def test_output_shapes(self):
        head = UqHead()
        features = torch.randn(2, 8, 8)
        class_logits, confidence = head(features)
        assert class_logits.shape == (2, 8, 4)
        assert confidence.shape == (2, 8, 1)

    def test_confidence_logits_shape(self):
        head = UqHead()
        features = torch.randn(3, 5, 8)
        _, confidence_logits = head(features)
        # Raw logits, sigmoid applied in loss function
        assert confidence_logits.shape == (3, 5, 1)
        # Verify sigmoid produces valid probabilities
        probs = torch.sigmoid(confidence_logits)
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_gradient_flows(self):
        head = UqHead()
        features = torch.randn(2, 4, 8)
        class_logits, confidence = head(features)
        loss = class_logits.sum() + confidence.sum()
        loss.backward()
        for p in head.parameters():
            assert p.grad is not None
            assert p.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# compute_uq_loss
# ---------------------------------------------------------------------------


class TestComputeUqLoss:
    def test_finite_loss(self):
        class_logits = torch.randn(2, 6, 4)
        confidence_logits = torch.randn(2, 6, 1)
        class_labels = torch.randint(0, 4, (2, 6))
        conf_targets = torch.randint(0, 2, (2, 6)).float()
        loss = compute_uq_loss(class_logits, confidence_logits, class_labels, conf_targets)
        assert torch.isfinite(loss)

    def test_think_mask_excludes_positions(self):
        torch.manual_seed(42)
        class_logits = torch.randn(1, 5, 4, requires_grad=True)
        confidence_logits = torch.randn(1, 5, 1, requires_grad=True)
        class_labels = torch.randint(0, 4, (1, 5))
        conf_targets = torch.randint(0, 2, (1, 5)).float()
        think_mask = torch.tensor([[True, True, False, False, False]])

        loss = compute_uq_loss(
            class_logits, confidence_logits, class_labels, conf_targets,
            think_mask=think_mask,
        )
        loss.backward()

        # Gradients at think positions should be zero
        assert (class_logits.grad[0, :2, :] == 0).all()

    def test_fully_masked_returns_zero(self):
        """When think_mask excludes all positions, loss is zero (not NaN)."""
        class_logits = torch.randn(1, 4, 4, requires_grad=True)
        confidence_logits = torch.randn(1, 4, 1, requires_grad=True)
        class_labels = torch.randint(0, 4, (1, 4))
        conf_targets = torch.randint(0, 2, (1, 4)).float()
        think_mask = torch.ones(1, 4, dtype=torch.bool)  # all masked

        loss = compute_uq_loss(
            class_logits, confidence_logits, class_labels, conf_targets,
            think_mask=think_mask,
        )
        assert torch.isfinite(loss)
        assert loss.item() == 0.0


# ---------------------------------------------------------------------------
# End-to-end integration
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_uq_head_trains_on_tiny_model(self):
        """Full pipeline: model forward -> collect norms -> extract features ->
        pseudo-labels -> UQ loss -> backward. Verify UQ loss decreases."""
        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        model.train()
        uq_head = UqHead()
        uq_cfg = UqFeatureConfig.for_num_layers(cfg.num_layers)

        optimizer = torch.optim.Adam(uq_head.parameters(), lr=1e-3)

        losses = []
        for _ in range(20):
            input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
            targets = torch.randint(0, cfg.vocab_size, (2, 8))

            with LayerNormCollector(model, uq_cfg.norm_layers) as collector:
                logits = model(input_ids=input_ids)
            norms = collector.get_norms()

            with torch.no_grad():
                features, uq_probs = extract_uq_features(norms, logits.detach(), uq_cfg)
                class_labels, conf_targets = compute_pseudo_labels(
                    logits.detach(), targets, features, probs=uq_probs,
                )

            class_logits, confidence = uq_head(features)
            loss = compute_uq_loss(
                class_logits, confidence, class_labels, conf_targets,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease (first 5 avg > last 5 avg)
        assert sum(losses[:5]) / 5 > sum(losses[-5:]) / 5

    def test_uq_with_coconut(self):
        """UQ + COCONUT: verify shapes and loss computation work together."""
        from ct87.coconut import ThoughtNorm, coconut_forward, coconut_loss

        torch.manual_seed(42)
        cfg = _tiny_config()
        model = HarmonyModel(cfg)
        tn = ThoughtNorm(cfg.hidden_dim, eps=cfg.rms_norm_eps)
        uq_head = UqHead()
        uq_cfg = UqFeatureConfig.for_num_layers(cfg.num_layers)

        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        targets = torch.randint(0, cfg.vocab_size, (2, 8))
        num_thoughts = 2

        with LayerNormCollector(model, uq_cfg.norm_layers) as collector:
            logits, think_mask = coconut_forward(
                model, tn, input_ids,
                think_token_id=127, num_thoughts=num_thoughts,
            )
        norms = collector.get_norms()

        # Build augmented targets
        think_targets = torch.full((2, num_thoughts), -100, dtype=targets.dtype)
        aug_targets = torch.cat([think_targets, targets], dim=1)

        # Main loss
        lm_loss = coconut_loss(logits, aug_targets, think_mask)

        # UQ features and loss on augmented sequence
        with torch.no_grad():
            features, uq_probs = extract_uq_features(norms, logits.detach(), uq_cfg)
            # Use real targets for non-think positions, 0 for think positions
            uq_targets = aug_targets.clone()
            uq_targets[uq_targets == -100] = 0
            class_labels, conf_targets = compute_pseudo_labels(
                logits.detach(), uq_targets, features, probs=uq_probs,
            )

        class_logits, confidence = uq_head(features.detach())
        uq_loss = compute_uq_loss(
            class_logits, confidence, class_labels, conf_targets,
            think_mask=think_mask,
        )

        assert torch.isfinite(lm_loss)
        assert torch.isfinite(uq_loss)
        assert features.shape == (2, 10, 8)  # augmented seq_len
