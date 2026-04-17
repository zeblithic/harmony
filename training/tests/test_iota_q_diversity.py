"""Tests for ι-Q-diversity (ZEB-130): MoE load-balancing aux loss on
retrieval-row marginal distribution. Spec:
docs/superpowers/specs/2026-04-17-iota-q-diversity-design.md
"""

from __future__ import annotations

import pytest

from ct87.model import HarmonyModelConfig


class TestConfigValidation:
    def test_qdiv_defaults(self):
        c = HarmonyModelConfig.tiny_engram_xattn_capgap()
        assert c.engram_qdiv_enabled is False
        assert c.engram_qdiv_lambda == 0.01
        assert c.engram_qdiv_warmup_steps == 200

    def test_qdiv_requires_inject_layers(self):
        c = HarmonyModelConfig.tiny_engram_xattn_capgap()
        c.engram_inject_layers = ()
        c.engram_qdiv_enabled = True
        with pytest.raises(ValueError, match="engram_inject_layers"):
            c.__post_init__()

    def test_qdiv_lambda_must_be_non_negative(self):
        c = HarmonyModelConfig.tiny_engram_xattn_capgap()
        c.engram_qdiv_enabled = True
        c.engram_qdiv_lambda = -0.1
        with pytest.raises(ValueError, match="engram_qdiv_lambda"):
            c.__post_init__()

    def test_qdiv_warmup_must_be_non_negative(self):
        c = HarmonyModelConfig.tiny_engram_xattn_capgap()
        c.engram_qdiv_enabled = True
        c.engram_qdiv_warmup_steps = -1
        with pytest.raises(ValueError, match="engram_qdiv_warmup_steps"):
            c.__post_init__()
