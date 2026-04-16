"""Tests for ZEB-128 engram consolidation."""
import torch
import pytest

from ct87.engram import EngramConsolidationDecoder


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
