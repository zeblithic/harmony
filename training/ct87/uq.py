"""UQ (Uncertainty Quantification) head training for ct87.

Implements the training-side mirror of the Rust UQ head
(crates/harmony-inference/src/uq_head.rs) and feature extraction
(crates/harmony-inference/src/uq_features.rs).

The UQ head is a tiny 429-parameter MLP that classifies each generation
step into one of four uncertainty categories (Confident, HighVolume,
SpectralCollapse, Uncertain) plus a scalar confidence score. It trains
as an auxiliary task alongside the main LM using self-supervised
pseudo-labels derived from model behavior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ct87.model import HarmonyModel


@dataclass
class UqFeatureConfig:
    """Configuration for UQ feature extraction.

    Mirrors Rust UqFeatureConfig in uq_features.rs:27-34.
    """

    norm_layers: list[int]  # 4 layer indices for L2 norm sampling
    top_k: int = 10  # top-k for probability mass feature (f7)

    @staticmethod
    def for_num_layers(n: int) -> UqFeatureConfig:
        """Derive config from layer count. Matches Rust for_num_layers."""
        if n == 24:
            return UqFeatureConfig(norm_layers=[0, 8, 16, 23])
        if n == 8:
            return UqFeatureConfig(norm_layers=[0, 2, 5, 7])
        last = n - 1
        i1 = min(last // 3, last)
        i2 = min(2 * last // 3, last)
        return UqFeatureConfig(norm_layers=[0, i1, i2, last])


class LayerNormCollector:
    """Context manager that collects per-layer L2 norms via forward hooks.

    Registers hooks on the specified transformer layers. Each hook computes
    the L2 norm of the layer output across hidden_dim at every sequence
    position, storing ``[batch, seq_len]`` tensors keyed by layer index.

    Handles COCONUT 2-pass: since hooks overwrite (not append), the last
    forward pass's norms are always what ``get_norms()`` returns. Also safe
    with gradient checkpointing — recomputed norms overwrite with identical
    values after we've already consumed them.
    """

    def __init__(self, model: HarmonyModel, layer_indices: list[int]):
        self._model = model
        self._layer_indices = layer_indices
        self._norms: dict[int, torch.Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

    def __enter__(self) -> LayerNormCollector:
        for idx in self._layer_indices:

            def _make_hook(layer_idx: int):
                def hook(module, input, output):  # noqa: ARG001
                    with torch.no_grad():
                        self._norms[layer_idx] = output.detach().norm(dim=-1)
                return hook

            self._hooks.append(
                self._model.layers[idx].register_forward_hook(_make_hook(idx))
            )
        return self

    def __exit__(self, *args) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def get_norms(self) -> dict[int, torch.Tensor]:
        """Return collected norms: ``{layer_idx: [batch, seq_len]}``."""
        return dict(self._norms)


def extract_uq_features(
    layer_norms: dict[int, torch.Tensor],
    logits: torch.Tensor,
    config: UqFeatureConfig,
) -> torch.Tensor:
    """Extract 8-dim UQ feature vectors at all sequence positions.

    Mirrors Rust ``extract_uq_features()`` from ``uq_features.rs:98-136``.

    Args:
        layer_norms: ``{layer_idx: [batch, seq_len]}`` L2 norms from
            :class:`LayerNormCollector`.
        logits: ``[batch, seq_len, vocab_size]`` raw logits.
        config: Feature extraction config.

    Returns:
        ``[batch, seq_len, 8]`` feature tensor.
    """
    # f1-f4: L2 norms at sampled layers
    f1 = layer_norms[config.norm_layers[0]]
    f2 = layer_norms[config.norm_layers[1]]
    f3 = layer_norms[config.norm_layers[2]]
    f4 = layer_norms[config.norm_layers[3]]

    # f5: linear regression slope
    f5 = (3.0 * (f4 - f1) + (f3 - f2)) / 10.0

    # Softmax for entropy and top-k
    probs = F.softmax(logits, dim=-1)

    # f6: Shannon entropy  -sum(p * ln(p))
    f6 = -(probs * probs.clamp(min=1e-10).log()).sum(dim=-1)

    # f7: top-k probability mass
    k = min(config.top_k, logits.shape[-1])
    f7 = torch.topk(probs, k=k, dim=-1).values.sum(dim=-1)

    # f8: stub (attention lookback, deferred)
    f8 = torch.zeros_like(f1)

    return torch.stack([f1, f2, f3, f4, f5, f6, f7, f8], dim=-1)


def compute_pseudo_labels(
    logits: torch.Tensor,
    targets: torch.Tensor,
    features: torch.Tensor,
    collapse_threshold: float = 0.1,
    confident_threshold: float = 0.5,
    top_k: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive pseudo-labels from model behavior.

    Priority order: SpectralCollapse > Confident > HighVolume > Uncertain.

    Args:
        logits: ``[batch, seq_len, vocab_size]``
        targets: ``[batch, seq_len]``
        features: ``[batch, seq_len, 8]`` from :func:`extract_uq_features`
        collapse_threshold: max norm threshold for SpectralCollapse
        confident_threshold: P(target) threshold for Confident
        top_k: top-k value used in feature extraction (for entropy threshold)

    Returns:
        class_labels: ``[batch, seq_len]`` long, values in {0,1,2,3}
        confidence_targets: ``[batch, seq_len]`` float in {0.0, 1.0}
    """
    batch, seq_len = targets.shape

    # Extract feature components
    f1 = features[..., 0]
    f2 = features[..., 1]
    f3 = features[..., 2]
    f4 = features[..., 3]
    f6 = features[..., 5]  # entropy
    f7 = features[..., 6]  # top-k mass

    # Start with Uncertain (3) everywhere
    labels = torch.full((batch, seq_len), 3, dtype=torch.long, device=targets.device)

    # HighVolume (1): high entropy AND dispersed probability
    entropy_threshold = math.log(float(top_k))
    high_volume = (f6 > entropy_threshold) & (f7 < 0.5)
    labels[high_volume] = 1

    # Confident (0): model assigns high prob to correct token (overrides HighVolume)
    probs = F.softmax(logits, dim=-1)
    target_probs = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    confident = target_probs > confident_threshold
    labels[confident] = 0

    # SpectralCollapse (2): all sampled norms very low (highest priority)
    max_norms = torch.stack([f1, f2, f3, f4], dim=-1).max(dim=-1).values
    collapsed = max_norms < collapse_threshold
    labels[collapsed] = 2

    # Confidence target: was the top prediction correct?
    predicted = logits.argmax(dim=-1)
    confidence_targets = (predicted == targets).float()

    return labels, confidence_targets


class UqHead(nn.Module):
    """Uncertainty Quantification Head — mirrors Rust UqHead exactly.

    Two-path classifier operating on 8 pre-extracted features:
    - Classifier: Linear(8->32) -> ReLU -> Linear(32->4) -> raw logits
    - Confidence: Linear(8->1) -> raw logit

    Forward returns raw logits for both paths. The Rust side applies
    softmax/sigmoid in its forward; here we defer activation to the loss
    functions (``F.cross_entropy`` and ``F.binary_cross_entropy_with_logits``)
    for numerical stability under autocast.

    Total parameters: 429.
    """

    NUM_FEATURES = 8
    HIDDEN_DIM = 32
    NUM_CLASSES = 4

    def __init__(self):
        super().__init__()
        self.classifier_fc1 = nn.Linear(self.NUM_FEATURES, self.HIDDEN_DIM, bias=True)
        self.classifier_fc2 = nn.Linear(self.HIDDEN_DIM, self.NUM_CLASSES, bias=True)
        self.confidence_linear = nn.Linear(self.NUM_FEATURES, 1, bias=True)
        self._init_weights()

    def _init_weights(self):
        """Match Rust UqHead::new() initialization.

        scale = 1/sqrt(fan_in), biases = zeros.
        """
        for linear in [self.classifier_fc1, self.classifier_fc2, self.confidence_linear]:
            fan_in = linear.weight.shape[1]
            nn.init.normal_(linear.weight, mean=0.0, std=1.0 / math.sqrt(fan_in))
            nn.init.zeros_(linear.bias)

    def forward(
        self, features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            features: ``[..., 8]`` — any leading dimensions.

        Returns:
            class_logits: ``[..., 4]`` raw logits (no softmax).
            confidence_logits: ``[..., 1]`` raw logits (no sigmoid).
        """
        h = F.relu(self.classifier_fc1(features))
        class_logits = self.classifier_fc2(h)
        confidence_logits = self.confidence_linear(features)
        return class_logits, confidence_logits


def compute_uq_loss(
    class_logits: torch.Tensor,
    confidence_logits: torch.Tensor,
    class_labels: torch.Tensor,
    confidence_targets: torch.Tensor,
    think_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Combined UQ loss: classification CE + confidence BCE.

    Uses ``binary_cross_entropy_with_logits`` on raw confidence logits
    for numerical stability under bfloat16 autocast.

    Args:
        class_logits: ``[batch, seq_len, 4]``
        confidence_logits: ``[batch, seq_len, 1]`` raw logits (pre-sigmoid)
        class_labels: ``[batch, seq_len]`` long
        confidence_targets: ``[batch, seq_len]`` float
        think_mask: optional ``[batch, seq_len]`` bool — positions to exclude

    Returns:
        Scalar loss.
    """
    logits_flat = class_logits.reshape(-1, 4)
    labels_flat = class_labels.reshape(-1)
    conf_flat = confidence_logits.squeeze(-1).reshape(-1)
    target_flat = confidence_targets.reshape(-1)

    if think_mask is not None:
        valid = ~think_mask.reshape(-1)
        logits_flat = logits_flat[valid]
        labels_flat = labels_flat[valid]
        conf_flat = conf_flat[valid]
        target_flat = target_flat[valid]

    if labels_flat.numel() == 0:
        return class_logits.new_tensor(0.0)

    ce = F.cross_entropy(logits_flat, labels_flat)
    bce = F.binary_cross_entropy_with_logits(conf_flat, target_flat)

    return ce + bce
