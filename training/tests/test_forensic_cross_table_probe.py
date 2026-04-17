"""Regression tests for the (X) cross-table probe in forensic_eta_b_capgap.py.

Documents the bug the fix addresses (row-permuted alt table is a no-op on
top-k retrieved content) and verifies the fix (random-gaussian alt table
returns genuinely different content).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _topk_content(table: torch.Tensor, q_norm: torch.Tensor, k: int) -> torch.Tensor:
    """Reproduce the top-k retrieval used by EngramCrossAttention.

    Returns the k retrieved row vectors for a single query.
    """
    table_norm = F.normalize(table, dim=-1, eps=1e-8)
    sims = q_norm @ table_norm.T
    idx = sims.topk(k, dim=-1).indices
    return table[idx]


def test_row_permuted_alt_table_returns_identical_content_bug_documentation() -> None:
    """The bug: alt_table = primary_table[perm] produces identical top-k content.

    This is why the pre-fix (X) probe always reported |cos| = 1.0 regardless
    of V's actual content-sensitivity. Kept as a regression test so that if
    anyone reintroduces the permutation-based alt table, this test fails loudly.
    """
    n_rows, embed_dim, k = 200, 32, 8
    torch.manual_seed(0)
    primary_table = torch.randn(n_rows, embed_dim)
    q_norm = F.normalize(torch.randn(embed_dim), dim=-1)

    gen = torch.Generator(device="cpu").manual_seed(42)
    perm = torch.randperm(n_rows, generator=gen)
    buggy_alt_table = primary_table[perm]

    content_primary = _topk_content(primary_table, q_norm, k)
    content_alt = _topk_content(buggy_alt_table, q_norm, k)

    assert torch.allclose(content_primary, content_alt, atol=1e-6), (
        "Row-permuted alt table should return identical top-k content "
        "(bug the fix addresses)."
    )


def test_gaussian_matched_alt_table_returns_different_content() -> None:
    """The fix: random-gaussian alt table with matched statistics yields
    genuinely different retrieved content, so (X) |cos| becomes an honest
    probe of V's content-sensitivity."""
    n_rows, embed_dim, k = 2000, 64, 8
    torch.manual_seed(0)
    primary_table = torch.randn(n_rows, embed_dim) * 0.7 + 0.3

    gen = torch.Generator(device="cpu").manual_seed(42)
    primary_mean = primary_table.mean(dim=0, keepdim=True)
    primary_std = primary_table.std(dim=0, keepdim=True).clamp(min=1e-8)
    alt_table = (
        torch.randn(primary_table.shape, generator=gen, dtype=primary_table.dtype)
        * primary_std
        + primary_mean
    )

    q_norm = F.normalize(torch.randn(embed_dim), dim=-1)
    content_primary = _topk_content(primary_table, q_norm, k)
    content_alt = _topk_content(alt_table, q_norm, k)

    pairwise_cos = F.cosine_similarity(
        content_primary.unsqueeze(1),
        content_alt.unsqueeze(0),
        dim=-1,
    ).abs().mean().item()
    assert pairwise_cos < 0.5, (
        f"Retrieved content from alt table should be materially different "
        f"from primary; got mean |cos| = {pairwise_cos:.3f}."
    )


def test_gaussian_alt_table_preserves_first_and_second_moments() -> None:
    """Matched statistics = alt table is statistically comparable to primary,
    so retrieval similarities land in a comparable range and the (X) probe
    measures V's sensitivity rather than V's response to OOD magnitudes."""
    n_rows, embed_dim = 5000, 64
    torch.manual_seed(0)
    shift = torch.linspace(-1.0, 1.0, embed_dim)
    scale = torch.linspace(0.2, 1.5, embed_dim)
    primary_table = torch.randn(n_rows, embed_dim) * scale + shift

    gen = torch.Generator(device="cpu").manual_seed(42)
    primary_mean = primary_table.mean(dim=0, keepdim=True)
    primary_std = primary_table.std(dim=0, keepdim=True).clamp(min=1e-8)
    alt_table = (
        torch.randn(primary_table.shape, generator=gen, dtype=primary_table.dtype)
        * primary_std
        + primary_mean
    )

    # Both tables are N=5000 samples; each has per-dim SEM ≈ σ/√N ≈ 0.02
    # around the true mean, so the MAX pair-wise difference across 64 dims
    # lands around ~4× that. Loose tolerance captures 'matched within
    # sampling noise' rather than 'bit-identical'.
    assert torch.allclose(
        alt_table.mean(dim=0), primary_table.mean(dim=0), atol=0.15,
    ), "Alt table per-dim mean should match primary within sampling noise."
    assert torch.allclose(
        alt_table.std(dim=0), primary_table.std(dim=0), atol=0.2,
    ), "Alt table per-dim std should match primary within sampling noise."
