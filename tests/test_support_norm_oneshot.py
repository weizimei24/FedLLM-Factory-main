"""CPU mathematical checks for the one-shot support-normalized aggregator."""

import pytest
import torch

from tools.support_norm_oneshot import (
    SCALE, Layer, check_average_baseline, low_rank_truncate, reconstruct, reference_average,
    support_count_a, support_count_b,
)


def synthetic_layer(seed=11, out_dim=18, in_dim=19):
    generator = torch.Generator().manual_seed(seed)
    global_pair = (torch.randn(out_dim, 8, generator=generator),
                   torch.randn(8, in_dim, generator=generator))
    local_pairs = [(torch.randn(out_dim, 8, generator=generator),
                    torch.randn(8, in_dim, generator=generator)) for _ in range(8)]
    return Layer('synthetic', global_pair, local_pairs)


def relative_error(actual, expected):
    return (actual - expected).norm().item() / expected.norm().item()


def exact_sum(layer):
    bg, ag = layer.global_pair
    return sum(SCALE * (b.double() @ a.double() - bg.double() @ ag.double())
               for b, a in layer.local_pairs)


def test_forced_support_eight_is_average_elementwise():
    layer = synthetic_layer()
    expected = exact_sum(layer) / 8
    actual = reconstruct(layer.u, layer.s, layer.vh, torch.full_like(layer.s, 8.0))
    assert relative_error(actual, expected) < 1e-10


def test_forced_support_one_is_direct_sum():
    layer = synthetic_layer()
    expected = exact_sum(layer)
    actual = reconstruct(layer.u, layer.s, layer.vh, torch.ones_like(layer.s))
    assert relative_error(actual, expected) < 1e-10


def test_eight_identical_rank_one_updates_have_support_eight():
    bg, ag = torch.zeros(12, 8), torch.zeros(8, 12)
    u, v = torch.arange(1, 13, dtype=torch.float32), torch.arange(12, 0, -1, dtype=torch.float32)
    bi, ai = bg.clone(), ag.clone()
    bi[:, 0], ai[0] = u, v
    layer = Layer('identical', (bg, ag), [(bi, ai) for _ in range(8)])
    active = layer.s > 1e-9
    torch.testing.assert_close(layer.n_a[active], torch.full_like(layer.n_a[active], 8.0),
                               rtol=1e-10, atol=1e-10)
    support = reconstruct(layer.u, layer.s, layer.vh, layer.n_a)
    assert relative_error(support, exact_sum(layer) / 8) < 1e-10


def test_orthogonal_rank_one_updates_have_support_one():
    # Distinct singular values avoid arbitrary rotations in a degenerate SVD.
    bg, ag = torch.zeros(12, 8), torch.zeros(8, 12)
    locals_ = []
    for i in range(8):
        bi, ai = bg.clone(), ag.clone()
        bi[i, 0], ai[0, i] = float(i + 1), 1.0
        locals_.append((bi, ai))
    layer = Layer('orthogonal', (bg, ag), locals_)
    active = layer.s > 1e-9
    torch.testing.assert_close(layer.n_a[active], torch.ones_like(layer.n_a[active]),
                               rtol=1e-10, atol=1e-10)
    support = reconstruct(layer.u, layer.s, layer.vh, layer.n_a)
    assert relative_error(support, exact_sum(layer)) < 1e-10


def test_support_counts_bounded_and_threshold_monotone():
    layer = synthetic_layer()
    assert torch.all((layer.n_a >= 1) & (layer.n_a <= 8))
    assert torch.all(support_count_b(layer.c, .1) >= support_count_b(layer.c, .3))


def test_shuffling_preserves_support_multiset_and_norm_match():
    layer = synthetic_layer()
    shuffled = layer.n_for('shuffled_support_42')
    torch.testing.assert_close(shuffled.sort().values, layer.n_a.sort().values)
    support_norm = (layer.s / layer.n_a).norm().item()
    matched_norm = (layer.s * layer.norm_ratio / 8).norm().item()
    assert abs(support_norm - matched_norm) / support_norm < 1e-12


def test_rank8_low_rank_svd_matches_dense_ordinary_svd():
    generator = torch.Generator().manual_seed(78)
    left = torch.randn(23, 16, generator=generator, dtype=torch.float64)
    right = torch.randn(16, 21, generator=generator, dtype=torch.float64)
    dense = left @ right
    u, s, vh = torch.linalg.svd(dense, full_matrices=False)
    reference = (u[:, :8] * s[:8]) @ vh[:8]
    actual = low_rank_truncate(left, right, 8)
    assert relative_error(actual, reference) < 1e-10


def test_average_baseline_gate_uses_prior_joint_values():
    old_nll, old_em = reference_average()
    assert abs(old_nll - 2.300) < .005
    assert abs(old_em - 4.00) < 1e-12
    check_average_baseline(old_nll + .004, old_em, old_nll, old_em)
    with pytest.raises(RuntimeError, match='stopping before other conditions'):
        check_average_baseline(old_nll + .006, old_em, old_nll, old_em)
