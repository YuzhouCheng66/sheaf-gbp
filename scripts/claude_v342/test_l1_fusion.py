"""Unit tests for l1_fusion (CPU, no models). Run: pytest scripts/claude_v342/test_l1_fusion.py -q"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
import l1_fusion as F  # noqa: E402

torch.manual_seed(0)
SHAPE = (3, 4, 5)  # (B, K, D) for the small tests


def _rand(n, lo=1e-3, hi=1e3):
    z = torch.randn(n, *SHAPE)
    lam = torch.exp(torch.empty(n, *SHAPE).uniform_(torch.log(torch.tensor(lo)).item(), torch.log(torch.tensor(hi)).item()))
    return z, lam


def test_two_node_closed_form():
    z, lam = _rand(2)
    w = 0.7
    x = F.fuse_direct(z, lam, [(0, 1)], w)
    det = lam[0] * lam[1] + w * (lam[0] + lam[1])
    x0 = (lam[0] * z[0] * (lam[1] + w) + w * lam[1] * z[1]) / det
    x1 = (lam[1] * z[1] * (lam[0] + w) + w * lam[0] * z[0]) / det
    assert torch.allclose(x[0], x0, rtol=1e-5, atol=1e-6)
    assert torch.allclose(x[1], x1, rtol=1e-5, atol=1e-6)


TOL = dict(rtol=1e-4, atol=1e-5)  # float32 solve vs. message arithmetic with lam spanning 1e-3..1e3


def test_gbp_equals_direct_two_nodes_one_round():
    for seed in range(5):
        torch.manual_seed(seed)
        z, lam = _rand(2)
        x = F.fuse_direct(z, lam, [(0, 1)], 1.3)
        m, _ = F.gbp_sync(z, lam, [(0, 1)], 1.3, rounds=1)
        assert torch.allclose(m, x, **TOL)


def test_gbp_equals_direct_on_chain_after_diameter_rounds():
    edges = [(0, 1), (1, 2), (2, 3)]
    z, lam = _rand(4)
    w = torch.tensor([0.5, 2.0, 1.0])
    x = F.fuse_direct(z, lam, edges, w)
    d = F.tree_diameter(4, edges)
    assert d == 3
    m, prec = F.gbp_sync(z, lam, edges, w, rounds=d)
    assert torch.allclose(m, x, **TOL)
    m1, _ = F.gbp_sync(z, lam, edges, w, rounds=1)
    assert not torch.allclose(m1, x, rtol=1e-3, atol=1e-3)  # one round is not enough on a diameter-3 chain
    assert (prec >= lam).all()


def test_w_zero_returns_own_z():
    z, lam = _rand(2)
    x = F.fuse_direct(z, lam, [(0, 1)], 0.0)
    assert torch.allclose(x, z, atol=1e-6)
    m, prec = F.gbp_sync(z, lam, [(0, 1)], 0.0, rounds=1)
    assert torch.allclose(m, z, atol=1e-6)
    assert torch.allclose(prec, lam, rtol=1e-6)


def test_precision_weighting_and_equal_precision_symmetry():
    z, _ = _rand(2)
    lam = torch.ones(2, *SHAPE)
    x = F.fuse_direct(z, lam, [(0, 1)], 1.0)
    # equal precision: both agents move toward each other by the same amount
    assert torch.allclose(x[0] - z[0], -(x[1] - z[1]), atol=1e-6)
    lam2 = lam.clone()
    lam2[0] = 1e3  # agent 0 is certain: it barely moves, agent 1 moves toward z_0
    x2 = F.fuse_direct(z, lam2, [(0, 1)], 1.0)
    assert (x2[0] - z[0]).abs().max() < 1e-2
    assert ((x2[1] - z[0]).abs() < (z[1] - z[0]).abs() + 1e-6).all()


def test_absent_message_falls_back_to_prior():
    z, lam = _rand(2)
    w = 1.3
    full, _ = F.gbp_sync(z, lam, [(0, 1)], w, rounds=1)
    drop = torch.zeros(2, dtype=torch.bool)
    drop[1] = True  # message 1->0 lost: agent 0 receives nothing
    m, prec = F.gbp_sync(z, lam, [(0, 1)], w, rounds=1, drop=drop[:, None, None, None])
    assert torch.equal(m[0], z[0])           # exactly its own z (the prior)
    assert torch.equal(prec[0], lam[0])
    assert torch.allclose(m[1], full[1])      # agent 1 still receives agent 0
    both = torch.ones(2, dtype=torch.bool)
    m2, _ = F.gbp_sync(z, lam, [(0, 1)], w, rounds=1, drop=both[:, None, None, None])
    assert torch.equal(m2, z)


def test_per_element_drop_mask():
    z, lam = _rand(2)
    full, _ = F.gbp_sync(z, lam, [(0, 1)], 2.0, rounds=1)
    drop = torch.zeros(2, *SHAPE, dtype=torch.bool)
    drop[1, 0] = True  # batch element 0 loses 1->0 only
    m, _ = F.gbp_sync(z, lam, [(0, 1)], 2.0, rounds=1, drop=drop)
    assert torch.equal(m[0, 0], z[0, 0])
    assert torch.allclose(m[0, 1:], full[0, 1:])
    assert torch.allclose(m[1], full[1])


def test_gradients_reach_z_lam_and_w():
    z, lam = _rand(2)
    z.requires_grad_(True)
    lam.requires_grad_(True)
    w = torch.tensor(1.0, requires_grad=True)
    x = F.fuse_direct(z, lam, [(0, 1)], w)
    x.pow(2).sum().backward()
    assert z.grad is not None and z.grad.abs().sum() > 0
    assert lam.grad is not None and lam.grad.abs().sum() > 0
    assert w.grad is not None and w.grad.abs() > 0


def test_bf16_inputs_solve_in_fp32_and_return_dtype():
    z, lam = _rand(2)
    x32 = F.fuse_direct(z, lam, [(0, 1)], 1.0)
    xb = F.fuse_direct(z.to(torch.bfloat16), lam.to(torch.bfloat16), [(0, 1)], 1.0)
    assert xb.dtype == torch.bfloat16
    assert torch.allclose(xb.float(), x32, rtol=2e-2, atol=2e-2)


def test_fp16_wire_is_close_to_exact_and_bytes_match_design():
    z, lam = _rand(2, lo=1e-3, hi=1e3)
    x = F.fuse_direct(z, lam, [(0, 1)], 1.0)
    m, _ = F.gbp_sync(z, lam, [(0, 1)], 1.0, rounds=1, wire_dtype=torch.float16)
    assert torch.allclose(m, x, rtol=5e-3, atol=5e-3)
    assert F.wire_bytes(32, 256) == 32 * 1024  # 32 KB per message: K x D_e means + K x D_e precisions, fp16


def test_float64_round_trip_is_exact_at_large_w_and_clamped_lam():
    # review M1: the C check runs in float64; in float32 the LU solve and the message arithmetic differ by ~1e-4 relative here
    torch.manual_seed(3)
    z = torch.randn(2, *SHAPE, dtype=torch.float64)
    lam = torch.where(torch.rand(2, *SHAPE) < 0.5, torch.tensor(1e-3), torch.tensor(1e3)).to(torch.float64)
    x = F.fuse_direct(z, lam, [(0, 1)], 100.0)
    m, prec = F.gbp_sync(z, lam, [(0, 1)], 100.0, rounds=1)
    assert x.dtype == torch.float64 and m.dtype == torch.float64 and prec.dtype == torch.float64
    assert float((m - x).abs().max()) <= 1e-10 * (1.0 + float(x.abs().max()))
    drop = torch.tensor([False, True])[:, None, None, None]
    m2, _ = F.gbp_sync(z, lam, [(0, 1)], 100.0, rounds=1, drop=drop)
    assert torch.equal(m2[0], z[0])


def test_edge_residual_shape_and_value():
    x = torch.zeros(2, 3, 4, 5)
    x[1] = 1.0
    r = F.edge_residual(x, [(0, 1)])
    assert r.shape == (1, 3)
    assert torch.allclose(r, torch.full((1, 3), (4 * 5) ** 0.5))


def test_bad_inputs_raise():
    z, lam = _rand(2)
    with pytest.raises(ValueError):
        F.fuse_direct(z, lam[:1], [(0, 1)], 1.0)
    with pytest.raises(ValueError):
        F.laplacian(2, [(0, 2)], 1.0)
    with pytest.raises(ValueError):
        F.laplacian(2, [(0, 1)], torch.tensor([1.0, 2.0]))
