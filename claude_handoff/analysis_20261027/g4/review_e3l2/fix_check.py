"""review E3-L2 (reviewer): the proposed M1 drop-in for g4_e3l2.source_topk ('ste' branch only) and its checks.
Forward must be bit-identical to the reviewed source_topk; backward = one-sided straight-through at every clamp bound
(the E3-L _FloorOneSided precedent, review S1 of E3-L). CPU, seconds."""
import os, sys
sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', 'scripts', 'claude_v342')))
import numpy as np
import torch
import g4_e3l2 as M


class _ClampOneSided(torch.autograd.Function):
    """forward clamp(x, lo, hi); backward: identity inside [lo, hi]; above hi only g >= 0 passes (a descent step lowers x),
    below lo only g <= 0 passes (a descent step raises x)"""

    @staticmethod
    def forward(ctx, x, lo, hi):
        ctx.save_for_backward(x); ctx.lo = float(lo); ctx.hi = float(hi)
        return x.clamp(lo, hi)

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        g = torch.where(x > ctx.hi, g.clamp(min=0.0), g)
        return torch.where(x < ctx.lo, g.clamp(max=0.0), g), None, None


def source_topk_fixed(z, lam, K, qmode, clamp=True):
    if qmode != 'ste' or not clamp:
        return M.source_topk(z, lam, K, qmode, clamp)
    sel = M.topk_select(lam, K); s = sel.to(lam.dtype)
    h = lam.unsqueeze(1) * z
    lq = (M.L.DELTA_LAM * torch.round(lam / M.L.DELTA_LAM)).clamp(M.LAMQ_MIN, M.LAMQ_MAX) * s
    hq = (M.L.DELTA_H * torch.round(h / M.L.DELTA_H)).clamp(-M.HQ_LIM, M.HQ_LIM) * s.unsqueeze(1)
    lam_b = torch.where(sel, _ClampOneSided.apply(lam, M.LAMQ_MIN - 0.5, M.LAMQ_MAX + 0.5), lam)   # selected cells only
    hg = _ClampOneSided.apply(h, -M.HQ_LIM - 0.5, M.HQ_LIM + 0.5) * s.unsqueeze(1)
    return hg + (hq - hg).detach(), lam_b + (lq - lam_b).detach(), sel


rng = np.random.default_rng(7)
n_ok = 0
for t in range(50):
    N = 3; g = (8, 8, 4)
    lam = torch.from_numpy(rng.gamma(0.7, 6.0, (N,) + g)) * torch.from_numpy(rng.random((N,) + g) < 0.7)
    z = torch.from_numpy(rng.normal(0, 3, (N, 8) + g))
    K = torch.from_numpy(rng.integers(1, 200, N))
    a1, b1, s1 = M.source_topk(z, lam, K, 'ste'); a2, b2, s2 = source_topk_fixed(z, lam, K, 'ste')
    assert torch.equal(a1, a2) and torch.equal(b1, b2) and torch.equal(s1, s2)                 # forward bit-identical
    l1 = lam.clone().requires_grad_(True); z1 = z.clone().requires_grad_(True)
    l2 = lam.clone().requires_grad_(True); z2 = z.clone().requires_grad_(True)
    G1 = torch.from_numpy(rng.normal(0, 1, a1.shape)); G2 = torch.from_numpy(rng.normal(0, 1, b1.shape))
    hq, lq, sel = M.source_topk(z1, l1, K, 'ste'); ((hq * G1).sum() + (lq * G2).sum()).backward()
    hq, lq, sel = source_topk_fixed(z2, l2, K, 'ste'); ((hq * G1).sum() + (lq * G2).sum()).backward()
    h = lam.unsqueeze(1) * z
    inside_h = (h.abs() <= M.HQ_LIM + 0.5).all(1)
    inside = (~sel | ((lam >= 0.5) & (lam <= 15.5))) & inside_h
    assert torch.allclose(l1.grad[inside], l2.grad[inside]) and torch.allclose(z1.grad.permute(0, 2, 3, 4, 1)[inside], z2.grad.permute(0, 2, 3, 4, 1)[inside])
    hi = sel & (lam > 15.5)
    assert (G2[hi] < 0).any() is not None and torch.equal(l2.grad[hi & inside_h], G2[hi & inside_h].clamp(min=0) + (G1 * z).sum(1)[hi & inside_h])
    lo = sel & (lam < 0.5)
    assert torch.equal(l2.grad[lo & inside_h], G2[lo & inside_h].clamp(max=0) + (G1 * z).sum(1)[lo & inside_h])
    n_ok += 1
print(f'fix_check: {n_ok}/50 random cases: forward bit-identical to g4_e3l2.source_topk; gradients identical inside the bounds, one-sided outside')


# ---------------------------------------------------------------- M1 as recommended (variant cons1s of grad_probe.py): h built from the clamped Lam
def source_topk_m1(z, lam, K, qmode, clamp=True):
    if not clamp:
        return M.source_topk(z, lam, K, qmode, clamp)
    sel = M.topk_select(lam, K); s = sel.to(lam.dtype)
    lam_c = torch.where(sel, _ClampOneSided.apply(lam, M.LAMQ_MIN, M.LAMQ_MAX), lam)        # selected: Lam in [1, 15], one-sided ST
    h = lam_c.unsqueeze(1) * z                                                                # h = Lam_c z (the transmitted precision)
    if qmode is None:
        return h * s.unsqueeze(1), lam_c * s, sel
    lq = (M.L.DELTA_LAM * torch.round(lam_c / M.L.DELTA_LAM)) * s
    hq = (M.L.DELTA_H * torch.round(h / M.L.DELTA_H)).clamp(-M.HQ_LIM, M.HQ_LIM) * s.unsqueeze(1)
    if qmode == 'round':
        return hq, lq, sel
    hg = _ClampOneSided.apply(h, -M.HQ_LIM - 0.5, M.HQ_LIM + 0.5) * s.unsqueeze(1)
    return hg + (hq - hg).detach(), lam_c + (lq - lam_c).detach(), sel


n2 = 0
for t in range(50):
    N = 3; g = (8, 8, 4)
    lam = torch.from_numpy(rng.gamma(0.7, 6.0, (N,) + g)) * torch.from_numpy(rng.random((N,) + g) < 0.7)
    z = torch.from_numpy(rng.normal(0, 3, (N, 8) + g)); K = torch.from_numpy(rng.integers(1, 200, N))
    for q in ('round', 'ste'):
        h1, l1, s1 = M.source_topk(z, lam, K, q); h2, l2, s2 = source_topk_m1(z, lam, K, q)
        inr = s1 & (lam >= 1) & (lam <= 15)
        assert torch.equal(s1, s2) and torch.equal(l1, l2)                                    # Lam_q and the selection identical
        assert torch.equal(h1.permute(0, 2, 3, 4, 1)[inr], h2.permute(0, 2, 3, 4, 1)[inr])  # h_q identical where 1 <= Lam <= 15
        assert (l2[s2] >= 1).all() and (h2.abs() <= 7).all()
    n2 += 1
print(f'fix_check M1: {n2}/50 random cases: Lam_q and selection identical to g4_e3l2; h_q identical on selected cells with 1 <= Lam <= 15')
