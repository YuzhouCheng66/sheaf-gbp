"""Unit test for eval_decision_v431.decide (Claude, 2026-10-26, batch 3S-DR). CPU, seconds: python test_eval_decision_v431.py"""
import torch
import eval_decision_v431 as E
assert (E.OCC_T, E.KNOWN_OCC_T) == (1.5, -0.5)
k = torch.tensor([-1., -0.6, -0.5, -0.4, 0., 0., 0., 2., -3.]); o = torch.tensor([3., 3., 1.5, 1.49, 1.49, 1.5, -2., 0.1, -3.])
v = E.decide(k, o, 'v420'); c = E.decide(k, o, 'calibrated')
assert v.dtype == c.dtype == torch.int8
assert v.tolist() == [-1, -1, -1, -1, 1, 1, 0, 1, -1], v.tolist()
assert c.tolist() == [-1, -1, 1, -1, 0, 1, 0, 0, -1], c.tolist()      # o>=1.5 & k>=-0.5 -> 1; else k>=0 -> 0; else -1
g = torch.Generator().manual_seed(0); K = torch.randn(4, 1000, generator=g) * 3; O = torch.randn(4, 1000, generator=g) * 3
ref = torch.where(K >= 0, (O >= 0).to(torch.int8), torch.full_like(O, -1, dtype=torch.int8))   # train_3d_v431.predict, verbatim
assert torch.equal(E.decide(K, O, 'v420'), ref)
C = E.decide(K, O, 'calibrated'); assert torch.equal(C == 1, (O >= 1.5) & (K >= -0.5)) and torch.equal(C == 0, (K >= 0) & ~((O >= 1.5) & (K >= -0.5)))
F = E.decide(K, O, 'occ_only')                                    # review M1: the fallback keeps v420's unknown set, hence its FPR
assert torch.equal(F == -1, K < 0) and torch.equal(F == -1, E.decide(K, O, 'v420') == -1) and torch.equal(F == 1, (K >= 0) & (O >= 1.5))
tg = torch.randint(-1, 2, (3, 500), generator=g).to(torch.int8); sf = (tg == 1) & (torch.rand(3, 500, generator=g) < .5); pr = torch.randint(-1, 2, (3, 500), generator=g).to(torch.int8)
def ref_sf(p, t, s):                                               # train_3d_v431.metrics3d surface_f1 (+ _f1, agg), same logic
    v = []
    for i in range(p.shape[0]):
        kn = t[i] >= 0
        if kn.sum() == 0:
            continue
        a = ((p[i] == 1) & kn) & ~((t[i] == 1) & ~s[i]); b = s[i]
        tp = (a & b).sum().item(); fp = (a & ~b).sum().item(); fn = (~a & b).sum().item()
        if tp + fp + fn:
            v.append(2 * tp / max(2 * tp + fp + fn, 1))
    return sum(v) / len(v)
assert abs(E.surface_f1_mean(pr, tg, sf) - ref_sf(pr, tg, sf)) < 1e-12
print('decide, occ_only and fast surface F1 ok')
