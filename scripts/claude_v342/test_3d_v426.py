"""Tests for train_3d_v426 (Claude, 2026-10-25, before any run). CPU: python test_3d_v426.py --source <sealed>
A. --precision learned is train_3d_v425 bit-for-bit for (d3, W2, 8x8x4) and (plan_column, W2, 16x16x1): state_dict, source, join, logits.
B. --precision coverage: every tensor identical to learned (the head keeps its shape); q == 4 x block coverage exactly where the mask is
   on and mu unchanged; in the join, on every coordinate with >= 2 evidence-bearing sources the selected precision equals the maximum
   coverage precision among them (the winner is the best-coverage source); gradient reaches no head parameter of the q column;
   three AdamW steps reduce the K0 loss.
C. CLI: --precision is set before the model is built and recorded in run.json (source text).
"""
import argparse, os, sys, time
from pathlib import Path
import torch
ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); args = ap.parse_args()
sys.path.insert(0, args.source); sys.path.insert(0, os.path.join(args.source, 'exploration_grid')); HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import mapping_3d_storey_data_v420 as W
import train_3d_v425 as T5
import train_3d_v426 as T6
t0 = time.time(); R1 = W.DIMS['R1']
fx = W.make_fixture('TRAIN', 7, 2, R1); own = torch.from_numpy(fx['private']); T5.CHART_M = T6.CHART_M = fx['chart_m']

def cfg(M, dec, w, layout):
    M.DECODER = dec; M.DECODER_WIDTH = w; M.BX, M.BY, M.BZ = M.LAYOUTS[layout]; M.D = M.BX * M.BY * M.BZ

def same(ma, mb, Ma, Mb):
    sa, sb = ma.state_dict(), mb.state_dict()
    assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
    with torch.no_grad():
        a = ma.source(own); b = mb.source(own); assert all(torch.equal(x, y) for x, y in zip(a, b))
        fa = Ma.rollout_max(*a, fx['edges'], stages=(0, 8)); fb = Mb.rollout_max(*b, fx['edges'], stages=(0, 8))
        for k in (0, 8):
            assert torch.equal(fa[k], fb[k]) and all(torch.equal(x, y) for x, y in zip(ma.decoder(fa[k]), mb.decoder(fb[k])))

T6.PRECISION = 'learned'
for dec, w, layout in (('d3', 2.0, '8x8x4'), ('plan_column', 2.0, '16x16x1')):
    cfg(T5, dec, w, layout); cfg(T6, dec, w, layout)
    same(T5.SheafModel3D(R1, 342101), T6.SheafModel3D(R1, 342101), T5, T6)
print(f'A. --precision learned == train_3d_v425 bit-for-bit (d3 W2 8x8x4; plan_column W2 16x16x1) ({time.time()-t0:.0f}s)')

cfg(T6, 'plan_column', 2.0, '16x16x1')
T6.PRECISION = 'learned'; mL = T6.SheafModel3D(R1, 342101)
with torch.no_grad():
    muL, qL, mkL = mL.source(own)                                                                  # under the learned mode
T6.PRECISION = 'coverage'; mC = T6.SheafModel3D(R1, 342101)
sL, sC = mL.state_dict(), mC.state_dict(); assert all(torch.equal(sL[k], sC[k]) for k in sL)
with torch.no_grad():
    muC, qC, mkC = mC.source(own)
    assert torch.equal(muL, muC) and torch.equal(mkL, mkC)
    cov = torch.nn.functional.avg_pool3d((own >= 0).float()[:, None], (8, 8, 32)).flatten(1)
    assert torch.equal(qC[mkC > 0], (4. * cov.clamp_min(1e-4 / 4.))[mkC > 0]) and not torch.equal(qL, qC)
    ev = (mkC * qC).double(); n = ev.shape[0]
    feats = T6.rollout_max(muC, qC, mkC, fx['edges'], stages=(8,))[8]
    p_sel = torch.exp(-feats[:, T6.D:].double())                                                   # selected precision per node (from log var)
    # reachable sources per node after 8 rounds on the forest = all nodes of its tree; for every coordinate with >= 2 evidence sources among
    # them the selected precision must equal the max of their (coverage precision + prior) up to the 8-bit wire quantisation
    import collections
    adj = collections.defaultdict(set)
    for i, j, _ in fx['edges']: adj[i].add(j); adj[j].add(i)
    def tree(i):
        seen, st = {i}, [i]
        while st:
            u = st.pop()
            for v in adj[u]:
                if v not in seen: seen.add(v); st.append(v)
        return sorted(seen)
    checked = 0; lo = 0.; hi = 0.
    for i in range(n):
        src = tree(i); E = ev[src]; cnt = (E > 0).sum(0)
        for d in torch.nonzero(cnt >= 2)[:, 0].tolist():
            best = float(E[:, d].max()); got = float(p_sel[i, d]); dev = (best - got) / best; lo = min(lo, dev); hi = max(hi, dev); checked += 1
    # the wire FLOORS the precision on a 255-level log codebook over [1e-8, 128] (train_v378.kbit_wire_quantize): one level = 9.16 %, so the
    # selected precision lies in (best x e^-0.0916, best]; a deviation above that would mean a lower-coverage source won
    assert checked > 100 and lo > -1e-6 and hi < 0.092, (checked, lo, hi)
    worst = hi
mC.train(); opt = torch.optim.AdamW(mC.parameters(), lr=1e-3); losses = []
for step in range(3):
    opt.zero_grad(); mu, q, mk = mC.source(own); fz = T6.rollout_max(mu, q, mk, fx['edges'], stages=(0,))
    l, _, _ = T6.stage_loss(mC.decoder, fz[0], T6.stream_targets(own, torch.from_numpy(fx['reached'][0])), R1); l.backward()
    if step == 0:
        g = mC.head.mlp[2].weight.grad; assert g is not None and torch.all(g[1] == 0) and torch.any(g[0] != 0)   # q column gets no gradient
    opt.step(); losses.append(float(l.detach()))
assert losses[-1] < losses[0], losses
print(f'B. coverage precision: tensors identical; q == 4 x coverage; join winner = best-coverage source on {checked} contested coordinates (max rel. dev {worst:.4f}); q column has zero gradient; loss {[round(x, 4) for x in losses]} ({time.time()-t0:.0f}s)')
src = (HERE / 'train_3d_v426.py').read_text()
assert src.index("PRECISION = args.precision") < src.index("model = SheafModel3D(dims, args.seed)") and "precision=PRECISION," in src
print('C. CLI sets --precision before the model is built and run.json records it')
print('ALL TESTS PASSED', f'{time.time()-t0:.0f}s')
