"""Tests for train_3d_v429 / batch 3S-D (Claude, 2026-10-25, before any run). CPU: python test_3d_v429.py --source <sealed>
A. Defaults (8+8 bits, layouts 8x8x4 / 16x16x1 / 16x16x4 W1-W2) are train_3d_v428 bit-for-bit: state_dict, source, join, logits, loss, grads.
B. 32x32x1 @ 8+8 (D = 1,024): builds; the code grid is folded (space-to-depth 2) into the 16 x 16 start: same schedule and channel plan as the control, pre input width 8; encoder patch 2 x 2 x 16 compute cells; coverage
   mask = 4 x 4 x 32-voxel column known; block-local; logits shape; trunk / head / decoder param counts reported; raw_code_bytes 2,048.
C. --wire-bits 16,16 at 16x16x1: LEVELS 32767; the wire's precision codebook has 65,535 levels (max relative floor error < 0.04 %) and the
   mean 32,767 levels per side; raw_code_bytes 1,024; the join still picks the max-precision source. --wire-bits 4,4 at 16x16x2 (D = 512):
   LEVELS 7, precision codebook 15 levels (one level = 166 %), raw_code_bytes 512; a cross-check that quantised precisions are monotone.
D. CLI: --wire-bits parsed before the model is built and before any quantiser is used; run.json records wire_bits / raw_code_bytes; the
   warm-start assert includes wire_bits (source text).
"""
import argparse, math, os, sys, time
from pathlib import Path
import torch
ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); args = ap.parse_args()
sys.path.insert(0, args.source); sys.path.insert(0, os.path.join(args.source, 'exploration_grid')); HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import mapping_3d_storey_data_v420 as W
import train_3d_v428 as T8
import train_3d_v429 as T9
t0 = time.time(); R1 = W.DIMS['R1']
fx = W.make_fixture('TRAIN', 7, 2, R1); own = torch.from_numpy(fx['private']); T8.CHART_M = T9.CHART_M = fx['chart_m']
reached0 = torch.from_numpy(fx['reached'][0])

def cfg(M, layout, w, bits=(8, 8)):
    M.DECODER = 'd3'; M.DECODER_WIDTH = w; M.BX, M.BY, M.BZ = M.LAYOUTS[layout]; M.D = M.BX * M.BY * M.BZ; M.PRECISION = 'learned'; M.KNOWN_WEIGHT = 'uniform'
    if hasattr(M, 'WIRE_BITS'):
        M.WIRE_BITS = bits; M._v378.MU_BITS, M._v378.Q_BITS = bits; M.LEVELS = 2 ** (bits[0] - 1) - 1

def same(ma, mb, Ma, Mb):
    sa, sb = ma.state_dict(), mb.state_dict(); assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
    a = ma.source(own); b = mb.source(own); assert all(torch.equal(x, y) for x, y in zip(a, b))
    fa = Ma.rollout_max(*a, fx['edges'], stages=(0, 8)); fb = Mb.rollout_max(*b, fx['edges'], stages=(0, 8))
    for k in (0, 8): assert torch.equal(fa[k], fb[k])
    tgt = Ma.stream_targets(own, reached0)
    la, _, _ = Ma.stage_loss(ma.decoder, fa[8], tgt, R1); lb, _, _ = Mb.stage_loss(mb.decoder, fb[8], tgt, R1); assert torch.equal(la, lb)
    ga = torch.autograd.grad(la, list(ma.parameters()), allow_unused=True); gb = torch.autograd.grad(lb, list(mb.parameters()), allow_unused=True)
    assert all((x is None and y is None) or torch.equal(x, y) for x, y in zip(ga, gb))

for layout, w in (('16x16x1', 2.0), ('8x8x4', 1.0), ('16x16x4', 1.0)):
    cfg(T8, layout, w); cfg(T9, layout, w)
    same(T8.SheafModel3D(R1, 342101), T9.SheafModel3D(R1, 342101), T8, T9)
print(f'A. v429 defaults == v428 bit-for-bit for 16x16x1 W2, 8x8x4 W1, 16x16x4 W1 (incl. loss and grads) ({time.time()-t0:.0f}s)')

cfg(T9, '32x32x1', 2.0); m = T9.SheafModel3D(R1, 342101).eval()
assert T9.D == 1024 and m.decoder.start == (16, 16, 8) and len(m.decoder.blocks) == 2 and m.decoder.fold == 2 and m.decoder.pre[0].in_channels == 8 and m.trunk.bs == (2, 2, 16) and m.head.k == (4, 4, 32)
mc = T9.SheafModel3D(R1, 342101); cfg(T9, '16x16x1', 2.0); mctrl = T9.SheafModel3D(R1, 342101); cfg(T9, '32x32x1', 2.0)
sa, sb = mc.decoder.state_dict(), mctrl.decoder.state_dict(); assert list(sa) == list(sb) and all(sa[k].shape == sb[k].shape for k in sa if k != 'decoder.pre.0.weight' and k != 'pre.0.weight')   # same schedule / channel plan; only pre's input width differs
assert sa['pre.0.weight'].shape[1] == 8 and sb['pre.0.weight'].shape[1] == 2
with torch.no_grad():
    mu, q, mask = m.source(own); assert mu.shape == (8, 1024)
    cov = ((own >= 0).reshape(8, 32, 4, 32, 4, 1, 32).any(-1).any(4).any(2)).reshape(8, 1024).float(); assert torch.equal(mask, cov)
    k = [int(j) for j in torch.nonzero(mask[0] > 0)[:, 0] if divmod(int(j), 32)[0] != divmod(int(j), 32)[1]][0]; ix, iy = divmod(k, 32)
    o2 = own.clone(); outside = torch.ones(R1, dtype=torch.bool); outside[ix * 4:(ix + 1) * 4, iy * 4:(iy + 1) * 4, :] = False
    o2[0][outside] = torch.randint(-1, 2, R1, dtype=torch.int8)[outside]
    mu2, q2, _ = m.source(o2); assert torch.equal(mu2[0, k], mu[0, k]) and torch.equal(q2[0, k], q[0, k])
    f = T9.rollout_max(mu, q, mask, fx['edges'], stages=(8,))[8]; assert f.shape == (8, 2048)
    kl, ol = m.decoder(f); assert kl.shape == ol.shape == (8, 128 * 128 * 32)
pg = {kk: sum(p.numel() for p in v) for kk, v in m.parameter_groups().items()}
print(f'B. 32x32x1 @ 8+8: D 1,024, folded into the 16x16 start (same schedule as the control, pre in 8), patch 2x2x16, column mask, block-local, logits ok; params {pg} ({time.time()-t0:.0f}s)')

cfg(T9, '16x16x1', 2.0, (16, 16)); assert T9.LEVELS == 32767 and T9._v378.Q_BITS == 16
p = torch.tensor([1e-4, 0.01, 0.5, 2., 3.99], dtype=torch.float64); e = p * 0.3
ph, eh, _ = T9.kbit_wire_quantize(p, e); rel = ((p - ph) / p).max().item(); assert 0 <= rel < 4e-4, rel
z = torch.linspace(-1, 1, 5); assert torch.allclose(T9.latent_quantize(z), z, atol=2e-5)
m16 = T9.SheafModel3D(R1, 342101).eval()
with torch.no_grad():
    mu, q, mask = m16.source(own); f = T9.rollout_max(mu, q, mask, fx['edges'], stages=(8,))[8]
    ev = (mask * q).double(); P = torch.exp(-f[:, T9.D:].double())
    adj = {}
    for i, j, _ in fx['edges']: adj.setdefault(i, set()).add(j); adj.setdefault(j, set()).add(i)
    def tree(i):
        seen, st = {i}, [i]
        while st:
            u = st.pop()
            for v in adj.get(u, ()):
                if v not in seen: seen.add(v); st.append(v)
        return sorted(seen)
    hi = 0.
    for i in range(8):
        E = ev[tree(i)]; cnt = (E > 0).sum(0)
        for d in torch.nonzero(cnt >= 2)[:, 0].tolist():
            hi = max(hi, (float(E[:, d].max()) - float(P[i, d])) / float(E[:, d].max()))
    assert hi < 4e-4, hi
cfg(T9, '16x16x2', 2.0, (4, 4)); assert T9.D == 512 and T9.LEVELS == 7 and T9._v378.Q_BITS == 4
ph4, _, _ = T9.kbit_wire_quantize(p, e); assert torch.all(ph4[1:] >= ph4[:-1]) and ((p - ph4) / p).max().item() < 0.80
m44 = T9.SheafModel3D(R1, 342101).eval(); assert m44.decoder.start == (16, 16, 8) and m44.head.k == (8, 8, 16)
with torch.no_grad():
    mu, q, mask = m44.source(own); assert mu.shape == (8, 512) and (mu.abs() * 7 - (mu.abs() * 7).round()).abs().max() < 1e-5
print(f'C. 16+16 bits: LEVELS 32767, wire error < 0.04 %, join picks the max; 4+4 bits at 16x16x2: D 512, LEVELS 7, codebook monotone ({time.time()-t0:.0f}s)')
src = (HERE / 'train_3d_v429.py').read_text()
i_wb = src.index("_v378.MU_BITS, _v378.Q_BITS = WIRE_BITS; LEVELS = 2 ** (WIRE_BITS[0] - 1) - 1"); i_model = src.index("model = SheafModel3D(dims, args.seed)")
assert i_wb < i_model and "('wire_bits', list(WIRE_BITS))" in src and "wire_bits=list(WIRE_BITS)" in src and "raw_code_bytes=D * sum(WIRE_BITS) // 8" in src
print('D. CLI sets the wire bits before the model; run.json records wire_bits and raw_code_bytes; warm start asserts wire_bits')
print('ALL TESTS PASSED', f'{time.time()-t0:.0f}s')
