"""Tests for train_3d_v424 (Claude, 2026-10-25). CPU: python test_3d_v424.py --source <sealed>
A. (W 1, 8x8x4) == train_3d_v420; (W 2, 8x8x4) == train_3d_v422; (W 1, 16x16x1) == train_3d_v423 -- state_dict, source, join, decoder
   logits bit-for-bit on a real fixture.
B. Diagnostic 16x16x4 (D = 1,024): builds; coverage mask = block known; decoder logits shape; rollout runs with D = 1,024 (prior, wire).
C. CLI: layout and width are set before the model is built, D follows the layout, D != 256 is refused without --allow-non256 (source text).
"""
import argparse, os, sys, time
from pathlib import Path
import torch
ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); args = ap.parse_args()
sys.path.insert(0, args.source); sys.path.insert(0, os.path.join(args.source, 'exploration_grid')); HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import mapping_3d_storey_data_v420 as W
import train_3d_v420 as T0
import train_3d_v422 as T2
import train_3d_v423 as T3
import train_3d_v424 as T4
t0 = time.time(); R1 = W.DIMS['R1']
fx = W.make_fixture('TRAIN', 7, 2, R1); own = torch.from_numpy(fx['private'])
for M in (T0, T2, T3, T4):
    M.CHART_M = fx['chart_m']

def same(ma, mb, Ma, Mb):
    sa, sb = ma.state_dict(), mb.state_dict()
    assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
    with torch.no_grad():
        a = ma.source(own); b = mb.source(own); assert all(torch.equal(x, y) for x, y in zip(a, b))
        fa = Ma.rollout_max(*a, fx['edges'], stages=(0, 8)); fb = Mb.rollout_max(*b, fx['edges'], stages=(0, 8))
        for k in (0, 8):
            assert torch.equal(fa[k], fb[k]) and all(torch.equal(x, y) for x, y in zip(ma.decoder(fa[k]), mb.decoder(fb[k])))

T4.DECODER_WIDTH = 1.0; T4.BX, T4.BY, T4.BZ = 8, 8, 4; T4.D = 256
same(T0.SheafModel3D(R1, 342101), T4.SheafModel3D(R1, 342101), T0, T4)
T2.DECODER_WIDTH = 2.0; T4.DECODER_WIDTH = 2.0
same(T2.SheafModel3D(R1, 342202), T4.SheafModel3D(R1, 342202), T2, T4)
T4.DECODER_WIDTH = 1.0; T3.BX, T3.BY, T3.BZ = T4.BX, T4.BY, T4.BZ = 16, 16, 1
same(T3.SheafModel3D(R1, 342101), T4.SheafModel3D(R1, 342101), T3, T4)
print(f'A. v424 == v420 (W1 8x8x4), == v422 (W2 8x8x4), == v423 (W1 16x16x1) bit-for-bit ({time.time()-t0:.0f}s)')

T4.BX, T4.BY, T4.BZ = 16, 16, 4; T4.D = 1024
m = T4.SheafModel3D(R1, 342101).eval()
with torch.no_grad():
    mu, q, mask = m.source(own)
    assert mu.shape == (8, 1024)
    cov = ((own >= 0).reshape(8, 16, 8, 16, 8, 4, 8).any(-1).any(4).any(2)).reshape(8, 1024).float()
    assert torch.equal(mask, cov)
    f = T4.rollout_max(mu, q, mask, fx['edges'], stages=(8,))[8]; assert f.shape == (8, 2048)
    kl, ol = m.decoder(f); assert kl.shape == ol.shape == (8, 128 * 128 * 32)
print(f'B. diagnostic 16x16x4: D = 1,024, mask = block known, rollout + decoder shapes ({time.time()-t0:.0f}s)')

src = (HERE / 'train_3d_v424.py').read_text()
i_w = src.index("DECODER_WIDTH = args.decoder_width"); i_l = src.index("BX, BY, BZ = LAYOUTS[args.layout]; D = BX * BY * BZ"); i_g = src.index("assert D == 256 or (args.allow_non256")
i_m = src.index("model = SheafModel3D(dims, args.seed)")
assert "global CHART_M, DECODER_WIDTH, BX, BY, BZ, D" in src and i_w < i_l < i_g < i_m and "layout=args.layout" in src
print('C. CLI order: width, layout, D, the 256 guard, then the model; run.json records both knobs')
print('ALL TESTS PASSED', f'{time.time()-t0:.0f}s')
