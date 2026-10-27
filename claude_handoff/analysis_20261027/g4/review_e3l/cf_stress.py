"""stress: the O13 closed-form guard near the Lam floor (softplus(-8) = 3.35e-4 on 'switched-off' supported blocks) with large z and w"""
import os, sys, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
import numpy as np, torch
sys.path.insert(0, os.path.abspath('../../../../scripts/claude_v342'))
import g4_e3l as L
G = L.G
rng = np.random.default_rng(11)
fl = float(torch.nn.functional.softplus(torch.tensor(L.A_FLOOR, dtype=torch.float64)))
res = []
for m, ed in ((8, [(i, i + 1) for i in range(7)]), (8, [(0, i) for i in range(1, 8)]), (5, [(0, 1), (1, 2), (2, 3), (3, 4)])):
    Fl = G.forest([(i, j, 1.0) for i, j in ed], m)
    for w in (1e3, 1e4, 1e5):
        for zs in (10., 100., 1000., 5000.):
            lam = np.full((m, 8, 8, 4), fl); lam[:, :2] = rng.gamma(2, 2, (m, 2, 8, 4))       # most cells at the floor, a quarter informative
            lam32 = torch.tensor(lam, dtype=torch.float32).double()                       # as the encoder gives it (float32)
            z = torch.tensor(rng.normal(0, zs, (m, 8, 8, 8, 4)), dtype=torch.float32)
            c = L.gbp_checks(z, lam32, Fl, w)
            res.append(dict(m=m, tree='chain' if ed[1][0] == 1 else 'star', w=w, z_std=zs, cf_max=c['cf_max'], viol=c['violations'], cf_cells=c['cf_cells'], excl=c['cf_excluded_tiny'],
                            passes=c['cf_max'] <= L.CF_TOL and c['violations'] == 0))
for r in res:
    print(r)
json.dump(res, open('cf_stress.json', 'w'), indent=1)
