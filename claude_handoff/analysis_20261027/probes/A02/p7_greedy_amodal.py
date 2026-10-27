"""A02 probe P7 (2026-10-27, CPU only): GREEDY AMODAL COMPLETION UNDER A SHARED PRIOR = "generate the unseen part to save bits on the seen".
P6 showed (i) any occupancy G that equals the own map on its KNOWN voxels renders to exactly the own map, and (ii) the true world
occupancy of a whole house costs less under ctx than one agent's visible hit set.  Here the sender actually CHOOSES the completion:
every voxel is coded in a fixed causal order under a static prior; on known voxels the sender codes the true value, on unknown voxels
it writes the prior's most likely value (cost -log2 max(p, 1 - p), usually ~0).  The decoder decodes G without knowing which voxels were
free choices, then renders (ray casting from the transmitted poses) -> exact own map, F1 = 1.
Prior: static context model (9 causal neighbours, all with non-negative offsets so a wavefront over x + y + z is exact, + layer + range
bucket; backoff to the 9-neighbour context), leave-one-house-out, trained on
  'amodal' = true world occupancy inside the 5.3 m ball(s) of the agent(s), solid outside (an oracle completion; the prior learns worlds);
  'hits'   = the observed hit sets (the payload of P1/P2).
Reported per own map and round-8 union (3 held-out houses, 48 maps): bits of hits under the hits-prior, of the oracle completion under
the amodal prior, and of the sender's greedy completion under the amodal prior (the achievable one).  Output: probes/A02/p7_greedy_amodal.json"""
import glob, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..'))); sys.path.insert(0, HERE)
from rate_baselines_lib import load, targets  # noqa: E402

NB9 = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0), (0, 0, 2), (0, 2, 0), (2, 0, 0)]
DIMS = (128, 128, 32)


def dist_bucket(centres, cell):
    X, Y, Z = DIMS
    xs = (np.arange(X) + .5) * cell[0]; ys = (np.arange(Y) + .5) * cell[1]; zs = (np.arange(Z) + .5) * cell[2]
    d = np.full(DIMS, np.inf, np.float32)
    for c in centres:
        dd = np.sqrt((xs[:, None, None] - c[0]) ** 2 + (ys[None, :, None] - c[1]) ** 2 + (zs[None, None, :] - c[2]) ** 2).astype(np.float32)
        d = np.minimum(d, dd)
    return np.digitize(d, [1., 2., 3., 4., 5.3]).astype(np.int8), d <= 5.3


def contexts(G):
    """9-neighbour causal context of every voxel of a full binary grid (padding reads 0)"""
    P = np.zeros((DIMS[0] + 2, DIMS[1] + 2, DIMS[2] + 2), np.int64); P[2:, 2:, 2:] = G
    c = np.zeros(DIMS, np.int64)
    for dx, dy, dz in NB9:
        c = c * 2 + P[2 - dx:2 - dx + DIMS[0], 2 - dy:2 - dy + DIMS[1], 2 - dz:2 - dz + DIMS[2]]
    return c


class Prior:
    def __init__(self):
        self.n = np.zeros((512 * 32 * 6, 2)); self.n9 = np.zeros((512, 2))

    def add(self, G, rb):
        c = contexts(G); z = np.broadcast_to(np.arange(DIMS[2])[None, None], DIMS)
        key = ((c * 32 + z) * 6 + rb).ravel(); g = G.ravel().astype(np.int64)
        self.n += np.bincount(key * 2 + g, minlength=self.n.size).reshape(-1, 2)
        self.n9 += np.bincount(c.ravel() * 2 + g, minlength=1024).reshape(-1, 2)
        return self

    def table(self, a=2.):
        p9 = (self.n9[:, 1] + a * .5) / (self.n9.sum(1) + a)
        back = np.repeat(p9, 32 * 6)
        self.p1 = (self.n[:, 1] + a * back) / (self.n.sum(1) + a)
        return self

    def bits_fixed(self, G, rb):
        c = contexts(G); z = np.broadcast_to(np.arange(DIMS[2])[None, None], DIMS)
        p1 = self.p1[((c * 32 + z) * 6 + rb).ravel()]; g = G.ravel()
        return float(-np.log2(np.where(g == 1, p1, 1 - p1)).sum()) / 8


def planes():
    X, Y, Z = DIMS
    xx, yy, zz = np.meshgrid(np.arange(X), np.arange(Y), np.arange(Z), indexing='ij')
    d = (xx + yy + zz).ravel(); order = np.argsort(d, kind='stable'); ds = d[order]
    cuts = np.searchsorted(ds, np.arange(ds.max() + 2))
    flat = order
    return [(xx.ravel()[flat[cuts[i]:cuts[i + 1]]], yy.ravel()[flat[cuts[i]:cuts[i + 1]]], zz.ravel()[flat[cuts[i]:cuts[i + 1]]]) for i in range(len(cuts) - 1)]


PLANES = None


def greedy(prior, cons, rb):
    """cons: int8 grid, -1 = free choice, 0/1 = fixed; returns (G, bytes, bytes_on_free_choices)"""
    P = np.zeros((DIMS[0] + 2, DIMS[1] + 2, DIMS[2] + 2), np.int8); bits = 0.; bits_free = 0.
    for xs, ys, zs in PLANES:
        c = np.zeros(len(xs), np.int64)
        for dx, dy, dz in NB9:
            c = c * 2 + P[xs + 2 - dx, ys + 2 - dy, zs + 2 - dz]
        p1 = prior.p1[(c * 32 + zs) * 6 + rb[xs, ys, zs]]
        k = cons[xs, ys, zs]
        v = np.where(k >= 0, k, (p1 >= .5).astype(np.int8))
        b = -np.log2(np.where(v == 1, p1, 1 - p1))
        bits += b.sum(); bits_free += b[k < 0].sum()
        P[xs + 2, ys + 2, zs + 2] = v
    return P[2:, 2:, 2:], bits / 8, bits_free / 8


if __name__ == '__main__':
    PLANES = planes()
    data = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'demo_web3d', 'data')
    houses = []
    for path in sorted(glob.glob(os.path.join(data, 'house_*.json'))):
        m, own, pred, gt = load(path); n = own.shape[0]
        R8 = np.asarray(m['reached'][m['hops'].index(8)], bool); T8 = targets(own, R8)
        pos = [np.asarray(p, float) for p in m['positions_m']]; occ = gt['occ'].astype(np.int8)
        items = []
        for kind, maps in (('own', own), ('union8', T8)):
            for a in range(n):
                cs = [pos[a]] if kind == 'own' else [pos[j] for j in np.nonzero(R8[a])[0]]
                rb, ball = dist_bucket(cs, m['cell_m'])
                amodal = np.where(ball, occ, np.int8(1)).astype(np.int8)
                assert ((maps[a] == 1) <= (amodal == 1)).all() and not ((maps[a] == 0) & (amodal == 1)).any()
                items.append(dict(kind=kind, agent=a, T=maps[a], rb=rb, amodal=amodal))
        houses.append(dict(name=m['house'], items=items)); print('loaded', m['house'], flush=True)
    rows = []
    for hold in (1, 2, 5):
        for kind in ('own', 'union8'):
            ph, pa = Prior(), Prior()
            for h_, H in enumerate(houses):
                if h_ == hold:
                    continue
                for it in H['items']:
                    if it['kind'] == kind:
                        ph.add((it['T'] == 1).astype(np.int8), it['rb']); pa.add(it['amodal'], it['rb'])
            ph.table(); pa.table()
            for it in houses[hold]['items']:
                if it['kind'] != kind:
                    continue
                T = it['T']
                G, b_greedy, b_free = greedy(pa, T.astype(np.int8), it['rb'])
                assert ((G == 1) == (T == 1))[T >= 0].all()
                r = dict(house=houses[hold]['name'], kind=kind, agent=it['agent'], hits_under_hits_prior=ph.bits_fixed((T == 1).astype(np.int8), it['rb']),
                         oracle_amodal=pa.bits_fixed(it['amodal'], it['rb']), greedy_amodal=b_greedy, greedy_bytes_on_free_choices=b_free,
                         hits_under_amodal_prior=pa.bits_fixed((T == 1).astype(np.int8), it['rb']), known=float((T >= 0).mean()))
                rows.append(r); print(json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    S = {kind: {k: dict(median=float(np.median([r[k] for r in rows if r['kind'] == kind])), mean=float(np.mean([r[k] for r in rows if r['kind'] == kind])))
                for k in rows[0] if k not in ('house', 'kind', 'agent')} for kind in ('own', 'union8')}
    print(json.dumps(S, indent=1))
    json.dump(dict(summary=S, rows=rows), open(os.path.join(HERE, 'p7_greedy_amodal.json'), 'w'), indent=1)
