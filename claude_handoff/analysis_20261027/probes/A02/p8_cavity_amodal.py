"""A02 probe P8 (2026-10-27, CPU only): converged tree messages in the fact codec with encoder-side amodal completion (P7), with and
without conditioning on the GBP cavity message (what the receiver side already holds and has itself sent to the sender).
For every directed forest edge s -> r of 3 held-out demo VAL houses (r1_i1, r2_i0, r3_i1; priors trained on the other 5 houses' edges):
U_S = union of the sender side's own maps, U_R = union of the receiver side's.  The receiver must learn U_S (it then has U_S | U_R exactly,
by rendering the sender side's poses through the decoded occupancy).  Payloads, static context priors (9 causal neighbours + layer +
range bucket of the sender side [+ the U_R state of the voxel: unknown / free / occupied]):
  hits      = hits(U_S) under a hits prior (no cavity)                    (P3/P2-style payload)
  greedy    = the sender's greedy amodal completion of U_S under an amodal prior (no cavity) -> the upward / first message
  cavity    = the same completion coded under an amodal prior that also sees U_R's state at the voxel -> the downward message
Free choices (voxels unknown in U_S) take the prior's argmax given everything before them.  Output: probes/A02/p8_cavity_amodal.json"""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..'))); sys.path.insert(0, HERE)
from rate_baselines_lib import load  # noqa: E402
import p7_greedy_amodal as P7  # noqa: E402
from p7_greedy_amodal import DIMS, NB9, dist_bucket, contexts, planes  # noqa: E402


class PriorX:
    """static prior with an extra per-voxel context symbol x in 0..nx-1 (U_R state); backoff to the 9-neighbour context"""
    def __init__(self, nx):
        self.nx = nx; self.n = np.zeros((512 * 32 * 6 * nx, 2)); self.n9 = np.zeros((512, 2))

    def key(self, c, z, rb, x):
        return ((c * 32 + z) * 6 + rb) * self.nx + x

    def add(self, G, rb, x):
        c = contexts(G); z = np.broadcast_to(np.arange(DIMS[2])[None, None], DIMS)
        k = self.key(c, z, rb, x).ravel(); g = G.ravel().astype(np.int64)
        self.n += np.bincount(k * 2 + g, minlength=self.n.size).reshape(-1, 2)
        self.n9 += np.bincount(c.ravel() * 2 + g, minlength=1024).reshape(-1, 2)

    def table(self, a=2.):
        p9 = (self.n9[:, 1] + a * .5) / (self.n9.sum(1) + a)
        self.p1 = (self.n[:, 1] + a * np.repeat(p9, 32 * 6 * self.nx)) / (self.n.sum(1) + a)
        return self

    def bits_fixed(self, G, rb, x):
        c = contexts(G); z = np.broadcast_to(np.arange(DIMS[2])[None, None], DIMS)
        p1 = self.p1[self.key(c, z, rb, x).ravel()]; g = G.ravel()
        return float(-np.log2(np.where(g == 1, p1, 1 - p1)).sum()) / 8

    def greedy(self, cons, rb, x):
        P = np.zeros((DIMS[0] + 2, DIMS[1] + 2, DIMS[2] + 2), np.int8); bits = 0.
        for xs, ys, zs in P7.PLANES:
            c = np.zeros(len(xs), np.int64)
            for dx, dy, dz in NB9:
                c = c * 2 + P[xs + 2 - dx, ys + 2 - dy, zs + 2 - dz]
            p1 = self.p1[self.key(c, zs, rb[xs, ys, zs].astype(np.int64), x[xs, ys, zs].astype(np.int64))]
            k = cons[xs, ys, zs]; v = np.where(k >= 0, k, (p1 >= .5).astype(np.int8))
            bits += (-np.log2(np.where(v == 1, p1, 1 - p1))).sum(); P[xs + 2, ys + 2, zs + 2] = v
        return P[2:, 2:, 2:], bits / 8


if __name__ == '__main__':
    P7.PLANES = planes()
    data = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'demo_web3d', 'data')
    houses = []
    for path in sorted(glob.glob(os.path.join(data, 'house_*.json'))):
        m, own, pred, gt = load(path); n = own.shape[0]; occ = gt['occ'].astype(np.int8)
        pos = [np.asarray(p, float) for p in m['positions_m']]
        adj = {i: set() for i in range(n)}
        for i, j in m['edges']:
            adj[i].add(j); adj[j].add(i)
        def comp(start, cut):
            out, st = {start}, [start]
            while st:
                u = st.pop()
                for v in adj[u]:
                    if v != cut and v not in out:
                        out.add(v); st.append(v)
            return out
        edges = []
        for i, j in m['edges']:
            for s, r in ((i, j), (j, i)):
                S, R = comp(s, r), comp(r, s)
                US = own[sorted(S)].max(0); UR = own[sorted(R)].max(0)
                rb, ball = dist_bucket([pos[a] for a in S], m['cell_m'])
                amodal = np.where(ball, occ, np.int8(1)).astype(np.int8)
                edges.append(dict(s=s, r=r, side=len(S), rside=len(R), US=US, x=(UR + 1).astype(np.int8), rb=rb, amodal=amodal))
        houses.append(dict(name=m['house'], edges=edges)); print('loaded', m['house'], len(edges), flush=True)
    rows = []
    zero = np.zeros(DIMS, np.int8)
    for hold in (1, 2, 5):
        ph, pa, pc = PriorX(1), PriorX(1), PriorX(3)
        for h_, H in enumerate(houses):
            if h_ == hold:
                continue
            for e in H['edges']:
                ph.add((e['US'] == 1).astype(np.int8), e['rb'], zero); pa.add(e['amodal'], e['rb'], zero); pc.add(e['amodal'], e['rb'], e['x'])
        ph.table(); pa.table(); pc.table()
        for e in houses[hold]['edges']:
            cons = e['US'].astype(np.int8)
            Gu, b_greedy = pa.greedy(cons, e['rb'], zero)
            Gc, b_cav = pc.greedy(cons, e['rb'], e['x'])
            assert ((Gc == 1) == (e['US'] == 1))[e['US'] >= 0].all() and ((Gu == 1) == (e['US'] == 1))[e['US'] >= 0].all()
            r = dict(house=houses[hold]['name'], s=e['s'], r=e['r'], side=e['side'], rside=e['rside'], hits=ph.bits_fixed((e['US'] == 1).astype(np.int8), e['rb'], zero),
                     greedy=b_greedy, cavity=b_cav, oracle_amodal=pa.bits_fixed(e['amodal'], e['rb'], zero), poses_bytes=20 * e['side'])
            rows.append(r); print(json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    S = {k: dict(median=float(np.median([r[k] for r in rows])), mean=float(np.mean([r[k] for r in rows]))) for k in rows[0] if k not in ('house', 's', 'r')}
    print(json.dumps(S, indent=1))
    json.dump(dict(summary=S, rows=rows), open(os.path.join(HERE, 'p8_cavity_amodal.json'), 'w'), indent=1)
