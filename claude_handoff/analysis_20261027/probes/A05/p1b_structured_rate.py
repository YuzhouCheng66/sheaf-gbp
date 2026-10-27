"""A05 probe 1b (CPU, demo data): lossless cost of a STRUCTURED occluder set G with H <= G <= not F (ray-cast exact, see p1).
G_s = full floor slab (public) + ceiling slab (1 B: wall height) + wall columns extruded over the full height (2-D plan of the
columns that show a structure hit below the lintel start) + lintel columns from the lintel start up (2-D map, 1 B door height)
+ the observed furniture hits (3-D). Every piece is world-occupied, so G_s never covers a known-free voxel, and it covers every hit:
RayCast(G_s, poses) = the map, exactly. Cost = ctx2d(wall plan) + ctx2d(lintel plan | wall plan) + ctx3d(furniture hits) + 2 B (+ poses).
Also: the same with the furniture hits coded given the wall plan and the slab (co-located context), and the share of the cost that
is furniture. Lintel start index: the lowest z of structure voxels in non-full-height structure columns of gt_occ (public: door height)."""
import glob, json, math, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, HERE)
from rate_baselines_lib import load, targets, ctx_bytes  # noqa: E402
from p1_hitset_rate import ctx_bin, _ctx  # noqa: E402  (module-level code of p1 is guarded below by __name__? no: we import functions only)

DATA = sys.argv[1] if len(sys.argv) > 1 else 'demo_web3d/data'
NB2 = [(0, 1), (1, 0), (1, 1), (1, -1), (0, 2), (2, 0)]


def ctx2d(mask, side=None, side_base=1):
    s = torch.from_numpy(mask.astype(np.int64)); X, Y = s.shape; c = torch.zeros_like(s)
    for dx, dy in NB2:
        sh = torch.zeros_like(s)
        if dy >= 0:
            sh[dx:, dy:] = s[:X - dx, :Y - dy]
        else:
            sh[dx:, :Y + dy] = s[:X - dx, -dy:]
        c = c * 2 + sh
    if side is not None:
        c = c * side_base + torch.from_numpy(side.astype(np.int64))
    return _ctx(s, c, 2)


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(DATA, 'house_*.json'))):
        m, own, pred, gt = load(path); n = own.shape[0]
        slab = gt['slab']; obj = gt['obj']; occ = gt['occ']; struct = occ & ~obj & ~slab
        full_col = struct[:, :, 1:20].all(2)                               # wall columns (full height between the slabs)
        part = struct.any(2) & ~full_col
        lk = int(np.nonzero(struct[part].any(0))[0].min()) if part.any() else 32
        T8 = targets(own, m['reached'][m['hops'].index(8)])
        adj = {i: set() for i in range(n)}
        for i, j in m['edges']:
            adj[i].add(j); adj[j].add(i)
        def side_of(s, r):
            side, stack = {s}, [s]
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if v != r and v not in side:
                        side.add(v); stack.append(v)
            return sorted(side)
        maps = [('own', own[a]) for a in range(n)] + [('union8', T8[a]) for a in range(n)]
        for i, j in m['edges']:
            for s, r in ((i, j), (j, i)):
                maps.append(('edge', own[side_of(s, r)].max(0)))
        for kind, M in maps:
            H = M == 1
            Hs = H & struct
            wall = Hs[:, :, :lk].any(2)                                     # columns with a structure hit below the lintel start
            lint = Hs.any(2) & ~wall
            Hf = H & obj
            # validity of G_s on the WORLD: every piece world-occupied (so never known-free) and covering every hit
            G = slab.copy(); G |= wall[:, :, None]; L3 = np.zeros_like(G); L3[:, :, lk:] = lint[:, :, None]; G |= L3; G |= Hf
            assert not (G & ~occ).any(), 'G_s covers a world-free voxel'
            assert not (H & ~G).any(), 'G_s misses a hit'
            r = dict(kind=kind, house=m['house'], lintel_k=lk, ctx_full=ctx_bytes(M), ctx_H=ctx_bin(H),
                     wall_cols=int(wall.sum()), lintel_cols=int(lint.sum()), furn_hits=int(Hf.sum()),
                     c_wall=ctx2d(wall), c_lint=ctx2d(lint, wall, 2), c_furn=ctx_bin(Hf),
                     c_furn_ctx=ctx_bin(Hf, (wall[:, :, None] | slab).astype(np.int64) + 0 * Hf, 2))
            r['structured'] = r['c_wall'] + r['c_lint'] + min(r['c_furn'], r['c_furn_ctx']) + 2
            rows.append(r)
        print(m['house'], 'lintel_k', lk, flush=True)

    S = {}
    for kind in ('own', 'union8', 'edge'):
        r = [x for x in rows if x['kind'] == kind]
        S[kind] = {k: float(np.median([x[k] for x in r])) for k in r[0] if isinstance(r[0][k], (int, float)) and k != 'lintel_k'}
        S[kind]['furn_share_of_structured'] = float(np.median([min(x['c_furn'], x['c_furn_ctx']) / x['structured'] for x in r]))
        S[kind]['n'] = len(r)
    json.dump(dict(summary=S, rows=rows), open(os.path.join(HERE, 'p1b_structured_rate.json'), 'w'), indent=1)
    print(json.dumps(S, indent=1))


if __name__ == '__main__':
    main()
