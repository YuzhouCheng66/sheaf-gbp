"""A05 probe 6 (CPU, demo data, exact poses): a REALISTIC deterministic box encoder (sees only its own map), for design 2.
surface : greedy maximal boxes of the agent's own furniture hits H_f (every box voxel is a hit, so every box is world-occupied: sound;
          K0 and R8 exact by the ray-cast theorem whatever the union).
solid   : greedy maximal boxes inside A = H_f | (own-unknown voxels within 2 voxels of H_f, not structure / slab), seeded at uncovered
          hits, each shrunk to the bounding box of the hits it covers (never covers an own known-free voxel, so K0 is exact; it may cover
          world-free unknown space, which can break R8 when another agent saw that space free) -- soundness measured against gt_occ.
For both: tokens per own map and per round-8 union (union = all reached agents' tokens, reduced by containment), K0 / R8 decode via
exact structure + tokens + ray cast, and a crude leave-one-house-out token code (as p4)."""
import glob, json, math, os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, HERE)
from rate_baselines_lib import load, targets, metrics, dilate6, DIMS  # noqa: E402
from p2b_raycast_decode import ray_table, cast, struct_parts, fpr  # noqa: E402
from p4_box_tokens import greedy_boxes, raster  # noqa: E402


def solid_boxes(Hf, allowed):
    """greedy maximal boxes inside `allowed`, seeded at uncovered hits, shrunk to the AABB of the hits they contain"""
    left = Hf.copy(); out = []; X, Y, Z = Hf.shape
    while left.any():
        x, y, z = np.argwhere(left)[0]
        z1 = z
        while z1 + 1 < Z and allowed[x, y, z1 + 1]: z1 += 1
        z0 = z
        while z0 - 1 >= 0 and allowed[x, y, z0 - 1]: z0 -= 1
        y1 = y
        while y1 + 1 < Y and allowed[x, y1 + 1, z0:z1 + 1].all(): y1 += 1
        x1 = x
        while x1 + 1 < X and allowed[x1 + 1, y:y1 + 1, z0:z1 + 1].all(): x1 += 1
        sub = Hf[x:x1 + 1, y:y1 + 1, z0:z1 + 1]
        ii = np.argwhere(sub)
        lo = ii.min(0); hi = ii.max(0)
        b = (x + lo[0], x + hi[0] + 1, y + lo[1], y + hi[1] + 1, z0 + lo[2], z0 + hi[2] + 1)
        out.append(b)
        left[b[0]:b[1], b[2]:b[3], b[4]:b[5]] = False
    return np.array(out, np.int64).reshape(-1, 6)


def reduce_contained(b):
    if len(b) == 0:
        return b
    b = np.unique(b, axis=0); keep = np.ones(len(b), bool)
    for i in range(len(b)):
        c = (b[:, 0] <= b[i, 0]) & (b[:, 1] >= b[i, 1]) & (b[:, 2] <= b[i, 2]) & (b[:, 3] >= b[i, 3]) & (b[:, 4] <= b[i, 4]) & (b[:, 5] >= b[i, 5])
        c[i] = False
        if (c & keep).any():
            keep[i] = False
    return b[keep]


def token_bits(tokens_by_map):
    """leave-one-house-out empirical code of raster-sorted tokens: per coordinate, delta to the previous token's coordinate (x0,y0,z0)
    or the extent (x1-x0, ...), adaptive frequencies from the other houses (as p4)"""
    def diffs(b):
        if len(b) == 0:
            return np.zeros((0, 6), np.int64)
        o = b[np.lexsort((b[:, 4], b[:, 2], b[:, 0]))]; d = o.copy()
        d[1:, [0, 2, 4]] = o[1:, [0, 2, 4]] - o[:-1, [0, 2, 4]]
        d[:, 1] = o[:, 1] - o[:, 0]; d[:, 3] = o[:, 3] - o[:, 2]; d[:, 5] = o[:, 5] - o[:, 4]
        return d
    houses = sorted({h for h, _ in tokens_by_map}); res = []
    for h in houses:
        tr = [diffs(b) for k, b in tokens_by_map if k != h and len(b)]
        train = np.concatenate(tr) if tr else np.zeros((0, 6), np.int64)
        tabs = []
        for c in range(6):
            vals, cnt = np.unique(train[:, c], return_counts=True); tabs.append((dict(zip(vals.tolist(), cnt.tolist())), cnt.sum()))
        for k, b in tokens_by_map:
            if k != h:
                continue
            d = diffs(b); bits = 0.
            for c in range(6):
                p, N = tabs[c]
                bits += sum(-math.log2((p.get(int(v), 0) + 0.5) / (N + 0.5 * 256)) for v in d[:, c])
            res.append(dict(house=k, tokens=len(b), bytes=bits / 8))
    return res


if __name__ == '__main__':
    poses = json.load(open(os.path.join(HERE, 'poses.json')))
    rows = []; toks = {('surface', 0): [], ('solid', 0): [], ('surface', 8): [], ('solid', 8): []}
    for path in sorted(glob.glob(os.path.join(ROOT, 'demo_web3d', 'data', 'house_*.json'))):
        t0 = time.time()
        m, own, pred, gt = load(path); n = own.shape[0]; key = m['house']
        traj = np.asarray(poses[key]['trajectory_m']); cell = tuple(poses[key]['cell']); lk = poses[key]['lintel_k']
        tables = {a: [ray_table(traj[k, a], cell) for k in range(traj.shape[0])] for a in range(n)}
        slab = gt['slab']; obj = gt['obj']; occ = gt['occ']; struct = occ & ~obj & ~slab
        slab_t = torch.from_numpy(slab.reshape(-1)); obj_t = torch.from_numpy(obj.reshape(-1)); struct_t = torch.from_numpy(struct.reshape(-1))
        R8 = np.asarray(m['reached'][m['hops'].index(8)], bool); T8 = targets(own, R8)
        own_tok = {}
        for a in range(n):
            t = own[a]; Hf = (t == 1) & obj
            Gs, wall, lint = struct_parts(t == 1, gt, lk)
            colwall = wall[:, :, None] | np.zeros(DIMS, bool)
            near = Hf.copy()
            for _ in range(2):
                near = near | dilate6(torch.from_numpy(near.reshape(1, -1)), DIMS)[0].numpy().reshape(DIMS)
            allowed = Hf | (near & (t < 0) & ~colwall & ~slab)
            for kind, B in (('surface', greedy_boxes(Hf).reshape(-1, 6)), ('solid', solid_boxes(Hf, allowed).reshape(-1, 6))):
                own_tok[(kind, a)] = B
                G = Gs | raster(B, DIMS)
                Pr = cast(G, tables[a])
                mt = metrics(torch.from_numpy(Pr.reshape(1, -1)), torch.from_numpy(t.reshape(1, -1)), slab_t, obj_t, struct_t)[0]
                vox = raster(B, DIMS)
                rows.append(dict(house=key, hop=0, agent=a, kind=kind, tokens=len(B), exact=bool((Pr == t).all()), unsound_voxels=int((vox & ~occ).sum()),
                                 fpr=fpr(Pr, t), **mt))
                toks[(kind, 0)].append((key, B))
        seen = set()
        for a in range(n):
            agents = tuple(np.nonzero(R8[a])[0])
            if agents in seen:
                continue
            seen.add(agents)
            t = T8[a]
            Gs, wall, lint = struct_parts(t == 1, gt, lk)
            tabs = [tb for b_ in agents for tb in tables[b_]]
            for kind in ('surface', 'solid'):
                B = reduce_contained(np.concatenate([own_tok[(kind, b_)] for b_ in agents]))
                G = Gs | raster(B, DIMS)
                Pr = cast(G, tabs)
                mt = metrics(torch.from_numpy(Pr.reshape(1, -1)), torch.from_numpy(t.reshape(1, -1)), slab_t, obj_t, struct_t)[0]
                rows.append(dict(house=key, hop=8, agent=a, kind=kind, tokens=len(B), exact=bool((Pr == t).all()), unsound_voxels=int((raster(B, DIMS) & ~occ).sum()),
                                 fpr=fpr(Pr, t), **mt))
                toks[(kind, 8)].append((key, B))
        print(key, f'{time.time() - t0:.0f}s', flush=True)
    S = {}
    for kind in ('surface', 'solid'):
        for hop in (0, 8):
            r = [x for x in rows if x['kind'] == kind and x['hop'] == hop]
            tb = token_bits(toks[(kind, hop)])
            S[f'{kind} hop{hop}'] = dict(tokens_median=float(np.median([x['tokens'] for x in r])), exact_share=float(np.mean([x['exact'] for x in r])),
                                         unsound_maps_share=float(np.mean([x['unsound_voxels'] > 0 for x in r])), unsound_voxels_median=float(np.median([x['unsound_voxels'] for x in r])),
                                         surface_f1=float(np.mean([x['surface_f1'] for x in r])), furn_f1=float(np.mean([x['furn_f1'] for x in r if x['furn_f1'] is not None])),
                                         fpr=float(np.mean([x['fpr'] for x in r])), token_bytes_median=float(np.median([x['bytes'] for x in tb])),
                                         bits_per_token=float(np.median([x['bytes'] * 8 / max(x['tokens'], 1) for x in tb])), n=len(r))
    json.dump(dict(summary=S, rows=rows), open(os.path.join(HERE, 'p6_encoder_boxes.json'), 'w'), indent=1)
    print(json.dumps(S, indent=1))
