"""A05 probe 4 (CPU, demo data): how many axis-aligned BOX TOKENS does the observed furniture need, and what would they cost?
gt_obj (the rasterised union of the generator's primitives) is decomposed greedily into maximal boxes on the voxel grid (largest
first; a box only covers obj voxels, so every box is world-occupied = sound). For each own map / round-8 union: visible boxes = boxes
that contain >= 1 observed furniture hit. G = exact structure + visible boxes satisfies hits <= G <= world, so RayCast(G, poses)
reproduces the map exactly (checked with the true poses of p2a). Token cost estimates: raw = 2 x 7 bit (x0, x1) + 2 x 7 (y) + 2 x 5 (z)
= 38 bit; 'delta' = an adaptive empirical code of each coordinate given the previous token sorted in raster order (a crude stand-in
for a learned prior), measured leave-one-house-out. Also: boxes per object (connected components of gt_obj)."""
import glob, json, math, os, sys, time
import numpy as np
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, HERE)
from rate_baselines_lib import load, targets  # noqa: E402
from p2b_raycast_decode import ray_table, cast, struct_parts  # noqa: E402


def greedy_boxes(mask):
    """maximal-box greedy cover of a boolean 3-D mask (boxes inside the mask; each voxel covered at least once)"""
    m = mask.copy(); boxes = []
    X, Y, Z = m.shape
    while m.any():
        x, y, z = np.argwhere(m)[0]                                    # lowest uncovered voxel in raster order
        # grow along z, then y, then x, staying inside the ORIGINAL mask (may overlap covered voxels)
        z1 = z
        while z1 + 1 < Z and mask[x, y, z1 + 1]: z1 += 1
        y1 = y
        while y1 + 1 < Y and mask[x, y1 + 1, z:z1 + 1].all(): y1 += 1
        x1 = x
        while x1 + 1 < X and mask[x1 + 1, y:y1 + 1, z:z1 + 1].all(): x1 += 1
        boxes.append((x, x1 + 1, y, y1 + 1, z, z1 + 1))
        m[x:x1 + 1, y:y1 + 1, z:z1 + 1] = False
    return np.array(boxes, np.int64)


def raster(boxes, shape):
    g = np.zeros(shape, bool)
    for x0, x1, y0, y1, z0, z1 in boxes:
        g[x0:x1, y0:y1, z0:z1] = True
    return g


if __name__ == '__main__':
    poses = json.load(open(os.path.join(HERE, 'poses.json')))
    rows = []; tokens = []
    for path in sorted(glob.glob(os.path.join(ROOT, 'demo_web3d', 'data', 'house_*.json'))):
        t0 = time.time()
        m, own, pred, gt = load(path); n = own.shape[0]; key = m['house']
        traj = np.asarray(poses[key]['trajectory_m']); cell = tuple(poses[key]['cell']); lk = poses[key]['lintel_k']
        obj = gt['obj']
        boxes = greedy_boxes(obj)
        lab, nobj = ndimage.label(obj)
        assert (raster(boxes, obj.shape) == obj).all()
        # box membership: which boxes contain a hit
        R8 = np.asarray(m['reached'][m['hops'].index(8)], bool); T8 = targets(own, R8)
        tables = {a: [ray_table(traj[k, a], cell) for k in range(traj.shape[0])] for a in range(n)}
        for hop, T in ((0, own), (8, T8)):
            for a in range(n):
                t = T[a]; Hf = (t == 1) & obj
                vis = np.array([Hf[x0:x1, y0:y1, z0:z1].any() for x0, x1, y0, y1, z0, z1 in boxes], bool)
                Gs, wall, lint = struct_parts(t == 1, gt, lk)
                G = Gs | raster(boxes[vis], obj.shape)
                agents = [a] if hop == 0 else list(np.nonzero(R8[a])[0])
                exact = None
                if hop == 0 or a == min(agents):
                    exact = bool((cast(G, [tb for b in agents for tb in tables[b]]) == t).all())
                objs_seen = len(set(lab[Hf].tolist()) - {0})
                rows.append(dict(house=key, hop=hop, agent=a, boxes_visible=int(vis.sum()), objects_seen=objs_seen, exact=exact))
                if hop == 0:
                    tokens.append((key, boxes[vis]))
        rows.append(dict(house=key, hop='house', boxes_total=len(boxes), objects=int(nobj), boxes_per_object=len(boxes) / max(nobj, 1)))
        print(key, len(boxes), 'boxes', nobj, 'objects', f'{time.time() - t0:.0f}s', flush=True)

    # crude token code: per coordinate, empirical entropy of the difference to the previous token (raster-sorted), leave-one-house-out
    houses = sorted({k for k, _ in tokens})
    def diffs(b):
        if len(b) == 0:
            return np.zeros((0, 6), np.int64)
        o = b[np.lexsort((b[:, 4], b[:, 2], b[:, 0]))]
        d = o.copy(); d[1:, [0, 2, 4]] = o[1:, [0, 2, 4]] - o[:-1, [0, 2, 4]]
        d[:, 1] = o[:, 1] - o[:, 0]; d[:, 3] = o[:, 3] - o[:, 2]; d[:, 5] = o[:, 5] - o[:, 4]   # extents
        return d
    per_map_bits = []
    for h in houses:
        train = np.concatenate([diffs(b) for k, b in tokens if k != h and len(b)])
        for k, b in tokens:
            if k != h:
                continue
            d = diffs(b); bits = 0.
            for c in range(6):
                vals, cnt = np.unique(train[:, c], return_counts=True); p = dict(zip(vals.tolist(), cnt.tolist())); N = cnt.sum(); V = 256
                bits += sum(-math.log2((p.get(int(v), 0) + 0.5) / (N + 0.5 * V)) for v in d[:, c])
            per_map_bits.append(dict(house=k, boxes=len(b), bits=bits, bits_per_box=bits / max(len(b), 1)))
    S = {}
    for hop in (0, 8):
        r = [x for x in rows if x['hop'] == hop]
        S[f'hop{hop}'] = dict(boxes_visible_median=float(np.median([x['boxes_visible'] for x in r])), boxes_visible_mean=float(np.mean([x['boxes_visible'] for x in r])),
                              objects_seen_median=float(np.median([x['objects_seen'] for x in r])),
                              exact_all=all(x['exact'] for x in r if x['exact'] is not None))
    h = [x for x in rows if x['hop'] == 'house']
    S['house'] = dict(boxes_total_median=float(np.median([x['boxes_total'] for x in h])), objects_median=float(np.median([x['objects'] for x in h])),
                      boxes_per_object=float(np.median([x['boxes_per_object'] for x in h])))
    S['k0_token_code'] = dict(bits_per_box_median=float(np.median([x['bits_per_box'] for x in per_map_bits if x['boxes']])),
                              bytes_median=float(np.median([x['bits'] / 8 for x in per_map_bits])))
    json.dump(dict(summary=S, rows=rows, token_code=per_map_bits), open(os.path.join(HERE, 'p4_box_tokens.json'), 'w'), indent=1)
    print(json.dumps(S, indent=1))
