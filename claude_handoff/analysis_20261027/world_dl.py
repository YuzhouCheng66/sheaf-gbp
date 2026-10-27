"""world_dl.py (Claude, 2026-10-27): description length of what each agent saw, if the codec knew the world generator and the sensor.
For VAL houses (first `--per-stratum` houses per room stratum that place 8 agents, the same rule as the demo export), per agent and for the
round-8 union: (a) the exact adaptive context code of the voxel map (ctx, as in rate_baselines.py), (b) a parametric description length
DL = visible contour vertices x 14 bit (x, y at the R1 cell) + visible objects x bits_obj + 16 bit (wall and door heights) + poses
(3 sensor poses per agent, 44 bit: the centre at the R1 cell + 3 jitters at 1 cm in +-0.1 m). bits_obj = kind (log2 22) + footprint corner
(14) + depth and width on the kind's range at the R1 cell + side (log2 5) + 4 bit of per-kind heights. An object is visible when one of its
primitive boxes' voxels is observed occupied; a vertex when an observed-occupied voxel lies within 2 cells of its column.
DL is an UPPER bound on the information an exact generator-aware codec needs (the map is a deterministic function of these numbers),
so ctx / DL bounds what any codec can gain over a strong lossless voxel coder at full fidelity. CPU only."""
import argparse, json, math, os, sys, time
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); ap.add_argument('--out', required=True); ap.add_argument('--per-stratum', type=int, default=8)
args = ap.parse_args()
sys.path.insert(0, args.source); sys.path.insert(0, os.path.join(args.source, 'exploration_grid')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mapping_3d_storey_data_v420 as W
import torch
from rate_baselines_lib import ctx_bytes

KINDS = len(W.SIZES_KIND)
rows = []
for rooms in (1, 2, 3):
    taken = 0
    for index in range(40):
        if taken >= args.per_stratum:
            break
        t0 = time.time()
        try:
            fx = W.make_fixture('VALIDATION', index, rooms)
        except ValueError:
            continue
        taken += 1
        spec = W.world_spec('VALIDATION', index, rooms)
        L = spec['chart_m']; NX = 128; cell = L / NX; own = fx['private']; n = own.shape[0]
        d = json.load(open(os.path.join(os.environ['HOUSEEXPO_JSON_DIR'], spec['base_spec']['house_id'] + '.json')))
        v = np.asarray(d['verts'], np.float64); centre = (v.min(0) + v.max(0)) / 2; vv = v - centre + L / 2       # chart metres
        vcell = np.clip(np.floor(vv / cell).astype(int), 0, NX - 1)
        obits = []; oboxes = []
        for o in spec['objects']:
            lo_hi = W.SIZES_KIND.get(o['kind'], ((0.3, 0.7), (0.3, 0.7), False))
            b = math.log2(KINDS) + 14 + sum(math.log2(max(hi - lo, 0) / cell + 1) for lo, hi in lo_hi[:2]) + math.log2(5) + 4
            obits.append(b)
            bx = []
            for p in o['prims']:
                if p[0] == 'box':
                    _, x0, y0, z0, x1, y1, z1 = p
                else:
                    _, ccx, ccy, r, z0, z1 = p; x0, y0, x1, y1 = ccx - r, ccy - r, ccx + r, ccy + r
                i0, i1 = int(math.floor(x0 / cell)), int(math.ceil(x1 / cell)); j0, j1 = int(math.floor(y0 / cell)), int(math.ceil(y1 / cell))
                k0, k1 = int(math.floor((W.FLOOR_T + z0) / 0.1)), int(math.ceil((W.FLOOR_T + z1) / 0.1))
                bx.append((max(i0, 0), min(max(i1, i0 + 1), NX), max(j0, 0), min(max(j1, j0 + 1), NX), max(k0, 0), min(max(k1, k0 + 1), 32)))
            oboxes.append(bx)
        obj = fx['obj'].astype(bool)
        def visible(m):
            occ_seen = (m == 1)
            vis_obj = [any((occ_seen[a:b, c:e, f:g] & obj[a:b, c:e, f:g]).any() for a, b, c, e, f, g in bx) for bx in oboxes]
            col = occ_seen.any(2); colp = np.zeros((NX + 4, NX + 4), bool); colp[2:-2, 2:-2] = col
            near = np.zeros_like(col)
            for dx in range(5):
                for dy in range(5):
                    near |= colp[dx:dx + NX, dy:dy + NX]
            vis_v = near[vcell[:, 0], vcell[:, 1]]
            return np.asarray(vis_obj, bool), vis_v
        def dl(vis_obj, vis_v, agents):
            return (14 * int(vis_v.sum()) + float(np.asarray(obits)[vis_obj].sum()) + 16 + 44 * agents) / 8
        for a in range(n):
            vo, vvv = visible(own[a])
            rows.append(dict(kind='own', house=f'r{rooms}_i{index}', agent=a, ctx=ctx_bytes(own[a]), dl=dl(vo, vvv, 1), n_obj=int(vo.sum()), n_vert=int(vvv.sum())))
        hi8 = list(W.HOPS).index(8); R = fx['reached'][hi8]
        for a in range(n):
            U = np.where(R[a][:, None, None, None], own, np.int8(-1)).max(0)
            vo = np.zeros(len(oboxes), bool); vvv = np.zeros(len(vcell), bool)
            for s in np.nonzero(R[a])[0]:
                o_, v_ = visible(own[s]); vo |= o_; vvv |= v_
            rows.append(dict(kind='union8', house=f'r{rooms}_i{index}', agent=a, reached=int(R[a].sum()), ctx=ctx_bytes(U), dl=dl(vo, vvv, int(R[a].sum())), n_obj=int(vo.sum()), n_vert=int(vvv.sum())))
        whole = (14 * len(vcell) + sum(obits) + 16) / 8
        rows.append(dict(kind='house', house=f'r{rooms}_i{index}', n_obj_total=len(oboxes), n_vert_total=len(vcell), dl_world=whole, chart_m=L))
        print(f'r{rooms}_i{index}: {len(oboxes)} objects, {len(vcell)} vertices, world DL {whole:.0f} B, {time.time() - t0:.0f}s', flush=True)
S = {}
for k in ('own', 'union8'):
    r = [x for x in rows if x['kind'] == k]
    S[k] = dict(n=len(r), ctx_median=float(np.median([x['ctx'] for x in r])), dl_median=float(np.median([x['dl'] for x in r])),
                ratio_ctx_over_dl_median=float(np.median([x['ctx'] / x['dl'] for x in r])), objects_median=float(np.median([x['n_obj'] for x in r])),
                vertices_median=float(np.median([x['n_vert'] for x in r])))
h = [x for x in rows if x['kind'] == 'house']
S['house'] = dict(n=len(h), dl_world_median=float(np.median([x['dl_world'] for x in h])), objects_median=float(np.median([x['n_obj_total'] for x in h])),
                  vertices_median=float(np.median([x['n_vert_total'] for x in h])))
json.dump(dict(summary=S, rows=rows), open(args.out, 'w'), indent=1); print(json.dumps(S, indent=1))
