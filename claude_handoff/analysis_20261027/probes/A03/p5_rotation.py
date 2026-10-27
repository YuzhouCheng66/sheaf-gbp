"""p5_rotation.py (A03): how much of the structured codec's advantage is the generator's AXIS-ALIGNED furniture?
The 6 demo VAL houses are re-furnished with the SAME objects, each rotated about its footprint centre by theta ~ U(-tmax, tmax)
(tmax = 0, 15, 45 deg; walls, slabs, lintels, poses unchanged). One conservative rasteriser for all three conditions (a cell is
occupied when any of 3 x 3 sub-samples falls inside a primitive; parts with no sub-sample keep the cell of their centre), so tmax = 0
is the control. Agents whose sensor voxel becomes occupied are skipped. Measured per own map: lossless ctx (ternary 3-D context code),
the non-learned codecs of p3 ('plan+box' with the oracle wall / furniture split, 'shell' label-free), their box counts and bytes
(per-field static entropies pooled over the condition, + 14 bit position per box), and the render exactness (2 agents per house).
Usage: python -B p5_rotation.py"""
import json, math, os, sys, time, zlib
from collections import Counter
import numpy as np
from a03np import observe_np, cell_of, load_fx, HERE, box_cover, paint, ctx2d_bytes, ctx_bytes_np

FLOOR_T = 0.10
SUB = 3
TMAX = (0, 15, 45)


def raster_objects(objects, L, rng, tmax):
    NX = 128; cx = L / NX; cz = 0.1
    xs = (np.arange(NX * SUB) + .5) * cx / SUB                          # sub-sample centres along x / y (metres)
    zc = (np.arange(32) + .5) * cz
    obj = np.zeros((NX, NX, 32), bool)
    for o in objects:
        prims = o['prims']
        bx = [(p[1], p[2], p[4], p[5]) if p[0] == 'box' else (p[1] - p[3], p[2] - p[3], p[1] + p[3], p[2] + p[3]) for p in prims]
        ox = (min(b[0] for b in bx) + max(b[2] for b in bx)) / 2; oy = (min(b[1] for b in bx) + max(b[3] for b in bx)) / 2
        th = math.radians(float(rng.uniform(-tmax, tmax))) if tmax else 0.
        c, s = math.cos(th), math.sin(th)
        for p in prims:
            if p[0] == 'box':
                _, x0, y0, z0, x1, y1, z1 = p
                pc = np.array([(x0 + x1) / 2 - ox, (y0 + y1) / 2 - oy]); hw = np.array([(x1 - x0) / 2, (y1 - y0) / 2])
                ctr = np.array([ox + c * pc[0] - s * pc[1], oy + s * pc[0] + c * pc[1]])
                rad = float(np.hypot(*hw)) + cx
            else:
                _, px, py, r, z0, z1 = p
                pc = np.array([px - ox, py - oy]); ctr = np.array([ox + c * pc[0] - s * pc[1], oy + s * pc[0] + c * pc[1]]); rad = r + cx
            i0 = max(0, int((ctr[0] - rad) / cx) * SUB); i1 = min(NX * SUB, (int((ctr[0] + rad) / cx) + 1) * SUB)
            j0 = max(0, int((ctr[1] - rad) / cx) * SUB); j1 = min(NX * SUB, (int((ctr[1] + rad) / cx) + 1) * SUB)
            if i1 <= i0 or j1 <= j0:
                continue
            X, Y = np.meshgrid(xs[i0:i1], xs[j0:j1], indexing='ij')
            dx, dy = X - ctr[0], Y - ctr[1]
            if p[0] == 'box':
                u = c * dx + s * dy; v = -s * dx + c * dy                   # into the part's own (unrotated) frame
                ins = (np.abs(u) <= hw[0]) & (np.abs(v) <= hw[1])
            else:
                ins = dx * dx + dy * dy <= r * r
            m = np.zeros((NX * SUB, NX * SUB), bool); m[i0:i1, j0:j1] = ins
            m2 = m.reshape(NX, SUB, NX, SUB).any((1, 3))
            if not m2.any():
                m2[min(NX - 1, max(0, int(ctr[0] // cx))), min(NX - 1, max(0, int(ctr[1] // cx)))] = True
            iz = (zc >= FLOOR_T + z0) & (zc < FLOOR_T + z1)
            if not iz.any():
                iz[min(31, max(0, int((FLOOR_T + (z0 + z1) / 2) // cz)))] = True
            obj |= m2[:, :, None] & iz[None, None, :]
    return obj


def run(args):
    name, tmax = args
    t0 = time.time()
    fx, meta = load_fx(name); L = meta['chart_m']; cell = cell_of(L); traj = fx['trajectory']
    occ0 = fx['occ'].astype(bool); slab = fx['slab'].astype(bool); obj0 = fx['obj'].astype(bool)
    struct = occ0 & ~obj0 & ~slab
    wall2d = struct[:, :, 5]; lint2d = struct[:, :, 23] & ~wall2d
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    obj = raster_objects(meta['objects'], L, rng, tmax) & ~wall2d[:, :, None] & ~slab
    occ = struct | slab | obj
    lz0 = int(np.argmax(struct[lint2d].any(0))) if lint2d.any() else 21
    rows = []
    for a in range(traj.shape[1]):
        pv = np.floor(traj[:, a] / np.asarray(cell)).astype(int)
        if any(occ[tuple(v)] for v in pv):
            continue
        own = observe_np(occ, cell, traj[:, a])
        hits = own == 1; free = own == 0
        row = dict(house=name, tmax=tmax, agent=a, ctx=ctx_bytes_np(own), hits_ctx=ctx_bytes_np(hits, 2, 0), furn_hits=int((hits & obj).sum()))
        sc = (hits & struct).any(2)
        plan = np.where(sc & wall2d, 1, np.where(sc & lint2d, 2, np.where(sc, 1, 0))).astype(np.int8)
        fb = box_cover(hits & obj, free)
        sb = box_cover(hits & ~slab, ~(hits & ~slab))
        row.update(plan_bytes=ctx2d_bytes(plan, 3), plan_boxes=[list(b) for b in fb], shell_boxes=[list(b) for b in sb])
        if a < 2:                                                         # render exactness spot check
            g = slab | (plan == 1)[:, :, None] | paint(fb)
            lz = np.zeros(32, bool); lz[lz0:] = True; g |= (plan == 2)[:, :, None] & lz[None, None, :]
            row['exact_plan'] = bool((observe_np(g, cell, traj[:, a]) == own).all())
            row['exact_shell'] = bool((observe_np(slab | paint(sb), cell, traj[:, a]) == own).all())
        rows.append(row)
    print(name, tmax, len(rows), 'agents', f'{time.time()-t0:.0f}s', flush=True)
    return rows


def bits_of(boxes, tab):
    sizes = (32, 128, 128, 32); tot = 8.
    for b in boxes:
        tot += 14
        for f, v in enumerate((b[4], b[1] - b[0], b[3] - b[2], b[5] - b[4])):
            cc = tab[f]; N = sum(cc.values()); tot += -math.log2((cc.get(v, 0) + 0.5) / (N + 0.5 * sizes[f]))
    return tot


if __name__ == '__main__':
    import multiprocessing as mp
    houses = ['r1_i0', 'r1_i1', 'r2_i0', 'r2_i1', 'r3_i0', 'r3_i1']
    with mp.Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as pool:
        R = [r for rr in pool.map(run, [(h, t) for t in TMAX for h in houses]) for r in rr]
    S = {}
    for t in TMAX:
        rr = [r for r in R if r['tmax'] == t]
        for kind in ('plan_boxes', 'shell_boxes'):
            tab = [Counter(), Counter(), Counter(), Counter()]
            for r in rr:
                for b in r[kind]:
                    tab[0][b[4]] += 1; tab[1][b[1] - b[0]] += 1; tab[2][b[3] - b[2]] += 1; tab[3][b[5] - b[4]] += 1
            for r in rr:
                r[kind + '_bytes'] = bits_of(r[kind], tab) / 8 + 60 / 8 + (r['plan_bytes'] if kind == 'plan_boxes' else 0.)
        med = lambda k: float(np.median([r[k] for r in rr]))
        S[t] = dict(n=len(rr), ctx=med('ctx'), hits_plus_poses=float(np.median([r['hits_ctx'] + 5.5 for r in rr])),
                    plan_box_bytes=med('plan_boxes_bytes'), shell_bytes=med('shell_boxes_bytes'),
                    n_plan_boxes=float(np.median([len(r['plan_boxes']) for r in rr])), n_shell_boxes=float(np.median([len(r['shell_boxes']) for r in rr])),
                    furn_hits=med('furn_hits'), ctx_over_planbox=float(np.median([r['ctx'] / r['plan_boxes_bytes'] for r in rr])),
                    ctx_over_shell=float(np.median([r['ctx'] / r['shell_boxes_bytes'] for r in rr])),
                    exact_plan=sum(1 for r in rr if r.get('exact_plan')), exact_shell=sum(1 for r in rr if r.get('exact_shell')),
                    checked=sum(1 for r in rr if 'exact_plan' in r))
        print(t, json.dumps(S[t]))
    json.dump(dict(summary=S, rows=[{k: v for k, v in r.items() if not k.endswith('_boxes')} for r in R]), open(os.path.join(HERE, 'p5_rotation.json'), 'w'), indent=1)
