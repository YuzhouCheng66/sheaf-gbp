"""A06 'program' codec prototype (Claude A06, 2026-10-27). CPU only. A map is coded as a WORLD consistent with it: a set of axis-aligned
boxes that cover every observed-occupied voxel O and avoid every observed-free voxel F (unknown voxels are don't-care, so boxes may extend
into them), plus a residual voxel set (O not covered by any accepted box) and the poses; the decoder renders boxes + residual and ray-casts
the poses (generator's observe(), verbatim) -> exactly the map. R8: the union map is coded the same way (a relay re-encodes its fused world).
Greedy encoder: seeds in scan order (z, x, y) over uncovered O; a box grows round-robin by one-voxel slabs in 6 directions while the slab
avoids F; accepted if it covers >= m new O voxels. Box fields are entropy-coded with static histograms learned on the OTHER houses
(leave-one-house-out), boxes sorted by x0 (delta-coded); residual with the 9-neighbour ctx coder, context-augmented by the rendered boxes.
Maps: own maps RE-SIMULATED at the demo agent centres with the jitters of a06_raycast.py (same seed) so the decoder can ray-cast.
Reports bytes (boxes + residual + poses) vs ctx(ternary) and ctx(O); exactness; and the lossy 'boxes only' variant scored with metrics."""
import json, math, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a06_common import houses, scene, metrics_rows, mean, RB  # noqa: E402
from a06_raycast import observe, fibonacci_sphere, ctx_bin, BEAMS, POSES, POSE_BYTES  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
M_ACCEPT = int(sys.argv[1]) if len(sys.argv) > 1 else 8


def psum(a):
    p = np.zeros((a.shape[0] + 1, a.shape[1] + 1, a.shape[2] + 1), np.int64)
    p[1:, 1:, 1:] = a.astype(np.int64).cumsum(0).cumsum(1).cumsum(2)
    return p


def bsum(p, x0, x1, y0, y1, z0, z1):
    return int(p[x1, y1, z1] - p[x0, y1, z1] - p[x1, y0, z1] - p[x1, y1, z0] + p[x0, y0, z1] + p[x0, y1, z0] + p[x1, y0, z0] - p[x0, y0, z0])


def cover(t, m_accept):
    O = t == 1; Fm = t == 0; X, Y, Z = t.shape
    pF = psum(Fm); left = O.copy(); boxes = []; skipped = np.zeros_like(O)
    pL = psum(left)
    order = np.argwhere(O.transpose(2, 0, 1))                     # (z, x, y) scan order
    for z, x, y in order:
        if not left[x, y, z] or skipped[x, y, z]:
            continue
        b = [x, x + 1, y, y + 1, z, z + 1]
        grew = True
        while grew:
            grew = False
            for d in range(6):
                nb = list(b)
                if d == 0 and b[1] < X: nb[0], nb[1] = b[1], b[1] + 1
                elif d == 1 and b[0] > 0: nb[0], nb[1] = b[0] - 1, b[0]
                elif d == 2 and b[3] < Y: nb[2], nb[3] = b[3], b[3] + 1
                elif d == 3 and b[2] > 0: nb[2], nb[3] = b[2] - 1, b[2]
                elif d == 4 and b[5] < Z: nb[4], nb[5] = b[5], b[5] + 1
                elif d == 5 and b[4] > 0: nb[4], nb[5] = b[4] - 1, b[4]
                else:
                    continue
                if bsum(pF, *nb) == 0:                            # the new slab avoids observed free space
                    if d == 0: b[1] += 1
                    elif d == 1: b[0] -= 1
                    elif d == 2: b[3] += 1
                    elif d == 3: b[2] -= 1
                    elif d == 4: b[5] += 1
                    else: b[4] -= 1
                    grew = True
        gain = bsum(pL, *b)
        if gain >= m_accept:
            boxes.append(tuple(int(v) for v in b)); left[b[0]:b[1], b[2]:b[3], b[4]:b[5]] = False; pL = psum(left)
        else:
            skipped[x, y, z] = True
    return boxes, left


def render(boxes, shape):
    w = np.zeros(shape, bool)
    for x0, x1, y0, y1, z0, z1 in boxes:
        w[x0:x1, y0:y1, z0:z1] = True
    return w


FIELDS = [('dx0', 129), ('y0', 128), ('z0', 32), ('w', 129), ('h', 129), ('d', 33)]


def fields(boxes):
    bs = sorted(boxes); prev = 0; rows = []
    for x0, x1, y0, y1, z0, z1 in bs:
        rows.append((x0 - prev, y0, z0, x1 - x0, y1 - y0, z1 - z0)); prev = x0
    return rows


def box_bits(rows, hist):
    bits = 16                                                      # box count
    for r in rows:
        for (name, n), v in zip(FIELDS, r):
            h = hist[name]; bits += -math.log2((h[v] + 0.5) / (h.sum() + 0.5 * n))
    return bits


def run_house(path, maps_cache):
    S = scene(path); m = S['m']; n = m['agents']; W = S['occ'].astype(np.uint8); cell = np.asarray(m['cell_m'], np.float64)
    rays = fibonacci_sphere(BEAMS)
    rng = np.random.default_rng(int.from_bytes(m['house'].encode(), 'little') % (2 ** 32))   # same jitters as a06_raycast.py
    centres = np.asarray(m['positions_m'], np.float64)
    jit = rng.uniform(-.1, .1, (POSES, n, 3)); jit[:, :, 2] = 0.
    traj = centres[None] + jit
    own = np.stack([observe(W, cell, traj[:, a], rays) for a in range(n)])
    reached = np.asarray(m['reached'][m['hops'].index(8)], bool)
    T8 = RB.targets(own, reached)
    out = dict(house=m['house'], own=[], union=[])
    for kind, maps in (('own', own), ('union', T8)):
        memo = {}
        for a in range(n):
            t0 = time.time(); t = maps[a]
            members = [a] if kind == 'own' else list(np.nonzero(reached[a])[0])
            key = tuple(members)
            if key not in memo:                                      # an R8 union is the same map for every agent of a component
                boxes, left = cover(t, M_ACCEPT)
                wb = render(boxes, t.shape)
                re = np.stack([observe((wb | left).astype(np.uint8), cell, traj[:, b], rays) for b in members]).max(0)
                lossy = np.stack([observe(wb.astype(np.uint8), cell, traj[:, b], rays) for b in members]).max(0)
                memo[key] = dict(n_boxes=len(boxes), fields=fields(boxes), n_resid=int(left.sum()), n_occ=int((t == 1).sum()),
                                 resid_B=ctx_bin(left, side=wb), ctx_ternary=RB.ctx_bytes(t), ctx_O=ctx_bin(t == 1), poses_B=POSE_BYTES * len(members),
                                 exact=bool((re == t).all()), lossy_pred=lossy, secs=time.time() - t0)
            row = dict(memo[key]); row['agent'] = a; out[kind].append(row)
    # score the lossy 'boxes only' decode with the training metrics (K0 against own, R8 against T8)
    for kind, maps in (('own', own), ('union', T8)):
        P = np.stack([r.pop('lossy_pred') for r in out[kind]])
        rows = metrics_rows(P, maps, S)
        for r, mr in zip(out[kind], rows):
            r.update({f'lossy_{k}': mr[k] for k in ('surface_f1', 'var_occ_f1', 'furn_f1')})
            r['lossy_fpr'] = None
    print(m['house'], 'covered', flush=True)
    return out


if __name__ == '__main__':
    torch.set_num_threads(2)
    part = os.path.join(OUT, f'boxes_m{M_ACCEPT}_partial.json')
    R = json.load(open(part)) if os.path.exists(part) else []
    done = {r['house'] for r in R}
    for path in houses():
        if os.path.basename(path)[6:-5] in done:
            continue
        for attempt in range(40):                           # the shared machine hits its commit limit now and then: wait and retry
            try:
                R.append(run_house(path, None)); break
            except (MemoryError, RuntimeError) as e:
                if not isinstance(e, MemoryError) and 'memory' not in str(e).lower():
                    raise
                print('memory pressure, retry', attempt, flush=True); time.sleep(30)
        json.dump(R, open(part, 'w'))
    # leave-one-house-out static histograms for the box fields -> bits
    for kind in ('own', 'union'):
        for h in R:
            hist = {name: np.zeros(n) for name, n in FIELDS}
            for g in R:
                if g['house'] == h['house']:
                    continue
                for r in g[kind]:
                    for row in r['fields']:
                        for (name, n), v in zip(FIELDS, row):
                            hist[name][v] += 1
            for r in h[kind]:
                r['box_B'] = box_bits(r['fields'], hist) / 8
                r['total_B'] = r['box_B'] + r['resid_B'] + r['poses_B']
    json.dump(R, open(os.path.join(OUT, f'boxes_m{M_ACCEPT}.json'), 'w'))
    for kind in ('own', 'union'):
        rr = [r for h in R for r in h[kind]]
        med = lambda k: float(np.median([r[k] for r in rr]))
        print(f"[{kind}] m={M_ACCEPT}: exact {sum(r['exact'] for r in rr)}/{len(rr)}; boxes med {med('n_boxes'):.0f}; residual voxels med {med('n_resid'):.0f} of {med('n_occ'):.0f}; "
              f"bytes med: boxes {med('box_B'):.0f} + residual {med('resid_B'):.0f} + poses {med('poses_B'):.0f} = {med('total_B'):.0f} (mean {np.mean([r['total_B'] for r in rr]):.0f}) | "
              f"ctx ternary {med('ctx_ternary'):.0f}, ctx O {med('ctx_O'):.0f} | boxes-only lossy: surf {mean([r['lossy_surface_f1'] for r in rr]):.3f} var {mean([r['lossy_var_occ_f1'] for r in rr]):.3f} furn {mean([r['lossy_furn_f1'] for r in rr]):.3f} at {np.median([r['box_B'] + r['poses_B'] for r in rr]):.0f} B")
