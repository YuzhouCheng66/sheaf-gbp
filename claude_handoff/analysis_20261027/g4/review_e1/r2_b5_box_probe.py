"""r2_b5_box_probe.py (code review of G4-E1, 2026-10-27): does the B5 arm of criterion (b) depend on HOW small structure is deleted?
Pre-registered B5 (D8-36 / C1 / C2 F13 / A03 p12): log-odds threshold x "delete boxes" with minimum support m (A03: keep only greedy
solid boxes covering >= m observed hits) x {1, 2x2x1}. The implementation deletes 26-connected components with < m voxels instead.
This probe swaps ONLY g4_noise_panels.b5_variants (monkeypatch, module unchanged) for the box version and reruns the module's run_b5
on the cam tier (public frame only), next to the module's own CC version on the same maps.
Box version: occ = known & L > tau; boxes = g4_exact_codec.box_cover(occ, known & ~occ) (A03 greedy cover, never enters an observed
free voxel); keep boxes whose box holds >= m occupied voxels; decoded ternary = known ? (voxel in a kept box) : unknown (painting
into unknown voxels is not shown, nothing else changes). 2x2x1 as the module (down_tern). Coding, fusion, scoring: module's run_b5.
CPU, single process.  python -B r2_b5_box_probe.py [--houses 1:0,1:1] [--seed 1]"""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'claude_v342'))
import g4_noise_panels as G  # noqa: E402
from g4_exact_codec import box_cover  # noqa: E402
W = G.W


def b5_variants_box(L):
    known = L != G.UNK; out = {}
    for tau in G.B5_TAUS:
        occ = known & (L > tau)
        boxes = box_cover(occ, known & ~occ)
        sup = np.array([int(occ[x0:x1, y0:y1, z0:z1].sum()) for x0, x1, y0, y1, z0, z1 in boxes], np.int64)
        for m in G.B5_MS:
            P = np.zeros(L.shape, bool)
            for b, s in zip(boxes, sup):
                if s >= m:
                    x0, x1, y0, y1, z0, z1 = b; P[x0:x1, y0:y1, z0:z1] = True
            D = np.where(known, P.astype(np.int8), np.int8(-1)).astype(np.int8)
            for f in G.B5_DS:
                out[(tau, m, f)] = D if f == (1, 1, 1) else G.down_tern(D, f)
    return out


def points(res):
    pts = {}
    for key, rec in res.items():
        for fu in ('sum', 'max'):
            rows = rec['pub'][fu]['R8_rows']
            pts[f'{key}|{fu}'] = (rec['pub'][fu]['rnet'], float(np.mean([r['furn_f1'] for r in rows if r['furn_f1'] is not None])))
    return pts


def run_house(rooms, index, seed, cname='cam'):
    t0 = time.time()
    fx = W.make_fixture('VALIDATION', index, rooms)
    occ = fx['occ']; cell = np.asarray(fx['cell'], np.float64); private = fx['private']; traj = fx['trajectory_m']
    positions = fx['positions_m']; n = private.shape[0]; shape = occ.shape; ds = 0.5 * float(cell.min())
    H = G.House(occ, fx['slab'], fx['obj'], cell); F = G.forest(fx['edges'], n)
    reached8 = np.asarray(fx['reached'][G.HOPS.index(8)], bool)
    T8 = G.stream_targets(private, reached8)
    K0T = [G.Target(private[a], H) for a in range(n)]
    R8T = [G.Target(T8[cm[0]], H, mult=(private[cm] >= 0).sum(0)) for cm in F['comps']]
    rays = W.fibonacci_sphere(W.BEAMS); c = G.CONDITIONS[cname]
    L_pub = []
    for a in range(n):
        z = G.frame_draws(seed, rooms, index, a)
        frame = (G.rotz(float(z[2] * c['inter_yaw'])), np.array([z[0] * c['inter_t'], z[1] * c['inter_t'], 0.]), np.asarray(positions[a], np.float64))
        scans = []
        for j in range(traj.shape[0]):
            hs, ls = G.trace_pose(occ, cell, traj[j, a], rays)
            scan, _ = G.measure(c, hs, ls, traj[j, a], G.pose_draws(seed, rooms, index, a, j, len(rays)), ds, rays)
            scans.append(G.to_public(scan, frame))
        L_pub.append(G.integrate_agent(shape, cell, ds, scans))
    # B4 (judged) for the quarter-rate budget and the B4 furniture reference
    msgs, preds = G.fuse(L_pub, F, 'sum')
    b4 = float(sum(G.int_ctx_bytes(*msgs[e]) for e in F['directed']))
    rows = [G.score(preds[cm[0]], R8T[ci]) for ci, cm in enumerate(F['comps']) for _ in cm]
    b4f = float(np.mean([r['furn_f1'] for r in rows if r['furn_f1'] is not None]))
    out = dict(house=f'r{rooms}_i{index}', seed=seed, cond=cname, B4_relay=b4, B4_R8_furn=b4f, fam={})
    print(out['house'], 'B4', round(b4), round(b4f, 4), f'{time.time() - t0:.0f}s', flush=True)
    orig = G.b5_variants
    for fam, fn in (('cc', orig), ('box', b5_variants_box)):
        G.b5_variants = fn
        try:
            res = G.run_b5(L_pub, L_pub, False, F, K0T, R8T, G.RelayStats(), {})
        finally:
            G.b5_variants = orig
        pts = points(res); out['fam'][fam] = pts
        P = list(pts.values()); q = 0.25 * b4
        print(out['house'], fam, 'hull@q', G.hull_value(P, q), 'best', max(y for _, y in P), 'rates', round(min(x for x, _ in P)), round(max(x for x, _ in P)),
              f'{time.time() - t0:.0f}s', flush=True)
    out['seconds'] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--houses', default='1:0,1:1')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default=os.path.join(HERE, 'r2_b5_box_probe.json'))
    args = ap.parse_args()
    res = json.load(open(args.out)) if os.path.exists(args.out) else []
    done = {(r['house'], r['seed']) for r in res}
    for hs in args.houses.split(','):
        rooms, index = map(int, hs.split(':'))
        if (f'r{rooms}_i{index}', args.seed) in done:
            continue
        res.append(run_house(rooms, index, args.seed))
        json.dump(res, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
