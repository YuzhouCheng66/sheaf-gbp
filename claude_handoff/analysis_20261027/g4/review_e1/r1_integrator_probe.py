"""r1_integrator_probe.py (code review of G4-E1, 2026-10-27): how much do criteria (b) / (c) depend on the log-odds integrator?
Reviewer's probe, does NOT modify g4_noise_panels.py; it imports its tracer / noise model / coders / metrics / B5 unchanged and only
swaps the per-scan integration rule. All three rules are computed from the SAME per-scan beam counts (nh, nm) of integrate_scan:
  V0 impl    : per scan L <- clamp(L + 17 nh - 8 nm)               (g4_noise_panels.integrate_agent, asserted equal)
  V1 octomap : per scan, a voxel with >= 1 endpoint gets ONE hit, else a voxel crossed by >= 1 beam gets ONE miss (OctoMap
               computeUpdate: occupied keys removed from the free key set), clamp [-40, 70] after every update (updateNodeLogOdds)
  V2 probe   : L = sum over scans of 17 nh - 8 nm, no clamp (A07 p3 / A10 / C1 probes)
Fusions computed directly (= the relay, which the module asserts equal): sum, maxabs, max_occ; B4 relay / flood bytes with the module's
int coder; B5 (module's run_b5) for V0 and V1 on cam, plus V1 with the threshold read as L >= tau (tau > 0).
CPU, single process.  python -B r1_integrator_probe.py [--houses 1:0,1:1,2:0,3:0] [--seed 1] [--b5]"""
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
W = G.W
from scipy import ndimage  # noqa: E402

VARIANTS = ('V0_impl', 'V1_octomap', 'V2_probe')


def agent_variants(shape, cell, ds, scans):
    V = int(np.prod(shape))
    L0 = np.zeros(V, np.int32); L1 = np.zeros(V, np.int32); L2 = np.zeros(V, np.int64); known = np.zeros(V, bool)
    for o, D, r, sm in scans:
        nh, nm = G.integrate_scan(shape, cell, ds, o, D, r, sm)
        t = (nh + nm) > 0
        L0[t] = np.clip(L0[t] + G.L_HIT * nh[t] + G.L_MISS * nm[t], G.L_MIN, G.L_MAX)
        h = nh > 0; m = (nm > 0) & ~h
        L1[h] = np.minimum(L1[h] + G.L_HIT, G.L_MAX); L1[m] = np.maximum(L1[m] + G.L_MISS, G.L_MIN)
        L2 += G.L_HIT * nh + G.L_MISS * nm
        known |= t
    return {'V0_impl': (L0.reshape(shape), known.reshape(shape)), 'V1_octomap': (L1.reshape(shape), known.reshape(shape)),
            'V2_probe': (L2.reshape(shape), known.reshape(shape))}


def tern(L, K):
    return np.where(K, (L > 0).astype(np.int8), np.int8(-1)).astype(np.int8)


def comb(kind, items):
    if kind == 'sum':
        S = np.zeros(items[0][0].shape, np.int64); K = np.zeros(items[0][0].shape, bool)
        for L, k in items:
            S += np.where(k, L, 0); K |= k
        return S, K
    if kind == 'maxabs':
        out = np.zeros(items[0][0].shape, np.int64); key = np.full(items[0][0].shape, -1, np.int64); K = np.zeros(items[0][0].shape, bool)
        for L, k in items:
            kk = np.where(k, 2 * np.abs(L.astype(np.int64)) + (L > 0), -1); m = kk > key
            out[m] = L[m]; key[m] = kk[m]; K |= k
        return out, K
    if kind == 'max_occ':
        return np.maximum.reduce([tern(L, k) for L, k in items]), None
    raise ValueError(kind)


def b5_variants_ge(L):
    """B5 with the threshold read as L >= tau for tau > 0 (L > 0 for tau = 0); everything else as g4_noise_panels.b5_variants"""
    known = L != G.UNK; out = {}
    for tau in G.B5_TAUS:
        occ = known & ((L > 0) if tau == 0 else (L >= tau))
        lab, _ = ndimage.label(occ, G.S26); sizes = np.bincount(lab.ravel())
        for m in G.B5_MS:
            small = sizes < m; small[0] = False
            D = np.where(known, (occ & ~small[lab]).astype(np.int8), np.int8(-1)).astype(np.int8)
            for f in G.B5_DS:
                out[(tau, m, f)] = D if f == (1, 1, 1) else G.down_tern(D, f)
    return out


def b5_points(res):
    pts = {}
    for key, rec in res.items():
        for fu in ('sum', 'max'):
            rows = rec['pub'][fu]['R8_rows']
            pts[f'{key}|{fu}'] = (rec['pub'][fu]['rnet'], float(np.mean([r['furn_f1'] for r in rows if r['furn_f1'] is not None])))
    return pts


def run_house(rooms, index, seed, conds, do_b5):
    t0 = time.time()
    fx = W.make_fixture('VALIDATION', index, rooms)
    occ = fx['occ']; cell = np.asarray(fx['cell'], np.float64); private = fx['private']; traj = fx['trajectory_m']
    positions = fx['positions_m']; n = private.shape[0]; shape = occ.shape; ds = 0.5 * float(cell.min())
    H = G.House(occ, fx['slab'], fx['obj'], cell); F = G.forest(fx['edges'], n)
    reached8 = np.asarray(fx['reached'][G.HOPS.index(8)], bool)
    T8 = G.stream_targets(private, reached8)
    K0T = [G.Target(private[a], H) for a in range(n)]
    R8T = [G.Target(T8[cm[0]], H, mult=(private[cm] >= 0).sum(0)) for cm in F['comps']]
    rays = W.fibonacci_sphere(W.BEAMS)
    traces = [[G.trace_pose(occ, cell, traj[j, a], rays) for j in range(traj.shape[0])] for a in range(n)]
    fm = [len(F['comps'][F['comp_of'][a]]) - 1 for a in range(n)]
    out = dict(house=f'r{rooms}_i{index}', seed=seed, agents=n, comps=F['comps'], conds={})
    for cname in conds:
        c = G.CONDITIONS[cname]; has_inter = c['inter_t'] > 0 or c['inter_yaw'] > 0
        K0v = {v: [] for v in VARIANTS}; PUBv = {v: [] for v in VARIANTS}
        for a in range(n):
            frame = None
            if has_inter:
                z = G.frame_draws(seed, rooms, index, a)
                frame = (G.rotz(float(z[2] * c['inter_yaw'])), np.array([z[0] * c['inter_t'], z[1] * c['inter_t'], 0.]), np.asarray(positions[a], np.float64))
            scans = []
            for j in range(traj.shape[0]):
                hs, ls = traces[a][j]
                scan, _ = G.measure(c, hs, ls, traj[j, a], G.pose_draws(seed, rooms, index, a, j, len(rays)), ds, rays)
                scans.append(scan)
            vk = agent_variants(shape, cell, ds, scans)
            if a == 0:                                                        # V0 must be the module's integrator exactly
                ref = G.integrate_agent(shape, cell, ds, scans)
                L0, K0_ = vk['V0_impl']; assert np.array_equal(np.where(K0_, L0, G.UNK).astype(np.int8), ref), 'V0 != module'
            vp = agent_variants(shape, cell, ds, [G.to_public(s, frame) for s in scans]) if has_inter else vk
            for v in VARIANTS:
                K0v[v].append(vk[v]); PUBv[v].append(vp[v])
        crec = {}
        for v in VARIANTS:
            r = {}
            k0rows = [G.score(tern(*K0v[v][a]), K0T[a]) for a in range(n)]
            r['K0_furn'] = float(np.mean([x['furn_f1'] for x in k0rows if x['furn_f1'] is not None]))
            r['K0_free'] = float(np.mean([x['free_recall'] for x in k0rows if x['free_recall'] is not None]))
            r['K0_exact'] = all(np.array_equal(tern(*K0v[v][a]), private[a]) for a in range(n))
            sets = [('pub', PUBv[v])] + ([('truepose', K0v[v])] if has_inter else [])
            for ms, maps in sets:
                for kind in ('sum', 'maxabs', 'max_occ'):
                    rows = []
                    for ci, cm in enumerate(F['comps']):
                        B, Kb = comb(kind, [maps[a] for a in cm])
                        p = B if kind == 'max_occ' else tern(B, Kb)
                        if ms == 'pub':
                            r.setdefault('R8_exact_' + kind, True)
                            r['R8_exact_' + kind] &= bool(np.array_equal(p, T8[cm[0]]))
                        row = G.score(p, R8T[ci])
                        rows += [row] * len(cm)
                    tag = kind if ms == 'pub' else kind + '_truepose'
                    r['R8_furn_' + tag] = float(np.mean([x['furn_f1'] for x in rows if x['furn_f1'] is not None]))
                    r['R8_free_' + tag] = float(np.mean([x['free_recall'] for x in rows if x['free_recall'] is not None]))
                    r['R8_fpr_' + tag] = float(np.mean([x['unknown_known_fpr'] for x in rows if x['unknown_known_fpr'] is not None]))
            maps = PUBv[v]
            r['B4_relay'] = float(sum(G.int_ctx_bytes(*comb('sum', [maps[b] for b in F['sides'][e]])) for e in F['directed']))
            r['B4_flood'] = float(sum(G.int_ctx_bytes(L, K) * k for (L, K), k in zip(maps, fm)))
            r['B4_maxabs_relay'] = float(sum(G.int_ctx_bytes(*comb('maxabs', [maps[b] for b in F['sides'][e]])) for e in F['directed']))
            r['B3_relay'] = float(sum(G.ctx_tern_bytes(comb('max_occ', [maps[b] for b in F['sides'][e]])[0]) for e in F['directed']))
            vals = np.concatenate([L[K] for L, K in maps])
            r['own_alphabet'] = int(len(np.unique(vals))); r['own_frac_saturated'] = float(np.mean((vals <= G.L_MIN) | (vals >= G.L_MAX)))
            crec[v] = r
            print(out['house'], cname, v, json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in r.items()}), f'{time.time() - t0:.0f}s', flush=True)
        if do_b5 and cname == 'cam':
            for v, fn in (('V0_impl', None), ('V1_octomap', None), ('V1_octomap_ge', b5_variants_ge)):
                base = v.replace('_ge', '')
                L_pub = [np.where(K, L, G.UNK).astype(np.int8) for L, K in PUBv[base]]
                L_k0 = [np.where(K, L, G.UNK).astype(np.int8) for L, K in K0v[base]]
                orig = G.b5_variants
                if fn is not None:
                    G.b5_variants = fn
                try:
                    res = G.run_b5(L_pub, L_k0, True, F, K0T, R8T, G.RelayStats(), {})
                finally:
                    G.b5_variants = orig
                crec.setdefault('B5', {})[v] = b5_points(res)
                pts = list(crec['B5'][v].values()); q = 0.25 * crec[base]['B4_relay']
                print(out['house'], 'B5', v, 'hull@q', G.hull_value(pts, q), 'best', max(y for _, y in pts), 'q', round(q), f'{time.time() - t0:.0f}s', flush=True)
        out['conds'][cname] = crec
    out['seconds'] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--houses', default='1:0,1:1,2:0,3:0')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--conds', default='clean,range_only,cam_nf,cam')
    ap.add_argument('--b5', action='store_true')
    ap.add_argument('--out', default=os.path.join(HERE, 'r1_integrator_probe.json'))
    args = ap.parse_args()
    res = []
    if os.path.exists(args.out):
        res = json.load(open(args.out))
    done = {(r['house'], r['seed']) for r in res}
    for hs in args.houses.split(','):
        rooms, index = map(int, hs.split(':'))
        if (f'r{rooms}_i{index}', args.seed) in done:
            continue
        res.append(run_house(rooms, index, args.seed, args.conds.split(','), args.b5))
        json.dump(res, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
