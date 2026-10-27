"""r1b_perbeam_probe.py (code review of G4-E1, 2026-10-27): the per-BEAM reading of "OctoMap-style log-odds, clamped at the OctoMap
default bounds": every beam is one OctoMap insertRay (one miss per crossed voxel, then one hit at the endpoint voxel), the beams of a
scan in index order, scans in pose order, and the clamp [-40, 70] applied after EVERY single update (updateNodeLogOdds). This is the
A07 / A10 / C1 probe convention ("one miss per voxel per beam") plus the D8-36 clamp. Compared with
  V0 impl    : per scan L <- clamp(L + 17 nh - 8 nm)        (g4_noise_panels.integrate_agent)
  V1 octomap : per scan one hit (any endpoint) else one miss (any crossing), clamp after every update (OctoMap insertPointCloud)
  V3 perbeam : as above (order-dependent; deterministic)
Same tracer / noise / CRN / metrics / coders as the module (imported, unchanged). Single process, CPU.
python -B r1b_perbeam_probe.py [--houses 1:0,1:1] [--seed 1] [--conds range_only,cam_nf,cam] [--b5]"""
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


def scan_events(shape, cell, ds, o, D, r, smax):
    """integrate_scan's marching, but returning the event list (voxel, beam, delta) instead of counts"""
    V = int(np.prod(shape)); shape_a = np.array(shape); cell = np.asarray(cell, np.float64)
    hit = ~np.isnan(r); end_f = np.full(len(D), -1, np.int64)
    ev_v, ev_b, ev_d = [], [], []
    if hit.any():
        hi = np.nonzero(hit)[0]
        e = np.floor((o[None] + D[hi] * r[hi, None]) / cell).astype(np.int64)
        ok = np.all((e >= 0) & (e < shape_a), 1)
        ef = (e[ok, 0] * shape[1] + e[ok, 1]) * shape[2] + e[ok, 2]
        end_f[hi[ok]] = ef
        ev_v.append(ef); ev_b.append(hi[ok]); ev_d.append(np.full(len(ef), G.L_HIT, np.int64))
    rr = np.where(hit, r - 1e-9, np.inf); sm = np.where(hit, np.iinfo(np.int32).max, smax)
    act = np.arange(len(D)); last_f = np.full(len(D), -1, np.int64); s = 0
    while len(act):
        s += 1; t = s * ds
        act = act[(t < rr[act]) & (s <= sm[act])]
        if not len(act):
            break
        idx = np.floor((o[None] + D[act] * t) / cell).astype(np.int64)
        inside = np.all((idx >= 0) & (idx < shape_a), 1)
        a2 = act[inside]; idx = idx[inside]
        f = (idx[:, 0] * shape[1] + idx[:, 1]) * shape[2] + idx[:, 2]
        new = (f != last_f[a2]) & (f != end_f[a2])
        ev_v.append(f[new]); ev_b.append(a2[new]); ev_d.append(np.full(int(new.sum()), G.L_MISS, np.int64)); last_f[a2] = f
    v = np.concatenate(ev_v); b = np.concatenate(ev_b); d = np.concatenate(ev_d)
    nh = np.bincount(v[d > 0], minlength=V); nm = np.bincount(v[d < 0], minlength=V)
    return v, b, d, nh, nm


def apply_sequential(L, v, b, d):
    """per voxel, apply its events in beam order, clamping after every update"""
    order = np.lexsort((b, v)); v = v[order]; d = d[order]
    starts = np.flatnonzero(np.r_[True, v[1:] != v[:-1]]); sizes = np.diff(np.r_[starts, len(v)])
    gs = np.argsort(-sizes, kind='stable'); starts = starts[gs]; sizes = sizes[gs]
    vox = v[starts]; cur = L[vox].astype(np.int64); neg = -sizes
    for k in range(int(sizes.max()) if len(sizes) else 0):
        nl = int(np.searchsorted(neg, -k, side='left'))                  # groups with more than k events (sizes sorted descending)
        cur[:nl] = np.clip(cur[:nl] + d[starts[:nl] + k], G.L_MIN, G.L_MAX)
    L[vox] = cur


def agent_maps(shape, cell, ds, scans):
    V = int(np.prod(shape))
    L0 = np.zeros(V, np.int64); L1 = np.zeros(V, np.int64); L3 = np.zeros(V, np.int64); known = np.zeros(V, bool)
    for o, D, r, sm in scans:
        v, b, d, nh, nm = scan_events(shape, cell, ds, o, D, r, sm)
        t = (nh + nm) > 0
        L0[t] = np.clip(L0[t] + G.L_HIT * nh[t] + G.L_MISS * nm[t], G.L_MIN, G.L_MAX)
        h = nh > 0; m = (nm > 0) & ~h
        L1[h] = np.minimum(L1[h] + G.L_HIT, G.L_MAX); L1[m] = np.maximum(L1[m] + G.L_MISS, G.L_MIN)
        apply_sequential(L3, v, b, d)
        known |= t
    f = lambda L: np.where(known, L, G.UNK).astype(np.int8).reshape(shape)  # noqa: E731
    return {'V0_impl': f(L0), 'V1_octomap': f(L1), 'V3_perbeam': f(L3)}


def mean_furn(rows, k='furn_f1'):
    return float(np.mean([r[k] for r in rows if r[k] is not None]))


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
    out = dict(house=f'r{rooms}_i{index}', seed=seed, conds={})
    for cname in conds:
        c = G.CONDITIONS[cname]; has_inter = c['inter_t'] > 0 or c['inter_yaw'] > 0
        K0m = {}; PUBm = {}
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
            mk = agent_maps(shape, cell, ds, scans)
            if a == 0:
                assert np.array_equal(mk['V0_impl'], G.integrate_agent(shape, cell, ds, scans)), 'V0 != module'
            mp = agent_maps(shape, cell, ds, [G.to_public(s, frame) for s in scans]) if has_inter else mk
            for k in mk:
                K0m.setdefault(k, []).append(mk[k]); PUBm.setdefault(k, []).append(mp[k])
        crec = {}
        for k in K0m:
            r = dict(K0_furn=mean_furn([G.score(G.tern(L), K0T[a]) for a, L in enumerate(K0m[k])]),
                     K0_free=mean_furn([G.score(G.tern(L), K0T[a]) for a, L in enumerate(K0m[k])], 'free_recall'))
            for kind in G.FUSIONS:
                msgs, preds = G.fuse(PUBm[k], F, kind)
                rows = [G.score(preds[cm[0]], R8T[ci]) for ci, cm in enumerate(F['comps']) for _ in cm]
                r['R8_furn_' + kind] = mean_furn(rows); r['R8_free_' + kind] = mean_furn(rows, 'free_recall')
                if kind == 'sum':
                    r['B4_relay'] = float(sum(G.int_ctx_bytes(*msgs[e]) for e in F['directed']))
            if k != 'V0_impl':
                same = [float(np.mean(G.tern(K0m[k][a]) == G.tern(K0m['V0_impl'][a]))) for a in range(n)]
                r['K0_tern_agree_with_V0'] = float(np.mean(same))
            crec[k] = r
            print(out['house'], cname, k, json.dumps({x: round(y, 4) for x, y in r.items()}), f'{time.time() - t0:.0f}s', flush=True)
        if do_b5 and cname == 'cam':
            res = G.run_b5(PUBm['V3_perbeam'], PUBm['V3_perbeam'], False, F, K0T, R8T, G.RelayStats(), {})
            pts = [(rec['pub'][fu]['rnet'], mean_furn(rec['pub'][fu]['R8_rows'])) for rec in res.values() for fu in ('sum', 'max')]
            crec['V3_perbeam']['B5_hull_at_quarter'] = G.hull_value(pts, 0.25 * crec['V3_perbeam']['B4_relay'])
            crec['V3_perbeam']['B5_best'] = max(y for _, y in pts)
            print(out['house'], 'B5 V3', crec['V3_perbeam']['B5_hull_at_quarter'], crec['V3_perbeam']['B5_best'], f'{time.time() - t0:.0f}s', flush=True)
        out['conds'][cname] = crec
    out['seconds'] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--houses', default='1:0,1:1')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--conds', default='range_only,cam_nf,cam')
    ap.add_argument('--b5', action='store_true')
    ap.add_argument('--out', default=os.path.join(HERE, 'r1b_perbeam_probe.json'))
    args = ap.parse_args()
    res = json.load(open(args.out)) if os.path.exists(args.out) else []
    done = {(r['house'], r['seed']) for r in res}
    for hs in args.houses.split(','):
        rooms, index = map(int, hs.split(':'))
        if (f'r{rooms}_i{index}', args.seed) in done:
            continue
        res.append(run_house(rooms, index, args.seed, args.conds.split(','), args.b5))
        json.dump(res, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
