"""Report-only probe (not a judged batch): split the cam-tier inter-agent loss into its yaw and translation parts, and read how much a
per-agent Manhattan yaw estimate (from the agent's OWN public-frame evidence, no communication) recovers.

Arms (each agent's believed scans moved by K_a, then V3 re-integration, forest sum, G4-E4's as-is / aligned rows):
  noreg        K = I                                   (= G4-E4 'noreg')
  oracle_yaw   removes the true yaw of A_a about the agent's believed position, keeps the translation
  oracle_trans removes the true translation, keeps the yaw
  truepose     K0 maps (= cam_nf, = G4-E4 'truepose')
  manhattan    K = rotation by -d_a about the agent's believed position; d_a = 1/4 circular mean of 4 x (wall-surfel normal angle),
               weighted by surfel weight, from planes_logodds(L_pub[a]) (G4-E4's surfels, public grid constants only)
Uses the frozen G4-E1 / G4-E4 modules by import only.  Usage: python -B probe_yaw.py --out DIR [--procs 12] [--panels val] [--seeds 1,2]
"""
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
sys.dont_write_bytecode = True
import argparse, glob, json, math, multiprocessing, time, traceback  # noqa: E401,E402
import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'claude_v342'))
import g4_registration as R  # noqa: E402
G = R.G; W = G.W; I3 = R.I3
ARMS = ('noreg', 'oracle_yaw', 'oracle_trans', 'manhattan', 'truepose')
COND = 'cam'


def manhattan_yaw(L, cell, prm):
    S = R.evidence(R.planes_logodds(L, cell, prm), cell, prm)['S']
    if not len(S['n']):
        return 0.0, 0
    phi = np.arctan2(S['n'][:, 1], S['n'][:, 0]); w = S['w']
    return 0.25 * math.atan2(float((w * np.sin(4 * phi)).sum()), float((w * np.cos(4 * phi)).sum())), int(len(phi))


def run_job(job):
    panel, rooms, index, seed, out_dir = job
    prm = R.MATCH; t0 = time.time()
    try:
        fx = W.make_fixture('VALIDATION', index, rooms)
    except ValueError as e:
        return dict(panel=panel, house=f'r{rooms}_i{index}', seed=seed, skipped=repr(e)[:200])
    occ = fx['occ']; cell = np.asarray(fx['cell'], np.float64); private = fx['private']; traj = fx['trajectory_m']
    positions = fx['positions_m']; n = private.shape[0]; shape = occ.shape; ds = 0.5 * float(cell.min())
    H = G.House(occ, fx['slab'], fx['obj'], cell); F = G.forest(fx['edges'], n)
    reached8 = np.asarray(fx['reached'][G.HOPS.index(8)], bool)
    T8 = G.stream_targets(private, reached8)
    R8T = [G.Target(T8[cm[0]], H, mult=(private[cm] >= 0).sum(0)) for cm in F['comps']]
    rays = W.fibonacci_sphere(W.BEAMS)
    traces = [[G.trace_pose(occ, cell, traj[j, a], rays) for j in range(traj.shape[0])] for a in range(n)]
    c = G.CONDITIONS[COND]; rstats = G.RelayStats()
    scans_k0, scans_pub, L_k0, L_pub, MAs, est = [], [], [], [], [], []
    for a in range(n):
        z = G.frame_draws(seed, rooms, index, a)
        frame = (G.rotz(float(z[2] * c['inter_yaw'])), np.array([z[0] * c['inter_t'], z[1] * c['inter_t'], 0.]), np.asarray(positions[a], np.float64))
        MA = R.frame_matrix(frame); MAs.append(MA)
        scans = [G.measure(c, *traces[a][j], traj[j, a], G.pose_draws(seed, rooms, index, a, j, len(rays)), ds, rays)[0] for j in range(traj.shape[0])]
        pub = [G.to_public(s, frame) for s in scans]
        L_k0.append(G.integrate_agent(shape, cell, ds, scans, ('V3',))['V3'])
        L_pub.append(G.integrate_agent(shape, cell, ds, pub, ('V3',))['V3'])
        scans_k0.append(scans); scans_pub.append(pub)
    Ks = {k: [] for k in ('oracle_yaw', 'oracle_trans', 'manhattan')}
    for a in range(n):
        MA = MAs[a]; th = R.yaw(MA); cen = np.asarray(positions[a], np.float64)[:2]
        pb = R.apply2(MA, cen)                                    # the agent's believed position (its own odometry, public frame)
        t = pb - cen
        Ks['oracle_yaw'].append(R.about(-th, pb, np.zeros(2)))
        Ks['oracle_trans'].append(R.se2(0., -t[0], -t[1]))
        d, ns = manhattan_yaw(L_pub[a], cell, prm)
        Ks['manhattan'].append(R.about(-d, pb, np.zeros(2)))
        est.append(dict(agent=a, true_yaw_deg=math.degrees(th), est_yaw_deg=math.degrees(d), err_deg=math.degrees(d - th),
                        t_cm=float(np.linalg.norm(t) * 100), surfels=ns))
    for a in range(n):                                            # sanity: K . A keeps exactly the intended part of the error
        cen = np.asarray(positions[a], np.float64)[:2]
        assert abs(R.yaw(Ks['oracle_yaw'][a] @ MAs[a])) < 1e-12
        assert np.allclose(R.apply2(Ks['oracle_trans'][a] @ MAs[a], cen), cen, atol=1e-12)
    arms = {}
    arms['noreg'] = R.eval_arm(L_pub, scans_pub, [I3] * n, F, R8T, T8, shape, cell, ds, prm, rstats)
    for k, K in Ks.items():
        Lk = [R._integrate(shape, cell, ds, scans_pub[a], K[a]) for a in range(n)]
        arms[k] = R.eval_arm(Lk, scans_pub, K, F, R8T, T8, shape, cell, ds, prm, rstats)
        del Lk
    arms['truepose'] = R.eval_arm(L_k0, scans_k0, [I3] * n, F, R8T, T8, shape, cell, ds, prm, rstats)
    out = dict(panel=panel, house=f'r{rooms}_i{index}', rooms=rooms, index=index, seed=seed, agents=n, est=est,
               arms={k: {row: v[row] for row in ('asis', 'aligned')} for k, v in arms.items()}, seconds=time.time() - t0)
    with open(os.path.join(out_dir, f'{panel}_r{rooms}_i{index}_s{seed}.json'), 'w') as fh:
        json.dump(out, fh)
    return out


def _safe(job):
    try:
        r = run_job(job)
        return r['house'], r.get('seconds'), r.get('skipped')
    except Exception:  # noqa: BLE001
        return job[1:4], None, traceback.format_exc()[-1500:]


def aggregate(out_dir):
    J = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(out_dir, '*_s*.json')))]
    J = [j for j in J if 'arms' in j]
    res = {}
    for panel in sorted({j['panel'] for j in J}):
        for seed in sorted({j['seed'] for j in J}):
            S = [j for j in J if j['panel'] == panel and j['seed'] == seed]
            for row in ('asis', 'aligned'):
                parts = [(j['house'], {a: R._house_sum(j['arms'][a][row]) for a in ARMS}) for j in S]
                means = {a: sum(p[1][a][0] for p in parts) / max(1, sum(p[1][a][1] for p in parts)) for a in ARMS}
                rec = {a: R._pooled_recovery(parts, a) for a in ARMS if a not in ('noreg', 'truepose')}
                res[f'{panel}|s{seed}|{row}'] = dict(houses=len(S), furn_f1=means, recovery=rec)
            errs = [e['err_deg'] for j in S for e in j['est']]; ty = [e['true_yaw_deg'] for j in S for e in j['est']]
            res[f'{panel}|s{seed}|manhattan_yaw'] = dict(agents=len(errs), true_rms_deg=float(np.sqrt(np.mean(np.square(ty)))),
                                                         err_rms_deg=float(np.sqrt(np.mean(np.square(errs)))),
                                                         err_abs_p90_deg=float(np.percentile(np.abs(errs), 90)),
                                                         err_abs_max_deg=float(np.max(np.abs(errs))))
    json.dump(res, open(os.path.join(out_dir, 'summary.json'), 'w'), indent=1)
    for k, v in res.items():
        print(k, json.dumps(v))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True); ap.add_argument('--procs', type=int, default=12)
    ap.add_argument('--panels', default='val'); ap.add_argument('--seeds', default='1,2'); ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--aggregate-only', action='store_true')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if not a.aggregate_only:
        G.setup_env()
        jobs = [(p, r, i, int(s), a.out) for p in a.panels.split(',') for s in a.seeds.split(',') for (r, i) in G.panel_specs(p)][:a.limit]
        t0 = time.time()
        with multiprocessing.get_context('spawn').Pool(a.procs, initializer=G.setup_env) as pool:
            for k, (h, sec, err) in enumerate(pool.imap_unordered(_safe, jobs)):
                print(f'[{k + 1}/{len(jobs)}] {h} {sec} {err or ""}', flush=True)
        print('total', time.time() - t0)
    aggregate(a.out)
