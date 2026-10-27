"""Reviewer diagnostic for G4-E4 (NOT part of the batch; does not modify g4_registration.py or g4_noise_panels.py).
One TRAIN house, one noise seed, cam tier. Imports the reviewed code and recomputes the three arms exactly as run_job does, then adds
report-only diagnostic rows:
  aligned_pose : per component the pose-space oracle alignment G = inv(mean_a(K_a . A_a)) about the component surfel centroid
                 (A_a = true -> believed inter-agent frame; truepose arm: G = I exactly), re-integrate under G . K_a, sum, score;
  oracle_refused: the reg arm with ONLY the gate-refused edges (estimate = identity) replaced by the true relative pose;
  oracle_all    : every tree edge at its true relative pose (must reproduce truepose up to the gauge -> pipeline fairness check);
  reg_root      : the reg arm in the 'root' gauge (aligned row should be gauge invariant).
Usage: python -B diag_decompose.py SPLIT ROOMS INDEX SEED OUT_JSON
"""
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
sys.dont_write_bytecode = True
SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'scripts', 'claude_v342'))
sys.path.insert(0, SCRIPTS)

import json, math, time  # noqa: E401,E402
import numpy as np  # noqa: E402
import g4_registration as R  # noqa: E402

G = R.G; W = G.W; I3 = R.I3


def mean_rows(rows):
    s = G.summarise_rows(G._expand(rows))
    return {k: s.get(k) for k in ('furn_f1', 'free_recall', 'unknown_known_fpr', 'surface_f1')}


def main():
    split, rooms, index, seed, out_path = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    assert split == 'TRAIN', 'reviewer diagnostics run on TRAIN only'
    t0 = time.time(); prm = dict(R.MATCH)
    fx = W.make_fixture(split, index, rooms)
    occ = fx['occ']; cell = np.asarray(fx['cell'], np.float64); private = fx['private']; traj = fx['trajectory_m']
    positions = fx['positions_m']; n = private.shape[0]; shape = occ.shape; ds = 0.5 * float(cell.min())
    H = G.House(occ, fx['slab'], fx['obj'], cell); F = G.forest(fx['edges'], n)
    reached8 = np.asarray(fx['reached'][G.HOPS.index(8)], bool)
    T8 = G.stream_targets(private, reached8)
    R8T = [G.Target(T8[cm[0]], H, mult=(private[cm] >= 0).sum(0)) for cm in F['comps']]
    rays = W.fibonacci_sphere(W.BEAMS)
    traces = [[G.trace_pose(occ, cell, traj[j, a], rays) for j in range(traj.shape[0])] for a in range(n)]
    c = G.CONDITIONS['cam']; rstats = G.RelayStats()
    scans_k0, scans_pub, MAs = [], [], []
    for a in range(n):
        z = G.frame_draws(seed, rooms, index, a)
        frame = (G.rotz(float(z[2] * c['inter_yaw'])), np.array([z[0] * c['inter_t'], z[1] * c['inter_t'], 0.]), np.asarray(positions[a], np.float64))
        MAs.append(R.frame_matrix(frame))
        scans = [G.measure(c, *traces[a][j], traj[j, a], G.pose_draws(seed, rooms, index, a, j, len(rays)), ds, rays)[0] for j in range(traj.shape[0])]
        scans_k0.append(scans); scans_pub.append([G.to_public(s, frame) for s in scans])
    integ = lambda scans, M: R._integrate(shape, cell, ds, scans, M)  # noqa: E731
    L_k0 = [integ(scans_k0[a], I3) for a in range(n)]
    L_pub = [integ(scans_pub[a], I3) for a in range(n)]
    reg = R.register_forest(L_pub, F, cell, prm, 'prior')
    parent, order = R.tree_order(F)
    true_d = {k: MAs[k[0]] @ R.inv(MAs[k[1]]) for k in reg['keys']}
    refused = [k for k in reg['keys'] if np.array_equal(reg['edges'][k]['M'], I3)]
    edges = []
    for k in reg['keys']:
        e = reg['edges'][k]; P = reg['surf_m'][k[1]]; cp = np.asarray(e['pivot'])
        edges.append(dict(edge=list(k), refused=k in refused, err_before=R.pose_err(I3, true_d[k], P, cp), err_after=R.pose_err(e['M'], true_d[k], P, cp),
                          identity_gate=e['identity_gate'], coarse_accepted=e['coarse_accepted'], icp_stop=e['icp_stop']))

    def pose_aligned(maps0, base, Ks, MAs_arm):
        Gs, Gmap = [], {}
        for ci, cm in enumerate(F['comps']):
            Gc = R.inv(R.mean_transform([Ks[a] @ MAs_arm[a] for a in cm], reg['pivots'][ci]))
            if np.allclose(Gc, I3, atol=1e-14):
                Gc = I3
            Gs.append(R.params(Gc))
            for a in cm:
                Gmap[a] = Gc
        al = [maps0[a] if np.array_equal(Gmap[a], I3) else integ(base[a], Gmap[a] @ Ks[a]) for a in range(n)]
        _, pr = G.fuse(al, F, 'sum', rstats)
        return R._rows(pr, R8T, F['comps']), Gs

    res = dict(house=f'{split}_r{rooms}_i{index}', seed=seed, agents=n, cell=cell.tolist(), components=F['comps'], tree_edges=F['und'],
               refused=[list(k) for k in refused], edges=edges, arms={})

    def run_arm(name, maps0, base, Ks, MAs_arm):
        t = time.time()
        ev = R.eval_arm(maps0, base, Ks, F, R8T, T8, shape, cell, ds, prm, rstats)
        pa, Gs = pose_aligned(maps0, base, Ks, MAs_arm)
        res['arms'][name] = dict(asis=mean_rows(ev['asis']), aligned=mean_rows(ev['aligned']), aligned_pose=mean_rows(pa),
                                 G_matcher=[a['G'] for a in ev['align']], G_pose=Gs, align_stop=[a['icp_stop'] for a in ev['align']],
                                 align_steps=[a['steps'] for a in ev['align']], seconds=time.time() - t)
        print(name, {k: round(v['furn_f1'], 4) for k, v in res['arms'][name].items() if isinstance(v, dict) and 'furn_f1' in v},
              'G_matcher', [[round(x, 3) for x in g] for g in res['arms'][name]['G_matcher']], 'G_pose', [[round(x, 3) for x in g] for g in Gs], flush=True)

    Iall = [I3] * n
    run_arm('noreg', L_pub, scans_pub, Iall, MAs)
    run_arm('truepose', L_k0, scans_k0, Iall, Iall)
    L_reg = [L_pub[a] if np.array_equal(reg['K'][a], I3) else integ(scans_pub[a], reg['K'][a]) for a in range(n)]
    run_arm('reg', L_reg, scans_pub, reg['K'], MAs)
    del L_reg
    L_root = [L_pub[a] if np.array_equal(reg['C'][a], I3) else integ(scans_pub[a], reg['C'][a]) for a in range(n)]
    run_arm('reg_root', L_root, scans_pub, reg['C'], MAs)
    del L_root
    if refused:
        d = {k: (true_d[k] if k in refused else reg['edges'][k]['M']) for k in reg['keys']}
        _, K = R.compose(F, d, 'prior', reg['pivots'])
        L = [L_pub[a] if np.array_equal(K[a], I3) else integ(scans_pub[a], K[a]) for a in range(n)]
        run_arm('oracle_refused', L, scans_pub, K, MAs)
        del L
    _, K = R.compose(F, dict(true_d), 'prior', reg['pivots'])
    L = [integ(scans_pub[a], K[a]) for a in range(n)]
    run_arm('oracle_all', L, scans_pub, K, MAs)
    del L
    A = res['arms']
    for row in ('asis', 'aligned', 'aligned_pose'):
        Fn, Fu = A['noreg'][row]['furn_f1'], A['truepose'][row]['furn_f1']
        res[f'recovery_{row}'] = {k: ((A[k][row]['furn_f1'] - Fn) / (Fu - Fn) if Fu - Fn > 0 else None) for k in A if k not in ('noreg', 'truepose')}
    res['seconds'] = time.time() - t0; res['peak_mb'] = G._peak_mb()
    print(json.dumps({k: res[k] for k in res if k.startswith('recovery')}, indent=1), flush=True)
    print('refused edges', res['refused'], 'seconds', round(res['seconds']), 'peak MB', res['peak_mb'], flush=True)
    with open(out_path, 'w') as fh:
        json.dump(res, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))


if __name__ == '__main__':
    main()
