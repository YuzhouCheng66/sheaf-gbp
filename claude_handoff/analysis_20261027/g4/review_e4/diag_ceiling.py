"""Reviewer diagnostic (NOT part of the batch): the pairwise-tree observability ceiling on one TRAIN house, cam tier.
ceiling_pairwise = every tree edge whose two agents share enough wall evidence AT THE TRUE RELATIVE POSE (the reviewed gate's overlap
tests on the K0 maps) gets its TRUE relative pose, every other tree edge keeps the identity (no estimator can do better than the prior
there without extra priors); compose (prior gauge), re-integrate, sum, score asis + aligned exactly as eval_arm does.
Recovery is computed against the noreg / truepose rows of diag_decompose.py (same house, same seed, deterministic).
Usage: python -B diag_ceiling.py ROOMS INDEX SEED DECOMPOSE_JSON OUT_JSON
"""
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'scripts', 'claude_v342')))

import json  # noqa: E402
import numpy as np  # noqa: E402
import g4_registration as R  # noqa: E402

G = R.G; W = G.W; I3 = R.I3


def main():
    rooms, index, seed, dec_path, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
    prm = dict(R.MATCH); dec = json.load(open(dec_path))
    fx = W.make_fixture('TRAIN', index, rooms)
    occ = fx['occ']; cell = np.asarray(fx['cell'], np.float64); private = fx['private']; traj = fx['trajectory_m']
    positions = fx['positions_m']; n = private.shape[0]; shape = occ.shape; ds = 0.5 * float(cell.min())
    H = G.House(occ, fx['slab'], fx['obj'], cell); F = G.forest(fx['edges'], n)
    reached8 = np.asarray(fx['reached'][G.HOPS.index(8)], bool); T8 = G.stream_targets(private, reached8)
    R8T = [G.Target(T8[cm[0]], H, mult=(private[cm] >= 0).sum(0)) for cm in F['comps']]
    rays = W.fibonacci_sphere(W.BEAMS); c = G.CONDITIONS['cam']; rstats = G.RelayStats()
    scans_k0, scans_pub, MAs = [], [], []
    for a in range(n):
        z = G.frame_draws(seed, rooms, index, a)
        frame = (G.rotz(float(z[2] * c['inter_yaw'])), np.array([z[0] * c['inter_t'], z[1] * c['inter_t'], 0.]), np.asarray(positions[a], np.float64))
        MAs.append(R.frame_matrix(frame))
        scans = [G.measure(c, *G.trace_pose(occ, cell, traj[j, a], rays), traj[j, a], G.pose_draws(seed, rooms, index, a, j, len(rays)), ds, rays)[0]
                 for j in range(traj.shape[0])]
        scans_k0.append(scans); scans_pub.append([G.to_public(s, frame) for s in scans])
    L_k0 = [R._integrate(shape, cell, ds, scans_k0[a], I3) for a in range(n)]
    ev = [R.evidence(R.planes_logodds(L, cell, prm), cell, prm) for L in L_k0]
    parent, order = R.tree_order(F)
    keys = sorted((parent[a], a) for a in order if parent[a] is not None)
    deltas, obs = {}, {}
    for (i, j) in keys:
        Si, Sj = ev[i]['S'], ev[j]['S']
        _, info = R.edge_cost(Sj, Si, I3, prm, full=True)
        ok, st = R.gate(Sj, Si, I3, info, R.overlap_pivot(Sj, Si, prm), prm)
        obs[f'{i}-{j}'] = dict(ok=ok, n_inl=st['n_inl'], wfrac=round(st['inl_wfrac'], 3), min_eig=round(st['min_eig'], 3))
        deltas[(i, j)] = MAs[i] @ R.inv(MAs[j]) if ok else I3
    piv = []
    L_pub_ev = [R.evidence(R.planes_logodds(R._integrate(shape, cell, ds, scans_pub[a], I3), cell, prm), cell, prm) for a in range(n)]
    for cm in F['comps']:
        P = np.concatenate([L_pub_ev[a]['S']['m'] for a in cm]); piv.append(P.mean(0))
    _, K = R.compose(F, deltas, 'prior', piv)
    L = [R._integrate(shape, cell, ds, scans_pub[a], K[a]) for a in range(n)]
    e = R.eval_arm(L, scans_pub, K, F, R8T, T8, shape, cell, ds, prm, rstats)
    row = {k: G.summarise_rows(G._expand(e[k]))['furn_f1'] for k in ('asis', 'aligned')}
    A = dec['arms']; res = dict(house=f'TRAIN_r{rooms}_i{index}', seed=seed, observable_tree_edges=obs, ceiling_pairwise=row, recovery={})
    for k in ('asis', 'aligned'):
        Fn, Fu = A['noreg'][k]['furn_f1'], A['truepose'][k]['furn_f1']
        res['recovery'][k] = dict(ceiling_pairwise=(row[k] - Fn) / (Fu - Fn), reg=(A['reg'][k]['furn_f1'] - Fn) / (Fu - Fn))
    print(json.dumps(res, indent=1), flush=True)
    with open(out, 'w') as fh:
        json.dump(res, fh, indent=1)


if __name__ == '__main__':
    main()
