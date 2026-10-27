"""Reviewer diagnostic (NOT part of the batch): co-visibility of ALL agent pairs vs the radio-tree edges, TRAIN house only.
Overlap of a pair = the reviewed matcher's gate statistics at the TRUE relative pose (K0 maps, cam_nf noise, no inter-agent error), i.e. how
much shared wall evidence two agents have at best. A pair is 'co-visible' iff it would pass the overlap part of the gate (>= gate_min_inl
inliers each way, inlier weight >= gate_min_wfrac, min_eig >= gate_min_eig). Question: are the endpoints of the tree edges without overlap
connected through co-visible pairs (then evidence relayed over the tree could register them) or not (then no estimator can)?
Usage: python -B diag_covis.py ROOMS INDEX SEED OUT_JSON
"""
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'scripts', 'claude_v342')))

import itertools, json  # noqa: E401,E402
import numpy as np  # noqa: E402
import g4_registration as R  # noqa: E402

G = R.G; W = G.W


def main():
    rooms, index, seed, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    prm = dict(R.MATCH)
    fx = W.make_fixture('TRAIN', index, rooms)
    occ = fx['occ']; cell = np.asarray(fx['cell'], np.float64); traj = fx['trajectory_m']; n = fx['private'].shape[0]; ds = 0.5 * float(cell.min())
    F = G.forest(fx['edges'], n); rays = W.fibonacci_sphere(W.BEAMS); c = G.CONDITIONS['cam_nf']
    maps = []
    for a in range(n):
        scans = [G.measure(c, *G.trace_pose(occ, cell, traj[j, a], rays), traj[j, a], G.pose_draws(seed, rooms, index, a, j, len(rays)), ds, rays)[0]
                 for j in range(traj.shape[0])]
        maps.append(G.integrate_agent(occ.shape, cell, ds, scans, ('V3',))['V3'])
    ev = [R.evidence(R.planes_logodds(L, cell, prm), cell, prm) for L in maps]
    pairs = {}
    for i, j in itertools.combinations(range(n), 2):
        Si, Sj = ev[i]['S'], ev[j]['S']
        if len(Si['m']) < 3 or len(Sj['m']) < 3:
            pairs[(i, j)] = dict(ok=False); continue
        _, info = R.edge_cost(Sj, Si, R.I3, prm, full=True)
        piv = R.overlap_pivot(Sj, Si, prm)
        ok, st = R.gate(Sj, Si, R.I3, info, piv, prm)
        pairs[(i, j)] = dict(ok=ok, n_inl=st['n_inl'], wfrac=round(st['inl_wfrac'], 3), min_eig=round(st['min_eig'], 3),
                             dist_m=float(np.linalg.norm(fx['positions_m'][i, :2] - fx['positions_m'][j, :2])))
    # connectivity of the co-visibility graph
    par = list(range(n))
    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    for (i, j), p in pairs.items():
        if p['ok']:
            par[find(i)] = find(j)
    covis_comps = sorted({tuple(sorted(a for a in range(n) if find(a) == find(b))) for b in range(n)})
    tree = [dict(edge=[i, j], **pairs[(i, j)], endpoints_covis_connected=find(i) == find(j)) for i, j in F['und']]
    # best case of registering against relayed evidence: agent a vs the union (max of the w / f planes) of every OTHER agent of its
    # component, all at the true poses (an upper bound on what a message-based registration could see)
    planes = [R.planes_logodds(L, cell, prm) for L in maps]
    vs_rest = {}
    for cm in F['comps']:
        for a in cm:
            others = [b for b in cm if b != a]
            if not others:
                continue
            wu = np.max([planes[b][0] for b in others], 0); fu = np.max([planes[b][1] for b in others], 0)
            eu = R.evidence((wu, fu), cell, prm); Sa = ev[a]['S']
            _, info = R.edge_cost(Sa, eu['S'], R.I3, prm, full=True)
            ok, st = R.gate(Sa, eu['S'], R.I3, info, R.overlap_pivot(Sa, eu['S'], prm), prm)
            vs_rest[a] = dict(ok=ok, n_inl_fwd=st['n_inl_fwd'], wfrac_fwd=round(st['inl_wfrac_fwd'], 3), min_eig=round(st['min_eig'], 3))
            print('agent', a, 'vs union of the rest: gate ok', ok, vs_rest[a], flush=True)
    # pooled evidence: co-visibility component Q1 (union of its agents) vs component Q2 (union), at the true poses; islands = connected
    # components of that graph = the groups whose relative pose is observable even when every agent's evidence can be pooled
    def union_ev(ag):
        return R.evidence((np.max([planes[b][0] for b in ag], 0), np.max([planes[b][1] for b in ag], 0)), cell, prm)
    Q = [list(q) for q in covis_comps]; qpairs = {}
    while True:                                                        # merge pooled groups until no pair of groups passes the gate
        qev = [union_ev(q) for q in Q]; merged = False
        for x, y in itertools.combinations(range(len(Q)), 2):
            Sx, Sy = qev[x]['S'], qev[y]['S']
            if len(Sx['m']) < 3 or len(Sy['m']) < 3:
                continue
            _, info = R.edge_cost(Sy, Sx, R.I3, prm, full=True)
            ok, st = R.gate(Sy, Sx, R.I3, info, R.overlap_pivot(Sy, Sx, prm), prm)
            qpairs[f'{Q[x]}-{Q[y]}'] = dict(ok=ok, n_inl=st['n_inl'], wfrac=round(st['inl_wfrac'], 3), min_eig=round(st['min_eig'], 3))
            if ok:
                Q = [q for k, q in enumerate(Q) if k not in (x, y)] + [sorted(Q[x] + Q[y])]; merged = True
                break
        if not merged:
            break
    islands = sorted(tuple(q) for q in Q)
    print('component-vs-component (pooled) gate:', qpairs, flush=True)
    print('observability islands with pooled evidence:', islands, flush=True)
    res = dict(house=f'TRAIN_r{rooms}_i{index}', seed=seed, tree=tree, covis_components=covis_comps, vs_rest=vs_rest, pooled_pairs=qpairs,
               pooled_islands=islands,
               n_covis_pairs=sum(p['ok'] for p in pairs.values()), n_pairs=len(pairs),
               pairs={f'{i}-{j}': p for (i, j), p in pairs.items()})
    for t in tree:
        print('tree edge', t['edge'], 'overlap ok', t['ok'], 'n_inl', t.get('n_inl'), 'wfrac', t.get('wfrac'), 'dist %.1f' % t.get('dist_m', -1),
              'endpoints co-visibility-connected', t['endpoints_covis_connected'], flush=True)
    print('co-visibility components', covis_comps, 'co-visible pairs', res['n_covis_pairs'], '/', res['n_pairs'], flush=True)
    with open(out, 'w') as fh:
        json.dump(res, fh, indent=1)


if __name__ == '__main__':
    main()
