"""Unit tests for g4_registration (Claude, 2026-10-27, batch G4-E4). CPU, about a minute:  python -B test_g4_registration.py
Real-fixture checks (no-registration arm = G4-E1 maps and rows, cam_nf corrections ~ 0, cam per-edge error falls, E_pose assertion
counted) are part of the smoke run (--smoke-asserts --check-e1-maps --e1-jobs ...), not of this file."""
import inspect, json, math, os, sys, tempfile

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g4_registration as R  # noqa: E402

G = R.G
PRM = dict(R.MATCH)


# ---------------------------------------------------------------- SE(2) algebra and the bridge to G4-E1's frames
def test_se2_algebra_and_frames():
    rng = np.random.default_rng(0)
    for _ in range(20):
        th, c, t = rng.normal(0, .05), rng.uniform(0, 10, 2), rng.normal(0, .1, 2)
        M = R.about(th, c, t); P = rng.uniform(0, 10, (50, 2))
        assert np.allclose(R.apply2(M, P), (P - c) @ R.se2(th)[:2, :2].T + c + t)
        assert np.allclose(R.inv(M) @ M, np.eye(3)) and abs(R.yaw(M) - th) < 1e-12
        assert np.allclose(R.apply2(M, c), c + t)
        # G4-E1's inter-agent frame and frame_matrix move a scan identically (z untouched, directions rotated)
        cen = np.r_[c, 1.3]; deg = math.degrees(th)
        frame = (G.rotz(deg), np.r_[t, 0.], cen)
        scan = (np.r_[rng.uniform(0, 10, 2), 1.3], rng.normal(size=(30, 3)), rng.uniform(.1, 5, 30), np.arange(30, dtype=np.int32))
        a = G.to_public(scan, frame); b = R.transform_scan(scan, R.frame_matrix(frame))
        assert np.allclose(a[0], b[0], atol=1e-12) and np.allclose(a[1], b[1], atol=1e-12) and a[2] is scan[2] and b[3] is scan[3]
    scan = (np.zeros(3), np.ones((2, 3)), np.ones(2), np.zeros(2, np.int32))
    assert R.transform_scan(scan, R.I3) is scan                        # identity correction: the scan itself (no-registration arm exact)
    assert np.array_equal(R.frame_matrix(None), np.eye(3))


def test_pose_err_and_mean_transform():
    c = np.array([2., 3.]); P = np.array([[1., 1.], [4., 2.], [2., 5.]])
    Mt = R.about(math.radians(1.), c, (.05, -.02))
    e = R.pose_err(Mt, Mt, P, c)
    assert abs(e['yaw_deg']) < 1e-12 and e['disp_pivot_cm'] < 1e-9 and e['disp_rms_cm'] < 1e-9
    e = R.pose_err(np.eye(3), Mt, P, c)
    assert abs(e['yaw_deg'] + 1.) < 1e-9 and abs(e['disp_pivot_cm'] - math.hypot(5, 2)) < 1e-9 and e['disp_max_cm'] >= e['disp_rms_cm']
    Ms = [R.about(.01, c, (.1, 0.)), R.about(-.03, c, (0., .2))]
    H = R.mean_transform(Ms, c)
    assert abs(R.yaw(H) + .01) < 1e-12 and np.allclose(R.apply2(H, c) - c, [.05, .1])


# ---------------------------------------------------------------- the pre-registered monotone assertion
def test_assert_monotone_fires():
    assert R.assert_monotone([3., 2., 2., 1.5]) == 3
    for bad in ([1., 1.0000001], [3., 2., 2.5]):
        try:
            R.assert_monotone(bad)
            raise AssertionError('increase not detected')
        except RuntimeError as e:                                       # explicit raise (review S4): survives python -O
            assert 'increased' in str(e)


# ---------------------------------------------------------------- tree composition = exact tree solution (+ gauge)
def _random_tree(n, rng):
    edges = [(int(rng.integers(0, a)), a, 1.) for a in range(1, n)]
    return G.forest(edges, n)


def test_compose_is_exact_on_trees():
    rng = np.random.default_rng(1)
    for trial in range(10):
        n = 8; F = _random_tree(n, rng) if trial % 2 else G.forest([(0, 1, 1.), (1, 2, 1.), (4, 5, 1.), (5, 6, 1.), (5, 7, 1.)], n)
        A = [R.about(rng.normal(0, .03), rng.uniform(0, 12, 2), rng.normal(0, .08, 2)) for _ in range(n)]   # true -> believed
        parent, order = R.tree_order(F)
        deltas = {(parent[a], a): A[parent[a]] @ R.inv(A[a]) for a in order if parent[a] is not None}       # exact relative poses
        piv = [rng.uniform(0, 12, 2) for _ in F['comps']]
        for gauge in ('root', 'prior'):
            C, K = R.compose(F, deltas, gauge, piv)
            for ci, cm in enumerate(F['comps']):
                Rs = [K[a] @ A[a] for a in cm]                         # true -> corrected: one common transform per component
                assert all(np.allclose(Rs[0], x, atol=1e-12) for x in Rs), (gauge, cm)
                if gauge == 'root':
                    assert np.allclose(Rs[0], A[cm[0]], atol=1e-12)
                else:                                                   # the prior gauge sits at the mean of the believed frames
                    H = R.mean_transform([C[a] for a in cm], piv[ci])
                    assert np.allclose(K[cm[0]], R.inv(H) @ C[cm[0]])
                    Mk = R.mean_transform([K[a] for a in cm], piv[ci])  # the corrections average to zero (unit prior on every frame)
                    assert abs(R.yaw(Mk)) < 1e-12 and np.allclose(R.apply2(Mk, piv[ci]), piv[ci], atol=1e-12)


def test_estimator_sees_no_truth():
    assert list(inspect.signature(R.register_forest).parameters) == ['maps', 'F', 'cell', 'prm', 'gauge']
    import ast
    used = set()
    for f in (R.register_forest, R.register_pair, R.coarse_search, R.icp_refine, R.edge_cost, R._match, R.gn_system, R.gate,
              R.overlap_pivot, R.surfels, R.evidence, R.planes_logodds, R.band_mask, R.compose, R.tree_order, R.mean_transform):
        for node in ast.walk(ast.parse(inspect.getsource(f))):          # identifiers only (docstrings may say "no truth")
            if isinstance(node, ast.Name):
                used.add(node.id)
            elif isinstance(node, ast.Attribute):
                used.add(node.attr)
    for word in ('frame_draws', 'make_fixture', 'stream_targets', 'Target', 'private', 'positions', 'positions_m', 'T8', 'R8T', 'occ',
                 'frame', 'frames', 'MAs', 'frame_matrix', 'planes_tern', 'to_public', 'trajectory_m'):
        assert word not in used, word


# ---------------------------------------------------------------- matcher on synthetic planes
def _plane(shape, cell, segs, M=np.eye(3), ss=4):
    """soft wall plane: supersampled rasterisation of axis-aligned wall rectangles (metres) seen through the SE(2) M (true -> map)"""
    X, Y = shape; cx = cell[0]; Minv = R.inv(M)
    u = (np.arange(X * ss) + .5) * cx / ss; v = (np.arange(Y * ss) + .5) * cx / ss
    Ug, Vg = np.meshgrid(u, v, indexing='ij'); P = R.apply2(Minv, np.stack([Ug.ravel(), Vg.ravel()], 1))
    hit = np.zeros(len(P), bool)
    for x0, x1, y0, y1 in segs:
        hit |= (P[:, 0] >= x0) & (P[:, 0] < x1) & (P[:, 1] >= y0) & (P[:, 1] < y1)
    return hit.reshape(X * ss, Y * ss).reshape(X, ss, Y, ss).mean((1, 3))


SEGS = [(.3, .45, .3, 5.7), (5.55, 5.7, .3, 5.7), (.3, 5.7, .3, .45), (.3, 5.7, 5.55, 5.7), (2.9, 3.05, .45, 3.2),
        (1.0, 1.6, 4.0, 4.9), (4.0, 4.9, 1.2, 1.5), (3.9, 4.1, 3.9, 4.1)]


def _ev(w, cell):
    f = np.where(w < .05, .5, 0.)                                       # a free plane everywhere off the walls
    return R.evidence((w, f), cell, PRM)


def test_register_pair_on_planes():
    cell = (0.1, 0.1, 0.1); shape = (60, 60); rng = np.random.default_rng(2)
    for trial in range(4):
        Ai = R.about(rng.normal(0, math.radians(1.)), (2.5, 3.), rng.normal(0, .05, 2))
        Aj = R.about(rng.normal(0, math.radians(1.)), (3.5, 2.5), rng.normal(0, .05, 2))
        ev_i, ev_j = _ev(_plane(shape, cell, SEGS, Ai), cell), _ev(_plane(shape, cell, SEGS, Aj), cell)
        res = R.register_pair(ev_i, ev_j, cell, PRM)
        Dt = Ai @ R.inv(Aj); P = ev_j['S']['m']
        before, after = R.pose_err(R.I3, Dt, P, np.array(res['pivot'])), R.pose_err(res['M'], Dt, P, np.array(res['pivot']))
        assert after['disp_rms_cm'] < 2.0 and abs(after['yaw_deg']) < 0.3, (trial, before, after, res['coarse'])   # raster floor ~1-1.6 cm
        assert after['disp_rms_cm'] < before['disp_rms_cm']
        R.assert_monotone(res['traj'])
        assert all(b < a for a, b in zip(res['traj'], res['traj'][1:]))  # every accepted step strictly lowers E_edge
    # no error: the estimate stays at (numerically) zero
    ev = _ev(_plane(shape, cell, SEGS), cell)
    res = R.register_pair(ev, ev, cell, PRM)
    assert np.allclose(res['M'], np.eye(3), atol=1e-9) and res['traj'][0] == res['E']


def test_coarse_search_finds_large_offsets():
    cell = (0.1, 0.1, 0.1); shape = (60, 60)
    Aj = R.about(math.radians(6.), (3., 3.), (.32, -.27))                 # beyond the ICP basin, inside the search window
    wi, wj = _plane(shape, cell, SEGS), _plane(shape, cell, SEGS, Aj)
    ev_i, ev_j = _ev(wi, cell), _ev(wj, cell)
    pivot = R.overlap_pivot(ev_j['S'], ev_i['S'], PRM)
    Mc, info = R.coarse_search(wi, wj, cell, pivot, PRM)
    e = R.pose_err(Mc, R.inv(Aj), ev_j['S']['m'], pivot)
    assert abs(e['yaw_deg']) <= .5 and e['disp_rms_cm'] < 6., (e, info)
    res = R.register_pair(ev_i, ev_j, cell, PRM)
    e = R.pose_err(res['M'], R.inv(Aj), ev_j['S']['m'], pivot)
    assert res['coarse_accepted'] and e['disp_rms_cm'] < 1.5 and abs(e['yaw_deg']) < .25, e   # 6 deg raster staircase: ~1 cm floor


def test_icp_steps_only_lower_the_energy():
    rng = np.random.default_rng(3); cell = (0.1, 0.1, 0.1); shape = (60, 60)
    ev_i = _ev(_plane(shape, cell, SEGS), cell)
    Aj = R.about(math.radians(1.5), (3., 3.), (.06, .04)); ev_j = _ev(_plane(shape, cell, SEGS, Aj), cell)
    for _ in range(5):
        M0 = R.about(rng.normal(0, .02), (3., 3.), rng.normal(0, .05, 2))
        E0 = R.edge_cost(ev_j['S'], ev_i['S'], M0, PRM); traj = [E0]
        M, E, it, stop = R.icp_refine(ev_j['S'], ev_i['S'], M0, E0, np.array([3., 3.]), PRM, traj)
        assert E <= E0 and len(traj) == it + 1 and all(b < a for a, b in zip(traj, traj[1:])), (traj, stop)
        assert R.edge_cost(ev_j['S'], ev_i['S'], M, PRM) == E


def test_gate_and_trust_region():
    cell = (0.1, 0.1, 0.1); shape = (60, 60)
    ev_a = _ev(_plane(shape, cell, [(.3, .45, .3, 2.5), (.3, 2.5, .3, .45)]), cell)          # two disjoint corners: no overlap
    ev_b = _ev(_plane(shape, cell, [(5.55, 5.7, 3.5, 5.7), (3.5, 5.7, 5.55, 5.7)]), cell)
    res = R.register_pair(ev_a, ev_b, cell, PRM)
    assert np.array_equal(res['M'], np.eye(3)) and res['icp_stop'] == 'gate' and not res['coarse_accepted'] and len(res['traj']) == 1
    ev = _ev(_plane(shape, cell, SEGS), cell); piv = np.array([3., 3.])
    for M, ok in ((np.eye(3), True), (R.about(math.radians(10.5), piv, (0., 0.)), False), (R.about(0., piv, (.61, 0.)), False),
                  (R.about(math.radians(9.), piv, (.5, -.5)), True)):
        assert R.gate(ev['S'], ev['S'], M, R.edge_cost(ev['S'], ev['S'], M, PRM, full=True)[1], piv, PRM)[1]['trust'] == ok
    # a single straight wall is degenerate (translation along it unobservable): the gate refuses it
    ev1 = _ev(_plane(shape, cell, [(.3, .45, .3, 5.7)]), cell)
    g = R.gate(ev1['S'], ev1['S'], np.eye(3), R.edge_cost(ev1['S'], ev1['S'], np.eye(3), PRM, full=True)[1], piv, PRM)
    assert not g[0] and g[1]['min_eig'] < PRM['gate_min_eig']


def test_surfels_are_subcell_and_oriented():
    cell = (0.1, 0.1, 0.1); X = Y = 40
    for x0 in (2.0, 2.025, 2.05, 2.075):                                 # a 1-cell thick wall at a sub-cell offset (exact at ss = 4)
        w = _plane((X, Y), cell, [(x0, x0 + .1, .5, 3.5)])
        f = np.zeros_like(w); f[:int(round(x0 / .1, 6)), :] = 1.         # free up to the wall face, low-x side only
        S = R.surfels(w, f, cell, PRM)
        mid = (S['m'][:, 1] > 1.) & (S['m'][:, 1] < 3.)
        assert abs(np.average(S['m'][mid, 0], weights=S['w'][mid]) - (x0 + .05)) < 2e-3, x0
        assert np.all(S['ori'][mid]) and np.all(S['n'][mid, 0] < -.99)   # normals point to the observed free side


# ---------------------------------------------------------------- the pipeline on a sensor-simulated room (G4-E1 functions)
def _room():
    shape = (64, 64, 30); cell = np.array([.1, .1, .1]); occ = np.zeros(shape, np.uint8)
    occ[:, :, 0] = 1; occ[:, :, 27:] = 1                                  # floor slab (z < .1) and ceiling
    occ[2:4, 2:62, :] = occ[60:62, 2:62, :] = occ[2:62, 2:4, :] = occ[2:62, 60:62, :] = 1
    occ[30:32, 4:36, :] = 1                                               # interior wall with a passage at y > 3.6 m
    occ[10:16, 40:48, 1:8] = 1; occ[44:48, 10:30, 1:20] = 1; occ[45, 45, 1:25] = 1
    return occ, cell


def test_register_forest_on_simulated_scans():
    occ, cell = _room(); rays = G.W.fibonacci_sphere(G.W.BEAMS); ds = 0.5 * float(cell.min())     # the real beam count
    cen = [np.array([1.6, 4.6, 1.3]), np.array([4.4, 4.8, 1.3]), np.array([5.3, 2.0, 1.3])]
    true = [(.04, -.03, .9), (-.05, .02, -1.2), (.02, .05, .6)]           # inter-agent errors (m, m, deg), as G4-E1 draws them
    c = G.CONDITIONS['range_only']                                       # range noise, no per-frame pose error
    maps, maps0, MAs = [], [], []
    for a, p0 in enumerate(cen):
        scans = []
        for j, dxy in enumerate(((0., 0.), (.08, -.06), (-.07, .05))):
            p = p0 + np.r_[dxy, 0.]
            hs, ls = G.trace_pose(occ, cell, p, rays)
            scans.append(G.measure(c, hs, ls, p, G.pose_draws(5, 1, 0, a, j, len(rays)), ds, rays)[0])
        frame = (G.rotz(true[a][2]), np.array([true[a][0], true[a][1], 0.]), p0)
        MAs.append(R.frame_matrix(frame))
        maps.append(G.integrate_agent(occ.shape, cell, ds, [G.to_public(s, frame) for s in scans], ('V3',))['V3'])
        maps0.append(G.integrate_agent(occ.shape, cell, ds, scans, ('V3',))['V3'])
    F = G.forest([(0, 1, 1.), (1, 2, 1.)], 3)
    reg = R.register_forest(maps, F, cell, PRM, 'prior')
    assert reg['monotone_checks'] >= 3 and all(b <= a for a, b in zip(reg['E_pose'], reg['E_pose'][1:]))
    for (i, j) in reg['keys']:
        e = reg['edges'][(i, j)]; Dt = MAs[i] @ R.inv(MAs[j]); P = reg['surf_m'][j]
        before, after = R.pose_err(R.I3, Dt, P, np.array(e['pivot'])), R.pose_err(e['M'], Dt, P, np.array(e['pivot']))
        assert after['disp_rms_cm'] < 0.5 * before['disp_rms_cm'] and after['disp_rms_cm'] < 2.5, (i, j, before, after)
    ae0 = R.agent_errors([R.I3] * 3, MAs, F, reg['pivots']); ae1 = R.agent_errors(reg['K'], MAs, F, reg['pivots'])
    assert ae1['ATE_cm'] < 0.5 * ae0['ATE_cm'] and ae1['ARE_deg'] < 0.5 * ae0['ARE_deg'], (ae0['ATE_cm'], ae1['ATE_cm'], ae0['ARE_deg'], ae1['ARE_deg'])
    # no inter-agent error: the correction is (near) zero
    reg0 = R.register_forest(maps0, F, cell, PRM, 'prior')
    for (i, j) in reg0['keys']:
        e = reg0['edges'][(i, j)]
        assert R.pose_err(e['M'], R.I3, reg0['surf_m'][j], np.array(e['pivot']))['disp_rms_cm'] < 1.5 and abs(math.degrees(R.yaw(e['M']))) < .3


# ---------------------------------------------------------------- verdict and aggregation
def _seed(Fn=.73, Fr=.84, Fu=.85, nf_n=.85, nf_r=.85, free=(.96, .96), fpr=(.02, .02), mono=True, houses=('r1_i0', 'r1_i1'), arms=R.ARMS,
          e1_bad=(), e1=True, ceil=None, best_Fr=None):
    """a per-seed summary as summarise_seed builds it; ceil = (F_oracle_all, F_ceiling_pairwise, F_ceiling_islands) adds the report arms"""
    row = lambda f, fr=.96, fp=.02: dict(furn_f1=f, free_recall=fr, unknown_known_fpr=fp)  # noqa: E731
    rows = R.ROWS + (('aligned_best',) if best_Fr is not None else ())
    cam = dict(arms={'noreg': {r: row(Fn, free[0], fpr[0]) for r in rows},
                     'reg': {r: row(best_Fr if r == 'aligned_best' else Fr, free[1], fpr[1]) for r in rows},
                     'truepose': {r: row(Fu) for r in rows}}, E_pose_monotone_all=mono)
    nf = dict(arms={'noreg': {r: row(nf_n) for r in rows}, 'reg': {r: row(nf_r) for r in rows}, 'truepose': {r: row(nf_n) for r in rows}},
              E_pose_monotone_all=True)
    if ceil is not None:
        for arm, f in zip(R.REPORT_ARMS, ceil):
            cam['arms'][arm] = {r: row(f) for r in rows}
    for a in R.ARMS:
        if a not in arms:
            del cam['arms'][a]
    if e1:
        for d in (cam, nf):
            d['e1_compare'] = dict(counts={'equal': len(houses) - len(e1_bad)} | ({'unequal': len(e1_bad)} if e1_bad else {}), bad=list(e1_bad))
    return dict(houses=list(houses), skipped=[], conds=dict(cam=cam, cam_nf=nf))


EXP = frozenset(['r1_i0', 'r1_i1'])


def _V(seeds, judged=True, extra=()):
    return R.verdict(seeds, EXP, judged, extra)


def test_verdict():
    V = lambda s, judged=True: _V(s, judged)['verdict']  # noqa: E731
    assert V({'1': _seed(), '2': _seed(Fr=.8321)}).startswith('LEVER')                    # (.8321 - .73) / .12 = .8508
    assert V({'1': _seed(), '2': _seed(Fr=.8318)}) == 'AMBIGUOUS'                          # .848 < .85 in one seed
    assert V({'1': _seed(Fr=.78), '2': _seed(Fr=.75)}).startswith('INSUFFICIENT')          # .42, .17
    assert V({'1': _seed(Fr=.78), '2': _seed(Fr=.80)}) == 'AMBIGUOUS'                      # .42, .58
    assert V({'1': _seed(Fr=.84), '2': _seed(Fr=.75)}) == 'AMBIGUOUS'                      # one seed LEVER, the other INSUFFICIENT
    assert V({'1': _seed(nf_r=.8449), '2': _seed()}) == 'AMBIGUOUS'                        # cam_nf worse by .51 pp
    assert V({'1': _seed(nf_r=.8451), '2': _seed()}).startswith('LEVER')                   # .49 pp is allowed
    assert V({'1': _seed(free=(.96, .9599)), '2': _seed()}) == 'AMBIGUOUS'                 # free recall may not drop at all
    assert V({'1': _seed(fpr=(.02, .0231)), '2': _seed()}) == 'AMBIGUOUS'                  # FPR worse by .31 pp
    assert V({'1': _seed(fpr=(.02, .0229)), '2': _seed()}).startswith('LEVER')
    assert V({'1': _seed(mono=False), '2': _seed()}) == 'AMBIGUOUS'
    assert V({'1': _seed(Fu=.73), '2': _seed()}) == 'AMBIGUOUS'                            # one seed: no gap, recovery undefined
    assert V({'1': _seed(Fu=.73, Fr=.70), '2': _seed(Fr=.75)}) == 'AMBIGUOUS'              # undefined + INSUFFICIENT: not INSUFFICIENT
    assert V({'1': _seed()}).startswith('INCOMPLETE')
    assert V({'1': _seed(), '3': _seed()}).startswith('INCOMPLETE')
    assert V({'1': _seed(), '2': _seed(houses=('r1_i0',))}).startswith('INCOMPLETE')
    assert V({'1': _seed(), '2': _seed(arms=('noreg', 'truepose'))}).startswith('INCOMPLETE')
    s = _seed(); del s['conds']['cam_nf']
    assert V({'1': s, '2': _seed()}).startswith('INCOMPLETE')
    assert V({'1': _seed(), '2': _seed()}, judged=False).startswith('NOT JUDGED')
    v = _V({'1': _seed(), '2': _seed()})
    r = v['per_seed']['1']['readouts']['aligned']
    assert abs(r['recovery'] - (.84 - .73) / (.85 - .73)) < 1e-12 and v['judged_row'] == 'aligned' and v['category'] == 'LEVER'
    # review S1: a failed job is a reason
    v = _V({'1': _seed(), '2': _seed()}, extra=['job val r1_i0 s2 failed: RuntimeError: E_pose increased'])
    assert v['verdict'].startswith('INCOMPLETE') and 'failed' in v['verdict'] and v['category'] is None


def test_verdict_requires_the_g4_e1_control():                          # review M1
    v = _V({'1': _seed(), '2': _seed(e1_bad=('r1_i1',))})
    assert v['verdict'].startswith('INCOMPLETE') and 'control arm != G4-E1' in v['verdict']
    v = _V({'1': _seed(), '2': _seed(e1=False)})
    assert v['verdict'].startswith('INCOMPLETE') and 'no comparison with G4-E1' in v['verdict']
    assert _V({'1': _seed(), '2': _seed(e1=False)}, judged=False)['verdict'].startswith('NOT JUDGED (report panel); readout: LEVER')


def test_verdict_reading_and_ceilings():                                # D8-39 / D8-40 addendum 2, review M3 and S2
    low = dict(Fr=.75)                                                  # recovery .17 -> INSUFFICIENT
    v = _V({'1': _seed(**low, ceil=(.85, .80, .84)), '2': _seed(**low, ceil=(.851, .79, .83))})
    assert v['category'] == 'INSUFFICIENT' and 'not identifiable' in v['reading'] and 'G5-B does NOT' in v['verdict']
    assert 'ceilings (cam, aligned recovery): pairwise s1 0.583' in v['verdict'] and v['oracle_check_ok'] and 'pipeline check failed' not in v['verdict']
    v = _V({'1': _seed(**low, ceil=(.85, .8321, .85)), '2': _seed(**low, ceil=(.85, .84, .85))})            # ceilings .851 / .917
    assert 'classical method insufficient' in v['reading'] and 'becomes a candidate' in v['verdict']
    v = _V({'1': _seed(**low, ceil=(.85, .8321, .85)), '2': _seed(**low, ceil=(.85, .8318, .85))})           # .848 in one seed
    assert 'not identifiable' in v['reading']
    v = _V({'1': _seed(**low), '2': _seed(**low)})
    assert 'undetermined' in v['reading']
    v = _V({'1': _seed(ceil=(.83, .8, .82)), '2': _seed(ceil=(.85, .8, .82))})                               # oracle recovery .83 -> off by .17
    assert v['category'] == 'LEVER' and not v['oracle_check_ok'] and 'alignment pipeline check failed' in v['verdict']
    v = _V({'1': _seed(best_Fr=.80), '2': _seed(best_Fr=.84)})                                               # aligned_best: .58 in seed 1
    assert v['category'] == 'LEVER' and v['category_aligned_best'] == 'AMBIGUOUS' and 'sensitive to alignment' in v['verdict']
    v = _V({'1': _seed(best_Fr=.84), '2': _seed(best_Fr=.84)})
    assert v['category_aligned_best'] == 'LEVER' and 'sensitive' not in v['verdict']


def test_e4b_entry():                                                   # addendum 3
    mk = lambda c1, c2: _V({'1': _seed(Fr=.75, ceil=(.85, .74, c1)), '2': _seed(Fr=.75, ceil=(.85, .74, c2))})  # noqa: E731
    e = R.e4b_entry(mk(.74 + .15 * .12 + 1e-9, .80))                    # islands - pairwise = .15 + eps (recovery units), .5
    assert e['value'] and e['complete'] and abs(e['per_seed']['1'] - .15) < 1e-6
    e = R.e4b_entry(mk(.74 + .14 * .12, .80))
    assert not e['value'] and e['complete']
    e = R.e4b_entry(_V({'1': _seed(Fr=.75, ceil=(.85, .74, .80))}))
    assert not e['value'] and not e['complete']


def _rowsF(*fs):
    return [[dict(furn_f1=f), 4] for f in fs]


def test_aligned_best_and_none_positions():
    best, below = R.aligned_best(_rowsF(.8, .5, None), _rowsF(.7, .6, None))
    assert [r[0]['furn_f1'] for r in best] == [.8, .6, None] and below == 1
    job = dict(panel='val', house='r1_i0', seed=1, conds=dict(cam=dict(arms=dict(
        noreg=dict(asis=_rowsF(.5, None), aligned=_rowsF(.6, None)), reg=dict(asis=_rowsF(.7, None), aligned=_rowsF(.8, None))))))
    R.check_none_positions(job)
    job['conds']['cam']['arms']['reg']['aligned'] = _rowsF(.8, .1)
    try:
        R.check_none_positions(job)
        raise AssertionError('different None positions not detected')
    except RuntimeError as e:
        assert 'undefined on different rows' in str(e)


def test_recovery_distribution():
    def job(h, n, r, t):
        return dict(house=h, conds=dict(cam=dict(arms=dict(noreg=dict(aligned=_rowsF(n)), reg=dict(aligned=_rowsF(r)), truepose=dict(aligned=_rowsF(t))))))
    J = [job('a', .7, .8, .9), job('b', .6, .6, .8), job('c', .5, .75, .75)]
    d = R.recovery_distribution(J, 'cam', 'aligned')
    assert abs(d['pooled'] - ((.8 + .6 + .75) - (.7 + .6 + .5)) / ((.9 + .8 + .75) - (.7 + .6 + .5))) < 1e-12
    assert d['per_house'] == {'a': (.8 - .7) / (.9 - .7), 'b': 0.0, 'c': 1.0} and abs(d['gap'] - (2.45 - 1.8) / 3) < 1e-12
    assert d['boot_ci95'][0] <= d['pooled'] <= d['boot_ci95'][1] and d == R.recovery_distribution(J, 'cam', 'aligned')   # deterministic


def test_e1_compare_at_aggregation():
    with tempfile.TemporaryDirectory() as tmp:
        rows = [[dict(furn_f1=.75, furn_tp=3), 8]]; rows_t = [[dict(furn_f1=.85, furn_tp=4), 8]]
        e1 = dict(fingerprint=dict(script_sha256='E1SHA'), conds=dict(cam=dict(by_int=dict(V3=dict(R8=dict(sum=rows, sum_truepose=rows_t)))),
                                                                          cam_nf=dict(by_int=dict(V3=dict(R8=dict(sum=rows_t))))))
        json.dump(e1, open(os.path.join(tmp, 'val_r1_i0_s1.json'), 'w'))
        job = dict(panel='val', rooms=1, index=0, seed=1, fingerprint=dict(g4_noise_panels_sha256='E1SHA'),
                   conds=dict(cam=dict(arms=dict(noreg=dict(asis=rows), truepose=dict(asis=rows_t))),
                              cam_nf=dict(arms=dict(noreg=dict(asis=rows_t), truepose=dict(asis=rows_t))),
                              harsh=dict(arms=dict(noreg=dict(asis=rows), truepose=dict(asis=rows)))))
        assert R.e1_compare_job(job, tmp, {}) == dict(cam='equal', cam_nf='equal', harsh='tier_missing')
        job['conds']['cam']['arms']['noreg']['asis'] = [[dict(furn_f1=.7500001, furn_tp=3), 8]]
        assert R.e1_compare_job(job, tmp, {})['cam'] == 'unequal'
        assert R.e1_compare_job(dict(job, seed=2), tmp, {})['cam'] == 'missing'
        assert R.e1_compare_job(dict(job, fingerprint=dict(g4_noise_panels_sha256='OTHER')), tmp, {})['cam_nf'] == 'other_e1_version'
        assert R.e1_compare_job(job, None, {})['cam'] == 'no_e1_dir'


def test_failed_job_is_recorded_and_makes_incomplete():                # review S1
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, 'jobs'))
        opts = dict(out=tmp, conds=['cam'], gauge='prior', job_format=R.JOB_FORMAT)
        orig = R.run_job
        try:
            def boom(job):
                raise RuntimeError('E_pose increased at step 3: 1.0 -> 2.0')
            R.run_job = boom
            path, sec, mb, sk, fl = R._run_and_save(('val', 1, 0, 1, opts))
        finally:
            R.run_job = orig
        rec = json.load(open(path))
        assert 'failed' in rec and 'E_pose increased' in rec['failed'] and fl and rec['fingerprint'] == R.fingerprint(opts)
        S = R.aggregate(tmp)
        v = S['panels']['val']['verdict']
        assert v['verdict'].startswith('INCOMPLETE') and any('failed' in x for x in v['incomplete_reasons']) and S['failed'][0]['house'] == 'r1_i0'


def test_run_job_feeds_the_believed_maps_to_the_estimator():           # review S6
    import ast
    calls = [n for n in ast.walk(ast.parse(inspect.getsource(R.run_job))) if isinstance(n, ast.Call) and getattr(n.func, 'id', None) == 'register_forest']
    assert len(calls) == 1 and isinstance(calls[0].args[0], ast.Name) and calls[0].args[0].id == 'L_pub'


def test_islands_merge_rule():
    # pairs 0-1 and 2-3 are co-visible; pooled {0,1} overlaps {4}; nothing joins {2,3} -> islands (0,1,4), (2,3), (5)
    pair = lambda x, y: (x, y) in {(0, 1), (2, 3)}  # noqa: E731
    groups = R.covis_groups([0, 1, 2, 3, 4, 5], pair)
    assert groups == [[0, 1], [2, 3], [4], [5]]
    pooled = lambda gx, gy: set(gx + gy) == {0, 1, 4}  # noqa: E731
    assert R.merge_islands(groups, pooled) == [(0, 1, 4), (2, 3), (5,)]
    # merging is repeated: {0,1} + {4} first, then the pooled {0,1,4} reaches {5}
    pooled2 = lambda gx, gy: set(gx + gy) in ({0, 1, 4}, {0, 1, 4, 5})  # noqa: E731
    assert R.merge_islands(groups, pooled2) == [(0, 1, 4, 5), (2, 3)]


def test_truth_built_report_arms():
    rng = np.random.default_rng(9); n = 6
    F = G.forest([(0, 1, 1.), (1, 2, 1.), (1, 3, 1.), (3, 4, 1.), (4, 5, 1.)], n)
    A = [R.about(rng.normal(0, .03), rng.uniform(0, 10, 2), rng.normal(0, .06, 2)) for _ in range(n)]
    parent, order = R.tree_order(F); keys = sorted((parent[a], a) for a in order if parent[a] is not None); piv = [np.array([5., 5.])]
    K_or = R.compose(F, R.truth_deltas(keys, A), 'prior', piv)[1]
    assert all(np.allclose(K_or[a] @ A[a], K_or[0] @ A[0], atol=1e-12) for a in range(n))           # oracle: one common transform
    K_one = R.islands_K(F, [tuple(range(n))], A, 'prior', piv)
    assert all(np.allclose(x, y, rtol=0., atol=1e-12) for x, y in zip(K_one, K_or))                  # one island = oracle_all (reuse)
    surf = [rng.uniform(0, 10, (int(rng.integers(3, 9)), 2)) for _ in range(n)]                     # island pivots from surfels, as reg's
    piv_s = [np.concatenate(surf).mean(0)]
    K_or_s = R.compose(F, R.truth_deltas(keys, A), 'prior', piv_s)[1]
    assert all(np.allclose(x, y, rtol=0., atol=1e-12) for x, y in zip(R.islands_K(F, [tuple(range(n))], A, 'prior', piv_s, surf), K_or_s))
    K2 = R.islands_K(F, [(0, 1, 2), (3, 4, 5)], A, 'prior', piv_s, surf)                             # two islands: island-level prior gauge
    for isl in ((0, 1, 2), (3, 4, 5)):
        c = np.concatenate([surf[a] for a in isl]).mean(0)
        Mk = R.mean_transform([K2[a] for a in isl], c)                                                # each island's mean correction = I
        assert abs(R.yaw(Mk)) < 1e-12 and np.allclose(R.apply2(Mk, c), c, atol=1e-12)
        assert all(np.allclose(K2[a] @ A[a], K2[isl[0]] @ A[isl[0]], atol=1e-12) for a in isl)        # members at their true relative poses
    empty = [np.zeros((0, 2))] * n                                                                    # no surfels: the component pivot
    assert all(np.array_equal(x, y) for x, y in zip(R.islands_K(F, [(0, 1, 2), (3, 4, 5)], A, 'prior', piv_s, empty),
                                                    R.islands_K(F, [(0, 1, 2), (3, 4, 5)], A, 'prior', piv_s)))
    isl = [(0, 1, 2), (3, 4, 5)]
    K_isl = R.islands_K(F, isl, A, 'root', piv)
    assert np.array_equal(K_isl[0], np.eye(3)) and np.array_equal(K_isl[3], np.eye(3))                # anchors keep their believed frame
    for a in (1, 2):
        assert np.allclose(K_isl[a] @ A[a], A[0])
    for a in (4, 5):
        assert np.allclose(K_isl[a] @ A[a], A[3])
    use = {k: k != (1, 3) for k in keys}                                                             # pairwise: edge 1-3 unobservable
    K_cp = R.compose(F, R.truth_deltas(keys, A, use), 'root', piv)[1]
    assert np.allclose(K_cp[3], K_cp[1]) and np.allclose(K_cp[5] @ A[5], K_cp[3] @ A[3])              # 1-3 kept at identity, 3-4-5 true


def test_eval_arm_alignment_direction_and_identity_fallback():          # review S6
    occ, cell = _room(); rays = G.W.fibonacci_sphere(G.W.BEAMS); ds = 0.5 * float(cell.min()); shape = occ.shape
    cen = [np.array([1.6, 4.6, 1.3]), np.array([4.4, 4.8, 1.3])]
    scans, clean = [], []
    for a, p in enumerate(cen):
        hs, ls = G.trace_pose(occ, cell, p, rays)
        scans.append([G.measure(G.CONDITIONS['clean'], hs, ls, p, G.pose_draws(5, 1, 0, a, 0, len(rays)), ds, rays)[0]])
        clean.append(G.W.observe(occ, cell, p[None], rays))
    slab = np.zeros(shape, bool); slab[:, :, 0] = True; slab[:, :, 27:] = True
    obj = np.zeros(shape, bool); obj[10:16, 40:48, 1:8] = True; obj[44:48, 10:30, 1:20] = True; obj[45, 45, 1:25] = True
    H = G.House(occ, slab, obj, cell); F = G.forest([(0, 1, 1.)], 2)
    T = np.maximum(clean[0], clean[1]); T8 = np.stack([T, T]); R8T = [G.Target(T, H, mult=(np.stack(clean) >= 0).sum(0))]
    M_off = R.about(math.radians(1.0), (3.2, 3.2), (.05, -.04))                                       # a global error of the fused map
    maps0 = [R._integrate(shape, cell, ds, scans[a], M_off) for a in range(2)]
    out = R.eval_arm(maps0, scans, [M_off, M_off], F, R8T, T8, shape, cell, ds, dict(R.MATCH), G.RelayStats())
    g = out['align'][0]['G']
    assert -1.5 < g[0] < -0.5, g                                                                      # G undoes the +1 deg error (direction)
    assert out['aligned'][0][0]['furn_f1'] > out['asis'][0][0]['furn_f1'] and out['aligned_below_asis'] == 0
    T_empty = np.full(shape, -1, np.int8); T8e = np.stack([T_empty, T_empty])                          # no truth surfels -> G = I
    out = R.eval_arm(maps0, scans, [M_off, M_off], F, R8T, T8e, shape, cell, ds, dict(R.MATCH), G.RelayStats())
    assert out['align'][0]['G'] == [0., 0., 0.] and out['aligned'] == out['asis']


if __name__ == '__main__':
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for k, f in tests:
        f(); print('ok', k, flush=True)
    print(f'all {len(tests)} tests passed')
