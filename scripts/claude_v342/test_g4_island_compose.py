"""Unit tests for g4_island_compose (Claude, 2026-10-27, batch G4-E6, D8-51). CPU, about a minute:  python -B test_g4_island_compose.py
Covers: the composition rule on toy trees (rejected edges carry nothing; inside an island = E4 compose; the island pivot is the mean
of the BELIEVED positions), the G4-E6 K path never reads the truth (mock / poisoned records), the judged row = E4 eval_arm (same G,
same B4 rows), the read-outs reproduce E3-R on a known task (R-A's independent recompute, noreg, cam / cam_nf; skipped if the local
inputs/ copy is absent), the O17 row comparison and the D8-51 verdict logic; after the review (code_review_g4_e6.md): S1 fresh
guards in the verdict block (report only), S2 mismatch diagnostics, S3 full-sha input pins, S4 report row ceiling_acc_islands,
review S7 (= coordinator S5) the 96-task assertion."""
import copy, json, math, os, sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g4_island_compose as M  # noqa: E402

R, G, E = M.R, M.G, M.E
I3 = np.eye(3)
TOY_EDGES = [(0, 1), (0, 5), (1, 2), (1, 3), (3, 4)]          # (parent, child) of the BFS tree rooted at 0 (= R.tree_order)


def _toy(accepted, rng, est=None):
    est = est or {e: [rng.normal(0, 1.5), rng.normal(0, .08), rng.normal(0, .08)] for e in TOY_EDGES}
    edges = [dict(i=i, j=j, est=list(est[(i, j)]), corrected=(i, j) in accepted, true=[float(x) for x in rng.normal(0, 1, 3)],
                  overlap_true=dict(ok=bool(rng.random() < .5)), err_before={}, err_after={}, failed=False, n_src=100, n_tgt=90)
             for (i, j) in TOY_EDGES]
    bpos = [rng.uniform(0, 8, 2) for _ in range(6)]
    return edges, bpos, est


def test_islands_and_rejected_edges_carry_nothing():
    rng = np.random.default_rng(1)
    acc = {(0, 1), (1, 2), (3, 4)}
    edges, bpos, est = _toy(acc, rng)
    K, C, isl = M.e6_corrections(6, edges, bpos)
    assert isl == [[0, 1, 2], [3, 4], [5]], isl
    assert np.allclose(K[5], I3)                                              # a singleton island: no correction at all
    for (i, j) in [(1, 3), (0, 5)]:                                           # rejected edges: any estimate -> the same K
        e2 = copy.deepcopy(edges)
        for e in e2:
            if (e['i'], e['j']) == (i, j):
                e['est'] = [7.0, .4, -.3]
        K2 = M.e6_corrections(6, e2, bpos)[0]
        assert all(np.array_equal(a, b) for a, b in zip(K, K2))
    e3 = copy.deepcopy(edges)                                                 # an accepted edge of island {0,1,2} does not move {3,4}
    e3[0]['est'] = [3.0, .2, .1]
    K3 = M.e6_corrections(6, e3, bpos)[0]
    assert all(np.array_equal(K[a], K3[a]) for a in (3, 4, 5)) and not np.allclose(K[1], K3[1])
    # inside an island the relative pose is exactly the accepted estimate: K_i^-1 K_j = Delta_ij
    for (i, j) in acc:
        D = M.K_of(est[(i, j)])
        assert np.allclose(R.inv(K[i]) @ K[j], D, atol=1e-12)


def test_island_equals_e4_compose():
    rng = np.random.default_rng(2)
    F = G.forest([(i, j, 1.) for i, j in TOY_EDGES], 6)
    parent, _ = R.tree_order(F)
    assert sorted((p, a) for a, p in parent.items() if p is not None) == sorted(TOY_EDGES)
    # (a) every edge accepted: one island = E4 compose with the unit-prior gauge about the same pivot
    edges, bpos, est = _toy(set(TOY_EDGES), rng)
    K, _, isl = M.e6_corrections(6, edges, bpos)
    piv = np.mean(bpos, 0)
    _, K4 = R.compose(F, {k: M.K_of(est[k]) for k in TOY_EDGES}, 'prior', [piv])
    assert isl == [list(range(6))] and all(np.allclose(a, b, atol=1e-12) for a, b in zip(K, K4))
    # (b) islands: E4 compose (rejected edge = identity) restricted to an island differs from G4-E6's C by one left factor, so the
    #     per-island unit-prior gauge (same pivot) gives the same K
    acc = {(0, 1), (1, 2), (3, 4)}
    edges, bpos, est = _toy(acc, rng)
    K, C, isl = M.e6_corrections(6, edges, bpos)
    C4, _ = R.compose(F, {k: (M.K_of(est[k]) if k in acc else I3) for k in TOY_EDGES}, 'root', None)
    for g in isl:
        X = C4[g[0]] @ R.inv(C[g[0]])
        assert all(np.allclose(C4[a], X @ C[a], atol=1e-12) for a in g)      # = compose up to the anchor's left factor
        pv = np.mean([bpos[a] for a in g], 0)
        Hinv = R.inv(R.mean_transform([C4[a] for a in g], pv))
        assert all(np.allclose(Hinv @ C4[a], K[a], atol=1e-12) for a in g)


def test_gauge_pivot_is_the_believed_position_mean():
    rng = np.random.default_rng(3)
    acc = {(0, 1), (1, 2), (3, 4)}
    edges, bpos, est = _toy(acc, rng)
    K, C, isl = M.e6_corrections(6, edges, bpos)
    for g in isl:
        pv = np.mean([bpos[a] for a in g], 0)
        Hinv = R.inv(R.mean_transform([C[a] for a in g], pv))
        assert all(np.allclose(K[a], Hinv @ C[a], atol=1e-14) for a in g)
        # the island's mean displacement of its believed-position centroid is zero (unit prior), its mean yaw is zero
        assert abs(np.mean([R.yaw(K[a]) for a in g])) < 1e-12
        assert np.allclose(np.mean([R.apply2(K[a], pv) - pv for a in g], 0), 0, atol=1e-12)
    other = [p + rng.normal(0, .5, 2) for p in bpos]                           # e.g. the true positions: a different K
    K2 = M.e6_corrections(6, edges, other)[0]
    assert not all(np.allclose(a, b) for a, b in zip(K, K2))
    # believed_positions = mean xy of the agent's own public-frame scan origins
    scans = [[(np.r_[rng.uniform(0, 8, 2), 1.3], None, None, None) for _ in range(3)] for _ in range(2)]
    bp = M.believed_positions(scans)
    assert all(np.allclose(bp[a], np.mean([s[0][:2] for s in scans[a]], 0)) for a in range(2))


class _Poison(dict):
    """an edge record whose truth fields raise when read"""
    ALLOWED = ('i', 'j', 'corrected', 'est')

    def __getitem__(self, k):
        assert k in self.ALLOWED, f'G4-E6 read the truth field {k!r}'
        return dict.__getitem__(self, k)

    def get(self, k, d=None):
        assert k in self.ALLOWED, f'G4-E6 read the truth field {k!r}'
        return dict.get(self, k, d)


def test_e6_K_never_reads_the_truth():
    rng = np.random.default_rng(4)
    acc = {(0, 1), (1, 2), (3, 4)}
    edges, bpos, est = _toy(acc, rng)
    F = G.forest([(i, j, 1.) for i, j in TOY_EDGES], 6)
    scans = [[(np.r_[bpos[a], 1.3], None, None, None)] for a in range(6)]
    e4c = dict(registration=dict(edges=[_Poison(e) for e in edges], K=[[float(x) for x in rng.normal(0, 1, 3)] for _ in range(6)]),
               inter_frames_true=[[0., 0., 0.]] * 6, observability=dict(islands=[[0, 1], [2, 3, 4, 5]]),
               arms=dict(ceiling_islands=dict(K=[[0., 0., 0.]] * 6)))
    # mock 1: the true poses are random (and None): G4-E6 / E4-rule / noreg corrections do not change
    Ka, _ = M.arm_Ks(e4c, 'cam_nf', 6, scans, F, MAs=None)
    prev = None
    for s in range(3):
        MAs = [R.about(rng.normal(0, .1), rng.uniform(0, 8, 2), rng.normal(0, .3, 2)) for _ in range(6)]
        Kb, _ = M.arm_Ks(e4c, 'cam', 6, scans, F, MAs=MAs)
        for arm in ('noreg', 'e6', 'e4rule'):
            assert all(np.array_equal(x, y) for x, y in zip(Ka[arm], Kb[arm])), arm
        if prev is not None:                                                  # sanity: the mock does reach the truth-built row
            assert not all(np.allclose(x, y) for x, y in zip(prev, Kb['ceiling_acc_islands']))
        prev = Kb['ceiling_acc_islands']
    # mock 2: every truth field of the E4 record scrambled -> the same G4-E6 K
    e5 = copy.deepcopy(edges)
    for e in e5:
        e.update(true=[9., 9., 9.], overlap_true=dict(ok=not e['overlap_true']['ok']), failed=True, err_after=dict(yaw_deg=99.))
    e4c2 = dict(e4c, registration=dict(e4c['registration'], edges=e5), observability=dict(islands=[[0, 1, 2, 3, 4, 5]]))
    Kc, _ = M.arm_Ks(e4c2, 'cam_nf', 6, scans, F)
    assert all(np.array_equal(x, y) for x, y in zip(Ka['e6'], Kc['e6']))
    # the run's call passes only (edges, believed positions); positions_m is not an argument of the G4-E6 path
    import inspect
    assert list(inspect.signature(M.e6_corrections).parameters) == ['n', 'edges', 'bpos']
    src = inspect.getsource(M.e6_corrections) + inspect.getsource(M.believed_positions) + inspect.getsource(M.accepted_islands)
    for bad in ('positions_m', 'MAs', "'true'", 'overlap_true', 'frame_draws', 'inter_frames_true', 'observability'):
        assert bad not in src, bad


def test_ceiling_acc_islands_row():
    """O24 (review S4, report only): inside every accepted-edge island the corrected frames agree exactly (true relative poses),
    the gauge is G4-E6's (believed-position pivot), and with exact edge estimates G4-E6 equals this row"""
    rng = np.random.default_rng(6)
    acc = {(0, 1), (1, 2), (3, 4)}
    MAs = [R.about(rng.normal(0, .02), rng.uniform(0, 8, 2), rng.normal(0, .06, 2)) for _ in range(6)]
    true_est = {(i, j): R.params(MAs[i] @ R.inv(MAs[j])) for (i, j) in TOY_EDGES}
    edges, bpos, _ = _toy(acc, rng, est=true_est)
    K6, _, isl = M.e6_corrections(6, edges, bpos)
    Kc = M.acc_islands_true_K(isl, MAs, bpos)
    for g in isl:
        for x in g:
            for y in g:
                assert np.allclose(R.inv(Kc[x] @ MAs[x]) @ (Kc[y] @ MAs[y]), I3, atol=1e-12)
    assert all(np.allclose(a, b, atol=1e-9) for a, b in zip(K6, Kc))           # exact estimates -> the same corrections


def test_rows_diff_reports_size_of_a_mismatch():                              # review S2
    mine = [[dict(furn_f1=.80, free_recall=.99, strata=dict(h_furn=[[1]], k_furn=[[1]])), 8], [dict(furn_f1=.5, free_recall=.9), 2]]
    theirs = [[dict(furn_f1=.80, free_recall=.99, strata=dict(h_furn=[[1]])), 8], [dict(furn_f1=.5002, free_recall=None), 2]]
    d = M.rows_diff(mine, theirs)
    assert d['n_unequal_comp'] == 1 and abs(d['max_abs']['furn_f1'] - .0002) < 1e-12 and d['max_abs']['free_recall'] == 'none_mismatch'
    assert d['keys_differing'] == ['free_recall', 'furn_f1']
    assert M.rows_diff(mine, theirs[:1])['structure'] == 'different component count'


def test_inputs_are_pinned_and_96_tasks():                                    # review S3 / S7 (coordinator S5)
    m, sha = M.e4_manifest()
    assert len(m) == M.N_EXPECTED and sha == M.E4_MANIFEST_SHA256
    assert M.pinned_e4_manifest() == m
    exp, skipped, missing = M.expected_jobs(M.PANELS, M.SEEDS)
    assert len(exp) == M.N_EXPECTED and not skipped and not missing and set(exp) == set(m)
    RO = M.load_readouts()                                                    # full-sha pins of checkpoints, thresholds and fit
    assert RO['info']['fit_sha256'] == M.FIT_SHA256
    assert [RO['info']['nets'][f'NET_{c[0]}']['ckpt_sha256'] for c in M.CKPTS] == [c[3] for c in M.CKPTS]
    assert [RO['info']['nets'][f'NET_{c[0]}']['threshold_sha256'] for c in M.CKPTS] == [c[4] for c in M.CKPTS]
    import tempfile
    d = tempfile.mkdtemp(prefix='e6t96_'); os.makedirs(os.path.join(d, 'jobs'))
    orig = M.expected_jobs
    try:
        M.expected_jobs = lambda p, s: (exp[:-1], [], [])                     # a directory that lost one job -> INCOMPLETE
        S = M.aggregate(d)
    finally:
        M.expected_jobs = orig
    assert S['verdict']['verdict'] == 'INCOMPLETE' and any('expected 96' in x for x in S['verdict']['reasons'])


def _room32():
    shape = (64, 64, 32); cell = np.array([.1, .1, .1]); occ = np.zeros(shape, np.uint8)
    occ[:, :, 0] = 1; occ[:, :, 27:] = 1
    occ[2:4, 2:62, :] = occ[60:62, 2:62, :] = occ[2:62, 2:4, :] = occ[2:62, 60:62, :] = 1
    occ[30:32, 4:36, :] = 1
    occ[10:16, 40:48, 1:8] = 1; occ[44:48, 10:30, 1:20] = 1; occ[45, 45, 1:25] = 1
    return occ, cell


def test_judged_row_is_e4_eval_arm():
    """the aligned row of eval_arm6 = E4's eval_arm (same G_c, same B4 asis / aligned rows); B6 / NET read the same aligned maps"""
    occ, cell = _room32(); rays = G.W.fibonacci_sphere(G.W.BEAMS); ds = 0.5 * float(cell.min()); shape = occ.shape
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
    M_off = R.about(math.radians(1.0), (3.2, 3.2), (.05, -.04))
    maps0 = [R._integrate(shape, cell, ds, scans[a], M_off) for a in range(2)]
    ref = R.eval_arm(maps0, scans, [M_off, M_off], F, R8T, T8, shape, cell, ds, dict(R.MATCH), G.RelayStats())
    mine = M.eval_arm6(maps0, scans, [M_off, M_off], F, R8T, T8, H, (shape, cell, ds), dict(R.MATCH), M.load_readouts(), G.RelayStats())
    assert [x['G'] for x in mine['align']] == [x['G'] for x in ref['align']]
    assert M.rows_equal_e4(mine['asis']['B4'], ref['asis']) and M.rows_equal_e4(mine['aligned']['B4'], ref['aligned'])
    assert set(mine['aligned']) == set(M.READOUTS)
    for ro in M.READOUTS:                                             # every read-out shares the known mask (k > 0) -> same FPR
        assert mine['aligned'][ro][0][0]['unknown_known_fpr'] == mine['aligned']['B4'][0][0]['unknown_known_fpr']


def _export(name):
    p = os.path.join(M.INP, 'export', name + '.npz')
    ra = os.path.join(M.A27, 'review_E3R', 'RA', 'out_recompute', name + '.json')
    return (p, ra) if (os.path.exists(p) and os.path.exists(ra)) else (None, None)


def test_readouts_reproduce_e3r_noreg():
    """noreg read-outs on a known task = R-A's independent recompute of E3-R (NET_342101 / NET_342202 = R8_net; B6_A / B6_B = R8_lin),
    furniture F1 and free recall value for value, cam and cam_nf"""
    name = 'val_r1_i0_s1'
    p, ra = _export(name)
    if p is None:
        print('  (skipped: local inputs/ copy absent)'); return
    z = np.load(p); ref = M._jload(ra)
    occ = z['occ']; private = z['private']; n = private.shape[0]
    H = G.House(occ, z['slab'].astype(bool), z['obj'].astype(bool), z['cell'])
    F = G.forest(E.edges_list(z['edges']), n)
    assert [list(c) for c in F['comps']] == ref['comps']
    R8T = [G.Target(private[cm].max(0), H, mult=(private[cm] >= 0).sum(0)) for cm in F['comps']]
    RO = M.load_readouts()
    for tn in ('cam', 'cam_nf'):
        maps = list(z[f'L_pub_{tn}']) if f'L_pub_{tn}' in z.files else list(z[f'L_k0_{tn}'])
        out = M.readouts(maps, F, R8T, H, RO, G.RelayStats())
        for sd, b6 in (('342101', 'B6_A'), ('342202', 'B6_B')):
            T = ref['tiers'][tn][sd]
            for ci in range(len(F['comps'])):
                assert out[f'NET_{sd}'][ci][0]['furn_f1'] == T['R8_net'][ci][0] and out[f'NET_{sd}'][ci][0]['free_recall'] == T['R8_net'][ci][2], (tn, sd)
                assert out[b6][ci][0]['furn_f1'] == T['R8_lin'][ci][0] and out[b6][ci][0]['free_recall'] == T['R8_lin'][ci][2], (tn, b6)
        print(f'  {tn}: NET_342101 {out["NET_342101"][0][0]["furn_f1"]:.4f}  NET_342202 {out["NET_342202"][0][0]["furn_f1"]:.4f}', flush=True)


def test_rows_equal_e4_ignores_only_the_k_strata():
    row = dict(furn_f1=.5, strata=dict(h_furn=[[1, 2, 3]], k_furn=[[1, 1, 1]], k_free=[[2, 3]]))
    theirs = [[dict(furn_f1=.5, strata=dict(h_furn=[[1, 2, 3]])), 4]]
    assert M.rows_equal_e4([[row, 4]], theirs)
    assert not M.rows_equal_e4([[dict(row, furn_f1=.6), 4]], theirs)
    assert M.rows_equal_e4([[row, 4]], None) is None


def _cells(val, fresh, nf=(0., 0., 0., 0.), free=(0., 0., 0., 0.), harm=(0, 0, 0, 0), fresh_free=(0., 0., 0., 0.), fresh_harm=(0, 0, 0, 0)):
    out = {'val': {}, 'fresh': {}}
    k = 0
    for s in M.SEEDS:
        out['val'][str(s)] = {}; out['fresh'][str(s)] = {}
        for nt in M.NETS:
            out['val'][str(s)][nt] = dict(delta_pp=val[k], nf_delta_pp=nf[k], free_delta_pp=free[k], n_harm=harm[k], harm_houses=[])
            out['fresh'][str(s)][nt] = dict(delta_pp=fresh[k], nf_delta_pp=0., free_delta_pp=fresh_free[k], n_harm=fresh_harm[k], harm_houses=[])
            k += 1
    return out


def test_verdict_d8_51():
    V = lambda *a, **k: M.verdict(_cells(*a, **k), [])['verdict']  # noqa: E731
    assert V((5, 5, 4, 3), (4, 3, 2.5, 3.5)) == 'LEVER'                       # fresh mean 3.25, min 2.5
    assert V((5, 5, 4, 2.9), (4, 4, 4, 4)) == 'AMBIGUOUS'                      # one VAL cell below +3 but >= +1
    assert V((5, 5, 4, 0.9), (4, 4, 4, 4)) == 'INSUFFICIENT'                   # one VAL cell < +1
    assert V((5, 5, 5, 5), (4, 4, 4, 0.9)) == 'AMBIGUOUS'                      # a fresh cell < +1
    assert V((5, 5, 5, 5), (3, 3, 3, 2.9)) == 'AMBIGUOUS'                      # fresh mean < +3
    assert V((5, 5, 5, 5), (4, 4, 4, 4), nf=(0, 0, 0, -1.01)) == 'INSUFFICIENT'  # cam_nf guard
    assert V((5, 5, 5, 5), (4, 4, 4, 4), nf=(0, 0, 0, -1.0)) == 'LEVER'        # the guard line itself passes
    assert V((5, 5, 5, 5), (4, 4, 4, 4), free=(0, 0, -0.51, 0)) == 'AMBIGUOUS'  # free recall guard
    assert V((5, 5, 5, 5), (4, 4, 4, 4), harm=(0, 4, 0, 0)) == 'AMBIGUOUS'     # > 3 / 24 harmed houses
    assert V((5, 5, 5, 5), (4, 4, 4, 4), harm=(3, 3, 3, 3)) == 'LEVER'
    assert M.verdict(_cells((5, 5, 5, 5), (4, 4, 4, 4)), ['1 tasks failed'])['verdict'] == 'INCOMPLETE'
    # S1: fresh guard read-outs are listed in the verdict block, report only (they never change the verdict)
    v = M.verdict(_cells((5, 5, 5, 5), (4, 4, 4, 4), fresh_free=(0, -2., 0, 0), fresh_harm=(5, 0, 0, 0)), [])
    assert v['verdict'] == 'LEVER'
    fg = v['fresh_guards_report_only']
    assert fg['n_harm'] == [5, 0, 0, 0] and fg['free_delta_pp'] == [0, -2., 0, 0] and fg['nf_delta_pp'] == [0., 0., 0., 0.]
    assert fg['harm_le_3_would_pass'] is False and fg['free_guard_would_pass'] is False and fg['nf_guard_would_pass'] is True
    assert '只报告' in fg['note']


def test_aggregate_marks_missing_and_failed_incomplete(tmp=None):
    import tempfile
    d = tempfile.mkdtemp(prefix='e6t_')
    os.makedirs(os.path.join(d, 'jobs'))
    with open(os.path.join(d, 'jobs', 'val_r1_i0_s1.json'), 'w') as fh:
        json.dump(dict(job='val_r1_i0_s1', failed='Traceback ...', script_sha256='x'), fh)
    S = M.aggregate(d, only=['val_r1_i0_s1', 'val_r2_i3_s1'])
    assert S['verdict']['verdict'] == 'INCOMPLETE' and S['failed'] == ['val_r1_i0_s1'] and S['missing'] == ['val_r2_i3_s1']


if __name__ == '__main__':
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for k, f in tests:
        f(); print('ok', k, flush=True)
    print(f'all {len(tests)} tests passed')
