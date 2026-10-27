"""Unit tests for g4_registration_msg (Claude, 2026-10-27, batch G4-E4b). CPU, about a minute:  python -B test_g4_registration_msg.py
Real-fixture checks (rows equal to G4-E1 and to E4, E_msg assertion counted, oracle ~1) are part of the smoke (--smoke-asserts)."""
import ast, inspect, json, math, os, sys, tempfile

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g4_registration_msg as M  # noqa: E402
import test_g4_registration as T4  # noqa: E402   (E4's synthetic room and planes)

R = M.R; G = M.G; I3 = M.I3; PRM = dict(R.MATCH)


def test_frozen_dependency():
    assert R.file_sha256(R.__file__) == M.E4_SHA256
    assert R.file_sha256(G.__file__).startswith('2313d75b')


def test_warp_plane():
    cell = (0.1, 0.1, 0.1); w = np.zeros((40, 40)); w[10, 20] = 1.
    assert M.warp_plane(w, I3, cell) is w
    s = M.warp_plane(w, R.about(0., (0., 0.), (0.2, -0.1)), cell)                   # +2 cells in x, -1 in y
    assert abs(s[12, 19] - 1.) < 1e-12 and abs(s.sum() - 1.) < 1e-12
    Mr = R.about(math.radians(90.), (2.05, 2.05), (0., 0.))                          # 90 deg about the centre of cell (20, 20)
    r = M.warp_plane(w, Mr, cell)
    x = R.apply2(Mr, np.array([[(10 + .5) * .1, (20 + .5) * .1]]))[0] / .1 - .5     # where the bright cell's centre goes
    k = np.unravel_index(np.argmax(r), r.shape)
    assert np.allclose(k, np.round(x)) and abs(r.max() - 1.) < 1e-9


def _forest():
    return G.forest([(0, 1, 1.), (1, 2, 1.), (1, 3, 1.), (3, 4, 1.), (5, 6, 1.)], 7)


def test_tree_sides():
    F = _forest(); sides = M.tree_sides(F)
    assert [k for k, _, _ in sides] == [(0, 1), (1, 2), (1, 3), (3, 4), (5, 6)]          # BFS order of the child
    for (i, j), A, B in sides:
        comp = F['comps'][F['comp_of'][j]]
        assert sorted(A + B) == sorted(comp) and not set(A) & set(B) and j in B and i in A
    assert dict(((k, B) for k, _, B in sides))[(1, 3)] == [3, 4] and dict(((k, B) for k, _, B in sides))[(0, 1)] == [1, 2, 3, 4]


def test_estimator_sees_no_truth():
    assert list(inspect.signature(M.register_messages).parameters) == ['maps', 'F', 'cell', 'prm', 'sweeps']
    used = set()
    for f in (M.register_messages, M.side_message, M._side_surfels, M.message_energy, M.tree_sides, M.warp_plane):
        for node in ast.walk(ast.parse(inspect.getsource(f))):
            if isinstance(node, ast.Name):
                used.add(node.id)
            elif isinstance(node, ast.Attribute):
                used.add(node.attr)
    for word in ('frame_draws', 'make_fixture', 'stream_targets', 'Target', 'private', 'positions', 'T8', 'R8T', 'occ', 'frame', 'MAs',
                 'frame_matrix', 'planes_tern', 'to_public', 'trajectory_m', 'truth_deltas', 'islands_K'):
        assert word not in used, word
    calls = [n for n in ast.walk(ast.parse(inspect.getsource(M.run_job))) if isinstance(n, ast.Call) and getattr(n.func, 'id', None) == 'register_messages']
    assert len(calls) == 1 and calls[0].args[0].id == 'L_pub'


def _sim(cen, true, cond='range_only'):
    occ, cell = T4._room(); rays = G.W.fibonacci_sphere(G.W.BEAMS); ds = 0.5 * float(cell.min())
    c = G.CONDITIONS[cond]; maps, MAs = [], []
    for a, p0 in enumerate(cen):
        scans = []
        for j, dxy in enumerate(((0., 0.), (.08, -.06), (-.07, .05))):
            p = p0 + np.r_[dxy, 0.]
            hs, ls = G.trace_pose(occ, cell, p, rays)
            scans.append(G.measure(c, hs, ls, p, G.pose_draws(5, 1, 0, a, j, len(rays)), ds, rays)[0])
        frame = (G.rotz(true[a][2]), np.array([true[a][0], true[a][1], 0.]), p0)
        MAs.append(R.frame_matrix(frame))
        maps.append(G.integrate_agent(occ.shape, cell, ds, [G.to_public(s, frame) for s in scans], ('V3',))['V3'])
    return maps, MAs, cell


CEN = [np.array([1.6, 4.6, 1.3]), np.array([4.4, 4.8, 1.3]), np.array([5.3, 2.0, 1.3])]
TRUE = [(.04, -.03, .9), (-.05, .02, -1.2), (.02, .05, .6)]


def test_two_agents_one_sweep_equals_e4():
    """single variable: with one tree edge both messages ARE the two agents' own surfels, so the first sweep is exactly E4's estimate"""
    maps, MAs, cell = _sim(CEN[:2], TRUE[:2])
    F = G.forest([(0, 1, 1.)], 2)
    e4 = R.register_forest(maps, F, cell, PRM, 'prior')
    m1 = M.register_messages(maps, F, cell, PRM, sweeps=1)
    assert np.array_equal(m1['C'][1], e4['C'][1]) and all(np.array_equal(x, y) for x, y in zip(m1['K'], e4['K']))
    assert m1['E_msg'][0] == e4['E_pose'][0] and len(m1['E_msg']) == 2                  # same energy at I; one kept edge update
    assert abs(m1['E_msg'][1] - e4['E_pose'][-1]) <= 1e-9 * e4['E_pose'][-1]           # = E4's final energy (moved surfels vs moved frame)


def test_register_messages_on_simulated_scans():
    maps, MAs, cell = _sim(CEN, TRUE)
    F = G.forest([(0, 1, 1.), (1, 2, 1.)], 3)
    reg = M.register_messages(maps, F, cell, PRM)
    assert reg['monotone_checks'] == len(reg['E_msg']) - 1 and all(b < a for a, b in zip(reg['E_msg'], reg['E_msg'][1:]))
    assert len(reg['E_msg']) >= 3                                                    # both edges moved in the first sweep
    for x in reg['log']:                                                             # review M1: every global rejection logs the child-only E_msg
        if x.get('reason') == 'E_msg not lower':
            assert 'E_child_only' in x and x['child_only_would_lower'] == (x['E_child_only'] < x['E_try'] or x['child_only_would_lower'])
    for (i, j) in reg['keys']:
        Dt = MAs[i] @ R.inv(MAs[j]); De = R.inv(reg['C'][i]) @ reg['C'][j]; P = reg['surf_m'][j]
        b, a = R.pose_err(I3, Dt, P, P.mean(0)), R.pose_err(De, Dt, P, P.mean(0))
        assert a['disp_rms_cm'] < 0.5 * b['disp_rms_cm'] and a['disp_rms_cm'] < 2.5, (i, j, b, a)
    ae0 = R.agent_errors([I3] * 3, MAs, F, reg['pivots']); ae1 = R.agent_errors(reg['K'], MAs, F, reg['pivots'])
    assert ae1['ATE_cm'] < 0.5 * ae0['ATE_cm']
    Mk = R.mean_transform(reg['K'], reg['pivots'][0])                                # prior gauge: the corrections average to I
    assert abs(R.yaw(Mk)) < 1e-12 and np.allclose(R.apply2(Mk, reg['pivots'][0]), reg['pivots'][0], atol=1e-12)
    # E_msg is the sum of the per-edge terms at the final state, and a global reject leaves the state unchanged
    E, per = M.message_energy([R.evidence(R.planes_logodds(L, cell, PRM), cell, PRM)['S'] for L in maps],
                              {a: reg['C'][a] for a in range(3)}, reg['sides'], PRM)
    assert E == reg['E_msg'][-1] and abs(sum(per) - E) < 1e-12


def _panel_verdict(cat, complete=True):
    return dict(category=cat if complete else None, complete=complete, incomplete_reasons=[] if complete else ['seed 2: 3 of 24 houses missing'])


def test_verdict_e4b():
    V = M.verdict_e4b
    assert V(_panel_verdict('LEVER'), _panel_verdict('LEVER'))['category'] == 'LEVER'
    assert V(_panel_verdict('LEVER'), _panel_verdict('AMBIGUOUS'))['category'] == 'AMBIGUOUS'           # fresh must also be a lever
    assert V(_panel_verdict('LEVER'), _panel_verdict('INSUFFICIENT'))['category'] == 'AMBIGUOUS'
    assert V(_panel_verdict('INSUFFICIENT'), _panel_verdict('LEVER'))['category'] == 'INSUFFICIENT'     # VAL decides "not enough"
    assert V(_panel_verdict('AMBIGUOUS'), _panel_verdict('AMBIGUOUS'))['category'] == 'AMBIGUOUS'
    v = V(_panel_verdict('LEVER'), _panel_verdict('LEVER', complete=False))
    assert v['category'] is None and v['verdict'].startswith('INCOMPLETE') and 'fresh' in v['verdict']
    v = V(_panel_verdict('LEVER'), _panel_verdict('LEVER'), ['val: control / truth rows != E4 (1: ...)'])
    assert v['verdict'].startswith('INCOMPLETE') and '!= E4' in v['verdict']
    assert V(None, _panel_verdict('LEVER'))['verdict'].startswith('INCOMPLETE')
    # review M3: the oracle_all flag (addendum 1) and the aligned_best flag (addendum 4)
    pv = lambda cat, best=None, oracle=True: dict(_panel_verdict(cat), category_aligned_best=best or cat, oracle_check_ok=oracle)  # noqa: E731
    v = V(pv('LEVER'), pv('LEVER', oracle=False))
    assert v['category'] == 'LEVER' and 'alignment pipeline check failed' in v['verdict'] and 'fresh' in v['verdict']
    v = V(pv('LEVER'), pv('LEVER', best='AMBIGUOUS'))
    assert v['category'] == 'LEVER' and v['category_aligned_best'] == 'AMBIGUOUS' and 'sensitive to alignment' in v['verdict']
    v = V(pv('AMBIGUOUS', best='INSUFFICIENT'), pv('AMBIGUOUS'))
    assert v['category'] == 'AMBIGUOUS' and v['category_aligned_best'] == 'INSUFFICIENT' and 'sensitive to alignment' in v['verdict']
    v = V(pv('INSUFFICIENT'), pv('LEVER'))
    assert 'sensitive' not in v['verdict'] and 'check failed' not in v['verdict']


def test_e4_compare_and_mechanism():
    rows = [[dict(furn_f1=.7, furn_tp=3), 8]]
    arms = {a: dict(asis=rows, aligned=rows) for a in M.COMPARE_ARMS}
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, 'jobs'))
        e4job = dict(panel='val', rooms=1, index=0, seed=1, conds=dict(cam=dict(arms=dict(arms, reg=dict(asis=rows, aligned=rows)))))
        json.dump(e4job, open(os.path.join(tmp, 'jobs', 'val_r1_i0_s1.json'), 'w'))
        job = dict(panel='val', rooms=1, index=0, seed=1, conds=dict(cam=dict(arms=dict(arms, reg=dict(asis=[[dict(furn_f1=.8), 8]], aligned=rows)))))
        assert M.e4_compare_job(job, tmp, {}) == dict(cam='equal')                                   # reg may differ, controls may not
        job2 = json.loads(json.dumps(job)); job2['conds']['cam']['arms']['ceiling_islands']['aligned'] = [[dict(furn_f1=.71, furn_tp=3), 8]]
        assert M.e4_compare_job(job2, tmp, {}) == dict(cam='unequal')
        assert M.e4_compare_job(dict(job, seed=2), tmp, {}) == dict(cam='missing') and M.e4_compare_job(job, None, {}) == dict(cam='no_e4_dir')

    def jb(seed, house, edges, disp, F, e4b=True, opt=None):
        """edges: E4b -> (kept_updates, rejected_global) pairs; E4 -> corrected flags"""
        rows = lambda f: [[dict(furn_f1=f), 8]]  # noqa: E731
        E = ([dict(corrected=k > 0, kept_updates=k, rejected_global=r) for k, r in edges] if e4b else [dict(corrected=c) for c in edges])
        reg = dict(edges=E, agent_err_after=dict(per_agent=[dict(disp_cm=d) for d in disp]))
        if e4b:
            above = (opt or 'le') == 'above'
            reg['optimizer_diag'] = dict(E_msg_E4b_final=5. if above else 4., E_msg_at_E4_solution=4.5, E_msg_at_true_poses=4.2, E_msg_initial=9.,
                                         E4b_final_above_E4_solution=above, true_poses_below_E4b_final=above, global_rejections=1, child_only_would_lower=1)
        return dict(panel='val', seed=seed, house=house, conds=dict(cam=dict(
            registration=reg, arms=dict(noreg=dict(aligned=rows(F[0])), reg=dict(aligned=rows(F[1])), truepose=dict(aligned=rows(F[2]))))))
    # seed 1: 4 edges, one gate-refused (0, 0), one globally rejected (0, 2); seed 2: three gate-refused
    e4b = [jb(1, 'a', [(2, 0), (1, 1), (0, 0), (0, 2)], [1., 2.], (.7, .78, .9)), jb(2, 'a', [(2, 0), (0, 0), (0, 0), (0, 0)], [5., 5.], (.7, .74, .9))]
    e4 = [jb(1, 'a', [True, True, False, False], [2., 3.], (.7, .73, .9), e4b=False), jb(2, 'a', [True, True, False, False], [4., 4.], (.7, .76, .9), e4b=False),
          jb(1, 'b', [False] * 4, [9.], (.5, .5, .9), e4b=False)]                         # house b is not in E4b: excluded
    m = M.mechanism(e4b, e4)
    s1, s2 = m['per_seed']['1'], m['per_seed']['2']
    assert s1['e4_houses_matched'] == 1 and s1['frac_gate_refused_edges_E4b'] == .25 and s1['frac_globally_rejected_edges_E4b'] == .25
    assert s1['frac_uncorrected_any_E4b'] == .5 and s1['frac_uncorrected_edges_E4'] == .5 and s1['gate_refused_at_least_halved']   # .25 <= .5 x .5
    assert s1['agent_err_lower'] and abs(s1['recovery_diff'] - .25) < 1e-9 and s1['optimizer_ok']
    assert not s2['gate_refused_at_least_halved'] and not s2['agent_err_lower'] and abs(s2['recovery_diff'] + .10) < 1e-9
    assert m['message_bottleneck_hypothesis'].startswith('not rejected')
    e4b[0]['conds']['cam']['arms']['reg']['aligned'] = [[dict(furn_f1=.75), 8]]                # diff .10 in seed 1 too
    assert M.mechanism(e4b, e4)['message_bottleneck_hypothesis'].startswith('rejected')
    e4b[1] = jb(2, 'a', [(2, 0), (0, 0), (0, 0), (0, 0)], [5., 5.], (.7, .74, .9), opt='above')   # seed 2: 0 % of jobs at or below E4's E_msg
    m = M.mechanism(e4b, e4)
    assert not m['per_seed']['2']['optimizer_ok'] and m['message_bottleneck_hypothesis'].startswith("limited by the D8-40 optimizer")
    o = M.optimizer_summary(e4b, 'val', 2, 'cam')
    assert o['E4b_final_above_E4_solution'] == 1 and o['frac_E4b_final_le_E4_solution'] == 0. and abs(o['excess_sum'] - .5) < 1e-12
    assert o['rejections_child_only_would_lower'] == 1 and o['true_poses_below_E4b_final'] == 1


def test_failed_job_record():
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, 'jobs'))
        opts = dict(out=tmp, conds=['cam'], job_format=M.JOB_FORMAT)
        orig = M.run_job
        try:
            def boom(job):
                raise RuntimeError('E_msg increased at step 2: 1.0 -> 2.0')
            M.run_job = boom
            path, *_, fl = M._run_and_save(('val', 1, 0, 1, opts))
        finally:
            M.run_job = orig
        rec = json.load(open(path))
        assert 'E_msg increased' in rec['failed'] and fl
        S = M.aggregate(tmp)
        assert S['decision']['verdict'].startswith('INCOMPLETE') and S['failed'][0]['house'] == 'r1_i0'
        assert any('no comparison with G4-E4' in r for r in S['decision']['reasons'])                 # review M3: no --e4-out


if __name__ == '__main__':
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for k, f in tests:
        f(); print('ok', k, flush=True)
    print(f'all {len(tests)} tests passed')
