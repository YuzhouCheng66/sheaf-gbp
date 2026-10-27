"""unit tests for g4_quant_wire.py (G4-E5, D8-52). Run: CUDA_VISIBLE_DEVICES= python -B -m pytest -q test_g4_quant_wire.py
(or python -B test_g4_quant_wire.py). Synthetic data and the public E1c fit counts only; no VAL / fresh map is read."""
import os
import sys

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import itertools, json, math  # noqa: E401,E402
import numpy as np  # noqa: E402
import g4_quant_wire as Q  # noqa: E402

E, G, C = Q.E, Q.G, Q.C


def _set_partitions(n, q):
    """all partitions of range(n) into exactly q non-empty labelled-by-first-appearance blocks"""
    def rec(i, lab, m):
        if i == n:
            if m == q:
                yield list(lab)
            return
        for b in range(min(m + 1, q)):
            lab.append(b)
            yield from rec(i + 1, lab, max(m, b + 1))
            lab.pop()
    yield from rec(0, [], 0)


# ---------------------------------------------------------------- quantiser
def test_dp_equals_bruteforce_contiguous():
    rng = np.random.default_rng(0)
    for trial in range(40):
        n = int(rng.integers(5, 10)); q = int(rng.integers(2, min(5, n) + 1))
        no = rng.integers(0, 50, n).astype(float); nf = rng.integers(0, 50, n).astype(float)
        if trial % 5 == 0:
            no[rng.integers(0, n)] = 0; nf[rng.integers(0, n)] = 0
        c = np.log((no + .5) / (nf + .5))
        lab, cost, order = Q.opt_partition(no, nf, c, q)
        assert len(np.unique(lab)) == q
        assert abs(Q.partition_cost(no, nf, lab) - cost) < 1e-9
        best = math.inf
        for cuts in itertools.combinations(range(1, n), q - 1):
            b = [0] + list(cuts) + [n]; lb = np.zeros(n, np.int64)
            for k in range(q):
                lb[order[b[k]:b[k + 1]]] = k
            best = min(best, Q.partition_cost(no, nf, lb))
        assert cost <= best + 1e-9 and cost >= best - 1e-9, (trial, cost, best)
        # bins are contiguous in c order and numbered in increasing c
        assert np.all(np.diff(lab[order]) >= 0)


def test_contiguous_in_llr_order_is_globally_optimal():
    """binary label: the best partition into q cells is contiguous in LLR order (so the DP is the MI-optimal quantiser)"""
    rng = np.random.default_rng(1)
    for trial in range(12):
        n = 7; q = 2 + trial % 2
        no = rng.integers(1, 40, n).astype(float); nf = rng.integers(1, 40, n).astype(float)
        c = np.log((no + .5) / (nf + .5))
        _, cost, _ = Q.opt_partition(no, nf, c, q)
        glob = min(Q.partition_cost(no, nf, np.asarray(lb)) for lb in _set_partitions(n, q))
        assert abs(cost - glob) < 1e-9, (trial, cost, glob)


def test_level_value_formula_matches_e1c():
    fit = Q._jload(Q.FIT)
    for half in ('A', 'B'):
        no, nf, t = Q.fit_counts(fit, half)
        single = Q.level_values(no, nf, np.arange(Q.NL))                 # singleton bins == E1c's per-symbol cq
        assert np.array_equal(single, np.asarray(t['cq'], np.int64))
        lab = np.zeros(Q.NL, np.int64); lab[Q.NL // 2:] = 1; v = Q.level_values(no, nf, lab)
        po, pf = no[lab == 0].sum(), nf[lab == 0].sum()
        ref = round((math.log((po + .5) / (no.sum() + .5 * Q.NL)) - math.log((pf + .5) / (nf.sum() + .5 * Q.NL))) / 0.05)
        assert v[0] == ref


def test_quantizer_build_deterministic_and_matches_file():
    fit = Q._jload(Q.FIT)
    A = Q.build_quantizer(fit)
    for h in ('A', 'B'):
        for arm in ('Q3', 'Q4', 'VOTE'):
            d = A[h][arm]
            assert np.array_equal(np.asarray(d['values'])[np.asarray(d['labels'])], np.asarray(d['table']))
            assert len(d['values']) == d['q'] and np.all(np.diff(d['values']) > 0)
        assert A[h]['Qinf']['table'] == fit['halves'][h]['tiers']['cam_nf']['cq']
    assert A['A']['Q3']['values'] == [-97, 5, 95] and A['A']['Q3']['c_order_cut_sizes'] == [31, 56, 24]   # R-B probe (section 4)
    p = os.path.join(Q.QUANT_DIR, 'criteria_quantizer.json')
    if os.path.exists(p):
        doc = Q._jload(p)
        for h in ('A', 'B'):
            for arm in Q.ARMS:
                assert doc['quantizer'][h][arm]['table'] == A[h][arm]['table']


# ---------------------------------------------------------------- floor: known mask, exact relay
def _toy(n=4, shape=(12, 10, 6), seed=0):
    rng = np.random.default_rng(seed)
    maps = []
    for a in range(n):
        L = rng.integers(Q.LMIN, Q.LMAX + 1, shape).astype(np.int8)
        L[rng.random(shape) < 0.45] = Q.UNK
        maps.append(L)
    edges = [(0, 1, 1.), (1, 2, 1.)] + ([(3, 4, 1.)] if n > 4 else [])
    return maps, G.forest(edges, n)


def test_known_mask_invariant_and_exact_relay():
    fit = Q._jload(Q.FIT); A = Q.build_quantizer(fit)
    maps, F = _toy(n=5)
    sL, _, mnL = E.relay_sums(maps, F, E.Evidence.from_spec('L').table)
    rng = np.random.default_rng(3)
    for h in ('A', 'B'):
        for arm in Q.ARMS:
            tab = np.asarray(A[h][arm]['table'], np.int64)
            sums, msgs, msgn = E.relay_sums(maps, F, tab)                  # asserts belief == direct component sum internally
            for ci, cm in enumerate(F['comps']):
                assert np.array_equal(sums[ci][1], sL[ci][1])
                S_direct = sum(E.stalk(maps[a], tab)[0] for a in cm)
                assert np.array_equal(sums[ci][0], S_direct)
                lg = rng.normal(size=S_direct.shape).astype(np.float32)
                p = E.net_readout(lg, sums[ci][1], 2000)
                assert np.array_equal(p >= 0, sL[ci][1] > 0)              # NET never fills unknown voxels
                assert np.array_equal(E.linear_readout(sums[ci][0], sums[ci][1], 7) >= 0, sL[ci][1] > 0)
            for e in F['directed']:
                assert np.array_equal(msgn[e][0], mnL[e][0])
            if arm != 'Qinf':                                              # level-count messages reproduce (S, k) exactly
                lm, own = Q.level_relay(maps, F, np.asarray(A[h][arm]['labels']), np.asarray(A[h][arm]['values']), msgs, msgn)
                for a, (N, K) in enumerate(own):
                    assert np.array_equal(K, maps[a] != Q.UNK) and np.array_equal(N.sum(0), K.astype(np.int32))


def test_quantised_stalk_is_local():
    """the quantised stalk of an agent depends only on its own map (C1): changing another agent's map leaves it unchanged"""
    fit = Q._jload(Q.FIT); A = Q.build_quantizer(fit); tab = np.asarray(A['A']['Q3']['table'])
    maps, _ = _toy(n=3)
    e0, k0 = E.stalk(maps[0], tab)
    maps[1] = np.full_like(maps[1], 5)
    e1, k1 = E.stalk(maps[0], tab)
    assert np.array_equal(e0, e1) and np.array_equal(k0, k1)
    assert set(np.unique(e0[k0]).tolist()) <= set(A['A']['Q3']['values'])


# ---------------------------------------------------------------- level-count coder
def _field(q, shape, kmax, seed, smooth=True):
    rng = np.random.default_rng(seed)
    X, Y, Z = shape
    k = rng.integers(0, kmax + 1, shape)
    if smooth:                                                             # spatially correlated known mask / levels
        k[:, : Y // 3] = 0
        base = (np.arange(X)[:, None, None] + np.arange(Z)[None, None, :] + 0 * np.arange(Y)[None, :, None]) % q
    else:
        base = rng.integers(0, q, shape)
    N = np.zeros((q,) + shape, np.int64)
    for idx in np.ndindex(*shape):
        for _ in range(int(k[idx])):
            l = base[idx] if rng.random() < 0.8 else rng.integers(0, q)
            N[(l,) + idx] += 1
    return N


def test_level_coder_roundtrip_and_ideal_length():
    cases = [(3, (7, 6, 5), 4, 0, True), (4, (6, 5, 6), 5, 1, False), (2, (5, 5, 4), 1, 2, True), (3, (4, 4, 3), 0, 3, True),
             (4, (8, 7, 4), 3, 4, True)]
    for q, shape, kmax, seed, smooth in cases:
        N = _field(q, shape, kmax, seed, smooth)
        n0, n1, hd = Q.lev_counts(N)
        seq, hd2 = Q.lev_symbols(N)
        assert hd == hd2
        c0 = np.zeros_like(n0); c1 = np.zeros_like(n1)
        for ctx, bit in seq:
            (c1 if bit else c0)[ctx] += 1
        assert np.array_equal(c0, n0) and np.array_equal(c1, n1), 'vectorised contexts != decoding-order contexts'
        ideal = G.kt_bin_bits(n0, n1)
        bits, hd3 = Q.lev_encode(N)
        Nd = Q.lev_decode(bits, shape, hd3)
        assert np.array_equal(Nd, N), f'round trip failed for case {(q, shape, kmax, seed)}'
        assert len(bits) <= ideal + 16, (len(bits), ideal)
        assert len(bits) >= ideal - 1e-6 - 2


def test_level_coder_onehot_agent_map():
    rng = np.random.default_rng(5)
    L = rng.integers(Q.LMIN, Q.LMAX + 1, (6, 5, 4)).astype(np.int8); L[rng.random(L.shape) < .5] = Q.UNK
    lab = (np.arange(Q.LMIN, Q.LMAX + 1) > 0).astype(np.int64) + (np.arange(Q.LMIN, Q.LMAX + 1) > 30)
    N, K = Q.onehot_levels(L, lab, 3)
    assert np.array_equal(N.sum(0) > 0, K) and N.max() <= 1
    bits, hd = Q.lev_encode(N.astype(np.int64))
    assert np.array_equal(Q.lev_decode(bits, L.shape, hd), N)
    assert abs(Q.lev_bytes(N) - (G.kt_bin_bits(*Q.lev_counts(N)[:2]) / 8 + 4)) < 1e-9


# ---------------------------------------------------------------- seeds, QPOST4, B_RO
def test_new_seeds_disjoint_from_reserved():
    assert not (set(Q.NEW_SEEDS) & set(E.RESERVED_SEEDS)) and not (set(Q.NEW_SEEDS) & set(E.GEN_SEEDS)) and not (set(Q.NEW_SEEDS) & {101, 102})
    assert Q.check_new_seeds(Q.NEW_SEEDS)
    for bad in (1, 2, 7, 101, 102, 200, 201, 202):
        try:
            Q.check_new_seeds([bad]); raise AssertionError(f'seed {bad} accepted')
        except AssertionError as e:
            assert 'reserved' in str(e), e
    def flat(dr):
        return np.concatenate([np.ravel(dr[k]) for k in sorted(dr)])
    for s_new in Q.NEW_SEEDS:                                              # per-pose and inter-agent draws differ from every old seed
        for s_old in (1, 2, 7, 101, 102, 200, 201, 202) + tuple(x for x in Q.NEW_SEEDS if x != s_new):
            assert not np.array_equal(flat(G.pose_draws(s_new, 1, 0, 0, 0, 16)), flat(G.pose_draws(s_old, 1, 0, 0, 0, 16)))
            assert not np.array_equal(G.frame_draws(s_new, 1, 0, 0), G.frame_draws(s_old, 1, 0, 0))


def test_qpost_levels_match_rc_definition():
    rng = np.random.default_rng(7)
    lg = rng.normal(0, 4, 5000).astype(np.float32); tk = -0.3; tau = 2.0
    d = lg.astype(np.float64) - tk
    rc = np.where(d < -tau, -3, np.where(d < 0, -1, np.where(d < tau, 1, 3)))
    assert np.array_equal(Q.QPOST_VALS[Q.qpost_levels(lg, tk)], rc)


def test_b_ro_root_choice_bruteforce():
    F = G.forest([(0, 1, 1.), (1, 2, 1.), (1, 3, 1.), (4, 5, 1.)], 6)
    rng = np.random.default_rng(2)
    eb = {e: float(rng.integers(1, 100)) for e in F['directed']}
    tern = [10., 20.]
    got = Q.b_ro(F, {'Sk': eb}, tern, flag=False)
    want = 0.
    for ci, cm in enumerate(F['comps']):
        best = min(sum(eb[e] for e in F['directed'] if e[0] in cm and r not in F['sides'][e]) for r in cm)
        # brute force: edges pointing to r are those whose receiver side contains r
        best2 = min(sum(eb[(s, t)] for (s, t) in F['directed'] if s in cm and r in (set(cm) - set(F['sides'][(s, t)]))) for r in cm)
        assert best == best2
        want += best + (len(cm) - 1) * tern[ci]
    assert abs(got - want) < 1e-9
    assert len(Q.toward_root(F, [0, 1, 2, 3], 0)) == 3


# ---------------------------------------------------------------- verdict rules (review M1, M2, S1, S3, S5, S7)
def _c(d, worst=None, free=0., surf=0., ratio=0.35, clean=0.998):
    return dict(d_furn=d, d_worst_house=d if worst is None else worst, worst_house='r1_i0', d_free=free, d_surface=surf, ratio=ratio,
                clean_furn=clean, d_fpr=0.)


def _cells(*ds, **kw):
    return {f'{sd}|{s}': _c(d, **kw) for (sd, s), d in zip([('342101', 3), ('342101', 4), ('342202', 3), ('342202', 4)], ds)}


def test_arm_verdict_mappings_and_undefined():
    C_ = dict(Q.CRITERIA)
    assert Q._arm_verdict(_cells(-.003, -.003, -.004, -.002), C_)['verdict'] == 'LEVER'
    assert Q._arm_verdict(_cells(-.003, -.011, -.004, -.002), C_)['verdict'] == 'NO'                  # (1) mean
    assert Q._arm_verdict(_cells(-.003, -.003, -.004, -.002, worst=-.051), C_)['verdict'] == 'NO'     # (1) worst house
    assert Q._arm_verdict(_cells(-.003, -.003, -.004, -.002, free=-.004), C_)['verdict'] == 'NO'      # (2) guard -> NO
    assert Q._arm_verdict(_cells(-.003, -.003, -.004, -.002, clean=.989), C_)['verdict'] == 'NO'
    assert Q._arm_verdict(_cells(-.003, -.003, -.004, -.002, ratio=.6), C_)['verdict'] == 'AMBIGUOUS'
    assert Q._arm_verdict(_cells(-.003, -.003, -.004, -.002, ratio=.8), C_)['verdict'] == 'NO'       # (3) > 0.75 -> NO
    cs = _cells(-.003, -.003, -.004, -.002); cs['342101|3']['d_worst_house'] = None
    assert Q._arm_verdict(cs, C_)['verdict'] == 'INCOMPLETE'                                          # no TypeError (S7)


def test_final_decision_fresh_rule_and_arm_selection():
    C_ = dict(Q.CRITERIA)
    L, N, A = dict(verdict='LEVER'), dict(verdict='NO'), dict(verdict='AMBIGUOUS')
    v3 = _cells(-.003, -.003, -.003, -.003)
    # LEVER, fresh mean below -1 pp -> downgraded to AMBIGUOUS
    f = Q.final_decision({'Q3': L, 'Q4': L}, {'Q3': v3, 'Q4': v3}, {'Q3': _cells(-.012, -.011, -.010, -.012), 'Q4': v3}, C_)
    assert f['verdict'] == 'AMBIGUOUS' and f['arm'] == 'Q3' and not f['fresh_mean_ok']
    # LEVER, fresh better and of opposite sign -> stays LEVER; the literal sign check is only reported
    f = Q.final_decision({'Q3': L, 'Q4': L}, {'Q3': v3, 'Q4': v3}, {'Q3': _cells(.001, .0005, .001, .0), 'Q4': v3}, C_)
    assert f['verdict'] == 'LEVER' and f['fresh_sign_opposite'] is True
    # NO is never upgraded by fresh (the old sign rule did), whatever fresh shows
    vn = _cells(-.02, -.02, -.02, -.02)
    for fr in (_cells(.002, .002, .002, .002), _cells(-.03, -.03, -.03, -.03)):
        f = Q.final_decision({'Q3': N, 'Q4': N}, {'Q3': vn, 'Q4': vn}, {'Q3': fr, 'Q4': fr}, C_)
        assert f['verdict'] == 'NO'
    # AMBIGUOUS never changes either
    f = Q.final_decision({'Q3': A, 'Q4': N}, {'Q3': v3, 'Q4': v3}, {'Q3': _cells(-.03, -.03, -.03, -.03), 'Q4': v3}, C_)
    assert f['verdict'] == 'AMBIGUOUS' and f['arm'] == 'Q3'
    # neither LEVER: the batch verdict is Q3's even when Q4 is "better" (S3); Q4 listed
    f = Q.final_decision({'Q3': N, 'Q4': A}, {'Q3': vn, 'Q4': v3}, {'Q3': vn, 'Q4': v3}, C_)
    assert f['verdict'] == 'NO' and f['arm'] == 'Q3' and f['q4_verdict'] == 'AMBIGUOUS'
    # Q3 not LEVER, Q4 LEVER -> fallback
    f = Q.final_decision({'Q3': A, 'Q4': L}, {'Q3': v3, 'Q4': v3}, {'Q3': v3, 'Q4': v3}, C_)
    assert f['verdict'] == 'LEVER' and f['arm'] == 'Q4' and 'FALLBACK' in f['note']


def test_aggregate_guards():
    import argparse, tempfile
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, 'jobs'))
        for seeds, cont in (('1,2', ''), ('1', '2'), ('3,4,1', '2'), ('3', '4')):                     # M2b: verdict only on 3, 4
            try:
                Q.aggregate_cmd(argparse.Namespace(out=d, seeds=seeds, continuity=cont, smoke=False, criteria=''))
                raise AssertionError(f'aggregate accepted --seeds {seeds}')
            except SystemExit as e:
                assert 'refused' in str(e)
        S = Q.aggregate(d, [3, 4], [1, 2], smoke=False, criteria_path=os.path.join(d, 'missing.json'))   # M2a: no fall-back
        assert S['decision']['final']['verdict'] == 'INCOMPLETE' and 'arms' not in S['decision']
        crit = os.path.join(d, 'crit.json')
        with open(crit, 'w') as fh:
            json.dump(dict(criteria=Q.CRITERIA, script_sha256='x'), fh)
        job = dict(stage='eval', panel='val', rooms=1, index=0, seed=3, failed=True, script_sha256='x', criteria_sha256='0' * 64, smoke=False)
        with open(os.path.join(d, 'jobs', 'val_r1_i0_s3.json'), 'w') as fh:
            json.dump(job, fh)
        S = Q.aggregate(d, [3, 4], [1, 2], smoke=False, criteria_path=crit)
        assert S['decision']['final']['verdict'] == 'INCOMPLETE'
        assert any('criteria sha' in r for r in S['decision']['final']['reasons'])
        try:                                                                                           # S5: --smoke on non-smoke jobs
            Q.aggregate(d, [1], [], smoke=True, criteria_path=crit); raise AssertionError('smoke aggregate accepted non-smoke jobs')
        except SystemExit as e:
            assert 'refused' in str(e)


def test_check_qpost_file():
    hs = sorted(set(Q.e3r_holdout_houses('342101')))
    assert hs == sorted(Q.e3r_holdout_houses('342202')) and len(hs) == 19
    files = [f"trainho_{h[len('TRAIN_'):]}_s{s}.npz" for h in hs for s in E.GEN_SEEDS]
    good = dict(smoke=False, criteria_sha256='c' * 64, houses=hs, seeds=list(E.GEN_SEEDS), files=files)
    Q.check_qpost_file(good, 'c' * 64, smoke=False)
    for bad in (dict(good, smoke=True), dict(good, criteria_sha256='d' * 64), dict(good, houses=hs[:2]), dict(good, files=files[:56]),
                dict(good, seeds=[200])):
        try:
            Q.check_qpost_file(bad, 'c' * 64, smoke=False); raise AssertionError('bad QPOST4 threshold file accepted')
        except AssertionError as e:
            assert 'accepted' not in str(e), e
    Q.check_qpost_file(dict(good, smoke=True), 'c' * 64, smoke=True)                                   # smoke eval may use it


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print('PASS', f.__name__, flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback
            print('FAIL', f.__name__, repr(e), flush=True); traceback.print_exc()
    print(f'{ok}/{len(fns)} passed', flush=True)
    sys.exit(0 if ok == len(fns) else 1)
