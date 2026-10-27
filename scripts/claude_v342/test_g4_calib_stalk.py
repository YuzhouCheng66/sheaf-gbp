"""unit tests for g4_calib_stalk.py (G4-E1c, D8-42). Synthetic arrays only (no fixture, no house), CPU, ~10 s.
run:  CUDA_VISIBLE_DEVICES= python -B test_g4_calib_stalk.py"""
import os
import sys

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import g4_calib_stalk as C  # noqa: E402

G = C.G
RNG = np.random.default_rng(20261027)


def rand_map(shape, p_unk=0.4, rng=RNG):
    L = rng.integers(C.LMIN, C.LMAX + 1, size=shape).astype(np.int8)
    L[rng.random(shape) < p_unk] = G.UNK
    return L


def small_world(shape=(12, 12, 6), rng=RNG):
    occ = np.zeros(shape, np.int8); slab = np.zeros(shape, bool); obj = np.zeros(shape, bool)
    slab[:, :, 0] = True; occ[:, :, 0] = 1
    occ[0, :, :] = 1; occ[:, 0, :] = 1                                  # two walls
    for _ in range(4):                                                  # furniture boxes
        x, y = rng.integers(3, shape[0] - 3, 2); h = rng.integers(1, 4)
        occ[x:x + 2, y:y + 2, 1:1 + h] = 1; obj[x:x + 2, y:y + 2, 1:1 + h] = True
    H = G.House(occ, slab, obj, np.array([0.1, 0.1, 0.1]))
    t = np.where(rng.random(shape) < 0.25, np.int8(-1), occ.astype(np.int8)).astype(np.int8)
    return occ, H, G.Target(t, H)


def test_table_smoothing_and_formula():
    no = np.zeros(C.NL, np.int64); nf = np.zeros(C.NL, np.int64)
    no[110] = 30; no[57] = 10; nf[0] = 90; nf[57] = 10                  # many cells empty
    c = C.fit_table(no, nf)
    assert np.isfinite(c).all() and c.shape == (C.NL,)
    po = (no + .5) / (40 + .5 * 111); pf = (nf + .5) / (100 + .5 * 111)
    assert np.allclose(c, np.log(po) - np.log(pf))
    assert c[110] > 0 > c[0] and abs(c[5] - np.log((100 + 55.5) / (40 + 55.5))) < 1e-12   # empty cell: ratio of the normalisers
    q = C.quantise(c)
    assert q.dtype == np.int64 and np.array_equal(q, np.rint(c / 0.05).astype(np.int64))


def test_hist_ignores_unk_and_scope():
    L = rand_map((9, 8, 7)); scope = RNG.random(L.shape) < 0.7; y = RNG.random(L.shape) < 0.3
    no, nf = C.hist_L(L, scope, y)
    k = (L != G.UNK) & scope
    assert no.sum() == (k & y).sum() and nf.sum() == (k & ~y).sum()
    for v in (C.LMIN, 0, C.LMAX):
        assert no[v - C.LMIN] == ((L == v) & k & y).sum() and nf[v - C.LMIN] == ((L == v) & k & ~y).sum()
    allunk = np.full((3, 3, 3), G.UNK, np.int8)
    a, b = C.hist_L(allunk, np.ones((3, 3, 3), bool), np.ones((3, 3, 3), bool))
    assert a.sum() == 0 and b.sum() == 0


def test_evidence_unk_and_indexing():
    cq = np.arange(C.NL, dtype=np.int64) * 3 - 7
    L = np.array([[[G.UNK, C.LMIN, 0, C.LMAX, 17, -8]]], np.int8)
    e, k = C.evidence(L, cq)
    assert k.tolist() == [[[False, True, True, True, True, True]]]
    assert e[0, 0, 0] == 0 and e[0, 0, 1] == cq[0] and e[0, 0, 2] == cq[40] and e[0, 0, 3] == cq[110] and e[0, 0, 4] == cq[57] and e[0, 0, 5] == cq[32]
    assert e.dtype == np.int32


def test_prior_counted_once():
    cq = C.quantise(RNG.normal(0, 2, C.NL)); l0q = -37; tq = 9; theta = tq - l0q
    L1 = rand_map((6, 6, 4)); L2 = rand_map((6, 6, 4))
    e1, k1 = C.evidence(L1, cq)                                         # one agent: l0 + c(L) > t
    want1 = np.where(k1, ((l0q + e1) > tq).astype(np.int8), np.int8(-1))
    assert np.array_equal(C.readout(e1, k1, theta), want1)
    F = G.forest([(0, 1, 1.0)], 2)                                      # two agents: l0 + c(L1) + c(L2) > t (the prior once, not twice)
    own = [C.evidence(L, cq) for L in (L1, L2)]
    _, bel = G.relay(own, F, G.comb_sum)
    S, K = bel[0]
    e2, k2 = own[1]
    assert np.array_equal(S, e1 + e2) and np.array_equal(K, k1 | k2)
    want2 = np.where(K, ((l0q + e1 + e2) > tq).astype(np.int8), np.int8(-1))
    assert np.array_equal(C.readout(S, K, theta), want2)
    twice = np.where(K, ((2 * l0q + e1 + e2) > tq).astype(np.int8), np.int8(-1))
    assert not np.array_equal(twice, want2)                              # the test would catch a double-counted prior


def test_zero_observation_agent_changes_nothing():
    cq = C.quantise(RNG.normal(0, 2, C.NL)); theta = 3
    L0, L1 = rand_map((7, 6, 5)), rand_map((7, 6, 5)); Lz = np.full((7, 6, 5), G.UNK, np.int8)
    F2 = G.forest([(0, 1, 1.0)], 2); F3 = G.forest([(0, 1, 1.0), (1, 2, 1.0)], 3); F3b = G.forest([(0, 2, 1.0), (2, 1, 1.0)], 3)
    _, b2 = G.relay([C.evidence(L, cq) for L in (L0, L1)], F2, G.comb_sum)
    for F in (F3, F3b):
        _, b3 = G.relay([C.evidence(L, cq) for L in (L0, L1, Lz)], F, G.comb_sum)
        for a in range(3):
            assert np.array_equal(b3[a][0], b2[0][0]) and np.array_equal(b3[a][1], b2[0][1])
            assert np.array_equal(C.readout(*b3[a], theta), C.readout(*b2[0], theta))
    n2 = [(k.astype(np.int32), k) for _, k in [C.evidence(L, cq) for L in (L0, L1)]]
    n3 = [(k.astype(np.int32), k) for _, k in [C.evidence(L, cq) for L in (L0, L1, Lz)]]
    assert np.array_equal(G.relay(n3, F3, G.comb_sum)[1][2][0], G.relay(n2, F2, G.comb_sum)[1][0][0])   # the k count too


def test_identity_table_is_b4_and_clean_readout_exact():
    ident = np.arange(C.LMIN, C.LMAX + 1, dtype=np.int64)               # cq(L) = L, theta = 0  ->  B4 bit for bit
    maps = [rand_map((8, 7, 5)) for _ in range(4)]
    F = G.forest([(0, 1, 1.), (1, 2, 1.), (1, 3, 1.)], 4)
    _, b4 = G.fuse(maps, F, 'sum')
    _, bel = G.relay([C.evidence(L, ident) for L in maps], F, G.comb_sum)
    for a in range(4):
        assert np.array_equal(C.readout(*bel[a], 0), b4[a])
    occ, H, T = small_world()                                           # clean V3 maps: hits only in occupied, misses only in free
    clean = []
    for _ in range(3):
        L = np.where(occ == 1, RNG.choice([17, 34, 51, 68, 70], occ.shape), RNG.choice([-8, -16, -24, -32, -40], occ.shape)).astype(np.int8)
        L[RNG.random(occ.shape) < 0.3] = G.UNK; clean.append(L)
    Fc = G.forest([(0, 1, 1.), (1, 2, 1.)], 3)
    _, bc = G.relay([C.evidence(L, ident) for L in clean], Fc, G.comb_sum)
    p = C.readout(*bc[0], 0); kn = p >= 0
    assert kn.any() and np.array_equal(p[kn], occ[kn].astype(np.int8))   # the read-out is exact on clean evidence


def test_sweep_and_counts_match_score():
    occ, H, T = small_world()
    for trial in range(4):
        S = RNG.integers(-300, 300, size=occ.shape).astype(np.int32); K = RNG.random(occ.shape) < 0.8
        kc = np.where(K, RNG.integers(1, 9, size=occ.shape), 0).astype(np.int32)
        th, trt, frt = C.thr_hists(S, K, kc, T, H, 1)
        thetas, f1, fr = C.sweep(th, trt, frt)
        for theta in (-301, -50, 0, 7, 120, 299):
            r = G.score(C.readout(S, K, theta), T)
            c = C.counts_at(th, trt, frt, theta)
            assert (c[0], c[1], c[2]) == (r['furn_tp'], r['furn_fp'], r['furn_fn']), (theta, c, r)
            assert abs(c[3] / c[4] - r['free_recall']) < 1e-12
            i = int(np.flatnonzero(thetas == theta)[0])
            assert abs(f1[i] - (r['furn_f1'] or 0.)) < 1e-12 and abs(fr[i] - r['free_recall']) < 1e-12
        wk = [0.9, 0.7, 0.6, 0.65, 0.6]; l0q = -35
        for tq in (-40, 0, 25):
            r = G.score(C.readout_w(S, K, kc, wk, l0q, tq), T)
            c = C.counts_at_w(th, trt, frt, wk, l0q, tq)
            assert (c[0], c[1], c[2]) == (r['furn_tp'], r['furn_fp'], r['furn_fn']) and abs(c[3] / c[4] - r['free_recall']) < 1e-12
        th2, trt2, frt2 = C.thr_hists(S, K, kc, T, H, 3)                # component weight = expansion
        assert np.array_equal(th2, 3 * th) and trt2 == 3 * trt and frt2 == 3 * frt


def test_select_theta_respects_free_recall():
    occ, H, T = small_world()
    S = RNG.integers(-200, 200, size=occ.shape).astype(np.int32); S[T.t == 1] += 120; K = RNG.random(occ.shape) < 0.9
    kc = np.where(K, 1, 0).astype(np.int32)
    S4 = RNG.integers(-100, 100, size=occ.shape).astype(np.int32)
    th6, trt, frt = C.thr_hists(S, K, kc, T, H, 1); th4, _, _ = C.thr_hists(S4, K, kc, T, H, 1)
    sel = C.select_theta(th6, th4, trt, frt)
    assert sel['free'] + 1e-15 >= sel['b4_free']
    thetas, f1, fr = C.sweep(th6, trt, frt)
    ok = fr + 1e-15 >= sel['b4_free']
    assert abs(sel['f1'] - f1[ok].max()) < 1e-12


def test_logistic_recovers_parameters():
    x = np.linspace(-6, 6, 61); a, b = -1.3, 0.62; p = 1 / (1 + np.exp(-(a + b * x))); n = 200000
    y1 = np.rint(p * n); y0 = n - y1
    ah, bh = C.logistic(x, y1, y0)
    assert abs(ah - a) < 0.01 and abs(bh - b) < 0.01
    _, w = C.logistic(x, y1, y0, offset=a, intercept=False)
    assert abs(w - b) < 0.01


def test_sparse_roundtrip_and_seeds():
    h = np.zeros((2, C.KMAX, C.VB), np.int64); h[0, 1, 5] = 7; h[1, 4, C.VB - 1] = 3
    assert np.array_equal(C._dense(C._sparse(h)), h)
    try:
        C.main(['fit', '--out', os.devnull, '--fit-seeds', '1,101'])
    except AssertionError:
        pass
    else:
        raise AssertionError('fit seeds overlapping {1, 2} must be refused')



# ================================================================ verdict and provenance (review M1 / M2 / M3; after rv_verdict_probe.py)
import copy, json, tempfile  # noqa: E401,E402


def _row(f, fr=.98, fpr=.02, sf=.92):
    return dict(furn_f1=f, free_recall=fr, unknown_known_fpr=fpr, surface_f1=sf)


def _allc(x):
    return {(h, s): x for h in 'AB' for s in ('1', '2')}


def _S(val_d, fresh_d=None, b4=.85, clean=(1., 1.), free_b6=.99, surf_b6=.92, drop=()):
    S = dict(panels={}, failed=[], skipped=[], e1_mismatch=[], agg_reasons=[])
    for panel, dd in (('val', val_d), ('fresh', fresh_d)):
        if dd is None:
            continue
        P = dict(seeds={})
        for s in ('1', '2'):
            if (panel, s) in drop:
                continue
            t = {'cam_nf': {'B4': dict(R8=_row(b4), K0=_row(.7))}, 'clean': {'B4': dict(R8=_row(1., 1., 0., 1.), K0=_row(1.))}}
            for h in 'AB':
                if (panel, s, h) in drop:
                    continue
                t['cam_nf'][f'B6_{h}'] = dict(R8=_row(b4 + dd[(h, s)] / 100, free_b6, .02, surf_b6), K0=_row(.75))
                t['clean'][f'B6_{h}'] = dict(R8=_row(clean[0], clean[1], 0., 1.), K0=_row(1.))
            houses = sorted(C.expected_houses(panel))
            if (panel, s, 'houses') in drop:
                houses = houses[:-1]
            P['seeds'][s] = dict(houses=houses, tiers=t)
        S['panels'][panel] = P
    return S


def _v(S, smoke=False):
    return C.verdict(copy.deepcopy(S), smoke=smoke)


def test_verdict_branches():
    good = _allc(4.)
    assert _v(_S(good, _allc(2.)))['verdict'] == 'LEVER'
    assert _v(_S({**good, ('B', '2'): 0.5}, _allc(2.)))['verdict'] == 'NOT_ENOUGH'
    assert _v(_S({**good, ('B', '2'): 2.}, _allc(2.)))['verdict'] == 'AMBIGUOUS'
    assert _v(_S(good, _allc(2.), clean=(.985, 1.)))['verdict'] == 'AMBIGUOUS'           # clean furniture < .99 -> not LEVER
    assert _v(_S(good, _allc(2.), clean=(1., .989)))['verdict'] == 'AMBIGUOUS'           # clean free recall < .99
    assert _v(_S(good, _allc(2.), free_b6=.98 - 1e-6))['verdict'] == 'AMBIGUOUS'        # free recall 1e-6 below B4
    assert _v(_S(good, _allc(2.), surf_b6=.92 - .006))['verdict'] == 'AMBIGUOUS'        # surface -0.6 pp
    assert _v(_S(good, _allc(2.), surf_b6=.92 - .004))['verdict'] == 'LEVER'            # surface -0.4 pp is allowed
    V = _v(_S(good, _allc(-0.5)))                                                        # fresh mean of opposite sign
    assert V['verdict'] == 'AMBIGUOUS' and V.get('fresh_downgrade')
    V = _v(_S(_allc(-2.), _allc(1.)))                                                    # NOT_ENOUGH downgraded too
    assert V['verdict'] == 'AMBIGUOUS' and V.get('fresh_downgrade')
    V = _v(_S(good, {**_allc(1.), ('A', '1'): -2.}))                                     # 4-cell mean sign rule; per-cell signs reported
    assert V['verdict'] == 'LEVER' and V['fresh_cell_signs']['A|s1'] == -1 and set(V['fresh_seed_mean_pp']) == {1, 2}


def test_verdict_incomplete_paths():
    good = _allc(4.)
    assert _v(_S(good))['verdict'] == 'INCOMPLETE'                                      # no fresh (M2)
    assert _v(_S(good, _allc(2.), drop={('fresh', '2')}))['verdict'] == 'INCOMPLETE'    # fresh seed missing
    assert _v(_S(good, _allc(2.), drop={('fresh', '1', 'houses')}))['verdict'] == 'INCOMPLETE'   # fresh house missing
    assert _v(_S(good, _allc(2.), drop={('fresh', '2', 'B')}))['verdict'] == 'INCOMPLETE'        # fresh cell missing
    assert _v(_S(good, _allc(2.), drop={('val', '2', 'B')}))['verdict'] == 'INCOMPLETE'          # VAL cell missing
    assert _v(_S(good, _allc(2.), drop={('val', '1', 'houses')}))['verdict'] == 'INCOMPLETE'
    for key, val in (('e1_mismatch', ['val_r1_i0_s1']), ('failed', ['val_r1_i0_s1']), ('skipped', ['fresh_r1_i8_s1']),
                     ('agg_reasons', ['fit.json is a smoke fit'])):
        S = _S(good, _allc(2.)); S[key] = val
        assert _v(S)['verdict'] == 'INCOMPLETE', key
    S = _S(good, None, drop={('val', '2')})                                               # smoke: readout shown, still INCOMPLETE
    assert _v(S, smoke=True)['verdict'].startswith('INCOMPLETE (smoke readout: LEVER')


def _fit(**kw):
    halves = {h: dict(n_houses=15, world_ids=[f'{h}{i}' for i in range(15)], tiers={tn: {} for tn in C.FIT_TIERS}) for h in C.HALVES}
    f = dict(smoke=False, seeds=[101, 102], failed=[], halves=halves, script_sha256='S')
    f.update(kw)
    return f


def test_fit_reasons():
    assert C.fit_reasons(_fit()) == []
    assert C.fit_reasons(None) == ['no fit.json']
    assert C.fit_reasons(_fit(smoke=True)) and C.fit_reasons(_fit(smoke=True), smoke=True) == []
    f = _fit(); del f['smoke']
    assert C.fit_reasons(f)                                                              # an old fit.json without the flag is refused
    f = _fit(); f['halves']['B']['n_houses'] = 11
    assert any('11 houses' in r for r in C.fit_reasons(f))
    for seeds in ([7], [1, 101], [2], []):
        assert C.fit_reasons(_fit(seeds=seeds)), seeds
    assert C.fit_reasons(_fit(failed=['A:r1_i0']))
    f = _fit(); f['halves']['B']['world_ids'] = ['A3'] + f['halves']['B']['world_ids'][1:]
    assert any('world ids' in r for r in C.fit_reasons(f))
    f = _fit(); del f['halves']['A']['tiers']['harsh']
    assert C.fit_reasons(f)
    assert C.fit_reasons(_fit(), fit_sha='x', job_fit_shas={'y'})                        # fit sha != jobs' fit sha
    assert C.fit_reasons(_fit(), fit_sha='x', job_fit_shas=set())                        # no eval job recorded it
    assert C.fit_reasons(_fit(), fit_sha='x', job_fit_shas={'x'}, job_script_shas={'T'})  # script sha mismatch
    assert C.fit_reasons(_fit(), fit_sha='x', job_fit_shas={'x'}, job_script_shas={'S'}) == []


def test_eval_refuses_bad_fit_and_aggregate_finds_fit():
    d = tempfile.mkdtemp(prefix='e1c_test_')
    for bad in (_fit(smoke=True), _fit(seeds=[7]), _fit(script_sha256='not-this-script')):
        p = os.path.join(d, 'fit.json'); json.dump(bad, open(p, 'w'))
        try:
            C.main(['eval', '--out', os.path.join(d, 'ev'), '--fit', p])
        except SystemExit as e:
            assert 'refusing fit.json' in str(e)
        else:
            raise AssertionError('a bad fit.json was not refused')
        assert not os.path.exists(os.path.join(d, 'ev', 'jobs'))                          # refused before any job
    os.makedirs(os.path.join(d, 'ag', 'jobs'))
    p = os.path.join(d, 'fit.json'); json.dump(_fit(), open(p, 'w'))
    json.dump(dict(fit=p), open(os.path.join(d, 'ag', 'eval_run.json'), 'w'))
    S = C.aggregate(os.path.join(d, 'ag'))                                               # no --fit: taken from eval_run.json
    assert S['fit_path'] == p and any('fit sha' in r for r in S['agg_reasons'])           # and checked against the (absent) jobs


def test_train_halves_disjoint_and_train_only():
    sp = C.train_specs(5)
    A = {(r, i) for h, r, i in sp if h == 'A'}; B = {(r, i) for h, r, i in sp if h == 'B'}
    assert len(A) == len(B) == 15 and not (A & B)
    assert all(i % 2 == 0 for _, i in A) and all(i % 2 == 1 for _, i in B)
    assert sorted({r for r, _ in A}) == [1, 2, 3]

def test_eval_job_path_synthetic():
    """run_eval_job end to end on a synthetic house (house_ctx / tier_maps stubbed): relay assertions, B6, W and B4t rows, bytes;
    with the identity table and theta = 0, B6 and B4t (theta4 = 0) reproduce B4 row for row"""
    occ, H, T = small_world((10, 9, 5))
    n = 4; maps = [rand_map(occ.shape, 0.5) for _ in range(n)]
    F = G.forest([(0, 1, 1.), (1, 2, 1.)], n)                           # two components: {0, 1, 2} and {3}
    K0T = [G.Target(np.where(m == G.UNK, np.int8(-1), occ.astype(np.int8)), H) for m in maps]
    R8T = [G.Target(T.t, H, mult=np.ones(occ.shape, np.int64)) for _ in F['comps']]
    ctx = dict(F=F, n=n, K0T=K0T, R8T=R8T)
    ident = list(range(C.LMIN, C.LMAX + 1))
    tb = dict(cq=ident, theta=0, l0q=0, tq=0, w=[1.] * C.KMAX, w_tq=0, b4_theta=0)
    d = tempfile.mkdtemp(prefix='e1c_job_')
    p = os.path.join(d, 'fit.json'); json.dump(dict(halves={h: dict(tiers={tn: tb for tn in C.FIT_TIERS}) for h in C.HALVES}), open(p, 'w'))
    old = C.house_ctx, C.tier_maps
    C.house_ctx = lambda split, rooms, index: ctx
    C.tier_maps = lambda c, tn, seed, k0cache, pub=True: (maps, maps, False)
    try:
        rec = C.run_eval_job(('nopanel', 9, 99, 1, dict(fit=p, fit_sha256=C.file_sha256(p), script_sha256='S', bytes=True)))
    finally:
        C.house_ctx, C.tier_maps = old
    assert rec['e1_match'] == {tn: None for tn in C.EVAL_TIERS} and rec['relay']['belief_checks'] > 0
    for tn in C.EVAL_TIERS:
        t = rec['tiers'][tn]
        for h in C.HALVES:
            assert t[f'B6_{h}']['R8'] == t['B4']['R8'] and t[f'B6_{h}']['K0'] == t['B4']['K0']
            strip = lambda rows: [[{k: v for k, v in r.items() if k != 'strata'}, w] for r, w in rows]  # noqa: E731
            assert t[f'B4t_{h}']['R8'] == strip(t['B4']['R8'])
            assert [r['furn_f1'] for r, _ in t[f'B6_{h}']['R8_w']] == [r['furn_f1'] for r, _ in t['B4']['R8']]
        assert ('rnet_relay' in t['B6_A']) == (tn == C.JUDGED)

if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t(); print('ok', t.__name__, flush=True)
    print(f'{len(tests)}/{len(tests)} passed')
