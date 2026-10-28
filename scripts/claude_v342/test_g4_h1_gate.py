"""unit tests for g4_h1_gate.py (batch G4-H1G, D8-75). CPU only. ~2-3 min.
Data read here: TRAIN files only (one TRAIN hold-out house of the G4-E5 export, one TRAIN H1-smoke OOD file and its H1L smoke record).
No H1 VAL file, no noise seed 3 / 4 file is read.
run:  CUDA_VISIBLE_DEVICES= python -B test_g4_h1_gate.py"""
import os
import sys

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import copy, json, tempfile, time  # noqa: E401,E402
import numpy as np  # noqa: E402
import g4_h1_gate as X  # noqa: E402

HL = X.HL; W = X.W; E3 = X.E3; L = X.L; R = X.R; G = X.G
TRAIN_HO_FILE = os.path.join(W.TRAIN_HO, 'trainho_r1_i19_s200.npz')
H1_TRAIN_FILE = os.path.join(HL.EXPORT_DIR, 'floating_train_r2_i9_s203.npz')
H1L_TRAIN_REC = os.path.join(X.G4, 'h1l_smoke', 'train', 'jobs', 'floating_train_r2_i9.json')


def _synthetic_lut(xmax_bin=9):
    """every cell with x < -0.5 (bins 0..9) strong, all Lam_b >= 1: a gate that surely fires on real beliefs"""
    st = np.zeros((X.LMAX + 1, X.NX), bool); st[1:, :xmax_bin + 1] = True
    return X.lut_of(st)


# ---------------------------------------------------------------- pins
def test_frozen_and_pins():
    X.check_self_frozen()
    for p, s in ((X.CRITERIA_W, X.CRITERIA_W_SHA256), (X.E7_SUMMARY, X.E7_SUMMARY_SHA256), (X.H1L_SUMMARY, X.H1L_SUMMARY_SHA256),
                 (X.H1L_SMOKE_SUMMARY, X.H1L_SMOKE_SUMMARY_SHA256), (X.E7_SMOKE_JOB, X.E7_SMOKE_JOB_SHA256)):
        assert X.sha(p) == s, p
    thr = X.load_thresholds()
    for ts in X.SEEDS:
        assert thr[ts]['w4']['w'] == 4.0 and thr[ts]['learned']['R8']['j_occ'] == HL._jload(os.path.join(HL.INP, HL.E3L3_CKPTS[ts][0], 'threshold.json'))['R8']['j_occ']


# ---------------------------------------------------------------- binning and the fitting rule
def test_cell_index_exact():
    lb = np.array([0, 1, 1, 1, 20, 20, 20, 20, 3, 3, 64], np.float32)
    hb = np.array([0, -1, 0, 1, -19, 1, 20, -20, -1, 2, -64], np.float32)
    ix = X.cell_index(hb, lb)
    want = [-1, 40 + 0, 40 + 20, 40 + 39, 800 + 1, 800 + 21, 800 + 39, 800 + 0, 120 + 13, 120 + 33, 64 * 40 + 0]
    assert ix.tolist() == want, ix.tolist()
    # agrees with floor((x + 1) / 0.05) away from float edge cases
    rng = np.random.default_rng(0)
    lam = rng.integers(1, 30, 5000).astype(np.float32); h = np.array([rng.integers(-l, l + 1) for l in lam.astype(int)], np.float32)
    x = h.astype(np.float64) / lam
    ref = np.minimum(np.floor((x + 1) / 0.05 + 1e-9), 39).astype(int)
    assert np.array_equal(X.cell_index(h, lam) % X.NX, ref)
    for bad_h, bad_l in ((np.array([0.5], np.float32), np.array([1], np.float32)), (np.array([3], np.float32), np.array([2], np.float32)),
                         (np.array([1], np.float32), np.array([0], np.float32)), (np.array([0], np.float32), np.array([65], np.float32))):
        try:
            X.cell_index(bad_h, bad_l)
        except RuntimeError:
            continue
        raise AssertionError(f'cell_index accepted {bad_h} / {bad_l}')


def test_fit_rule_995_200():
    C = np.zeros((X.LMAX + 1, X.NX, X.NCOL), np.int64)
    C[1, 0, :2] = (200, 199)          # q = .995 exactly, n = 200 -> strong
    C[1, 1, :2] = (199, 199)          # n = 199 -> not
    C[2, 0, :2] = (1000, 994)         # q = .994 -> not
    C[2, 1, :2] = (1000, 995)         # q = .995 -> strong
    C[3, 5, :2] = (10 ** 7, 9950000)  # exactly .995 at large n -> strong
    C[3, 6, :2] = (10 ** 7, 9949999)  # just below -> not
    C[0, 0, :2] = (10 ** 6, 10 ** 6)  # Lam_b = 0 never strong
    s = X.fit_strong(C)
    assert s[1, 0] and not s[1, 1] and not s[2, 0] and s[2, 1] and s[3, 5] and not s[3, 6] and not s[0, 0]
    assert int(s.sum()) == 3
    assert X.Q_NUM / X.Q_DEN == 0.995 and X.N_MIN == 200 and X.D8_75['q_min'] == 0.995 and X.D8_75['n_min'] == 200


# ---------------------------------------------------------------- the gate changes only decoder-occupied voxels in strong cells
def test_apply_gate_synthetic():
    rng = np.random.default_rng(1)
    p = rng.integers(-1, 2, (20, 20, 8)).astype(np.int8)
    idx = np.where(p >= 0, rng.integers(0, (X.LMAX + 1) * X.NX, p.shape), -1).astype(np.int32)
    lut = rng.random((X.LMAX + 1) * X.NX) < 0.3
    q, ch = X.apply_gate(p, idx, lut)
    strong = np.zeros(p.shape, bool); strong[idx >= 0] = lut[idx[idx >= 0]]
    assert np.array_equal(ch, (p == 1) & strong)
    assert np.array_equal(q >= 0, p >= 0)                       # known area unchanged
    assert np.array_equal(q[~ch], p[~ch]) and (q[ch] == 0).all()
    assert ((p[ch]) == 1).all() and ch.sum() > 0
    q2, ch2 = X.apply_gate(p, idx, np.zeros_like(lut))
    assert np.array_equal(q2, p) and not ch2.any()


def _train_house():
    z = np.load(TRAIN_HO_FILE)
    H = G.House(z['occ'], z['slab'], z['obj'], z['cell']); private = z['private']; n = private.shape[0]
    F = G.forest(R.edges_list(z['edges']), n)
    K0T = [G.Target(private[a], H) for a in range(n)]
    R8T = [G.Target(private[cm].max(0), H, mult=(private[cm] >= 0).sum(0)) for cm in F['comps']]
    return list(z['L_k0_cam_nf']), F, H, K0T, R8T


def test_guard_path_known_bytes_k0_unchanged():
    """TRAIN house, frozen E3.eval_arm at w = 4: the recording wrapper changes nothing; the gated twin keeps known mask, FPR rows, wire / bytes,
    K0 and every non-R8 field; the gate fires only on decoder-occupied voxels of strong cells; ungated rows are reproduced from the capture"""
    Lm, F, H, K0T, R8T = _train_house()
    M = HL.load_models(1); thr = X.load_thresholds(); c = X.arm_cfg(M, thr, '342101', 'w4')
    a0 = E3.eval_arm(c['model'], c['w'], c['thr'], c['tb'], Lm, Lm, F, H, K0T, R8T, 2)
    orig = L.decode_np
    with X.DecodeCapture() as cap:
        a = E3.eval_arm(c['model'], c['w'], c['thr'], c['tb'], Lm, Lm, F, H, K0T, R8T, 2)
    assert L.decode_np is orig, 'decode_np not restored'
    assert X._jr(a) == X._jr(a0), 'the recording wrapper changed eval_arm'
    lut = _synthetic_lut()
    ag, ok = X.gate_eval_arm(a, cap.calls, F, Lm, R8T, H, c['jo'], lut, check_rows=True)
    assert ok['known_identical'] and ok['fpr_identical'] and ok['bytes_identical'] and ok['ungated_rows_reproduced']
    assert ok['gate']['changed'] > 0
    for k in a:
        if k not in ('R8', 'whole_miss'):
            assert X._jr(ag[k]) == X._jr(a[k]), k
    assert [r[0]['unknown_known_fpr'] for r in ag['R8']] == [r[0]['unknown_known_fpr'] for r in a['R8']]
    # voxel level: recompute one component by hand
    hb, lb, lg = cap.calls[0]; cm = F['comps'][0]
    kc = sum((Lm[x] != X.UNK).astype(np.int32) for x in cm)
    p = R.net_readout(lg[0], kc, c['jo']); ix = X.cell_index(hb.numpy()[0, 0], lb.numpy()[0])
    q, ch = X.apply_gate(p, ix, lut)
    x = np.where(lb.numpy()[0] > 0, hb.numpy()[0, 0] / np.maximum(lb.numpy()[0], 1), 9.0)
    assert np.array_equal(ch, (p == 1) & (x < -0.5 + 1e-9) & (lb.numpy()[0] > 0))
    assert np.array_equal(q >= 0, kc > 0)
    # gated free recall can only rise, furniture tp can only fall
    for ru, rg in zip(a['R8'], ag['R8']):
        if ru[0]['free_recall'] is not None:
            assert rg[0]['free_recall'] >= ru[0]['free_recall']
        assert rg[0]['furn_tp'] <= ru[0]['furn_tp'] and rg[0]['furn_fp'] <= ru[0]['furn_fp']


# ---------------------------------------------------------------- H1: ungated arm == the H1L record (the gate, per file)
def test_h1_ungated_equals_h1l_record():
    """TRAIN H1-smoke file: the rewritten H1 evaluation (learned w, ungated) == the frozen H1L smoke record field by field; the gated
    record keeps the controls and changes only E3-L3 layers"""
    rec = HL._jload(H1L_TRAIN_REC)
    assert rec['script_sha256'] == X.SELF_FROZEN['g4_h1_e3l3']
    hs = rec['seeds']['203']
    assert X.sha(H1_TRAIN_FILE) == hs['file_sha256']
    M = HL.load_models(1); thr = X.load_thresholds()
    cfgs = {'learned': {ts: X.arm_cfg(M, thr, ts, 'learned') for ts in X.SEEDS}}
    luts = {ts: {'learned': _synthetic_lut()} for ts in X.SEEDS}
    out = X.h1_evaluate_file(H1_TRAIN_FILE, M, cfgs, luts, gbp_check=True)
    u, g = out[('learned', 'ungated')], out[('learned', 'gated')]
    for key in ('R8', 'flips', 'flips_u', 'flip_split', 'z', 'components', 'families', 'layer1', 'zone_voxels', 'ood_voxels', 'relay', 'n_agents'):
        assert X._jr(u[key]) == hs[key], key
    for arm in HL.ARMS:
        a = {k: v for k, v in u['e3'][arm].items() if not k.startswith('seconds')}
        b = {k: v for k, v in hs['e3'][arm].items() if not k.startswith('seconds')}
        assert X._jr(a) == b, arm
    ok, bad = HL.compare_controls(g, hs)
    assert ok, bad
    assert any(X._jr(g['R8'][arm]) != X._jr(u['R8'][arm]) for arm in HL.ARMS), 'synthetic gate should change something'
    for arm in HL.ARMS:
        st = g['gate'][arm]
        assert st['changed'] == sum(st['truth']) and sum(v[0] + v[1] + v[2] for v in st['cells'].values()) == st['changed']
        assert all(int(k.split('_')[1]) <= 9 for k in st['cells'])
        for (ru, _), (rg, _) in zip(u['R8'][arm]['rows'], g['R8'][arm]['rows']):
            assert rg['unknown_known_fpr'] == ru['unknown_known_fpr']


# ---------------------------------------------------------------- comparators used by the gates
def test_gate_comparators_detect_differences():
    S = HL._jload(X.H1L_SMOKE_SUMMARY)
    assert X.gate_h1l(S, S, smoke=True)['equal']
    S2 = copy.deepcopy(S); S2['variants']['floating']['203']['R8']['E3L3_342101']['furn_f1'] += 1e-15
    r = X.gate_h1l(S2, S, smoke=True)
    assert not r['equal'] and 'furn_f1' in r['diffs'][0]
    S3 = copy.deepcopy(S); S3['verdict']['reasons'] = ['x']
    assert X.gate_h1l(S3, S, smoke=True)['equal'] and not X.gate_h1l(S3, S, smoke=False)['equal']
    E7S = HL._jload(X.E7_SUMMARY)
    mine = dict(panels={p: dict(seeds={s: dict(tiers={tn: {f'E3L3_{ts}': copy.deepcopy(T[f'E3L3W_{ts}']) for ts in X.SEEDS} for tn, T in D['tiers'].items()})
                                     for s, D in P['seeds'].items()}) for p, P in E7S['panels'].items()})
    r = X.gate_e7(mine, E7S, 'w4')
    assert r['equal'] and r['compared'] == 2 * 2 * 3 * 2, r
    assert not X.gate_e7(mine, E7S, 'learned')['equal']        # the learned arm differs from w = 4
    m2 = copy.deepcopy(mine); m2['panels']['val']['seeds']['3']['tiers']['cam_nf']['E3L3_342202']['bytes_median'] += 1
    assert not X.gate_e7(m2, E7S, 'w4')['equal']
    m3 = copy.deepcopy(mine); del m3['panels']['fresh']['seeds']['4']
    assert not X.gate_e7(m3, E7S, 'w4')['equal']
    assert X.gate_e7(mine, E7S, 'w4', tiers=['cam_nf'])['compared'] == 2 * 2 * 1 * 2


# ---------------------------------------------------------------- decision branches (D8-75)
def _cells(d=0.0, free=0.0, fpr=True, by=True, n=24):
    return {f'{ts}|{ns}': dict(d_furn=d, d_free=free, fpr_identical=fpr, bytes_identical=by, n_houses=n, houses_same=True)
            for ts in X.SEEDS for ns in X.GUARD_SEEDS}


def test_decide_branches():
    U = dict(novel='AMBIGUOUS', scale='AMBIGUOUS', floating='HARMFUL')
    good = dict(novel='AMBIGUOUS', scale='SAFE', floating='AMBIGUOUS')
    assert X.decide([], U, good, True, _cells(-0.001), _cells(-0.001))['verdict'] == 'LEVER'
    assert X.decide([], U, good, True, _cells(-0.003), _cells(-0.003))['verdict'] == 'LEVER'          # exactly -0.3 pp passes
    # NO: precision guard
    assert X.decide([], U, good, True, _cells(-0.0031), _cells(0.0))['verdict'] == 'NO'
    assert X.decide([], U, good, True, _cells(0.0, fpr=False), _cells(0.0))['verdict'] == 'NO'
    assert X.decide([], U, good, True, _cells(0.0, by=False), _cells(0.0))['verdict'] == 'NO'
    assert X.decide([], U, good, True, _cells(0.0, free=-1e-6), _cells(0.0))['verdict'] == 'NO'
    c = _cells(0.0); c['342202|4']['d_furn'] = -0.004
    assert X.decide([], U, good, True, c, _cells(0.0))['verdict'] == 'NO'
    assert X.decide([], U, dict(good, floating='HARMFUL'), True, _cells(-0.004), _cells(0.0))['verdict'] == 'NO'   # NO outranks AMBIGUOUS
    # AMBIGUOUS: hallucination not fixed / not-worse broken / ctrl / fresh demotion
    d = X.decide([], U, dict(good, floating='HARMFUL'), True, _cells(0.0), _cells(0.0))
    assert d['verdict'] == 'AMBIGUOUS' and d['why_not_lever'] == ['floating_not_harmful']
    assert X.decide([], U, dict(good, novel='HARMFUL'), True, _cells(0.0), _cells(0.0))['verdict'] == 'AMBIGUOUS'
    assert X.decide([], dict(U, scale='SAFE'), dict(good, scale='AMBIGUOUS'), True, _cells(0.0), _cells(0.0))['verdict'] == 'AMBIGUOUS'
    assert X.decide([], U, good, False, _cells(0.0), _cells(0.0))['verdict'] == 'AMBIGUOUS'
    d = X.decide([], U, good, True, _cells(0.0), _cells(-0.0031))
    assert d['verdict'] == 'AMBIGUOUS' and 'fresh' in d['why_not_lever'][0]
    # INCOMPLETE
    assert X.decide(['gate: learned-w ungated H1 != h1l'], U, good, True, _cells(0.0), _cells(0.0))['verdict'] == 'INCOMPLETE'
    c = _cells(0.0); del c['342101|3']
    assert X.decide([], U, good, True, c, _cells(0.0))['verdict'] == 'INCOMPLETE'
    assert X.decide([], U, good, True, _cells(0.0, n=23), _cells(0.0))['verdict'] == 'INCOMPLETE'
    assert X.decide([], U, dict(good, scale=None), True, _cells(0.0), _cells(0.0))['verdict'] == 'INCOMPLETE'
    assert X.decide([], U, good, True, _cells(0.0), {})['verdict'] == 'INCOMPLETE'


# ---------------------------------------------------------------- never overwrite, tables
def test_outputs_and_tables_guards():
    for bad in (os.path.join(X.G4, 'h1l'), os.path.join(X.G4, 'e7', 'x'), os.path.join(X.G4, 'h1g_other')):
        try:
            X.safe_out(bad)
        except SystemExit:
            continue
        raise AssertionError(f'safe_out accepted {bad}')
    X.safe_out(os.path.join(X.OUT_ROOT, 'h1', 'w4', 'gated'))
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'r.json'); opts = dict(script_sha256='a', tables_sha256='b')
        assert X._rec_done(p, opts) is False
        X._jdump(dict(script_sha256='a', tables_sha256='b'), p); assert X._rec_done(p, opts) is True
        for bad in (dict(script_sha256='a', tables_sha256='c'), dict(script_sha256='z', tables_sha256='b'), dict(script_sha256='a', tables_sha256='b', failed=True)):
            X._jdump(bad, p)
            try:
                X._rec_done(p, opts)
            except SystemExit:
                continue
            raise AssertionError(f'_rec_done accepted {bad}')
        st = np.zeros((X.LMAX + 1, X.NX), bool); st[2, 0] = st[3, 1] = True
        T = dict(batch='G4-H1G', stage='fit', smoke=True, rule=dict(n_x_bins=X.NX, lam_max=X.LMAX, q_min=0.995, n_min=200),
                 tables={ts: {wl: dict(strong=[[2, 0], [3, 1]]) for wl in X.WLABS} for ts in X.SEEDS})
        tp = os.path.join(td, 'gate_tables.json'); X._jdump(T, tp)
        _, s, luts = X.load_tables(tp)
        assert np.array_equal(luts['342101']['w4'], X.lut_of(st)) and s == X.sha(tp)
        for kw in (dict(expect_sha='0' * 64), dict(allow_smoke=False)):
            try:
                X.load_tables(tp, **kw)
            except SystemExit:
                continue
            raise AssertionError(f'load_tables accepted {kw}')


if __name__ == '__main__':
    t0 = time.time(); n = 0
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith('test_') and callable(v)):
        t = time.time(); fn(); n += 1
        print(f'ok  {name}  {time.time() - t:.1f}s', flush=True)
    print(f'all {n} tests passed in {time.time() - t0:.0f}s')
