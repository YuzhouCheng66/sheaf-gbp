"""unit tests for g4_e6_e3l3.py (batch G4-E6L, D8-70). CPU only. ~1 min.
The only VAL data read here is val_r1_i0_s1 (VAL noise seed 1, already looked at by R-I) to reproduce the PUBLISHED formal E3-L3 reading
(D8-65), and frozen G4-E6 job records (published, D8-60) for the gate tests. No G4-E6L task is run here (the smoke does that).
run:  CUDA_VISIBLE_DEVICES= python -B test_g4_e6_e3l3.py"""
import os
import sys

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import copy, glob, json, shutil, tempfile, time  # noqa: E401,E402
import numpy as np  # noqa: E402
import g4_e6_e3l3 as X  # noqa: E402

HL = X.HL; OI = X.OI; G = X.G; E = X.E
AN = X.A27
EXPORT_E3R = os.path.join(AN, 'review_E3R', 'inputs', 'export')
RI = os.path.join(AN, 'review_E3L3', 'RI')
KEYS = ('furn_f1', 'furn_f1_tol1', 'free_recall', 'surface_f1', 'unknown_known_fpr')


# ---------------------------------------------------------------- formal reading reproduced through this script's read-out
def test_e3l3_rows_reproduce_formal_task():
    """X.e3l3_rows (k from the maps, each agent its own belief) on val_r1_i0_s1 cam_nf == the formal rows (R-I's per-task rows, whose
    24-task mean equals the D8-65 summary bit for bit: see test_g4_h1_e3l3.test_ri_rows_are_the_formal_reading)"""
    rec = json.load(open(os.path.join(EXPORT_E3R, 'records', 'val_r1_i0_s1.json')))
    path = os.path.join(EXPORT_E3R, 'val_r1_i0_s1.npz')
    assert HL.sha(path) == rec['file_sha256']
    z = np.load(path)
    H = G.House(z['occ'], z['slab'], z['obj'], z['cell']); private = z['private']; n = private.shape[0]
    F = G.forest(E.edges_list(z['edges']), n)
    R8T = [G.Target(private[cm].max(0), H, mult=(private[cm] >= 0).sum(0)) for cm in F['comps']]
    maps = list(z['L_k0_cam_nf'])
    ref = json.load(open(os.path.join(RI, 'out', 'val_1_0_1.json')))['tiers']['cam_nf']['R8']
    RO = OI.load_readouts()
    e3 = HL.load_e3l3(2)
    for ts in X.SEEDS:
        assert np.array_equal(e3[ts]['table'], RO['b6'][f'B6_{X.PAIR[ts]}'].table)
        rows, info = X.e3l3_rows(e3[ts], maps, F, R8T, H, gbp_check=True)
        assert info['known_mismatch'] == 0 and info['gbp']['pass']
        want = ref[f'E3L3_{ts}']['rows']
        assert len(rows) == len(want) == n and all(w == 1 for _, w in rows)
        for (a, _), b in zip(rows, want):
            for k in KEYS:
                assert a[k] == b[k], (ts, k)
    # the frozen NET / B6 read-outs on the same maps reproduce R-I's rows too (same pipeline as the G4-E6 gate)
    out = OI.readouts(maps, F, R8T, H, RO, G.RelayStats())
    for ro in ('NET_342101', 'NET_342202', 'B6_A', 'B6_B'):
        exp = G._expand(out[ro])
        assert [r['furn_f1'] for r in exp] == [r['furn_f1'] for r in ref[ro]['rows']], ro


# ---------------------------------------------------------------- sha pins and the consistency gate
def test_pins_and_manifests():
    X.check_frozen()
    assert len(OI.pinned_e4_manifest()) == 96
    m = X.pinned_e6_manifest()
    assert len(m) == 96 and set(m) == set(OI.expected_jobs(X.PANELS, X.NOISE_SEEDS)[0])
    for p in sorted(glob.glob(os.path.join(X.E6_JOBS, '*.json')))[:5]:
        assert json.load(open(p))['script_sha256'] == X.E6_SCRIPT_SHA256
    d = tempfile.mkdtemp()
    try:
        for p in sorted(glob.glob(os.path.join(X.E6_JOBS, '*.json')))[:3]:
            shutil.copy(p, d)
        _, s = X.e6_manifest(d)
        assert s != X.E6_MANIFEST_SHA256
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_task_refuses_wrong_job_sha():
    opts = dict(out=tempfile.mkdtemp(), threads=1, script_sha256=HL.sha(X.__file__), hl_sha256=HL.sha(HL.__file__), w='learned', w_thr=None,
                w_thr_sha256=None, gbp_check=False, e4_sha256=OI.pinned_e4_manifest(), e6_sha256=X.pinned_e6_manifest())
    try:
        bad = dict(opts, e6_sha256=dict(opts['e6_sha256'], val_r1_i0_s1='0' * 64))
        try:
            X.run_task(('val', 1, 0, 1, bad)); raise AssertionError('must refuse')
        except RuntimeError as e:
            assert 'G4-E6 job val_r1_i0_s1 sha' in str(e)
        bad = dict(opts, e4_sha256=dict(opts['e4_sha256'], val_r1_i0_s1='0' * 64))
        try:
            X.run_task(('val', 1, 0, 1, bad)); raise AssertionError('must refuse')
        except RuntimeError as e:
            assert 'E4 job val_r1_i0_s1 sha' in str(e)
        try:
            X.run_task(('val', 1, 0, 1, dict(opts, script_sha256='0' * 64))); raise AssertionError('must refuse')
        except RuntimeError as e:
            assert 'changed since the run started' in str(e)
        try:
            X.run_task(('val', 1, 0, 1, dict(opts, hl_sha256='0' * 64))); raise AssertionError('must refuse')
        except RuntimeError as e:
            assert 'g4_h1_e3l3.py changed' in str(e)
    finally:
        shutil.rmtree(opts['out'], ignore_errors=True)


def test_gate_rows():
    j = json.load(open(os.path.join(X.E6_JOBS, 'val_r2_i3_s1.json')))
    theirs = j['conds']['cam']['arms']['e6']['aligned']
    mine = copy.deepcopy(theirs)
    eq, diag = X._gate_rows(mine, theirs)
    assert all(eq.values()) and not diag and set(eq) == set(X.GATED)
    mine['NET_342202'][0][0]['furn_f1'] += 1e-12
    mine['B6_A'][0][0]['free_recall'] = None
    eq, diag = X._gate_rows(mine, theirs)
    assert not eq['NET_342202'] and not eq['B6_A'] and eq['NET_342101'] and eq['B4']
    assert diag['NET_342202']['n_unequal_comp'] == 1 and diag['B6_A']['max_abs']['free_recall'] == 'none_mismatch'
    eq, _ = X._gate_rows(mine, None)
    assert not any(eq.values())


# ---------------------------------------------------------------- aggregation arithmetic and the D8-70 branches
def _row(f1, free=0.99, tp=90, fp=5, fn=5):
    return dict(furn_f1=f1, free_recall=free, furn_tp=tp, furn_fp=fp, furn_fn=fn)


def _job(name, e3, net, e3_noreg, net_noreg, tp_e3, tp_net, n_isl=2, free_e3=0.99, free_net=0.99):
    """one fake task: 2 components of 2 agents; NET rows per component [row, 2], E3L3 rows per agent [row, 1]"""
    def comp(v, free):
        return [[_row(v, free), 2], [_row(v, free), 2]]

    def agents(v, free):
        return [[_row(v, free), 1] for _ in range(4)]
    arms = {}
    for cond, arm, ve, vn in (('cam', 'e6', e3, net), ('cam', 'noreg', e3_noreg, net_noreg), ('cam_nf', 'noreg', tp_e3, tp_net)):
        al = {}
        for ts in X.SEEDS:
            al[f'E3L3_{ts}'] = agents(ve, free_e3 if (cond, arm) == ('cam', 'e6') else 0.99)
            al[f'NET_{ts}'] = comp(vn, free_net if (cond, arm) == ('cam', 'e6') else 0.99)
            al[f'B6_{X.PAIR[ts]}'] = comp(vn - 0.07, 0.98)
        arms.setdefault(cond, {})[arm] = dict(aligned=al)
    return dict(job=name, conds=dict(cam=dict(arms=arms['cam'], n_islands=n_isl), cam_nf=dict(arms=arms['cam_nf'])), timing=dict(total=1.0))


def test_cell_stats():
    J = [_job('val_r1_i0_s1', 0.88, 0.89, 0.80, 0.81, 0.96, 0.97, n_isl=1), _job('val_r1_i1_s1', 0.86, 0.89, 0.79, 0.81, 0.96, 0.97, n_isl=4)]
    c = X.cell_stats(J, '342101')
    assert abs(c['e3_e6'] - 0.87) < 1e-12 and abs(c['net_e6'] - 0.89) < 1e-12 and abs(c['d_pp'] + 2.0) < 1e-9
    assert abs(c['comp_eff_pp'] - (0.87 - 0.795) * 100) < 1e-9
    assert c['n_harm'] == 1 and c['harm_houses'] == ['val_r1_i1_s1']             # -3 pp < -2 pp; the -1 pp house is not harmed
    assert abs(c['island_loss_pp']['E3L3_342101'] - (0.96 - 0.87) * 100) < 1e-9 and abs(c['island_loss_pp']['NET_342101'] - 8.0) < 1e-9
    assert c['by_islands'] == {'1': dict(houses=1, mean_d_pp=c['per_house_pp']['val_r1_i0_s1']), '4': dict(houses=1, mean_d_pp=c['per_house_pp']['val_r1_i1_s1'])}
    p = c['paired_e6']
    assert p['n_rows'] == 8 and p['n_rows_both_defined'] == 8 and p['pooled_d_pp'] == 0.0     # same fake counts on both sides
    rows_a = [_row(None, tp=0, fp=0, fn=0), _row(0.5, tp=1, fp=1, fn=1)]; rows_b = [_row(0.9, tp=9, fp=1, fn=1), _row(0.6, tp=3, fp=1, fn=1)]
    q = X.paired(rows_a, rows_b)
    assert q['n_rows_both_defined'] == 1 and abs(q['paired_row_d_pp'] + 10.0) < 1e-9 and abs(q['a']['f1'] - 0.5) < 1e-12 and abs(q['b']['f1'] - 24 / 28) < 1e-12


def _cell(d, comp=7.0, free=0.0, harm=0):
    return dict(d_pp=d, comp_eff_pp=comp, free_d_pp=free, n_harm=harm, harm_houses=[], per_house_pp={})


def _cells(val_d, fresh_d, **kw):
    out = {}
    for panel, ds in (('val', val_d), ('fresh', fresh_d)):
        out[panel] = {str(s): {ts: _cell(ds[k], **kw) for k, ts in enumerate(X.SEEDS)} for s in X.NOISE_SEEDS}
    return out


def test_verdict_branches():
    assert X.verdict(_cells([-0.5, -0.9], [-0.5, -0.5]), [])['verdict'] == 'NONINFERIOR'
    assert X.verdict(_cells([-1.0, -0.2], [-0.5, -0.5]), [])['verdict'] == 'NONINFERIOR'            # exactly -1.0 passes
    assert X.verdict(_cells([-1.2, -0.5], [-0.5, -0.5]), [])['verdict'] == 'AMBIGUOUS'
    assert X.verdict(_cells([-2.1, -0.5], [-0.5, -0.5]), [])['verdict'] == 'NOT_NONINFERIOR'
    v = X.verdict(_cells([-0.5, -0.5], [-1.5, -1.5]), [])
    assert v['verdict'] == 'AMBIGUOUS' and v['notes'] and not v['fresh_ok']                          # fresh only demotes
    assert X.verdict(_cells([-2.5, -0.5], [-0.5, -0.5], comp=7.0), [])['verdict'] == 'NOT_NONINFERIOR'
    assert X.verdict(_cells([-0.5, -0.5], [-0.5, -0.5], comp=2.9), [])['verdict'] == 'AMBIGUOUS'      # composition not effective
    assert X.verdict(_cells([-0.5, -0.5], [-0.5, -0.5], free=-0.6), [])['verdict'] == 'AMBIGUOUS'
    assert X.verdict(_cells([-0.5, -0.5], [-0.5, -0.5], harm=4), [])['verdict'] == 'AMBIGUOUS'
    assert X.verdict(_cells([-0.5, -0.5], [-0.5, -0.5], harm=3), [])['verdict'] == 'NONINFERIOR'
    c = _cells([-0.5, -0.5], [-0.5, -0.5]); c['fresh']['1']['342101']['free_d_pp'] = -5.0; c['fresh']['2']['342202']['n_harm'] = 9
    assert X.verdict(c, [])['verdict'] == 'NONINFERIOR'                                                # fresh guards are report only
    v = X.verdict(_cells([-0.5, -0.5], [-0.5, -0.5]), [], finite_w=True)
    assert v['verdict'] == 'REPORT ONLY (finite w)' and v['would_be'] == 'NONINFERIOR'
    assert X.verdict(_cells([-0.5, -0.5], [-0.5, -0.5]), ['3 gate_failed'])['verdict'] == 'INCOMPLETE'
    c = _cells([-0.5, -0.5], [-0.5, -0.5]); c['val']['2']['342202']['d_pp'] = None
    assert X.verdict(c, [])['verdict'] == 'INCOMPLETE'
    assert X.CRIT == dict(noninf_pp=-1.0, comp_eff_pp=3.0, free_guard_pp=-0.5, harm_pp=-2.0, harm_max=3, fresh_mean_pp=-1.0, not_noninf_pp=-2.0)


def test_aggregate_blocks():
    """aggregate on fake task files: gate failure, known mismatch, GaBP failure, missing task -> INCOMPLETE reasons"""
    d = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(d, 'jobs'))
        names = ['val_r1_i0_s1', 'val_r1_i1_s1']
        for k, nm in enumerate(names):
            j = _job(nm, 0.88, 0.89, 0.80, 0.81, 0.96, 0.97)
            j.update(script_sha256='s', hl_sha256='h', frozen={}, w='learned', w_thr_sha256=None, gbp_check=True, readouts={}, e3l3={},
                     gate=dict(ok=(k == 0), problems=([] if k == 0 else ['cam/e6: aligned rows != G4-E6 job'])))
            for cond in ('cam', 'cam_nf'):
                for arm, a in j['conds'][cond]['arms'].items():
                    a['e3'] = {f'E3L3_{ts}': {'known_mismatch': (5 if (k == 1 and cond == 'cam_nf') else 0), 'gbp': {'pass': k == 0}}
                               for ts in X.SEEDS}
            json.dump(j, open(os.path.join(d, 'jobs', nm + '.json'), 'w'))
        S = X.aggregate(d, smoke=True, only=names + ['val_r1_i2_s1'])
        r = ' '.join(S['verdict']['reasons'])
        assert S['verdict']['verdict'] == 'INCOMPLETE' and 'gate_failed' in r and 'known_mismatch' in r and 'gbp_failed' in r and 'missing' in r
        assert 'val' in S['cells'] and abs(S['cells']['val']['1']['342101']['d_pp'] + 1.0) < 1e-9
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == '__main__':
    t0 = time.time(); n = 0
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith('test_') and callable(v)):
        t = time.time(); fn(); n += 1
        print(f'ok  {name}  {time.time() - t:.1f}s', flush=True)
    print(f'all {n} tests passed in {time.time() - t0:.0f}s')
