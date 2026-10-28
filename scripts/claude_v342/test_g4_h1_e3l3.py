"""unit tests for g4_h1_e3l3.py (batch G4-H1L, D8-69). CPU only. ~1-2 min.
The only VAL data read here is one task R-I already looked at (val_r1_i0_s1, VAL noise seed 1, cam_nf) and only to reproduce the PUBLISHED
formal E3-L3 reading (D8-65); no G4-H1L VAL file is read. The prototype cross-check uses a TRAIN OOD file (R-I's scratch export).
run:  CUDA_VISIBLE_DEVICES= python -B test_g4_h1_e3l3.py"""
import os
import sys

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import copy, glob, json, shutil, tempfile, time  # noqa: E401,E402
import numpy as np  # noqa: E402
import g4_h1_e3l3 as HL  # noqa: E402

G = HL.G; R = HL.R; OP = HL.OP; L = HL.L
AN = HL.AN
EXPORT_E3R = os.path.join(AN, 'review_E3R', 'inputs', 'export')
RI = os.path.join(AN, 'review_E3L3', 'RI')
E3L3_SUMMARY = os.path.join(AN, 'g4', 'e3l3', 'summary.json')
RI_OOD = os.environ.get('RI_OOD_EXPORT', r'C:\Users\27118\AppData\Local\Temp\claude\C--Users-27118-AppData-Local-Claude-Data-claude-pzs8sxrjxfjjc-scratch-workspaces-7462828b-8825-4be3-955b-e422393bdef0-4bc58eb2-7206-4518-ad18-b22859b86201-scratch-2026-10-23-91f83a\160d1fa0-6b2e-4733-9167-0e4000fcc081\scratchpad\ri_ood_export')
KEYS = ('furn_f1', 'furn_f1_tol1', 'free_recall', 'surface_f1', 'unknown_known_fpr')
_M = None


def models():
    global _M
    if _M is None:
        _M = HL.load_models(2)
    return _M


# ---------------------------------------------------------------- formal reading reproduced
def test_ri_rows_are_the_formal_reading():
    """R-I's 24 per-task VAL s1 cam_nf E3L3 rows average to the formal D8-65 summary bit for bit (ties the per-task reference to it)"""
    S = json.load(open(E3L3_SUMMARY))
    for ts in HL.SEEDS:
        f = S['panels']['val']['seeds']['1']['tiers']['cam_nf'][f'E3L3_{ts}']['R8']
        rows = []
        for p in sorted(glob.glob(os.path.join(RI, 'out', 'val_*_1.json'))):
            rows += json.load(open(p))['tiers']['cam_nf']['R8'][f'E3L3_{ts}']['rows']
        assert len(rows) == 192
        assert float(np.mean([r['furn_f1'] for r in rows])) == f['furn_f1']
        assert float(np.mean([r['free_recall'] for r in rows])) == f['free_recall']


def test_decode_reproduces_formal_task():
    """E3-L3 decode (HL.e3l3_preds, learned w, per-hop rounding, each agent its own belief) on val_r1_i0_s1 cam_nf == the formal rows"""
    rec = json.load(open(os.path.join(EXPORT_E3R, 'records', 'val_r1_i0_s1.json')))
    path = os.path.join(EXPORT_E3R, 'val_r1_i0_s1.npz')
    assert HL.sha(path) == rec['file_sha256']
    z = np.load(path)
    H = G.House(z['occ'], z['slab'], z['obj'], z['cell']); private = z['private']; n = private.shape[0]
    F = G.forest(R.edges_list(z['edges']), n)
    R8T = [G.Target(private[cm].max(0), H, mult=(private[cm] >= 0).sum(0)) for cm in F['comps']]
    maps = list(z['L_k0_cam_nf'])
    kcs = [sum((maps[a] != HL.UNK).astype(np.int32) for a in cm) for cm in F['comps']]
    ref = json.load(open(os.path.join(RI, 'out', 'val_1_0_1.json')))['tiers']['cam_nf']['R8']
    M = models()
    for ts in HL.SEEDS:
        ep, zq, fp, info = HL.e3l3_preds(M['e3'][ts], maps, F, kcs, gbp_check=True)
        assert info['known_mismatch'] == 0 and info['gbp']['pass'] and info['w_label'] == 'learned'
        assert abs(info['w'] - {'342101': 1240.3677978515625, '342202': 1357.7166748046875}[ts]) < 1e-9
        rows = [R.scored(p, R8T[ci], H, kcs[ci]) for ci in range(len(F['comps'])) for p in ep[ci]]
        want = ref[f'E3L3_{ts}']['rows']
        assert len(rows) == len(want) == n
        for a, b in zip(rows, want):
            for k in KEYS:
                assert a[k] == b[k], (ts, k, a[k], b[k])
        assert set(np.unique(zq[fp]).tolist()) <= {-1.0, 0.0, 1.0}


def test_finite_w_needs_thresholds_and_asserts_known():
    try:
        HL.load_e3l3(1, w=4.0, w_thr=None)
        raise AssertionError('finite w without thresholds must be refused')
    except RuntimeError as e:
        assert '--w-thr' in str(e)
    d = tempfile.mkdtemp()
    try:
        p = os.path.join(d, 'thr.json')
        json.dump({'342101': {'R8': {'j_occ': 1990}}, '342202': 2001}, open(p, 'w'))
        assert HL.parse_w_thr(p) == {'342101': 1990, '342202': 2001}
        e3 = HL.load_e3l3(1, w=0.9, w_thr=HL.parse_w_thr(p))      # w < 1: the Lam message w / (w + 1) rounds to 0 -> known != k > 0
        z = np.load(os.path.join(EXPORT_E3R, 'val_r1_i0_s1.npz'))
        maps = list(z['L_k0_cam_nf'])[:2]
        F = G.forest([(0, 1, 1.0)], 2)
        kcs = [sum((maps[a] != HL.UNK).astype(np.int32) for a in (0, 1))]
        _, _, _, info = HL.e3l3_preds(e3['342101'], maps, F, kcs, gbp_check=False)
        assert info['known_mismatch'] > 0 and info['j_occ'] == 1990 and info['w_label'] == 'finite:0.9'
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- prototype cross-check (TRAIN OOD file)
def test_matches_ri_prototype_on_train_ood():
    """evaluate_file on R-I's TRAIN novel export == R-I's ri_probe E3L3 zone / family / phantom / own-pairing flip counts and rows;
    B4 / B6 rows == OP.evaluate's (the H1 code) on the same file"""
    path = os.path.join(RI_OOD, 'novel_train_r2_i19_s203.npz')
    if not os.path.exists(path):
        print('   (skipped: R-I scratch export not present)'); return
    ref = json.load(open(os.path.join(RI, 'out', 'ood_novel_2_19_203.json')))
    r = HL.evaluate_file(path, models(), gbp_check=False)
    for ts in HL.SEEDS:
        arm = f'E3L3_{ts}'; e = ref['e3'][arm]
        assert r['R8'][arm]['zones'] == e['zones']
        assert r['R8'][arm]['families'] == e['families']
        assert np.allclose(r['R8'][arm]['phantoms'], e['phantoms'], rtol=0, atol=1e-12)
        assert r['flips'][arm][f'B6_{HL.PAIR[ts]}'] == e['flips']
        for (row, w), b in zip(r['R8'][arm]['rows'], e['rows']):
            assert w == 1 and row['furn_f1'] == b['furn_f1'] and row['free_recall'] == b['free_recall']
        assert r['e3'][arm]['known_mismatch'] == 0
    op = ref['op']
    for arm in HL.CONTROLS:
        assert HL._jr(r['R8'][arm]['rows']) == op['R8'][arm]['rows']
        assert HL._jr(r['R8'][arm]['zones']) == op['R8'][arm]['zones']


# ---------------------------------------------------------------- sha and consistency gates
def _smoke_record():
    return json.load(open(os.path.join(AN, 'g4', 'h1_smoke2', 'train', 'jobs', 'ctrl_train_r1_i9.json')))


def test_gate_npz_sha_and_h1_record():
    d = tempfile.mkdtemp()
    try:
        rec = _smoke_record()
        opts = dict(out=d, export=HL.EXPORT_DIR, h1_jobs=d, threads=1, script_sha256=HL.sha(HL.__file__), w='learned', w_thr=None,
                    w_thr_sha256=None, gbp_check=False)
        job = ('ctrl', 'TRAIN', 'train', 1, 9, (203,), opts)
        bad = copy.deepcopy(rec); bad['seeds']['203']['file_sha256'] = '0' * 64
        json.dump(bad, open(os.path.join(d, 'ctrl_train_r1_i9.json'), 'w'))
        out = HL.run_job(job)
        assert out.get('failed') and 'sha' in out['error'] and 'H1 record' in out['error'], out.get('error')
        bad = copy.deepcopy(rec); bad['script_sha256'] = 'f' * 64
        json.dump(bad, open(os.path.join(d, 'ctrl_train_r1_i9.json'), 'w'))
        out = HL.run_job(job)
        assert out.get('failed') and '3a6959da' in out['error']
        out = HL.run_job(job[:-1] + (dict(opts, script_sha256='0' * 64),))
        assert out.get('failed') and 'changed since the run started' in out['error']
        os.remove(os.path.join(d, 'ctrl_train_r1_i9.json'))
        out = HL.run_job(job)
        assert out.get('failed') and 'missing' in out['error']
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_compare_controls():
    rec = _smoke_record()['seeds']['203']
    mine = dict(R8={arm: copy.deepcopy(rec['R8'][arm]) for arm in HL.CONTROLS}, components=rec['components'])
    ok, bad = HL.compare_controls(mine, rec)
    assert ok and not bad
    mine['R8']['B6_A']['rows'][0][0]['furn_f1'] += 1e-15
    mine['R8']['B4']['phantoms'][0] += 1
    ok, bad = HL.compare_controls(mine, rec)
    assert not ok and 'B6_A:rows' in bad and 'B4:phantoms' in bad


def test_frozen_and_pins():
    HL.check_frozen()
    assert HL.sha(HL.H1_SUMMARY) == HL.H1_SUMMARY_SHA256 and HL.sha(HL.FIT) == HL.FIT_SHA256
    assert {k: v for k, v in HL.CRIT.items() if not k.startswith('ctrl')} == {k: v for k, v in OP.CRIT.items() if not k.startswith('ctrl')}
    assert HL.CRIT['ctrl_lo'] == 0.043 and HL.CRIT['ctrl_hi'] == 0.086 and OP.CRIT['ctrl_lo'] == 0.053
    for p in sorted(glob.glob(os.path.join(HL.H1_JOBS, '*.json'))):            # the formal H1 records: 96 completed VAL jobs of 3a6959da
        r = json.load(open(p))
        assert r['script_sha256'] == HL.H1_SCRIPT_SHA256 and r['split'] == 'VALIDATION' and not r.get('failed') and sorted(r['seeds']) == ['1', '2']
    assert len(glob.glob(os.path.join(HL.H1_JOBS, '*.json'))) == 96


# ---------------------------------------------------------------- criteria branches
def _V(e3_all, b6_all, e3_zone, b6_zone, flip, ph_e3, ph_b6, free_e3=0.997, free_b6=0.995, b4_all=0.80, km=0, gbp=True):
    def arm(all_, zone, ph, free):
        return dict(furn_f1=all_, furn_f1_tol1=all_, surface_f1=0.9, free_recall=free, phantoms_per_house=ph, phantoms_nofloor_tall_per_house=0.0,
                    zones=dict(zone=dict(f1=zone, f1_tol1=zone), rest=dict(f1=None, fp=0, n_true=0)), families={})
    R8 = dict(B4=arm(b4_all, None, 9.0, 0.98), B6_A=arm(b6_all, b6_zone, ph_b6, free_b6), B6_B=arm(b6_all, b6_zone, ph_b6, free_b6))
    for ts in HL.SEEDS:
        R8[f'E3L3_{ts}'] = arm(e3_all, e3_zone, ph_e3, free_e3)
    fl = {f'E3L3_{ts}': {b: {'le_-0': dict(prec=flip, n=10, n_vox=10.0, prec_zone=flip, n_zone=10)} for b in ('B6_A', 'B6_B')} for ts in HL.SEEDS}
    return dict(R8=R8, flips=fl, flip_split={f'E3L3_{ts}': dict(fam={}, h={}) for ts in HL.SEEDS}, z={},
                e3={f'E3L3_{ts}': dict(known_mismatch=km, gbp_pass=gbp, gbp_cf_max=0.0, gbp_violations=0) for ts in HL.SEEDS}, house_list=['r1_i0'])


def _S(by_variant):
    return dict(variants={v: {str(s): copy.deepcopy(V) for s in (1, 2)} for v, V in by_variant.items()}, failed=[], skipped=[], foreign=[], h1_mismatch=[],
                versions={})


def test_verdict_branches():
    ctrl = _V(0.965, 0.905, None, None, 0.85, 0.5, 4.0)                    # +6.0 pp: inside [+4.3, +8.6]
    safe = _V(0.93, 0.92, 0.935, 0.92, 0.80, 1.0, 1.0)
    amb = _V(0.93, 0.92, 0.935, 0.92, 0.60, 1.0, 1.0)                     # flip .60 in [.50, .70)
    harm_flip = _V(0.93, 0.92, 0.935, 0.92, 0.45, 1.0, 1.0)
    harm_zone = _V(0.93, 0.92, 0.905, 0.92, 0.80, 1.0, 1.0)               # zone -1.5 pp
    harm_ph = _V(0.93, 0.92, 0.935, 0.92, 0.80, 3.5, 1.0)                 # phantoms > B6 + 2
    low_all = _V(0.919, 0.92, 0.935, 0.92, 0.80, 1.0, 1.0)                # overall < 0 -> not safe, not harmful
    vd = HL.verdict(_S(dict(ctrl=ctrl, novel=safe, scale=amb, floating=harm_flip)), ['r1_i0'], (1, 2))
    assert vd['verdict'] == {'novel': 'SAFE', 'scale': 'AMBIGUOUS', 'floating': 'HARMFUL'} and vd['ctrl_ok']
    vd = HL.verdict(_S(dict(ctrl=ctrl, novel=harm_zone, scale=harm_ph, floating=low_all)), ['r1_i0'], (1, 2))
    assert vd['verdict'] == {'novel': 'HARMFUL', 'scale': 'HARMFUL', 'floating': 'AMBIGUOUS'}
    vd = HL.verdict(_S(dict(ctrl=_V(0.95, 0.905, None, None, 0.85, 0.5, 4.0), novel=safe, scale=safe, floating=safe)), ['r1_i0'], (1, 2))
    assert vd['verdict'] == {'novel': 'SAFE', 'scale': 'SAFE', 'floating': 'SAFE'}       # +4.5 pp: in the new band (D8-53's +5.3 would fail)
    vd = HL.verdict(_S(dict(ctrl=_V(0.946, 0.905, None, None, 0.85, 0.5, 4.0), novel=safe, scale=safe, floating=safe)), ['r1_i0'], (1, 2))
    assert vd['verdict'].startswith('INCONCLUSIVE')                                       # +4.1 pp < +4.3
    vd = HL.verdict(_S(dict(ctrl=_V(0.995, 0.905, None, None, 0.85, 0.5, 4.0), novel=safe, scale=safe, floating=safe)), ['r1_i0'], (1, 2))
    assert vd['verdict'].startswith('INCONCLUSIVE')                                       # +9.0 pp > +8.6
    vd = HL.verdict(_S(dict(ctrl=ctrl, novel=safe, scale=safe, floating=safe)), ['r1_i0'], (1, 2), finite_w=True)
    assert vd['verdict'] == 'REPORT ONLY (finite w)' and vd['would_be'] == {'novel': 'SAFE', 'scale': 'SAFE', 'floating': 'SAFE'}
    for mut in ('failed', 'foreign', 'h1_mismatch', 'skipped'):
        S = _S(dict(ctrl=ctrl, novel=safe, scale=safe, floating=safe)); S[mut] = ['x']
        assert HL.verdict(S, ['r1_i0'], (1, 2))['verdict'] == 'INCOMPLETE', mut
    S = _S(dict(ctrl=ctrl, novel=safe, scale=safe, floating=safe)); S['versions'] = {'ckpt_sha256': ['a', 'b']}
    assert HL.verdict(S, ['r1_i0'], (1, 2))['verdict'] == 'INCOMPLETE'
    assert HL.verdict(_S(dict(ctrl=ctrl, novel=_V(0.93, 0.92, 0.935, 0.92, 0.8, 1, 1, km=3), scale=safe, floating=safe)), ['r1_i0'], (1, 2))['verdict'] == 'INCOMPLETE'
    assert HL.verdict(_S(dict(ctrl=ctrl, novel=_V(0.93, 0.92, 0.935, 0.92, 0.8, 1, 1, gbp=False), scale=safe, floating=safe)), ['r1_i0'], (1, 2))['verdict'] == 'INCOMPLETE'
    assert HL.verdict(_S(dict(ctrl=ctrl, novel=safe, scale=safe, floating=safe)), ['r1_i0', 'r2_i0'], (1, 2))['verdict'] == 'INCOMPLETE'
    S = _S(dict(ctrl=ctrl, novel=safe, scale=safe, floating=safe)); del S['variants']['scale']['2']
    assert HL.verdict(S, ['r1_i0'], (1, 2))['verdict'] == 'INCOMPLETE'
    none_flip = _V(0.93, 0.92, 0.935, 0.92, None, 1.0, 1.0)               # P3: empty flip set passes
    assert HL.verdict(_S(dict(ctrl=ctrl, novel=none_flip, scale=safe, floating=safe)), ['r1_i0'], (1, 2))['verdict']['novel'] == 'SAFE'
    low_free = _V(0.93, 0.92, 0.935, 0.92, 0.80, 1.0, 1.0, free_e3=0.985, free_b6=0.995)
    assert HL.verdict(_S(dict(ctrl=ctrl, novel=low_free, scale=safe, floating=safe)), ['r1_i0'], (1, 2))['verdict']['novel'] == 'AMBIGUOUS'
    edge = _V(0.93, 0.92, 0.91, 0.92, 0.70, 1.5, 1.0)                     # zone exactly -1 pp, flip exactly .70, phantoms exactly +0.5 -> SAFE
    assert HL.verdict(_S(dict(ctrl=ctrl, novel=edge, scale=safe, floating=safe)), ['r1_i0'], (1, 2))['verdict']['novel'] == 'SAFE'


if __name__ == '__main__':
    t0 = time.time(); n = 0
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith('test_') and callable(v)):
        t = time.time(); fn(); n += 1
        print(f'ok  {name}  {time.time() - t:.1f}s', flush=True)
    print(f'all {n} tests passed in {time.time() - t0:.0f}s')
