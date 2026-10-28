"""unit tests for g4_e3l3_w.py (batch G4-E7, D8-67). Synthetic arrays / tiny synthetic checkpoints, plus ONE real-data reconciliation on a
VAL noise-seed-1 task (val_r1_i0_s1, checkpoint 342101, cam_nf) against R-J's probes (review_E3L3/RJ) and G4-E5's Qinf row. CPU only.
Never reads noise seeds 3, 4. run:  CUDA_VISIBLE_DEVICES= python -B test_g4_e3l3_w.py"""
import os
import sys

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, tempfile  # noqa: E401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import g4_e3l3_w as M  # noqa: E402
import test_g4_e3l as TL  # noqa: E402   (helpers only: random trees / maps, fake houses; read only)

E = M.E
L = M.L
R = M.R
G = M.G
RNG = np.random.default_rng(20261028)
torch.set_num_threads(2)
RJ = os.path.join(M.AN, 'review_E3L3', 'RJ')


def _ternary_case(m, g, rng):
    Fl = G.forest(TL.rand_tree(m, rng), m)
    lam = (rng.random((m,) + g) < 0.6).astype(np.float64); lam[:, 0, 0, 0] = 0.0
    z = rng.integers(-1, 2, (m,) + g).astype(np.float64) * lam
    return Fl, torch.from_numpy(lam), torch.from_numpy(z)


# ---------------------------------------------------------------- finite w: exact GaBP, energy, known mask, bounded precision
def test_finite_w_gabp_closed_form_and_monotone():
    for w in M.W_GRID:
        for t in range(25):
            m = int(RNG.integers(1, 9))
            Fl, lam, z = _ternary_case(m, (4, 4, 3), RNG)
            c = L.gbp_checks(z.unsqueeze(1), lam, Fl, w)
            assert c['cf_max'] <= 1e-10 and c['cf_rel'] <= 1e-10 and c['violations'] == 0 and c['steps'] >= 1, (w, t, c)


def test_finite_w_chain_analytic():
    """chain 0 - 1 - 2, only agent 0 observes: Lam_b(1) = w / (w + 1), Lam_b(2) = w / (w + 2) (float GaBP; the evidence decays with tree
    distance), x_b = z_0 everywhere (one observation); the quantised wire keeps Lam_q >= 1 for w >= 1 (known preserved)"""
    Fl = G.forest([(0, 1, 1.0), (1, 2, 1.0)], 3)
    lam = torch.zeros(3, 1, 1, 1, dtype=torch.float64); lam[0] = 1.0
    z = torch.zeros_like(lam); z[0] = -1.0
    for w in M.W_GRID:
        hb, lb, _ = L.gabp((lam * z).unsqueeze(1), lam, Fl, w, None)
        assert abs(float(lb[1]) - w / (w + 1)) < 1e-12 and abs(float(lb[2]) - w / (w + 2)) < 1e-12
        x = L.belief_mean(hb, lb)
        assert torch.allclose(x[:, 0], torch.full_like(x[:, 0], -1.0))
        hq, lq, msgs = L.gabp((lam * z).unsqueeze(1), lam, Fl, w, 'round')
        assert bool((lq > 0).all()) and float(lq.max()) <= 2.0
        c = L.gbp_checks(z.unsqueeze(1), lam, Fl, w)
        assert c['cf_max'] < 1e-12 and c['violations'] == 0


def test_finite_w_known_mask_and_bounded_messages():
    """quantised wire at w in {3, 4, 6, 8}: Lam_b > 0 == k > 0 for every agent (known mask bit-identical to the learned-w / E3-R one); every
    message precision <= w (Lam_{j->i} = w Lam~ / (w + Lam~) < w); integer messages; |h| <= Lam on every message"""
    for w in M.W_GRID:
        for t in range(20):
            m = int(RNG.integers(2, 9))
            Fl, lam, z = _ternary_case(m, (5, 4, 3), RNG)
            hb, lb, msgs = L.gabp((lam * z).unsqueeze(1), lam, Fl, w, 'round')
            k = lam.sum(0)
            for a in range(m):
                assert torch.equal(lb[a] > 0, k > 0), (w, t, a)
            for hm, lm in msgs.values():
                hi, li = L.msg_ints(hm, lm)
                assert li.max() <= w and (np.abs(hi[0]) <= li).all()
            hl, ll, _ = L.gabp((lam * z).unsqueeze(1), lam, Fl, 1240.0, 'round')        # learned-w scale: the integer sums (consensus)
            assert all(torch.equal(ll[a], k) for a in range(m))


# ---------------------------------------------------------------- W2: the w* rule
def _per_seed(f1, ratio, learned=(0.977, 0.977)):
    """f1 / ratio: {label: (342101, 342202)}"""
    out = {}
    for i, sd in enumerate(('342101', '342202')):
        w = {'learned': dict(furn_f1=learned[i], bytes_ratio_median=0.33)}
        for lab in f1:
            w[lab] = dict(furn_f1=f1[lab][i], bytes_ratio_median=ratio[lab][i])
        out[sd] = dict(w=w)
    return out


def test_choose_w_rule():
    f_ok = {'w3': (.9760, .9760), 'w4': (.9762, .9765), 'w6': (.9768, .9768), 'w8': (.9769, .9769)}
    # both conditions: the largest w that meets both on both checkpoints
    ws, br, note, rows = M.choose_w(_per_seed(f_ok, {'w3': (.10, .10), 'w4': (.12, .125), 'w6': (.14, .12), 'w8': (.17, .16)}))
    assert ws == 4.0 and br == 'both' and note is None and rows['w6']['bytes_ok'] == {'342101': False, '342202': True}
    # boundaries: exactly -0.15 pp and exactly 0.13 pass
    f_b = dict(f_ok, w8=(.977 - .0015, .977 - .0015))
    ws, br, _, _ = M.choose_w(_per_seed(f_b, {'w3': (.10, .10), 'w4': (.11, .11), 'w6': (.12, .12), 'w8': (.13, .13)}))
    assert ws == 8.0 and br == 'both'
    # furniture fails at w = 8 on one checkpoint only -> 6
    f_c = dict(f_ok, w8=(.9769, .9750))
    ws, br, _, _ = M.choose_w(_per_seed(f_c, {k: (.10, .10) for k in f_ok}))
    assert ws == 6.0 and br == 'both'
    # no w meets bytes: among furniture-ok, smallest worst ratio; ties -> larger w; the note says 0.15 x is expected to be missed
    f_d = dict(f_ok, w3=(.970, .970))
    ws, br, note, _ = M.choose_w(_per_seed(f_d, {'w3': (.10, .10), 'w4': (.15, .16), 'w6': (.16, .15), 'w8': (.20, .20)}))
    assert ws == 6.0 and br == 'furniture_only' and '0.15' in note          # w4 and w6 tie at 0.16 -> larger w
    # no w meets furniture -> None, not run
    ws, br, note, _ = M.choose_w(_per_seed({k: (.97, .97) for k in f_ok}, {k: (.10, .10) for k in f_ok}))
    assert ws is None and br == 'none' and '不跑' in note


def test_official_check_rows():
    ss = dict(w={'learned': dict(R8=dict(j_occ=1998, f1=.9769, free=.9975, fpr=.019, free_min=.995, feasible_occ=True),
                                 K0=dict(j_occ=2002, f1=.938, free=.9933, fpr=.0048, free_min=.991, feasible_occ=True, k0_fallback_r8=False),
                                 bytes_ratio_median=.3277, bytes_ratio_max=.3589, bytes_median=246229.3, n=57)},
              control=dict(R8=dict(free=.99749, fpr=.019, furn_f1=.9806), K0=dict(free=.99345, fpr=.0048, furn_f1=.9447)),
              z=dict(z_minus=.08, z_zero=.06, z_plus=.86, z_entropy_bits=.76, tanh_saturation=.08, footprint=100))
    off = json.loads(json.dumps(dict(R8=dict(ss['w']['learned']['R8'], control=ss['control']['R8']), K0=dict(ss['w']['learned']['K0'], control=ss['control']['K0']),
                                     holdout_bytes=dict(median_ratio=.3277, max_ratio=.3589, median_bytes=246229.3, n=57), z=ss['z'])))
    c = M.official_check(ss, off)
    assert c['j_occ_equal'] and c['all_equal'] and c['max_abs_diff'] == 0.0
    off['R8']['f1'] = .9768; off['K0']['j_occ'] = 2001
    c = M.official_check(ss, off)
    assert not c['j_occ_equal'] and not c['all_equal'] and abs(c['max_abs_diff'] - 1.0) < 1e-12


# ---------------------------------------------------------------- W4: the D8-67 verdict
def _cell(d=-0.001, worst=-0.008, ratio=0.45, free=-0.0005, surf=-0.0005, clean=0.995, cam=-0.004, n=24):
    return dict(d_furn=d, worst_house_d=worst, worst_house='r1_i0', bytes_ratio=ratio, free_delta=free, surface_delta=surf, clean_arm_furn=clean,
                cam_d_furn=cam, n_houses=n, paired=dict(d_pooled=d, d_rowmean=d))


def _cells(**kw):
    return {f'{ts}|{ns}': _cell(**kw) for ts in ('342101', '342202') for ns in ('3', '4')}


def test_judge_branches():
    J = M.judge
    assert J(_cells(), _cells(), [])['verdict'] == 'LEVER'
    # demotions (only demote LEVER)
    v = J(_cells(cam=-0.012), _cells(), [])['verdict']
    assert v.startswith('AMBIGUOUS (LEVER demoted') and 'cam' in v
    v = J(_cells(), _cells(d=-0.0035), [])['verdict']
    assert v.startswith('AMBIGUOUS (LEVER demoted') and 'fresh' in v
    assert J(_cells(d=-0.005, cam=-0.02), _cells(d=-0.01), [])['verdict'] == 'AMBIGUOUS'        # never upgraded / never demoted further
    # LEVER boundaries (inclusive): d = -0.3 pp, worst = -2 pp, ratio = .60, free / surface = -0.3 pp, clean = .99
    assert J(_cells(d=-0.003, worst=-0.02, ratio=0.60, free=-0.003, surf=-0.003, clean=0.99), _cells(), [])['verdict'] == 'LEVER'
    # AMBIGUOUS: just outside a LEVER line but no NO line
    for kw in (dict(d=-0.0031), dict(worst=-0.021), dict(ratio=0.61)):
        assert J(_cells(**kw), _cells(), [])['verdict'] == 'AMBIGUOUS', kw
    # NO: one cell d < -1 pp, or bytes > .80, or a margin guard fails
    c = _cells(); c['342202|4'] = _cell(d=-0.011)
    assert J(c, _cells(), [])['verdict'] == 'NO'
    c = _cells(); c['342101|3'] = _cell(ratio=0.81)
    assert J(c, _cells(), [])['verdict'] == 'NO'
    for kw in (dict(free=-0.0031), dict(surf=-0.004), dict(clean=0.989)):
        c = _cells(); c['342101|4'] = _cell(**kw)
        assert J(c, _cells(), [])['verdict'] == 'NO', kw
    assert J(_cells(d=-0.01), _cells(), [])['verdict'] == 'AMBIGUOUS'                          # exactly -1 pp is not NO
    assert J(_cells(ratio=0.80), _cells(), [])['verdict'] == 'AMBIGUOUS'
    # NO is never changed by the demotion rules
    c = _cells(cam=-0.05); c['342101|3'] = _cell(d=-0.02)
    assert J(c, _cells(d=-0.02), [])['verdict'] == 'NO'
    # INCOMPLETE: missing cell, a reason from the caller, a house count != 24, a missing clean / cam row
    c = _cells(); del c['342101|4']
    assert J(c, _cells(), [])['verdict'].startswith('INCOMPLETE')
    assert J(_cells(), _cells(), ['GaBP check failed for E3L3W_342101'])['verdict'].startswith('INCOMPLETE')
    assert J(_cells(n=23), _cells(), [])['verdict'].startswith('INCOMPLETE')
    f = _cells(); del f['342202|3']
    assert J(_cells(), f, [])['verdict'].startswith('INCOMPLETE')
    c = _cells(); c['342202|3']['clean_arm_furn'] = None
    assert J(c, _cells(), [])['verdict'].startswith('INCOMPLETE')
    r = J(_cells(d=-0.002), _cells(), [])
    assert r['paired_report']['342101|3']['same_side_lever'] and abs(r['cam_mean_d'] + 0.004) < 1e-12


def test_constants_d8_67():
    C = M.D8_67
    assert (C['lever_furn_margin'], C['lever_worst_house'], C['lever_bytes_ratio']) == (-0.003, -0.020, 0.60)
    assert (C['guard_free'], C['guard_surface'], C['clean_min'], C['cf_tol']) == (-0.003, -0.003, 0.99, 1e-5)
    assert (C['cam_demote_mean'], C['fresh_demote_mean'], C['no_furn_margin'], C['no_bytes_ratio']) == (-0.010, -0.003, -0.010, 0.80)
    assert C['noise_seeds'] == ['3', '4'] and C['train_seeds'] == ['342101', '342202'] and C['houses_per_cell'] == 24
    assert tuple(M.W_GRID) == (3.0, 4.0, 6.0, 8.0) and M.W_FURN_MARGIN == 0.0015 and M.W_BYTES_MAX == 0.13 and M.MAX_WORKERS == 10
    assert M.JUDGE_SEEDS == (3, 4) and M.CONT_SEEDS == (1, 2) and M.HO_SEEDS == (200, 201, 202)
    s = M.check_frozen()
    assert s['g4_e3l3.py'].startswith('c7523cc8') and len(s) == 5


# ---------------------------------------------------------------- synthetic end-to-end: hold-out house, wselect task, eval task, aggregate
def _tiny_ckpts(d, sd='342101'):
    """a tiny E3-L3 checkpoint (non-trivial code), a tiny E3-R NET checkpoint with the same evidence table, and their threshold files"""
    tb = np.arange(R.LMIN, R.LMAX + 1)
    ev = R.Evidence(tb, 0, 0, dict(kind='L', spec='L'))
    torch.manual_seed(0)
    m = E.E3L3(4, E.W0)
    with torch.no_grad():
        m.enc.head.weight.normal_(0, 2.0); m.enc.head.bias.fill_(0.1)
    e3l3 = os.path.join(d, 'e3l3'); os.makedirs(e3l3, exist_ok=True)
    torch.save(dict(state_dict=m.state_dict(), recipe=dict(width=4, w0=E.W0, evidence=ev.recipe()), seed=int(sd)), os.path.join(e3l3, 'final.pt'))
    M._jdump(dict(R8=dict(j_occ=2000), K0=dict(j_occ=2000, k0_fallback_r8=False)), os.path.join(e3l3, 'threshold.json'))
    net = R.UNet3(3, 4, skip=True, bias0=0.0)
    e3r = os.path.join(d, 'e3r'); os.makedirs(e3r, exist_ok=True)
    torch.save(dict(state_dict=net.state_dict(), recipe=dict(width=4, cin=3, skip=True, bias0=0.0, evidence=ev.recipe()), seed=int(sd)), os.path.join(e3r, 'final.pt'))
    M._jdump(dict(j=2000, k0=dict(j_sep=2000), free=None, houses=None), os.path.join(e3r, 'threshold.json'))
    return dict(e3l3_ckpt=os.path.join(e3l3, 'final.pt'), e3l3_thr=os.path.join(e3l3, 'threshold.json'), e3r_ckpt=os.path.join(e3r, 'final.pt'),
                e3r_thr=os.path.join(e3r, 'threshold.json'))


def _write_ho(d, H, seeds=M.HO_SEEDS, tamper=None):
    ex = os.path.join(d, 'ho'); os.makedirs(os.path.join(ex, 'records'), exist_ok=True); recs = []
    for i, s in enumerate(seeds):
        priv = H.private.copy()
        if tamper == 'private' and i == 2:
            priv[0, 0, 0, 0] = 1 - max(priv[0, 0, 0, 0], 0)
        arr = dict(occ=H.occ.astype(np.int8), slab=H.slab, obj=H.obj, cell=H.cell, private=priv, edges=R.edges_array(H.edges),
                   L_k0_cam_nf=H.maps[1 + i % (len(H.maps) - 1)])
        fn = f'trainho_r1_i9_s{s}.npz'; fp = os.path.join(ex, fn); np.savez_compressed(fp, **arr)
        r = dict(stage='export', panel='trainho', rooms=1, index=9, seed=s, split='TRAIN', script_sha256='a63ce7d2' + '0' * 56, components=H.comps,
                 file=fn, file_sha256=R.file_sha256(fp))
        M._jdump(r, os.path.join(ex, 'records', fn.replace('.npz', '.json'))); recs.append(r)
    return ex, recs


def test_hohouse_layout_and_guards():
    H = TL.fake_house(n=4, edges=((0, 1), (1, 2)), noisy_draws=3, rng=np.random.default_rng(7))
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        ex, recs = _write_ho(d, H)
        Hd = M.HOHouse(ex, recs[::-1])
        assert Hd.name == 'TRAIN_r1_i9' and Hd.draws == ['clean', 'cam_nf_s200', 'cam_nf_s201', 'cam_nf_s202'] and Hd.maps[0] is None
        assert Hd.occ.dtype == bool and np.array_equal(Hd.occ, H.occ) and Hd.comps == H.comps and Hd.n == 4
        for i in range(3):
            assert np.array_equal(Hd.maps[1 + i], H.maps[1 + i])
        hr = M.ho_records(ex)
        assert list(hr) == ['TRAIN_r1_i9'] and len(hr['TRAIN_r1_i9']) == 3
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        ex, recs = _write_ho(d, H, tamper='private')
        try:
            M.HOHouse(ex, recs)
        except AssertionError as e:
            assert 'private differs' in str(e)
        else:
            raise AssertionError('a draw with different static arrays must be refused')
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        ex, recs = _write_ho(d, H)
        recs[1]['file_sha256'] = '0' * 64
        try:
            M.HOHouse(ex, recs)
        except AssertionError as e:
            assert 'sha256' in str(e)
        else:
            raise AssertionError('an export file whose sha differs from its record must be refused')


def test_wsel_task_tiny():
    H = TL.fake_house(n=4, edges=((0, 1), (1, 2), (2, 3)), noisy_draws=3, rng=np.random.default_rng(11))
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        ex, recs = _write_ho(d, H); p = _tiny_ckpts(d)
        rec = M._wsel_task((recs, dict(seed='342101', export=ex, threads=1, w_grid=list(M.W_GRID), checks=True, **p)))
        assert not rec.get('failed'), rec.get('traceback')
        assert rec['labels'] == ['learned', 'w3', 'w4', 'w6', 'w8'] and rec['draws'] == ['cam_nf_s200', 'cam_nf_s201', 'cam_nf_s202']
        assert all(v == 0 for v in rec['known_mismatch'].values())
        for lab in rec['labels']:
            g = rec['gbp'][lab]
            assert g['components'] == 3 and g['cf_max'] <= 1e-5 and g['violations'] == 0, (lab, g)
            assert rec['tot'][lab][2:] == rec['ctrl_counts']['R8'][2:4], 'FPR counts must equal the control (same known masks)'
            for dn in rec['draws']:
                assert rec['bytes'][lab][dn] > 0 and rec['ctrl_bytes'][dn] > 0
        assert rec['lam_msg_max']['w3'] <= 3 and rec['lam_msg_max']['learned'] == 3          # learned w: integer sums of the 4-agent chain
        assert rec['bytes']['w3']['cam_nf_s200'] != rec['bytes']['learned']['cam_nf_s200']
        ss = M.summarise_seed([rec])
        assert set(ss['w']) == {'learned', 'w3', 'w4', 'w6', 'w8'} and ss['w']['w4']['n'] == 3 and ss['w']['w4']['gbp']['pass']
        assert ss['w']['w4']['K0']['j_occ'] == ss['w']['learned']['K0']['j_occ'], 'K0 thresholds are w-free'
        bad = json.loads(json.dumps(rec)); bad['known_mismatch']['w4'] = 5
        try:
            M.summarise_seed([bad])
        except AssertionError as e:
            assert 'known' in str(e)
        else:
            raise AssertionError('a known-mask mismatch must stop the selection')


def _write_val_export(d, H, seed=1):
    ex = os.path.join(d, 'ex'); os.makedirs(os.path.join(ex, 'records'), exist_ok=True)
    arr = dict(occ=H.occ.astype(np.int8), slab=H.slab, obj=H.obj, cell=H.cell, private=H.private, edges=R.edges_array(H.edges))
    for tn in R.EVAL_TIERS:
        arr[f'L_k0_{tn}'] = H.maps[0] if tn == 'clean' else H.maps[1]
    arr['L_pub_cam'] = H.maps[1][:, ::-1].copy()
    fn = f'val_r1_i0_s{seed}.npz'; fp = os.path.join(ex, fn); np.savez_compressed(fp, **arr)
    rec = dict(stage='export', panel='val', rooms=1, index=0, seed=seed, split='VALIDATION', file=fn, file_sha256=R.file_sha256(fp), components=H.comps,
               e1_match=None)
    return ex, rec


def test_eval_task_tiny_and_aggregate():
    H = TL.fake_house(n=4, edges=((0, 1), (1, 2), (3, 0)), noisy_draws=1, rng=np.random.default_rng(13))
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        ex, rec_e = _write_val_export(d, H); p = _tiny_ckpts(d)
        out = os.path.join(d, 'out')
        thr_a = dict(R8=dict(j_occ=1990), K0=dict(j_occ=2000, k0_fallback_r8=False))
        opts = dict(exports={'1': ex}, out=out, seeds_ckpt=['342101'], paths={'342101': p}, thr_arm={'342101': thr_a}, w_star=4.0, tiers=list(R.EVAL_TIERS),
                    threads=1, chunk=2, script_sha256=R.file_sha256(M.__file__), criteria_sha256='c' * 64, ckpt_shas={'342101': 'x'}, smoke=True)
        rec = M._eval_and_save((rec_e, opts))
        assert not rec.get('failed'), rec.get('traceback')
        for tn in R.EVAL_TIERS:
            c = rec['checks'][tn]['342101']
            assert c['k0_identical'] and c['fpr_identical'] and c['known_mismatch_arm'] == 0 and c['known_mismatch_ctrl'] == 0 and c['net_fpr_identical']
            a = rec['tiers'][tn]['E3L3W_342101']; b = rec['tiers'][tn]['E3L3_342101']
            assert a['w'] == 4.0 and b['w'] > 900 and a['thr']['R8'] == 1990 and b['thr']['R8'] == 2000
            assert a['gbp']['violations'] == 0 and a['gbp']['cf_max'] <= 1e-5 and b['gbp']['cf_max'] <= 1e-5 and a['gbp']['components'] == 1
            assert 'strata' not in a['K0_sep'][0] and (('strata' in a['R8'][0][0]) == (tn == 'cam_nf'))
            assert len(rec['tiers'][tn]['NET_342101']['K0_sep']) == H.n and (rec['tiers'][tn]['NET_342101']['bytes'] is not None) == (tn == 'cam_nf')
        assert rec['tiers']['cam_nf']['E3L3W_342101']['bytes'] != rec['tiers']['cam_nf']['E3L3_342101']['bytes']
        try:
            M._eval_and_save((rec_e, opts))
        except AssertionError as e:
            assert 'never overwrite' in str(e)
        else:
            raise AssertionError('job files are never overwritten')
        crit = dict(script_sha256=R.file_sha256(M.__file__))
        S = M.aggregate(out, crit, 'c' * 64, smoke=True)
        assert not S['reasons'] and not S['failed'] and S['gbp']['E3L3W_342101']['pass'] and S['gbp']['E3L3_342101']['pass']
        assert all(a['k0_identical'] and a['known_mismatch'] == 0 for a in S['assertions'].values())
        D = S['panels']['val']['seeds']['1']
        cell = M.make_cell(D, '342101')
        assert cell['n_houses'] == 1 and cell['worst_house'] == 'r1_i0' and abs(cell['bytes_ratio'] - cell['report_bytes_ratio_per_house_median']) < 1e-12
        assert cell['fpr_delta'] == 0.0 and cell['clean_arm_furn'] is not None and cell['cam_d_furn'] is not None and cell['net']['bytes'] > 0
        assert abs(cell['d_furn'] - (cell['arm_R8_furn'] - cell['ctrl_R8_furn'])) < 1e-15 and 'paired' in cell
        S2 = M.aggregate(out, crit, 'd' * 64, smoke=True)
        assert any('criteria sha' in x for x in S2['reasons'])
        r = M.judge({'342101|1': cell}, {}, M.integrity_reasons(S, crit, smoke=True))
        assert r['verdict'].startswith('INCOMPLETE')


def test_eval_gate_judged_seeds():
    """judged noise seeds 3, 4 run only with the coordinator-confirmed criteria sha; the smoke never reads them; a smoke criteria file or a
    criteria file written by another script version cannot drive the real eval"""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        cp = os.path.join(d, 'criteria_w.json')
        M._jdump(dict(stage='wselect', batch='G4-E7', smoke=False, w_star=4.0, w_star_label='w4', script_sha256=R.file_sha256(M.__file__)), cp)
        sha = R.file_sha256(cp)
        for argv, msg in ((['--seeds', '3,4'], 'confirm-criteria-sha'), (['--seeds', '4', '--confirm-criteria-sha', 'f' * 64], 'confirm-criteria-sha'),
                          (['--seeds', '3', '--smoke', '--confirm-criteria-sha', sha], 'smoke never reads'), (['--seeds', '5'], 'no TEST')):
            try:
                M.main(['eval', '--criteria', cp, '--out', os.path.join(d, 'o')] + argv)
            except AssertionError as e:
                assert msg in str(e), (argv, str(e))
            else:
                raise AssertionError(f'{argv} must be refused')
        cs = os.path.join(d, 'criteria_smoke.json')
        M._jdump(dict(stage='wselect', batch='G4-E7', smoke=True, w_star=4.0, w_star_label='w4', script_sha256=R.file_sha256(M.__file__)), cs)
        try:
            M.main(['eval', '--criteria', cs, '--out', os.path.join(d, 'o'), '--seeds', '1'])
        except AssertionError as e:
            assert 'smoke criteria' in str(e)
        else:
            raise AssertionError('a smoke criteria file must not drive the real eval')
        co = os.path.join(d, 'criteria_old.json')
        M._jdump(dict(stage='wselect', batch='G4-E7', smoke=False, w_star=4.0, w_star_label='w4', script_sha256='0' * 64), co)
        try:
            M.main(['eval', '--criteria', co, '--out', os.path.join(d, 'o'), '--seeds', '1'])
        except AssertionError as e:
            assert 'sha mismatch' in str(e)
        else:
            raise AssertionError('criteria written by another script version must be refused')
        cn = os.path.join(d, 'criteria_none.json')
        M._jdump(dict(stage='wselect', batch='G4-E7', smoke=False, w_star=None, note='x', script_sha256=R.file_sha256(M.__file__)), cn)
        try:
            M.main(['eval', '--criteria', cn, '--out', os.path.join(d, 'o'), '--seeds', '1'])
        except AssertionError as e:
            assert 'not run' in str(e)
        else:
            raise AssertionError('w* = None: the batch is not run')


# ---------------------------------------------------------------- real data: learned w reproduces E3-L3 as run (R-J probes, noise seed 1 only)
def test_real_learned_w_reproduces_official_seed1():
    """val_r1_i0_s1 (VAL noise seed 1), checkpoint 342101, cam_nf, every agent decodes its own belief, official threshold j = 1998:
    learned w == R-J P4 'ACT' furniture and R-J P2 O7 bytes (R-J reproduced g4/e3l3/summary.json's per-house bytes to the bit);
    w = 4 / w = 8 with the same frozen threshold == R-J P5 'W4' / 'W8'; NET == G4-E5's Qinf row of the same task."""
    tag = 'val_r1_i0_s1'
    rec_e = M._jload(os.path.join(M.EXPORT_OLD, 'records', tag + '.json'))
    p = M.ckpt_paths('342101'); M.ckpt_shas('342101')
    thr = M._jload(p['e3l3_thr']); assert thr['R8']['j_occ'] == 1998
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        opts = dict(exports={'1': M.EXPORT_OLD}, out=d, seeds_ckpt=['342101'], paths={'342101': p}, thr_arm={'342101': thr}, w_star=4.0, tiers=['cam_nf'],
                    threads=1, chunk=2, script_sha256='x', criteria_sha256='x', ckpt_shas={}, smoke=True)
        rec = M.eval_task((rec_e, opts))
    assert not rec.get('failed'), rec.get('traceback')
    T = rec['tiers']['cam_nf']; c = T['E3L3_342101']; a = T['E3L3W_342101']; nb = T['NET_342101']
    clip = M._jload(os.path.join(RJ, 'rj_clip_342101_s1.json'))['houses'][tag + '.npz']['ACT']
    lev = M._jload(os.path.join(RJ, 'rj_lev_342101_s1_0_24.json'))['houses'][tag + '.npz']['O7']
    wfin = M._jload(os.path.join(RJ, 'rj_wfin_342101_s1.json'))['houses'][tag + '.npz']
    assert M._house_furn(c['R8']) == clip['furn'] == 0.9703763010408326, (M._house_furn(c['R8']), clip['furn'])
    assert c['bytes'] == lev == clip['bytes'] == 244812.29848869468, (c['bytes'], lev)
    assert M._house_furn(a['R8']) == wfin['W4']['furn'] and a['bytes'] == wfin['W4']['bytes'], (M._house_furn(a['R8']), a['bytes'], wfin['W4'])
    assert rec['checks']['cam_nf']['342101']['k0_identical'] and rec['checks']['cam_nf']['342101']['known_mismatch_arm'] == 0
    assert a['gbp']['cf_max'] <= 1e-5 and a['gbp']['violations'] == 0 and c['gbp']['cf_max'] <= 1e-5
    assert abs(rec['w_learned']['342101'] - 1240.3677978515625) < 1e-9
    q = M._jload(os.path.join(M.E5_JOBS, tag + '.json'))['tiers']['cam_nf']['342101']['Qinf']
    assert M._house_furn(nb['R8']) == M._house_furn(q['R8_net']), (M._house_furn(nb['R8']), M._house_furn(q['R8_net']))
    assert abs(nb['bytes']['sum'] + nb['bytes']['count'] - q['bytes']['Sk']) < 1e-6, (nb['bytes'], q['bytes']['Sk'])
    # w = 8 through the same frozen eval_arm
    z = np.load(os.path.join(M.EXPORT_OLD, rec_e['file']))
    Hh = G.House(z['occ'], z['slab'], z['obj'], z['cell']); private = z['private']; n = private.shape[0]
    F = G.forest(R.edges_list(z['edges']), n)
    K0T = [G.Target(private[k], Hh) for k in range(n)]; R8T = [G.Target(private[cm].max(0), Hh, mult=(private[cm] >= 0).sum(0)) for cm in F['comps']]
    ck, model = E.load_model(p['e3l3_ckpt'], 'cpu'); tb = torch.as_tensor(np.asarray(ck['recipe']['evidence']['table']), dtype=torch.int32)
    torch.set_num_threads(1)
    Lm = list(z['L_k0_cam_nf'])
    a8 = E.eval_arm(model, 8.0, thr, tb, Lm, Lm, F, Hh, K0T, R8T, 2)
    torch.set_num_threads(2)
    assert M._house_furn(a8['R8']) == wfin['W8']['furn'] and a8['bytes'] == wfin['W8']['bytes'], (M._house_furn(a8['R8']), a8['bytes'], wfin['W8'])


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t0 = __import__('time').time()
        t(); print('ok', t.__name__, f'{__import__("time").time() - t0:.1f}s', flush=True)
    print(f'{len(tests)}/{len(tests)} passed')
