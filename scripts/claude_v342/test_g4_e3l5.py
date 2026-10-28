"""Unit tests for batch E3-L5 (D8-76): mapping_3d_storey_data_v420div.py, g4_e3l5_gen.py, g4_e3l5.py, chain_day8_E3L5.sh, run_e3l5_gen.sh.
CPU only, no GPU, no VAL / TEST maps (TRAIN worlds only; the cache-format test reads the local 3-house smoke cache if present).
Run: CUDA_VISIBLE_DEVICES= python -B -m pytest -q scripts/claude_v342/test_g4_e3l5.py"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import g4_e3l5 as E5  # noqa: E402
import g4_e3l5_gen as GEN  # noqa: E402

G = E5.G
G.setup_env(*(x or d for x, d in zip(E5.R._HX, (G.HOUSEEXPO_JSON_DIR, G.HOUSEEXPO_INDEX))))
import mapping_3d_storey_data_v420 as W  # noqa: E402
import mapping_3d_storey_data_v420ood as WO  # noqa: E402
import mapping_3d_storey_data_v420div as D  # noqa: E402

ROOT = G.ROOT
SMOKE_CACHE = os.path.join(ROOT, 'claude_handoff', 'analysis_20261027', 'probes', 'e3l5_smoke', 'cache_e3l5')
CHAIN = os.path.join(HERE, 'chain_day8_E3L5.sh')
GENSH = os.path.join(HERE, 'run_e3l5_gen.sh')
HAVE_DATA = os.path.isdir(os.environ.get('HOUSEEXPO_JSON_DIR', ''))
needs_data = pytest.mark.skipif(not HAVE_DATA, reason='HouseExpo JSON not available')


def _div_houses(k):
    a = D.assignment()
    return [h for h in sorted(a) if a[h] == D.VARIANT][:k]


def _orig_houses(k):
    a = D.assignment()
    return [h for h in sorted(a) if a[h] == D.ORIG][:k]


# ---------------------------------------------------------------- 1. Y families never appear in train_div
@needs_data
def test_train_div_has_no_held_out_family():
    fams, kinds = {}, set()
    for r, i in _div_houses(8):
        spec = D.world_spec('TRAIN', i, r)
        for o in spec['objects']:
            assert o['kind'] not in D.Y_KINDS and o['family'] not in D.Y_FAMILIES and o['placement'] == 'floor'
            assert all(p[0] in ('box', 'cyl', 'obox') for p in o['prims']), 'ell primitive in train_div'
            fams[o['family']] = fams.get(o['family'], 0) + 1; kinds.add(o['kind'])
        st = D.family_stats('TRAIN', i, r)
        assert st['y_objects'] == 0 and st['y_voxels'] == 0 and st['y_prims'] == 0
        assert set(st['families']) <= set(D.X_FAMILIES)
    assert set(fams) == set(D.X_FAMILIES), f'not every X family drawn in 8 houses: {fams}'
    assert not kinds & {'l_shape', 'step_shape'}, 'composites are neither X nor Y'


def test_check_div_objects_rejects_every_y_kind_and_attachment():
    good = dict(kind='wall_shelf', family='wall_shelf', placement='floor', prims=[['box', 0, 0, 1, 1, 1, 1.1]])
    D.check_div_objects([good])
    for bad in (dict(good, kind='ball', family='ellipsoid'), dict(good, kind='clutter', family='clutter'), dict(good, kind='hanging', family='hanging'),
                dict(good, kind='coat_rack', family='pole'), dict(good, kind='ellipsoid', family='ellipsoid'), dict(good, placement='table_top'),
                dict(good, prims=[['ell', 1, 1, 1, .2, .2, .2]]), dict(good, family='composite'), dict(good, family='orig')):
        with pytest.raises(AssertionError):
            D.check_div_objects([good, bad])


def test_rot_table_clutter_is_dropped():
    rng = np.random.default_rng(0)
    seen_top = False
    for _ in range(40):                      # v420ood draws table-top clutter with p = .5; train_div keeps the draw but drops the primitives
        d_, w_, ag, pay = WO._plan('novel', 'rot_table', None, rng, 2.6)
        seen_top |= bool(pay['top'])
        prims, att = WO._build('novel', 'rot_table', (1, 1, 1 + w_, 1 + d_), None, False, dict(pay, top=[]), None, rng, 2.6)
        assert att == [] and prims
    assert seen_top, 'the v420ood rot_table never drew clutter (test not informative)'


# ---------------------------------------------------------------- 2. plan, held-out rule and the per-house assignment
def test_plan_and_holdout_equal_cache_e3r_v1():
    plan = D.train_plan()
    assert len(plan) == 205 and len(set(plan)) == 205
    full = [(int(s['rooms']), int(s['index'])) for s in W.scene_plan('TRAIN', 70)]
    assert set(full) - set(plan) == {(1, 26), (1, 29), (1, 46), (1, 49), (1, 66)}
    assert sum(D.is_holdout(i) for _, i in plan) == 19 and sum(not D.is_holdout(i) for _, i in plan) == 186
    assert D.is_holdout(9) == GEN.E.is_holdout(9) and D.is_holdout(19) and not D.is_holdout(10)
    assert (D.HOLDOUT_MOD, D.HOLDOUT_REM) == (GEN.E.HOLDOUT_MOD, GEN.E.HOLDOUT_REM) and D.TRAIN_PER_STRATUM == GEN.E.TRAIN_PER_STRATUM


def test_assignment_deterministic_balanced_and_index_parity_free():
    a = D.assignment(); D._assignment.cache_clear(); b = D.assignment()
    assert a == b and set(a) == set(D.train_plan())
    for rooms in (1, 2, 3):
        for ho in (False, True):
            grp = [h for h in a if h[0] == rooms and D.is_holdout(h[1]) == ho]
            assert sum(a[h] == D.VARIANT for h in grp) == len(grp) // 2
    ho = [h for h in a if D.is_holdout(h[1])]
    assert {a[h] for h in ho} == {D.VARIANT, D.ORIG}, 'held-out houses must contain both halves'
    assert sum(a[h] == D.VARIANT for h in a if not D.is_holdout(h[1])) == 92
    assert sum(a[h] == D.VARIANT for h in ho) == 8


def test_variant_of_refuses_other_splits_and_unknown_houses():
    with pytest.raises(AssertionError):
        D.variant_of('VALIDATION', 0, 1)
    with pytest.raises(AssertionError):
        D.variant_of('TEST', 0, 1)
    with pytest.raises(KeyError):
        D.variant_of('TRAIN', 26, 1)          # skipped by cache-e3r-v1
    with pytest.raises(AssertionError):
        D.world_spec('VALIDATION', 0, 1)


@needs_data
def test_identity_reproduces_v420_and_plan_is_unchanged():
    for r, i in _div_houses(1) + _orig_houses(1):
        s0 = W.world_spec('TRAIN', i, r); s1 = D.world_spec('TRAIN', i, r, identity=True)
        assert s0['world_id'] == s1['world_id'] and json.dumps(s0['objects']) == json.dumps(s1['objects'])
        r0 = W.rasterize(s0, W.DIMS['R1']); r1 = D.rasterize(s1, W.DIMS['R1'])
        assert all(np.array_equal(r0[k], r1[k]) for k in ('occ', 'slab', 'obj'))
    for r, i in _div_houses(2):
        sv = W.world_spec('TRAIN', i, r); sd = D.world_spec('TRAIN', i, r)
        assert sd['world_id'] == f"v420div:train_div:TRAIN:{sv['house_id']}" and sd['plan_world_id'] == sv['world_id']
        assert (sd['wall_h'], sd['door_h']) == (sv['wall_h'], sv['door_h'])
        rv = W.rasterize(sv, W.DIMS['R1']); rd = D.rasterize(sd, W.DIMS['R1'])
        assert np.array_equal(rv['slab'], rd['slab'])
        assert np.array_equal((rv['occ'] == 1) & ~rv['obj'], (rd['occ'] == 1) & ~rd['obj']), 'walls / slabs / lintels must be bit-identical'
        assert not np.array_equal(rv['obj'], rd['obj'])


@needs_data
def test_mixed_module_dispatch():
    m = D.module_mixed()
    assert m.make_fixture is D.make_fixture_mixed and m.world_spec is D.world_spec_mixed and m.BEAMS == W.BEAMS and m.HOPS == W.HOPS
    (ro, io), = _orig_houses(1); (rd, idv), = _div_houses(1)
    assert m.world_spec('TRAIN', io, ro)['world_id'].startswith('v420:TRAIN:')
    assert m.world_spec('TRAIN', idv, rd)['world_id'].startswith('v420div:train_div:TRAIN:')
    assert GEN.attempt_of('v420div:train_div:TRAIN:abc') == 0 and GEN.attempt_of('v420div:train_div:TRAIN:abc:a2') == 2 and GEN.attempt_of('v420:TRAIN:abc') == 0


# ---------------------------------------------------------------- 3. cache format = cache-e3r-v1 (local smoke cache)
MANIFEST_KEYS = {'format', 'script_sha256', 'noise_panels_sha256', 'data_module_sha256', 'houseexpo_index_sha256', 'seeds', 'holdout_rule', 'p_clean_at_train',
                 'houses', 'skipped', 'failed', 'n_train', 'n_holdout', 'total_bytes'}
HOUSE_KEYS = {'house', 'rooms', 'index', 'holdout', 'agents', 'components', 'file', 'file_sha256', 'file_bytes', 'world_id', 'seeds'}
NPZ = {'occ': np.int8, 'slab': np.bool_, 'obj': np.bool_, 'cell': np.float64, 'private': np.int8, 'comp_of': np.int64, 'edges': np.float64,
       'L_clean': np.int8, 'L_cam_nf_s200': np.int8, 'L_cam_nf_s201': np.int8, 'L_cam_nf_s202': np.int8}


@pytest.mark.skipif(not os.path.exists(os.path.join(SMOKE_CACHE, 'manifest.json')), reason='local smoke cache not generated')
def test_cache_format_equals_e3r():
    man = json.load(open(os.path.join(SMOKE_CACHE, 'manifest.json')))
    assert set(man) == MANIFEST_KEYS and man['format'] == GEN.E.CACHE_FORMAT and man['seeds'] == [200, 201, 202]
    assert man['holdout_rule'] == 'TRAIN index % 10 == 9' and man['p_clean_at_train'] == 0.1
    assert man['data_module_sha256'] == G.file_sha256(D.__file__) and man['script_sha256'] == G.file_sha256(GEN.__file__)
    for h in man['houses']:
        assert set(h) == HOUSE_KEYS
        assert h['world_id'].startswith(GEN.expected_world_prefix(h['rooms'], h['index']))
        with np.load(os.path.join(SMOKE_CACHE, h['file'])) as z:
            assert set(z.files) == set(NPZ)
            for k, dt in NPZ.items():
                assert z[k].dtype == dt, (k, z[k].dtype)
    hs = GEN.E.load_cache(SMOKE_CACHE, man, 'all')               # E3-R's own loader reads it (sha verified)
    assert len(hs) == len(man['houses']) and hs[0].draws == ['clean', 'cam_nf_s200', 'cam_nf_s201', 'cam_nf_s202']
    div = json.load(open(os.path.join(SMOKE_CACHE, 'e3l5_div.json')))
    assert div['manifest_sha256'] == G.file_sha256(os.path.join(SMOKE_CACHE, 'manifest.json')) and div['y_totals'] == dict(objects=0, voxels=0, prims=0)


# ---------------------------------------------------------------- 4. the wrapper only swaps the cache
def _fake_cache(tmp, ok=True):
    c = os.path.join(tmp, 'cache'); os.makedirs(c)
    json.dump(dict(format=1, houses=[], seeds=[200, 201, 202]), open(os.path.join(c, 'manifest.json'), 'w'))
    ms = G.file_sha256(os.path.join(c, 'manifest.json'))
    json.dump(dict(batch='E3-L5', manifest_sha256=ms, y_totals=dict(objects=0, voxels=0, prims=0), ok=ok, formal=True, house_set_equals_e3r=True,
                   per_variant={}, problems=[]), open(os.path.join(c, 'e3l5_div.json'), 'w'))
    return c, ms


def test_train_forwards_arguments_unchanged(monkeypatch):
    tmp = tempfile.mkdtemp()
    try:
        cache, ms = _fake_cache(tmp)
        out = os.path.join(tmp, 'run')
        seen = {}
        saved = {k: getattr(E5.L3, k) for k in ('E3L3', 'W0', 'ARM', 'D8_63', 'load_model')}

        def fake_main(argv):
            seen['argv'] = list(argv)
            seen['state'] = {k: getattr(E5.L3, k) for k in saved}
            os.makedirs(out, exist_ok=True)
            json.dump(dict(updates_done=9562, updates_planned=9562, complete=True, w=1234.5), open(os.path.join(out, 'final.json'), 'w'))
            return {}
        monkeypatch.setattr(E5.L3, 'main', fake_main)
        rest = ['--cache', cache, '--manifest-sha', ms, '--evidence', 'c:fit.json:A:cam_nf', '--seed', '342101', '--out', out, '--updates', '9562',
                '--log-every', '200', '--probe-every', '1000', '--ctrl-holdout', 'x.json', '--expect-gpu', 'RTX 5090']
        E5.main(['train', '--prereg-updates', '9562'] + rest)
        assert seen['argv'] == ['train'] + rest, 'the trainer must receive the arguments unchanged'
        assert seen['state'] == saved, 'model class / w0 / arm / criteria of g4_e3l3 must stay untouched (single variable = the cache)'
        meta = json.load(open(os.path.join(out, 'e3l5.json')))
        assert meta['cache_manifest_sha256'] == ms and meta['prereg'] == 'D8-76' and not meta['second_difference']
        assert json.load(open(os.path.join(out, 'final.json')))['e3l5']['w_final'] == 1234.5
        with pytest.raises(SystemExit):                              # never overwrite
            E5.main(['train', '--prereg-updates', '9562'] + rest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_train_refuses_wrong_cache_or_count(monkeypatch):
    tmp = tempfile.mkdtemp()
    try:
        monkeypatch.setattr(E5.L3, 'main', lambda argv: pytest.fail('trainer must not start'))
        cache, ms = _fake_cache(tmp)
        base = ['train', '--prereg-updates', '9562', '--cache', cache, '--evidence', 'L', '--seed', '342101', '--out', os.path.join(tmp, 'r')]
        with pytest.raises(AssertionError):                          # formal run without the pinned manifest sha
            E5.main(base + ['--updates', '9562'])
        with pytest.raises(AssertionError):                          # wrong pin
            E5.main(base + ['--updates', '9562', '--manifest-sha', '0' * 64])
        with pytest.raises(AssertionError):                          # more than the pre-registered count
            E5.main(base + ['--updates', '9563', '--manifest-sha', ms])
        with pytest.raises(SystemExit):                              # a second difference must be written down
            E5.main(base + ['--updates', '7000', '--manifest-sha', ms])
        with pytest.raises(AssertionError):                          # the E3-R cache (no e3l5_div.json) is refused
            c2 = os.path.join(tmp, 'e3r'); os.makedirs(c2); json.dump({}, open(os.path.join(c2, 'manifest.json'), 'w'))
            E5.check_cache(c2, None, smoke=True)
        cache_bad, ms_bad = _fake_cache(os.path.join(tmp, 'b') if os.makedirs(os.path.join(tmp, 'b')) is None else tmp, ok=False)
        with pytest.raises(AssertionError):
            E5.check_cache(cache_bad, ms_bad)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_fixed_w_loader_is_exact_and_restored():
    import torch
    tmp = tempfile.mkdtemp()
    try:
        m = E5.L3.E3L3(4, E5.L3.W0)
        p = os.path.join(tmp, 'ck.pt')
        torch.save(dict(state_dict=m.state_dict(), recipe=dict(width=4, w0=E5.L3.W0)), p)
        ck, mm = E5._load_model_fixed_w(p, 'cpu', w=4.0)
        assert float(mm.w.detach()) == 4.0
        rec = E5._sel_task_w((dict(house='TRAIN_r1_i9', file='missing.npz', file_sha256='0'), dict(ckpt=p, cache=tmp, seeds=[200], threads=1, w_fixed=4.0)))
        assert rec.get('failed') and rec['w_fixed'] == 4.0
        assert E5.L3.load_model is E5._ORIG_LOAD, 'L3.load_model must be restored'
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- 5. the D8-76 criteria (hard-coded) and their branches
def test_criteria_numbers_are_d8_76():
    C = E5.D8_76
    assert (C['c1_flip_min'], C['c1_zone_margin'], C['c1_phantom_margin'], C['c2_flip_gain']) == (0.50, -0.010, 2.0, 0.05)
    assert (C['c3_furn_margin'], C['guard_free'], C['guard_surface'], C['clean_min'], C['k0_margin'], C['c4_bytes_ratio']) == (-0.005, -0.005, -0.005, 0.99, -0.03, 0.15)
    assert C['judged_w'] == 4.0 and C['val_seeds'] == ['3', '4'] and C['h1_seeds'] == ['1', '2'] and C['train_seeds'] == ['342101', '342202']
    assert set(C['y_families']) == set(D.Y_FAMILIES) and set(C['x_families']) == set(D.X_FAMILIES) and C['h1_variants'] == ['novel', 'floating']


def _cells(flip=0.6, dzone=0.0, ph=1.0, phb=1.0, cmp_flip=0.5, dfurn=0.0, bytes_r=0.12, free=0.0, surf=0.0, fpr=0.0, clean=0.995, dk0=-0.01):
    h, v = {}, {}
    for ts in ('342101', '342202'):
        for hs, vs in (('1', '3'), ('2', '4')):
            a = dict(f1_Y=0.9 + dzone, flip_prec_Y=flip, flip_n_Y=100, flip_prec_all=flip, phantoms_per_house=ph)
            h[f'{ts}|{hs}'] = dict(arm=a, e3l3w4=dict(a, flip_prec_Y=cmp_flip), learned=dict(a), b6_f1_Y=0.9, b6_phantoms_per_house=phb,
                                   d_zone_Y=dzone)
            v[f'{ts}|{vs}'] = dict(d_furn=dfurn, bytes_ratio=bytes_r, free_delta=free, surface_delta=surf, fpr_delta=fpr, clean_furn=clean, d_K0=dk0,
                                   learned=dict(d_furn_vs_e3l3w4=dfurn, bytes_ratio=0.3))
    return v, h


def test_judge_branches():
    J = lambda **k: E5.judge(*_cells(**k), [], [])['verdict']  # noqa: E731
    assert J() == 'LEVER'
    assert J(flip=0.50, dzone=-0.010, ph=3.0, phb=1.0, cmp_flip=0.45, dfurn=-0.005, bytes_r=0.15, free=-0.005, surf=-0.005, dk0=-0.03, clean=0.99) == 'LEVER'
    assert J(flip=0.49) == 'NOT_ENOUGH'                         # criterion 1 (flip)
    assert J(dzone=-0.0101) == 'NOT_ENOUGH'                     # criterion 1 (zone)
    assert J(ph=3.01, phb=1.0) == 'NOT_ENOUGH'                  # criterion 1 (phantoms)
    assert J(cmp_flip=0.56) == 'AMBIGUOUS'                      # criterion 2
    assert J(dfurn=-0.0051) == 'AMBIGUOUS'                      # criterion 3 (furniture vs E3-L3 w = 4)
    assert J(free=-0.006) == 'AMBIGUOUS' and J(surf=-0.006) == 'AMBIGUOUS' and J(clean=0.989) == 'AMBIGUOUS' and J(dk0=-0.031) == 'AMBIGUOUS'
    assert J(fpr=1e-9) == 'AMBIGUOUS'                           # FPR must be identical
    assert J(bytes_r=0.151) == 'AMBIGUOUS'                      # criterion 4
    assert J(flip=0.49, bytes_r=0.9) == 'NOT_ENOUGH'            # 1 fails -> NOT_ENOUGH whatever else
    v, h = _cells(); assert E5.judge(v, h, [], ['342202'])['verdict'].startswith('AMBIGUOUS (lever blocked: truncated')
    v, h = _cells(flip=None); assert E5.judge(v, h, [], [])['verdict'] == 'LEVER'      # no counter-evidence flip at all: passes (D8-53 P3)
    v, h = _cells(); h['342101|1']['e3l3w4']['flip_prec_Y'] = None
    assert E5.judge(v, h, [], [])['verdict'] == 'AMBIGUOUS'      # comparator undefined, arm defined: criterion 2 not met
    v, h = _cells(); del v['342202|4']; assert E5.judge(v, h, [], [])['verdict'].startswith('INCOMPLETE')
    v, h = _cells(); assert E5.judge(v, h, ['failed jobs'], [])['verdict'].startswith('INCOMPLETE')
    v, h = _cells(); assert E5.judge(v, h, [], [], arm_key='learned')['verdict'] == 'AMBIGUOUS'   # learned w: bytes .3 > .15


def test_flip_zone_counts_equals_ood_panel():
    OP, _ = E5._h1mods()
    rng = np.random.default_rng(3); sh = (6, 5, 4)

    class T_:
        known = rng.random(sh) < .9; tr = rng.random(sh) < .3
    class H_:
        region = rng.random(sh) < .95
    p = rng.integers(-1, 2, sh).astype(np.int8); pb = rng.integers(-1, 2, sh).astype(np.int8); S = rng.integers(-5, 5, sh)
    Z = rng.random(sh) < .4
    mine = E5.flip_zone_counts(p, pb, S, T_, H_, dict(Y=Z))['Y']
    theirs = OP.flip_counts(p, pb, S, T_, H_, Z)['le_-0']
    assert mine == theirs[2:4]


def test_h1_cells_pooling():
    def arm(c, fl, ph):
        return dict(zones_y={'Y': dict(c=c, h=[[0, 0, 0]] * 3), 'f:clutter': dict(c=c, h=[[0, 0, 0]] * 3)}, flips={'le_-0': fl},
                    flips_zone={'f:clutter': fl[2:4]}, phantoms=[ph, 0, 0, 0], rows=[])
    def x(c_arm, c_b6, fl, ph, phb):
        a = {f'{n}_{ts}': arm(c_arm, fl, ph) for n in ('E3L5W4', 'E3L3W4', 'E3L5') for ts in ('342101', '342202')}
        b = {nm: dict(zones_y={'Y': dict(c=c_b6, h=[[0, 0, 0]] * 3), 'f:clutter': dict(c=c_b6, h=[[0, 0, 0]] * 3)}, phantoms=[phb, 0, 0, 0], rows=[])
             for nm in ('B4', 'B6_A', 'B6_B')}
        return dict(arms=a, controls=b)
    S = dict(files={'novel': {'1': [dict(house='r1_i0', x=x([8, 2, 2] + [0] * 6, [6, 2, 4] + [0] * 6, [5, 5, 3, 1], 1.0, 2.0))]},
                    'floating': {'1': [dict(house='r1_i0', x=x([2, 0, 0] + [0] * 6, [2, 0, 0] + [0] * 6, [1, 1, 1, 3], 3.0, 2.0))]}})
    C = dict(E5.D8_76, h1_seeds=['1'], h1_houses=1)
    cells, reasons = E5.h1_cells(S, C)
    c = cells['342101|1']
    assert not reasons and c['n_maps'] == 2
    assert abs(c['arm']['f1_Y'] - 2 * 10 / (2 * 10 + 2 + 2)) < 1e-12 and abs(c['b6_f1_Y'] - 2 * 8 / (2 * 8 + 2 + 4)) < 1e-12
    assert c['arm']['flip_prec_Y'] == 4 / 8 and c['arm']['phantoms_per_house'] == 2.0 and c['b6_phantoms_per_house'] == 2.0
    assert c['families']['clutter']['group'] == 'Y'


# ---------------------------------------------------------------- 6. chain / launcher static checks
def _bash_ok(path):
    bash = shutil.which('bash')
    if not bash:
        pytest.skip('bash not available')
    r = subprocess.run([bash, '-n', path], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_chain_static():
    _bash_ok(CHAIN)
    s = open(CHAIN, encoding='utf-8').read()
    code = '\n'.join(l for l in s.splitlines() if not l.lstrip().startswith('#'))
    assert 'pkill' not in code and 'killall' not in code and 'kill -9' not in code
    for v in ('E3L5_SHA', 'MAN5_SHA', 'DIV5_SHA', 'VER5_SHA', 'CTRLH5_SHA'):
        assert re.search(rf'^{v}=FILL_{v}\b', s, re.M), f'{v} must be a FILL_ placeholder for the coordinator'
    assert 'if grep -q -e "FILL_"' in s and '[ "$PREREG" = "D8-76" ] && [ -n "$APPROVAL" ]' in s
    assert re.search(r'^BUDGET=3000\b', s, re.M) and re.search(r'^UPD_PREREG=9562\b', s, re.M) and 'START+6600' in s and 'START+7150' in s
    assert '--max-updates 400' in s and '--time-from 200' in s and '"inclusive_seconds": 7200' in s
    assert code.index('run_seed 342101') < code.index('run_seed 342202'), 'two seeds sequentially'
    train_lines = [l for l in code.splitlines() if '$P -B scripts/g4_e3l5.py train' in l]
    assert len(train_lines) == 4 and all('--cache $CACHE5' in l and '--manifest-sha $MAN5_SHA' in l and '--ctrl-holdout $REF5/ctrl_holdout_e3l5.json' in l for l in train_lines)
    assert 'cache-e3r-v1 ' not in code and 'CACHE=' not in code, 'the E3-R cache must not be the training cache'
    for l in code.splitlines():
        if any(f'g4_e3l5.py {c}' in l for c in ('select', 'eval', 'h1', 'aggregate')):
            assert 'CUDA_VISIBLE_DEVICES= ' in l and 'nice -n 10' in l and 'timeout -k 30' in l, l
    used = set(re.findall(r'scripts/([A-Za-z0-9_]+\.py)', code))
    checked = set(re.findall(r'\(sha scripts/([A-Za-z0-9_]+\.py)\)', code))
    assert used <= checked, f'scripts run without a sha check: {used - checked}'
    for f in ('g4_ood_panel.py', 'g4_h1_e3l3.py', 'mapping_3d_storey_data_v420ood.py', 'mapping_3d_storey_data_v420div.py', 'g4_e3l3.py'):
        assert f in checked
    pins = dict(re.findall(r'^([A-Z0-9_]+_SHA)=([0-9a-f]{64})\b', s, re.M))
    assert pins['H1L_SHA'] == E5.FROZEN['g4_h1_e3l3'] and pins['OP_SHA'] == E5.FROZEN['g4_ood_panel'] and pins['OOD_SHA'] == E5.FROZEN['mapping_3d_storey_data_v420ood']
    assert pins['E3L3_SHA'] == E5.FROZEN['g4_e3l3'] and pins['GEN5_SHA'] == G.file_sha256(GEN.__file__) and pins['DIV_SHA'] == G.file_sha256(D.__file__)
    for d in re.findall(r'^(?:OUT|LOG|TMPDIR|CACHE5|REF5|REF4|REF)=(\S+)', s, re.M):
        assert d.startswith('/mnt/datasets/') or d.startswith('runs/'), d
    assert 'cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021' in s and 'bigboy' in s.lower()


def test_chain_two_leases_static():
    """the optional 3rd argument 'two-leases' (coordinator request): one seed per lease, own SOLO wait / preflight, two lease files, one count"""
    s = open(CHAIN, encoding='utf-8').read()
    code = '\n'.join(l for l in s.splitlines() if not l.lstrip().startswith('#'))
    assert 'MODE="${3:-}"' in code and 'case "$MODE" in ""|two-leases) ;;' in code
    i_if = code.index('if [ "$MODE" = "two-leases" ]; then')
    i_a, i_b = code.index('lease2 A 342101 "$EV_342101"'), code.index('lease2 B 342202 "$EV_342202"')
    i_else = code.index('\nelse\n', i_b)
    i_one = code.index('cat > $OUT/lease.json', i_else)
    i_fi = code.index('log "E3-L5 lease over after $((END-START)) s; the GPU is not used again by this chain"\nfi\n')
    i_cpu = code.index('CTRLH=$REF5/ctrl_holdout_e3l5.json')
    assert i_if < i_a < i_b < i_else < i_one < i_fi < i_cpu, 'two-leases branch, then the unchanged one-lease branch, then the shared CPU stages'
    body = code[code.index('lease2(){'):i_a]
    assert re.search(r'^BUDGET2=6000\b', s, re.M) and 'SDT=$((NOW+BUDGET2))' in body and 'DE=$((START+6600)); DT=$((START+7150))' in body
    assert 'cat > $OUT/lease_$LS.json' in body and '"user_gpu_approval_utc": "$APPROVAL"' in body and '"seed": $seed' in body and '"inclusive_seconds": 7200' in body
    assert '> $OUT/lease_end_$LS.json' in body and "open('$OUT/updates_$LS.json', 'w')" in body
    assert 'for f in lease_$LS.json updates_$LS.json lease_end_$LS.json pf-s$seed s$seed; do' in body and 'never overwrite' in body
    assert body.count('while busy; do sleep 120; done') == 2 and '--max-updates 400' in body and '--time-from 200' in body
    assert "json.load(open('$OUT/updates_A.json'))['updates']" in body and 'if [ "$UPDFIT" -ge "$UPD_A" ]; then' in body and 'INCOMPLETE' in body
    assert 'pkill' not in body and 'killall' not in body and body.count('timeout -k 30') == 2
    one = code[i_else:i_fi]                                         # the one-lease branch (default, no 3rd argument)
    assert 'run_seed 342101 "$EV_342101"\nrun_seed 342202 "$EV_342202"' in one and '"sequential": true, "per_seed_budget_seconds": $BUDGET,' in one
    assert 'lease2' not in one and 'BUDGET2' not in one


def _two_lease_harness(tmp, rate_a, rate_b):
    """run the chain's lease2() for real in bash with stubs: a fake trainer (writes final.json with the given s/update), no GPU, no sleeps"""
    s = open(CHAIN, encoding='utf-8').read().splitlines()
    a = s.index(next(l for l in s if l.startswith('lease2(){')))
    b = next(j for j in range(a, len(s)) if s[j] == '}')
    fn = '\n'.join(s[a:b + 1])
    py = sys.executable.replace('\\', '/')
    fake = os.path.join(tmp, 'fake_train.py').replace('\\', '/')
    open(fake, 'w').write(
        'import argparse, json, os, sys\n'
        'p = argparse.ArgumentParser(); [p.add_argument(x) for x in ("--out", "--updates", "--max-updates", "--seed", "--second-note")]\n'
        'a, _ = p.parse_known_args(sys.argv[3:])\n'
        'mx = int(a.max_updates or 0); n = mx if mx else int(a.updates)\n'
        'r = float(os.environ["FAKE_RATE_" + a.seed])\n'
        'os.makedirs(a.out, exist_ok=True)\n'
        'json.dump(dict(rate_s_per_update=r, load_seconds=30.0, first_window_s=20.0, probe_seconds_max=30.0, rate_cv=0.01, updates_done=n, updates_planned=n,\n'
        '               complete=True, truncated=False, train_seconds=n * r, seconds=n * r + 50, w=1000.0, second_note=a.second_note), open(os.path.join(a.out, "final.json"), "w"))\n'
        'print("fake trainer", a.seed, n)\n')
    wrap = os.path.join(tmp, 'P.sh').replace('\\', '/')
    open(wrap, 'w', newline='\n').write(f'#!/bin/bash\nif [ "$2" = "scripts/g4_e3l5.py" ]; then exec "{py}" "{fake}" "$@"; fi\nexec "{py}" "$@"\n')
    out = os.path.join(tmp, 'out').replace('\\', '/')
    os.makedirs(out)
    script = '\n'.join([
        '#!/bin/bash', f'OUT={out}', f'P="bash {wrap}"', 'UPD_PREREG=9562', 'BUDGET2=6000', 'PREREG=D8-76', 'APPROVAL=2026-10-28T12:00Z', 'E3R=runs/e3r',
        'REF5=ref5', 'CACHE5=cache5', 'MAN5_SHA=m', 'EV_342101=evA', 'EV_342202=evB', f'LOGF={out}/chain.log',
        'log(){ echo "$*" >> $LOGF; }', 'check_shas(){ :; }', 'busy(){ false; }', 'sleep(){ :; }', 'sha(){ echo x; }',
        'incomplete(){ echo "{\\"verdict\\": \\"INCOMPLETE: $1\\"}" > $OUT/summary.json; log "INCOMPLETE $1"; }',
        fn, 'lease2 A 342101 "$EV_342101"', 'lease2 B 342202 "$EV_342202"', 'echo HARNESS_DONE', 'lease2 A 342101 "$EV_342101"', 'echo NOT_REACHED'])
    sp = os.path.join(tmp, 'h.sh')
    open(sp, 'w', newline='\n').write(script + '\n')
    env = dict(os.environ, FAKE_RATE_342101=str(rate_a), FAKE_RATE_342202=str(rate_b))
    r = subprocess.run([shutil.which('bash'), sp.replace('\\', '/')], capture_output=True, text=True, env=env, timeout=600)
    return r, out


@pytest.mark.skipif(not shutil.which('bash'), reason='bash not available')
@pytest.mark.parametrize('rate_a, rate_b, exp_a, exp_b, b_incomplete', [
    (0.3526, 0.3526, 9562, 9562, False),          # both fit the pre-registered count
    (0.3526, 0.9, 9562, None, True),              # lease B cannot fit lease A's count -> fit_B, second difference, INCOMPLETE
    (0.7, 0.3526, None, None, False),             # lease A below 9,562 -> second difference; lease B reuses A's count
])
def test_chain_two_leases_dry_run(rate_a, rate_b, exp_a, exp_b, b_incomplete):
    tmp = tempfile.mkdtemp()
    try:
        r, out = _two_lease_harness(tmp, rate_a, rate_b)
        assert 'HARNESS_DONE' in r.stdout and 'NOT_REACHED' not in r.stdout and r.returncode == 2, (r.stdout, r.stderr)   # re-running lease A aborts
        la, lb = json.load(open(os.path.join(out, 'lease_A.json'))), json.load(open(os.path.join(out, 'lease_B.json')))
        assert (la['seed'], lb['seed'], la['lease'], lb['lease']) == (342101, 342202, 'A', 'B')
        assert la['user_gpu_approval_utc'] == lb['user_gpu_approval_utc'] == '2026-10-28T12:00Z' and la['inclusive_seconds'] == 7200
        assert la['per_seed_budget_seconds'] == 6000 and la['train_deadline_epoch'] - la['lease_start_epoch'] == 6600
        assert la['total_deadline_epoch'] - la['lease_start_epoch'] == 7150 and lb['lease_start_epoch'] >= la['lease_start_epoch']
        for x in ('lease_end_A.json', 'lease_end_B.json', 'pf-s342101', 'pf-s342202'):
            assert os.path.exists(os.path.join(out, x)), x
        ua, ub = json.load(open(os.path.join(out, 'updates_A.json'))), json.load(open(os.path.join(out, 'updates_B.json')))
        fa = json.load(open(os.path.join(out, 's342101', 'final.json'))); fb = json.load(open(os.path.join(out, 's342202', 'final.json')))
        if exp_a is not None:
            assert ua['updates'] == exp_a
        assert fa['updates_done'] == ua['updates'] and fb['updates_done'] == ub['updates'] and ub['updates_lease_a'] == str(ua['updates'])
        if exp_b is not None:
            assert ub['updates'] == exp_b
        if b_incomplete:
            assert ub['updates'] == ub['updates_fit'] < ua['updates'] and 'INCOMPLETE' in ub['second_difference_note'] and fb['second_note'] == ub['second_difference_note']
        else:
            assert ub['updates'] == ua['updates'] and (ub['second_difference_note'] or '') == (ua['second_difference_note'] or '')
            assert ua['second_difference'] == (ua['updates'] < 9562) and (ua['updates'] == 9562 or 'lease B uses lease A' in ua['second_difference_note'])
        assert 'exists (never overwrite)' in open(os.path.join(out, 'chain.log')).read()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_gen_launcher_static():
    _bash_ok(GENSH)
    s = open(GENSH, encoding='utf-8').read()
    code = '\n'.join(l for l in s.splitlines() if not l.lstrip().startswith('#'))
    assert 'export CUDA_VISIBLE_DEVICES=' in code and re.search(r'^WORKERS=([0-9]+)', s, re.M) and int(re.search(r'^WORKERS=([0-9]+)', s, re.M).group(1)) <= 9
    assert 'pkill' not in code and 'killall' not in code
    for d in re.findall(r'^(?:CACHE|REF5|LOG|TMPDIR)=(\S+)', s, re.M):
        assert d.startswith('/mnt/datasets/') or d.startswith('runs/'), d
    assert 'manifest.json ] && { log "ABORT' in s, 'must refuse an existing cache'
    assert GEN.MAX_WORKERS == 9
