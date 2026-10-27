"""unit tests for mapping_3d_storey_data_v420ood.py and g4_ood_panel.py (batch G4-H1, D8-53). CPU only; TRAIN houses only (VAL is
the verdict panel and is never touched here). ~1-2 min.
run:  CUDA_VISIBLE_DEVICES= python -B test_g4_ood_panel.py"""
import os
import sys

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math, time  # noqa: E401,E402
import numpy as np  # noqa: E402
import g4_ood_panel as P  # noqa: E402

G = P.G; E = P.E; WO = P.WO; W = WO.W
RNG = np.random.default_rng(20261027)
TRAIN_HOUSES = [('TRAIN', 9, 1), ('TRAIN', 9, 2), ('TRAIN', 19, 3)]


def toy_spec(L=8.0):
    return dict(chart_m=L)


def mask(p, dims=(64, 64, 32), L=8.0):
    return WO.prim_mask(p, toy_spec(L), dims)


# ---------------------------------------------------------------- generator
def test_identity_reproduces_v420():
    for s, i, r in TRAIN_HOUSES:
        a = W.world_spec(s, i, r); b = WO.world_spec(WO.IDENTITY, s, i, r)
        assert a['objects'] == b['objects'] and a['world_id'] == b['world_id'] and a['wall_h'] == b['wall_h'] and a['door_h'] == b['door_h']
        ra, rb = W.rasterize(a, (128, 128, 32)), WO.rasterize(b, (128, 128, 32))
        for k in ('occ', 'slab', 'obj'):
            assert np.array_equal(ra[k], rb[k]), k


def test_plan_bitwise_identical_across_variants():
    for s, i, r in TRAIN_HOUSES:
        ref = W.rasterize(W.world_spec(s, i, r), (128, 128, 32)); plan = ref['occ'].astype(bool) & ~ref['obj']
        for v in WO.VARIANTS:
            sp = WO.world_spec(v, s, i, r); rr = WO.rasterize(sp, (128, 128, 32))
            assert np.array_equal(rr['slab'], ref['slab']), (v, 'slab')
            assert np.array_equal(rr['occ'].astype(bool) & ~rr['obj'], plan), (v, 'walls / lintels / slabs differ')
            assert sp['wall_h'] == W.world_spec(s, i, r)['wall_h'] and sp['door_h'] == W.world_spec(s, i, r)['door_h']
            assert sp['verts_m'] == W.world_spec(s, i, r)['verts_m']


def test_world_id_and_independent_rng():
    s, i, r = TRAIN_HOUSES[1]
    base = W.world_spec(s, i, r)
    specs = {v: WO.world_spec(v, s, i, r) for v in WO.VARIANTS}
    for v, sp in specs.items():
        assert sp['world_id'] == f"v420ood:{v}:{base['split']}:{base['house_id']}" and sp['variant'] == v and sp['plan_world_id'] == base['world_id']
        assert WO.world_spec(v, s, i, r)['objects'] == sp['objects'], 'not deterministic'
    # ctrl uses the original library with a NEW world seed: different furniture from v420
    assert [o['prims'] for o in specs['ctrl']['objects']] != [o['prims'] for o in base['objects']]
    assert {o['kind'] for o in specs['ctrl']['objects']} <= set(W.SIZES_KIND)
    # the furniture rng follows the world id (independent streams)
    seeds = {v: W._seed(sp['world_id'] + ':geometry') for v, sp in specs.items()}
    assert len(set(seeds.values())) == 4 and W._seed(base['world_id'] + ':geometry') not in seeds.values()


def test_variant_object_classes():
    new = set(WO.NEW_SIZES)
    for s, i, r in TRAIN_HOUSES:
        nv = WO.world_spec('novel', s, i, r)['objects']
        assert nv and {o['kind'] for o in nv} <= new - {'hanging'} and all(o['ood'] for o in nv)
        for o in nv:
            if o['kind'].startswith('rot_'):
                assert 15.0 <= o['angle_deg'] <= 75.0 and all(p[0] in ('obox', 'cyl') for p in o['prims'])
            if o['kind'] == 'clutter':
                for p in o['prims']:
                    assert p[0] == 'obox' and 0.10 - 1e-9 <= 2 * p[3] <= 0.25 + 1e-9 and 0.10 - 1e-9 <= 2 * p[4] <= 0.25 + 1e-9 and 0.10 - 1e-9 <= p[7] - p[6] <= 0.25 + 1e-9
            if o['kind'] == 'coat_rack':
                assert 0.010 <= o['prims'][1][3] <= 0.015 and o['prims'][1][0] == 'cyl'
            if o['kind'] == 'wall_shelf':
                assert 1.0 <= min(p[3] for p in o['prims']) <= 1.9 + 1e-9
        fl = WO.world_spec('floating', s, i, r)
        assert fl['objects'] and {o['kind'] for o in fl['objects']} <= {'wall_shelf', 'hanging'}
        for o in fl['objects']:
            z0 = min((p[3] if p[0] == 'box' else p[4] if p[0] == 'cyl' else p[6] if p[0] == 'obox' else p[3] - p[6]) for p in o['prims'])
            assert z0 >= 1.0 - 1e-9, (o['kind'], z0)
        rr = WO.rasterize(fl, (128, 128, 32)); hz = (np.arange(32) + .5) * 0.1 - W.FLOOR_T
        assert not rr['obj'][:, :, hz < 0.95].any(), 'a floating object touches the lower 0.95 m'
        sc = WO.world_spec('scale', s, i, r)['objects']
        assert sc and all(o['scale'] in (0.6, 1.5) and o['ood'] for o in sc) and {o['kind'] for o in sc} <= set(W.SIZES_KIND)
        ct = WO.world_spec('ctrl', s, i, r)['objects']
        assert not any(o['ood'] for o in ct)


def test_scale_is_uniform_about_anchor():
    rng = np.random.default_rng(3)
    fp = (2.0, 1.0, 2.0 + 1.2 * 1.5, 1.0 + 0.5 * 1.5)                     # scaled footprint of a wall-backed cabinet on y0
    prims, _ = WO._build('scale', 'cabinet', fp, 'y0', True, None, 1.5, rng, 2.7)
    x0, y0, x1, y1 = WO.footprint(prims)
    assert abs(x0 - fp[0]) < 1e-9 and abs(x1 - fp[2]) < 1e-9 and abs(y0 - fp[1]) < 1e-9 and abs(y1 - fp[3]) < 1e-9
    h = prims[0][6]; assert 1.5 * 0.75 - 1e-9 <= h <= min(1.5 * 0.95, 2.7) + 1e-9
    fp = (2.0, 1.0, 2.0 + 0.9 * 1.5, 1.0 + 0.6 * 1.5)                     # wardrobe 1.8-2.1 m x 1.5 is clipped at the ceiling
    prims, _ = WO._build('scale', 'wardrobe', fp, 'y0', True, None, 1.5, rng, 2.5)
    assert prims[0][6] == 2.5


# ---------------------------------------------------------------- primitives
def test_obox_theta0_equals_box_and_90_swaps():
    dims = (64, 64, 32); c = 8.0 / 64
    for _ in range(300):
        cx, cy = RNG.uniform(1, 7, 2); hx, hy = RNG.choice([RNG.uniform(0.005, 0.06), RNG.uniform(0.07, 1.2)], 2)
        z0 = RNG.uniform(0, 1.5); z1 = z0 + RNG.choice([0.02, RNG.uniform(0.05, 1.0)])
        box = ('box', cx - hx, cy - hy, z0, cx + hx, cy + hy, z1)
        a = mask(box, dims); b = mask(('obox', cx, cy, hx, hy, 0.0, z0, z1), dims)
        if not np.array_equal(a, b):                                      # only exact cell-boundary ties may differ
            fx = [(cx - hx) / c, (cx + hx) / c, cx / c, (cy - hy) / c, (cy + hy) / c, cy / c]
            assert any(abs(v - round(v * 2) / 2) < 1e-9 for v in fx), (box, int(a.sum()), int(b.sum()))
        b90 = mask(('obox', cx, cy, hy, hx, math.pi / 2, z0, z1), dims)
        assert abs(int(b90.sum()) - int(a.sum())) <= max(2, 0.02 * a.sum()), (box, int(a.sum()), int(b90.sum()))


def test_obox_count_matches_area():
    dims = (128, 128, 32); L = 16.0; c = L / 128
    errs = []
    for _ in range(200):
        hx, hy = RNG.uniform(0.3, 1.2), RNG.uniform(0.3, 1.2); th = RNG.uniform(0, math.pi)
        cx, cy = RNG.uniform(4, 12, 2)
        m = WO.prim_mask(('obox', cx, cy, hx, hy, th, 0.5, 0.6), toy_spec(L), dims)[:, :, :].any(2)
        n = int(m.sum()); A = 4 * hx * hy; per = 4 * (hx + hy)
        assert abs(n * c * c - A) <= per * c, (n, A / c / c)               # within one cell layer of the boundary
        errs.append(n - A / (c * c))
    assert abs(np.mean(errs)) <= 1.0, np.mean(errs)                      # unbiased to within one cell on average
    assert np.max(np.abs(errs)) <= 0.5 * (4 * (1.2 + 1.2)) / c


def test_thin_part_rule():
    dims = (64, 64, 32); c = 8.0 / 64
    # both axes thinner than a cell: exactly the cell containing the centre
    m = mask(('obox', 3.21, 4.07, 0.01, 0.02, 0.6, 0.30, 0.31), dims)
    assert m.sum() == 1 and m[int(3.21 // c), int(4.07 // c), int((W.FLOOR_T + 0.305) // 0.1)]
    # a 2 cm plate 1.5 m long at 37 deg: a continuous line of cells crossing the medial line, at least length / cell of them
    th = math.radians(37); m2 = mask(('obox', 4.0, 4.0, 0.75, 0.01, th, 1.0, 1.5), dims).any(2)
    lab, nl = __import__('scipy').ndimage.label(m2, structure=np.ones((3, 3), bool))
    assert nl == 1 and m2.sum() >= 1.5 / c
    xs, ys = np.nonzero(m2); u = ((xs + .5) * c - 4) * math.cos(th) + ((ys + .5) * c - 4) * math.sin(th)
    v = -((xs + .5) * c - 4) * math.sin(th) + ((ys + .5) * c - 4) * math.cos(th)
    assert np.all(np.abs(v) <= 0.5 * c * (abs(math.cos(th)) + abs(math.sin(th))) + 1e-9) and np.all(np.abs(u) <= 0.75 + 1e-9)
    # thin in z only: one z layer (v420 rule)
    m3 = mask(('obox', 4.0, 4.0, 0.5, 0.3, 0.3, 1.0, 1.02), dims)
    assert len(np.unique(np.nonzero(m3)[2])) == 1
    # v420 cylinder pole of 2-3 cm keeps one column
    m4 = mask(('cyl', 4.03, 3.97, 0.012, 0.0, 1.8), dims)
    assert m4.any(2).sum() == 1
    # ellipsoids: a tiny one -> the centre cell; flat in z -> one layer; a sphere = brute-force cell-centre inclusion
    m5 = mask(('ell', 4.01, 3.99, 0.5, 0.03, 0.03, 0.03), dims)
    assert m5.sum() == 1 and m5[int(4.01 // c), int(3.99 // c), int((W.FLOOR_T + 0.5) // 0.1)]
    m6 = mask(('ell', 4.0, 4.0, 1.0, 0.5, 0.4, 0.02), dims)
    assert len(np.unique(np.nonzero(m6)[2])) == 1 and m6.sum() > 0.5 * math.pi * 0.5 * 0.4 / c / c
    m7 = mask(('ell', 4.02, 3.96, 0.45, 0.45, 0.45, 0.45), dims)
    X, Y, Z = np.meshgrid((np.arange(64) + .5) * c, (np.arange(64) + .5) * c, (np.arange(32) + .5) * 0.1, indexing='ij')
    ref = ((X - 4.02) ** 2 + (Y - 3.96) ** 2 + (Z - W.FLOOR_T - 0.45) ** 2) <= 0.45 ** 2
    assert np.array_equal(m7, ref)
    assert abs(m7.sum() * c * c * 0.1 - 4 / 3 * math.pi * 0.45 ** 3) < 0.1 * 4 / 3 * math.pi * 0.45 ** 3


def test_footprint_and_rotation():
    prims = WO.rotate_local([('box', -1.0, -0.5, 0.0, 1.0, 0.5, 0.5)], math.radians(30))
    x0, y0, x1, y1 = WO.footprint(prims)
    ex = math.cos(math.radians(30)) * 1.0 + math.sin(math.radians(30)) * 0.5
    ey = math.sin(math.radians(30)) * 1.0 + math.cos(math.radians(30)) * 0.5
    assert abs(x1 - ex) < 1e-12 and abs(x0 + ex) < 1e-12 and abs(y1 - ey) < 1e-12 and abs(y0 + ey) < 1e-12
    assert WO.footprint([('box', 0, 0, 0, 1, 2, 1), ('cyl', 3, 3, 0.5, 0, 1)]) == W._footprint([('box', 0, 0, 0, 1, 2, 1), ('cyl', 3, 3, 0.5, 0, 1)])


def test_partition_masks_match_objects():
    for v in WO.VARIANTS:
        s, i, r = TRAIN_HOUSES[0]
        sp = WO.world_spec(v, s, i, r); rr = WO.rasterize(sp, (128, 128, 32)); oa = WO.ood_arrays(sp, (128, 128, 32), rr)
        n = len(sp['objects']); off = oa['objvox_off']
        assert len(off) == n + 1 and len(oa['obj_kinds']) == n and len(oa['obj_ood']) == n
        assert np.all(np.diff(off) > 0), 'an object has no voxel'
        V = np.zeros(rr['occ'].size, bool); Vo = np.zeros(rr['occ'].size, bool)
        for k in range(n):
            ix = oa['objvox_idx'][off[k]:off[k + 1]]; V[ix] = True
            if oa['obj_ood'][k]:
                Vo[ix] = True
        assert np.array_equal(V.reshape(rr['occ'].shape) & ~rr['slab'], rr['obj'])
        assert np.array_equal(Vo.reshape(rr['occ'].shape), oa['ood_mask'])
        assert (oa['obj_ood'].all() if v != 'ctrl' else not oa['obj_ood'].any())
        if v == 'ctrl':
            assert not oa['ood_mask'].any()


def test_module_for_and_house_ctx():
    m = WO.module_for('novel')
    for k in ('BEAMS', 'fibonacci_sphere', 'array_hash', 'scene_plan', 'HOPS', 'v371'):
        assert getattr(m, k) is getattr(W, k)
    old = E.W
    try:
        E.W = m
        t0 = time.time(); ctx = E.house_ctx('TRAIN', 1, 9)
        fx = ctx['fx']
        assert fx['world_id'].startswith('v420ood:novel:TRAIN:') and fx['variant'] == 'novel' and fx['ood_mask'].shape == ctx['occ'].shape
        assert set(W.make_fixture.__code__.co_varnames[:5]) <= set(WO.make_fixture.__code__.co_varnames)
        L_k0, L_pub, hi = E.tier_maps(ctx, 'cam_nf', 203, {})
        assert not hi and len(L_k0) == ctx['n'] and L_k0[0].dtype == np.int8
        print(f'   house_ctx + cam_nf maps on TRAIN r1_i9 (novel): {time.time() - t0:.1f}s')
    finally:
        E.W = old


# ---------------------------------------------------------------- scoring layers
def _toy_house(shape=(12, 12, 8)):
    occ = np.zeros(shape, np.int8); slab = np.zeros(shape, bool); slab[:, :, 0] = True; occ[slab] = 1
    obj = np.zeros(shape, bool); obj[3:5, 3:5, 1:3] = True; occ[obj] = 1
    H = G.House(occ, slab, obj, (0.1, 0.1, 0.1))
    t = np.zeros(shape, np.int8); t[slab] = 1; t[obj] = 1
    return occ, H, G.Target(t, H)


def test_phantom_blocks_toy():
    occ, H, T = _toy_house()
    near = __import__('scipy').ndimage.binary_dilation(occ == 1, structure=P.CUBE)
    kc = np.ones(occ.shape, np.int32)
    p = np.zeros(occ.shape, np.int8); p[occ == 1] = 1
    assert P.phantom_blocks(p, kc, H, near)[0] == 0                    # the true object is no phantom
    p2 = p.copy(); p2[9:11, 9, 4:6] = 1                                # a 4-voxel block far from everything
    p2[5, 5, 3] = 1                                                    # diagonal neighbour of the object (Chebyshev 1): not a phantom
    n, vox, nb, _ = P.phantom_blocks(p2, kc, H, near)
    assert n == 1 and vox == 4
    kc0 = kc.copy(); kc0[9:11, 9, 4:6] = 0
    assert P.phantom_blocks(p2, kc0, H, near)[0] == 0                  # unknown voxels are never counted


def test_phantom_s3_floor_row():
    """S3: floor-hugging blocks are invisible to the verdict definition (the floor slab is truth within one cell); the report row drops
    the floor slab from the truth and counts blocks reaching z >= 2"""
    occ, H, T = _toy_house()
    nd = __import__('scipy').ndimage
    near = nd.binary_dilation(occ == 1, structure=P.CUBE)
    floor = np.zeros(occ.shape, bool); floor[:, :, 0] = True
    near_nf = nd.binary_dilation((occ == 1) & ~floor, structure=P.CUBE)
    kc = np.ones(occ.shape, np.int32)
    p = np.zeros(occ.shape, np.int8); p[occ == 1] = 1
    p[9, 9, 1] = 1                                                     # one floor layer only: neither row
    p[9, 2, 1:3] = 1                                                   # floor-hugging but reaching z = 2: S3 row only
    p[2, 10, 4:6] = 1                                                  # floating far block: both rows
    n, vox, nb, tall = P.phantom_blocks(p, kc, H, near, near_nf, 2)
    assert (n, tall) == (1, 2), (n, tall)
    assert P.phantom_blocks(p, kc, H, near)[3] == 0                    # no S3 row without near_nf


def test_flip_split_toy():
    occ, H, T = _toy_house()
    shape = occ.shape
    near_fam = np.full(shape, 1, np.int64); near_fam[:6] = 0            # x < 6 -> family 0, else family 1
    cheb1 = np.zeros(shape, bool); cheb1[5, 3:5, 1:3] = True
    pn = np.zeros(shape, np.int8); pb = np.zeros(shape, np.int8); S = np.full(shape, -5, np.int32)
    pn[3:5, 3:5, 1:3] = 1                                              # 8 true voxels (family 0, layer 1 has 4)
    pn[5, 3:5, 1:3] = 1                                                # 4 false voxels at Chebyshev 1 (family 0)
    pn[8, 8, 4] = 1                                                    # 1 false voxel, family 1
    d = P.flip_split(pn, pb, S, T, H, near_fam, 2, cheb1, 1)
    assert d['fam'].tolist() == [[8, 4, 4], [0, 1, 0], [0, 0, 0]], d['fam'].tolist()
    assert d['h'][:3, 0].sum() == 8 and d['h'][:3, 1].sum() == 5 and d['h'][3].tolist() == [4, 2]   # bands; layer 1: 4 true, 2 false


def test_guards_formal_png_skip_sha():
    import tempfile, json
    for argv in (['run', '--split', 'VALIDATION', '--formal', '--houses', 'r1:0'],
                 ['run', '--split', 'VALIDATION', '--formal', '--plan', 'ctrl:r1:0'],
                 ['run', '--split', 'TRAIN', '--smoke', '--formal', '--houses', 'r1:9'],
                 ['run', '--split', 'VALIDATION', '--smoke', '--houses', 'r1:0'],
                 ['geom', '--split', 'TRAIN', '--houses', 'r1:9', '--png', os.path.join(P.ROOT, 'claude_handoff', 'x_png')]):
        try:
            P.main(argv); raise RuntimeError(f'not refused: {argv}')
        except AssertionError:
            pass
    d = tempfile.mkdtemp()
    opts = dict(script_sha256=P.sha(P.__file__), gen_sha256=P.sha(WO.__file__), v420_sha256=P.sha(W.__file__), e3r_sha256=P.sha(E.__file__),
                noise_sha256=P.sha(G.__file__), threads=1, export=d, out=d)
    old, old_w = E.house_ctx, E.W
    try:
        def boom(*a, **k):
            raise ValueError('boom')
        def agents(*a, **k):
            raise ValueError('could not place agents')
        E.house_ctx = boom
        r = P.run_job(('novel', 'TRAIN', 'train', 1, 9, (203,), opts))
        assert r.get('failed') and not r.get('skipped') and 'boom' in r['error']
        E.house_ctx = agents
        r = P.run_job(('novel', 'TRAIN', 'train', 1, 9, (203,), opts))
        assert r.get('skipped') and not r.get('failed')
        r = P.run_job(('novel', 'TRAIN', 'train', 1, 9, (203,), dict(opts, gen_sha256='0' * 64)))
        assert r.get('failed') and 'changed since the run started' in r['error']
    finally:
        E.house_ctx = old; E.W = old_w
    # S4b / S4c: foreign jobs and skips block a formal aggregation; a smoke aggregation never takes VAL jobs
    os.makedirs(os.path.join(d, 'jobs'), exist_ok=True)
    for name, rec in (('novel_train_r1_i9', dict(variant='novel', split='TRAIN', panel='train', rooms=1, index=9, failed=True)),
                      ('ctrl_val_r1_i0', dict(variant='ctrl', split='VALIDATION', panel='val', rooms=1, index=0, skipped='could not place agents'))):
        json.dump(rec, open(os.path.join(d, 'jobs', name + '.json'), 'w'))
    S = P.aggregate(d)
    assert S['foreign'] == ['novel_train_r1_i9'] and S['skipped'] == [('ctrl', 'r1_i0')] and S['verdict']['verdict'] == 'INCOMPLETE'
    assert any('another split' in x for x in S['verdict']['reasons']) and any('skipped' in x for x in S['verdict']['reasons'])
    S = P.aggregate(d, smoke=True)
    assert S['foreign'] == ['ctrl_val_r1_i0'] and S['failed'] == ['novel_train_r1_i9']


def test_flip_and_zone_counts_toy():
    occ, H, T = _toy_house()
    shape = occ.shape
    pn = np.zeros(shape, np.int8); pb = np.zeros(shape, np.int8)
    S = np.full(shape, -100, np.int32)
    pn[3:5, 3:5, 1:3] = 1                                              # NET right on the object (8 voxels, all true)
    pn[6, 6, 2] = 1                                                    # NET wrong voxel against free evidence
    S[3:5, 3, 1:3] = 10                                                # 4 of the true voxels have positive evidence (not S <= 0)
    zone = G.dil6(T.tr)
    fc = P.flip_counts(pn, pb, S, T, H, zone)
    assert fc['le_-0'][:2] == [4, 1]                                    # precision 4 / 5
    fc2 = P.flip_counts(pn, pb, S * 0 - 41, T, H, zone)                # S x 0.05 = -2.05 nats <= -2
    assert fc2['le_-2'][:2] == [8, 1] and fc2['le_-4'][:2] == [0, 0]
    kc = np.ones(shape, np.int32)
    allz = np.ones(shape, bool)
    zc = P.zone_counts(pn, T, H, kc, dict(zone=allz))['zone']['c']
    row = G.score(pn, T)
    assert (zc[0], zc[1], zc[2]) == (row['furn_tp'], row['furn_fp'], row['furn_fn'])
    assert abs(P._tol1(zc.astype(float)) - row['furn_f1_tol1']) < 1e-12
    zz = P.zone_counts(pn, T, H, kc, dict(zone=zone, rest=~zone))
    assert zz['zone']['c'][0] + zz['rest']['c'][0] == zc[0] and zz['zone']['c'][1] + zz['rest']['c'][1] == zc[1]


def _fake_V(net_all, b6_all, net_zone, b6_zone, flip, ph_net, ph_b6, free_net=0.99, free_b6=0.99, b4=0.85):
    def arm(all_, zone_, ph, free):
        return dict(furn_f1=all_, furn_f1_tol1=all_, free_recall=free, surface_f1=0.9, phantoms_per_house=ph,
                    zones=dict(zone=dict(f1=zone_, f1_tol1=zone_), rest=dict(f1=None, fp=0)))
    R8 = {'B4': arm(b4, b4, 0, free_b6)}
    for h in 'AB':
        R8[f'B6_{h}'] = arm(b6_all, b6_zone, ph_b6, free_b6)
    for ts in ('342101', '342202'):
        R8[f'NET_{ts}'] = arm(net_all, net_zone, ph_net, free_net)
    fl = {'le_-0': dict(n=10, prec=flip, n_zone=5, prec_zone=flip)}
    return dict(R8=R8, flips={f'NET_{ts}': {'B6_A': fl, 'B6_B': fl} for ts in ('342101', '342202')}, house_list=['r1_i0'])


def _fake_S(per_variant):
    return dict(failed=[], skipped=[], script_sha256=['x'], gen_sha256=['y'], ckpt_sha256=['z'],
                variants={v: {str(s): V for s in (1, 2)} for v, V in per_variant.items()})


def test_verdict_logic():
    ctrl = _fake_V(0.97, 0.90, None, None, 0.9, 1.0, 1.0)
    safe = _fake_V(0.95, 0.93, 0.92, 0.925, 0.8, 1.2, 1.0)             # zone -0.5 pp, overall +2, flip .8, phantoms +0.2
    amb = _fake_V(0.95, 0.93, 0.92, 0.925, 0.6, 1.2, 1.0)              # flip .6: not safe, not harmful
    harm = _fake_V(0.95, 0.93, 0.90, 0.92, 0.8, 1.2, 1.0)              # zone -2 pp
    harm_ph = _fake_V(0.95, 0.93, 0.92, 0.92, 0.8, 3.5, 1.0)           # phantoms > B6 + 2
    vd = P.verdict(_fake_S(dict(ctrl=ctrl, novel=safe, scale=amb, floating=harm)), ['r1_i0'], (1, 2))
    assert vd['ctrl_ok'] and vd['verdict'] == dict(novel='SAFE', scale='AMBIGUOUS', floating='HARMFUL'), vd['verdict']
    c = vd['variants']['novel']['own']['cells']['342202_s2']                  # S1: ctrl of the same checkpoint and seed, report only
    assert c['b6'] == 'B6_B' and c['ctrl']['flip_prec'] == 0.9 and c['ctrl']['ph_net'] == 1.0 and c['safe']
    S = _fake_S(dict(ctrl=ctrl, novel=safe, scale=safe, floating=safe)); S['skipped'] = [('novel', 'r2_i3')]
    assert P.verdict(S, ['r1_i0'], (1, 2))['verdict'] == 'INCOMPLETE'           # S4c: a skip blocks the formal verdict
    S = _fake_S(dict(ctrl=ctrl, novel=safe, scale=safe, floating=safe)); S['foreign'] = ['novel_train_r1_i9']
    assert P.verdict(S, ['r1_i0'], (1, 2))['verdict'] == 'INCOMPLETE'           # S4b
    vd = P.verdict(_fake_S(dict(ctrl=ctrl, novel=harm_ph, scale=safe, floating=safe)), ['r1_i0'], (1, 2))
    assert vd['verdict']['novel'] == 'HARMFUL'
    bad_ctrl = _fake_V(0.99, 0.90, None, None, 0.9, 1.0, 1.0)          # +9 pp: outside [+5.3, +8.6]
    vd = P.verdict(_fake_S(dict(ctrl=bad_ctrl, novel=safe, scale=safe, floating=safe)), ['r1_i0'], (1, 2))
    assert vd['verdict'].startswith('INCONCLUSIVE')
    S = _fake_S(dict(ctrl=ctrl, novel=safe, scale=safe, floating=safe)); S['failed'] = ['novel_val_r1_i0']
    assert P.verdict(S, ['r1_i0'], (1, 2))['verdict'] == 'INCOMPLETE'
    S = _fake_S(dict(ctrl=ctrl, novel=safe, scale=safe, floating=safe))
    assert P.verdict(S, ['r1_i0', 'r2_i0'], (1, 2))['verdict'] == 'INCOMPLETE'   # a missing house
    del S['variants']['floating']['2']
    assert P.verdict(S, ['r1_i0'], (1, 2))['verdict'] == 'INCOMPLETE'
    lowfree = _fake_V(0.95, 0.93, 0.92, 0.925, 0.8, 1.2, 1.0, free_net=0.98, free_b6=0.99)
    vd = P.verdict(_fake_S(dict(ctrl=ctrl, novel=lowfree, scale=safe, floating=safe)), ['r1_i0'], (1, 2))
    assert vd['verdict']['novel'] == 'AMBIGUOUS'


if __name__ == '__main__':
    t0 = time.time(); n = 0
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith('test_') and callable(v)):
        t = time.time(); fn(); n += 1
        print(f'ok  {name}  {time.time() - t:.1f}s', flush=True)
    print(f'all {n} tests passed in {time.time() - t0:.0f}s')
