"""Unit tests for g4_noise_panels (Claude, 2026-10-27, batch G4-E1). CPU, about a minute:  python -B test_g4_noise_panels.py
(also runs under pytest). Real-fixture checks (clean tier = observe() voxel-for-voxel on every agent, noise statistics, relay
assertions) are part of the smoke run (--smoke-asserts), not of this file."""
import itertools, math, os, sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g4_noise_panels as G  # noqa: E402

ANALYSIS = os.path.join(G.ROOT, 'claude_handoff', 'analysis_20261027')


# ---------------------------------------------------------------- small synthetic world
def _world(shape=(24, 20, 12), cell=(0.1, 0.1, 0.1)):
    occ = np.zeros(shape, np.uint8)
    occ[0, :, :] = occ[-1, :, :] = occ[:, 0, :] = occ[:, -1, :] = 1           # walls
    occ[:, :, 0] = 1; occ[:, :, -1] = 1                                       # slabs
    occ[6:10, 5:9, 1:5] = 1                                                   # table-like box
    occ[15, 12:14, 1:9] = 1                                                   # thin post
    occ[:4, 8:12, 1:12] = 0; occ[0, 9:11, 1:8] = 0                            # opening to the outside (rays leave the grid)
    return occ, np.asarray(cell, np.float64)


def _poses(cell):
    return np.array([[1.25, 1.05, 0.65], [1.31, 0.93, 0.62], [1.71, 1.37, 0.71]]) * (np.asarray(cell) / 0.1)


def _clean_scans(occ, cell, poses, rays):
    ds = 0.5 * float(cell.min()); out = []
    for j, p in enumerate(poses):
        hs, ls = G.trace_pose(occ, cell, p, rays)
        dr = G.pose_draws(1, 1, 0, 0, j, len(rays))
        scan, st = G.measure(G.CONDITIONS['clean'], hs, ls, p, dr, ds, rays)
        assert st['dropped'] == 0 and st['outliers'] == 0 and st['s_err2'] == 0 and st['pose'] == [0., 0., 0., 0.]
        out.append(scan)
    return out, ds


def test_clean_integration_equals_observe():
    for cell in ((0.1, 0.1, 0.1), (0.109375, 0.109375, 0.1), (0.0625, 0.0625, 0.1)):
        occ, cell = _world(cell=cell)
        rays = G.W.fibonacci_sphere(4096); poses = _poses(cell)
        ref = G.W.observe(occ, cell, poses, rays)
        scans, ds = _clean_scans(occ, cell, poses, rays)
        maps = G.integrate_agent(occ.shape, cell, ds, scans)
        assert sorted(maps) == sorted(G.INTEGRATORS) and G.GOVERNING == 'V3'
        for I, L in maps.items():                                               # every integrator degenerates to observe() when clean
            assert np.array_equal(G.tern(L), ref), (I, int((G.tern(L) != ref).sum()))
            assert (L[ref == 1] >= G.L_HIT).all() and (L[ref == 0] <= G.L_MISS).all() and G.is_unk(L)[ref == -1].all()
            assert L.dtype == (np.int32 if I in G.UNCLAMPED else np.int8)
        # inter-agent error of zero size and the identity frame leave the scan untouched
        assert G.to_public(scans[0], None) is scans[0]


def _noisy_scans(n_rays=1024):
    occ, cell = _world(); rays = G.W.fibonacci_sphere(n_rays); poses = _poses(cell); ds = 0.5 * float(cell.min()); scans = []
    for j, p in enumerate(poses):
        hs, ls = G.trace_pose(occ, cell, p, rays)
        scans.append(G.measure(G.CONDITIONS['cam'], hs, ls, p, G.pose_draws(3, 1, 0, 0, j, len(rays)), ds, rays)[0])
    return occ, cell, ds, scans


def test_integrators_match_naive_definitions():
    assert (G.L_HIT, G.L_MISS, G.L_MIN, G.L_MAX) == (17, -8, -40, 70)
    for n_rays in (1024, 4096):
        occ, cell, ds, scans = _noisy_scans(n_rays)
        maps = G.integrate_agent(occ.shape, cell, ds, scans)
        V = int(np.prod(occ.shape)); kn = np.zeros(V, bool)
        ref = {I: np.zeros(V, np.int64) for I in G.INTEGRATORS}
        for o, D, r, sm in scans:
            v, b, d, nh, nm = G.scan_events(occ.shape, cell, ds, o, D, r, sm)
            assert np.array_equal(np.bincount(v, minlength=V), nh + nm)
            assert len(set(zip(v.tolist(), b.tolist()))) == len(v)             # at most one event per (voxel, beam)
            t = (nh + nm) > 0; kn |= t
            ref['V0'][t] = np.clip(ref['V0'][t] + 17 * nh[t] - 8 * nm[t], -40, 70)                       # V0: one clamp per scan
            ref['V2'] += 17 * nh - 8 * nm                                                                 # V2: no clamp
            for x in np.nonzero(t)[0]:                                                                    # V1: one update per scan
                ref['V1'][x] = min(max(ref['V1'][x] + (17 if nh[x] else -8), -40), 70)
            for i in np.lexsort((b, v)):                                                                  # V3: per beam, clamp each
                ref['V3'][v[i]] = min(max(ref['V3'][v[i]] + int(d[i]), -40), 70)
        for I in G.INTEGRATORS:
            sentinel = np.iinfo(np.int32).min if I in G.UNCLAMPED else -128
            assert np.array_equal(np.where(kn, ref[I], sentinel).reshape(occ.shape), maps[I].astype(np.int64)), I
        assert maps['V2'][maps['V2'] != np.iinfo(np.int32).min].min() < G.L_MIN                       # V2 really is unclamped
        k = maps['V3'] != G.UNK
        assert maps['V3'][k].min() == G.L_MIN and maps['V3'][k].max() == G.L_MAX                      # clamps bind
        assert np.array_equal(G.integrate_scan(occ.shape, cell, ds, *scans[0])[0], G.scan_events(occ.shape, cell, ds, *scans[0])[3])


def test_apply_sequential_equals_naive_loop():
    rng = np.random.default_rng(0)
    for trial in range(20):
        V = 50; n = 400
        v = rng.integers(0, V, n); b = rng.permutation(n)
        d = np.where(rng.random(n) < .4, G.L_HIT, G.L_MISS).astype(np.int8)
        L = rng.integers(-40, 71, V).astype(np.int64); ref = L.copy()
        for i in np.lexsort((b, v)):
            ref[v[i]] = min(max(ref[v[i]] + int(d[i]), G.L_MIN), G.L_MAX)
        G.apply_sequential(L, v, b, d)
        assert np.array_equal(L, ref), trial


def test_noise_injection_statistics():
    rng = np.random.default_rng(0); R = 200000
    rays = rng.standard_normal((R, 3)); rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    hit = rng.integers(0, 80, R).astype(np.int32); last = np.where(hit > 0, hit - 1, 60).astype(np.int32); ds = 0.05
    p = np.array([1., 1., 1.])
    for name in ('cam', 'lidar', 'harsh'):
        c = G.CONDITIONS[name]
        dr = G.pose_draws(7, 1, 3, 2, 1, R)
        (o, D, r, sm), st = G.measure(c, hit, last, p, dr, ds, rays)
        assert abs(st['dropped'] / R - c['p_drop']) < 0.003, name
        assert len(D) == R - st['dropped']
        assert abs(st['outliers'] / st['hits_kept'] - c['p_out']) < 0.0015, name
        n = st['n_err']; sd_abs = math.sqrt(st['s_err2'] / n - (st['s_err'] / n) ** 2); sd_rel = math.sqrt(st['s_rel2'] / n - (st['s_rel'] / n) ** 2)
        if c['sigma0']:
            assert abs(sd_abs / c['sigma0'] - 1) < 0.02, (name, sd_abs)
        if c['k']:
            assert abs(sd_rel / c['k'] - 1) < 0.02, (name, sd_rel)
        z = dr['z_pose']
        assert np.allclose(o, p + [z[0] * c['pose_t'], z[1] * c['pose_t'], z[2] * c['pose_t'] / 2])
        assert np.allclose(D, rays[dr['u_drop'] >= c['p_drop']] @ G.rotz(z[3] * c['pose_yaw']).T)
        assert np.isnan(r[hit[dr['u_drop'] >= c['p_drop']] == 0]).all() and (r[~np.isnan(r)] >= 0.05).all()
    # common random numbers: cam and cam_nf scans are identical before the inter-agent frame
    dr = G.pose_draws(1, 2, 3, 4, 0, R)
    a, _ = G.measure(G.CONDITIONS['cam'], hit, last, p, dr, ds, rays); b, _ = G.measure(G.CONDITIONS['cam_nf'], hit, last, p, dr, ds, rays)
    assert all(np.array_equal(x, y, equal_nan=True) for x, y in zip(a, b))


def test_ctx_tern_matches_rate_baselines():
    sys.path.insert(0, ANALYSIS)
    import torch  # noqa: F401  (rate_baselines_lib needs it)
    from rate_baselines_lib import ctx_bytes, down_up
    rng = np.random.default_rng(1)
    for shape in ((12, 10, 8), (20, 16, 6)):
        t = rng.integers(-1, 2, shape).astype(np.int8); t[:, :, :2] = -1; t[3:7, 2:9, 2:5] = 1
        assert abs(G.ctx_tern_bytes(t) - ctx_bytes(t)) < 1e-6
        c, u = down_up(t, (2, 2, 1), .5)
        assert np.array_equal(G.down_tern(t, (2, 2, 1)), c) and np.array_equal(G.up_tern(c, (2, 2, 1)), u)


def _seq_int_bits(v, known):
    """brute-force sequential adaptive coder of the B4 stream (ternary pass then magnitude pass), KT predictive probabilities"""
    X, Y, Z = v.shape; t = np.where(known, (v > 0).astype(int), -1); s = t + 1
    cnt3 = defaultdict(lambda: [0, 0, 0]); bits = 0.
    for x, y, z in itertools.product(range(X), range(Y), range(Z)):
        ctx = tuple(int(s[x - dx, y - dy, z - dz]) if (x >= dx and y >= dy and z >= dz) else 0 for dx, dy, dz in G.NB)
        c = cnt3[ctx]; sym = int(s[x, y, z]); bits -= math.log2((c[sym] + .5) / (sum(c) + 1.5)); c[sym] += 1
    u = np.where(v > 0, v - 1, -v); k = np.zeros_like(u)
    k[known] = np.floor(np.log2(u[known] + 1)).astype(int)
    cnt2 = defaultdict(lambda: [0, 0])

    def emit(ctx, bit):
        nonlocal bits
        c = cnt2[ctx]; bits -= math.log2((c[bit] + .5) / (c[0] + c[1] + 1.)); c[bit] += 1
    for x, y, z in itertools.product(range(X), range(Y), range(Z)):
        if not known[x, y, z]:
            continue
        st = int(v[x, y, z] > 0); kk = int(k[x, y, z]); cls = 0
        nbs = [(nx, ny, nz) for nx, ny, nz in ((x, y, z - 1), (x, y - 1, z), (x - 1, y, z))
               if min(nx, ny, nz) >= 0 and known[nx, ny, nz] and int(v[nx, ny, nz] > 0) == st]
        if nbs:
            up = int(u[nbs[0]]); kp = min(int(math.floor(math.log2(up + 1))), 9); cls = 1 + kp
            nsame = sum(int(u[q]) == up for q in nbs[1:])
            emit(('f', st, kp, nsame), int(u[x, y, z]) == up)
            if int(u[x, y, z]) == up:
                continue
        for j in range(kk):
            emit(('p', st, cls, min(j, 11)), 1)
        emit(('p', st, cls, min(kk, 11)), 0)
        rem = int(u[x, y, z]) + 1 - (1 << kk)
        for b in range(kk):
            emit(('s', st, kk, b), (rem >> (kk - 1 - b)) & 1)
    return bits / 8 + 4


def test_int_coder_is_exact_adaptive_code_length():
    rng = np.random.default_rng(2)
    for shape, hi in (((6, 5, 4), 70), ((5, 7, 6), 490)):
        known = rng.random(shape) < .7
        v = np.where(rng.random(shape) < .5, rng.integers(1, hi, shape), -rng.integers(0, hi, shape)); v[~known] = 0
        v[0, 0, :] = 3000; known[0, 0, :] = True                               # long exp-Golomb prefixes (> 11 bins, capped context)
        v[1:4, :, 1:3] = -40; v[2, 1:3, 3] = 52; known[1:4, :, 1:3] = True; known[2, 1:3, 3] = True   # runs: the equal-to-neighbour flag
        assert abs(G.int_ctx_bytes(v, known) - _seq_int_bits(v, known)) < 1e-6
    L = np.full((4, 4, 4), G.UNK, np.int8); L[1:3, 1:3, 1:3] = -40
    assert G.int_ctx_bytes(L, L != G.UNK) < 20                                 # a constant map is cheap


def _ref_bt(t, tree_depth=16, res=0.1):
    X, Y, Z = t.shape; S = 1 << int(math.ceil(math.log2(max(X, Y, Z)))); dc = int(round(math.log2(S)))
    g = np.zeros((S, S, S), int); g[:X, :Y, :Z] = np.where(t < 0, 0, np.where(t == 1, 2, 1))

    def build(level, x, y, z):
        if level == 0:
            return ('leaf', int(g[x, y, z])) if g[x, y, z] else None
        ch = [build(level - 1, 2 * x + (i & 1), 2 * y + ((i >> 1) & 1), 2 * z + ((i >> 2) & 1)) for i in range(8)]
        if all(c == ('leaf', 1) for c in ch):
            return ('leaf', 1)
        if all(c == ('leaf', 2) for c in ch):
            return ('leaf', 2)
        if all(c is None for c in ch):
            return None
        return ('inner', ch)
    node = build(dc, 0, 0, 0)
    if node is None:
        return f'# Octomap OcTree binary file\n# (feel free to add / change comments, but leave the first line as it is!)\n#\nid OcTree\nsize 0\nres {res}\ndata\n'.encode()
    for lv in range(dc + 1, tree_depth + 1):
        ch = [None] * 8; ch[7 if lv == tree_depth else 0] = node; node = ('inner', ch)

    def code(c):
        return 0 if c is None else (3 if c[0] == 'inner' else c[1])

    def ser(nd):
        ch = nd[1]; out = bytes([sum(code(ch[i]) << (2 * i) for i in range(4)), sum(code(ch[i]) << (2 * (i - 4)) for i in range(4, 8))])
        for c in ch:
            if c is not None and c[0] == 'inner':
                out += ser(c)
        return out

    def count(nd):
        return 1 + (sum(count(c) for c in nd[1] if c is not None) if nd[0] == 'inner' else 0)
    header = ('# Octomap OcTree binary file\n# (feel free to add / change comments, but leave the first line as it is!)\n#\n'
              f'id OcTree\nsize {count(node)}\nres {res}\ndata\n')
    return header.encode() + ser(node)


def test_octomap_bt_matches_recursive_reference():
    rng = np.random.default_rng(3)
    for shape in ((8, 8, 4), (16, 12, 8), (5, 9, 3)):
        t = rng.integers(-1, 2, shape).astype(np.int8)
        t[:4, :4, :2] = 0; t[4:8, :4, :] = 1; t[:, -2:, :] = -1               # prunable blocks at several levels
        assert G.octomap_bt(t) == _ref_bt(t), shape
    t = np.full((8, 8, 8), -1, np.int8)
    assert G.octomap_bt(t) == _ref_bt(t) and b'size 0' in G.octomap_bt(t)
    t = np.zeros((8, 8, 8), np.int8)                                            # a fully known uniform cube prunes to one leaf
    assert G.octomap_bt(t) == _ref_bt(t) and b'size 14\n' in G.octomap_bt(t)       # 13 single-child ancestors (depth 16) + 1 leaf


def _random_forest(n, rng, k_comp=2):
    edges = []; size = n // k_comp
    for a in range(1, n):
        if a % size == 0:
            continue                                                            # start a new component
        lo = (a // size) * size
        edges.append((int(rng.integers(max(lo, a - 3), a)), a, 1.0))
    return edges


def test_relay_counts_every_agent_once():
    rng = np.random.default_rng(4); n = 8
    for trial in range(5):
        F = G.forest(_random_forest(n, rng), n)
        maps = [np.where(rng.random((6, 5, 4)) < .3, G.UNK, rng.integers(-40, 71, (6, 5, 4))).astype(np.int8) for _ in range(n)]
        st = G.RelayStats()
        msgs, preds = G.fuse(maps, F, 'sum', st)
        for (s, r), (S, K) in msgs.items():                                     # every message = exact sum over the sender side
            side = F['sides'][(s, r)]
            assert np.array_equal(S, sum(np.where(maps[a] == G.UNK, 0, maps[a].astype(np.int32)) for a in side))
            assert np.array_equal(K, np.any([maps[a] != G.UNK for a in side], 0))
        for a in range(n):
            cm = F['comps'][F['comp_of'][a]]
            Ssum = sum(np.where(maps[b] == G.UNK, 0, maps[b].astype(np.int32)) for b in cm); Kany = np.any([maps[b] != G.UNK for b in cm], 0)
            assert np.array_equal(preds[a], np.where(Kany, (Ssum > 0).astype(np.int8), -1))
        assert st.belief_checks == n and st.combines == len(F['directed']) + n
        for kind in ('maxabs', 'max_occ'):
            G.fuse(maps, F, kind)
    # the provenance assertion fires on any double counting (e.g. a message fed back to its sender)
    try:
        G._merge([(np.zeros(2), 0b011), (np.zeros(2), 0b010)], G.comb_maxocc, None)
        raise RuntimeError('double counting not detected')
    except AssertionError:
        pass


def test_maxabs_is_a_join():
    rng = np.random.default_rng(5)
    a, b, c = (np.where(rng.random(500) < .2, G.UNK, rng.integers(-40, 71, 500)).astype(np.int8) for _ in range(3))
    f = lambda *x: G.comb_maxabs(list(x))  # noqa: E731
    assert np.array_equal(f(a, b), f(b, a)) and np.array_equal(f(f(a, b), c), f(a, f(b, c))) and np.array_equal(f(a, a), a)
    assert np.array_equal(f(np.array([5, -5, -6, G.UNK], np.int8), np.array([-5, 5, 5, -1], np.int8)), np.array([5, 5, -6, -1], np.int8))


def _a07_furn_tol1(p, t, region):
    from scipy import ndimage
    S6 = ndimage.generate_binary_structure(3, 1); dil = lambda m: ndimage.binary_dilation(m, S6)  # noqa: E731
    known = t >= 0; po = (p == 1) & known & region; to = (t == 1) & known & region
    npr, nr = po.sum(), to.sum()
    if npr + nr == 0:
        return None
    pp = (po & dil(to)).sum() / npr if npr else 1.; rr = (to & dil(po)).sum() / nr if nr else 1.
    return 2 * pp * rr / max(pp + rr, 1e-12)


def test_metrics_match_train_3d_v431():
    import torch
    import train_3d_v431 as T
    rng = np.random.default_rng(6); dims = (16, 14, 10)
    occ = np.zeros(dims, bool); occ[0] = occ[-1] = True; occ[:, :, 0] = True; occ[5:9, 4:8, 1:4] = True; occ[11, 9, 1:7] = True
    slab = np.zeros(dims, bool); slab[:, :, 0] = True; obj = np.zeros(dims, bool); obj[5:9, 4:8, 1:4] = True; obj[11, 9, 1:7] = True
    H = G.House(occ.astype(np.uint8), slab, obj, (0.1, 0.1, 0.1))
    struct = occ & ~obj & ~slab
    for _ in range(4):
        t = np.where(rng.random(dims) < .35, -1, occ.astype(np.int8)).astype(np.int8)
        p = t.copy(); flip = rng.random(dims) < .15; p[flip] = rng.integers(-1, 2, int(flip.sum()))
        ref = T.metrics3d(torch.from_numpy(p.reshape(1, -1)), torch.from_numpy(t.reshape(1, -1)), dims, torch.from_numpy(slab.reshape(-1)),
                          torch.from_numpy(obj.reshape(-1)), torch.from_numpy(struct.reshape(-1)))[0]
        got = G.score(p, G.Target(t, H, mult=(t >= 0).astype(np.int64)), strata=True)
        for k in ('occ_f1', 'surface_f1', 'surface_f1_tol1', 'var_occ_f1', 'furn_f1', 'obj_recall', 'accuracy', 'unknown_known_fpr'):
            tol = 1e-6 if k in ('accuracy', 'unknown_known_fpr') else 1e-9                     # torch takes these means in float32
            assert (ref[k] is None and got[k] is None) or abs(ref[k] - got[k]) < tol, (k, ref[k], got[k])
        assert abs(got['furn_f1_tol1'] - _a07_furn_tol1(p, t, H.region)) < 1e-12
        assert abs(got['free_recall'] - ((p == 0) & (t == 0)).sum() / (t == 0).sum()) < 1e-12
        hf = np.asarray(got['strata']['h_furn']).sum(0); mf = np.asarray(got['strata']['m_furn']).sum(0)
        assert hf.tolist() == mf.tolist() == [got['furn_tp'], got['furn_fp'], got['furn_fn']]       # strata partition the region


def test_b5_denoiser():
    L = np.full((10, 10, 6), -8, np.int8); L[:, :, 5] = G.UNK
    L[2:5, 2:5, 1:4] = 40                                                       # 27-voxel object
    L[8, 8, 2] = 17; L[7, 1, 1] = 30; L[7, 2, 2] = 30                           # speckle: 1 voxel, and 2 diagonal (26-connected) voxels
    v = G.b5_variants(L)
    f1 = (1, 1, 1)
    assert (v[('cc', 0, 4, f1)] == 1).sum() == 27 and (v[('cc', 0, 32, f1)] == 1).sum() == 0
    assert (v[('cc', 17, 4, f1)] == 1).sum() == 27                              # tau 17 drops the single-hit voxel, m 4 the pair
    assert (v[('cc', 34, 4, f1)] == 1).sum() == 27 and v[('cc', 34, 4, f1)][8, 8, 2] == 0
    assert np.array_equal(v[('cc', 0, 4, (2, 2, 1))], G.down_tern(v[('cc', 0, 4, f1)], (2, 2, 1)))
    for key, D in v.items():
        if key[3] == f1:
            assert ((D == -1) == (L == G.UNK)).all(), key                       # denoising never changes the known set
    # V1 rule (review S4): tau > 0 uses >=, tau 0 keeps >
    known, occ = G.b5_occ(np.array([0, 17, 34, -8, G.UNK], np.int8), 17, tau_ge=True)
    assert occ.tolist() == [False, True, True, False, False] and G.b5_occ(np.array([0, 17], np.int8), 0, True)[1].tolist() == [False, True]
    # box family: a 1 x 1 x 7 leg standing on a slab is one 26-component, but its own A03 box (the leg + the slab voxel below it,
    # support 8) is deleted for m >= 16, while the 8 x 8 x 1 slab box (support 64) survives
    L = np.full((8, 8, 9), -8, np.int8); L[:, :, 0] = 40; L[3, 3, 1:8] = 40
    v = G.b5_variants(L)
    assert (v[('cc', 0, 32, f1)][3, 3, 1:8] == 1).all()                         # connected to the slab: cc keeps it
    assert (v[('box', 0, 8, f1)][3, 3, 1:8] == 1).all() and (v[('box', 0, 16, f1)][3, 3, 1:8] == 0).all()
    assert (v[('box', 0, 32, f1)][:, :, 0] == 1).all()


def test_box_cover_is_a03_verbatim():
    sys.path.insert(0, os.path.join(ANALYSIS, 'probes', 'A03'))
    import a03np
    rng = np.random.default_rng(8)
    for _ in range(6):
        must = rng.random((9, 8, 7)) < .25; forbid = (rng.random((9, 8, 7)) < .3) & ~must
        boxes = G.box_cover(must, forbid)
        assert boxes == a03np.box_cover(must, forbid)
        P = np.zeros(must.shape, bool)
        for x0, x1, y0, y1, z0, z1 in boxes:
            P[x0:x1, y0:y1, z0:z1] = True
        assert (P >= must).all() and not (P & forbid).any()                     # covers every must voxel, never a forbidden one


def test_hull_and_verdict():
    pts = [(100., .70), (300., .80), (1000., .90)]
    assert G.hull_value(pts, 50.) is None and abs(G.hull_value(pts, 200.) - .75) < 1e-12 and G.hull_value(pts, 5000.) == .90
    assert abs(G.hull_value(pts + [(250., .60)], 250.) - .775) < 1e-12
    I = 'V3'; s0 = 6066.; houses = ['r1_i0', 'r1_i1']; expected = frozenset(houses)

    def D(furn, rnet, b5=.7, cam_nf=.95, exact=True, bytes_ok=True, max_occ=.6, maxabs=.74, b5_present=True, hs=houses):
        rows = lambda f, fpr=.001, fr=.97: dict(furn_f1=f, unknown_known_fpr=fpr, free_recall=fr)  # noqa: E731
        cl = dict(exact=dict(K0_exact=exact), K0=dict(free_recall=1., unknown_known_fpr=0., accuracy=1.),
                  R8=dict(sum=dict(free_recall=1., unknown_known_fpr=0., accuracy=1.)), b4_relay_equals_s0_all_houses=bytes_ok,
                  rnet_median=dict(B4_int_relay=300000.))
        cam = dict(R8=dict(sum=rows(furn), maxabs=rows(maxabs), max_occ=rows(max_occ)), K0=dict(furn_f1=.8),
                   rnet_median=dict(B4_relay=rnet, B4_int_relay=rnet, B4_best_info=rnet))
        if b5_present:
            cam['B5_hull_pub'] = dict(at_quarter_B4=b5, best_any=b5)
        return dict(houses=list(hs), skipped=[], conds=dict(clean={I: cl}, cam={I: cam}, cam_nf={I: dict(R8=dict(sum=rows(cam_nf)))}))
    V = lambda seeds, judged=True: G.verdict(seeds, s0, I, expected, judged)  # noqa: E731
    v = V({'1': D(.80, 20000., .81), '2': D(.84, 30000., .80)})
    assert v['complete'] and v['verdict'].startswith('LEARNING_ROOM') and v['registration_panel'] == 'must be reported' and v['fusion_operator'] == 'sum'
    assert set(v['per_seed']['1']['c']) == {'max_occ', 'maxabs'}
    assert V({'1': D(.80, 20000., .83), '2': D(.84, 30000.)})['verdict'] == 'AMBIGUOUS'                        # B5 > B4 + 2 pp in seed 1
    assert V({'1': D(.93, 20000.), '2': D(.80, 7000.)})['verdict'].startswith('STOP')                         # (.92 | < 1.3 x) per seed
    assert V({'1': D(.80, 20000.), '2': D(.80, 20000., exact=False)})['verdict'].startswith('INCONCLUSIVE')      # (a): maps
    assert V({'1': D(.80, 20000.), '2': D(.80, 20000., bytes_ok=False)})['verdict'].startswith('INCONCLUSIVE')   # (a): bytes = S0
    # (c) is conjunctive: beating max_occ is not enough
    assert V({'1': D(.80, 20000., maxabs=.78), '2': D(.80, 20000.)})['fusion_operator'].startswith('undecided')
    assert V({'1': D(.80, 20000., max_occ=.78), '2': D(.80, 20000.)})['fusion_operator'].startswith('undecided')
    # B5 present but no point at <= 1/4 B4: the arm passes and is flagged
    v = V({'1': D(.80, 20000., None), '2': D(.80, 20000., None)})
    assert v['verdict'].startswith('LEARNING_ROOM') and v['per_seed']['1']['B5_note'].startswith('B5 cannot run')
    # review M5: everything else is INCOMPLETE
    assert V({'1': D(.80, 20000.)})['verdict'].startswith('INCOMPLETE')                                        # one seed
    assert V({'1': D(.80, 20000.), '3': D(.80, 20000.)})['verdict'].startswith('INCOMPLETE')                   # wrong seeds
    assert V({'1': D(.80, 20000.), '2': D(.80, 20000., hs=houses[:1])})['verdict'].startswith('INCOMPLETE')    # house sets differ
    assert V({'1': D(.80, 20000., b5_present=False), '2': D(.80, 20000.)})['verdict'].startswith('INCOMPLETE')  # cam without B5
    bad = D(.80, 20000.); del bad['conds']['cam_nf']
    assert V({'1': bad, '2': D(.80, 20000.)})['verdict'].startswith('INCOMPLETE')
    assert V({'1': D(.80, 20000.), '2': D(.80, 20000.)}, judged=False)['verdict'].startswith('NOT JUDGED')


def test_aggregate_refuses_mixed_fingerprints():
    import json, tempfile
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, 'jobs'))
        for k, fp in enumerate(({'job_format': G.JOB_FORMAT, 'script_sha256': 'a'}, {'job_format': G.JOB_FORMAT, 'script_sha256': 'b'})):
            json.dump(dict(panel='val', house=f'r1_i{k}', seed=1, skipped='x', fingerprint=fp), open(os.path.join(tmp, 'jobs', f'j{k}.json'), 'w'))
        try:
            G.aggregate(tmp)
            raise RuntimeError('mixed fingerprints were aggregated')
        except SystemExit as e:
            assert 'fingerprints' in str(e)
    opts = dict(out='x', conds=['clean'], s0_rnet=6066., integrators=['V3'])
    assert G.fingerprint(opts) == G.fingerprint(dict(opts, out='y')) and 'out' not in G.fingerprint(opts)


def test_unknown_sentinel_by_dtype():
    a8 = np.array([G.UNK, 0, 5], np.int8); a32 = np.array([np.iinfo(np.int32).min, -128, 5], np.int32)
    assert G.is_unk(a8).tolist() == [True, False, False] and G.is_unk(a32).tolist() == [True, False, False]
    assert G.tern(a32).tolist() == [-1, 0, 1]
    S = np.array([5, 40000, 0], np.int32); K = np.array([True, True, False])
    assert G.pack_sum(S, K).dtype == np.dtype('<i4') and G.pack_sum(S[:1], K[:1]).dtype == np.dtype('<i2')


def test_panels_never_touch_test():
    val, fresh = G.panel_specs('val'), G.panel_specs('fresh')
    assert len(val) == len(fresh) == 24 and all(i < 8 for _, i in val) and all(8 <= i < 16 for _, i in fresh)
    assert sorted({r for r, _ in val}) == [1, 2, 3]
    try:
        G.W.scene_plan('TEST', 1)
        raise RuntimeError('TEST must be closed')
    except ValueError:
        pass


if __name__ == '__main__':
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for k, f in tests:
        f(); print('ok', k, flush=True)
    print(f'all {len(tests)} tests passed')
