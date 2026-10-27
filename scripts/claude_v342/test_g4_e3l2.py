"""unit tests for g4_e3l2.py (batch E3-L2, D8-59). Synthetic arrays and tiny synthetic caches, CPU only, a few minutes.
run:  CUDA_VISIBLE_DEVICES= python -B test_g4_e3l2.py"""
import os
import sys

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, math, tempfile  # noqa: E401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import g4_e3l2 as M  # noqa: E402
import test_g4_e3l as TL  # noqa: E402   (helpers only: fake houses, random trees / maps; read only)

L = M.L
R = M.R
G = M.G
RNG = np.random.default_rng(20261027)
torch.set_num_threads(2)
TABLE = torch.as_tensor(np.arange(R.LMIN, R.LMAX + 1), dtype=torch.int32)


# ---------------------------------------------------------------- the hard-capacity source
def test_topk_and_clamp_ste():
    g = (4, 4, 2); N = 3
    lam = torch.from_numpy(RNG.gamma(1.5, 3.0, (N,) + g)); lam[:, 0] = 0.0                 # x = 0 slab untouched
    lam[1, 1, 0, 0] = 1e-4; lam[2] = 0.0; lam[2, 3, 3, 1] = 40.0                             # a tiny selected Lam, a lone large Lam
    z = torch.from_numpy(RNG.normal(0, 4, (N, 8) + g))
    for K in (1, 3, 7, 1000):
        sel = M.topk_select(lam, K)
        for a in range(N):
            t = lam[a] > 0; n = int(t.sum())
            assert int(sel[a].sum()) == min(K, n) and not (sel[a] & ~t).any()
            if 0 < int(sel[a].sum()) < n:                                                  # the selected are the K largest
                assert float(lam[a][sel[a]].min()) >= float(lam[a][t & ~sel[a]].max())
    K = torch.tensor([5, 1000, 2])
    sel = M.topk_select(lam, K)
    assert [int(sel[a].sum()) for a in range(N)] == [5, int((lam[1] > 0).sum()), 1]
    lam_ = lam.clone().requires_grad_(True); z_ = z.clone().requires_grad_(True)
    hq, lq, sel = M.source_topk(z_, lam_, K, 'ste')
    hr, lr, sel2 = M.source_topk(z, lam, K, 'round')
    assert torch.equal(sel, sel2) and torch.equal(hq.detach(), hr) and torch.equal(lq.detach(), lr)
    s = sel.double(); lam_c = torch.where(sel, lam.clamp(M.LAMQ_MIN, M.LAMQ_MAX), lam)       # review M1 semantics
    assert torch.equal(lr, s * torch.round(lam).clamp(M.LAMQ_MIN, M.LAMQ_MAX)) and lr.max() <= M.LAMQ_MAX   # Lam_q unchanged by M1
    assert torch.equal(hr, s.unsqueeze(1) * torch.round(lam_c.unsqueeze(1) * z).clamp(-M.HQ_LIM, M.HQ_LIM)) and hr.abs().max() <= M.HQ_LIM
    assert float(lr[2].max()) == M.LAMQ_MAX and float(lr[1][1, 0, 0]) == 1.0                # clamped at 15; a tiny selected Lam -> 1
    (hq.sum() * 2 + lq.sum() * 3).backward()
    h = lam_c.unsqueeze(1) * z
    inside = (~sel | ((lam >= M.LAMQ_MIN) & (lam <= M.LAMQ_MAX))) & (h.abs() <= M.HQ_LIM + 0.5).all(1)
    # inside every bound: Lam identity (unselected touched cells too: the selection is learnt through it), h only on the selected
    assert torch.equal(lam_.grad[inside], (3 + 2 * (s.unsqueeze(1) * z).sum(1))[inside])
    assert torch.equal(z_.grad.movedim(1, -1)[inside], (2 * (s * lam_c).unsqueeze(1).expand_as(z)).movedim(1, -1)[inside])
    hf, lf, _ = M.source_topk(z, lam, K, None)                                               # the float potentials of the selected cells use Lam_c
    assert torch.equal(lf, lam_c * s) and torch.allclose(hf, lam_c.unsqueeze(1) * z * s.unsqueeze(1))


def _m1_reference(z, lam, K, qmode):
    """the reviewer's reference (review_e3l2/fix_check.py, source_topk_m1 + _ClampOneSided), copied verbatim in substance"""
    class _C(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, lo, hi):
            ctx.save_for_backward(x); ctx.lo = float(lo); ctx.hi = float(hi)
            return x.clamp(lo, hi)

        @staticmethod
        def backward(ctx, g):
            x, = ctx.saved_tensors
            g = torch.where(x > ctx.hi, g.clamp(min=0.0), g)
            return torch.where(x < ctx.lo, g.clamp(max=0.0), g), None, None
    sel = M.topk_select(lam, K); s = sel.to(lam.dtype)
    lam_c = torch.where(sel, _C.apply(lam, M.LAMQ_MIN, M.LAMQ_MAX), lam)
    h = lam_c.unsqueeze(1) * z
    if qmode is None:
        return h * s.unsqueeze(1), lam_c * s, sel
    lq = (L.DELTA_LAM * torch.round(lam_c / L.DELTA_LAM)) * s
    hq = (L.DELTA_H * torch.round(h / L.DELTA_H)).clamp(-M.HQ_LIM, M.HQ_LIM) * s.unsqueeze(1)
    if qmode == 'round':
        return hq, lq, sel
    hg = _C.apply(h, -M.HQ_LIM - 0.5, M.HQ_LIM + 0.5) * s.unsqueeze(1)
    return hg + (hq - hg).detach(), lam_c + (lq - lam_c).detach(), sel


def test_m1_matches_review_reference():
    """review M1 acceptance: forward and gradients equal the reviewer's reference on the review's 50 random inputs (rng 7)"""
    rng = np.random.default_rng(7)
    for t in range(50):
        N = 3; g = (8, 8, 4)
        lam = torch.from_numpy(rng.gamma(0.7, 6.0, (N,) + g)) * torch.from_numpy(rng.random((N,) + g) < 0.7)
        z = torch.from_numpy(rng.normal(0, 3, (N, 8) + g)); K = torch.from_numpy(rng.integers(1, 200, N))
        for q in ('round', None):
            a = M.source_topk(z, lam, K, q); b = _m1_reference(z, lam, K, q)
            assert all(torch.equal(x, y) for x, y in zip(a, b))
        G1 = torch.from_numpy(rng.normal(0, 1, (N, 8) + g)); G2 = torch.from_numpy(rng.normal(0, 1, (N,) + g))
        grads = []
        for f in (M.source_topk, _m1_reference):
            l_ = lam.clone().requires_grad_(True); z_ = z.clone().requires_grad_(True)
            hq, lq, sel = f(z_, l_, K, 'ste'); ((hq * G1).sum() + (lq * G2).sum()).backward()
            grads.append((hq.detach(), lq.detach(), l_.grad, z_.grad))
        assert all(torch.equal(x, y) for x, y in zip(*grads))
        hq, lq, sel = grads[0][0], grads[0][1], M.topk_select(lam, K)
        assert (lq[sel] >= 1).all() and (hq.abs() <= 7).all()


def test_m1_dead_zone_and_one_sided_gradients():
    """review M1: a selected cell with a tiny raw Lam still sends h_q = round(Lam_c z) (no dead zone); beyond every bound only the
    gradient that pulls the value back passes"""
    lam = torch.zeros(1, 4, 1, 1, dtype=torch.float64); lam[0, :, 0, 0] = torch.tensor([1e-3, 20.0, 5.0, 2.0])
    z = torch.zeros(1, 1, 4, 1, 1, dtype=torch.float64); z[0, 0, :, 0, 0] = torch.tensor([3.0, 0.1, 2.0, 0.3])
    hq, lq, sel = M.source_topk(z, lam, 4, 'round')
    assert sel.all() and lq.flatten().tolist() == [1.0, 15.0, 5.0, 2.0] and hq.flatten().tolist() == [3.0, 2.0, 7.0, 1.0]   # dead-zone regression
    for gsign in (+1.0, -1.0):
        l_ = lam.clone().requires_grad_(True); z_ = z.clone().requires_grad_(True)
        hq, lq, _ = M.source_topk(z_, l_, 4, 'ste')
        (gsign * lq.sum()).backward()
        g = l_.grad.flatten()
        # Lam = 1e-3 < 1: only the raising gradient (g <= 0) passes; Lam = 20 > 15: only the lowering one (g >= 0); inside: identity
        assert float(g[0]) == (gsign if gsign < 0 else 0.0) and float(g[1]) == (gsign if gsign > 0 else 0.0) and float(g[2]) == gsign
    for gsign in (+1.0, -1.0):                                       # |h| = |5 x 2| = 10 > 7.5: only a gradient that shrinks |h| passes
        l_ = lam.clone().requires_grad_(True); z_ = z.clone().requires_grad_(True)
        hq, lq, _ = M.source_topk(z_, l_, 4, 'ste')
        (gsign * hq.sum()).backward()
        gz = z_.grad.flatten()
        assert float(gz[2]) == (gsign * 5.0 if gsign > 0 else 0.0) and float(gz[3]) == gsign * 2.0
    hf, lf, _ = M.source_topk(z, lam, 4, None)                     # float potentials: (Lam_c z s, Lam_c s)
    assert lf.flatten().tolist() == [1.0, 15.0, 5.0, 2.0] and torch.allclose(hf.flatten(), torch.tensor([3.0, 1.5, 10.0, 0.6], dtype=torch.float64))


def test_m1_toy_training_keeps_hq():
    """review M1: a toy that keeps training (Adam, 300 steps): a bounded latent (3 tanh, like an encoder output), a target that needs the
    latent, and a strong steady pull on Lam downwards. The raw Lam of the selected cells falls deep into the old dead zone (median < 0.5),
    yet the selected cells keep sending h_q != 0 and the latent keeps its gradient; at the same final state the pre-M1 formula
    (h_q = round(Lam_raw z)) would send almost nothing"""
    torch.manual_seed(0)
    a = torch.full((1, 16, 4, 2), 1.0, requires_grad=True); zz = torch.randn(1, 8, 16, 4, 2, requires_grad=True)
    target = torch.randn(1, 8, 16, 4, 2) * 2
    opt = torch.optim.Adam([a, zz], lr=0.05); nz = []; gz = []
    for _ in range(300):
        lam = torch.nn.functional.softplus(L.floor_one_sided(a)); z = 3 * torch.tanh(zz)
        hq, lq, sel = M.source_topk(z, lam, 64, 'ste')
        loss = ((hq / lq.clamp(min=1).unsqueeze(1) - target) ** 2).mean() + 5.0 * lam.mean()
        opt.zero_grad(); loss.backward(); opt.step()
        nz.append(float((hq.detach().movedim(1, -1)[sel] != 0).float().mean())); gz.append(float(zz.grad.abs().sum()))
    lam = lam.detach(); z = z.detach()
    assert float(lam[sel].median()) < 0.5                                         # the toy did enter the old dead zone
    assert min(nz[50:]) > 0.5 and min(gz[50:]) > 0.1, (min(nz[50:]), min(gz[50:]))
    legacy = torch.round(lam.unsqueeze(1) * z).clamp(-M.HQ_LIM, M.HQ_LIM).movedim(1, -1)[sel]
    assert float((legacy != 0).float().mean()) < 0.1 < nz[-1]


def test_selected_lamq_at_least_one():
    """D8-59: every selected cell has Lam_q in [1, 15] (no collapse), whatever the encoder outputs; unselected cells carry nothing"""
    for scale in (1e-6, 0.3, 2.0, 1e3):
        lam = torch.from_numpy(RNG.random((4, 8, 8, 4)) * scale) * torch.from_numpy(RNG.random((4, 8, 8, 4)) < 0.6)
        z = torch.from_numpy(RNG.normal(0, 10, (4, 8, 8, 8, 4)))
        for K in (16, 128):
            hq, lq, sel = M.source_topk(z, lam, K, 'round')
            assert (lq[sel] >= 1).all() and (lq[sel] <= 15).all() and not lq[~sel].any() and not hq.permute(0, 2, 3, 4, 1)[~sel].any()
            hi, li = L.msg_ints(hq[0], lq[0])                                                # a valid O7 wire message
            assert int((li > 0).sum()) == int(sel[0].sum()) and L.wire_bytes(hi, li) > 0


def test_padding_not_in_gabp_and_fixed_shape():
    """fixed-shape blocks of 8 with zero padding: the real rows equal an unpadded encoding, the padded rows never reach GaBP, and the
    whole batch forward equals the unchunked one"""
    torch.manual_seed(0)
    model = M.E3L2(width=8).eval()
    H = TL.fake_house(n=5, edges=((0, 1), (1, 2), (2, 3), (3, 4)))
    f = torch.from_numpy(RNG.normal(0, 1, (11, 3, 16, 16, 8)).astype(np.float32)); kn = torch.from_numpy(RNG.random((11, 16, 16, 8)) < 0.4)
    with torch.no_grad():
        z1, l1, P = M.encode_fixed(model, f, kn, 8); z0, l0, P0 = M.encode_fixed(model, f, kn, None)
    assert P == 5 and P0 == 0 and z1.shape[0] == 11 and torch.allclose(z1, z0, atol=1e-5) and torch.allclose(l1, l0, atol=1e-5)
    specs = [(0, 1, (0, 1, 2), 0, 1, 200), (0, 0, (3, 4), 5, 4, 3), (0, 1, (2,), 2, 2, 128)]
    with torch.no_grad():
        a = M.forward_batch(model, specs, [H], TABLE, 'round', chunk=8); b = M.forward_batch(model, specs, [H], TABLE, 'round', chunk=None)
    assert a[3]['npad'] == 2 and a[3]['n'] == 6 and a[3]['hq'].shape[0] == 6 and a[3]['dec'] == [1, 4, 5] and a[3]['K'] == [200] * 3 + [3] * 2 + [128]
    assert torch.allclose(a[0], b[0], atol=1e-5)
    called = []
    orig = L.gabp
    L.gabp = lambda h, lam, Fl, w, q=None: (called.append(h.shape[0]), orig(h, lam, Fl, w, q))[1]
    try:
        with torch.no_grad():
            M.forward_batch(model, specs, [H], TABLE, 'round', chunk=8)
    finally:
        L.gabp = orig
    assert called == [3, 2, 1]                                                               # GaBP only on the samples' agents


def test_forward_equals_e3l_when_unconstrained():
    """K >= the touched cells and no clamp: the E3-L2 source and the whole training forward equal E3-L's (same weights)"""
    torch.manual_seed(3)
    e3l = L.E3L(width=8)
    m2 = M.E3L2(width=8)
    miss = m2.load_state_dict({k: v for k, v in e3l.state_dict().items() if not k.startswith('ent.')}, strict=True)
    assert not miss.missing_keys and not miss.unexpected_keys
    H = TL.fake_house()
    specs5 = [(0, 1, (0, 1, 2), 0, 1), (0, 0, (3,), 5, 3), (0, 1, (1, 2), 3, 2)]
    lg5, yk5, yo5, s5 = L.forward_batch(e3l, specs5, [H], TABLE, 'ste')
    lg6, yk6, yo6, s6 = M.forward_batch(m2, [s + (10 ** 6,) for s in specs5], [H], TABLE, 'ste', clamp=False)
    assert torch.equal(s5['hq'].detach(), s6['hq'].detach()) and torch.equal(s5['lq'].detach(), s6['lq'].detach())
    assert torch.allclose(lg5, lg6, atol=1e-5) and torch.equal(yk5, yk6) and torch.equal(yo5, yo6)
    with torch.no_grad():                                                                     # and the clamp is what makes them differ
        z, lam = m2.encode(R.features_t(*L.evidence_t(H.t_maps[1][[0, 1]], TABLE)[:1], (H.t_maps[1][[0, 1]] != G.UNK).int()), H.t_maps[1][[0, 1]] != G.UNK)
        hqc, lqc, sel = M.source_topk(z, lam, 10 ** 6, 'round', clamp=True)
        hqe, lqe, _, _ = e3l.source(z, lam, 'round')
    assert (lqc[sel] >= 1).all() and torch.equal(lqc[sel], lqe[sel].clamp(M.LAMQ_MIN, M.LAMQ_MAX)) and torch.equal(sel, lam > 0)


def test_e3r_is_special_case():
    """kept from E3-L: Enc = identity (z = evidence, voxel grid, D = 1), Lam = count, w -> inf, integer wire: beliefs = E3-R's (S, k);
    here through the E3-L2 source with K >= every touched cell and no clamp"""
    n = 5; shape = (8, 8, 4)
    Fl = G.forest([(0, 1, 1.), (1, 2, 1.), (1, 3, 1.), (3, 4, 1.)], n)
    cq = RNG.integers(-100, 150, R.NL)
    maps = [TL.rand_map(shape, 0.5) for _ in range(n)]
    st = [R.stalk(Lm, cq) for Lm in maps]
    e = torch.from_numpy(np.stack([s for s, _ in st]).astype(np.float64)).unsqueeze(1); k = torch.from_numpy(np.stack([kk for _, kk in st]).astype(np.float64))
    S, kc = R.direct_sums(maps, list(range(n)), cq)
    hs, ls, sel = M.source_topk(e, k, 10 ** 9, 'round', clamp=False)
    assert torch.equal(sel, k > 0) and torch.equal(ls, k) and torch.equal(hs, k.unsqueeze(1) * e)
    hq, lq, msgs = L.gabp(hs, ls, Fl, 1e15, 'round')
    for a in range(n):
        assert np.array_equal(lq[a].numpy().astype(np.int64), kc) and np.array_equal(hq[a, 0].numpy().astype(np.int64), S)
        fx = R.features_np(np.rint(hq[a, 0].numpy()).astype(np.int32), lq[a].numpy().astype(np.int32))
        assert np.array_equal(fx, R.features_np(S, kc))
    for (s, r), (hm, lm) in msgs.items():
        side = Fl['sides'][(s, r)]
        assert np.array_equal(lm.numpy().astype(np.int64), sum(st[b][1].astype(np.int64) for b in side))


def test_gabp_exact_on_selected_cells():
    """the GaBP exactness / energy checks of E3-L run on the float potentials of the selected cells"""
    for t in range(10):
        m = 2 + t % 6
        Fl, lam, z, w = TL.rand_case(m, (4, 4, 2), 8, RNG)
        hf, lf, sel = M.source_topk(z, lam, 5 + t, None)
        c = L.gbp_checks(z, lf, Fl, w)
        assert c['cf_max'] <= L.CF_TOL and c['violations'] == 0


def test_k_sampling():
    rng = np.random.default_rng(1)
    ks = np.array([M.sample_k(rng) for _ in range(20000)])
    assert ks.min() >= 128 and ks.max() <= 2048 and set(np.unique(ks)) <= set(range(128, 2049))
    lk = np.log(ks)
    h, _ = np.histogram(lk, bins=8, range=(math.log(128), math.log(2048)))
    assert (np.abs(h / h.sum() - 1 / 8) < 0.01).all()                                         # log-uniform
    H = TL.fake_house()
    sp = [M.sample_spec(rng, [H]) for _ in range(200)]
    assert all(len(s) == 6 and 128 <= s[5] <= 2048 for s in sp)


# ---------------------------------------------------------------- K selection, thresholds, timing
def test_choose_k_rule():
    r = {str(K): v for K, v in zip(M.K_GRID, (0.04, 0.05, 0.07, 0.09, 0.11, 0.12, 0.13, 0.17, 0.2))}
    assert M.choose_k(r) == 768
    r['768'] = 0.1201
    assert M.choose_k(r) == 512
    assert M.choose_k({str(K): 0.2 for K in M.K_GRID}) is None
    assert M.choose_k(r, cap=0.15) == 1024 and M.BYTES_SELECT == 0.12


def test_threshold_rules():
    assert abs(M.occ_free_min(0.9975) - 0.9950) < 1e-12 and M.GUARD_FPR_SELECT == 0.0015
    # pick_occ with the -0.25 pp floor: a threshold with free recall control - 0.2 pp is admissible, - 0.3 pp is not
    A = np.zeros(L.NBIN); B = np.zeros(L.NBIN); C = np.zeros(L.NBIN)
    A[2100] = 50; B[2100] = 1; C[1950] = 997; C[2050] = 3                     # 1000 free voxels: bins >= 2051 keep all, >= 1951 keep 3
    j, feas, f1, fr = L.pick_occ(A, B, C, 60, 1000, M.occ_free_min(0.999))     # floor .9965: needs j > 2050 (free 1.0), or j in (1950, 2050] (.997)
    assert feas and fr[j] >= 0.9965 and j <= 1951 + 100
    j2, feas2, _, fr2 = L.pick_occ(A, B, C, 60, 1000, 1.01)
    assert not feas2 and fr2[j2] == fr2.max()
    r8 = dict(j_known=2010, j_occ=1990, t_known=0.2, t_occ=-0.2)
    k0 = M.k0_thresholds(dict(j_known=2005, j_occ=1980, feasible_known=True, feasible_occ=False, t_known=.1, t_occ=-.4), r8)
    assert k0['k0_fallback_r8'] and (k0['j_known'], k0['j_occ']) == (2010, 1990) and k0['own'] == dict(j_known=2005, j_occ=1980)
    k0 = M.k0_thresholds(dict(j_known=2005, j_occ=1980, feasible_known=True, feasible_occ=True, t_known=.1, t_occ=-.4), r8)
    assert not k0['k0_fallback_r8'] and (k0['j_known'], k0['j_occ']) == (2005, 1980)


def test_step_rate():
    marks = {50 * i: 0.3 + 0.16 * 50 * (i - 1) for i in range(1, 9)}                      # first window slow (autotune), then 0.16 s / update
    t = M.step_rate(marks, 200)
    assert abs(t['rate'] - 0.16) < 1e-9 and t['cv'] < 1e-9 and t['warning'] is None and [w[0] for w in t['windows']] == [200, 250, 300, 350]
    marks[400] = marks[350] + 50 * 0.30                                                      # one slow window -> CV >= 10 % -> median
    t = M.step_rate(marks, 200)
    assert t['warning'] and abs(t['rate'] - 0.16) < 1e-9 and t['mean_rate'] > 0.18
    assert M.step_rate({}, 200)['rate'] is None
    t = M.step_rate({50 * i: 0.16 * 50 * i for i in range(1, 5)}, 200)                      # review S5: too few windows -> the start < 200 is flagged
    assert t['window_start'] < 200 and not t['window_start_ok'] and 'start at update' in t['warning']
    t = M.step_rate({50 * i: 0.16 * 50 * i for i in range(1, 9)}, 200)
    assert t['window_start'] == 200 and t['window_start_ok'] and t['warning'] is None


# ---------------------------------------------------------------- verdict (D8-59)
def _row(furn, free=.997, surf=.99, fpr=.018, k0=.94):
    return dict(R8=dict(furn_f1=furn, furn_f1_tol1=furn + .01, free_recall=free, surface_f1=surf, unknown_known_fpr=fpr), K0=dict(furn_f1=k0), K0_sep=dict(furn_f1=k0))


def _summ(d=-.005, ratio=.12, d_fresh=-.004, d_cam=-.01, clean=.995, k0=-.01, missing=(), gbp_ok=True):
    Sl = dict(panels={}, failed=[], skipped=[], e1_mismatch=[], agg_reasons=[],
              gbp={f'E3L2_{t}': dict(cf_max=1e-12 if gbp_ok else 1e-3, violations=0, components=9, **{'pass': gbp_ok}) for t in ('342101', '342202')})
    Sc = dict(panels={})
    for panel, dd in (('val', d), ('fresh', d_fresh)):
        houses = [f'r{r}_i{i}' for r, i in G.panel_specs(panel)]
        Pl = dict(seeds={}); Pc = dict(seeds={})
        for ns in ('1', '2'):
            tl = dict(cam_nf={}, cam={}, clean={}); tc = dict(cam_nf={}, cam={}, clean={})
            for ts in ('342101', '342202'):
                c = _row(.97); c['rnet_relay_median'] = dict(sum=665000., count=88000.)
                tc['cam_nf'][f'NET_{ts}'] = c; tc['cam'][f'NET_{ts}'] = _row(.80); tc['clean'][f'NET_{ts}'] = _row(.997)
                dv = dd[(ts, ns)] if isinstance(dd, dict) else dd
                if (panel, ts, ns) not in missing:
                    a = _row(.97 + dv, k0=.94 + k0); a['bytes_median'] = ratio * 753000.; a['k_star'] = [512]
                    a['curve'] = {'128': dict(R8=_row(.9)['R8'], bytes_median=.03 * 753000.)}
                    tl['cam_nf'][f'E3L2_{ts}'] = a
                tl['cam'][f'E3L2_{ts}'] = _row(.80 + d_cam); tl['clean'][f'E3L2_{ts}'] = _row(clean)
            Pl['seeds'][ns] = dict(houses=houses, tiers=tl); Pc['seeds'][ns] = dict(houses=houses, tiers=tc)
        Sl['panels'][panel] = Pl; Sc['panels'][panel] = Pc
    return Sl, Sc


def _runs(d, done=(15000, 15000), planned=(15000, 15000), truncated=(False, False), kstar=(512, 512), thr=(True, True)):
    out = []
    for i, sd in enumerate(('342101', '342202')):
        r = os.path.join(d, f'run{sd}'); os.makedirs(r, exist_ok=True)
        M._jdump(dict(seed=int(sd), train_houses=186, recipe=dict(evidence=dict(desc=dict(spec='c:x:A:cam_nf')))), os.path.join(r, 'run.json'))
        M._jdump(dict(complete=(done[i] == planned[i] and not truncated[i]), truncated=truncated[i], updates_done=done[i], updates_planned=planned[i]),
                 os.path.join(r, 'final.json'))
        M._jdump(dict(k_star=kstar[i], smallest_k=128, per_k={str(K): dict(median_ratio=0.05 * (j + 1)) for j, K in enumerate(M.K_GRID)}), os.path.join(r, 'kselect.json'))
        if thr[i]:
            M._jdump(dict(k_star=kstar[i], n_houses=19, R8={str(kstar[i]): dict(j_known=2000)}, K0=dict(j_known=2000)), os.path.join(r, 'threshold.json'))
        out.append(r)
    return out


def test_verdict_d8_59():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        Rn = _runs(d)

        def V(Sl, Sc, runs=Rn, full=False):
            p = os.path.join(d, 'ctrl.json'); M._jdump(Sc, p)
            v = M.verdict(Sl, runs, p)
            return v if full else v['verdict']
        assert V(*_summ()) == 'LEVER'
        v = V(*_summ(), full=True)
        assert all(c['k_star'] == [512] and abs(c['bytes_ratio'] - .12) < 1e-12 and '128' in c['curve'] for c in v['cells'].values())
        assert v['stages']['342101']['k_star'] == 512
        assert V(*_summ(d=-.0105)) == 'AMBIGUOUS' and V(*_summ(d=-.031)) == 'NOT_ENOUGH'
        assert V(*_summ(ratio=.151)) == 'AMBIGUOUS' and V(*_summ(ratio=.501)) == 'NOT_ENOUGH' and V(*_summ(ratio=.15)) == 'LEVER'
        assert V(*_summ(d_cam=-.031)).startswith('AMBIGUOUS (lever demoted: cam')
        assert V(*_summ(d_fresh=-.0105)).startswith('AMBIGUOUS (lever demoted: fresh') and V(*_summ(d_fresh=+.01)) == 'LEVER'
        assert V(*_summ(d=-.031, d_fresh=-.05)) == 'NOT_ENOUGH' and V(*_summ(d=-.0105, d_fresh=.05)) == 'AMBIGUOUS'
        assert V(*_summ(clean=.989)) == 'AMBIGUOUS' and V(*_summ(k0=-.031)) == 'AMBIGUOUS' and V(*_summ(gbp_ok=False)) == 'AMBIGUOUS'
        Sl, Sc = _summ(); Sl['panels']['val']['seeds']['1']['tiers']['cam_nf']['E3L2_342101']['R8']['unknown_known_fpr'] = .018 + .0031
        assert V(Sl, Sc) == 'AMBIGUOUS'
        Sl, Sc = _summ(); Sl['panels']['val']['seeds']['2']['tiers']['cam_nf']['E3L2_342202']['R8']['free_recall'] = .997 - .0051
        assert V(Sl, Sc) == 'AMBIGUOUS'
        # INCOMPLETE: K* missing (no K <= 0.12), threshold missing, < 2,000 updates, different counts, missing cells / control / K0 rows
        assert V(*_summ(), runs=_runs(os.path.join(d, 'k'), kstar=(512, None))).startswith('INCOMPLETE')
        assert 'no K with holdout median byte ratio' in V(*_summ(), runs=_runs(os.path.join(d, 'k2'), kstar=(None, 512)))
        assert V(*_summ(), runs=_runs(os.path.join(d, 't'), thr=(True, False))).startswith('INCOMPLETE')
        assert V(*_summ(), runs=_runs(os.path.join(d, 'u'), done=(15000, 1999), truncated=(False, True))).startswith('INCOMPLETE')
        assert V(*_summ(), runs=_runs(os.path.join(d, 'p'), done=(15000, 14000), planned=(15000, 14000))).startswith('INCOMPLETE')
        assert V(*_summ(), runs=_runs(os.path.join(d, 'tr'), done=(15000, 12000), truncated=(False, True))) == 'AMBIGUOUS'
        assert V(*_summ(missing={('val', '342101', '1')})).startswith('INCOMPLETE')
        assert V(*_summ(missing={('fresh', '342202', '2')})).startswith('INCOMPLETE')
        Sl, Sc = _summ(); del Sc['panels']['val']['seeds']['1']['tiers']['cam_nf']['NET_342202']
        assert V(Sl, Sc).startswith('INCOMPLETE')
        Sl, Sc = _summ(); del Sl['panels']['val']['seeds']['1']['tiers']['cam_nf']['E3L2_342202']['K0_sep']
        assert V(Sl, Sc).startswith('INCOMPLETE')
        Sl, Sc = _summ(); del Sl['gbp']['E3L2_342101']
        assert V(Sl, Sc).startswith('INCOMPLETE')
        assert M.verdict(_summ()[0], Rn, None)['verdict'].startswith('INCOMPLETE')
        assert M.D8_59['lever_bytes_ratio'] == 0.15 and M.D8_59['lever_furn_margin'] == -0.01 and M.D8_59['fresh_min_mean'] == -0.01
        assert M.D8_59['not_enough_bytes_ratio'] == 0.5 and M.D8_59['k0_margin'] == -0.03 and M.D8_59['guard_fpr'] == 0.003


# ---------------------------------------------------------------- end to end on a synthetic cache (train -> ctrlprep -> kselect -> select -> eval)
def test_end_to_end_tiny():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        cache, houses = TL._write_cache(d)
        fit = os.path.join(d, 'fit.json'); cq = RNG.integers(-60, 90, R.NL)
        tb = dict(cq=list(map(int, cq)), theta=5, l0q=-3, tq=2, l0=-0.15)
        with open(fit, 'w') as fh:
            json.dump(dict(complete=True, halves=dict(A=dict(tiers=dict(cam_nf=tb)), B=dict(tiers=dict(cam_nf=dict(tb, cq=[0] * R.NL))))), fh)
        ev = f'c:{fit}:A:cam_nf'; man_sha = R.file_sha256(os.path.join(cache, 'manifest.json'))
        e3r = os.path.join(d, 'e3r')
        R.main(['train', '--cache', cache, '--evidence', ev, '--seed', '342101', '--out', os.path.join(e3r, 's342101'), '--updates', '2', '--device', 'cpu',
                '--batch', '1', '--log-every', '1', '--val-every', '2', '--holdout-samples', '1', '--warmup', '1'])
        out = os.path.join(d, 's342101')
        fin = M.main(['train', '--cache', cache, '--evidence', ev, '--seed', '342101', '--out', out, '--updates', '4', '--device', 'cpu', '--batch', '2',
                      '--width', '8', '--log-every', '1', '--val-every', '4', '--holdout-samples', '2', '--warmup', '1', '--manifest-sha', man_sha,
                      '--rate-every', '2', '--time-from', '2', '--snap', '3', '--ctrl-ckpt', os.path.join(e3r, 's342101', 'checkpoints', 'final.pt')])
        assert fin['complete'] and fin['updates_done'] == 4 and fin['snapshot']['updates'] == 3 and fin['rate_s_per_update'] > 0
        assert os.path.exists(os.path.join(out, 'checkpoints', 'upd3.pt')) and fin['snapshot']['sha256'] == R.file_sha256(os.path.join(out, 'checkpoints', 'upd3.pt'))
        rc = json.load(open(os.path.join(out, 'rate_curve.json')))
        for k in ('hq_nonzero', 'lam_clip_hi', 'lam_clip_lo', 'h_clip', 'lam_raw_median', 'grad_encoder', 'grad_total'):   # review S6
            assert all(k in r for r in rc), k
        hist = json.load(open(os.path.join(out, 'history.json')))
        assert all('hq_nonzero' in h and 'grad_encoder' in h for h in hist) and 'rate_window_start_ok' in fin
        assert [r['updates'] for r in rc] == [1, 2, 4] and all(len(r['K']) == 2 and r['support_cells'] <= r['touched_cells'] for r in rc)
        ctrl = os.path.join(d, 'ctrl.json')
        cp = M.main(['ctrlprep', '--cache', cache, '--e3r', e3r, '--out', ctrl, '--smoke', '--seeds-smoke', '342101', '--workers', '1'])
        assert cp['houses'] == ['TRAIN_r1_i9'] and set(cp['bytes']['342101']['TRAIN_r1_i9']) == {'cam_nf_s200'} and 0 < cp['readouts']['342101']['R8']['free'] <= 1
        bad = json.load(open(ctrl)); bad['manifest_sha256'] = '0' * 64; badp = os.path.join(d, 'ctrl_bad.json'); M._jdump(bad, badp)
        for cmd in ('kselect', 'select'):                                                       # review S1
            try:
                M.main([cmd, '--run', out, '--ctrl', badp, '--workers', '1', '--smoke'] + (['--bytes-max', '100'] if cmd == 'kselect' else []))
            except (AssertionError, FileNotFoundError) as e:
                assert cmd == 'select' or 'manifest' in str(e)
            else:
                raise AssertionError(f'{cmd} must refuse a control file of another cache')
        bad = json.load(open(ctrl)); bad['e3r']['342101']['ckpt_sha256'] = '0' * 64; M._jdump(bad, badp)
        try:
            M.main(['kselect', '--run', out, '--ctrl', badp, '--workers', '1', '--smoke', '--bytes-max', '100'])
        except AssertionError as e:
            assert 'checkpoint' in str(e)
        else:
            raise AssertionError('kselect must refuse a control file whose E3-R checkpoint sha differs')
        ks = M.main(['kselect', '--run', out, '--ctrl', ctrl, '--workers', '1', '--smoke', '--bytes-max', '100'])
        assert ks['k_star'] == max(M.K_GRID) and all(ks['per_k'][str(K)]['median_bytes'] > 0 for K in M.K_GRID)
        th = M.main(['select', '--run', out, '--ctrl', ctrl, '--workers', '1', '--smoke', '--kset', 'all'])
        assert th['k_star'] == ks['k_star'] and set(th['R8']) == {str(K) for K in M.K_GRID} and 'k0_fallback_r8' in th['K0']
        assert all(abs(v['free_min'] - (cp['readouts']['342101']['R8']['free'] - 0.0025)) < 1e-12 for v in th['R8'].values())
        assert all(abs(v['fpr_max'] - (cp['readouts']['342101']['R8']['fpr'] + 0.0015)) < 1e-12 for v in th['R8'].values())
        M.main(['kselect', '--run', out, '--ctrl', ctrl, '--name', 'upd3', '--workers', '1', '--smoke', '--bytes-max', '100'])
        ts = M.main(['select', '--run', out, '--ctrl', ctrl, '--name', 'upd3', '--kset', 'kstar', '--workers', '1', '--smoke'])
        assert list(ts['R8']) == [str(ts['k_star'])] and os.path.exists(os.path.join(out, 'threshold_upd3.json'))
        ex = os.path.join(d, 'export'); os.makedirs(os.path.join(ex, 'records'))
        H = TL.fake_house(noisy_draws=1, rng=np.random.default_rng(42))
        arr = dict(occ=H.occ.astype(np.int8), slab=H.slab, obj=H.obj, cell=H.cell, private=H.private, edges=R.edges_array(H.edges))
        for tn in R.EVAL_TIERS:
            arr[f'L_k0_{tn}'] = H.maps[0] if tn == 'clean' else H.maps[1]
        arr['L_pub_cam'] = H.maps[1][::-1].copy()
        fp = os.path.join(ex, 'train_r1_i0_s203.npz'); np.savez_compressed(fp, **arr)
        rec_e = dict(panel='train', rooms=1, index=0, seed=203, split='TRAIN', file='train_r1_i0_s203.npz', file_sha256=R.file_sha256(fp), components=H.comps)
        q = dict(Qinf=cq.astype(np.int64), Q3=np.sign(cq).astype(np.int64) * 10, VOTE=np.sign(cq).astype(np.int64))
        ckf, cks = os.path.join(out, 'checkpoints', 'final.pt'), os.path.join(out, 'checkpoints', 'upd3.pt')
        rec = M.eval_job((rec_e, dict(export=ex, ckpts=[ckf], snaps=[cks], thr={ckf: os.path.join(out, 'threshold.json'), cks: os.path.join(out, 'threshold_upd3.json')},
                                      ckpt_shas={}, tiers=list(R.EVAL_TIERS), threads=1, chunk=2, script_sha256='x', qtables=q)))
        assert not rec.get('failed'), rec.get('traceback')
        a = rec['tiers']['cam_nf']['E3L2_342101']
        assert a['k_star'] == ks['k_star'] and len(a['R8']) == H.n and len(a['K0_sep']) == H.n and a['bytes'] > 0 and a['bytes'] <= a['bytes_upper']
        assert a['gbp']['violations'] == 0 and a['gbp']['cf_max'] <= L.CF_TOL and set(a['curve']) == {str(K) for K in M.K_GRID if K != a['k_star']}
        assert 'E3L2s_342101' in rec['tiers']['cam_nf'] and 'E3L2s_342101' not in rec['tiers']['cam'] and 'curve' not in rec['tiers']['cam']['E3L2_342101']
        zl = rec['tiers']['cam_nf']['zero_learning']
        assert abs(zl['Qinf']['ratio'] - zl['Qinf']['fr1'] / zl['ctrl_qinf']) < 1e-12 and zl['VOTE']['fr1'] < zl['Qinf']['fr1']
        os.makedirs(os.path.join(d, 'ev', 'jobs')); M._jdump(rec, os.path.join(d, 'ev', 'jobs', 'train_r1_i0_s203.json'))
        S = M.aggregate(os.path.join(d, 'ev'))
        e = S['panels']['train']['seeds']['203']['tiers']['cam_nf']['E3L2_342101']
        assert S['gbp']['E3L2_342101']['pass'] and e['bytes_median'] == a['bytes'] and len(e['curve']) == len(M.K_GRID) - 1 and 'zero_learning' in S['panels']['train']['seeds']['203']['tiers']['cam_nf']
        assert M.verdict(S, [out], None, smoke=True)['verdict'].startswith('INCOMPLETE')
        # no K meets the byte cap: kselect records K* = None (INCOMPLETE), select still reads out the smallest K (flagged, report only)
        ks0 = M.main(['kselect', '--run', out, '--ctrl', ctrl, '--name', 'upd3', '--workers', '1', '--smoke', '--bytes-max', '1e-9'])
        assert ks0['k_star'] is None
        t0 = M.main(['select', '--run', out, '--ctrl', ctrl, '--name', 'upd3', '--kset', 'kstar', '--workers', '1', '--smoke'])
        assert t0['k_star'] == min(M.K_GRID) and t0['k_star_missing'] and list(t0['R8']) == [str(min(M.K_GRID))]


def test_chain_script_guards():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chain_day7_E3L2.sh'), encoding='utf-8') as fh:
        c = fh.read()
    assert 'export PYTHONPATH=/mnt/datasets/yuzhou-sheaf-gbp-research/e3r-pyroot:/mnt/datasets/yuzhou-sheaf-gbp-research/e3r-pyroot/exploration_grid' in c
    i_self = c.index('import self-check'); i_busy = c.index('while busy; do sleep 120; done'); i_lease = c.index('START=$(date +%s)')
    i_end = c.index('lease over after'); i_ks = c.index('g4_e3l2.py kselect'); i_sel = c.index('g4_e3l2.py select')
    assert i_self < i_busy < i_lease < i_end < i_ks < i_sel, 'K and thresholds must be selected after the lease'
    assert '--max-updates 400' in c and '--time-from 200' in c and '0.90 * avail' in c and '--reserve $SEL' not in c and '\nSEL=' not in c and '--snap 6510' in c
    assert 'BUDGET=3000' in c and '$((START+6600))' in c and '$((START+7150))' in c and '-lt 2000' in c and 'D8-59' in c and 'OUT=runs/e3l2' in c
    assert 'pkill' not in c and '--resume' not in c and 'export --' not in c
    assert '[ -e $OUT ]' in c and 'mkdir $OUT.lock' in c and 'grep -q "FILL_"' in c
    for k in ('E3L2_SHA', 'E3L_SHA', 'E3R_SUM_SHA', 'TH1_SHA', 'TH2_SHA', 'EXPREC_SHA', 'CTRLH_SHA', 'E3L_SUM_SHA', 'QFILE_SHA', 'CK1_SHA', 'CK2_SHA', 'MAN_SHA', 'PYR_SHA', 'G_SHA'):
        assert f'"${k}"' in c, k
    assert 'E3L_SHA=' + L.file_sha256(L.__file__) in c, 'the frozen g4_e3l.py sha pinned in the chain must be the imported one'


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t(); print('ok', t.__name__, flush=True)
    print(f'{len(tests)}/{len(tests)} passed')
