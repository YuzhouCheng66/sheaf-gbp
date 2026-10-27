"""unit tests for g4_e3l.py (batch E3-L, D8-54). Synthetic arrays, CPU only, a few minutes.
run:  CUDA_VISIBLE_DEVICES= python -B test_g4_e3l.py"""
import os
import sys

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, tempfile  # noqa: E401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import g4_e3l as L  # noqa: E402

R = L.R
G = L.G
RNG = np.random.default_rng(20261027)
torch.set_num_threads(2)


# ---------------------------------------------------------------- helpers
def rand_tree(m, rng):
    return [(i, int(rng.integers(i)), 1.0) for i in range(1, m)]


def rand_case(m, g, D, rng, p_zero=0.4, w=None):
    Fl = G.forest(rand_tree(m, rng), m)
    lam = rng.gamma(1.5, 2.0, (m,) + g) * (rng.random((m,) + g) > p_zero)
    lam[:, 0, 0, 0] = 0.0                                                  # one cell without any information
    z = rng.normal(0, 3, (m, D) + g)
    return Fl, torch.from_numpy(lam), torch.from_numpy(z), (w if w is not None else float(rng.choice([0.3, 3.0, 1000.0])))


def rand_map(shape, p_unk=0.4, rng=RNG):
    Lm = rng.integers(R.LMIN, R.LMAX + 1, size=shape).astype(np.int8)
    Lm[rng.random(shape) < p_unk] = G.UNK
    return Lm


def fake_house(n=4, shape=(16, 16, 8), edges=((0, 1), (1, 2)), noisy_draws=1, rng=RNG):
    """an LHouse-like object (occ, private maps consistent with occ, clean / noisy maps, forest)"""
    H = L.LHouse.__new__(L.LHouse)
    occ = np.zeros(shape, bool); occ[:, :, 0] = True; occ[0] = True
    for _ in range(5):
        x, y = rng.integers(2, shape[0] - 3, 2); occ[x:x + 2, y:y + 2, 1:3] = True
    obj = occ.copy(); obj[:, :, 0] = False; obj[0] = False
    slab = np.zeros(shape, bool); slab[:, :, 0] = True
    private = np.full((n,) + shape, -1, np.int8)
    for a in range(n):
        kn = rng.random(shape) < 0.5; kn[:, :, 4:] = False                 # the upper half never known: empty latent blocks
        private[a][kn] = occ[kn].astype(np.int8)
    clean = np.where(private == 1, np.int8(70), np.where(private == 0, np.int8(-40), np.int8(G.UNK))).astype(np.int8)
    maps = [clean] + [np.where(private >= 0, rand_map((n,) + shape, 0.2, rng), np.int8(G.UNK)).astype(np.int8) for _ in range(noisy_draws)]
    H.name = 'fake'; H.private = private; H.n = n; H.draws = ['clean'] + [f'cam_nf_s{200 + i}' for i in range(noisy_draws)]
    H.maps = np.stack(maps); H.occ = occ; H.slab = slab; H.obj = obj; H.cell = np.array([0.1, 0.1, 0.1])
    H.edges = [(int(i), int(j), 1.0) for i, j in edges]; H.F = G.forest(H.edges, n); H.comps = [list(c) for c in H.F['comps']]; H._fl = {}
    pk = np.zeros(shape, np.uint8)
    for a in range(n):
        pk |= ((private[a] >= 0).astype(np.uint8) << a)
    H.pk = pk
    return H.to('cpu')


# ---------------------------------------------------------------- GaBP
def test_gabp_equals_closed_form():
    for t in range(40):
        m = 1 + t % 8
        Fl, lam, z, w = rand_case(m, (5, 4, 3), 3, RNG)
        hb, lb, msgs = L.gabp(lam.unsqueeze(1) * z, lam, Fl, w, None)
        assert len(msgs) == 2 * (m - 1)
        x = L.belief_mean(hb, lb).reshape(m, 3, -1).numpy()
        xc, inf = L.closed_form(z.reshape(m, 3, -1).numpy(), lam.reshape(m, -1).numpy(), Fl, w)
        assert not inf[0] and np.abs(x[:, :, inf] - xc[:, :, inf]).max() <= 1e-9, (m, w)
        assert np.allclose(lb.reshape(m, -1).numpy()[:, ~inf], 0) and np.allclose(x[:, :, ~inf], 0)
        c = L.gbp_checks(z, lam, Fl, w)
        assert c['cf_max'] <= L.CF_TOL and c['violations'] == 0
    # a message depends only on its sender side: every schedule that respects it gives the same messages (two passes == relay order)
    Fl, lam, z, w = rand_case(6, (3, 3, 2), 2, RNG, w=2.0)
    _, _, m1 = L.gabp(lam.unsqueeze(1) * z, lam, Fl, w, None)
    for (s, r), (hm, lm) in m1.items():
        side = Fl['sides'][(s, r)]
        sub = sorted(side + [r]); Fs = L.local_forest(Fl, sub)                 # the sender side plus the receiver with no evidence
        lam2 = lam[sub].clone(); lam2[sub.index(r)] = 0
        _, _, m2 = L.gabp(lam2.unsqueeze(1) * z[sub], lam2, Fs, w, None)
        assert torch.allclose(m2[(sub.index(s), sub.index(r))][1], lm) and torch.allclose(m2[(sub.index(s), sub.index(r))][0], hm)


def test_closed_form_scope_tiny_precision():
    """smoke regression: softplus can give Lam ~ 1e-40; such cells are excluded from the closed-form comparison (float64-singular,
    Lam_q = 0 on the wire) and counted; informative cells in the same component are still compared"""
    Fl, lam, z, _ = rand_case(6, (4, 4, 2), 3, RNG, p_zero=0.0, w=1000.0)
    lam[:, :2] = 1e-40
    c = L.gbp_checks(z, lam, Fl, 1000.0)
    assert c['cf_max'] <= L.CF_TOL and c['violations'] == 0 and c['cf_excluded_tiny'] == 2 * 4 * 2 and c['cf_cells'] == 32 - 16
    assert 0 <= c['cf_rel'] <= 1e-9 and abs(c['cf_rel'] * c['cf_scale'] - c['cf_max']) <= 1e-12 * max(1.0, c['cf_scale'])   # review S6
    hb, lb, _ = L.gabp(lam.unsqueeze(1) * z, lam, Fl, 1000.0, None)
    x = L.belief_mean(hb, lb)
    assert torch.isfinite(x).all()
    hq, lq = L.quantize(lam.unsqueeze(1) * z, lam, 'round')
    assert not lq[:, :2].any() and not hq[:, :, :2].any()


def test_w_infinity_information_weighted_mean():
    for m in (2, 3, 5, 8):
        Fl, lam, z, _ = rand_case(m, (4, 4, 2), 4, RNG)
        hb, lb, _ = L.gabp(lam.unsqueeze(1) * z, lam, Fl, 1e12, None)
        x = L.belief_mean(hb, lb)
        tot = lam.sum(0); avg = (lam.unsqueeze(1) * z).sum(0) / torch.where(tot > 0, tot, torch.ones_like(tot)).unsqueeze(0)
        for a in range(m):
            assert torch.allclose(x[a], torch.where(tot > 0, avg, torch.zeros_like(avg)), rtol=1e-6, atol=1e-6)
            assert torch.allclose(lb[a], tot, rtol=1e-9)
    # finite w: agents disagree; the disagreement shrinks as w grows
    Fl, lam, z, _ = rand_case(5, (4, 4, 2), 2, RNG, p_zero=0.0)
    dis = []
    for w in (0.1, 1.0, 10.0, 100.0):
        hb, lb, _ = L.gabp(lam.unsqueeze(1) * z, lam, Fl, w, None); x = L.belief_mean(hb, lb)
        dis.append(float(((x - x.mean(0)) ** 2).mean()))
    assert all(a > b for a, b in zip(dis[:-1], dis[1:]))


def test_e3r_is_special_case():
    """Enc = identity (z = calibrated evidence e, voxel grid, D = 1), Lam = count (1[touched]), w -> inf: every belief = (S / k, k),
    i.e. E3-R's evidence sum; with the integer wire (w huge) the quantised relay gives S and k EXACTLY, and E3-R's network input"""
    n = 5; shape = (8, 8, 4)
    Fl = G.forest([(0, 1, 1.), (1, 2, 1.), (1, 3, 1.), (3, 4, 1.)], n)
    cq = RNG.integers(-100, 150, R.NL)
    maps = [rand_map(shape, 0.5) for _ in range(n)]
    st = [R.stalk(Lm, cq) for Lm in maps]
    e = torch.from_numpy(np.stack([s for s, _ in st]).astype(np.float64)).unsqueeze(1); k = torch.from_numpy(np.stack([kk for _, kk in st]).astype(np.float64))
    S, kc = R.direct_sums(maps, list(range(n)), cq)
    hb, lb, _ = L.gabp(k.unsqueeze(1) * e, k, Fl, 1e12, None)
    x = L.belief_mean(hb, lb)
    for a in range(n):
        assert np.allclose(lb[a].numpy(), kc) and np.allclose((x[a, 0] * lb[a]).numpy(), S, atol=1e-6)
        ok = kc > 0
        assert np.allclose(x[a, 0].numpy()[ok], (S / np.maximum(kc, 1))[ok])
    hq, lq, msgs = L.gabp(k.unsqueeze(1) * e, k, Fl, 1e15, 'round')
    for a in range(n):
        assert np.array_equal(lq[a].numpy().astype(np.int64), kc) and np.array_equal(hq[a, 0].numpy().astype(np.int64), S)
        fx = R.features_np(np.rint(hq[a, 0].numpy()).astype(np.int32), lq[a].numpy().astype(np.int32))
        assert np.array_equal(fx, R.features_np(S, kc))                       # = the input of E3-R's NET on the component sum
    for (s, r), (hm, lm) in msgs.items():                                     # = E1's relayed sum / count messages
        side = Fl['sides'][(s, r)]
        assert np.array_equal(lm.numpy().astype(np.int64), sum(st[b][1].astype(np.int64) for b in side))
        assert np.array_equal(hm[0].numpy().astype(np.int64), sum(st[b][0].astype(np.int64) for b in side))


def test_energy_monotone_descent():
    for t in range(30):
        m = 2 + t % 7
        Fl, lam, z, w = rand_case(m, (4, 3, 2), 3, RNG)
        hb, lb, _ = L.gabp(lam.unsqueeze(1) * z, lam, Fl, w, None)
        zf = z.reshape(m, 3, -1).numpy(); lf = lam.reshape(m, -1).numpy()
        en, xfin = L.descent_energies(zf, lf, Fl, w, L.belief_mean(hb, lb).reshape(m, 3, -1).numpy(), lb.reshape(m, -1).numpy())
        tol = 1e-9 * max(1, en[0])
        assert all(b <= a + tol for a, b in zip(en[:-1], en[1:])) and len(en) == 1 + (m - 1) + 1 + m
        xc, inf = L.closed_form(zf, lf, Fl, w)
        assert np.abs(xfin[:, :, inf] - xc[:, :, inf]).max() < 1e-8
        assert abs(en[-1] - L.energy(np.where(inf[None, None], xc, xfin), zf, lf, Fl, w)) <= tol
        # the final energy is the minimum: any perturbation increases it
        xp = xfin + RNG.normal(0, 1e-3, xfin.shape)
        assert L.energy(xp, zf, lf, Fl, w) >= en[-1] - tol


def test_naive_belief_sequence_not_monotone():
    """why O13 uses the block-coordinate reading: 'current belief means' can raise the energy (chain 1-2-3, z = -10, 0, 10)"""
    Fl = G.forest([(0, 1, 1.), (1, 2, 1.)], 3)
    z = np.array([-10., 0., 10.]).reshape(3, 1, 1); lam = np.ones((3, 1)); w = 1.0
    x0 = z.copy(); e0 = L.energy(x0, z, lam, Fl, w)
    x1 = z.copy(); x1[1] = (0 + 0.5 * -10) / 1.5                              # node 1's belief after only the message 0 -> 1
    assert L.energy(x1, z, lam, Fl, w) > e0
    zt = torch.from_numpy(z.reshape(3, 1, 1, 1, 1)); lt = torch.from_numpy(lam.reshape(3, 1, 1, 1))
    assert L.gbp_checks(zt, lt, Fl, w)['violations'] == 0


# ---------------------------------------------------------------- encoder support, quantisation, rate
def test_support_constraint():
    torch.manual_seed(0)
    model = L.E3L(width=8).eval()
    maps = [rand_map((16, 16, 8), 0.3) for _ in range(3)]
    maps[0][:, :, 4:] = G.UNK; maps[1][:8] = G.UNK; maps[2][:] = G.UNK
    z, lam = L.encode_maps(model, maps, torch.as_tensor(np.arange(R.LMIN, R.LMAX + 1), dtype=torch.int32), 'cpu')
    assert z.shape == (3, L.D_LAT, 4, 4, 4) and lam.shape == (3, 4, 4, 4)
    for a, Lm in enumerate(maps):
        blk = (Lm != G.UNK).reshape(4, 4, 4, 4, 4, 2).any((1, 3, 5))
        assert (lam[a].numpy()[~blk] == 0).all() and (lam[a].numpy()[blk] > 0).all()
    assert not lam[2].any()
    hq, lq, sym, msk = model.source(z, lam, 'round')
    assert not hq[2].any() and not lq[2].any() and sym.shape == (3, L.D_LAT + 1, 64) and not msk[2].any()
    assert abs(float(lam[0][lam[0] > 0].mean()) - 2.0) < 1.0                  # Lam starts near softplus(log(e^2 - 1)) = 2
    with torch.no_grad():                                                      # the straight-through floor: never a dead softplus
        model.enc.head.bias[L.D_LAT] = -200.0
    model.train()
    kn = torch.from_numpy(np.stack([m != G.UNK for m in maps[:2]]))
    f = R.features_t(torch.zeros(kn.shape, dtype=torch.int32), kn.to(torch.int32)).float()
    _, lam2 = model.encode(f, kn)
    sup = lam2 > 0
    assert sup.any() and torch.allclose(lam2[sup], torch.nn.functional.softplus(torch.tensor(L.A_FLOOR)))
    lam2.sum().backward()                                                      # a gradient that would push a further down: blocked
    assert float(model.enc.head.bias.grad[L.D_LAT]) == 0.0
    model.zero_grad(); _, lam2 = model.encode(f, kn)
    (-lam2.sum()).backward()                                                   # a gradient that pulls a back up: passes (review S1)
    assert float(model.enc.head.bias.grad[L.D_LAT]) < 0


def test_floor_one_sided_straight_through():
    """review S1: forward max(a, -8); backward unchanged where a >= -8, only the non-positive part where a < -8"""
    a = torch.tensor([-20.0, -9.0, -8.0, -3.0, 4.0], requires_grad=True)
    y = L.floor_one_sided(a)
    assert torch.equal(y.detach(), torch.tensor([-8.0, -8.0, -8.0, -3.0, 4.0]))
    g = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0])
    y.backward(g)
    assert torch.equal(a.grad, torch.tensor([0.0, -2.0, 3.0, -4.0, 5.0]))
    # under sustained downward pressure a stops at the floor's pre-activation; one step of upward pressure moves it back immediately
    p = torch.tensor([-8.5], requires_grad=True); opt = torch.optim.SGD([p], lr=1.0)
    for _ in range(5):
        opt.zero_grad(); torch.nn.functional.softplus(L.floor_one_sided(p)).sum().backward(); opt.step()
    assert float(p.detach()) == -8.5                                           # no further drift below the floor
    opt.zero_grad(); (-1e4 * torch.nn.functional.softplus(L.floor_one_sided(p))).sum().backward(); opt.step()
    assert float(p.detach()) > -8.5


def test_quantization_roundtrip_and_ste():
    h = torch.from_numpy(RNG.normal(0, 4, (8, 5, 4, 3))); lam = torch.from_numpy(RNG.gamma(1., 1.5, (5, 4, 3)))
    hq, lq = L.quantize(h, lam, 'round')
    hi, li = L.msg_ints(hq, lq)
    assert np.array_equal(li * L.DELTA_LAM, lq.numpy()) and np.array_equal(hi * L.DELTA_H, hq.numpy())
    assert (np.abs(lq.numpy() - lam.numpy()) <= L.DELTA_LAM / 2 + 1e-12).all()
    assert not hi[:, li == 0].any() and (li == 0).any()
    on = li > 0
    assert (np.abs(hq.numpy() - h.numpy())[:, on] <= L.DELTA_H / 2 + 1e-12).all()
    hq2, lq2 = L.quantize(hq, lq, 'round')                                    # idempotent: what is sent is what is received
    assert torch.equal(hq2, hq) and torch.equal(lq2, lq)
    hs = h.clone().requires_grad_(True); ls = lam.clone().requires_grad_(True)
    a, b = L.quantize(hs, ls, 'ste')
    assert torch.equal(a.detach(), hq) and torch.equal(b.detach(), lq)
    (a.sum() * 2 + b.sum() * 3).backward()
    assert torch.equal(hs.grad, 2 * torch.from_numpy(on.astype(np.float64)).expand_as(h)) and torch.equal(ls.grad, torch.full_like(lam, 3.))
    by = L.int_bytes(hi, li)
    assert by > 4 * 9 and by == L.int_bytes(hi.copy(), li.copy())
    wb = L.wire_bytes(hi, li)
    assert 2 + 4 * 9 < wb == L.wire_bytes(hi.copy(), li.copy())
    # the quantised GaBP wire: every message is an integer map, and the receiver's belief from the dequantised integers = the pipeline's
    Fl, lam3, z3, w = rand_case(4, (4, 4, 2), 3, RNG, w=5.0)
    h3q, l3q = L.quantize(lam3.unsqueeze(1) * z3, lam3, 'round')
    hb, lb, msgs = L.gabp(h3q, l3q, Fl, w, 'round')
    for a in range(4):
        hh = h3q[a].clone(); ll = l3q[a].clone()
        for k in Fl['nb'][a]:
            mh, ml = L.msg_ints(*msgs[(k, a)]); hh += torch.from_numpy(mh * L.DELTA_H); ll += torch.from_numpy(ml * L.DELTA_LAM)
        assert torch.equal(hh, hb[a]) and torch.equal(ll, lb[a])


def test_entropy_model_proxy_vs_wire():
    """D8-54 supplement 2: the proxy counts the support ONCE (Bernoulli + in-support values); fitted, it is of the same order as the
    measured wire code length (mask once + in-support values given the mask); the old 9x-mask upper bound is larger than the wire"""
    torch.manual_seed(1)
    g = (32, 32, 16)
    on = np.zeros(g, bool); on[4:20, 6:22, 2:10] = True; on &= RNG.random(g) < 0.9            # a spatially coherent support
    li = on * (1 + RNG.poisson(1.0, g))
    hi = np.where(on[None], np.rint(RNG.laplace(0, 1.2, (8,) + g)), 0).astype(np.int64)
    sym = torch.from_numpy(np.concatenate([li[None], hi]).reshape(1, 9, -1).astype(np.float32))
    m = torch.from_numpy(on.reshape(1, -1).astype(np.float32))
    ent = L.EntropyModel()
    opt = torch.optim.Adam(ent.parameters(), lr=0.05)
    for _ in range(300):
        loss = ent.bits(sym, m).mean() / sym.shape[-1]; opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        sb, vb = ent.bits_parts(sym, m)
        proxy = float(sb[0] + vb[0].sum())
        assert abs(float(torch.sigmoid(ent.sup_logit)) - on.mean()) < 0.01             # q = the support fraction
        # values outside the support cost nothing: changing them does not change the proxy
        sym2 = sym.clone(); sym2[0, 1:, ~torch.from_numpy(on.reshape(-1))] = 5.0
        assert float(ent.bits(sym2, m)) == float(ent.bits(sym, m))
    wire = 8 * L.wire_bytes(hi, li.astype(np.int64)); upper = 8 * L.int_bytes(hi, li.astype(np.int64))
    emp_vals = sum(L._ent(np.bincount((sym[0, c].numpy()[on.reshape(-1)] + 64).astype(np.int64))) * on.sum() for c in range(9))
    n = on.size; q = on.mean(); emp_sup = -n * (q * np.log2(q) + (1 - q) * np.log2(1 - q))
    assert proxy >= 0.99 * (emp_vals + emp_sup), (proxy, emp_vals, emp_sup)            # cross-entropy >= empirical entropy
    assert 1 / 3 < wire / proxy < 3, (wire, proxy)
    assert wire < upper and 8 * L.mask_bytes(on) < 0.5 * emp_sup                       # the ctx coder exploits the coherent mask
    with torch.no_grad():
        assert 0.5 < float(ent.bits(sym, m, noisy=True)) / proxy < 2                    # noise relaxation ~ the discrete length


def test_wire_coder_sign_stream_brute_force():
    """the in-support sign stream with the mask given = a sequential adaptive binary KT coder (raster order) with the ternary
    contexts of G.ctx_tern_bits; and value_bytes_given_mask = that + the int_ctx magnitude streams + 4 B"""
    g = (6, 5, 4)
    kn = RNG.random(g) < 0.6; v = np.where(kn, RNG.integers(-3, 4, g), 0)
    t = np.where(kn, (v > 0).astype(np.int8), np.int8(-1))
    X, Y, Z = g; cnt = {}; bits = 0.0
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                if not kn[x, y, z]:
                    continue
                ctx = tuple(int(t[x - dx, y - dy, z - dz]) if (x - dx >= 0 and y - dy >= 0 and z - dz >= 0) else -1 for dx, dy, dz in G.NB)
                n0, n1 = cnt.get(ctx, (0, 0)); bit = int(v[x, y, z] > 0)
                p = ((n1 if bit else n0) + 0.5) / (n0 + n1 + 1.0); bits -= np.log2(p)
                cnt[ctx] = (n0 + (1 - bit), n1 + bit)
    assert abs(L.sign_bits_given_mask(v, kn) - bits) < 1e-6
    e0, e1, p0, p1, s0, s1 = G.int_magnitude_events(v, kn)
    assert abs(L.value_bytes_given_mask(v, kn) - ((bits + G.kt_bin_bits(e0, e1) + G.kt_bin_bits(p0, p1) + G.kt_bin_bits(s0, s1)) / 8 + 4)) < 1e-6
    full = G.int_ctx_bytes(v, kn)                                                      # = mask + signs + magnitudes: paying for the mask
    assert L.value_bytes_given_mask(v, kn) < full
    li = np.zeros(g, np.int64); hi = np.zeros((8,) + g, np.int64)
    assert L.wire_bytes(hi, li) <= L.mask_bytes(li > 0) + 9 * 4 + 1e-9                 # empty support: headers + an all-zero mask


# ---------------------------------------------------------------- sampling
def test_subtree_sampling_connected():
    H = fake_house(n=7, edges=((0, 1), (1, 2), (1, 3), (3, 4), (5, 6)))
    rng = np.random.default_rng(3)
    seen = set(); sizes = {}
    for _ in range(3000):
        h, d, sub, aug, dec = L.sample_spec(rng, [H])
        Fl = H.local(sub)                                                      # asserts connectivity
        assert len(Fl['comps']) == 1 and dec in sub and (set(sub) <= {0, 1, 2, 3, 4} or set(sub) <= {5, 6})
        seen.add(sub); cm = 5 if 0 in sub or 1 in sub or 3 in sub or 2 in sub or 4 in sub else 2
        sizes.setdefault(cm, []).append(len(sub))
    assert {(0, 1), (1, 2), (1, 3), (3, 4), (5, 6), (0, 1, 2, 3, 4), (5,), (2,)} <= seen
    assert (0, 2) not in seen and (2, 3) not in seen and (0, 4) not in seen   # never a disconnected subset
    for cm, s in sizes.items():
        c = np.bincount(s, minlength=cm + 1)[1:]
        assert (np.abs(c / c.sum() - 1 / cm) < 0.05).all()                    # |s| uniform in 1..n


def test_forward_batch_shapes_and_labels():
    torch.manual_seed(0)
    H = fake_house(); model = L.E3L(width=8)
    tl = torch.as_tensor(np.arange(R.LMIN, R.LMAX + 1), dtype=torch.int32)
    specs = [(0, 1, (0, 1, 2), 0, 1), (0, 0, (3,), 5, 3), (0, 1, (1, 2), 3, 2)]
    lg, yk, yo, src = L.forward_batch(model, specs, [H], tl, 'ste')
    sym, msk = src['sym'], src['msk']
    assert lg.shape == (3, 2, 16, 16, 8) and sym.shape == (6, 9, 64) and msk.shape == (6, 64) and src['dec'] == [1, 3, 5]
    assert torch.equal(msk.detach(), (src['lq'].detach() > 0).flatten(1).to(msk.dtype))
    sc = (H.private[[0, 1, 2]] >= 0).any(0)
    assert np.array_equal(yk[0].numpy(), sc) and np.array_equal(yo[0].numpy(), H.occ)
    assert np.array_equal(yk[1].numpy(), L.aug_t(torch.from_numpy(H.private[3] >= 0), 5).numpy())
    lk, lo = L.bce_terms(lg, yk, yo)
    (lk + lo + model.ent.bits(sym, msk, noisy=True, detach=True).mean() / L.R_STAR).backward()
    assert model.log_w.grad is not None and model.enc.head.weight.grad is not None and model.ent.mu.grad is None and model.ent.sup_logit.grad is None
    rc = L.measure_rate(model, src)
    assert rc['agents'] == 3 and rc['measured_bits'] > 0 and rc['proxy_bits'] > 0


# ---------------------------------------------------------------- read-out and threshold selection
def test_readout_and_selection_logic():
    """three-state read-out (known head first, then occupancy head) and the sequential selection reproduce G.score pooled numbers;
    the known threshold is the smallest with FPR <= control + 0.3 pp; the occupancy threshold maximises F1 s.t. free recall"""
    Hs = [fake_house(rng=np.random.default_rng(s)) for s in (1, 2)]
    items = []
    for Hd in Hs:
        Hh = G.House(Hd.occ.astype(np.int8), Hd.slab, Hd.obj, Hd.cell)
        for cm in Hd.comps:
            T = G.Target(Hd.private[cm].max(0), Hh)
            for _ in cm:
                kl = np.where(T.known, RNG.choice([1.0, 3.0], T.t.shape), RNG.choice([-2.0, 0.5, 2.0], T.t.shape)).astype(np.float32)
                ol = np.where(T.occ, RNG.choice([-0.5, 1.0, 2.5], T.t.shape), RNG.choice([-1.5, 0.7, 1.5], T.t.shape)).astype(np.float32)
                items.append((np.stack([kl, ol]), T, Hh, (Hd.maps[1][cm] != G.UNK).any(0)))
    orig = L.agent_logits
    L.agent_logits = lambda model, houses, device, table_t, which, chunk=2: iter(items)
    try:
        ctrl = dict(R8=dict(fpr=0.05, free=0.3, furn_f1=0.5), K0=dict(fpr=0.05, free=0.3, furn_f1=0.5))
        base = sum(int((kc & ~T.known).sum()) for _, T, _, kc in items) / sum(int((~T.known).sum()) for _, T, _, _ in items)
        ctrl['R8']['fpr'] = ctrl['K0']['fpr'] = base
        sel = L.select_thresholds(torch.nn.Identity(), Hs, 'cpu', None, ctrl)['R8']
    finally:
        L.agent_logits = orig

    def pooled(jk, jo):
        tp = fp = fn = fh = ft = fu = ut = 0
        for lg, T, Hh, _ in items:
            p = L.readout(lg[0], lg[1], jk, jo)
            assert ((p == -1) == (R.logit_bin(lg[0]) < jk)).all()
            r = G.score(p, T); nf = int(T.free.sum()); nu = int((~T.known).sum())
            tp += r['furn_tp']; fp += r['furn_fp']; fn += r['furn_fn']; fh += (r['free_recall'] or 0) * nf; ft += nf
            fu += (r['unknown_known_fpr'] or 0) * nu; ut += nu
        return 2 * tp / max(2 * tp + fp + fn, 1), fh / ft, fu / ut
    f1, fr, fpr = pooled(sel['j_known'], sel['j_occ'])
    assert abs(f1 - sel['f1']) < 1e-12 and abs(fr - sel['free']) < 1e-9 and abs(fpr - sel['fpr']) < 1e-9 and fpr <= base + L.GUARD_FPR_SELECT + 1e-12
    cand = sorted({int(R.logit_bin(v)) for v in (-2.0, 0.5, 1.0, 2.0, 3.0)} | {int(R.logit_bin(v)) + 1 for v in (-2.0, 0.5, 2.0)})
    feas_k = [j for j in cand if pooled(j, 0)[2] <= base + L.GUARD_FPR_SELECT + 1e-12]
    assert sel['j_known'] <= min(feas_k) and sel['feasible_known']
    oc = sorted({int(R.logit_bin(v)) for v in (-1.5, -0.5, 0.7, 1.0, 1.5, 2.5)} | {int(R.logit_bin(v)) + 1 for v in (-1.5, -0.5, 0.7, 1.0, 1.5, 2.5)} | {0})
    sc = {j: pooled(sel['j_known'], j) for j in oc}
    best = max(v[0] for v in sc.values() if v[1] + 1e-12 >= 0.3)
    assert abs(sel['f1'] - best) < 1e-12 and sel['free'] + 1e-12 >= 0.3 and sel['feasible_occ']
    # infeasible free constraint -> flagged, smallest bin of the maximal free recall
    jo, feas, f1a, fra = L.pick_occ(np.ones(L.NBIN), np.ones(L.NBIN), np.ones(L.NBIN), 10, 10 ** 6, 0.5)
    assert not feas and fra[jo] == fra.max()


# ---------------------------------------------------------------- verdict
def _row(furn, free=.997, surf=.99, fpr=.018, tol1=.99, k0=.94):
    return dict(R8=dict(furn_f1=furn, furn_f1_tol1=tol1, free_recall=free, surface_f1=surf, unknown_known_fpr=fpr), K0=dict(furn_f1=k0), K0_sep=dict(furn_f1=k0))


def _summaries(d=-.005, ratio=.1, d_fresh=-.004, d_cam=-.01, clean=.995, k0=-.01, missing=(), gbp_ok=True):
    Sl = dict(panels={}, failed=[], skipped=[], e1_mismatch=[], agg_reasons=[], gbp={f'E3L_{t}': dict(cf_max=1e-12 if gbp_ok else 1e-3, violations=0, components=9, pass_=True)
                                                                                   for t in ('342101', '342202')})
    for g in Sl['gbp'].values():
        g['pass'] = g.pop('pass_') and gbp_ok
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
                    a = _row(.97 + dv, k0=.94 + k0); a['bytes_median'] = ratio * 753000.
                    tl['cam_nf'][f'E3L_{ts}'] = a
                tl['cam'][f'E3L_{ts}'] = _row(.80 + d_cam); tl['clean'][f'E3L_{ts}'] = _row(clean)
            Pl['seeds'][ns] = dict(houses=houses, tiers=tl); Pc['seeds'][ns] = dict(houses=houses, tiers=tc)
        Sl['panels'][panel] = Pl; Sc['panels'][panel] = Pc
    return Sl, Sc


def _runs(d, done=(6000, 6000), planned=(6000, 6000), truncated=(False, False)):
    out = []
    for i, sd in enumerate(('342101', '342202')):
        r = os.path.join(d, f'run{sd}'); os.makedirs(r, exist_ok=True)
        with open(os.path.join(r, 'run.json'), 'w') as fh:
            json.dump(dict(seed=int(sd), train_houses=186, recipe=dict(evidence=dict(desc=dict(spec='c:x:A:cam_nf')))), fh)
        with open(os.path.join(r, 'final.json'), 'w') as fh:
            json.dump(dict(complete=(done[i] == planned[i] and not truncated[i]), truncated=truncated[i], updates_done=done[i], updates_planned=planned[i]), fh)
        with open(os.path.join(r, 'threshold.json'), 'w') as fh:
            json.dump(dict(n_houses=19), fh)
        out.append(r)
    return out


def test_verdict_d8_54():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        Rn = _runs(d)

        def V(Sl, Sc, runs=Rn):
            p = os.path.join(d, 'ctrl.json')
            with open(p, 'w') as fh:
                json.dump(Sc, fh)
            return L.verdict(Sl, runs, p)['verdict']
        assert V(*_summaries()) == 'LEVER'
        assert V(*_summaries(d=-.0105)) == 'AMBIGUOUS'                       # just below the -1 pp lever margin
        assert V(*_summaries(d=-.031)) == 'NOT_ENOUGH'
        low = {(t, n): -.005 for t in ('342101', '342202') for n in ('1', '2')}; low[('342202', '2')] = -.031
        assert V(*_summaries(d=low)) == 'NOT_ENOUGH'                          # any cell < -3 pp
        assert V(*_summaries(ratio=.16)) == 'AMBIGUOUS' and V(*_summaries(ratio=.51)) == 'NOT_ENOUGH'
        assert V(*_summaries(d_cam=-.031)).startswith('AMBIGUOUS (lever demoted: cam')
        assert V(*_summaries(d_cam=-.029)) == 'LEVER'
        assert V(*_summaries(d_fresh=+.002)) == 'LEVER'                         # literal sign opposite: reported only (supplement 4)
        p = os.path.join(d, 'ctrl2.json'); Sl, Sc = _summaries(d_fresh=+.002)
        with open(p, 'w') as fh:
            json.dump(Sc, fh)
        fi = L.verdict(Sl, Rn, p)['fresh']
        assert fi['report_literal_sign_opposite'] and fi['fresh_ok']
        assert V(*_summaries(d_fresh=-.0095)) == 'LEVER'
        assert V(*_summaries(d_fresh=-.0105)).startswith('AMBIGUOUS (lever demoted: fresh')
        assert V(*_summaries(d=-.031, d_fresh=-.05)) == 'NOT_ENOUGH'                  # never changes NOT_ENOUGH
        assert V(*_summaries(d=-.0105, d_fresh=+.05)) == 'AMBIGUOUS'                  # never upgrades
        assert V(*_summaries(clean=.985)) == 'AMBIGUOUS' and V(*_summaries(k0=-.031)) == 'AMBIGUOUS' and V(*_summaries(k0=-.029)) == 'LEVER'
        assert V(*_summaries(gbp_ok=False)) == 'AMBIGUOUS'
        Sl, Sc = _summaries(); Sl['panels']['val']['seeds']['1']['tiers']['cam_nf']['E3L_342101']['R8']['free_recall'] = .997 - .006
        assert V(Sl, Sc) == 'AMBIGUOUS'
        Sl, Sc = _summaries(); Sl['panels']['val']['seeds']['1']['tiers']['cam_nf']['E3L_342101']['R8']['unknown_known_fpr'] = .018 + .0031
        assert V(Sl, Sc) == 'AMBIGUOUS'
        Sl, Sc = _summaries(); Sl['panels']['val']['seeds']['1']['tiers']['cam_nf']['E3L_342101']['R8']['unknown_known_fpr'] = .018 + .0029
        assert V(Sl, Sc) == 'LEVER'
        Sl, Sc = _summaries(); Sl['panels']['val']['seeds']['2']['tiers']['cam_nf']['E3L_342202']['R8']['surface_f1'] = .99 - .006
        assert V(Sl, Sc) == 'AMBIGUOUS'
        # INCOMPLETE: missing cell / fresh / control / clean / house set / runs / update counts; truncation only blocks the lever
        assert V(*_summaries(missing={('val', '342101', '1')})).startswith('INCOMPLETE')
        assert V(*_summaries(missing={('fresh', '342202', '2')})).startswith('INCOMPLETE')
        Sl, Sc = _summaries(); del Sc['panels']['val']['seeds']['1']['tiers']['cam_nf']['NET_342202']
        assert V(Sl, Sc).startswith('INCOMPLETE')
        Sl, Sc = _summaries(); del Sl['panels']['val']['seeds']['2']['tiers']['clean']['E3L_342101']
        assert V(Sl, Sc).startswith('INCOMPLETE')
        Sl, Sc = _summaries(); Sc['panels']['val']['seeds']['1']['houses'] = Sc['panels']['val']['seeds']['1']['houses'][1:]
        assert V(Sl, Sc).startswith('INCOMPLETE')
        Sl, Sc = _summaries(); del Sl['gbp']['E3L_342202']
        assert V(Sl, Sc).startswith('INCOMPLETE')
        Sl, Sc = _summaries(); Sl['failed'] = ['val_r1_i0_s1']
        assert V(Sl, Sc).startswith('INCOMPLETE')
        assert V(*_summaries(), runs=_runs(os.path.join(d, 'u'), done=(6000, 1999), truncated=(False, True))).startswith('INCOMPLETE')
        assert V(*_summaries(), runs=_runs(os.path.join(d, 'p'), done=(6000, 5000), planned=(6000, 5000))).startswith('INCOMPLETE')
        assert V(*_summaries(), runs=_runs(os.path.join(d, 't'), done=(6000, 5000), truncated=(False, True))) == 'AMBIGUOUS'
        assert V(*_summaries(d=-.031), runs=_runs(os.path.join(d, 't2'), done=(6000, 5000), truncated=(False, True))) == 'NOT_ENOUGH'
        assert L.verdict(_summaries()[0], Rn, None)['verdict'].startswith('INCOMPLETE')


# ---------------------------------------------------------------- end to end (gen-format cache -> E3-R control -> E3-L train / select / eval / aggregate)
def _write_cache(d):
    cache = os.path.join(d, 'cache'); os.makedirs(cache)
    houses = []
    for i, (idx, ho) in enumerate(((0, False), (9, True))):
        H = fake_house(noisy_draws=1, rng=np.random.default_rng(10 + i))
        comp_of = np.zeros(H.n, np.int64)
        for ci, cm in enumerate(H.comps):
            comp_of[cm] = ci
        name = f'TRAIN_r1_i{idx}'; p = os.path.join(cache, name + '.npz')
        np.savez_compressed(p, occ=H.occ.astype(np.int8), slab=H.slab, obj=H.obj, cell=H.cell, private=H.private, comp_of=comp_of,
                            edges=R.edges_array(H.edges), L_clean=H.maps[0], L_cam_nf_s200=H.maps[1])
        houses.append(dict(house=name, rooms=1, index=idx, holdout=ho, agents=H.n, components=H.comps, file=name + '.npz', file_sha256=R.file_sha256(p)))
    with open(os.path.join(cache, 'manifest.json'), 'w') as fh:
        json.dump(dict(format=R.CACHE_FORMAT, seeds=[200], houses=houses), fh)
    return cache, houses


def test_end_to_end_tiny():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        cache, houses = _write_cache(d)
        fit = os.path.join(d, 'fit.json'); cq = RNG.integers(-60, 90, R.NL)
        tb = dict(cq=list(map(int, cq)), theta=5, l0q=-3, tq=2, l0=-0.15)
        with open(fit, 'w') as fh:
            json.dump(dict(complete=True, halves=dict(A=dict(tiers=dict(cam_nf=tb)), B=dict(tiers=dict(cam_nf=dict(tb, cq=[0] * R.NL))))), fh)
        ev = f'c:{fit}:A:cam_nf'; man_sha = R.file_sha256(os.path.join(cache, 'manifest.json'))
        rr = os.path.join(d, 'e3r')                                            # the paired control (E3-R trainer, 2 updates)
        R.main(['train', '--cache', cache, '--evidence', ev, '--seed', '342101', '--out', rr, '--updates', '2', '--device', 'cpu', '--batch', '1',
                '--log-every', '1', '--val-every', '2', '--holdout-samples', '1', '--warmup', '1'])
        out = os.path.join(d, 'e3l')
        common = ['train', '--cache', cache, '--seed', '342101', '--updates', '3', '--device', 'cpu', '--batch', '2', '--width', '8', '--log-every', '1',
                  '--val-every', '3', '--holdout-samples', '2', '--warmup', '1', '--manifest-sha', man_sha, '--rate-every', '2',
                  '--ctrl-ckpt', os.path.join(rr, 'checkpoints', 'final.pt'), '--ctrl-threshold', os.path.join(rr, 'threshold.json')]
        try:
            L.main(common + ['--evidence', f'c:{fit}:B:cam_nf', '--out', os.path.join(d, 'bad')])
        except AssertionError as e:
            assert 'paired' in str(e)
        else:
            raise AssertionError('an evidence table different from the paired E3-R checkpoint must be refused')
        fin = L.main(common + ['--evidence', ev, '--out', out])
        assert fin['complete'] and fin['updates_done'] == 3 and fin['threshold']['R8']['j_known'] >= 0
        rcv = json.load(open(os.path.join(out, 'rate_curve.json')))
        assert [r['updates'] for r in rcv] == [1, 2] and all(r['measured_bits'] > 0 and r['proxy_bits'] > 0 for r in rcv)
        assert all(0 <= r['support_cells'] <= r['touched_cells'] and 'lam_dual' in r and 'r_meas' in r for r in rcv)   # review S1 trajectories
        hist = json.load(open(os.path.join(out, 'history.json')))
        assert hist and all('support_cells' in h and 'touched_cells' in h and 'r_meas_bits' in h for h in hist)
        assert fin['threshold_status'] == 'ok'
        # review S5: a failing threshold selection keeps the checkpoint and is flagged (no threshold.json -> the verdict's INCOMPLETE)
        badthr = os.path.join(d, 'bad_threshold.json')
        with open(badthr, 'w') as fh:
            json.dump({k: v for k, v in json.load(open(os.path.join(rr, 'threshold.json'))).items() if k != 'k0'}, fh)
        bad = os.path.join(d, 'e3l_badsel')
        fb = L.main([x if x != os.path.join(rr, 'threshold.json') else badthr for x in common] + ['--evidence', ev, '--out', bad])
        assert fb['threshold_status'] == 'FAILED' and 'k0' in fb['threshold_error']
        assert os.path.exists(os.path.join(bad, 'checkpoints', 'final.pt')) and not os.path.exists(os.path.join(bad, 'threshold.json'))
        fj = json.load(open(os.path.join(bad, 'final.json')))
        assert fj['threshold_status'] == 'FAILED' and fj['checkpoint_sha256'] == R.file_sha256(os.path.join(bad, 'checkpoints', 'final.pt'))
        assert abs(rcv[-1]['r_meas'] - np.mean([r['measured_bits'] for r in rcv])) < 1e-9 and fin['r_meas_bits'] == rcv[-1]['r_meas']
        th = json.load(open(os.path.join(out, 'threshold.json')))
        assert th['tf32'] is False and th['n_houses'] == 1 and abs(th['R8']['control']['fpr'] - th['R8']['control_fpr_check']) < 1e-12
        run = json.load(open(os.path.join(out, 'run.json')))
        assert all(c['cf_max'] <= L.CF_TOL and c['violations'] == 0 for c in run['probe_checks'])
        try:
            L.main(['select', '--run', out])
        except SystemExit:
            pass
        else:
            raise AssertionError('select must not overwrite threshold.json silently')
        sel = L.main(['select', '--run', out, '--overwrite'])
        assert sel['R8']['j_known'] == th['R8']['j_known'] and sel['R8']['j_occ'] == th['R8']['j_occ'] and sel['K0']['j_occ'] == th['K0']['j_occ']
        # a VAL-format export of one fake house, eval + aggregate (smoke)
        ex = os.path.join(d, 'export'); os.makedirs(os.path.join(ex, 'records'))
        H = fake_house(noisy_draws=1, rng=np.random.default_rng(42))
        arr = dict(occ=H.occ.astype(np.int8), slab=H.slab, obj=H.obj, cell=H.cell, private=H.private, edges=R.edges_array(H.edges))
        for tn in R.EVAL_TIERS:
            arr[f'L_k0_{tn}'] = H.maps[0] if tn == 'clean' else H.maps[1]
        arr['L_pub_cam'] = H.maps[1][::-1].copy()
        fp = os.path.join(ex, 'val_r1_i0_s1.npz'); np.savez_compressed(fp, **arr)
        with open(os.path.join(ex, 'records', 'val_r1_i0_s1.json'), 'w') as fh:
            json.dump(dict(panel='val', rooms=1, index=0, seed=1, split='VALIDATION', file='val_r1_i0_s1.npz', file_sha256=R.file_sha256(fp),
                           components=H.comps, e1_match=dict(cam_nf=True, cam=True, clean=True)), fh)
        rec = L.eval_job((json.load(open(os.path.join(ex, 'records', 'val_r1_i0_s1.json'))),
                          dict(export=ex, ckpts=[os.path.join(out, 'checkpoints', 'final.pt')], ckpt_shas={}, tiers=list(R.EVAL_TIERS), threads=1, chunk=2,
                               script_sha256='x')))
        assert not rec.get('failed'), rec.get('traceback')
        a = rec['tiers']['cam_nf']['E3L_342101']
        assert len(a['R8']) == H.n and len(a['K0_sep']) == H.n and a['bytes'] > 0 and a['gbp']['violations'] == 0 and a['gbp']['cf_max'] <= L.CF_TOL
        assert 0 <= a['gbp']['cf_rel_max'] <= 1e-6 and a['gbp']['cf_scale_max'] > 0
        assert len(a['rate']) == H.n and all(0 < r['measured_bits'] <= r['upper_bits'] for r in a['rate']) and a['bytes'] <= a['bytes_upper']
        os.makedirs(os.path.join(d, 'ev', 'jobs'))
        with open(os.path.join(d, 'ev', 'jobs', 'val_r1_i0_s1.json'), 'w') as fh:
            json.dump(rec, fh)
        S = L.aggregate(os.path.join(d, 'ev'))
        assert S['gbp']['E3L_342101']['pass'] and S['panels']['val']['seeds']['1']['tiers']['cam_nf']['E3L_342101']['bytes_median'] == a['bytes']
        assert L.verdict(S, [out], None, smoke=True)['verdict'].startswith('INCOMPLETE')


def test_chain_script_guards():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chain_day7_E3L.sh'), encoding='utf-8') as fh:
        c = fh.read()
    assert 'export PYTHONPATH=/mnt/datasets/yuzhou-sheaf-gbp-research/e3r-pyroot:/mnt/datasets/yuzhou-sheaf-gbp-research/e3r-pyroot/exploration_grid' in c
    i_self = c.index('import self-check'); i_busy = c.index('while busy; do sleep 120; done'); i_lease = c.index('START=$(date +%s)')
    assert i_self < i_busy < i_lease
    assert 'g4_e3l.py export' not in c and 'g4_e3r.py export' not in c, 'the E3-R export is reused, never rewritten'
    assert '0.90 * avail' in c and 'BUDGET=3000' in c and '$((START+6600))' in c and '$((START+7150))' in c and '-lt 2000' in c
    assert 'pkill' not in c and '--resume' not in c and '--init-from' not in c
    assert 'D8-54' in c and 'runs/e3l' in c and '[ -e $OUT ]' in c and 'mkdir $OUT.lock' in c
    for k in ('E3L_SHA', 'E3R_SUM_SHA', 'TH1_SHA', 'TH2_SHA', 'EXPREC_SHA', 'CK1_SHA', 'CK2_SHA', 'MAN_SHA', 'PYR_SHA', 'FIT_SHA', 'G_SHA', 'E3R_SHA'):
        assert c.count(f'${k}') >= 1, k                                      # checked by the chain
    assert c.count('--ctrl-ckpt $E3R/s$seed/checkpoints/final.pt') == 1 and 'CUDA_VISIBLE_DEVICES= timeout -k 30 14400 $P -B scripts/g4_e3l.py eval' in c
    assert 'grep -q "FILL_"' in c


def test_dual_step_bounded():
    assert L.dual_step(0.0, 0.5 * L.R_STAR) == 0.0                            # projected at 0
    assert abs(L.dual_step(0.1, 1.5 * L.R_STAR) - (0.1 + 0.5 * L.ETA)) < 1e-15  # unclipped inside [-1, 1] (= the literal rule, normalised)
    assert abs(L.dual_step(0.1, 50 * L.R_STAR) - (0.1 + L.ETA)) < 1e-15      # an untrained entropy model cannot blow lam up
    lam = 0.0
    for _ in range(300):
        lam = L.dual_step(lam, 40 * L.R_STAR)
    assert abs(lam - 300 * L.ETA) < 1e-12
    for _ in range(300):
        lam = L.dual_step(lam, 0.0)
    assert lam == 0.0


def test_constants_d8_54():
    assert L.R_STAR == 20000 and L.BLOCK == (4, 4, 2) and L.D_LAT == 8 and L.W0 > 100 and L.GUARD_FPR == 0.003 and L.GUARD_FPR_SELECT == 0.0015
    assert L.RATE_EVERY == 50 and L.RATE_WINDOW == 4 and L.D8_54['fresh_min_mean'] == -0.01 and 'fresh_sign_check' not in L.D8_54
    C = L.D8_54
    assert (C['lever_furn_margin'], C['lever_bytes_ratio'], C['guard_free'], C['guard_surface'], C['guard_fpr'], C['clean_min'], C['k0_margin'],
            C['cam_demote_margin'], C['not_enough_furn_margin'], C['not_enough_bytes_ratio'], C['min_updates']) == \
        (-0.01, 0.15, 0.005, 0.005, 0.003, 0.99, -0.03, -0.03, -0.03, 0.5, 2000)
    m = L.E3L()
    assert L.n_params(m.dec) == L.n_params(L.Dec(10, 16, 2)) and L.n_params(m) <= 1_500_000
    x = torch.zeros(1, 3, 16, 16, 8)
    z, a = m.enc(x)
    assert z.shape == (1, 8, 4, 4, 4) and a.shape == (1, 4, 4, 4)
    up = L.upsample(torch.arange(8.).reshape(1, 2, 2, 2))
    assert up.shape == (1, 8, 8, 4) and up[0, 3, 4, 1] == 0 * 4 + 1 * 2 + 0 and up[0, 4, 7, 3] == 1 * 4 + 1 * 2 + 1


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t(); print('ok', t.__name__, flush=True)
    print(f'{len(tests)}/{len(tests)} passed')
