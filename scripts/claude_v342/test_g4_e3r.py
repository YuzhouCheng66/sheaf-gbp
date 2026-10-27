"""unit tests for g4_e3r.py (batch E3-R, D8-43). Synthetic arrays and one tiny ray-cast scene, CPU only, ~1 min.
run:  CUDA_VISIBLE_DEVICES= python -B test_g4_e3r.py"""
import os
import sys

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, math, tempfile  # noqa: E401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import g4_e3r as E  # noqa: E402

G = E.G
RNG = np.random.default_rng(20261027)
torch.set_num_threads(2)


# ---------------------------------------------------------------- helpers
def rand_map(shape, p_unk=0.4, rng=RNG):
    L = rng.integers(E.LMIN, E.LMAX + 1, size=shape).astype(np.int8)
    L[rng.random(shape) < p_unk] = G.UNK
    return L


def fake_fit(path, cq, theta, l0q):
    tb = dict(cq=list(map(int, cq)), theta=int(theta), l0q=int(l0q), tq=int(theta + l0q), l0=l0q * E.UNIT)
    with open(path, 'w') as fh:
        json.dump(dict(complete=True, halves=dict(A=dict(tiers=dict(cam_nf=tb)), B=dict(tiers=dict(cam_nf=dict(tb, cq=[0] * E.NL))))), fh)


def small_forest(n, comps):
    """edges forming the given components as paths"""
    edges = []
    for cm in comps:
        for a, b in zip(cm[:-1], cm[1:]):
            edges.append((a, b, 1.0))
    return G.forest(edges, n)


def fake_house(n=4, shape=(16, 16, 8), comps=((0, 1, 2), (3,)), noisy_draws=1, rng=RNG):
    """a HouseData-like object: occ, private maps consistent with occ, clean V3-like maps (sign = truth), noisy maps"""
    H = E.HouseData.__new__(E.HouseData)
    occ = np.zeros(shape, bool); occ[:, :, 0] = True; occ[0] = True
    for _ in range(5):
        x, y = rng.integers(2, shape[0] - 3, 2); occ[x:x + 2, y:y + 2, 1:3] = True
    obj = occ.copy(); obj[:, :, 0] = False; obj[0] = False
    slab = np.zeros(shape, bool); slab[:, :, 0] = True
    private = np.full((n,) + shape, -1, np.int8)
    for a in range(n):
        kn = rng.random(shape) < 0.5
        private[a][kn] = occ[kn].astype(np.int8)
    clean = np.where(private == 1, np.int8(70), np.where(private == 0, np.int8(-40), np.int8(G.UNK))).astype(np.int8)
    maps = [clean]
    for _ in range(noisy_draws):
        m = rand_map((n,) + shape, p_unk=0.45, rng=rng); maps.append(m)
    H.name = 'fake'; H.private = private; H.n = n; H.draws = ['clean'] + [f'cam_nf_s{200 + i}' for i in range(noisy_draws)]
    H.maps = np.stack(maps); H.occ = occ; H.slab = slab; H.obj = obj; H.cell = np.array([0.1, 0.1, 0.1]); H.comps = [list(c) for c in comps]
    pk = np.zeros(shape, np.uint8)
    for a in range(n):
        pk |= ((private[a] >= 0).astype(np.uint8) << a)
    H.pk = pk
    return H.to('cpu')


class SumNet(torch.nn.Module):
    """a stand-in read-out: logit = 2 x S-channel - 0.3 x log(1 + k) (deterministic, no parameters to train)"""

    def forward(self, x):
        return 2.0 * x[:, 0] - 0.3 * x[:, 1]


# ---------------------------------------------------------------- tests
def test_evidence_pluggable():
    ev = E.Evidence.from_spec('L')
    assert np.array_equal(ev.table, np.arange(E.LMIN, E.LMAX + 1)) and ev.theta == 0 and ev.tag() == 'B4'
    cq = RNG.integers(-100, 150, E.NL)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'fit.json'); fake_fit(p, cq, 37, -40)
        ec = E.Evidence.from_spec(f'c:{p}:A:cam_nf')
        assert np.array_equal(ec.table, cq) and ec.theta == 37 and ec.l0q == -40 and ec.tag() == 'B6_A' and ec.desc['sha256'] == E.file_sha256(p)
        er = E.Evidence.from_recipe(json.loads(json.dumps(ec.recipe())))
        assert np.array_equal(er.table, ec.table) and er.theta == ec.theta and er.l0q == ec.l0q
        bad = os.path.join(d, 'bad.json')
        with open(bad, 'w') as fh:
            json.dump(dict(halves=dict(A=dict(tiers=dict(cam_nf=dict(cq=list(map(int, cq)), theta=1, l0q=0, tq=5))))), fh)
        try:
            E.Evidence.from_spec(f'c:{bad}:A:cam_nf')
        except AssertionError:
            pass
        else:
            raise AssertionError('theta != tq - l0q must be refused')
    L = rand_map((7, 6, 5))
    e, k = E.stalk(L, cq)
    assert np.array_equal(k, L != G.UNK) and (e[~k] == 0).all() and np.array_equal(e[k], cq[L[k].astype(np.int64) - E.LMIN])
    eL, _ = E.stalk(L, ev.table)
    assert np.array_equal(eL[k], L[k].astype(np.int32))


def test_prior_counted_once_not_in_input():
    """the network input is the bare evidence sum: no l0 anywhere (two tables that differ only in l0q give identical inputs);
    unknown-to-all voxels have S = 0, k = 0 exactly; features_t == features_np"""
    cq = RNG.integers(-100, 150, E.NL)
    maps = [rand_map((8, 8, 4)) for _ in range(3)]
    with tempfile.TemporaryDirectory() as d:
        p1 = os.path.join(d, 'a.json'); p2 = os.path.join(d, 'b.json'); fake_fit(p1, cq, 10, -39); fake_fit(p2, cq, 70, 21)
        e1 = E.Evidence.from_spec(f'c:{p1}:A:cam_nf'); e2 = E.Evidence.from_spec(f'c:{p2}:A:cam_nf')
    S1, k1 = E.direct_sums(maps, [0, 1, 2], e1.table); S2, k2 = E.direct_sums(maps, [0, 1, 2], e2.table)
    f1, f2 = E.features_np(S1, k1), E.features_np(S2, k2)
    assert np.array_equal(f1, f2), 'the prior leaked into the network input'
    none = k1 == 0
    assert none.any() and (S1[none] == 0).all() and (f1[0][none] == 0).all() and (f1[1][none] == 0).all()
    one = (k1 == 1)
    assert np.allclose(f1[0][one], S1[one] * E.UNIT / E.S_SCALE)
    ft = E.features_t(torch.from_numpy(S1), torch.from_numpy(k1)).numpy()
    assert ft.shape == f1.shape and np.allclose(ft, f1, rtol=0, atol=1e-6)
    import inspect
    assert list(inspect.signature(E.features_np).parameters) == ['S', 'kc'] and list(inspect.signature(E.features_t).parameters) == ['S', 'kc']
    assert np.allclose(f1[2][0, 0], E.zplane(4)) and abs(E.zplane(32)[0] + E.zplane(32)[-1]) < 1e-6


def test_relay_sums_equal_e1_fuse():
    n = 6; comps = [[0, 1, 2, 3], [4, 5]]
    F = small_forest(n, comps)
    maps = [rand_map((9, 8, 6), p_unk=0.5) for _ in range(n)]
    ev = E.Evidence.from_spec('L'); rs = G.RelayStats()
    sums, msgs, msgn = E.relay_sums(maps, F, ev.table, rs)
    _, bel = G.fuse(maps, F, 'sum')
    for ci, cm in enumerate(F['comps']):
        S, kc = sums[ci]
        direct = G.comb_sum([G.sum_payload(maps[a]) for a in cm])
        assert np.array_equal(S, direct[0]) and np.array_equal(kc > 0, direct[1])
        assert np.array_equal(kc, sum((maps[a] != G.UNK).astype(np.int32) for a in cm))
        for a in cm:
            assert np.array_equal(E.linear_readout(S, kc, 0), bel[a]), 'linear L read-out != E1 sum belief'
    assert rs.belief_checks == n and len(msgs) == len(F['directed']) == len(msgn)
    for (s, r), (Sm, Km) in msgs.items():                              # messages: the sender side's sum (receiver branch excluded)
        side = F['sides'][(s, r)]
        assert np.array_equal(Sm, sum(G.sum_payload(maps[a])[0] for a in side))
        assert np.array_equal(msgn[(s, r)][0], sum((maps[a] != G.UNK).astype(np.int32) for a in side))


def test_zero_observation_agent_changes_nothing():
    n = 4; comps = [[0, 1, 2], [3]]
    maps = [rand_map((8, 7, 5)) for _ in range(n)]
    cq = RNG.integers(-100, 150, E.NL)
    F = small_forest(n, comps)
    base, _, _ = E.relay_sums(maps, F, cq)
    maps2 = maps + [np.full((8, 7, 5), G.UNK, np.int8)]                   # a 5th agent that observed nothing, joined to agent 1
    F2 = G.forest([(0, 1, 1.), (1, 2, 1.), (1, 4, 1.)], 5)
    s2, _, _ = E.relay_sums(maps2, F2, cq)
    ci = [i for i, cm in enumerate(F2['comps']) if 4 in cm][0]
    S, kc = s2[ci]
    assert np.array_equal(S, base[0][0]) and np.array_equal(kc, base[0][1])
    assert np.array_equal(E.features_np(S, kc), E.features_np(*base[0]))
    e, k = E.stalk(maps2[4], cq)
    assert not e.any() and not k.any()


def test_known_mask_equals_baseline():
    maps = [rand_map((8, 8, 4), p_unk=0.6) for _ in range(3)]
    S, kc = E.direct_sums(maps, [0, 1, 2], np.arange(E.LMIN, E.LMAX + 1))
    _, bel = G.fuse(maps, small_forest(3, [[0, 1, 2]]), 'sum')
    for j in (0, 1000, 2000, 3999):
        for logit in (RNG.normal(0, 5, S.shape), np.full(S.shape, 100.), np.full(S.shape, -100.)):
            p = E.net_readout(logit, kc, j)
            assert np.array_equal(p >= 0, bel[0] >= 0) and np.array_equal(p >= 0, kc > 0)
    lg = np.array([E.bin_edge(5) - 1e-9, E.bin_edge(5), E.bin_edge(5) + 1e-9, -1e9, 1e9])
    assert E.logit_bin(lg).tolist() == [4, 5, 5, 0, E.NBIN - 1]


def test_clean_samples_label_equals_input():
    """clean draw: known (k > 0) == the scope (union of the subset's private known maps), and sign(sum L) == clean truth there"""
    H = fake_house()
    tl = torch.as_tensor(np.arange(E.LMIN, E.LMAX + 1), dtype=torch.int32)
    for sub in [(0,), (1, 2), (0, 1, 2), (3,)]:
        S, kc, y, m = E.sample_tensors(H, 0, sub, tl, 'scope_known')
        S, kc, y, m = S.numpy(), kc.numpy(), y.numpy(), m.numpy()
        scope = (H.private[list(sub)] >= 0).any(0)
        assert np.array_equal(kc > 0, scope) and np.array_equal(m, scope)
        assert np.array_equal((S > 0)[scope], y[scope]) and np.array_equal(np.where(scope, y.astype(np.int8), -1), H.private[list(sub)].max(0))
        _, _, _, m2 = E.sample_tensors(H, 0, sub, tl, 'scope')
        assert np.array_equal(m2.numpy(), scope)


def test_clean_v3_on_a_raycast_scene():
    """E1 crit. (a) on a tiny ray-cast scene: the clean V3 ternary is the generator's observe() map (the clean label = the input)"""
    shape = (24, 24, 12); cell = np.array([0.1, 0.1, 0.1])
    occ = np.zeros(shape, np.int8); occ[:, :, 0] = 1; occ[0] = 1; occ[:, 0] = 1; occ[10:13, 8:11, 1:4] = 1
    rays = E.W.fibonacci_sphere(3000); p = np.array([1.5, 1.6, 0.6]); ds = 0.05
    hs, ls = G.trace_pose(occ, cell, p, rays)
    scan, _ = G.measure(G.CONDITIONS['clean'], hs, ls, p, G.pose_draws(200, 1, 0, 0, 0, len(rays)), ds, rays)
    L = G.integrate_agent(shape, cell, ds, [scan], ('V3',))['V3']
    obs = E.W.observe(occ, cell, p[None], rays)
    assert np.array_equal(G.tern(L), obs)
    S, kc = E.direct_sums([L], [0], np.arange(E.LMIN, E.LMAX + 1))
    kn = obs >= 0
    assert np.array_equal(kc > 0, kn) and np.array_equal((S > 0)[kn], (occ == 1)[kn])


def test_sample_tensors_match_numpy_and_augment():
    H = fake_house(noisy_draws=2)
    cq = RNG.integers(-100, 150, E.NL); tl = torch.as_tensor(cq, dtype=torch.int32)
    for d, sub in [(1, (0, 2)), (2, (1,)), (1, (0, 1, 2))]:
        S, kc, y, m = E.sample_tensors(H, d, sub, tl)
        dS, dk = E.direct_sums(list(H.maps[d]), list(sub), cq)
        assert np.array_equal(S.numpy(), dS) and np.array_equal(kc.numpy(), dk)
        scope = (H.private[list(sub)] >= 0).any(0)
        assert np.array_equal(m.numpy(), scope & (dk > 0))
        x = E.features_t(S, kc)
        for aug in range(8):
            xa, ya, ma = E.augment(x, y, m, aug)
            assert xa.shape == x.shape
            # the label moves with the input: a voxel's (S, y) pair is preserved as a multiset
            pa = sorted(zip(xa[0].flatten().tolist(), ya.flatten().tolist())); p0 = sorted(zip(x[0].flatten().tolist(), y.flatten().tolist()))
            assert pa == p0
        xa, ya, _ = E.augment(x, y, m, 1)
        assert torch.equal(xa[0, 3, 5], x[0, 5, 3]) and torch.equal(ya[3, 5], y[5, 3])
    rng = np.random.default_rng(1)
    specs = [E.sample_spec(rng, [H]) for _ in range(400)]
    sizes = [len(s[2]) for s in specs]; clean = np.mean([s[1] == 0 for s in specs])
    assert set(sizes) == {1, 2, 3} and abs(clean - E.P_CLEAN) < 0.06
    assert all(set(s[2]) <= set(H.comps[0]) or set(s[2]) <= set(H.comps[1]) for s in specs)


def test_model_size_and_receptive_field():
    torch.manual_seed(0)
    m = E.UNet3(3, 16).double().eval()
    assert E.n_params(m) <= 1_000_000 and E.n_params(m) == 340113 and E.n_params(E.UNet3(3, 16, skip=True)) == 340115
    ms = E.UNet3(3, 16, skip=True, bias0=-0.85).double().eval()                 # at init: the linear read-out (head ~ 0)
    xs_ = torch.zeros(1, 3, 16, 16, 8, dtype=torch.float64); xs_[0, 0] = torch.randn(16, 16, 8, dtype=torch.float64)
    with torch.no_grad():
        ys_ = ms(xs_)
    assert (ys_ - (xs_[:, 0] * E.S_SCALE - 0.85)).abs().max() < 0.05
    x = torch.zeros(1, 3, 64, 32, 16, dtype=torch.float64); x[0, 2] = torch.linspace(-1, 1, 16)
    y0 = m(x)
    x1 = x.clone(); x1[0, 0, 32, 16, 8] = 1.0
    d = (m(x1) - y0).abs()[0] > 1e-12
    xs = np.nonzero(d.any(2).any(1).numpy())[0]
    assert xs.min() <= 32 - 4 and xs.max() >= 32 + 4, 'receptive field < 9 voxels'
    assert xs.min() >= 32 - 24 and xs.max() <= 32 + 24, 'output depends on voxels outside the expected receptive field'
    assert not d[:4].any() and not d[-4:].any()


def test_threshold_selection_matches_score():
    """select_threshold's pooled F1 / free recall at the chosen bin equal G.score on net_readout; the free-recall constraint holds"""
    Hs = [fake_house(noisy_draws=2, rng=np.random.default_rng(s)) for s in (1, 2)]
    ev = E.Evidence.from_spec('L')
    sel = E.select_threshold(SumNet(), ev, Hs, 'cpu')
    tp = fp = fn = fh = ft = 0; ltp = lfp = lfh = 0
    for Hd in Hs:
        Hh = G.House(Hd.occ.astype(np.int8), Hd.slab, Hd.obj, Hd.cell)
        for d in range(1, len(Hd.draws)):
            for cm in Hd.comps:
                T = G.Target(Hd.private[cm].max(0), Hh); w = len(cm)
                S, kc = E.direct_sums(list(Hd.maps[d]), cm, ev.table)
                lg = E.net_logits(SumNet(), [S], [kc], 'cpu')[0]
                r = G.score(E.net_readout(lg, kc, sel['j']), T); rl = G.score(E.linear_readout(S, kc, 0), T)
                tp += r['furn_tp'] * w; fp += r['furn_fp'] * w; fn += r['furn_fn'] * w
                fh += r['free_recall'] * int(T.free.sum()) * w; ft += int(T.free.sum()) * w
                ltp += rl['furn_tp'] * w; lfp += rl['furn_fp'] * w; lfh += rl['free_recall'] * int(T.free.sum()) * w
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    assert abs(f1 - sel['f1']) < 1e-12 and abs(fh / ft - sel['free']) < 1e-9
    assert abs(lfh / ft - sel['linear']['free']) < 1e-9 and sel['free'] + 1e-12 >= sel['linear']['free']
    assert abs(2 * ltp / max(2 * ltp + lfp + (tp + fn - ltp), 1) - sel['linear']['f1']) < 1e-12
    ktp = kfp = kfn = kfh = kft = kfl = 0                                  # the K0-specific threshold, same rule on single agents
    for Hd in Hs:
        Hh = G.House(Hd.occ.astype(np.int8), Hd.slab, Hd.obj, Hd.cell)
        for d in range(1, len(Hd.draws)):
            for a in range(Hd.n):
                T = G.Target(Hd.private[a], Hh); e, k = E.stalk(Hd.maps[d][a], ev.table); ki = k.astype(np.int32)
                lg = E.net_logits(SumNet(), [e], [ki], 'cpu')[0]
                r = G.score(E.net_readout(lg, ki, sel['k0']['j_sep']), T); rl = G.score(E.linear_readout(e, ki, 0), T)
                ktp += r['furn_tp']; kfp += r['furn_fp']; kfn += r['furn_fn']
                nf = int(T.free.sum()); kfh += (r['free_recall'] or 0) * nf; kft += nf; kfl += (rl['free_recall'] or 0) * nf
    assert abs(2 * ktp / max(2 * ktp + kfp + kfn, 1) - sel['k0']['f1_sep']) < 1e-12 and abs(kfh / kft - sel['k0']['free_sep']) < 1e-9
    assert abs(kfl / kft - sel['k0']['linear_free']) < 1e-9 and sel['k0']['free_sep'] + 1e-12 >= sel['k0']['linear_free']


def _arm(furn, k0=.7, k0sep=None, fpr=.02, free=.97, surf=.9):
    R8 = dict(furn_f1=furn, furn_f1_tol1=furn + .05, unknown_known_fpr=fpr, free_recall=free, surface_f1=surf, var_occ_f1=.8)
    cal = {f'k{i}': dict(n=10, slope=1.0, ece=.01) for i in range(1, 5)}; cal['k5+'] = dict(n=10, slope=1.0, ece=.01); cal['kappa'] = 1.
    a = dict(R8=R8, K0=dict(furn_f1=k0), calibration_R8=cal)
    if k0sep is not None:
        a['K0_sep'] = dict(furn_f1=k0sep)
    return a


def _panel(name, d_net, net_k0sep=.80, lin_k0sep=.70, missing=()):
    P = dict(seeds={})
    for ns in ('1', '2'):
        tiers = dict(cam_nf=dict(B4=_arm(.85, k0=.66), B6_A=_arm(.88, k0=.70), B6_B=_arm(.879, k0=.69)), clean=dict())
        for ts in ('342101', '342202'):
            if (ts, ns) not in missing:
                tiers['cam_nf'][f'NET_{ts}'] = _arm(.88 + (d_net[(ts, ns)] if isinstance(d_net, dict) else d_net), k0=.75, k0sep=net_k0sep)
            tiers['cam_nf'][f'LIN_{ts}'] = _arm(.88, k0=.70, k0sep=lin_k0sep)
            tiers['clean'][f'NET_{ts}'] = _arm(.995, free=.995)
        P['seeds'][ns] = dict(houses=[f'r{r}_i{i}' for r, i in G.panel_specs(name)], tiers=tiers)
    return P


def _summary(d_val, d_fresh=.05, **kw):
    S = dict(panels=dict(val=_panel('val', d_val, **kw)), failed=[], skipped=[], e1_mismatch=[], agg_reasons=[])
    if d_fresh is not None:
        S['panels']['fresh'] = _panel('fresh', d_fresh, **{k: v for k, v in kw.items() if k != 'missing'})
    return S


def _runs(d, done=(6000, 6000), planned=(6000, 6000), truncated=(False, False), thresholds=(True, True), seeds=('342101', '342202')):
    out = []
    for i, sd in enumerate(seeds):
        r = os.path.join(d, f'run{sd}{i}'); os.makedirs(r, exist_ok=True)
        with open(os.path.join(r, 'run.json'), 'w') as fh:
            json.dump(dict(seed=int(sd), train_houses=184, recipe=dict(evidence=dict(desc=dict(spec='L')))), fh)
        with open(os.path.join(r, 'final.json'), 'w') as fh:
            json.dump(dict(complete=(done[i] == planned[i] and not truncated[i]), truncated=truncated[i], updates_done=done[i], updates_planned=planned[i]), fh)
        if thresholds[i]:
            with open(os.path.join(r, 'threshold.json'), 'w') as fh:
                json.dump(dict(n_houses=19, houses=['h'] * 19, device='cuda'), fh)
        out.append(r)
    return out


def test_verdict_interface():
    """D8-44 verdict incl. the review's probes 1, 3, 5, 6 and every M3 case"""
    crit = dict(lever_r8_margin=.03, not_enough_margin=.01, k0_margin=-.01)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        R = _runs(d)
        V = lambda S, c=crit, runs=R: E.verdict(S, c, runs)['verdict']  # noqa: E731
        assert V(_summary(.05)) == 'LEVER'
        v = E.verdict(_summary(.05), crit, R)
        assert all(c['control'] == 'B6_A' for c in v['cells'].values()) and v['holdout_houses'] == {'342101': 19, '342202': 19}
        assert V(_summary(.05), None).startswith('PENDING_CRITERIA')
        assert V(_summary(.0, .0)) == 'NOT_ENOUGH'
        low = {(t, n): .05 for t in ('342101', '342202') for n in ('1', '2')}; low[('342202', '2')] = .005
        assert V(_summary(low)) == 'NOT_ENOUGH'
        mid = dict(low); mid[('342202', '2')] = .02
        assert V(_summary(mid)) == 'AMBIGUOUS'
        assert V(_summary(.05, -.05)) == 'AMBIGUOUS (fresh sign opposite to VAL)'
        # review probe 1 / 3: the fresh panel absent or one fresh seed missing -> INCOMPLETE (M3.2)
        assert V(_summary(.05, None)).startswith('INCOMPLETE')
        S = _summary(.05); del S['panels']['fresh']['seeds']['2']
        assert V(S).startswith('INCOMPLETE')
        S = _summary(.05); del S['panels']['fresh']['seeds']['1']['tiers']['cam_nf']['NET_342202']
        assert V(S).startswith('INCOMPLETE')
        S = _summary(.05); S['panels']['fresh']['seeds']['1']['houses'] = S['panels']['fresh']['seeds']['1']['houses'][1:]
        assert V(S).startswith('INCOMPLETE')
        r0, i0 = G.panel_specs('fresh')[0]
        S['skipped'] = [f'val_r{r0}_i{i0}_s1', f'fresh_r{r0}_i{i0}_s2']          # another panel's / seed's skip must not excuse it (S8)
        assert V(S).startswith('INCOMPLETE')
        S['skipped'] = [f'fresh_r{r0}_i{i0}_s1']
        assert V(S) == 'LEVER'
        # a missing VAL cell, a missing control arm (M3.3), a missing clean NET row (M3.4), K0_sep rows (M2)
        assert V(_summary(.05, missing={('342101', '1')})).startswith('INCOMPLETE')
        S = _summary(.05); del S['panels']['val']['seeds']['1']['tiers']['cam_nf']['B6_B']
        assert V(S).startswith('INCOMPLETE')
        S = _summary(.05); del S['panels']['val']['seeds']['2']['tiers']['clean']['NET_342202']
        assert V(S).startswith('INCOMPLETE')
        S = _summary(.05); del S['panels']['val']['seeds']['2']['tiers']['cam_nf']['LIN_342101']['K0_sep']
        assert V(S).startswith('INCOMPLETE')
        # review probe 6: NET K0_sep .68 vs LIN K0_sep .70 (-2 pp < -1 pp), control K0 .66 -> no lever (M2)
        S = _summary(.05, net_k0sep=.68, lin_k0sep=.70)
        for P in S['panels'].values():
            for ns in P['seeds']:
                P['seeds'][ns]['tiers']['cam_nf']['B6_A']['K0']['furn_f1'] = .66
        v = E.verdict(S, crit, R)
        assert v['verdict'] == 'AMBIGUOUS' and all(abs(c['d_K0_sep_net_minus_lin'] + .02) < 1e-9 and not c['guards']['k0'] for c in v['cells'].values())
        assert V(_summary(.05, net_k0sep=.695, lin_k0sep=.70)) == 'LEVER'                # -0.5 pp is inside the -1 pp guard
        # guards: free recall C - 0.5 pp, surface C - 0.5 pp, clean >= .99, FPR must equal the control (S5)
        S = _summary(.05); S['panels']['val']['seeds']['1']['tiers']['cam_nf']['NET_342101']['R8']['free_recall'] = .97 - .006
        assert V(S) == 'AMBIGUOUS'
        S = _summary(.05); S['panels']['val']['seeds']['1']['tiers']['cam_nf']['NET_342101']['R8']['free_recall'] = .97 - .004
        assert V(S) == 'LEVER'
        S = _summary(.05); S['panels']['val']['seeds']['1']['tiers']['cam_nf']['NET_342101']['R8']['surface_f1'] = .9 - .006
        assert V(S) == 'AMBIGUOUS'
        S = _summary(.05); S['panels']['val']['seeds']['2']['tiers']['clean']['NET_342202']['R8']['furn_f1'] = .98
        assert V(S) == 'AMBIGUOUS'
        S = _summary(.05); S['panels']['val']['seeds']['1']['tiers']['cam_nf']['NET_342101']['R8']['unknown_known_fpr'] = .021
        assert V(S).startswith('INCOMPLETE')
        S = _summary(.05); S['e1_mismatch'] = ['val_r1_i0_s1']
        assert V(S).startswith('INCOMPLETE')
        # training runs (M3.1): probe 5 = truncated seed with VAL d = 0 -> NOT_ENOUGH (not INCOMPLETE); truncation only blocks LEVER
        Rt = _runs(os.path.join(d, 't'), done=(6000, 5000), truncated=(False, True))
        assert V(_summary(.0, .0), runs=Rt) == 'NOT_ENOUGH'
        v = E.verdict(_summary(.05), crit, Rt)
        assert v['verdict'] == 'AMBIGUOUS' and v['truncated_seeds'] == ['342202'] and v['notes']
        assert V(_summary(.05), runs=_runs(os.path.join(d, 'u'), done=(6000, 1999), truncated=(False, True))).startswith('INCOMPLETE')
        assert V(_summary(.05), runs=_runs(os.path.join(d, 'p'), done=(6000, 5000), planned=(6000, 5000))).startswith('INCOMPLETE')
        assert V(_summary(.05), runs=_runs(os.path.join(d, 'h'), thresholds=(True, False))).startswith('INCOMPLETE')
        assert V(_summary(.05), runs=_runs(os.path.join(d, 's'), seeds=('342101',))).startswith('INCOMPLETE')
        assert V(_summary(.05), runs=None).startswith('INCOMPLETE')
        assert V(_summary(.05), runs=[]).startswith('INCOMPLETE')


def test_e1c_rule_and_incomplete_fit():
    """D8-44 evidence rule from the E1c summary (M4); incomplete fits refused"""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        fit = os.path.join(d, 'e1c_fit.json'); fake_fit(fit, RNG.integers(-100, 150, E.NL), 40, -39); sha = E.file_sha256(fit)
        for v, want_c in (('LEVER', True), ('AMBIGUOUS', True), ('AMBIGUOUS (fresh sign)', True), ('NOT_ENOUGH', False), ('INCOMPLETE', False),
                          ('INCOMPLETE (smoke readout: LEVER)', False)):
            ev, vv, probs = E.e1c_rule(dict(verdict=dict(verdict=v), fit_sha256=[sha], fit_sha256_file=sha), sha, '/ref/e1c_fit.json')
            assert not probs and vv == v
            assert ev == ({'342101': 'c:/ref/e1c_fit.json:A:cam_nf', '342202': 'c:/ref/e1c_fit.json:B:cam_nf'} if want_c else {'342101': 'L', '342202': 'L'})
        _, _, probs = E.e1c_rule(dict(verdict=dict(verdict='LEVER'), fit_sha256=[sha, 'x'], fit_sha256_file=sha), sha, '/r')
        assert probs
        _, _, probs = E.e1c_rule(dict(verdict=dict(verdict='LEVER'), fit_sha256=[sha], fit_sha256_file='y'), sha, '/r')
        assert probs
        summ = os.path.join(d, 's.json')
        with open(summ, 'w') as fh:
            json.dump(dict(verdict=dict(verdict='LEVER'), fit_sha256=[sha], fit_sha256_file=sha), fh)
        ok = E.main(['e1c-rule', '--summary', summ, '--fit', fit, '--ref-fit-path', '/ref/f.json', '--expect-342101', 'c:/ref/f.json:A:cam_nf',
                     '--expect-342202', 'c:/ref/f.json:B:cam_nf'])
        assert ok['342202'] == 'c:/ref/f.json:B:cam_nf'
        for bad in (['--expect-342101', 'L', '--expect-342202', 'L'], ['--expect-342101', 'c:/ref/f.json:B:cam_nf', '--expect-342202', 'c:/ref/f.json:A:cam_nf']):
            try:
                E.main(['e1c-rule', '--summary', summ, '--fit', fit, '--ref-fit-path', '/ref/f.json'] + bad)
            except SystemExit as e:
                assert e.code == 2
            else:
                raise AssertionError('a filled evidence that contradicts the D8-44 rule must be refused')
        inc = os.path.join(d, 'inc.json')
        with open(inc, 'w') as fh:
            json.dump(dict(complete=False, halves=_jload_fit(fit)['halves']), fh)
        try:
            E.Evidence.from_spec(f'c:{inc}:A:cam_nf')
        except AssertionError:
            pass
        else:
            raise AssertionError('an incomplete E1c fit must be refused')
        assert E.Evidence.from_spec(f'c:{inc}:A:cam_nf', allow_incomplete=True).theta == 40


def _jload_fit(p):
    with open(p) as fh:
        return json.load(fh)


def test_e1_rows_match_and_fingerprint():
    k0 = [dict(furn_f1=0.5, furn_tp=np.int64(3), strata=dict(h_furn=[[1, 2, 3]]))]; r8 = [[dict(furn_f1=0.75), 8]]
    e1 = dict(K0=json.loads(json.dumps(k0, default=int)), R8=dict(sum=[[dict(furn_f1=0.75), 8]]))
    assert E.e1_rows_match([dict(furn_f1=0.5, furn_tp=3, strata=dict(h_furn=[[1, 2, 3]]))], r8, e1)
    assert not E.e1_rows_match([dict(furn_f1=0.5000001, furn_tp=3, strata=dict(h_furn=[[1, 2, 3]]))], r8, e1)
    assert not E.e1_rows_match([dict(furn_f1=0.5, furn_tp=3, strata=dict(h_furn=[[1, 2, 3]]))], [[dict(furn_f1=0.75), 7]], e1)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        a = {k: rand_map((6, 5, 4)) for k in ('L_k0_cam_nf', 'L_k0_cam', 'L_pub_cam', 'private')}
        p1 = os.path.join(d, 'a.npz'); np.savez_compressed(p1, **a); fj = os.path.join(d, 'a.json')
        E.main(['fingerprint', '--npz', p1, '--write', fj])
        p2 = os.path.join(d, 'b.npz'); np.savez_compressed(p2, **{k: v for k, v in a.items() if k != 'private'})
        E.main(['fingerprint', '--npz', p2, '--expect', fj])                    # a subset of arrays is fine if the required ones match
        b = dict(a); b['L_pub_cam'] = b['L_pub_cam'].copy(); b['L_pub_cam'][0, 0, 0] ^= 1
        p3 = os.path.join(d, 'c.npz'); np.savez_compressed(p3, **b)
        for p in (p3,):
            try:
                E.main(['fingerprint', '--npz', p, '--expect', fj])
            except SystemExit as e:
                assert 'FAILED' in str(e)
            else:
                raise AssertionError('a changed array must fail the fingerprint check')
        p4 = os.path.join(d, 'd.npz'); np.savez_compressed(p4, **{k: v for k, v in a.items() if k != 'L_k0_cam'})
        try:
            E.main(['fingerprint', '--npz', p4, '--expect', fj])
        except SystemExit as e:
            assert 'missing' in str(e)
        else:
            raise AssertionError('a missing required array must fail the fingerprint check')


def test_train_two_steps_and_select_end_to_end():
    """S11: gen-format cache (2 synthetic houses) -> train 2 updates on CPU -> threshold.json (all three thresholds) -> select reproduces it"""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        cache = os.path.join(d, 'cache'); os.makedirs(cache)
        houses = []
        for i, (idx, ho) in enumerate(((0, False), (9, True))):
            H = fake_house(noisy_draws=1, rng=np.random.default_rng(10 + i))
            comp_of = np.zeros(H.n, np.int64)
            for ci, cm in enumerate(H.comps):
                comp_of[cm] = ci
            name = f'TRAIN_r1_i{idx}'; p = os.path.join(cache, name + '.npz')
            np.savez_compressed(p, occ=H.occ.astype(np.int8), slab=H.slab, obj=H.obj, cell=H.cell, private=H.private, comp_of=comp_of,
                                edges=np.zeros((0, 3)), L_clean=H.maps[0], L_cam_nf_s200=H.maps[1])
            houses.append(dict(house=name, rooms=1, index=idx, holdout=ho, agents=H.n, components=H.comps, file=name + '.npz', file_sha256=E.file_sha256(p)))
        with open(os.path.join(cache, 'manifest.json'), 'w') as fh:
            json.dump(dict(format=E.CACHE_FORMAT, seeds=[200], houses=houses), fh)
        out = os.path.join(d, 'run')
        fin = E.main(['train', '--cache', cache, '--evidence', 'L', '--seed', '342101', '--out', out, '--updates', '2', '--device', 'cpu', '--batch', '1',
                      '--log-every', '1', '--val-every', '1', '--holdout-samples', '1', '--warmup', '1', '--manifest-sha', E.file_sha256(os.path.join(cache, 'manifest.json'))])
        assert fin['complete'] and fin['updates_done'] == 2 and not fin['truncated']
        th = _jload_fit(os.path.join(out, 'threshold.json'))
        assert th['n_houses'] == 1 and th['tf32'] is False and isinstance(th['k0']['linear_theta_sep'], int) and 'j_sep' in th['k0']
        run = _jload_fit(os.path.join(out, 'run.json'))
        assert run['recipe']['skip'] and run['train_houses'] == 1 and run['holdout_houses'] == 1
        try:
            E.main(['select', '--run', out])
        except SystemExit:
            pass
        else:
            raise AssertionError('select must not overwrite threshold.json silently')
        sel = E.main(['select', '--run', out, '--overwrite'])
        assert sel['j'] == th['j'] and sel['k0']['j_sep'] == th['k0']['j_sep'] and sel['k0']['linear_theta_sep'] == th['k0']['linear_theta_sep']


def test_linear_k0_threshold_brute_force():
    """M2: the linear read-out's K0 threshold (integer theta on single-agent e) = the best feasible theta of a brute-force scan"""
    Hs = [fake_house(noisy_draws=2, rng=np.random.default_rng(s)) for s in (3, 4)]
    ev = E.Evidence.from_spec('L')
    sel = E.select_threshold(SumNet(), ev, Hs, 'cpu')
    rows = []
    for Hd in Hs:
        Hh = G.House(Hd.occ.astype(np.int8), Hd.slab, Hd.obj, Hd.cell)
        for d in range(1, len(Hd.draws)):
            for a in range(Hd.n):
                e, k = E.stalk(Hd.maps[d][a], ev.table); rows.append((e, k.astype(np.int32), G.Target(Hd.private[a], Hh)))

    def pooled(th):
        tp = fp = fn = fh = ft = 0
        for e, k, T in rows:
            r = G.score(E.linear_readout(e, k, th), T); nf = int(T.free.sum())
            tp += r['furn_tp']; fp += r['furn_fp']; fn += r['furn_fn']; fh += (r['free_recall'] or 0) * nf; ft += nf
        return 2 * tp / max(2 * tp + fp + fn, 1), fh / ft
    _, ref_free = pooled(ev.theta)
    scan = {th: pooled(th) for th in range(E.LMIN - 1, E.LMAX + 1)}
    best = max(f for f, fr in scan.values() if fr + 1e-12 >= ref_free)
    f_sel, fr_sel = pooled(sel['k0']['linear_theta_sep'])
    assert abs(f_sel - best) < 1e-12 and fr_sel + 1e-12 >= ref_free
    assert abs(f_sel - sel['k0']['linear_f1_sep']) < 1e-12 and abs(fr_sel - sel['k0']['linear_free_sep']) < 1e-9
    feasible_best = [th for th, (f, fr) in scan.items() if abs(f - best) < 1e-12 and fr + 1e-12 >= ref_free]
    assert sel['k0']['linear_theta_sep'] <= min(feasible_best)                 # ties -> the smallest theta (an equivalent predicate)


def test_selfcheck_import_paths():
    """M1: selfcheck passes when the world's dependencies come from the given root and fails otherwise"""
    assert E.main(['selfcheck', '--pyroot', E.ROOT]) is True
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        try:
            E.main(['selfcheck', '--pyroot', d])
        except SystemExit as e:
            assert 'FAILED' in str(e)
        else:
            raise AssertionError('modules outside the pinned root must fail the self-check')


def test_chain_script_guards():
    """static checks of chain_day7_E3R.sh (M1, M4, M5, S3, ops rules)"""
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chain_day7_E3R.sh'), encoding='utf-8') as fh:
        c = fh.read()
    assert 'export PYTHONPATH=/mnt/datasets/yuzhou-sheaf-gbp-research/e3r-pyroot:/mnt/datasets/yuzhou-sheaf-gbp-research/e3r-pyroot/exploration_grid' in c
    i_self = c.index('import self-check'); i_busy = c.index('while busy; do sleep 120; done'); i_lease = c.index('START=$(date +%s)')
    assert i_self < i_busy < i_lease, 'the import self-check must run before the lease'
    i_rule = c.index('g4_e3r.py e1c-rule'); assert i_rule < i_lease
    assert '[ -e $EXPORT ]' in c and 'PYR_SHA' in c and 'E1C_SUM_SHA' in c and 'E1JOBS_SHA' in c
    assert '0.90 * avail' in c and '0.95 * avail' not in c
    assert 'pkill' not in c and '--resume' not in c and '--init-from' not in c
    assert 'D8-44' in c and 'D8-4x' not in c


def test_calibration_recovers_slope():
    cal = np.zeros((2, E.KMAX, E.NBIN), np.int64)
    x = E.LOGIT_LO + (np.arange(E.NBIN) + .5) * E.LOGIT_STEP
    for k, (a, b) in enumerate(((0., 1.0), (.2, 0.9), (.2, 0.8), (.2, 0.7), (.2, 0.6))):
        p = 1 / (1 + np.exp(-(a + b * x))); n = np.where(np.abs(x) < 6, 4000, 0)
        cal[0, k] = np.rint(n * p); cal[1, k] = n - cal[0, k]
    c = E.calibration(cal)
    for k, b in enumerate((1.0, 0.9, 0.8, 0.7, 0.6)):
        key = f'k{k + 1}' if k < E.KMAX - 1 else f'k{E.KMAX}+'
        assert abs(c[key]['slope'] - b) < 0.01
    assert c['k1']['ece'] < 0.005 and c['k5+']['ece'] > 0.02             # calibrated layer vs over-confident layer
    assert abs(c['kappa'] - 0.6) < 0.02


def test_reserved_seeds_and_train_seeds():
    for argv in (['gen', '--cache', os.devnull, '--seeds', '1,200'], ['gen', '--cache', os.devnull, '--seeds', '101']):
        try:
            E.main(argv)
        except AssertionError:
            pass
        else:
            raise AssertionError(f'{argv}: reserved noise seeds must be refused')
    assert set(E.GEN_SEEDS).isdisjoint(E.RESERVED_SEEDS) and E.TRAIN_SEEDS == (342101, 342202)
    assert all(E.is_holdout(i) == (i % 10 == 9) for i in range(70)) and sum(E.is_holdout(i) for _, i in E.train_plan()) == 21


def test_lr_schedule():
    assert abs(E.lr_at(0, 1000, 1e-3, 100) - 1e-5) < 1e-12 and abs(E.lr_at(99, 1000, 1e-3, 100) - 1e-3 * 0.5 * (1 + math.cos(math.pi * 99 / 1000))) < 1e-12
    assert E.lr_at(1000, 1000, 1e-3, 100) < 1e-12


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t(); print('ok', t.__name__, flush=True)
    print(f'{len(tests)}/{len(tests)} passed')
