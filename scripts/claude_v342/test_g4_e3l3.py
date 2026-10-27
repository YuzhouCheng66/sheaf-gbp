"""unit tests for g4_e3l3.py (batch E3-L3, D8-63). Synthetic arrays and tiny synthetic caches, CPU only, a few minutes.
run:  CUDA_VISIBLE_DEVICES= python -B test_g4_e3l3.py"""
import os
import sys

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, re, tempfile  # noqa: E401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import g4_e3l3 as M  # noqa: E402
import test_g4_e3l as TL  # noqa: E402   (helpers only: fake houses, random trees / maps, synthetic caches; read only)

L = M.L
L2 = M.L2
R = M.R
G = M.G
RNG = np.random.default_rng(20261028)
torch.set_num_threads(2)
TABLE = torch.as_tensor(np.arange(R.LMIN, R.LMAX + 1), dtype=torch.int32)


def _model(width=4, seed=0, w0=M.W0):
    torch.manual_seed(seed)
    m = M.E3L3(width, w0)
    with torch.no_grad():                                         # a non-trivial code (the UNet3 head starts at ~0, i.e. z = 0)
        m.enc.head.weight.normal_(0, 2.0); m.enc.head.bias.fill_(0.1)
    return m


def _ternary_case(m, g, rng, w):
    Fl = G.forest(TL.rand_tree(m, rng), m)
    lam = (rng.random((m,) + g) < 0.6).astype(np.float64); lam[:, 0, 0, 0] = 0.0
    z = rng.integers(-1, 2, (m,) + g).astype(np.float64) * lam
    return Fl, torch.from_numpy(lam), torch.from_numpy(z), w


# ---------------------------------------------------------------- encoder, source, footprint
def test_encoder_bounded_ternary_footprint():
    model = _model(4)
    Lm = torch.from_numpy(TL.rand_map((3, 16, 16, 8), 0.5, RNG))
    e, kn = L.evidence_t(Lm, TABLE)
    f = R.features_t(e, kn.to(torch.int32))
    zhat, lam = model.encode(f, kn)
    assert zhat.shape == kn.shape and lam.shape == kn.shape
    assert torch.equal(lam, kn.to(lam.dtype)), 'Lam must be the footprint indicator'
    assert float(zhat.abs().max()) < 1.0 and not bool(zhat[~kn].any()), 'z_hat bounded and 0 off the footprint'
    for q in ('round', 'ste', None):
        h, lq, z = M.source(zhat, lam, q)
        assert set(torch.unique(z).tolist()) <= {-1.0, 0.0, 1.0} and not bool(z[~kn].any())
        assert torch.equal(h[:, 0], z * lam) and torch.equal(lq, lam)
        assert len(set(torch.unique(z[kn]).tolist())) >= 2, 'the test model must produce a non-trivial code'
    zh = zhat.detach().clone().requires_grad_(True)
    h, lq, z = M.source(zh, lam, 'ste')
    gr = torch.from_numpy(RNG.normal(0, 1, tuple(z.shape))).float()
    (z * gr).sum().backward()
    assert torch.equal(zh.grad, gr), 'straight-through rounding: identity backward'
    assert torch.equal(z.detach(), torch.round(zhat)) and float((torch.round(zhat) - zhat).abs().max()) <= 0.5


def test_encoder_gradient_reaches_u_and_init_is_zero_code():
    torch.manual_seed(1)
    m = M.E3L3(4)
    Lm = torch.from_numpy(TL.rand_map((2, 16, 16, 8), 0.5, RNG)); e, kn = L.evidence_t(Lm, TABLE)
    zhat, lam = m.encode(R.features_t(e, kn.to(torch.int32)), kn)
    assert float(zhat.abs().max()) < 0.5, 'UNet3 head N(0, 1e-3): the initial code is z = 0 everywhere'
    h, lq, z = M.source(zhat, lam, 'ste')
    (z * torch.randn_like(z)).sum().backward()
    assert m.enc.head.weight.grad is not None and float(m.enc.head.weight.grad.abs().sum()) > 0


# ---------------------------------------------------------------- GaBP: closed form, energy, w -> inf
def test_gabp_closed_form_and_monotone():
    for t in range(30):
        m = int(RNG.integers(1, 9))
        Fl, lam, z, w = _ternary_case(m, (4, 4, 3), RNG, float(RNG.choice([0.3, 3.0, 1000.0, float(np.exp(RNG.normal(6, 1)))])))
        c = L.gbp_checks(z.unsqueeze(1), lam, Fl, w)
        assert c['cf_max'] <= 1e-9 and c['violations'] == 0, (t, c)


def test_w_infinity_equals_integer_sums():
    for t in range(30):
        m = int(RNG.integers(1, 9))
        for w in (1e9, M.W0):                                      # the initial w = 1000 is already exact for m <= 8 (|h|, Lam <= 7 per side)
            Fl, lam, z, _ = _ternary_case(m, (5, 4, 3), RNG, w)
            h = (lam * z).unsqueeze(1)
            hb, lb, msgs = L.gabp(h, lam, Fl, w, 'round')
            S = z.sum(0); k = lam.sum(0)
            for a in range(m):
                assert torch.equal(hb[a, 0], S) and torch.equal(lb[a], k), (t, w, a)
            for (s, r), (hm, lm) in msgs.items():                  # each message = the integer sums (sum z, k) of the sender side
                side = _side(Fl, s, r)
                assert torch.equal(hm[0], z[side].sum(0)) and torch.equal(lm, lam[side].sum(0))
            hf, lf, _ = L.gabp(h, lam, Fl, w, None)                # float GaBP: the information-weighted mean, = S / k
            x = L.belief_mean(hf, lf)[:, 0]
            pos = k > 0
            assert float((x[:, pos] - (S / k.clamp(min=1))[pos]).abs().max()) < (1e-6 if w >= 1e9 else 0.05)


def _side(Fl, s, r):
    """the members on s's side of the edge s - r"""
    seen = {s, r}; out = [s]; i = 0
    while i < len(out):
        for v in Fl['nb'][out[i]]:
            if v not in seen:
                seen.add(v); out.append(v)
        i += 1
    return out


def test_small_w_breaks_known_and_is_caught():
    """P3: with w <= 1 a quantised Lam message w / (w + 1) rounds to 0, so Lam_b > 0 no longer equals k > 0 -- the mismatch counter sees it"""
    Fl = G.forest([(1, 0, 1.0)], 2)
    lam = torch.zeros(2, 2, 1, 1, dtype=torch.float64); lam[0, 0] = 1.0; lam[1, 1] = 1.0
    z = torch.ones_like(lam) * lam
    hb, lb, _ = L.gabp((lam * z).unsqueeze(1), lam, Fl, 0.9, 'round')
    k = lam.sum(0)
    assert int(((lb[0] > 0) != (k > 0)).sum()) == 1
    hb, lb, _ = L.gabp((lam * z).unsqueeze(1), lam, Fl, 1.5, 'round')
    assert int(((lb[0] > 0) != (k > 0)).sum()) == 0


# ---------------------------------------------------------------- known = Lam_b > 0 == E3-R k > 0, forward batch, loss mask
def test_known_equals_e3r_k_bitwise():
    H = TL.fake_house(n=5, edges=((0, 1), (1, 2), (3, 4)), noisy_draws=1, rng=np.random.default_rng(3))
    model = _model(4); w = float(model.w.detach())
    with torch.no_grad():
        for d in range(len(H.draws)):
            maps = H.maps[d]
            zhat, lam = L.encode_maps(model, list(maps), TABLE, 'cpu')
            h, lq, z = M.source(zhat, lam, 'round')
            for cm in H.comps:
                hb, lb, _ = L.gabp(h[cm], lq[cm], H.local(cm), w, 'round')
                _, kc = R.direct_sums(maps, cm, np.arange(R.LMIN, R.LMAX + 1))
                for i in range(len(cm)):
                    assert np.array_equal(lb[i].numpy() > 0, kc > 0)
                    lg = np.zeros(kc.shape, np.float32)
                    p = R.net_readout(lg, kc, 0)
                    assert np.array_equal(p >= 0, lb[i].numpy() > 0)       # the read-out's known mask is the E3-R one
                    assert np.array_equal(lb[i].numpy(), kc.astype(np.float64)) and np.array_equal(hb[i, 0].numpy(), z[cm].sum(0).numpy())


def test_forward_batch_mask_padding_and_gradients():
    houses = [TL.fake_house(n=4, edges=((0, 1), (1, 2), (2, 3)), rng=np.random.default_rng(5 + i)) for i in range(2)]
    model = _model(4)
    rng = np.random.default_rng(0)
    specs = [L.sample_spec(rng, houses) for _ in range(3)]
    lg, yo, m, src = M.forward_batch(model, specs, houses, TABLE, 'ste', chunk=8)
    n = sum(len(s[2]) for s in specs)
    assert lg.shape == (3, 16, 16, 8) and src['n'] == n and src['npad'] == (-n) % 8
    for b, (h, d, sub, aug, dec) in enumerate(specs):
        Lsub = houses[h].t_maps[d][list(sub)]
        k = L.aug_t((Lsub != R.UNK).sum(0), aug)
        scope = L.aug_t((houses[h].t_pk.int() & sum(1 << a for a in sub)) != 0, aug)
        assert torch.equal(m[b], scope & (k > 0)), 'loss mask = scope & k > 0 (E3-R scope_known)'
        assert torch.equal(src['lb'][b], k.to(src['lb'].dtype))
    loss = R.masked_bce(lg, yo.float(), m.float()); loss.backward()
    ge = sum(float(q.grad.abs().sum()) for q in model.enc.parameters() if q.grad is not None)
    gd = sum(float(q.grad.abs().sum()) for q in model.dec.parameters() if q.grad is not None)
    assert ge > 0 and gd > 0
    if any(len(s[2]) > 1 for s in specs):
        assert model.log_w.grad is not None
    lg2, _, m2, _ = M.forward_batch(model, specs, houses, TABLE, 'ste', chunk=None)  # padding never changes the result
    assert torch.allclose(lg, lg2, atol=1e-5) and torch.equal(m, m2)


# ---------------------------------------------------------------- bytes (O7)
def test_wire_bytes_o7_mask_once():
    for t in range(5):
        m = int(RNG.integers(2, 6))
        Fl, lam, z, w = _ternary_case(m, (16, 16, 8), RNG, 1e9)
        h = (lam * z).unsqueeze(1)
        _, _, msgs = L.gabp(h, lam, Fl, w, 'round')
        for e in Fl['directed']:
            hi, li = L.msg_ints(*msgs[e])
            kn = li > 0
            wb = L.wire_bytes(hi, li)
            assert hi.shape[0] == 1
            assert abs(wb - (L.mask_bytes(kn) + L.value_bytes_given_mask(li, kn) + L.value_bytes_given_mask(hi[0], kn))) < 1e-9
            assert wb <= L.int_bytes(hi, li) + 1e-9, 'mask once <= mask counted per channel (upper)'
    # a constant all-zero-h message is cheap; random signs cost more
    li = np.ones((16, 16, 8), np.int64); h0 = np.zeros((1, 16, 16, 8), np.int64); h1 = RNG.integers(-1, 2, (1, 16, 16, 8))
    assert L.wire_bytes(h0, li) < L.wire_bytes(h1, li)


# ---------------------------------------------------------------- thresholds
def test_threshold_rules():
    nb = R.NBIN
    A = np.zeros(nb, np.int64); B = np.zeros(nb, np.int64); C = np.zeros(nb, np.int64)
    A[3000] = 90; A[2500] = 10; B[2600] = 5; B[1500] = 50; C[1000] = 990          # predicted free = bins below j; max free recall .99
    acc = dict(R8=[A, B, C], K0=[A, B, C]); tot = dict(R8=[100, 1000, 7, 1000], K0=[100, 1000, 7, 1000])
    ctrl = dict(R8=dict(free=.99, fpr=.007, furn_f1=.9), K0=dict(free=.9975, fpr=.007, furn_f1=.8))
    s = M.pick_r8_k0(acc, tot, ctrl)
    r8 = s['R8']
    assert r8['feasible_occ'] and abs(r8['free_min'] - (.99 - .0025)) < 1e-15
    # feasible: j >= 1001 (free .99 >= .9875). F1: (1500, 2500] -> tp 100, fp 5 = 200 / 205 (max); (2600, 3000] -> 180 / 190; (1000, 1500] -> 200 / 255
    assert r8['j_occ'] == 1501, r8['j_occ']                         # ties -> the smallest bin
    assert abs(r8['f1'] - 200 / 205) < 1e-12 and abs(r8['fpr'] - .007) < 1e-15 and abs(r8['free'] - .99) < 1e-12
    k0 = s['K0']                                                    # free >= .995: infeasible (max .99) -> the R8 threshold, flagged
    assert not k0['feasible_occ'] and k0['k0_fallback_r8'] and k0['j_occ'] == r8['j_occ'] and k0['own']['j_occ'] == 1001
    s2 = M.pick_r8_k0(acc, tot, dict(ctrl, K0=dict(free=.99, fpr=.007, furn_f1=.8)))
    assert s2['K0']['feasible_occ'] and not s2['K0']['k0_fallback_r8'] and s2['K0']['j_occ'] == 1501
    s3 = M.pick_r8_k0(acc, tot, dict(ctrl, R8=dict(free=.9999, fpr=.007, furn_f1=.9)))   # R8 infeasible: max free, smallest bin, flagged
    assert not s3['R8']['feasible_occ'] and s3['R8']['j_occ'] == 1001


# ---------------------------------------------------------------- verdict D8-63
def _summ(d=-.005, ratio=.12, d_fresh=-.004, d_cam=-.01, clean=.995, k0=-.01, missing=(), gbp_ok=True, fpr_delta=0.0, mism=0, free=.997, surf=.99):
    Sl = dict(panels={}, failed=[], skipped=[], e1_mismatch=[], agg_reasons=[],
              gbp={f'E3L3_{t}': dict(cf_max=1e-12 if gbp_ok else 1e-3, violations=0, components=9, **{'pass': gbp_ok}) for t in ('342101', '342202')},
              known_mismatch={f'E3L3_{t}': mism for t in ('342101', '342202')})
    Sc = dict(panels={})
    for panel, dd in (('val', d), ('fresh', d_fresh)):
        houses = [f'r{r}_i{i}' for r, i in G.panel_specs(panel)]
        Pl = dict(seeds={}); Pc = dict(seeds={})
        for ns in ('1', '2'):
            tl = dict(cam_nf={}, cam={}, clean={}); tc = dict(cam_nf={}, cam={}, clean={})
            for ts in ('342101', '342202'):
                c = TL._row(.97); c['rnet_relay_median'] = dict(sum=665000., count=88000.)
                tc['cam_nf'][f'NET_{ts}'] = c; tc['cam'][f'NET_{ts}'] = TL._row(.80); tc['clean'][f'NET_{ts}'] = TL._row(.997)
                if (panel, ts, ns) not in missing:
                    a = TL._row(.97 + dd, k0=.94 + k0, fpr=.018 + fpr_delta, free=free, surf=surf); a['bytes_median'] = ratio * 753000.
                    a['known_mismatch'] = mism
                    tl['cam_nf'][f'E3L3_{ts}'] = a
                tl['cam'][f'E3L3_{ts}'] = TL._row(.80 + d_cam); tl['clean'][f'E3L3_{ts}'] = TL._row(clean)
            Pl['seeds'][ns] = dict(houses=houses, tiers=tl); Pc['seeds'][ns] = dict(houses=houses, tiers=tc)
        Sl['panels'][panel] = Pl; Sc['panels'][panel] = Pc
    return Sl, Sc


def _runs(d, done=(9000, 9000), planned=(9000, 9000), truncated=(False, False), thr=(True, True)):
    out = []
    for i, sd in enumerate(('342101', '342202')):
        r = os.path.join(d, f'run{sd}'); os.makedirs(r, exist_ok=True)
        M._jdump(dict(seed=int(sd), train_houses=186, recipe=dict(evidence=dict(desc=dict(spec='c:x:A:cam_nf')))), os.path.join(r, 'run.json'))
        M._jdump(dict(complete=(done[i] == planned[i] and not truncated[i]), truncated=truncated[i], updates_done=done[i], updates_planned=planned[i]),
                 os.path.join(r, 'final.json'))
        if thr[i]:
            M._jdump(dict(n_houses=19, R8=dict(j_occ=2000), K0=dict(j_occ=2000)), os.path.join(r, 'threshold.json'))
        out.append(r)
    return out


def test_verdict_d8_63():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        Rn = _runs(d)

        def V(Sl, Sc, runs=Rn, full=False):
            p = os.path.join(d, 'ctrl.json'); M._jdump(Sc, p)
            v = M.verdict(Sl, runs, p)
            return v if full else v['verdict']
        assert V(*_summ()) == 'LEVER'
        v = V(*_summ(), full=True)
        assert all(abs(c['bytes_ratio'] - .12) < 1e-12 and c['guards_pass'] for c in v['cells'].values())
        assert V(*_summ(d=-.0105)) == 'AMBIGUOUS' and V(*_summ(d=-.031)) == 'NOT_ENOUGH' and V(*_summ(d=-.0099)) == 'LEVER'
        assert V(*_summ(ratio=.151)) == 'AMBIGUOUS' and V(*_summ(ratio=.501)) == 'NOT_ENOUGH' and V(*_summ(ratio=.15)) == 'LEVER'
        assert V(*_summ(d_cam=-.031)).startswith('AMBIGUOUS (lever demoted: cam') and V(*_summ(d_cam=-.029)) == 'LEVER'
        assert V(*_summ(d_fresh=-.0105)).startswith('AMBIGUOUS (lever demoted: fresh') and V(*_summ(d_fresh=+.01)) == 'LEVER'
        assert V(*_summ(d=-.031, d_fresh=-.05)) == 'NOT_ENOUGH' and V(*_summ(d=-.0105, d_fresh=.05)) == 'AMBIGUOUS'   # fresh never upgrades
        assert V(*_summ(d=-.031, d_cam=-.10)) == 'NOT_ENOUGH'                                                            # cam only demotes LEVER
        assert V(*_summ(clean=.989)) == 'AMBIGUOUS' and V(*_summ(k0=-.031)) == 'AMBIGUOUS' and V(*_summ(k0=-.029)) == 'LEVER'
        assert V(*_summ(free=.997 - .0051)) == 'AMBIGUOUS' and V(*_summ(surf=.99 - .0051)) == 'AMBIGUOUS' and V(*_summ(free=.997 - .0049)) == 'LEVER'
        # assertions -> INCOMPLETE: GaBP, known mask, FPR not bit-identical
        assert V(*_summ(gbp_ok=False)).startswith('INCOMPLETE') and 'GaBP assertion failed' in V(*_summ(gbp_ok=False))
        assert V(*_summ(mism=3)).startswith('INCOMPLETE') and V(*_summ(fpr_delta=1e-6)).startswith('INCOMPLETE')
        Sl, Sc = _summ(); del Sl['gbp']['E3L3_342101']
        assert V(Sl, Sc).startswith('INCOMPLETE')
        # INCOMPLETE: threshold missing, < 2,000 updates, different counts, missing cells / control / K0 rows, no control, no runs, failed jobs
        assert V(*_summ(), runs=_runs(os.path.join(d, 't'), thr=(True, False))).startswith('INCOMPLETE')
        assert V(*_summ(), runs=_runs(os.path.join(d, 'u'), done=(9000, 1999), truncated=(False, True))).startswith('INCOMPLETE')
        assert V(*_summ(), runs=_runs(os.path.join(d, 'p'), done=(9000, 8000), planned=(9000, 8000))).startswith('INCOMPLETE')
        assert V(*_summ(), runs=_runs(os.path.join(d, 'tr'), done=(9000, 7000), truncated=(False, True))) == 'AMBIGUOUS'   # truncation blocks LEVER
        assert V(*_summ(missing={('val', '342101', '1')})).startswith('INCOMPLETE')
        assert V(*_summ(missing={('fresh', '342202', '2')})).startswith('INCOMPLETE')
        Sl, Sc = _summ(); del Sc['panels']['val']['seeds']['1']['tiers']['cam_nf']['NET_342202']
        assert V(Sl, Sc).startswith('INCOMPLETE')
        Sl, Sc = _summ(); del Sl['panels']['val']['seeds']['1']['tiers']['cam_nf']['E3L3_342202']['K0_sep']
        assert V(Sl, Sc).startswith('INCOMPLETE')
        Sl, Sc = _summ(); Sl['failed'] = ['val_r1_i0_s1']
        assert V(Sl, Sc).startswith('INCOMPLETE')
        assert M.verdict(_summ()[0], Rn, None)['verdict'].startswith('INCOMPLETE') and M.verdict(_summ()[0], None, None)['verdict'].startswith('INCOMPLETE')
        C = M.D8_63
        assert C['prereg'] == 'D8-63' and C['lever_bytes_ratio'] == 0.15 and C['lever_furn_margin'] == -0.01 and C['fresh_min_mean'] == -0.01
        assert C['not_enough_bytes_ratio'] == 0.5 and C['not_enough_furn_margin'] == -0.03 and C['k0_margin'] == -0.03 and C['cam_demote_margin'] == -0.03
        assert C['guard_free'] == 0.005 and C['guard_surface'] == 0.005 and C['clean_min'] == 0.99 and C['guard_fpr'] == 0.0 and C['min_updates'] == 2000
        assert M.FREE_MARGIN_SELECT == 0.0025 and M.BUDGET_S == 6000 and C['judged_tier'] == 'cam_nf' and C['train_seeds'] == ['342101', '342202']


def test_report_refs():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        e3l = os.path.join(d, 'e3l.json'); e5 = os.path.join(d, 'e5.json')
        M._jdump(dict(panels=dict(val=dict(seeds={'1': dict(tiers=dict(cam_nf=dict(E3L_342101=dict(R8=dict(furn_f1=.79), bytes_median=9e4))))}))), e3l)
        M._jdump(dict(panels=dict(val={'1': dict(cells={'342101': dict(cam_nf=dict(QPOST4=dict(furn=.97, d_furn=-.002, ratio=.19)))})})), e5)
        rep = M._report_refs({'342101|1': {}, '342202|2': {}}, e3l=e3l, e5=e5)
        assert rep['342101|1']['E3L']['R8_furn'] == .79 and rep['342101|1']['QPOST4']['bytes_ratio'] == .19
        assert rep['342202|2']['E3L'] is None and rep['342202|2']['QPOST4'] is None


# ---------------------------------------------------------------- end to end on a synthetic cache (train -> select -> eval -> aggregate)
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
        ctrl = os.path.join(d, 'ctrl.json')
        L2.main(['ctrlprep', '--cache', cache, '--e3r', e3r, '--out', ctrl, '--smoke', '--seeds-smoke', '342101', '--workers', '1'])
        out = os.path.join(d, 's342101')
        fin = M.main(['train', '--cache', cache, '--evidence', ev, '--seed', '342101', '--out', out, '--updates', '4', '--device', 'cpu', '--batch', '2',
                      '--width', '4', '--log-every', '1', '--probe-every', '2', '--holdout-samples', '2', '--warmup', '1', '--manifest-sha', man_sha,
                      '--time-from', '2', '--snap', '3', '--ctrl-ckpt', os.path.join(e3r, 's342101', 'checkpoints', 'final.pt'), '--ctrl-holdout', ctrl])
        assert fin['complete'] and fin['updates_done'] == 4 and fin['snapshot']['updates'] == 3 and fin['rate_s_per_update'] > 0 and fin['probes'] == 2
        assert os.path.exists(os.path.join(out, 'checkpoints', 'upd3.pt'))
        hist = json.load(open(os.path.join(out, 'history.json')))
        pr = hist[-1]['probe']
        for k in ('furn_f1_at0', 'furn_f1_oracle', 'z_minus', 'z_zero', 'z_plus', 'z_entropy_bits', 'tanh_saturation', 'w', 'known_mismatch', 'bytes_ratio_median'):
            assert k in pr, k
        assert pr['known_mismatch'] == 0 and abs(pr['z_minus'] + pr['z_zero'] + pr['z_plus'] - 1) < 1e-9 and pr['bytes_ratio_median'] > 0
        th = M.main(['select', '--run', out, '--ctrl', ctrl, '--workers', '1'])
        cp = json.load(open(ctrl))
        assert abs(th['R8']['free_min'] - (cp['readouts']['342101']['R8']['free'] - 0.0025)) < 1e-12 and th['known_mismatch'] == 0
        assert th['R8']['fpr'] == cp['readouts']['342101']['R8']['fpr'] and 'k0_fallback_r8' in th['K0'] and th['holdout_bytes']['median_ratio'] > 0
        try:
            M.main(['select', '--run', out, '--ctrl', ctrl, '--workers', '1'])
        except SystemExit as e:
            assert 'never overwrite' in str(e)
        else:
            raise AssertionError('select must never overwrite threshold.json')
        ex = os.path.join(d, 'export'); os.makedirs(os.path.join(ex, 'records'))
        H = TL.fake_house(noisy_draws=1, rng=np.random.default_rng(42))
        arr = dict(occ=H.occ.astype(np.int8), slab=H.slab, obj=H.obj, cell=H.cell, private=H.private, edges=R.edges_array(H.edges))
        for tn in R.EVAL_TIERS:
            arr[f'L_k0_{tn}'] = H.maps[0] if tn == 'clean' else H.maps[1]
        arr['L_pub_cam'] = H.maps[1][::-1].copy()
        fp = os.path.join(ex, 'train_r1_i0_s203.npz'); np.savez_compressed(fp, **arr)
        rec_e = dict(panel='train', rooms=1, index=0, seed=203, split='TRAIN', file='train_r1_i0_s203.npz', file_sha256=R.file_sha256(fp), components=H.comps)
        M._jdump(rec_e, os.path.join(ex, 'records', 'train_r1_i0_s203.json'))
        ckf = os.path.join(out, 'checkpoints', 'final.pt')
        rec = M.eval_job((rec_e, dict(export=ex, ckpts=[ckf], thr={ckf: os.path.join(out, 'threshold.json')}, ckpt_shas={}, tiers=list(R.EVAL_TIERS),
                                      threads=1, chunk=2, script_sha256='x')))
        assert not rec.get('failed'), rec.get('traceback')
        a = rec['tiers']['cam_nf']['E3L3_342101']
        assert len(a['R8']) == H.n and len(a['K0_sep']) == H.n and 0 < a['bytes'] <= a['bytes_upper'] and a['known_mismatch'] == 0
        assert a['gbp']['violations'] == 0 and a['gbp']['cf_max'] <= L.CF_TOL and a['gbp']['components'] == len(H.comps)
        assert rec['tiers']['cam']['E3L3_342101']['known_mismatch'] == 0 and len(a['rate']) == H.n and sum(a['z'][:3]) == a['z'][4]
        # the R8 read-out's FPR is the control's: known = k > 0 of the component (L_pub)
        for ci, cm in enumerate(H.comps):
            kc = sum((arr['L_pub_cam'][q] != R.UNK).astype(np.int32) for q in cm)
            assert kc.shape == H.occ.shape
        out_e = os.path.join(d, 'ev')
        M.main(['eval', '--export', ex, '--out', out_e, '--ckpt', ckf, '--workers', '1'])
        S = M.aggregate(out_e)
        e = S['panels']['train']['seeds']['203']['tiers']['cam_nf']['E3L3_342101']
        assert S['gbp']['E3L3_342101']['pass'] and S['known_mismatch']['E3L3_342101'] == 0 and e['bytes_median'] > 0 and 'self_map' in e and 'z' in e
        assert M.verdict(S, [out], None, smoke=True)['verdict'].startswith('INCOMPLETE')
        try:
            M.main(['eval', '--export', ex, '--out', out_e, '--ckpt', ckf, '--workers', '1'])
        except SystemExit as e2:
            assert 'never overwrite' in str(e2)
        else:
            raise AssertionError('eval must never overwrite job files')


# ---------------------------------------------------------------- the bigboy chain (static)
def test_chain_script_guards():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chain_bigboy_E3L3.sh'), encoding='utf-8') as fh:
        c = fh.read()
    assert 'cd ~/claude-sheaf-gbp/work' in c or 'cd $HOME/claude-sheaf-gbp/work' in c
    assert 'export PYTHONPATH=$HOME/claude-sheaf-gbp/data/e3r-pyroot:$HOME/claude-sheaf-gbp/data/e3r-pyroot/exploration_grid' in c
    assert 'export TMPDIR=$HOME/claude-sheaf-gbp/tmp' in c and 'export CUDA_DEVICE_ORDER=PCI_BUS_ID' in c
    assert 'P=$HOME/miniconda3/envs/mlpsp/bin/python' in c and 'conda ' not in c and 'pip ' not in c
    assert 'LOG=$HOME/claude-sheaf-gbp/work/runs/chain_bigboy.log' in c and 'OUT=runs/e3l3' in c
    # GPUs: only 0 and 1; seeds pinned to them; never 2 / 3
    assert 'CUDA_VISIBLE_DEVICES=0 ' in c and 'CUDA_VISIBLE_DEVICES=1 ' in c
    assert not re.search(r'CUDA_VISIBLE_DEVICES=[23]', c) and not re.search(r'CUDA_VISIBLE_DEVICES=0,1', c)
    gl = [ln for ln in c.splitlines() if re.search(r'CUDA_VISIBLE_DEVICES=\d', ln) and ' train ' in ln]
    assert len(gl) == 4, 'two preflights + two trainings'
    for ln in gl:
        assert ("CUDA_VISIBLE_DEVICES=0 " in ln and '--seed 342101' in ln and '"$EV_342101"' in ln) or \
               ("CUDA_VISIBLE_DEVICES=1 " in ln and '--seed 342202' in ln and '"$EV_342202"' in ln), ln
        assert "--expect-gpu 'RTX 6000 Ada'" in ln and ln.rstrip().endswith('&') and 'timeout -k 30' in ln
    assert sum('--ctrl-ckpt $E3R/s342101' in ln for ln in gl) == 1 and sum('--ctrl-ckpt $E3R/s342202' in ln for ln in gl) == 1
    for ln in c.splitlines():                                     # every CPU stage blanks the GPUs
        if re.search(r'g4_e3l3\.py (select|eval|aggregate|selfcheck)', ln):
            assert 'CUDA_VISIBLE_DEVICES= ' in ln, ln
    i_self = c.index('import self-check'); i_busy = c.index('while gpus_busy'); i_pf = c.index('--max-updates 400'); i_tr = c.index('--updates $UPD')
    i_sel = c.index('g4_e3l3.py select'); i_ev = c.index('g4_e3l3.py eval'); i_ag = c.index('g4_e3l3.py aggregate')
    assert i_self < i_busy < i_pf < i_tr < i_sel < i_ev < i_ag
    assert '--time-from 200' in c and '0.90 * avail' in c and 'BUDGET=6000' in c and '-lt 2000' in c and 'D8-63' in c and '--snap 6000' in c
    assert 'max(r1, r2)' in c and '--probe-every 400' in c
    assert 'pkill' not in c and 'killall' not in c and '--resume' not in c and 'export --' not in c and 'rm -rf' not in c
    assert '[ -e $OUT ]' in c and 'mkdir $OUT.lock' in c and 'grep -q "FILL_"' in c and 'du -sm' in c
    assert '--workers 10' in c and '--workers 1[1-9]' not in c and not re.search(r'--workers (1[1-9]|[2-9][0-9])', c)
    for k in ('E3L3_SHA', 'E3L2_SHA', 'E3L_SHA', 'E3R_SHA', 'G_SHA', 'E3R_SUM_SHA', 'TH1_SHA', 'TH2_SHA', 'EXPREC_SHA', 'CTRLH_SHA', 'CK1_SHA', 'CK2_SHA',
              'MAN_SHA', 'PYR_SHA', 'FIT_SHA'):
        assert f'"${k}"' in c, k
    assert 'E3L_SHA=' + L.file_sha256(L.__file__) in c and 'E3L2_SHA=' + L2.file_sha256(L2.__file__) in c
    assert 'E3R_SHA=' + R.file_sha256(R.__file__) in c and 'G_SHA=' + G.file_sha256(G.__file__) in c
    assert 'CTRLH_SHA=f0b55ed9' in c


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t(); print('ok', t.__name__, flush=True)
    print(f'{len(tests)}/{len(tests)} passed')
