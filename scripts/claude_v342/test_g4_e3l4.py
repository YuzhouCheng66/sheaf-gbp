"""unit tests for g4_e3l4.py (batch E3-L4, D8-68). Synthetic arrays and tiny synthetic caches, CPU only, a few minutes.
run:  CUDA_VISIBLE_DEVICES= python -B test_g4_e3l4.py
Optional: E3L3_CKPT=<an E3-L3 final.pt> (default: the local review_E3R/inputs copy or uk-linux e3l4-ref/e3l3_ckpt) for the
'learned w' forward test; without a checkpoint the test falls back to random weights at w = 1240.37 and says so."""
import os
import sys

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, math, re, shutil, subprocess, tempfile  # noqa: E401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import g4_e3l4 as M  # noqa: E402
import test_g4_e3l as TL  # noqa: E402   (helpers only: fake houses, random trees / maps, synthetic caches; read only)

L3 = M.L3
L2 = M.L2
L = M.L
R = M.R
G = M.G
HERE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(20261028)
torch.set_num_threads(2)
TABLE = torch.as_tensor(np.arange(R.LMIN, R.LMAX + 1), dtype=torch.int32)
E3L3_ORIG = M._ORIG['E3L3']
CKPT_CANDIDATES = [os.environ.get('E3L3_CKPT', ''),
                   os.path.join(HERE, '..', '..', 'claude_handoff', 'analysis_20261027', 'review_E3R', 'inputs', 'e3l3_s342101', 'final.pt'),
                   '/mnt/datasets/yuzhou-sheaf-gbp-research/e3l4-ref/e3l3_ckpt/s342101_final.pt']


def _nontrivial(m):
    with torch.no_grad():                                         # a non-trivial code (the UNet3 head starts at ~0, i.e. z = 0)
        m.enc.head.weight.normal_(0, 2.0); m.enc.head.bias.fill_(0.1)
    return m


def _ternary_case(m, g, rng):
    Fl = G.forest(TL.rand_tree(m, rng), m)
    lam = (rng.random((m,) + g) < 0.6).astype(np.float64); lam[:, 0, 0, 0] = 0.0
    z = rng.integers(-1, 2, (m,) + g).astype(np.float64) * lam
    return Fl, torch.from_numpy(lam), torch.from_numpy(z)


# ---------------------------------------------------------------- constants, w* grid, exact exp(log w*)
def test_constants_and_w_star():
    C = M.D8_68(4.0)
    assert C['prereg'] == 'D8-68' and C['noise_seeds'] == ['3', '4'] and C['budget_seconds'] == 3000 and C['updates_prereg'] == 9562 and C['w_star'] == 4.0
    for k in ('lever_furn_margin', 'lever_bytes_ratio', 'guard_free', 'guard_surface', 'clean_min', 'k0_margin', 'cam_demote_margin',
              'not_enough_furn_margin', 'not_enough_bytes_ratio', 'fresh_min_mean', 'min_updates', 'truncated_blocks_lever', 'guard_fpr', 'train_seeds',
              'judged_panel', 'report_panel', 'judged_tier', 'cam_tier', 'clean_tier'):
        assert C[k] == L3.D8_63[k], k                                         # the D8-63 numbers, unchanged
    assert C['lever_bytes_ratio'] == 0.15 and C['lever_furn_margin'] == -0.01 and C['not_enough_furn_margin'] == -0.03 and C['guard_fpr'] == 0.0
    assert M.UPDATES_PREREG == 9562 and M.BUDGET_S == 3000 and M.W_GRID == (3.0, 4.0, 6.0, 8.0) and M.ARM == 'E3L4'
    for w in M.W_GRID:
        assert M.check_w_star(w) == w
        assert float(torch.exp(torch.nn.Parameter(torch.tensor(math.log(w)))).detach()) == w, 'exp(float32(log w*)) must be exactly w* on the CPU'
    for bad in (1.0, 0.9, 5.0, 40.0, float('nan')):
        try:
            M.check_w_star(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError(f'w* = {bad} must be refused')
    assert M.check_w_star(1240.37, smoke=True) == 1240.37
    assert L3.E3L3 is E3L3_ORIG and L3.ARM == 'E3L3' and L3.W0 == 1000.0, 'importing g4_e3l4 must not change g4_e3l3'


def test_parser_never_swallows_forwarded_options():
    ap = M.build_parser()
    a, rest = ap.parse_known_args(['train', '--w-star', '4', '--prereg-updates', '9562', '--updates', '5', '--out', 'x', '--seed', '342101', '--max-updates', '3'])
    assert a.w_star == 4.0 and a.prereg_updates == 9562 and rest == ['--updates', '5', '--out', 'x', '--seed', '342101', '--max-updates', '3'], rest
    s = M._sub_args(rest, '--out', '--updates', '--max-updates', '--seed', '--smoke')
    assert s.updates == '5' and s.out == 'x' and s.max_updates == '3' and not s.smoke
    a, rest = ap.parse_known_args(['select', '--w-star', '6', '--run', 'r', '--ctrl', 'c', '--workers', '9'])
    assert a.w_star == 6.0 and rest == ['--run', 'r', '--ctrl', 'c', '--workers', '9']


# ---------------------------------------------------------------- Q1: omega frozen, zero gradient, initial weights = E3-L3's
def test_omega_frozen_and_gradient_zero():
    houses = [TL.fake_house(n=4, edges=((0, 1), (1, 2), (2, 3)), rng=np.random.default_rng(5 + i)) for i in range(2)]
    with M._as_e3l4(4.0):
        torch.manual_seed(342101)
        m = _nontrivial(L3.E3L3(4, L3.W0))                               # exactly what L3.train constructs, inside configure(w*)
        assert type(m) is M.E3L4 and not m.log_w.requires_grad and float(m.w) == 4.0
        lw0 = m.log_w.detach().clone()
        convs = [q for _, q in m.named_parameters() if q.ndim > 1]; rest = [q for _, q in m.named_parameters() if q.ndim <= 1]
        assert any(q is m.log_w for q in rest), 'log_w stays in the no-decay group (L3.train unchanged)'
        opt = torch.optim.AdamW([dict(params=convs, weight_decay=1e-4), dict(params=rest, weight_decay=0.0)], lr=1e-2)
        rng = np.random.default_rng(0)
        hd0 = m.enc.head.weight.detach().clone()
        for _ in range(3):
            specs = [L.sample_spec(rng, houses) for _ in range(3)]
            while not any(len(s[2]) > 1 for s in specs):
                specs = [L.sample_spec(rng, houses) for _ in range(3)]
            lg, yo, mk, src = L3.forward_batch(m, specs, houses, TABLE, 'ste', chunk=8)
            loss = R.masked_bce(lg, yo.float(), mk.float())
            opt.zero_grad(set_to_none=True); loss.backward()
            assert m.log_w.grad is None, 'omega must receive no gradient'
            assert sum(float(q.grad.abs().sum()) for q in m.enc.parameters() if q.grad is not None) > 0
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        assert torch.equal(m.log_w.detach(), lw0) and float(m.w) == 4.0, 'omega unchanged by AdamW'
        assert not torch.equal(m.enc.head.weight.detach(), hd0), 'the encoder does train'
    # the same seed gives E3-L3's initial encoder / decoder bit for bit (log_w consumes no random numbers)
    torch.manual_seed(342202); a = E3L3_ORIG(4, 1000.0)
    with M._as_e3l4(6.0):
        torch.manual_seed(342202); b = L3.E3L3(4, L3.W0)
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sa) == set(sb) and all(torch.equal(sa[k], sb[k]) for k in sa if k != 'log_w')
    assert float(torch.exp(sb['log_w'])) == 6.0 and abs(float(torch.exp(sa['log_w'])) - 1000.0) < 1e-3
    assert L3.E3L3 is E3L3_ORIG and L3.ARM == 'E3L3' and L3.W0 == 1000.0, '_as_e3l4 restores g4_e3l3'


def _restore():
    """the CLI paths call configure() for the rest of the process (as in a chain step); tests put g4_e3l3 back afterwards"""
    for k, v in M._ORIG.items():
        setattr(L3, k, v)
    M._W['w'] = None


# ---------------------------------------------------------------- GaBP at finite w*: closed form, energy, known mask, bounded messages
def test_gabp_closed_form_and_monotone_at_w_star():
    for w in M.W_GRID:
        for t in range(25):
            m = int(RNG.integers(1, 9))
            Fl, lam, z = _ternary_case(m, (4, 4, 3), RNG)
            c = L.gbp_checks(z.unsqueeze(1), lam, Fl, w)
            assert c['cf_max'] <= 1e-9 and c['violations'] == 0, (w, t, c)
            h = (lam * z).unsqueeze(1)
            hb, lb, msgs = L.gabp(h, lam, Fl, w, 'round')                   # the wire: per-hop rounding at step 1
            k = lam.sum(0)
            for a in range(m):
                assert torch.equal(lb[a] > 0, k > 0), 'known = Lam_b > 0 == k > 0 at finite w* (w > 1)'
            for (s, r), (hm, lm) in msgs.items():
                assert float(lm.max()) <= w and float(hm.abs().max()) <= w, 'a message precision is bounded by w (w Lam / (w + Lam) < w)'
    # finite w* is not the integer sum: on a 3-chain the far end receives an attenuated count
    Fl = G.forest([(1, 0, 1.0), (2, 1, 1.0)], 3)
    lam = torch.zeros(3, 1, 1, 1, dtype=torch.float64); lam[:] = 1.0; lam[2] = 6.0
    z = torch.ones_like(lam)
    for w in M.W_GRID:
        hb, lb, _ = L.gabp((lam * z).unsqueeze(1), lam, Fl, w, None)
        assert float(lb[0].max()) < float(lam.sum()) - 1e-6, 'finite w: the belief precision is below the full count'
        c = L.gbp_checks(z.unsqueeze(1), lam, Fl, w)
        assert c['cf_max'] <= 1e-12 and c['violations'] == 0


# ---------------------------------------------------------------- w = E3-L3's learned value -> the E3-L4 forward == g4_e3l3's
def _e3l3_ckpt():
    for p in CKPT_CANDIDATES:
        if p and os.path.exists(p):
            return p
    return None


def test_forward_equals_e3l3_at_learned_w():
    p = _e3l3_ckpt()
    if p:
        ck = torch.load(p, map_location='cpu', weights_only=False)
        sd = ck['state_dict']; width = ck['recipe']['width']
    else:
        print('  (no E3-L3 checkpoint found: random weights at w = 1240.37)', flush=True)
        torch.manual_seed(7); sd = _nontrivial(E3L3_ORIG(4, 1240.37)).state_dict(); width = 4
    w_learned = float(torch.exp(sd['log_w'].float()))
    a = E3L3_ORIG(width, 1000.0); a.load_state_dict(sd); a.eval()
    houses = [TL.fake_house(n=4, edges=((0, 1), (1, 2), (2, 3)), rng=np.random.default_rng(31 + i)) for i in range(2)]
    rng = np.random.default_rng(3)
    specs = [L.sample_spec(rng, houses) for _ in range(3)]
    with M._as_e3l4(w_learned):
        b = L3.E3L3(width, L3.W0); b.load_state_dict(sd); b.eval()
        assert type(b) is M.E3L4 and not b.log_w.requires_grad and float(b.w) == float(a.w.detach()) == w_learned
        with torch.no_grad():
            for q in ('round', 'ste'):
                la, ya, ma, _ = L3.forward_batch(a, specs, houses, TABLE, q, chunk=8)
                lb_, yb, mb, _ = L3.forward_batch(b, specs, houses, TABLE, q, chunk=8)
                assert torch.equal(la, lb_) and torch.equal(ma, mb) and torch.equal(ya, yb), q
            H = houses[0]                                                     # the eval path: encode, GaBP ('round'), decode, O7 bytes
            out = []
            for mdl in (a, b):
                zhat, lam = L.encode_maps(mdl, list(H.maps[1]), TABLE, 'cpu')
                h, lq, z = L3.source(zhat, lam, 'round')
                Fl = H.local(H.comps[0])
                hb, lbb, msgs = L.gabp(h[H.comps[0]], lq[H.comps[0]], Fl, mdl.w, 'round')
                by = sum(L.wire_bytes(*L.msg_ints(*msgs[e])) for e in Fl['directed'])
                out.append((L.decode_np(mdl, hb, lbb), by, z))
            assert np.array_equal(out[0][0], out[1][0]) and out[0][1] == out[1][1] and torch.equal(out[0][2], out[1][2])
        if p:
            ck2, mdl = L3.load_model(p, 'cpu')                                # g4_e3l3's loader inside the E3-L4 context
            assert type(mdl) is M.E3L4 and float(mdl.w) == w_learned
    assert L3.E3L3 is E3L3_ORIG


# ---------------------------------------------------------------- Q7: E1 rule per noise seed
def test_aggregate_e1_rule_for_new_seeds():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        os.makedirs(os.path.join(d, 'jobs'))
        none = dict(clean=None, cam_nf=None, cam=None); true = dict(clean=True, cam_nf=True, cam=True)
        for panel, i, s, e1 in (('val', 0, 3, none), ('val', 1, 4, none), ('val', 0, 1, none), ('val', 1, 1, true), ('val', 2, 3, dict(none, cam=False))):
            M._jdump(dict(stage='eval', panel=panel, rooms=1, index=i, seed=s, split='VALIDATION', e1_match=e1, script_sha256='x', ckpts={'a': 'b'}, tiers={}),
                     os.path.join(d, 'jobs', f'{panel}_r1_i{i}_s{s}.json'))
        S = M.aggregate(d)
        assert sorted(S['e1_not_applicable']) == ['val_r1_i0_s3', 'val_r1_i1_s4']
        assert sorted(S['e1_mismatch']) == ['val_r1_i0_s1', 'val_r1_i2_s3'], S['e1_mismatch']


# ---------------------------------------------------------------- verdict D8-68
def _summ(d=-.005, ratio=.12, d_fresh=-.004, d_cam=-.01, clean=.995, k0=-.01, missing=(), gbp_ok=True, fpr_delta=0.0, mism=0, free=.997, surf=.99,
          w=4.0, seeds=('3', '4'), ctrl_failed=(), ctrl_sha=None, ctrl_ck=('CK1', 'CK2')):
    arm = lambda t: f'E3L4_{t}'  # noqa: E731
    Sl = dict(panels={}, failed=[], skipped=[], e1_mismatch=[], agg_reasons=[],
              gbp={arm(t): dict(cf_max=1e-12 if gbp_ok else 1e-3, violations=0, components=9, **{'pass': gbp_ok}) for t in ('342101', '342202')},
              known_mismatch={arm(t): mism for t in ('342101', '342202')})
    Sc = dict(panels={}, failed=list(ctrl_failed), skipped=[], script_sha256=[ctrl_sha or R.file_sha256(R.__file__)],
              ckpts=[json.dumps({f'/e3r/{c}.pt': c for c in ctrl_ck}, sort_keys=True)])
    for panel, dd in (('val', d), ('fresh', d_fresh)):
        houses = [f'r{r}_i{i}' for r, i in G.panel_specs(panel)]
        Pl = dict(seeds={}); Pc = dict(seeds={})
        for ns in seeds:
            tl = dict(cam_nf={}, cam={}, clean={}); tc = dict(cam_nf={}, cam={}, clean={})
            for ts in ('342101', '342202'):
                c = TL._row(.97); c['rnet_relay_median'] = dict(sum=665000., count=88000.); c['R8']['furn_f1_pooled'] = .975
                tc['cam_nf'][f'NET_{ts}'] = c; tc['cam'][f'NET_{ts}'] = TL._row(.80); tc['clean'][f'NET_{ts}'] = TL._row(.997)
                if (panel, ts, ns) not in missing:
                    a = TL._row(.97 + dd, k0=.94 + k0, fpr=.018 + fpr_delta, free=free, surf=surf); a['bytes_median'] = ratio * 753000.
                    a['known_mismatch'] = mism; a['w'] = w; a['R8']['furn_f1_pooled'] = .975 + dd; a['belief_disagreement_rms'] = 0.1
                    tl['cam_nf'][arm(ts)] = a
                tl['cam'][arm(ts)] = TL._row(.80 + d_cam); tl['clean'][arm(ts)] = TL._row(clean)
            Pl['seeds'][ns] = dict(houses=houses, tiers=tl); Pc['seeds'][ns] = dict(houses=houses, tiers=tc)
        Sl['panels'][panel] = Pl; Sc['panels'][panel] = Pc
    return Sl, Sc


def _runs(d, done=(9562, 9562), planned=(9562, 9562), truncated=(False, False), thr=(True, True), w=(4.0, 4.0), meta=(True, True),
          second=(False, False), note=('', ''), wfinal=(4.0, 4.0), block=(True, True)):
    out = []
    for i, sd in enumerate(('342101', '342202')):
        r = os.path.join(d, f'run{sd}'); os.makedirs(r, exist_ok=True)
        M._jdump(dict(seed=int(sd), train_houses=186, recipe=dict(evidence=dict(desc=dict(spec='c:x:A:cam_nf'))),
                      pair=dict(ckpt=f'/e3r/{sd}.pt', ckpt_sha256=('CK1', 'CK2')[i], seed=int(sd))), os.path.join(r, 'run.json'))
        m = dict(prereg='D8-68', seed=int(sd), w_star=w[i], omega_frozen=True, updates_prereg=9562, updates=planned[i], second_difference=second[i],
                 second_difference_note=(note[i] or None))
        if meta[i]:
            M._jdump(m, os.path.join(r, 'e3l4.json'))
        fj = dict(complete=(done[i] == planned[i] and not truncated[i]), truncated=truncated[i], updates_done=done[i], updates_planned=planned[i])
        if block[i]:
            fj['e3l4'] = dict(m, w_final=wfinal[i], w_final_equals_w_star=(wfinal[i] == w[i]))
        M._jdump(fj, os.path.join(r, 'final.json'))
        if thr[i]:
            M._jdump(dict(n_houses=19, R8=dict(j_occ=2000), K0=dict(j_occ=2000)), os.path.join(r, 'threshold.json'))
        out.append(r)
    return out


def test_verdict_d8_68():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        Rn = _runs(d)

        def V(Sl, Sc, runs=Rn, full=False, w=4.0):
            p = os.path.join(d, 'ctrl.json'); M._jdump(Sc, p)
            v = M.verdict(Sl, runs, p, w_star=w)
            assert L3.D8_63 is M._ORIG['D8_63'] and L3.ARM == 'E3L3' and L3.E3L3 is E3L3_ORIG, 'verdict must restore g4_e3l3'
            return v if full else v['verdict']
        assert V(*_summ()) == 'LEVER'
        v = V(*_summ(), full=True)
        assert set(v['cells']) == {'342101|3', '342101|4', '342202|3', '342202|4'} and set(v['fresh_cells']) == set(v['cells'])
        c = v['cells']['342101|3']
        assert abs(c['bytes_ratio'] - .12) < 1e-12 and c['guards_pass'] and 'arm_R8_furn' in c and 'e3l3_R8_furn' not in c and 'arm_K0_sep' in c
        assert abs(c['d_furn_pooled'] + .005) < 1e-12 and c['belief_disagreement_rms'] == 0.1
        assert v['criteria']['prereg'] == 'D8-68' and v['w_star'] == 4.0 and not v['second_difference']['seeds']
        # the D8-63 branches on noise seeds 3 / 4
        assert V(*_summ(d=-.0105)) == 'AMBIGUOUS' and V(*_summ(d=-.031)) == 'NOT_ENOUGH' and V(*_summ(d=-.0099)) == 'LEVER'
        assert V(*_summ(ratio=.151)) == 'AMBIGUOUS' and V(*_summ(ratio=.501)) == 'NOT_ENOUGH' and V(*_summ(ratio=.15)) == 'LEVER'
        assert V(*_summ(d_cam=-.031)).startswith('AMBIGUOUS (lever demoted: cam') and V(*_summ(d_cam=-.029)) == 'LEVER'
        assert V(*_summ(d_fresh=-.0105)).startswith('AMBIGUOUS (lever demoted: fresh') and V(*_summ(d_fresh=+.01)) == 'LEVER'
        assert V(*_summ(d=-.031, d_fresh=-.05)) == 'NOT_ENOUGH' and V(*_summ(d=-.0105, d_fresh=.05)) == 'AMBIGUOUS'   # fresh never upgrades
        assert V(*_summ(d=-.031, d_cam=-.10)) == 'NOT_ENOUGH'                                                            # cam only demotes LEVER
        assert V(*_summ(clean=.989)) == 'AMBIGUOUS' and V(*_summ(k0=-.031)) == 'AMBIGUOUS' and V(*_summ(k0=-.029)) == 'LEVER'
        assert V(*_summ(free=.997 - .0051)) == 'AMBIGUOUS' and V(*_summ(surf=.99 - .0051)) == 'AMBIGUOUS' and V(*_summ(free=.997 - .0049)) == 'LEVER'
        # assertions -> INCOMPLETE (GaBP at finite w, known mask, FPR)
        assert 'GaBP assertion failed for E3L4_342101' in V(*_summ(gbp_ok=False))
        assert V(*_summ(mism=3)).startswith('INCOMPLETE') and V(*_summ(fpr_delta=1e-6)).startswith('INCOMPLETE')
        # truncation blocks LEVER only; < 2,000 / different counts / missing threshold -> INCOMPLETE
        assert V(*_summ(), runs=_runs(os.path.join(d, 'tr'), done=(9562, 9000), truncated=(False, True))) == 'AMBIGUOUS'
        assert V(*_summ(), runs=_runs(os.path.join(d, 'u'), done=(1999, 1999), planned=(1999, 1999), second=(True, True), note=('n', 'n'))).startswith('INCOMPLETE')
        assert V(*_summ(), runs=_runs(os.path.join(d, 'p'), planned=(9562, 9000), done=(9562, 9000), second=(False, True), note=('', 'n'))).startswith('INCOMPLETE')
        assert V(*_summ(), runs=_runs(os.path.join(d, 't'), thr=(True, False))).startswith('INCOMPLETE')
        # the second difference: written down -> judged normally (and listed); not written down -> INCOMPLETE
        r2 = _runs(os.path.join(d, 's2'), done=(9000, 9000), planned=(9000, 9000), second=(True, True), note=('fit 9000', 'fit 9000'))
        v2 = V(*_summ(), runs=r2, full=True)
        assert v2['verdict'] == 'LEVER' and v2['second_difference']['seeds'] == ['342101', '342202'] and v2['second_difference']['planned_updates'] == [9000]
        assert 'second difference' in V(*_summ(), runs=_runs(os.path.join(d, 's3'), done=(9000, 9000), planned=(9000, 9000)))
        # E3-L4 specific INCOMPLETE: w* bookkeeping, final w, cell w, control summary
        assert V(*_summ(), runs=_runs(os.path.join(d, 'm'), meta=(True, False))).startswith('INCOMPLETE')
        assert V(*_summ(), runs=_runs(os.path.join(d, 'w'), w=(4.0, 6.0))).startswith('INCOMPLETE')
        assert V(*_summ(), runs=_runs(os.path.join(d, 'wf'), wfinal=(4.0, 4.0000001))).startswith('INCOMPLETE')
        assert V(*_summ(), runs=_runs(os.path.join(d, 'b'), block=(False, True))).startswith('INCOMPLETE')
        assert V(*_summ(w=6.0)).startswith('INCOMPLETE') and V(*_summ(), w=6.0).startswith('INCOMPLETE')
        assert 'control: failed' in V(*_summ(ctrl_failed=('val_r1_i0_s3',)))
        assert 'g4_e3r sha' in V(*_summ(ctrl_sha='0' * 64)) and 'paired with training seed 342202' in V(*_summ(ctrl_ck=('CK1', 'CKX')))
        # judged on noise seeds 3 / 4 only: a summary of seeds 1 / 2 is not a verdict
        assert V(*_summ(seeds=('1', '2'))).startswith('INCOMPLETE')
        assert V(*_summ(missing={('val', '342101', '3')})).startswith('INCOMPLETE') and V(*_summ(missing={('fresh', '342202', '4')})).startswith('INCOMPLETE')
        Sl, Sc = _summ(); del Sc['panels']['val']['seeds']['3']['tiers']['cam_nf']['NET_342202']
        assert V(Sl, Sc).startswith('INCOMPLETE')
        Sl, Sc = _summ(); Sl['failed'] = ['val_r1_i0_s3']
        assert V(Sl, Sc).startswith('INCOMPLETE')
        assert M.verdict(_summ()[0], Rn, None, w_star=4.0)['verdict'].startswith('INCOMPLETE')
        assert M.verdict(_summ()[0], None, None, w_star=4.0)['verdict'].startswith('INCOMPLETE')


# ---------------------------------------------------------------- end to end on a synthetic cache (train -> select -> control -> eval -> aggregate)
def test_end_to_end_tiny():
    try:
        _end_to_end_tiny()
    finally:
        _restore()


def _end_to_end_tiny():
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
        e3r_ck = os.path.join(e3r, 's342101', 'checkpoints', 'final.pt')
        out = os.path.join(d, 's342101')
        common = ['train', '--w-star', '4', '--prereg-updates', '9562', '--cache', cache, '--evidence', ev, '--seed', '342101', '--out', out, '--updates', '4',
                  '--device', 'cpu', '--batch', '2', '--width', '4', '--log-every', '1', '--probe-every', '2', '--holdout-samples', '2', '--warmup', '1',
                  '--manifest-sha', man_sha, '--time-from', '2', '--snap', '3', '--ctrl-ckpt', e3r_ck, '--ctrl-holdout', ctrl]
        for bad, why in ((common, 'second difference'), ([c if c != '9562' else '9561' for c in common], 'D8-68')):
            try:
                M.main(bad)
            except (SystemExit, AssertionError) as e:
                assert why in str(e), str(e)
            else:
                raise AssertionError('must refuse: ' + why)
        assert not os.path.exists(os.path.join(out, 'run.json'))
        if os.path.exists(os.path.join(out, 'e3l4.json')):
            raise AssertionError('a refused train must not write e3l4.json')
        fin = M.main(common + ['--second-note', 'unit test: 4 updates'])
        assert fin['complete'] and fin['updates_done'] == 4 and fin['e3l4']['w_final'] == 4.0 and fin['e3l4']['second_difference']
        meta = json.load(open(os.path.join(out, 'e3l4.json')))
        assert meta['w_star'] == 4.0 and meta['omega_frozen'] and meta['updates_prereg'] == 9562 and meta['second_difference_note'] == 'unit test: 4 updates'
        ck = torch.load(os.path.join(out, 'checkpoints', 'final.pt'), map_location='cpu', weights_only=False)
        assert torch.equal(ck['state_dict']['log_w'], torch.tensor(math.log(4.0))) and ck['recipe']['w0'] == 4.0 and ck['w'] == 4.0
        hist = json.load(open(os.path.join(out, 'history.json')))
        assert all(h['probe']['w'] == 4.0 and h['probe']['known_mismatch'] == 0 for h in hist if 'probe' in h)
        try:
            M.main(common + ['--second-note', 'again'])
        except SystemExit as e:
            assert 'never overwrite' in str(e)
        else:
            raise AssertionError('train must never overwrite e3l4.json')
        th = M.main(['select', '--w-star', '4', '--run', out, '--ctrl', ctrl, '--workers', '1'])
        assert th['known_mismatch'] == 0 and 'k0_fallback_r8' in th['K0']
        try:
            M.main(['select', '--w-star', '6', '--run', out, '--ctrl', ctrl, '--workers', '1'])
        except AssertionError as e:
            assert 'w*' in str(e)
        else:
            raise AssertionError('select with a different w* must be refused')
        ex = os.path.join(d, 'export'); os.makedirs(os.path.join(ex, 'records'))
        H = TL.fake_house(noisy_draws=1, rng=np.random.default_rng(42))
        arr = dict(occ=H.occ.astype(np.int8), slab=H.slab, obj=H.obj, cell=H.cell, private=H.private, edges=R.edges_array(H.edges))
        for tn in R.EVAL_TIERS:
            arr[f'L_k0_{tn}'] = H.maps[0] if tn == 'clean' else H.maps[1]
        arr['L_pub_cam'] = H.maps[1][::-1].copy()
        fp = os.path.join(ex, 'train_r1_i0_s203.npz'); np.savez_compressed(fp, **arr)
        rec_e = dict(panel='train', rooms=1, index=0, seed=203, split='TRAIN', file='train_r1_i0_s203.npz', file_sha256=R.file_sha256(fp), components=H.comps,
                     e1_match=dict(clean=None, cam_nf=None, cam=None))
        M._jdump(rec_e, os.path.join(ex, 'records', 'train_r1_i0_s203.json'))
        ckf = os.path.join(out, 'checkpoints', 'final.pt')
        try:
            M.main(['eval', '--w-star', '4', '--export', ex, '--out', os.path.join(d, 'nope'), '--ckpt', ckf, '--expect-seeds', '203', '--workers', '1'])
        except AssertionError as e:
            assert 'VALIDATION' in str(e)
        else:
            raise AssertionError('a non-VALIDATION export must be refused outside --smoke')
        out_e = os.path.join(d, 'ev')
        M.main(['eval', '--w-star', '4', '--export', ex, '--out', out_e, '--ckpt', ckf, '--expect-seeds', '203', '--workers', '1', '--smoke'])
        job = json.load(open(os.path.join(out_e, 'jobs', 'train_r1_i0_s203.json')))
        assert not job.get('failed'), job.get('traceback')
        a = job['tiers']['cam_nf']['E3L4_342101']
        assert a['w'] == 4.0 and a['gbp']['violations'] == 0 and a['gbp']['cf_max'] <= L.CF_TOL and a['known_mismatch'] == 0 and 0 < a['bytes'] <= a['bytes_upper']
        assert job['script_sha256'] == R.file_sha256(M.__file__) and 'E3L3_342101' not in job['tiers']['cam_nf']
        try:
            M.main(['eval', '--w-star', '4', '--export', ex, '--out', out_e, '--ckpt', ckf, '--expect-seeds', '203', '--workers', '1', '--smoke'])
        except SystemExit as e:
            assert 'never overwrite' in str(e)
        else:
            raise AssertionError('eval must never overwrite job files')
        # the control: E3-R's own eval on the same maps + ctrlagg
        cdir = os.path.join(d, 'ctrl34')
        R.main(['eval', '--export', ex, '--out', cdir, '--ckpt', e3r_ck, '--workers', '1', '--smoke'])
        Sc = M.main(['ctrlagg', '--out', cdir, '--expect-seeds', '203', '--smoke'])
        assert 'NET_342101' in Sc['panels']['train']['seeds']['203']['tiers']['cam_nf'] and Sc['script_sha256'] == [R.file_sha256(R.__file__)]
        S = M.main(['aggregate', '--w-star', '4', '--out', out_e, '--runs', out, '--control-summary', os.path.join(cdir, 'summary.json'), '--smoke'])
        assert S['decision']['verdict'].startswith('INCOMPLETE') and S['gbp']['E3L4_342101']['pass'] and S['known_mismatch']['E3L4_342101'] == 0
        assert S['e1_not_applicable'] == [] and S['decision']['second_difference']['seeds'] == ['342101']
        assert not any('paired with training seed 342101' in r for r in S['decision']['e3l4_incomplete_reasons']), S['decision']['e3l4_incomplete_reasons']
        try:
            M.main(['aggregate', '--w-star', '4', '--out', out_e, '--runs', out, '--smoke'])
        except SystemExit as e:
            assert 'never overwrite' in str(e)
        else:
            raise AssertionError('aggregate must never overwrite summary.json')


# ---------------------------------------------------------------- the uk-linux chain (static)
def _chain():
    with open(os.path.join(HERE, 'chain_day8_E3L4.sh'), encoding='utf-8') as fh:
        return fh.read()


def test_chain_script_guards():
    c = _chain()
    lines = [ln for ln in c.splitlines() if not ln.lstrip().startswith('#')]          # code lines only (the header names the stages in prose)
    code = '\n'.join(lines)
    bash = shutil.which('bash')
    if bash:
        r = subprocess.run([bash, '-n', os.path.join(HERE, 'chain_day8_E3L4.sh')], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    assert 'cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1' in c and '[ "$PREREG" = "D8-68" ]' in c
    with open(os.path.join(HERE, 'chain_day7_E3L2.sh'), encoding='utf-8') as fh:   # PYTHONPATH / HouseExpo env identical to the E3-L2 chain
        ref = fh.read().splitlines()
    for key in ('export HOUSEEXPO_JSON_DIR=', 'export PYTHONPATH=', 'P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python', 'PYROOT=', 'CACHE='):
        mine = [ln.split('#')[0].rstrip() for ln in lines if ln.startswith(key)]; theirs = [ln.split('#')[0].rstrip() for ln in ref if ln.startswith(key)]
        assert len(mine) == 1 and mine == theirs, key
    assert 'export TMPDIR=/mnt/datasets/yuzhou-sheaf-gbp-research/tmp-claude' in c and 'OUT=runs/e3l4' in c and 'LOG=runs/chain_day8.log' in c
    assert 'conda ' not in code and 'pip ' not in code and 'pkill' not in code and 'killall' not in code and 'kill -9' not in code and 'rm -rf' not in code
    assert '--resume' not in code and 'rm ' not in code
    assert not re.search(r'(^|[;&|]|\$\(|\{)\s*kill\b', code, re.M), 'no kill command: only timeout -k on our own children'
    assert not re.search(r'g4_e3r\.py (export|gen|train)', c) and 'export --' not in c, 'no export / generation (never TEST)'
    # FILL_: only w*; every other pin is a real sha that matches the local files
    fills = re.findall(r'^(\w+)=FILL_\w*', c, re.M)
    assert fills == ['W_STAR'], fills
    assert 'grep -q -e "FILL_" -e "@E3L4" <<<"$W_STAR $E3L4_SHA"' in c and 'case "$W_STAR" in 3|4|6|8)' in c
    pins = dict(re.findall(r'^(\w+_SHA)=([0-9a-f]{64})\b', c, re.M))
    files = dict(E3L4_SHA='g4_e3l4.py', E3L3_SHA='g4_e3l3.py', E3L2_SHA='g4_e3l2.py', E3L_SHA='g4_e3l.py', E3R_SHA='g4_e3r.py', G_SHA='g4_noise_panels.py')
    for k, f in files.items():
        assert pins.get(k) == R.file_sha256(os.path.join(HERE, f)), f'{k} pin != sha of {f}'
    assert pins['E3L3_SHA'].startswith('c7523cc8') and pins['CTRLH_SHA'].startswith('f0b55ed9') and pins['E3R_SUM_SHA'].startswith('8a0bd5dd')
    for k in ('E3L4_SHA', 'E3L3_SHA', 'E3L2_SHA', 'E3L_SHA', 'E3R_SHA', 'G_SHA', 'D_SHA', 'V365_SHA', 'V371_SHA', 'IDX_SHA', 'PYR_SHA', 'MAN_SHA', 'FIT_SHA',
              'CTRLH_SHA', 'E3R_SUM_SHA', 'TH1_SHA', 'TH2_SHA', 'CK1_SHA', 'CK2_SHA', 'EXPREC_SHA', 'E5REC_SHA', 'E5RUN_SHA', 'E5_SUM_SHA', 'E3L3_SUM_SHA'):
        assert k in pins, k
        assert f'"${k}"' in c, f'{k} is pinned but never checked'
    i_chk = c.index('check_shas\nmkdir $OUT.lock')
    assert i_chk < c.index('[ -e $OUT ]') and 'trap \'rmdir $OUT.lock' in c
    # order: self-check -> SOLO wait -> preflight -> two sequential trainings -> select -> control -> eval -> aggregate -> report only
    idx = [code.index(s) for s in ('g4_e3l4.py selfcheck', 'while busy', '--max-updates 400', 'run_seed 342101', 'run_seed 342202', 'g4_e3l4.py select',
                                'g4_e3r.py eval', 'g4_e3l4.py ctrlagg', 'g4_e3l4.py eval --w-star $W_STAR --export $EXPORT34', 'g4_e3l4.py aggregate',
                                'g4_e3l4.py eval --w-star $W_STAR --export $EXPORT12', 'g4_e3l4.py report12')]
    assert idx == sorted(idx), idx
    # lease arithmetic and the budget rule
    assert 'DE=$((START+6600)); DT=$((START+7150))' in c and 'BUDGET=3000' in c and 'UPD_PREREG=9562' in c and '-lt 2000' in c
    assert 'min(fit, $UPD_PREREG)' in c and '0.90 * avail / r' in c and '--time-from 200' in c and '--probe-every 400' in c and '--snap 6000' in c
    assert 'SDT=$((NOW+BUDGET))' in c and '[ $SDT -gt $DT ] && SDT=$DT' in c and '[ $SDE -gt $DE ] && SDE=$DE' in c
    assert 'the GPU is not used again by this chain' in c and 'SECOND DIFFERENCE' in c
    gpu = [ln for ln in lines if 'g4_e3l4.py train --' in ln]
    assert len(gpu) == 2, 'one preflight + the training line inside run_seed'
    for ln in gpu:
        assert ln.lstrip().startswith('timeout -k 30') and "--expect-gpu 'RTX 5090'" in ln and '--w-star $W_STAR' in ln and '--prereg-updates $UPD_PREREG' in ln
        assert 'CUDA_VISIBLE_DEVICES' not in ln and not ln.rstrip().endswith('&'), 'sequential, on the only GPU'
    assert '--second-note "$SECOND_NOTE"' in gpu[1] and '--ctrl-ckpt $E3R/s$seed/checkpoints/final.pt' in gpu[1] and '--updates $UPD ' in gpu[1]
    # every CPU stage: GPUs blanked, under timeout, nice, <= 10 processes
    for ln in lines:
        if '$P -B scripts/' in ln and re.search(r'g4_e3l4\.py (select|eval|aggregate|selfcheck|ctrlagg|report12)|g4_e3r\.py eval', ln):
            assert 'CUDA_VISIBLE_DEVICES= ' in ln and 'timeout -k 30' in ln, ln
            if 'selfcheck' not in ln:
                assert 'nice -n 10' in ln, ln
            if '--workers' in ln:
                assert '--workers $WORKERS' in ln, ln
            if re.search(r'g4_e3l4\.py (select|eval|aggregate|report12|selfcheck)', ln):
                assert '--w-star $W_STAR' in ln, ln
    assert 'WORKERS=9' in c and not re.search(r'--workers (1[0-9]|[2-9][0-9])', c)
    assert '--expect-seeds 3,4' in c and '--expect-seeds 1,2' in c and '--control-summary $OUT/ctrl34/summary.json' in c
    assert '--control-summary $E3R/eval/summary.json' in c and '--e5-summary $REF4/e5_summary.json' in c
    # writes only under /mnt/datasets (absolute) or the run directory (relative to the claude-diag dir)
    for ln in lines:
        for m in re.finditer(r'(?<![-=])>\s*(\S+)', ln):                   # redirections ('->' arrows in log text are prose)
            tgt = m.group(1).rstrip('\'";)')
            if tgt.startswith('&') or tgt in ('/dev/null',) or tgt[0].isdigit() or tgt.startswith('='):
                continue                                                  # 2>&1, /dev/null, and '>' comparisons inside python snippets
            assert tgt.startswith(('$OUT', '$PF', 'runs/', '/mnt/datasets', '$LOG', "'$OUT")) or tgt == '>', (tgt, ln)
    assert re.search(r'^REF4=/mnt/datasets/yuzhou-sheaf-gbp-research/e3l4-ref\b', c, re.M) and 'EXPORT34=$REF4/e5_export' in c


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t(); print('ok', t.__name__, flush=True)
    print(f'{len(tests)}/{len(tests)} passed')
