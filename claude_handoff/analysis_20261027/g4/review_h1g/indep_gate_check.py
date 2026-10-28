"""Independent review recomputation for G4-H1G (reviewer, 2026-10-28). Does NOT import g4_h1_gate.py.
Uses only frozen modules (g4_h1_e3l3 = HL, g4_e3l3_w = W, and through them g4_e3l3 / g4_e3l / g4_e3r / g4_noise_panels).
mode fit : TRAIN hold-out TRAIN_r1_i19 (draws cam_nf_s200-202), 2 ckpts x {learned w, w = 4}: per-(Lam_b, xbin) counts vs the smoke and the
           formal fit job records; gate_tables counts == sum of the 19 formal fit job records; strong cells re-derived with exact fractions;
           in-house gating (changed voxels, truth split, pooled furniture / surface) vs what the record counts imply.
mode h1  : one H1 TRAIN smoke file (floating_train_r2_i9 seed 203; TRAIN only): changed voxels / truth split / per-cell counts and the
           gated R8 rows vs the smoke gated record.
Never reads VAL H1 files or noise seeds 3, 4. Writes only to claude_handoff/analysis_20261027/g4/review_h1g/.
"""
import os, sys
sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import json, glob, time, hashlib
from fractions import Fraction
import numpy as np

ROOT = r'C:\Users\27118\Desktop\sheaf-gbp\experiments\sheaf-gbp-p6-local-skill'
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'claude_v342'))
import g4_h1_e3l3 as HL  # frozen
import g4_e3l3_w as W    # frozen
assert 'g4_h1_gate' not in sys.modules
E3, L, R, G = HL.E3, HL.L, HL.R, HL.G
torch = HL.torch
UNK = R.UNK
G4 = os.path.join(ROOT, 'claude_handoff', 'analysis_20261027', 'g4')
OUT = os.path.join(G4, 'review_h1g')
TABLES = os.path.join(G4, 'h1g', 'fit', 'gate_tables.json')
CRIT_W = os.path.join(G4, 'e7', 'criteria_w.json')
SEEDS = ('342101', '342202')
NXB, LM = 40, 64


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def jl(p):
    with open(p) as f:
        return json.load(f)


def bins(hb, lb):
    """my binning: exact rational floor((h/L + 1) / (1/20)) via Fraction-equivalent integer math done per unique (h, L) pair;
    plus the naive float32 bin from L.belief_mean for a boundary-sensitivity count"""
    h = hb.astype(np.int64); l = lb.astype(np.int64)
    assert np.array_equal(h, hb) and np.array_equal(l, lb)
    xb = np.full(l.shape, -1, np.int64)
    pos = l > 0
    pairs = np.unique(np.stack([h[pos], l[pos]], 1), axis=0) if pos.any() else np.zeros((0, 2), np.int64)
    lut = {}
    for hh, ll in pairs.tolist():
        v = (Fraction(hh, ll) + 1) / Fraction(1, 20)
        b = v.numerator // v.denominator
        lut[(hh, ll)] = min(b, NXB - 1)
    if pos.any():
        keys = h[pos] * 1000 + l[pos]
        m = {hh * 1000 + ll: b for (hh, ll), b in lut.items()}
        xb[pos] = np.vectorize(m.get)(keys)
    return xb


def strong_from_counts(counts):
    S = set()
    for lam, rows in counts.items():
        for xb, r in enumerate(rows):
            n, nf = r[0], r[1]
            if int(lam) > 0 and n >= 200 and Fraction(nf, n) >= Fraction(995, 1000):
                S.add((int(lam), xb))
    return S


def thresholds():
    crit = jl(CRIT_W); out = {}
    for ts in SEEDS:
        dn, ck, th = HL.E3L3_CKPTS[ts]
        out[ts] = dict(learned=jl(os.path.join(HL.INP, dn, 'threshold.json'))['R8']['j_occ'], w4=crit['thresholds'][ts]['w4']['R8']['j_occ'])
    return out


def mode_fit():
    T = jl(TABLES); res = dict(tables_sha=sha(TABLES))
    # (a) gate_tables counts == sum of the 19 formal fit job records; strong cells re-derived exactly
    jobs = sorted(glob.glob(os.path.join(G4, 'h1g', 'fit', 'jobs', '*.json')))
    res['n_fit_jobs'] = len(jobs)
    A = {}
    for p in jobs:
        r = jl(p)
        assert not r.get('failed')
        for wl in ('learned', 'w4'):
            acc = A.setdefault((r['seed'], wl), {})
            for lam, rows in r['counts'][wl].items():
                a = acc.setdefault(lam, np.zeros((NXB, 8), np.int64)); a += np.asarray(rows, np.int64)
    res['tables'] = {}
    for ts in SEEDS:
        assert sorted(T['houses']) == sorted({jl(p)['house'] for p in jobs if jl(p)['seed'] == ts})
        for wl in ('learned', 'w4'):
            t = T['tables'][ts][wl]
            mine = {k: v.tolist() for k, v in A[(ts, wl)].items() if v.any()}
            same_counts = mine == t['counts']
            S_mine = strong_from_counts(mine); S_tab = {tuple(x) for x in t['strong']}
            # q and n of each strong cell + the borderline non-strong cells (n >= 200, q in [0.99, 0.995))
            border = []
            for lam, rows in mine.items():
                for xb, r in enumerate(rows):
                    if r[0] >= 200 and 0.99 <= r[1] / r[0] < 0.995:
                        border.append([int(lam), xb, r[0], r[1] / r[0]])
            res['tables'][f'{ts}|{wl}'] = dict(counts_equal_sum_of_jobs=same_counts, strong_equal=S_mine == S_tab, n_strong=len(S_tab),
                                                strong=[[l, x, mine[str(l)][x][0], mine[str(l)][x][1] / mine[str(l)][x][0]] for l, x in sorted(S_tab)],
                                                borderline_q_0p99_0p995=border, w=t['w'], j_occ=t['j_occ'],
                                                n_cells_lt200=sum(1 for rows in mine.values() for r in rows if 0 < r[0] < 200))
    # (b) one TRAIN hold-out house recomputed from scratch
    house = 'TRAIN_r1_i19'
    recs = W.ho_records(W.TRAIN_HO, [house])
    Hd = W.HOHouse(W.TRAIN_HO, recs[house])
    assert Hd.draws[1:] == ['cam_nf_s200', 'cam_nf_s201', 'cam_nf_s202']
    Hh = G.House(Hd.occ.astype(np.int8), Hd.slab, Hd.obj, Hd.cell)
    Ts = [G.Target(Hd.private[cm].max(0), Hh) for cm in Hd.comps]
    M = HL.load_models(1); thr = thresholds()
    res['house'] = house; res['per'] = {}
    for ts in SEEDS:
        m0 = M['e3'][ts]
        smoke_rec = jl(os.path.join(G4, 'h1g', 'smoke', 'fit', 'jobs', f'{ts}_{house}.json'))
        formal_rec = jl(os.path.join(G4, 'h1g', 'fit', 'jobs', f'{ts}_{house}.json'))
        for wl in ('learned', 'w4'):
            w = m0['w'] if wl == 'learned' else 4.0; jo = int(thr[ts][wl])
            S_tab = {tuple(x) for x in T['tables'][ts][wl]['strong']}
            cnt = {}; tot = np.zeros(5, np.int64); float_bin_diff = 0; rows_ = 0
            ch_tot = np.zeros(4, np.int64)          # changed, target free, target occ, target unknown
            fu = np.zeros(3, np.int64); fg = np.zeros(3, np.int64); su = []; sg = []
            with torch.no_grad():
                for d in range(1, 4):
                    maps = Hd.maps[d]
                    zh, lam = L.encode_maps(m0['model'], list(maps), m0['tb'], 'cpu')
                    hp, lq, _ = E3.source(zh, lam, 'round')
                    for ci, cm in enumerate(Hd.comps):
                        Fl = L.local_forest(Hd.F, cm)
                        hb, lb, _ = L.gabp(hp[cm], lq[cm], Fl, w, 'round')
                        xm = L.belief_mean(hb, lb).numpy()
                        lg = L.decode_np(m0['model'], hb, lb, HL.DEC_CHUNK)
                        kc = sum((maps[a] != UNK).astype(np.int32) for a in cm); Tt = Ts[ci]
                        for i in range(len(cm)):
                            rows_ += 1
                            lbi = lb[i].numpy(); hbi = hb[i, 0].numpy()
                            pk = lbi > 0
                            assert np.array_equal(pk, kc > 0)
                            xb = bins(hbi, lbi)
                            xf = np.floor((xm[i, 0][pk].astype(np.float32) + np.float32(1)) / np.float32(0.05)).astype(np.int64).clip(None, NXB - 1)
                            float_bin_diff += int((xf != xb[pk]).sum())
                            p = R.net_readout(lg[i], kc, jo)
                            po = (p == 1) & pk
                            masks = (pk & Tt.known, pk & Tt.free, pk & ~Tt.known, po & Tt.known, po & Tt.free,
                                     po & Tt.known & Hh.region & Tt.tr, po & Tt.known & Hh.region & ~Tt.tr, po & ~Tt.known)
                            li = lbi.astype(np.int64)
                            for j, mk in enumerate(masks):
                                if mk.any():
                                    u, c = np.unique(np.stack([li[mk], xb[mk]], 1), axis=0, return_counts=True)
                                    for (a, b), cc in zip(u.tolist(), c.tolist()):
                                        cnt.setdefault(str(a), np.zeros((NXB, 8), np.int64))[b, j] += cc
                            pr = po & Tt.known & Hh.region
                            tot += [int((pr & Tt.tr).sum()), int((pr & ~Tt.tr).sum()), int((Tt.tr & ~pr).sum()), int(((p == 0) & Tt.free).sum()), int(Tt.free.sum())]
                            # my gate
                            sm = np.zeros(p.shape, bool)
                            for (a, b) in S_tab:
                                sm |= (li == a) & (xb == b)
                            ch = (p == 1) & sm
                            q = p.copy(); q[ch] = 0
                            assert np.array_equal(q >= 0, p >= 0)
                            ch_tot += [int(ch.sum()), int((ch & Tt.free).sum()), int((ch & Tt.occ).sum()), int((ch & ~Tt.known).sum())]
                            for arr, pp, sl in ((fu, p, su), (fg, q, sg)):
                                prr = (pp == 1) & Tt.known & Hh.region
                                arr += [int((prr & Tt.tr).sum()), int((prr & ~Tt.tr).sum()), int((Tt.tr & ~prr).sum())]
                                sl.append(R.scored(pp, Tt, Hh, kc)['surface_f1'])
            mine = {k: v.tolist() for k, v in cnt.items()}
            exp_changed = sum(formal_rec['counts'][wl][str(l)][x][3] + formal_rec['counts'][wl][str(l)][x][7] for l, x in S_tab if str(l) in formal_rec['counts'][wl])
            exp_free = sum(formal_rec['counts'][wl][str(l)][x][4] for l, x in S_tab if str(l) in formal_rec['counts'][wl])
            exp_unk = sum(formal_rec['counts'][wl][str(l)][x][7] for l, x in S_tab if str(l) in formal_rec['counts'][wl])
            f1 = lambda a: 2 * a[0] / max(2 * a[0] + a[1] + a[2], 1)
            res['per'][f'{ts}|{wl}'] = dict(
                w=float(w), j_occ=jo, agent_rows=rows_, counts_equal_smoke=(mine == smoke_rec['counts'][wl]), counts_equal_formal=(mine == formal_rec['counts'][wl]),
                totals_mine=tot.tolist(), totals_equal_smoke=(tot.tolist() == smoke_rec['totals'][wl]), totals_equal_formal=(tot.tolist() == formal_rec['totals'][wl]),
                float32_bin_disagreements=float_bin_diff,
                gate_changed=int(ch_tot[0]), gate_changed_target_free=int(ch_tot[1]), gate_changed_target_occ=int(ch_tot[2]), gate_changed_target_unknown=int(ch_tot[3]),
                expected_from_record=dict(changed=int(exp_changed), target_free=int(exp_free), target_unknown=int(exp_unk)),
                gate_matches_record=bool(ch_tot[0] == exp_changed and ch_tot[1] == exp_free and ch_tot[3] == exp_unk),
                pooled_furn_f1=dict(ungated=f1(fu), gated=f1(fg)), mean_surface_f1=dict(ungated=float(np.mean(su)), gated=float(np.mean(sg)),
                                                                                    d_pp=float(100 * (np.mean(sg) - np.mean(su)))))
            print(ts, wl, json.dumps(res['per'][f'{ts}|{wl}']), flush=True)
    return res


def mode_h1():
    T = jl(TABLES)
    name = 'floating_train_r2_i9'; seed = '203'
    rec = {wl: jl(os.path.join(G4, 'h1g', 'smoke', 'h1', d, 'gated', 'jobs', name + '.json')) for wl, d in (('learned', 'w_learned'), ('w4', 'w4'))}
    recu = {wl: jl(os.path.join(G4, 'h1g', 'smoke', 'h1', d, 'ungated', 'jobs', name + '.json')) for wl, d in (('learned', 'w_learned'), ('w4', 'w4'))}
    s = rec['learned']['seeds'][seed]
    assert rec['learned']['split'] == 'TRAIN' and rec['learned']['panel'] == 'train'
    path = os.path.join(HL.EXPORT_DIR, s['file'])
    assert sha(path) == s['file_sha256']
    z = np.load(path)
    H = G.House(z['occ'], z['slab'], z['obj'], z['cell']); private = z['private']; n = private.shape[0]
    F = G.forest(R.edges_list(z['edges']), n)
    Ts = [G.Target(private[cm].max(0), H, mult=(private[cm] >= 0).sum(0)) for cm in F['comps']]
    Lm = list(z['L_k0_cam_nf'])
    M = HL.load_models(1); thr = thresholds(); out = dict(file=s['file'], per={})
    for ts in SEEDS:
        m0 = M['e3'][ts]
        with torch.no_grad():
            zh, lam = L.encode_maps(m0['model'], Lm, m0['tb'], 'cpu')
            hp, lq, _ = E3.source(zh, lam, 'round')
        for wl in ('learned', 'w4'):
            w = m0['w'] if wl == 'learned' else 4.0; jo = int(thr[ts][wl])
            S_tab = {tuple(x) for x in T['tables'][ts][wl]['strong']}
            changed = 0; truth = [0, 0, 0]; cells = {}; rows_g = []; rows_u = []
            with torch.no_grad():
                for ci, cm in enumerate(F['comps']):
                    Fl = L.local_forest(F, cm)
                    hb, lb, _ = L.gabp(hp[cm], lq[cm], Fl, w, 'round')
                    lg = L.decode_np(m0['model'], hb, lb, HL.DEC_CHUNK)
                    kc = sum((Lm[a] != UNK).astype(np.int32) for a in cm); Tt = Ts[ci]
                    for i in range(len(cm)):
                        lbi = lb[i].numpy(); hbi = hb[i, 0].numpy(); li = lbi.astype(np.int64)
                        xb = bins(hbi, lbi)
                        p = R.net_readout(lg[i], kc, jo)
                        sm = np.zeros(p.shape, bool)
                        for (a, b) in S_tab:
                            sm |= (li == a) & (xb == b)
                        ch = (p == 1) & sm
                        q = p.copy(); q[ch] = 0
                        changed += int(ch.sum())
                        for j, mk in enumerate((ch & Tt.free, ch & Tt.occ, ch & ~Tt.known)):
                            truth[j] += int(mk.sum())
                            if mk.any():
                                u, c = np.unique(np.stack([li[mk], xb[mk]], 1), axis=0, return_counts=True)
                                for (a, b), cc in zip(u.tolist(), c.tolist()):
                                    cells.setdefault(f'{a}_{b}', [0, 0, 0])[j] += cc
                        rows_g.append(R.scored(q, Tt, H, kc)); rows_u.append(R.scored(p, Tt, H, kc))
            g = rec[wl]['seeds'][seed]['gate'][f'E3L3_{ts}']
            rg = [r[0] for r in rec[wl]['seeds'][seed]['R8'][f'E3L3_{ts}']['rows']]
            ru = [r[0] for r in recu[wl]['seeds'][seed]['R8'][f'E3L3_{ts}']['rows']]
            jr = lambda x: json.loads(json.dumps(x))
            out['per'][f'{ts}|{wl}'] = dict(changed=changed, truth=truth, rec_changed=g['changed'], rec_truth=g['truth'],
                                            cells_equal=(cells == g['cells']), gated_rows_equal=(jr(rows_g) == jr(rg)), ungated_rows_equal=(jr(rows_u) == jr(ru)),
                                            n_rows=len(rows_g),
                                            mean_furn=dict(u=float(np.mean([r['furn_f1'] for r in rows_u])), g=float(np.mean([r['furn_f1'] for r in rows_g]))),
                                            mean_surf=dict(u=float(np.mean([r['surface_f1'] for r in rows_u])), g=float(np.mean([r['surface_f1'] for r in rows_g]))))
            print(ts, wl, json.dumps(out['per'][f'{ts}|{wl}']), flush=True)
    return out


if __name__ == '__main__':
    mode = sys.argv[1]
    t0 = time.time()
    HL.check_frozen(); W.check_frozen()
    r = mode_fit() if mode == 'fit' else mode_h1()
    r['seconds'] = time.time() - t0; r['utc'] = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    with open(os.path.join(OUT, f'indep_{mode}.json'), 'w') as f:
        json.dump(r, f, indent=1)
    print('done', mode, f'{time.time() - t0:.0f}s')
