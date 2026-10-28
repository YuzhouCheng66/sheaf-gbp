"""Reviewer's independent recomputation for G4-E7 (D8-67), stage 0 (wselect), one TRAIN hold-out house x one checkpoint at w = 4.
Does NOT import g4_e3l3_w.py. Uses frozen modules only for: the model + encoder (E.load_model, L.encode_maps), dec_input (frozen),
the O7 byte counter (L.wire_bytes), logit bins (R.logit_bin), targets (G.House / G.Target), edges_list.
Own code: tree / adjacency, two-pass GaBP with per-hop rounding (float32, same op order as the wire), an exact-rational re-rounding
cross-check, belief means, decoding call order (one agent at a time), histograms, known-mask check, float64 closed form + energies.
Never reads noise seeds 3, 4.  usage: python -B indep_w4_recompute.py HOUSE CKPT_SEED"""
import os, sys, json, time, hashlib
sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(v, '1')
import numpy as np
from fractions import Fraction

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
SCR = os.path.join(ROOT, 'scripts', 'claude_v342')
sys.path.insert(0, SCR)
import torch
import g4_e3l3 as E   # frozen (c7523cc8)
L = E.L; R = E.R; G = E.G
torch.set_num_threads(1)

AN = os.path.join(ROOT, 'claude_handoff', 'analysis_20261027')
HO = os.path.join(AN, 'g4', 'e5_export', 'train_holdout')
W = 4.0


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def adjacency(edges, n):
    nb = {i: [] for i in range(n)}
    for i, j, _ in edges:
        nb[i].append(j); nb[j].append(i)
    comps, seen = [], set()
    for s in range(n):
        if s in seen:
            continue
        c, st = [], [s]; seen.add(s)
        while st:
            u = st.pop(); c.append(u)
            for v in nb[u]:
                if v not in seen:
                    seen.add(v); st.append(v)
        comps.append(sorted(c))
    return nb, comps


def my_gabp_q(h, lam, nb, members, w):
    """h, lam: float32 numpy [n, X, Y, Z] (h = lam * z). Quantised two-pass GaBP by memoised recursion over directed edges of the tree
    restricted to `members`. Returns beliefs (hb, lb) for members, the integer messages, and an exact-rational re-rounding mismatch count."""
    mem = set(members); msgs = {}; exact_mis = [0, 0]
    wf = np.float32(w)

    def msg(s, r):
        if (s, r) in msgs:
            return msgs[(s, r)]
        lt = lam[s].copy(); ht = h[s].copy()
        for k in nb[s]:
            if k != r and k in mem:
                mh, ml = msg(k, s); lt = lt + ml; ht = ht + mh
        f = (wf / (wf + lt)).astype(np.float32)
        lq = np.round((lt * f).astype(np.float32)).astype(np.float32)
        hq = (np.round((ht * f).astype(np.float32)) * (lq > 0)).astype(np.float32)
        # exact rational cross-check (integers lt, ht; w integer): round-half-even of w*lt/(w+lt), w*ht/(w+lt)
        wi = int(w); lti = lt.astype(np.int64); hti = ht.astype(np.int64)
        assert np.array_equal(lti, lt) and np.array_equal(hti, ht)
        den = wi + lti
        def rhe(num, den):           # round half to even of num/den for den > 0 (vectorised, exact)
            q = np.floor_divide(num, den); r2 = 2 * (num - q * den)
            up = (r2 > den) | ((r2 == den) & (q % 2 == 1))
            return q + up
        lqe = rhe(wi * lti, den); hqe = rhe(wi * hti, den) * (lqe > 0)
        exact_mis[0] += int((lqe != lq.astype(np.int64)).sum()); exact_mis[1] += int((hqe != hq.astype(np.int64)).sum())
        msgs[(s, r)] = (hq, lq)
        return msgs[(s, r)]

    hb, lb = [], []
    for a in members:
        l = lam[a].copy(); hh = h[a].copy()
        for k in nb[a]:
            if k in mem:
                mh, ml = msg(k, a); l = l + ml; hh = hh + mh
        hb.append(hh); lb.append(l)
    for a in members:                         # every directed edge once (also leaves' incoming, already done)
        for k in nb[a]:
            if k in mem:
                msg(a, k)
    return np.stack(hb), np.stack(lb), msgs, exact_mis


def my_gabp_float64(z, lam, nb, members, w):
    mem = set(members); msgs = {}
    def msg(s, r):
        if (s, r) in msgs:
            return msgs[(s, r)]
        lt = lam[s].copy(); ht = lam[s] * z[s]
        for k in nb[s]:
            if k != r and k in mem:
                mh, ml = msg(k, s); lt = lt + ml; ht = ht + mh
        f = w / (w + lt); msgs[(s, r)] = (ht * f, lt * f)
        return msgs[(s, r)]
    xs = []
    for a in members:
        l = lam[a].copy(); hh = lam[a] * z[a]
        for k in nb[a]:
            if k in mem:
                mh, ml = msg(k, a); l = l + ml; hh = hh + mh
        xs.append(np.where(l > 0, hh / np.where(l > 0, l, 1.0), 0.0))
    return np.stack(xs)


def closed_form(z, lam, und, w):
    m = z.shape[0]; inf = lam.sum(0) > 0
    Lp = np.zeros((m, m))
    for i, j in und:
        Lp[i, i] += 1; Lp[j, j] += 1; Lp[i, j] -= 1; Lp[j, i] -= 1
    li = lam[:, inf].T; A = np.repeat((w * Lp)[None], li.shape[0], 0)
    A[:, np.arange(m), np.arange(m)] += li
    b = li * z[:, inf].T
    x = np.zeros_like(z); x[:, inf] = np.linalg.solve(A, b[:, :, None])[:, :, 0].T
    return x, inf


def energy(x, z, lam, und, w):
    e = 0.5 * float((lam * (x - z) ** 2).sum())
    for i, j in und:
        e += 0.5 * w * float(((x[i] - x[j]) ** 2).sum())
    return e


def main():
    house, sd = sys.argv[1], sys.argv[2]
    rooms, idx = house.split('_r')[1].split('_i')
    t0 = time.time()
    recs = [json.load(open(os.path.join(HO, 'records', f'trainho_r{rooms}_i{idx}_s{s}.json'))) for s in (200, 201, 202)]
    ck, model = E.load_model(os.path.join(AN, 'review_E3R', 'inputs', f'e3l3_s{sd}', 'final.pt'), 'cpu')
    tb = torch.as_tensor(np.asarray(ck['recipe']['evidence']['table']), dtype=torch.int32)
    ref = json.load(open(os.path.join(AN, 'g4', 'e7', 'wselect_jobs', f'{sd}_{house}.json')))
    crit = json.load(open(os.path.join(AN, 'g4', 'e7', 'criteria_w.json')))
    NB = R.NBIN
    acc = np.zeros((3, NB), np.int64); tot = np.zeros(4, np.int64); mism = 0; bytes_ = {}; lam_max = 0; exact = [0, 0]
    lam_hist = np.zeros(8, np.int64); cfmax = 0.0; en = []
    for r in recs:
        p = os.path.join(HO, r['file']); assert sha(p) == r['file_sha256'], p
        with np.load(p) as zf:
            occ = zf['occ']; slab = zf['slab']; obj = zf['obj']; cell = zf['cell']; private = zf['private']; edges = R.edges_list(zf['edges'])
            maps = np.ascontiguousarray(zf['L_k0_cam_nf'])
        n = private.shape[0]
        nb, comps = adjacency(edges, n)
        assert comps == [list(map(int, c)) for c in r['components']], (comps, r['components'])
        Hh = G.House((occ == 1).astype(np.int8), slab, obj, cell)
        with torch.no_grad():
            zhat, lam_t = L.encode_maps(model, list(maps), tb, 'cpu')            # frozen encoder (chunk 4 as E3-L3 select)
            z_t = torch.round(zhat)                                              # ternary source, own rounding
        lam = lam_t[:, :].numpy().astype(np.float32)                              # [n, X, Y, Z]
        z = z_t[:, 0].numpy().astype(np.float32) if z_t.dim() == 5 else z_t.numpy().astype(np.float32)
        h = (lam * z).astype(np.float32)
        by = 0.0
        for cm in comps:
            T = G.Target(private[cm].max(0), Hh)
            hb, lb, msgs, em = my_gabp_q(h, lam, nb, cm, W); exact[0] += em[0]; exact[1] += em[1]
            for (s, rr), (hq, lq) in msgs.items():
                li = lq.astype(np.int64); hi = hq.astype(np.int64)[None]
                by += L.wire_bytes(hi, li); lam_max = max(lam_max, int(li.max()))
                lam_hist += np.bincount(li[li > 0].ravel(), minlength=8)[:8]
            kc = (maps[cm] != R.UNK).sum(0)
            x = np.where(lb > 0, hb / np.where(lb > 0, lb, np.float32(1)), np.float32(0)).astype(np.float32)
            for i in range(len(cm)):
                with torch.no_grad():
                    xi = torch.from_numpy(x[i:i + 1]); lbi = torch.from_numpy(lb[i:i + 1])
                    lg = model.dec(E.dec_input(xi, lbi))[0].float().numpy()
                pk = lb[i] > 0; mism += int((pk != (kc > 0)).sum())
                ob = R.logit_bin(lg)
                acc[0] += np.bincount(ob[pk & T.known & Hh.region & T.tr], minlength=NB)
                acc[1] += np.bincount(ob[pk & T.known & Hh.region & ~T.tr], minlength=NB)
                acc[2] += np.bincount(ob[pk & T.free], minlength=NB)
                tot += [int(T.tr.sum()), int(T.free.sum()), int((pk & ~T.known).sum()), int((~T.known).sum())]
            # float64 closed form and energies at w = 4 (unquantised messages)
            z64 = z[cm].astype(np.float64); l64 = lam[cm].astype(np.float64)
            idx_ = {a: i for i, a in enumerate(cm)}
            und = sorted({(min(idx_[i], idx_[j]), max(idx_[i], idx_[j])) for i, j, _ in edges if i in idx_ and j in idx_})
            nbl = {idx_[a]: [idx_[b] for b in nb[a] if b in idx_] for a in cm}
            xg = my_gabp_float64(z64, l64, nbl, list(range(len(cm))), W)
            xc, inf = closed_form(z64, l64, und, W)
            cfmax = max(cfmax, float(np.abs(xg[:, inf] - xc[:, inf]).max()))
            en.append(dict(e_z=energy(z64, z64, l64, und, W), e_gabp=energy(xg, z64, l64, und, W), e_cf=energy(xc, z64, l64, und, W)))
        bytes_[f"cam_nf_s{r['seed']}"] = by
    j = crit['thresholds'][sd]['w4']['R8']['j_occ']
    tp = int(acc[0][j:].sum()); fp = int(acc[1][j:].sum()); fn = int(tot[0] - tp)
    out = dict(house=house, seed=sd, w=W, seconds=time.time() - t0,
               bytes=bytes_, ref_bytes=ref['bytes']['w4'], bytes_equal=all(bytes_[k] == ref['bytes']['w4'][k] for k in bytes_),
               bytes_max_abs_diff=max(abs(bytes_[k] - ref['bytes']['w4'][k]) for k in bytes_),
               acc_equal=bool(np.array_equal(acc, np.asarray(ref['acc']['w4']))), acc_absdiff_sum=int(np.abs(acc - np.asarray(ref['acc']['w4'])).sum()),
               tot=tot.tolist(), ref_tot=ref['tot']['w4'], tot_equal=tot.tolist() == ref['tot']['w4'],
               known_mismatch=mism, ref_known_mismatch=ref['known_mismatch']['w4'], lam_msg_max=lam_max, ref_lam_msg_max=ref['lam_msg_max']['w4'],
               lam_msg_hist_1_7=lam_hist[1:].tolist(), exact_rational_rounding_mismatch=dict(lam=exact[0], h=exact[1]),
               house_furn_f1_at_w4_threshold=dict(j=j, tp=tp, fp=fp, fn=fn, f1=2 * tp / max(2 * tp + fp + fn, 1)),
               closed_form_max_abs=cfmax, energies=en,
               energy_ok=all(e['e_gabp'] <= e['e_z'] + 1e-9 * max(1, abs(e['e_z'])) and abs(e['e_gabp'] - e['e_cf']) <= 1e-9 * max(1, abs(e['e_cf'])) for e in en),
               ctrl_bytes=ref['ctrl_bytes'], ratio_vs_net={k: bytes_[k] / ref['ctrl_bytes'][k] for k in bytes_})
    print(json.dumps(out, indent=1))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'indep_{sd}_{house}_w4.json'), 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
