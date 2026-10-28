"""R-J probe (review_E3L3, 2026-10-28, CPU only, report only, NOT a verdict): how much of E3-L3's O7 wire is redundancy
between the two channels of the SAME GaBP message (h = sum z, Lam = k) and between the two directions of the SAME tree edge?
Lossless re-codings only: every variant decodes the identical integer message (h, Lam), so beliefs / accuracy are bit-identical.
  O7  = L.wire_bytes(h, Lam)  (verdict coder of E3-L3: mask once + Lam | mask + h | mask, h coded independently of Lam)
  J1  = mask + Lam | mask + d | mask with d = Lam - h = n0 + 2 n_minus >= 0  (same frozen coder primitives, h replaced by d)
  K0  = own binarised-KT coder, independent channels (mask | 9 nb ; k-1 | 3 causal nb ; zigzag(h) | 3 causal nb)   (calibration)
  K1  = own coder, joint: mask ; k-1 ; d | (3 causal nb of d, min(k,3))
  K2  = K1 for the first message on each tree edge (sender side smaller = earlier in Fl['order']); the second message on the edge
        is coded with the first one (known to both endpoints) as side information: mask | (nb, A_mask), k-1 | (nb, A_k class),
        d | (nb, min(k,3), A_d class).  Decodable in the collect-distribute order of L.gabp.
Uses the frozen modules read only (g4_e3l3 / g4_e3l / g4_noise_panels), VAL cam_nf exports of review_E3R/inputs/export, the local
copies of the E3-L3 final checkpoints. Writes only into this directory."""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
import json, time
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'claude_v342'))
import torch
import g4_e3l3 as E
L = E.L; R = E.R; G = E.G
torch.set_num_threads(4)
INP = os.path.join(ROOT, 'claude_handoff', 'analysis_20261027', 'review_E3R', 'inputs')
C3 = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def shift(a, d, fill):
    out = np.full(a.shape, fill, a.dtype); X, Y, Z = a.shape; dx, dy, dz = d
    out[dx:, dy:, dz:] = a[:X - dx, :Y - dy, :Z - dz]; return out


def mask_bits(kn, side=None, ns=1):
    t = kn.astype(np.int64); c = np.zeros(kn.shape, np.int64)
    for d in G.NB:
        c = c * 2 + shift(t, d, 0)
    if side is not None:
        c = c * ns + side
    n = np.bincount(c.ravel(), minlength=512 * ns); n1 = np.bincount(c[kn], minlength=512 * ns)
    return G.kt_bin_bits(n - n1, n1) + 16


def unary_bits(s, kn, extra=None, nx=1, I=4):
    s = s.astype(np.int64); nbc = np.zeros(s.shape, np.int64)
    for d in C3:
        st = np.where(shift(kn, d, False), 1 + np.minimum(shift(s, d, 0), 2), 0)
        nbc = nbc * 4 + st
    base = nbc * nx + (extra if extra is not None else 0)
    sv = s[kn]; bv = base[kn]; M = int(sv.max(initial=0)); nctx = I * 64 * nx
    n0 = np.zeros(nctx, np.int64); n1 = np.zeros(nctx, np.int64)
    for i in range(M):
        sel = sv >= i; ci = min(i, I - 1) * 64 * nx + bv[sel]; b = sv[sel] > i
        n1 += np.bincount(ci[b], minlength=nctx); n0 += np.bincount(ci[~b], minlength=nctx)
    return G.kt_bin_bits(n0, n1) + 16


def zigzag(h):
    return np.where(h >= 0, 2 * h, -2 * h - 1)


def msg_fields(m):
    hi, li = L.msg_ints(*m); h = hi[0]; kn = li > 0; d = li - h
    assert (d[kn] >= 0).all() and (d[kn] <= 2 * li[kn]).all() and not d[~kn].any()
    return hi, li, h, kn, d


def first_bits(li, kn, d, h):
    k1 = np.where(kn, li - 1, 0)
    mk = mask_bits(kn); kb = unary_bits(k1, kn)
    K0 = mk + kb + unary_bits(np.where(kn, zigzag(h), 0), kn)
    K1 = mk + kb + unary_bits(d, kn, np.minimum(li, 3) - 1 + 0 * li, 3) if kn.any() else mk
    return K0 / 8, K1 / 8


def second_bits(li, kn, d, A):
    la, kna, da = A
    sideM = kna.astype(np.int64)
    sideK = np.where(kna, np.where(la >= 2, 2, 1), 0)
    sideD = np.where(kna, np.where(da > 0, 2, 1), 0)
    k1 = np.where(kn, li - 1, 0)
    b = mask_bits(kn, sideM, 2) + unary_bits(k1, kn, sideK, 3)
    b += unary_bits(d, kn, (np.clip(li, 1, 3) - 1) * 3 + sideD, 9)
    return b / 8


def house(path, model, w, table_t):
    z = np.load(path); private = z['private']; n = private.shape[0]
    F = G.forest(R.edges_list(z['edges']), n)
    Lp = list(z['L_pub_cam_nf']) if 'L_pub_cam_nf' in z.files else list(z['L_k0_cam_nf'])
    with torch.no_grad():
        zp, lp = L.encode_maps(model, Lp, table_t, 'cpu')
        hp, lpq, zq = E.source(zp, lp, 'round')
        out = dict(n=n, O7=0.0, O7_mask=0.0, O7_lam=0.0, O7_h=0.0, J1=0.0, K0=0.0, K1=0.0, K2=0.0, first_O7=0.0, second_O7=0.0,
                   first_K1=0.0, second_K1=0.0, second_K2=0.0, msgs=0, self_O7=0.0)
        for a in range(n):
            hi, li = L.msg_ints(hp[a], lpq[a]); out['self_O7'] += L.wire_bytes(hi, li)
        for cm in F['comps']:
            if len(cm) < 2:
                continue
            Fl = L.local_forest(F, cm)
            hb, lb, msgs = L.gabp(hp[cm], lpq[cm], Fl, w, 'round')
            seen = {}
            for (s, r) in Fl['order']:
                hi, li, h, kn, d = msg_fields(msgs[(s, r)])
                mb = L.mask_bytes(kn); vl = L.value_bytes_given_mask(li, kn); vh = L.value_bytes_given_mask(hi[0], kn)
                o7 = L.wire_bytes(hi, li); assert abs(o7 - (mb + vl + vh)) < 1e-6
                j1 = mb + vl + L.value_bytes_given_mask(d, kn)
                k0, k1 = first_bits(li, kn, d, h)
                key = frozenset((s, r)); first = key not in seen
                if first:
                    seen[key] = (li, kn, d); k2 = k1; out['first_O7'] += o7; out['first_K1'] += k1
                else:
                    k2 = second_bits(li, kn, d, seen[key]); out['second_O7'] += o7; out['second_K1'] += k1; out['second_K2'] += k2
                out['O7'] += o7; out['O7_mask'] += mb; out['O7_lam'] += vl; out['O7_h'] += vh
                out['J1'] += j1; out['K0'] += k0; out['K1'] += k1; out['K2'] += k2; out['msgs'] += 1
    return out


def main():
    seed = sys.argv[1]; noise = sys.argv[2] if len(sys.argv) > 2 else '1'; lim = int(sys.argv[3]) if len(sys.argv) > 3 else 99
    ck, model = E.load_model(os.path.join(INP, f'e3l3_s{seed}', 'final.pt'), 'cpu')
    w = float(model.w.detach()); table_t = torch.as_tensor(np.asarray(ck['recipe']['evidence']['table']), dtype=torch.int32)
    files = sorted(f for f in os.listdir(os.path.join(INP, 'export')) if f.startswith('val_') and f.endswith(f'_s{noise}.npz'))[:lim]
    res = {}; t0 = time.time()
    for f in files:
        res[f] = house(os.path.join(INP, 'export', f), model, w, table_t)
        print(f, json.dumps({k: round(v, 1) if isinstance(v, float) else v for k, v in res[f].items()}), f'{time.time() - t0:.0f}s', flush=True)
        with open(os.path.join(os.path.dirname(__file__), f'rj_probe_{seed}_s{noise}.json'), 'w') as fh:
            json.dump(dict(seed=seed, noise=noise, w=w, houses=res), fh, indent=1)


if __name__ == '__main__':
    main()
