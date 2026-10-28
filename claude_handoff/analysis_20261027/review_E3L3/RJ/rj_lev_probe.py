"""R-J probe 2 (report only, CPU, NOT a verdict): zero-learning LOSSLESS re-codings of E3-L3's GaBP messages.
For every directed message (s -> r) of the frozen E3-L3 checkpoint on VAL cam_nf exports:
  O7      = L.wire_bytes(h, Lam)                              (verdict coder; h coded independently of Lam)
  LEV     = Q.lev_bytes(N) of the per-level counts N = (n_-1, n_0, n_+1) relayed over the sender side (G4-E5's level-count coder,
            frozen, decodable; (h, Lam) = (n_+ - n_-, sum N) asserted == the integer GaBP message). H(h, Lam) <= H(N), so this is an
            achievable upper bound for a joint (h, Lam) coder of the same context class.
  LEV_SI  = LEV for the FIRST message on each tree edge (sender side smaller, earlier in Fl['order']); the SECOND message is coded
            with the first one as side information known to both endpoints (all three passes' contexts x side class of the voxel in
            the first message: 0 unseen / 1 dominant level +1 / 2 other) -- an extended copy of Q.lev_counts written here.
  GLOBAL  = LEV of the component's total level-count field (what every agent ends up knowing), and (n-1) x GLOBAL as a scale.
Writes only into this directory."""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
import json, time
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'claude_v342'))
import torch
import g4_e3l3 as E
import g4_quant_wire as Q
L = E.L; R = E.R; G = E.G
torch.set_num_threads(4)
INP = os.path.join(ROOT, 'claude_handoff', 'analysis_20261027', 'review_E3R', 'inputs')


def lev_counts_side(N, side, ns):
    """Q.lev_counts with every context multiplied by a side class side[v] in [0, ns) (decoder knows it: first message on the edge)"""
    N = np.asarray(N, np.int64); hd = Q.lev_header(N); q = hd['q']; k = N.sum(0); K = k > 0
    s1, s2, s3 = Q.lev_ctx_sizes(q); o2, o3 = s1 * ns, (s1 + s2) * ns
    n0 = np.zeros((s1 + s2 + s3) * ns, np.int64); n1 = np.zeros_like(n0)

    def add(ctx, bit):
        n1[:] += np.bincount(ctx[bit], minlength=len(n1)); n0[:] += np.bincount(ctx[~bit], minlength=len(n0))
    Ki = K.astype(np.int64); c1 = np.zeros(K.shape, np.int64)
    for i, d in enumerate(G.NB):
        c1 |= Q._shift(Ki, d, 0) << i
    add((c1 * ns + side).ravel(), K.ravel())
    kz = np.minimum(Q._shift(k, (0, 0, 1), 0), Q.K_NC - 1)
    kyx = np.minimum(np.maximum(Q._shift(k, (0, 1, 0), 0), Q._shift(k, (1, 0, 0), 0)), Q.K_NC - 1)
    m = k - 1
    for i in range(max(hd['kmax'] - 1, 0)):
        sel = K & (m >= i)
        if not sel.any():
            break
        add(o2 + ((min(i, Q.K_IC - 1) * Q.K_NC + kz[sel]) * Q.K_NC + kyx[sel]) * ns + side[sel], m[sel] > i)
    cls = np.argmax(N, 0).astype(np.int64); cls[~K] = q
    Q1 = q + 1
    nbc = (Q._shift(cls, (0, 0, 1), q) * Q1 + Q._shift(cls, (0, 1, 0), q)) * Q1 + Q._shift(cls, (1, 0, 0), q)
    r = k.copy()
    for j, l in enumerate(hd['coded']):
        nl = N[l]; rc = np.minimum(r, Q.R_C) - 1
        for i in range(int(r.max()) if K.any() else 0):
            sel = K & (nl >= i) & (r > i)
            if not sel.any():
                break
            add(o3 + (((j * Q.R_C + rc[sel]) * Q.I_C + min(i, Q.I_C - 1)) * Q1 ** 3 + nbc[sel]) * ns + side[sel], nl[sel] > i)
        r = r - nl
    assert np.array_equal(r, N[hd['drop']])
    return G.kt_bin_bits(n0, n1) / 8 + Q.LEV_HEADER_B


def side_of(Fl, s, r):
    nb = Fl['nb']; seen = {s}; st = [s]
    while st:
        u = st.pop()
        for v in nb[u]:
            if v != r and v not in seen and not (u == s and v == r):
                seen.add(v); st.append(v)
    seen.discard(r); return sorted(seen)


def house(path, model, w, table_t):
    z = np.load(path); private = z['private']; n = private.shape[0]
    F = G.forest(R.edges_list(z['edges']), n)
    Lp = list(z['L_pub_cam_nf']) if 'L_pub_cam_nf' in z.files else list(z['L_k0_cam_nf'])
    out = dict(n=n, O7=0.0, LEV=0.0, LEV_SI=0.0, first_O7=0.0, second_O7=0.0, first_LEV=0.0, second_LEV=0.0, second_LEV_SI=0.0,
               GLOBAL=0.0, GLOBALxN1=0.0, msgs=0, self_O7=0.0, self_LEV=0.0, kmax=0)
    with torch.no_grad():
        zp, lp = L.encode_maps(model, Lp, table_t, 'cpu')
        hp, lpq, zq = E.source(zp, lp, 'round')
    zi = zq[:, 0].numpy().astype(np.int64) if zq.dim() == 5 else zq.numpy().astype(np.int64)
    zi = zi.reshape((n,) + zi.shape[-3:]); fp = lpq.numpy() > 0
    own = np.stack([np.stack([(zi[a] == -1) & fp[a], (zi[a] == 0) & fp[a], (zi[a] == 1) & fp[a]]).astype(np.int64) for a in range(n)])
    for a in range(n):
        hi, li = L.msg_ints(hp[a], lpq[a]); out['self_O7'] += L.wire_bytes(hi, li); out['self_LEV'] += Q.lev_bytes(own[a])
    for cm in F['comps']:
        if len(cm) < 2:
            continue
        Fl = L.local_forest(F, cm)
        with torch.no_grad():
            hb, lb, msgs = L.gabp(hp[cm], lpq[cm], Fl, w, 'round')
        Ntot = own[cm].sum(0); g = Q.lev_bytes(Ntot); out['GLOBAL'] += g; out['GLOBALxN1'] += g * (len(cm) - 1)
        seen = {}
        for (s, r) in Fl['order']:
            hi, li = L.msg_ints(*msgs[(s, r)])
            Nm = own[[cm[a] for a in side_of(Fl, s, r)]].sum(0)
            assert np.array_equal(Nm[2] - Nm[0], hi[0]) and np.array_equal(Nm.sum(0), li), 'level counts != integer GaBP message'
            out['kmax'] = max(out['kmax'], int(li.max()))
            o7 = L.wire_bytes(hi, li); lv = Q.lev_bytes(Nm)
            key = frozenset((s, r))
            if key not in seen:
                seen[key] = Nm; lsi = lv; out['first_O7'] += o7; out['first_LEV'] += lv
            else:
                A = seen[key]; kA = A.sum(0)
                side = np.where(kA > 0, np.where(np.argmax(A, 0) == 2, 1, 2), 0).astype(np.int64)
                lsi = lev_counts_side(Nm, side, 3); out['second_O7'] += o7; out['second_LEV'] += lv; out['second_LEV_SI'] += lsi
            out['O7'] += o7; out['LEV'] += lv; out['LEV_SI'] += lsi; out['msgs'] += 1
    return out


def main():
    seed = sys.argv[1]; noise = sys.argv[2]; lo, hi_ = int(sys.argv[3]), int(sys.argv[4])
    ck, model = E.load_model(os.path.join(INP, f'e3l3_s{seed}', 'final.pt'), 'cpu')
    w = float(model.w.detach()); table_t = torch.as_tensor(np.asarray(ck['recipe']['evidence']['table']), dtype=torch.int32)
    files = sorted(f for f in os.listdir(os.path.join(INP, 'export')) if f.startswith('val_') and f.endswith(f'_s{noise}.npz'))[lo:hi_]
    res = {}; t0 = time.time(); op = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'rj_lev_{seed}_s{noise}_{lo}_{hi_}.json')
    for f in files:
        res[f] = house(os.path.join(INP, 'export', f), model, w, table_t)
        print(f, json.dumps({k: round(v) if isinstance(v, float) else v for k, v in res[f].items()}), f'{time.time() - t0:.0f}s', flush=True)
        with open(op, 'w') as fh:
            json.dump(dict(seed=seed, noise=noise, w=w, houses=res), fh, indent=1)


if __name__ == '__main__':
    main()
