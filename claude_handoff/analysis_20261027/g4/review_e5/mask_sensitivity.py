"""reviewer: share of the known mask inside the Qinf control's k message (the mask is coded twice in S+k: S's ternary pass and k's)."""
import os, sys, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rev_e5_recompute as R
sys.path.insert(0, os.path.join(R.ROOT, 'scripts', 'claude_v342'))
import g4_e3r as E
G = E.G
TB = R.tables()
for f in ('val_r1_i0_s1.npz', 'val_r1_i1_s1.npz'):
    z = np.load(os.path.join(R.EXP, f)); L = z['L_k0_cam_nf']; n = L.shape[0]
    comps, sides = R.comps_sides(z['edges'], n)
    known = [L[a] != R.UNK for a in range(n)]
    tab = TB['A']['Qinf']
    e = [np.where(known[a], tab[np.clip(L[a].astype(np.int64) - R.LMIN, 0, R.NL - 1)], 0).astype(np.int32) for a in range(n)]
    mask = sk = 0.
    for (s, r), sd in sides.items():
        S = sum(e[a] for a in sd); kc = sum(known[a].astype(np.int32) for a in sd); K = kc > 0
        mask += G.ctx_tern_bits(np.where(K, np.int8(1), np.int8(-1))) / 8
        sk += G.int_ctx_bytes(S, K) + G.int_ctx_bytes(kc, K)
    print(f'{f}: Qinf Sk {sk:.0f} B, mask coded in the k message {mask:.0f} B ({100 * mask / sk:.1f} %); Sk without the second mask {sk - mask:.0f} B')
