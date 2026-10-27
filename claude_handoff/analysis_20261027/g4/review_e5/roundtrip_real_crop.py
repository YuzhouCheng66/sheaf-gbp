"""reviewer: the implementer's real arithmetic coder (lev_encode / lev_decode) on crops of a REAL Q3 level-count message
(val_r1_i0_s1, seed 1, half A, the largest-side directed edge); decoded == original and bits vs the ideal length."""
import os, sys, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rev_e5_recompute as R
sys.path.insert(0, os.path.join(R.ROOT, 'scripts', 'claude_v342'))
import g4_quant_wire as Q
TB = R.tables(); lab = TB['A']['Q3_lab']
z = np.load(os.path.join(R.EXP, 'val_r1_i0_s1.npz')); L = z['L_k0_cam_nf']; n = L.shape[0]
comps, sides = R.comps_sides(z['edges'], n)
e = max(sides, key=lambda x: len(sides[x]))
N = np.zeros((3,) + L.shape[1:], np.int64)
for a in sides[e]:
    kn = L[a] != R.UNK; li = lab[np.clip(L[a].astype(np.int64) - R.LMIN, 0, R.NL - 1)]
    for l in range(3):
        N[l] += kn & (li == l)
K = N.sum(0) > 0
xs, ys = np.nonzero(K.any(2))
cx, cy = int(np.median(xs)), int(np.median(ys))
for (w, h) in ((24, 24), (40, 40)):
    crop = np.ascontiguousarray(N[:, cx - w // 2: cx + w // 2, cy - h // 2: cy + h // 2, :])
    t0 = time.time()
    bits, hd = Q.lev_encode(crop); dec = Q.lev_decode(bits, crop.shape[1:], hd)
    ideal = Q.lev_bits(crop)
    print(f'edge {e} side {sides[e]} crop {crop.shape[1:]} known {int((crop.sum(0) > 0).sum())} kmax {hd["kmax"]} drop {hd["drop"]}: '
          f'round trip {np.array_equal(dec, crop)}, real bits {len(bits)}, ideal {ideal:.1f}, overhead {len(bits) - ideal:+.1f} bits '
          f'({100 * (len(bits) - ideal) / max(ideal, 1):+.3f} %), {time.time() - t0:.0f}s')
t0 = time.time()
bits, hd = Q.lev_encode(N); dec = Q.lev_decode(bits, N.shape[1:], hd); ideal = Q.lev_bits(N)
print(f'edge {e} FULL message {N.shape[1:]} known {int(K.sum())} kmax {hd["kmax"]}: round trip {np.array_equal(dec, N)}, real bytes {len(bits) / 8 + 4:.1f}, '
      f'lev_bytes {Q.lev_bytes(N):.1f}, overhead {len(bits) - ideal:+.1f} bits, {time.time() - t0:.0f}s')
