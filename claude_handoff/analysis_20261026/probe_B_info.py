"""probe_B_info (analyst B, 2026-10-26, CPU only, report-only): how much information does the K0 target (an agent's own map) contain,
per house, per evidenced code coordinate, per height band and for furniture vs structure? Uses the 24-house VAL panel (probe_B_genval.py).
Two estimators of the information in a ternary map v in {-1, 0, 1}^(128x128x32):
  (1) lzma (preset 9 | extreme) of the bit-packed planes (z fastest): a generic UPPER bound;
  (2) a causal context model: -log2 p(v | 6 causal neighbours) with p pooled over all 192 own maps (adaptive-free plug-in, model cost
      ignored, so slightly OPTIMISTIC = a lower-ish estimate). Bits are attributed per voxel, so they can be split by height band,
      by furniture / structure / slab, and by code coordinate.
Also: surface voxels (the scored set) and their lzma size; evidenced coordinates at 16x16, 32x32, 64x64 lateral grids.
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 19 python probe_B_info.py --val val24.pkl --diag <3S-X diag.json> --out info.json
"""
import argparse, json, lzma, pickle, sys, time
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--val', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--diag', default='/mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021/runs/v430-stackdata/diag-control-s342101/diag.json')
args = ap.parse_args()
V = pickle.load(open(args.val, 'rb')); plan = V['plan']; fxs = V['fixtures']
t0 = time.time()


def lz_bits(a):
    return 8 * len(lzma.compress(np.packbits(a.astype(np.uint8).reshape(-1)).tobytes(), preset=9 | lzma.PRESET_EXTREME))


def surface(v):
    free = v == 0
    nb = np.zeros_like(free)
    for ax in range(3):
        for s in (1, -1):
            sh = np.roll(free, s, axis=ax)
            idx = [slice(None)] * 3; idx[ax] = slice(0, 1) if s == 1 else slice(-1, None); sh[tuple(idx)] = False
            nb |= sh
    return (v == 1) & nb


def context(v):
    """6 causal neighbours (x-1, y-1, z-1, x-1y-1, x-1z-1, y-1z-1), out of range = -1 (unknown) -> context id in [0, 3^6)."""
    p = np.pad(v, 1, constant_values=-1).astype(np.int16) + 1                      # 0, 1, 2
    X, Y, Z = v.shape
    c = lambda dx, dy, dz: p[1 + dx:1 + dx + X, 1 + dy:1 + dy + Y, 1 + dz:1 + dz + Z]
    nbs = [c(-1, 0, 0), c(0, -1, 0), c(0, 0, -1), c(-1, -1, 0), c(-1, 0, -1), c(0, -1, -1)]
    cid = np.zeros(v.shape, np.int32)
    for n in nbs:
        cid = cid * 3 + n
    return cid


maps = []
for hi, (sp, fx) in enumerate(zip(plan, fxs)):
    if fx is None:
        continue
    for a in range(fx['private'].shape[0]):
        maps.append((hi, a, fx['private'][a]))
# pooled context counts
NC = 3 ** 6
counts = np.zeros((NC, 3), np.float64)
ctxs = []
for hi, a, v in maps:
    cid = context(v); ctxs.append(cid)
    np.add.at(counts, (cid.reshape(-1), (v.reshape(-1) + 1)), 1)
prob = (counts + .5) / (counts + .5).sum(1, keepdims=True)
print('context model built %.0fs' % (time.time() - t0), flush=True)
bands = {'0.0-0.3m': (0, 3), '0.3-0.9m': (3, 9), '0.9-1.5m': (9, 15), '1.5-2.1m': (15, 21), '2.1-3.2m': (21, 32)}
rows = []
for (hi, a, v), cid in zip(maps, ctxs):
    fx = fxs[hi]; obj = fx['obj']; slab = fx['slab']; struct = fx['struct'].reshape(v.shape)
    bits = -np.log2(prob[cid, v + 1])                                                # per-voxel code length
    surf = surface(v); known = v >= 0
    r = dict(house=hi, agent=a, rooms=plan[hi]['rooms'], chart_m=float(fx['chart_m']), known=int(known.sum()), occ=int((v == 1).sum()), surf=int(surf.sum()),
             surf_obj=int((surf & obj).sum()), surf_struct=int((surf & struct).sum()), surf_slab=int((surf & slab).sum()),
             ctx_bits=float(bits.sum()), ctx_bits_obj_vox=float(bits[obj].sum()), ctx_bits_struct_vox=float(bits[struct].sum()), ctx_bits_slab_vox=float(bits[slab].sum()))
    for b, (z0, z1) in bands.items():
        r[f'ctx_bits_{b}'] = float(bits[:, :, z0:z1].sum()); r[f'surf_{b}'] = int(surf[:, :, z0:z1].sum()); r[f'surf_obj_{b}'] = int((surf & obj)[:, :, z0:z1].sum())
    for B in (16, 32, 64):
        k = 128 // B
        colknown = known.reshape(B, k, B, k, 32).any((1, 3, 4)); r[f'ev{B}'] = int(colknown.sum())
        colbits = bits.reshape(B, k, B, k, 32).sum((1, 3, 4)); r[f'ctx_bits_per_ev{B}_p50'] = float(np.median(colbits[colknown])); r[f'ctx_bits_per_ev{B}_p90'] = float(np.percentile(colbits[colknown], 90))
    if a < 2:                                                                       # lzma is slow: 2 agents per house
        r['lz_full_bits'] = lz_bits(np.concatenate([known.reshape(-1), (v == 1).reshape(-1)])); r['lz_known_bits'] = lz_bits(known); r['lz_surf_bits'] = lz_bits(surf)
        r['lz_surf_noobj_bits'] = lz_bits(surf & ~obj)
    rows.append(r)
print('rows %d %.0fs' % (len(rows), time.time() - t0), flush=True)
R = {k: np.array([r[k] for r in rows if k in r], float) for k in rows[0]}
lzr = [r for r in rows if 'lz_full_bits' in r]
summ = dict(n_maps=len(rows), ctx_bits_mean=float(R['ctx_bits'].mean()), ctx_bits_p10=float(np.percentile(R['ctx_bits'], 10)), ctx_bits_p90=float(np.percentile(R['ctx_bits'], 90)),
            lz_full_bits_mean=float(np.mean([r['lz_full_bits'] for r in lzr])), lz_known_bits_mean=float(np.mean([r['lz_known_bits'] for r in lzr])),
            lz_surf_bits_mean=float(np.mean([r['lz_surf_bits'] for r in lzr])), lz_surf_noobj_bits_mean=float(np.mean([r['lz_surf_noobj_bits'] for r in lzr])),
            surf_mean=float(R['surf'].mean()), surf_obj_share=float(R['surf_obj'].sum() / R['surf'].sum()), surf_struct_share=float(R['surf_struct'].sum() / R['surf'].sum()),
            surf_slab_share=float(R['surf_slab'].sum() / R['surf'].sum()),
            ctx_bits_share_obj_vox=float(R['ctx_bits_obj_vox'].sum() / R['ctx_bits'].sum()), ctx_bits_share_struct_vox=float(R['ctx_bits_struct_vox'].sum() / R['ctx_bits'].sum()),
            ctx_bits_share_slab_vox=float(R['ctx_bits_slab_vox'].sum() / R['ctx_bits'].sum()))
for B in (16, 32, 64):
    summ[f'ev{B}_mean'] = float(R[f'ev{B}'].mean()); summ[f'ctx_bits_per_ev{B}_mean'] = float((R['ctx_bits'] / R[f'ev{B}']).mean())
    summ[f'ctx_bits_per_ev{B}_p50_median'] = float(np.median(R[f'ctx_bits_per_ev{B}_p50'])); summ[f'surf_per_ev{B}'] = float((R['surf'] / R[f'ev{B}']).mean())
for b in bands:
    summ[f'ctx_bits_share_{b}'] = float(R[f'ctx_bits_{b}'].sum() / R['ctx_bits'].sum()); summ[f'surf_share_{b}'] = float(R[f'surf_{b}'].sum() / R['surf'].sum())
    summ[f'surf_obj_frac_in_{b}'] = float(R[f'surf_obj_{b}'].sum() / max(R[f'surf_{b}'].sum(), 1))
# per house vs L and vs the 3S-X per-house K0
H = {}
for r in rows:
    H.setdefault(r['house'], []).append(r)
hs = sorted(H)
L = np.array([H[h][0]['chart_m'] for h in hs]); hb = np.array([np.mean([r['ctx_bits'] for r in H[h]]) for h in hs]); hs_ = np.array([np.mean([r['surf'] for r in H[h]]) for h in hs])
hobj = np.array([np.sum([r['surf_obj'] for r in H[h]]) / np.sum([r['surf'] for r in H[h]]) for h in hs]); hev = np.array([np.mean([r['ev32'] for r in H[h]]) for h in hs])
summ['corr_L_ctxbits'] = float(np.corrcoef(L, hb)[0, 1]); summ['corr_L_surf'] = float(np.corrcoef(L, hs_)[0, 1]); summ['corr_L_bits_per_ev32'] = float(np.corrcoef(L, hb / hev)[0, 1])
summ['ctxbits_small_le13'] = float(hb[L <= 13].mean()); summ['ctxbits_big_ge17'] = float(hb[L >= 17].mean())
summ['bits_per_ev32_small_le13'] = float((hb / hev)[L <= 13].mean()); summ['bits_per_ev32_big_ge17'] = float((hb / hev)[L >= 17].mean())
summ['surf_small_le13'] = float(hs_[L <= 13].mean()); summ['surf_big_ge17'] = float(hs_[L >= 17].mean())
try:
    d = json.load(open(args.diag)); K = {(r['rooms'], r['index']): r for r in d['per_scene']}
    k0 = np.array([K[(plan[h]['rooms'], plan[h]['index'])]['K0_surface'] for h in hs])
    def pcorr(x, y, z):
        A = np.c_[np.ones_like(z), z]; rx = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]; ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]; return float(np.corrcoef(rx, ry)[0, 1])
    summ.update(corr_K0_ctxbits=float(np.corrcoef(hb, k0)[0, 1]), corr_K0_bits_per_ev32=float(np.corrcoef(hb / hev, k0)[0, 1]), corr_K0_objshare=float(np.corrcoef(hobj, k0)[0, 1]),
                corr_K0_L=float(np.corrcoef(L, k0)[0, 1]), pcorr_K0_L_given_bits_per_ev32=pcorr(L, k0, hb / hev), pcorr_K0_bits_per_ev32_given_L=pcorr(hb / hev, k0, L),
                pcorr_K0_objshare_given_L=pcorr(hobj, k0, L))
except Exception as e:                                                               # noqa
    summ['diag_error'] = repr(e)
summ['per_house'] = [dict(house=int(h), rooms=plan[h]['rooms'], index=plan[h]['index'], L=float(l), ctx_bits=float(b), bits_per_ev32=float(b / e), surf=float(s), obj_share=float(o))
                     for h, l, b, e, s, o in zip(hs, L, hb, hev, hs_, hobj)]
summ['seconds'] = time.time() - t0
json.dump(dict(summary=summ, rows=rows), open(args.out, 'w'), indent=1)
print(json.dumps({k: v for k, v in summ.items() if k != 'per_house'}, indent=1))
