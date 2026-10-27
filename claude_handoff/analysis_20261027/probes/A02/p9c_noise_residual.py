"""A02 debate probe P9c (2026-10-27, CPU only, one process): what does the 'exact facts + render' codec cost once range noise breaks
the render identity (P9)?  2 agents in each of r1_i0, r2_i0, r3_i1; OctoMap-style log-odds integration (P9 'logodds'); range noise
clean / 2 cm / 1.5 % of range.  Payload of an exact codec = hit set H (adaptive binary ctx, as P1) + correction of every voxel where
ray-cast(H, poses) differs from the noisy own map (adaptive binary ctx of the mismatch mask + 1 bit per mismatched voxel for the value);
compared with the ternary ctx of the noisy map itself (the classical lossless coder needs no identity).  Output: probes/A02/p9c_noise_residual.json"""
import json, os, sys, zlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..'))); sys.path.insert(0, HERE)
from rate_baselines_lib import load, ctx_bytes  # noqa: E402
from p1_render_hits import observe, ctx_bytes_binary  # noqa: E402
from p9_noise_identity import trace, integrate, state, RAYS  # noqa: E402

rows = []
for h in ('r1_i0', 'r2_i0', 'r3_i1'):
    m, own, pred, gt = load(os.path.join(ROOT, 'demo_web3d', 'data', f'house_{h}.json')); del pred, own
    occ = gt['occ'].astype(np.uint8); cell = np.asarray(m['cell_m'], np.float64); shape = occ.shape
    for a in (0, 1):
        rng = np.random.default_rng(zlib.crc32(m['house'].encode()) * 16 + a)
        centre = np.asarray(m['positions_m'][a], np.float64); jit = rng.uniform(-.1, .1, (3, 3)); jit[:, 2] = 0.
        poses = centre[None] + jit; tr = [trace(occ, cell, p) for p in poses]
        for sn, sig, k in (('clean', 0., 0.), ('d2cm', .02, 0.), ('r1.5pct', 0., .015)):
            nrng = np.random.default_rng(5000 + 17 * a + len(sn))
            hits = np.zeros(shape, np.int32); misses = np.zeros(shape, np.int32)
            for p, (s_hit, s_exit, ds, nsteps) in zip(poses, tr):
                d = np.where(s_hit > 0, s_hit * ds, np.nan); hm = ~np.isnan(d)
                if sig or k:
                    d[hm] = np.clip(d[hm] + nrng.normal(0, 1, hm.sum()) * (sig + k * d[hm]), 0.05, None)
                integrate(shape, cell, p, s_hit, s_exit, ds, nsteps, d, hits, misses)
            M = state(hits, misses, 'logodds'); H = M == 1
            R = observe(H.astype(np.uint8), cell, poses, RAYS); diff = R != M
            b_h = ctx_bytes_binary(H) - 2; b_res = (ctx_bytes_binary(diff) - 2 + diff.sum() / 8) if diff.any() else 0.
            row = dict(house=h, agent=a, setting=sn, n_occ=int(H.sum()), n_mismatch=int(diff.sum()), hits_bytes=b_h, residual_bytes=float(b_res),
                       exact_codec_bytes=float(b_h + b_res), ternary_ctx_bytes=ctx_bytes(M))
            rows.append(row); print(row, flush=True)
S = {}
for sn in ('clean', 'd2cm', 'r1.5pct'):
    rr = [r for r in rows if r['setting'] == sn]
    S[sn] = {k: float(np.median([r[k] for r in rr])) for k in ('n_occ', 'n_mismatch', 'hits_bytes', 'residual_bytes', 'exact_codec_bytes', 'ternary_ctx_bytes')}
print(json.dumps(S, indent=1))
json.dump(dict(summary=S, rows=rows), open(os.path.join(HERE, 'p9c_noise_residual.json'), 'w'), indent=1)
