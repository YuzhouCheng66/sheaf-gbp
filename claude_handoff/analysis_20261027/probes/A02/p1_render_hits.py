"""A02 probe P1 (2026-10-27, CPU only): is the ternary own map a deterministic function of (hit set, poses)?  And what do the parts cost?
For each demo VAL house (6) and agent (8): draw 3 poses = public centre + uniform +-0.1 m jitter (own RNG; the demo export has no
jitter), simulate the own map with the generator's exact sensor (observe() copied from mapping_3d_storey_data_v420), then
  (a) identity: render(hits, exact poses) == own map?  (hits = observed-occupied voxels, everything else treated as free)
  (b) pose quantisation: jitter rounded to 1 cm / 1 mm / 0.1 mm -> voxels that differ (the residual a lossless codec must add)
  (c) code lengths (adaptive KT, 9 causal neighbours, as rate_baselines_lib.ctx_bytes): full ternary map, known mask, hit set alone,
      and the same for the REAL exported own maps (sanity: the synthetic maps must look like the real ones).
Output: probes/A02/p1_render_hits.json"""
import glob, json, math, os, sys, time, zlib
import numpy as np
import torch
from multiprocessing import get_context

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from rate_baselines_lib import load, ctx_bytes, NB, LG  # noqa: E402

SENSOR_RANGE = 5.0; BEAMS = 65536; POSES = 3


def fibonacci_sphere(n):                                      # copied from mapping_3d_storey_data_v420
    i = np.arange(n) + .5
    phi = np.arccos(1 - 2 * i / n); theta = math.pi * (1 + 5 ** .5) * i
    return np.stack((np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)), 1)


def observe(occ, cell, poses, rays):                         # copied verbatim (logic) from mapping_3d_storey_data_v420.observe
    shape = np.array(occ.shape); cell = np.asarray(cell, np.float64)
    out = np.full(occ.shape, -1, np.int8)
    ds = 0.5 * float(cell.min()); nsteps = int(math.ceil(SENSOR_RANGE / ds))
    for p in poses:
        alive = np.ones(len(rays), bool)
        for s in range(1, nsteps + 1):
            idx_all = np.floor((p[None] + rays * (s * ds)) / cell).astype(np.int64)
            inside = np.all((idx_all >= 0) & (idx_all < shape), 1) & alive
            if not inside.any():
                break
            ii = idx_all[inside]
            hit = occ[ii[:, 0], ii[:, 1], ii[:, 2]] == 1
            fr = ii[~hit]; out[fr[:, 0], fr[:, 1], fr[:, 2]] = np.maximum(out[fr[:, 0], fr[:, 1], fr[:, 2]], 0)
            hh = ii[hit]; out[hh[:, 0], hh[:, 1], hh[:, 2]] = 1
            ai = np.nonzero(inside)[0]; alive[ai[hit]] = False
            alive &= inside | ~alive
    return out


def ctx_bytes_binary(mask):
    s = torch.from_numpy(mask.astype(np.int64)); X, Y, Z = s.shape; c = torch.zeros_like(s)
    for dx, dy, dz in NB:
        sh = torch.zeros_like(s); sh[dx:, dy:, dz:] = s[:X - dx, :Y - dy, :Z - dz]; c = c * 2 + sh
    idx = (c * 2 + s).reshape(-1)
    cnt = torch.bincount(idx, minlength=2 ** (len(NB) + 1)).reshape(-1, 2).double()
    N = cnt.sum(1); used = N > 0; cnt, N = cnt[used], N[used]
    bits = ((LG(N + 1.) - LG(1.)) - (LG(cnt + 0.5) - LG(0.5)).sum(1)).sum() / math.log(2)
    return float(bits) / 8 + 2


def job(args):
    path, a = args
    torch.set_num_threads(1)
    m, own, pred, gt = load(path)
    occ = gt['occ'].astype(np.uint8); cell = tuple(m['cell_m']); rays = fibonacci_sphere(BEAMS)
    rng = np.random.default_rng(zlib.crc32(m["house"].encode()) * 16 + a)
    centre = np.asarray(m['positions_m'][a], np.float64)
    jit = rng.uniform(-.1, .1, (POSES, 3)); jit[:, 2] = 0.
    poses = centre[None] + jit
    t0 = time.time(); syn = observe(occ, cell, poses, rays); t_obs = time.time() - t0
    hits = (syn == 1).astype(np.uint8)
    ren = observe(hits, cell, poses, rays)
    row = dict(house=m['house'], agent=a, t_observe=t_obs, known=float((syn >= 0).mean()), n_hits=int(hits.sum()),
               identity_mismatch=int((ren != syn).sum()))
    for name, q in (('1cm', 0.01), ('1mm', 0.001), ('0.1mm', 0.0001)):
        pq = centre[None] + np.round(jit / q) * q
        rq = observe(hits, cell, pq, rays)
        d = rq != syn
        row[f'mismatch_{name}'] = int(d.sum())
        row[f'mismatch_{name}_occ_or_surface'] = int((d & ((syn == 1) | (rq == 1))).sum())
        row[f'resid_bytes_{name}'] = ctx_bytes_binary(d) - 2 if d.any() else 0.
    row.update(ctx_full_syn=ctx_bytes(syn), ctx_known_syn=ctx_bytes_binary(syn >= 0), ctx_hits_syn=ctx_bytes_binary(hits.astype(bool)),
               ctx_full_real=ctx_bytes(own[a]), ctx_known_real=ctx_bytes_binary(own[a] >= 0), ctx_hits_real=ctx_bytes_binary(own[a] == 1),
               known_real=float((own[a] >= 0).mean()), n_hits_real=int((own[a] == 1).sum()))
    print(f"{m['house']} a{a}: identity mismatch {row['identity_mismatch']}, 1cm {row['mismatch_1cm']}, 1mm {row['mismatch_1mm']}, "
          f"ctx full {row['ctx_full_syn']:.0f} hits {row['ctx_hits_syn']:.0f} (real {row['ctx_full_real']:.0f} / {row['ctx_hits_real']:.0f}) {t_obs:.1f}s", flush=True)
    return row


if __name__ == '__main__':
    data = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'demo_web3d', 'data')
    paths = sorted(glob.glob(os.path.join(data, 'house_*.json')))
    jobs = [(p, a) for p in paths for a in range(8)]
    with get_context('spawn').Pool(16) as pool:
        rows = pool.map(job, jobs, chunksize=1)
    S = {}
    for k in rows[0]:
        if k in ('house', 'agent'):
            continue
        v = np.asarray([r[k] for r in rows], float)
        S[k] = dict(median=float(np.median(v)), mean=float(v.mean()), p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)), max=float(v.max()))
    print(json.dumps(S, indent=1))
    json.dump(dict(summary=S, rows=rows), open(os.path.join(HERE, 'p1_render_hits.json'), 'w'), indent=1)
