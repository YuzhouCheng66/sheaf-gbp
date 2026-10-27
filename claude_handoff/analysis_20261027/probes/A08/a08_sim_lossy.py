"""A08 probe 3 (CPU, simulation): rate vs F1 of geometry codecs whose decoder RAY-CASTS visibility from the transmitted poses.
Worlds = the 6 demo houses' GT rasters; agents at the stored centres with our own +-0.1 m jitter (probe 1 law); targets re-simulated.
Every variant sends: poses (5 B) + heights (2 B) + the observed 2-D wall plan (ctx) + a furniture description; the decoder renders
slabs + wall columns + lintels + furniture and ray-casts from the sender's poses (K0) or from every reached sender (R8, per-source
casts joined by per-voxel max = the current lattice join).  Variants of the furniture description:
  vox      lossless observed furniture voxels (ctx_bin)                       -> exact by theorem
  box      greedy consistency-set box cover (adaptive box code)               -> exact by construction
  box>=k   boxes covering < k observed furniture voxels dropped (k = 2, 4, 8, 16)
  c221     furniture OR-pooled to 2x2x1 cells (ctx_bin of the coarse grid), decoded as full coarse cells
  c222     same at 2x2x2;  c441  same at 4x4x1
  none     no furniture at all
Scored with the training metrics (surface / variable / furniture-region F1, unknown->known FPR) against the simulated targets."""
import glob, json, os, sys, time
import numpy as np, torch
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from a08_common import load, observe, metrics, ctx_bin, ctx_tern2d, gt_layers, plan_split, box_cover, render_boxes, box_bits, render_structure  # noqa: E402

DATA = sys.argv[1]; OUT = sys.argv[2]
VARIANTS = ['vox', 'box', 'box>=2', 'box>=4', 'box>=8', 'box>=16', 'c221', 'c222', 'c441', 'none']


def pack(M):
    return np.packbits(M >= 0), np.packbits(M == 1)


def unpack(k, o):
    k = np.unpackbits(k)[:128 * 128 * 32].astype(bool).reshape(128, 128, 32); o = np.unpackbits(o)[:128 * 128 * 32].astype(bool).reshape(128, 128, 32)
    return np.where(k, o.astype(np.int8), np.int8(-1))


def coarse(Of, f):
    bx, by, bz = f
    c = Of.reshape(128 // bx, bx, 128 // by, by, 32 // bz, bz).any((1, 3, 5))
    return c, c.repeat(bx, 0).repeat(by, 1).repeat(bz, 2)


def load_gt(path):
    """light loader: metadata + the three GT planes only (the full load() also decodes 48 prediction planes)."""
    import base64, gzip
    m = json.load(open(path)); raw = gzip.decompress(base64.b64decode(m.pop('bin_gz_b64')))
    def plane(name):
        o, n = m['offsets'][name]
        return np.unpackbits(np.frombuffer(raw[o:o + n], dtype=np.uint8))[:128 * 128 * 32].astype(bool).reshape(128, 128, 32)
    return m, {g: plane(f'gt_{g}') for g in ('occ', 'slab', 'obj')}


def job(args):
    path, a, seed = args
    m, gt = load_gt(path); lay = gt_layers(gt); cell = tuple(m['cell_m']); rng = np.random.default_rng(seed)
    c = np.asarray(m['positions_m'][a], float)
    poses = np.stack([c + np.r_[rng.uniform(-.1, .1, 2), 0.] for _ in range(3)])
    M = observe(gt['occ'], cell, poses); O = M == 1; F = M == 0
    plan, Of = plan_split(O, gt, lay); S = render_structure(plan, gt, lay)
    base = 5. + 2. + ctx_tern2d(plan)
    bx = box_cover(Of, ~F); Dbox = render_boxes(bx)
    cnt = [int(Of[x0:x1, y0:y1, z0:z1].sum()) for x0, x1, y0, y1, z0, z1 in bx]
    desc = {}
    desc['vox'] = (S | Of, base + ctx_bin(Of))
    desc['box'] = (S | Dbox, base + box_bits(bx)[1] / 8)
    for k in (2, 4, 8, 16):
        keep = [b for b, n in zip(bx, cnt) if n >= k]
        desc[f'box>={k}'] = (S | render_boxes(keep), base + box_bits(keep)[1] / 8)
    for name, f in (('c221', (2, 2, 1)), ('c222', (2, 2, 2)), ('c441', (4, 4, 1))):
        cg, up = coarse(Of, f); desc[name] = (S | up, base + ctx_bin(cg))
    desc['none'] = (S, base)
    out = dict(house=m['house'], agent=a, target=pack(M), n_boxes=len(bx), var={})
    for v in VARIANTS:
        D, b = desc[v]
        out['var'][v] = dict(bytes=float(b), pred=pack(observe(D, cell, poses)))
    return out


def score(P, T, gt):
    slab = torch.from_numpy(gt['slab'].reshape(-1)); obj = torch.from_numpy(gt['obj'].reshape(-1))
    struct = torch.from_numpy((gt['occ'] & ~gt['obj'] & ~gt['slab']).reshape(-1))
    rows = metrics(torch.from_numpy(P.reshape(len(P), -1)), torch.from_numpy(T.reshape(len(T), -1)), slab, obj, struct)
    for r, p, t in zip(rows, P, T):
        r['fpr'] = float(((p >= 0) & (t < 0)).sum()) / float((t < 0).sum())
    return rows


if __name__ == '__main__':
    t0 = time.time()
    paths = sorted(glob.glob(os.path.join(DATA, 'house_*.json')))
    jobs = [(p, a, 1000 * i + a) for i, p in enumerate(paths) for a in range(8)]
    with Pool(8, maxtasksperchild=2) as pool:
        res = pool.map(job, jobs, chunksize=1)
    print(f'casts done {time.time() - t0:.0f}s', flush=True)
    summary = {v: {'K0': [], 'R8': [], 'bytes': []} for v in VARIANTS}
    for i, p in enumerate(paths):
        m, gt = load_gt(p); R8 = np.asarray(m['reached'][m['hops'].index(8)], bool)
        rr = [r for r in res if r['house'] == m['house']]; rr.sort(key=lambda r: r['agent'])
        T = np.stack([unpack(*r['target']) for r in rr]); T8 = np.stack([T[R8[a]].max(0) for a in range(8)])
        for v in VARIANTS:
            P = np.stack([unpack(*r['var'][v]['pred']) for r in rr]); P8 = np.stack([P[R8[a]].max(0) for a in range(8)])
            summary[v]['K0'] += score(P, T, gt); summary[v]['R8'] += score(P8, T8, gt); summary[v]['bytes'] += [r['var'][v]['bytes'] for r in rr]
    agg = {}
    for v in VARIANTS:
        d = dict(bytes_median=float(np.median(summary[v]['bytes'])), bytes_mean=float(np.mean(summary[v]['bytes'])))
        for hop in ('K0', 'R8'):
            for k in ('surface_f1', 'var_occ_f1', 'furn_f1', 'fpr'):
                vals = [r[k] for r in summary[v][hop] if r[k] is not None]
                d[f'{hop}_{k}'] = float(np.mean(vals))
        agg[v] = d
        print(f"{v:8s} own bytes med {d['bytes_median']:6.0f} | K0 surf {d['K0_surface_f1']:.3f} var {d['K0_var_occ_f1']:.3f} furn {d['K0_furn_f1']:.3f} fpr {d['K0_fpr']:.4f} | "
              f"R8 surf {d['R8_surface_f1']:.3f} var {d['R8_var_occ_f1']:.3f} furn {d['R8_furn_f1']:.3f} fpr {d['R8_fpr']:.4f}")
    json.dump(dict(agg=agg, n_boxes=[r['n_boxes'] for r in res]), open(OUT, 'w'), indent=1)
    print(f'total {time.time() - t0:.0f}s')
