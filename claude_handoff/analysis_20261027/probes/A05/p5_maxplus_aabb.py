"""A05 probe 5 (CPU, demo data, exact poses): the simplest EXACTLY FUSABLE furniture code -- a max-plus (support-function) stalk.
Per public block b (bx x by x bz voxels) the stalk is the axis-aligned bounding box of the furniture hits inside b, written as six
max-pooled numbers (max x, max -x, ...). It is a lattice homomorphism: the code of a union of maps = the elementwise max of the codes,
so the join over the forest yields exactly the code of the union target (no source selection, no fusion gap). Decoder: fill every
block's box, add the exact structure layer, ray-cast from the poses. Variants: one box per block, or K boxes per block by fixed
z-slabs (the box of the hits in each z-slab). Rate: active-block index (ctx_bin of the block grid) + the six coordinates coded with an
adaptive KT model whose context is the same coordinate of the causal x-neighbour block (a crude stand-in for a learned prior)."""
import glob, json, math, os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, HERE)
from rate_baselines_lib import load, targets, metrics, DIMS  # noqa: E402
from p1_hitset_rate import ctx_bin  # noqa: E402
from p2b_raycast_decode import ray_table, cast, struct_parts, fpr  # noqa: E402

LG = math.lgamma


def aabb_code(Hf, b, kz):
    """per block (and per z-slab of height b[2] // kz): AABB of hits in local coords; returns active mask [Bx,By,Bz,kz] and lo/hi [.., 3]"""
    X, Y, Z = Hf.shape; bx, by, bz = b; sz = bz // kz
    h = Hf.reshape(X // bx, bx, Y // by, by, Z // bz, kz, sz).transpose(0, 2, 4, 5, 1, 3, 6)   # [Bx,By,Bz,kz, bx,by,sz]
    act = h.any((4, 5, 6))
    ix = np.arange(bx)[:, None, None]; iy = np.arange(by)[None, :, None]; iz = np.arange(sz)[None, None, :]
    big = 10 ** 6
    lo = np.stack([np.where(h, ix, big).min((4, 5, 6)), np.where(h, iy, big).min((4, 5, 6)), np.where(h, iz, big).min((4, 5, 6))], -1)
    hi = np.stack([np.where(h, ix, -1).max((4, 5, 6)), np.where(h, iy, -1).max((4, 5, 6)), np.where(h, iz, -1).max((4, 5, 6))], -1)
    return act, lo, hi


def aabb_fill(act, lo, hi, b, kz, shape):
    X, Y, Z = shape; bx, by, bz = b; sz = bz // kz
    g = np.zeros(shape, bool)
    for i, j, k, s in np.argwhere(act):
        x0, y0, z0 = lo[i, j, k, s]; x1, y1, z1 = hi[i, j, k, s]
        g[i * bx + x0:i * bx + x1 + 1, j * by + y0:j * by + y1 + 1, k * bz + s * sz + z0:k * bz + s * sz + z1 + 1] = True
    return g


def coord_bits(act, lo, hi, b, kz):
    """adaptive KT code of the 6 coordinates of every active box, context = same coordinate of the x-1 neighbour box (or 'none')"""
    sz = b[2] // kz; card = [b[0], b[1], sz, b[0], b[1], sz]
    vals = np.concatenate([lo, hi], -1)                                   # [Bx,By,Bz,kz,6]
    bits = 0.
    for c in range(6):
        A = card[c]; counts = {}
        for i, j, k, s in np.argwhere(act):
            ctx = int(vals[i - 1, j, k, s, c]) if i > 0 and act[i - 1, j, k, s] else A
            v = int(vals[i, j, k, s, c])
            tab = counts.setdefault(ctx, np.zeros(A))
            bits += -math.log2((tab[v] + 0.5) / (tab.sum() + 0.5 * A)); tab[v] += 1
    return bits / 8


if __name__ == '__main__':
    poses = json.load(open(os.path.join(HERE, 'poses.json')))
    VARIANTS = [((4, 4, 4), 1), ((4, 4, 4), 2), ((4, 4, 8), 4), ((2, 2, 4), 1), ((8, 8, 8), 4)]
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, 'demo_web3d', 'data', 'house_*.json'))):
        t0 = time.time()
        m, own, pred, gt = load(path); n = own.shape[0]; key = m['house']
        traj = np.asarray(poses[key]['trajectory_m']); cell = tuple(poses[key]['cell']); lk = poses[key]['lintel_k']
        tables = {a: [ray_table(traj[k, a], cell) for k in range(traj.shape[0])] for a in range(n)}
        slab = gt['slab']; obj = gt['obj']; occ = gt['occ']; struct = occ & ~obj & ~slab
        slab_t = torch.from_numpy(slab.reshape(-1)); obj_t = torch.from_numpy(obj.reshape(-1)); struct_t = torch.from_numpy(struct.reshape(-1))
        R8 = np.asarray(m['reached'][m['hops'].index(8)], bool); T8 = targets(own, R8)
        for hop, T in ((0, own), (8, T8)):
            seen = set()
            for a in range(n):
                agents = [a] if hop == 0 else list(np.nonzero(R8[a])[0])
                if hop == 8 and tuple(agents) in seen:
                    continue
                seen.add(tuple(agents))
                t = T[a]; Hf = (t == 1) & obj
                Gs, wall, lint = struct_parts(t == 1, gt, lk)
                tabs = [tb for b_ in agents for tb in tables[b_]]
                for b, kz in VARIANTS:
                    act, lo, hi = aabb_code(Hf, b, kz)
                    if hop == 8:                                          # homomorphism check: union code == elementwise max / min of the agents' codes
                        codes = [aabb_code((own[b_] == 1) & obj, b, kz) for b_ in agents]
                        A_ = np.any([c[0] for c in codes], 0)
                        LO = np.min([np.where(c[0][..., None], c[1], 10 ** 6) for c in codes], 0); HI = np.max([np.where(c[0][..., None], c[2], -1) for c in codes], 0)
                        hom = bool((A_ == act).all() and (LO[act] == lo[act]).all() and (HI[act] == hi[act]).all())
                    else:
                        hom = None
                    G = Gs | (aabb_fill(act, lo, hi, b, kz, DIMS) & ~struct & ~slab)
                    Pr = cast(G, tabs)
                    mt = metrics(torch.from_numpy(Pr.reshape(1, -1)), torch.from_numpy(t.reshape(1, -1)), slab_t, obj_t, struct_t)[0]
                    rate_idx = ctx_bin(act.reshape(act.shape[0], act.shape[1], -1)) if act.any() else 2.
                    rows.append(dict(house=key, hop=hop, agent=a, block=list(b), kz=kz, active=int(act.sum()), fpr=fpr(Pr, t), homomorphic=hom,
                                     rate_index=rate_idx, rate_coords=coord_bits(act, lo, hi, b, kz), **mt))
        print(key, f'{time.time() - t0:.0f}s', flush=True)
    S = {}
    for hop in (0, 8):
        for b, kz in VARIANTS:
            r = [x for x in rows if x['hop'] == hop and x['block'] == list(b) and x['kz'] == kz]
            S[f"{'K0' if hop == 0 else 'R8'} block{b[0]}x{b[1]}x{b[2]} kz{kz}"] = dict(
                active=float(np.median([x['active'] for x in r])), furn_bytes=float(np.median([x['rate_index'] + x['rate_coords'] for x in r])),
                surface_f1=float(np.mean([x['surface_f1'] for x in r])), var_f1=float(np.mean([x['var_occ_f1'] for x in r])),
                furn_f1=float(np.mean([x['furn_f1'] for x in r if x['furn_f1'] is not None])), fpr=float(np.mean([x['fpr'] for x in r])),
                homomorphic_all=all(x['homomorphic'] for x in r if x['homomorphic'] is not None) if hop == 8 else None, n=len(r))
    json.dump(dict(summary=S, rows=rows), open(os.path.join(HERE, 'p5_maxplus_aabb.json'), 'w'), indent=1)
    for k, v in S.items():
        print(f'{k:28s} ' + ' '.join(f'{kk} {vv:.4f}' if isinstance(vv, float) else f'{kk} {vv}' for kk, vv in v.items()))
