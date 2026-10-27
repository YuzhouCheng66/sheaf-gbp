"""A05 probe 2b (CPU, demo data, exact poses from p2a): what does a RAY-CAST decoder score when its geometry is imperfect?
Decode = RayCast(G_hat, poses of the agents the target covers) -> ternary map (free = traversed, occupied = first hit, else unknown);
scored with the training metric (surface / variable / furniture-region F1, target-known scope) and the unknown->known FPR.
G_hat = exact structure (slabs + extruded wall plan + lintels, lossless, from the target's own hits) + a furniture part:
  exact      : the furniture hits (sanity: must be 1.0 / FPR 0)
  none       : no furniture (structure-only code)
  model      : the current model's predicted-occupied voxels that are not world slab / structure (3S-X s342101 + calibrated rule)
  model_nb   : as model, minus predictions inside the one-voxel band around structure (a furniture-only decoder)
  thin_miss  : furniture hits minus every thin part (pole / sheet / panel of gt_obj, as p3)
  thick1     : furniture hits dilated by one voxel (6-nbhd) into world-free, non-structure space
  drop20     : 20 % of the furniture hits removed at random
  shell2     : furniture hits + world furniture voxels within 2 voxels of them (sound completion; must stay exact)
  down_bxbybz: furniture hits OR-pooled on bx x by x bz blocks and broadcast back (classical coarse furniture), with its ctx rate
For 'model', 'thin_miss', 'thick1', 'drop20' the same G_hat is also scored with the ORACLE known mask (target known, occupancy from
G_hat) to separate ray-cast amplification from the geometry error itself."""
import glob, json, math, os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, HERE)
from rate_baselines_lib import load, targets, metrics, dilate6, DIMS  # noqa: E402
from p1_hitset_rate import ctx_bin  # noqa: E402
from p1b_structured_rate import ctx2d  # noqa: E402
from p3_furniture_anatomy import thin_axes  # noqa: E402

SENSOR_RANGE = 5.0


def fib(n):
    i = np.arange(n) + .5
    phi = np.arccos(1 - 2 * i / n); theta = math.pi * (1 + 5 ** .5) * i
    return np.stack((np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)), 1)


RAYS = fib(65536)


def ray_table(p, cell, shape=DIMS):
    """int32 [R, S] flat voxel index of every march sample of pose p (-1 once outside the grid), same arithmetic as v420.observe"""
    cell = np.asarray(cell, np.float64); shape = np.asarray(shape)
    ds = 0.5 * float(cell.min()); S = int(math.ceil(SENSOR_RANGE / ds)); steps = np.arange(1, S + 1, dtype=np.float64) * ds
    out = np.empty((len(RAYS), S), np.int32)
    for c0 in range(0, len(RAYS), 8192):
        r = RAYS[c0:c0 + 8192]
        idx = np.floor((p[None, None, :] + r[:, None, :] * steps[None, :, None]) / cell).astype(np.int64)
        inb = np.all((idx >= 0) & (idx < shape), 2)
        lin = (idx[..., 0] * shape[1] + idx[..., 1]) * shape[2] + idx[..., 2]
        out[c0:c0 + 8192] = np.where(inb, lin, -1)
    return out


def cast(G, tables):
    """union over poses of the first-hit march through occluder set G (bool [X,Y,Z]); returns int8 ternary map"""
    flat = G.reshape(-1); o = np.full(flat.shape, -1, np.int8)
    for lin in tables:
        S = lin.shape[1]
        valid = lin >= 0
        hit = valid & flat[np.where(valid, lin, 0)]
        stop = ~valid | hit
        first = np.where(stop.any(1), stop.argmax(1), S)
        before = np.arange(S)[None, :] < first[:, None]
        fr = lin[before]; o[fr] = np.maximum(o[fr], 0)
        rows = np.nonzero(first < S)[0]; hh = rows[hit[rows, first[rows]]]
        o[lin[hh, first[hh]]] = 1
    return o.reshape(G.shape)


def struct_parts(H, gt, lk):
    slab = gt['slab']; obj = gt['obj']; struct = gt['occ'] & ~obj & ~slab
    Hs = H & struct; wall = Hs[:, :, :lk].any(2); lint = Hs.any(2) & ~wall
    G = slab.copy(); G |= wall[:, :, None]; L3 = np.zeros_like(G); L3[:, :, lk:] = lint[:, :, None]; G |= L3
    return G, wall, lint


def down_or(m, b):
    X, Y, Z = m.shape
    c = m.reshape(X // b[0], b[0], Y // b[1], b[1], Z // b[2], b[2]).any((1, 3, 5))
    return c, c.repeat(b[0], 0).repeat(b[1], 1).repeat(b[2], 2)


def fpr(pred, t):
    u = t < 0
    return float((pred[u] >= 0).mean()) if u.any() else None


def run_house(path):
    poses = json.load(open(os.path.join(HERE, 'poses.json')))
    m, own, pred, gt = load(path); n = own.shape[0]; key = m['house']
    traj = np.asarray(poses[key]['trajectory_m']); cell = tuple(poses[key]['cell']); lk = poses[key]['lintel_k']
    t0 = time.time()
    tables = {a: [ray_table(traj[k, a], cell) for k in range(traj.shape[0])] for a in range(n)}
    slab = gt['slab']; obj = gt['obj']; occ = gt['occ']; struct = occ & ~obj & ~slab
    band = dilate6(torch.from_numpy(struct.reshape(1, -1)), DIMS)[0].numpy().reshape(DIMS)
    tx, ty, tz = thin_axes(obj); thin = tx | ty | tz
    slab_t = torch.from_numpy(slab.reshape(-1)); obj_t = torch.from_numpy(obj.reshape(-1)); struct_t = torch.from_numpy(struct.reshape(-1))
    rng = np.random.default_rng(342101)
    R8 = np.asarray(m['reached'][m['hops'].index(8)], bool)
    T8 = targets(own, R8)
    rows = []
    for hop, T in ((0, own), (8, T8)):
        done = {}
        for a in range(n):
            t = T[a]; H = t == 1; Hf = H & obj
            agents = [a] if hop == 0 else list(np.nonzero(R8[a])[0])
            tabs = [tb for b in agents for tb in tables[b]]
            Gs, wall, lint = struct_parts(H, gt, lk)
            P = pred[hop][a]
            mf = (P == 1) & ~slab & ~struct
            near = dilate6(torch.from_numpy(Hf.reshape(1, -1)), DIMS)[0].numpy().reshape(DIMS)
            shell = Hf.copy()
            for _ in range(2):
                shell = shell | (dilate6(torch.from_numpy(shell.reshape(1, -1)), DIMS)[0].numpy().reshape(DIMS) & obj)
            scen = dict(exact=Hf, none=np.zeros_like(Hf), model=mf, model_nb=mf & ~band, thin_miss=Hf & ~thin,
                        thick1=Hf | (near & ~occ & ~band), drop20=Hf & (rng.random(Hf.shape) >= 0.2), shell2=shell)
            for b in ((2, 2, 1), (2, 2, 2), (4, 4, 4)):
                c, up = down_or(Hf, b); scen[f'down_{b[0]}{b[1]}{b[2]}'] = up & ~struct & ~slab
            # structure-only scenarios are identical for every agent of a component at R8: reuse
            sig = (hop, tuple(agents))
            for name, Gf in scen.items():
                if hop == 8 and name not in ('model', 'model_nb') and (sig, name) in done:
                    res = dict(done[(sig, name)]); res['agent'] = a; rows.append(res); continue
                G = Gs | Gf
                Pr = cast(G, tabs)
                mt = metrics(torch.from_numpy(Pr.reshape(1, -1)), torch.from_numpy(t.reshape(1, -1)), slab_t, obj_t, struct_t)[0]
                res = dict(house=key, hop=hop, agent=a, scen=name, mode='raycast', fpr=fpr(Pr, t), **mt)
                if name.startswith('down_'):
                    bb = tuple(int(ch) for ch in name[5:])
                    c, _ = down_or(Hf, bb)
                    res['rate_furn'] = ctx_bin(c) if min(c.shape) > 2 else None
                    res['rate_plan'] = ctx2d(wall) + ctx2d(lint, wall, 2) + 2
                if name == 'exact':
                    res['rate_furn'] = ctx_bin(Hf); res['rate_plan'] = ctx2d(wall) + ctx2d(lint, wall, 2) + 2
                    res['exact_ok'] = bool((Pr == t).all())
                rows.append(res); done[(sig, name)] = res
                if name in ('model', 'model_nb', 'thin_miss', 'thick1', 'drop20'):
                    Po = np.where(t >= 0, G.astype(np.int8), np.int8(-1))
                    mo = metrics(torch.from_numpy(Po.reshape(1, -1)), torch.from_numpy(t.reshape(1, -1)), slab_t, obj_t, struct_t)[0]
                    rows.append(dict(house=key, hop=hop, agent=a, scen=name, mode='oracle_known', fpr=0.0, **mo))
            # the model as it is now (its own known head, calibrated rule): reference row
            mr = metrics(torch.from_numpy(P.reshape(1, -1)), torch.from_numpy(t.reshape(1, -1)), slab_t, obj_t, struct_t)[0]
            rows.append(dict(house=key, hop=hop, agent=a, scen='model_as_is', mode='known_head', fpr=fpr(P, t), **mr))
        print(key, 'hop', hop, f'{time.time() - t0:.0f}s', flush=True)
    json.dump(rows, open(os.path.join(HERE, f'p2b_rows_{key}.json'), 'w'))
    return key


if __name__ == '__main__':
    import multiprocessing as mp
    paths = sorted(glob.glob(os.path.join(ROOT, 'demo_web3d', 'data', 'house_*.json')))
    torch.set_num_threads(4)
    todo = [p for p in paths if not os.path.exists(os.path.join(HERE, 'p2b_rows_' + os.path.basename(p)[6:-5] + '.json'))]
    if todo:
        with mp.get_context('spawn').Pool(2) as pool:
            print(pool.map(run_house, todo), flush=True)
    allrows = [r for p in paths for r in json.load(open(os.path.join(HERE, 'p2b_rows_' + os.path.basename(p)[6:-5] + '.json')))]
    S = {}
    for hop in (0, 8):
        for mode in ('raycast', 'oracle_known', 'known_head'):
            for scen in sorted({r['scen'] for r in allrows if r['hop'] == hop and r['mode'] == mode}):
                rr = [r for r in allrows if r['hop'] == hop and r['mode'] == mode and r['scen'] == scen]
                agg = {}
                for k in ('surface_f1', 'var_occ_f1', 'furn_f1', 'fpr', 'rate_furn', 'rate_plan'):
                    v = [r[k] for r in rr if r.get(k) is not None]
                    if v:
                        agg[k] = float(np.mean(v)) if k not in ('rate_furn', 'rate_plan') else float(np.median(v))
                if any('exact_ok' in r for r in rr):
                    agg['exact_ok_all'] = all(r['exact_ok'] for r in rr if 'exact_ok' in r)
                agg['n'] = len(rr)
                S[f"{'K0' if hop == 0 else 'R8'}|{mode}|{scen}"] = agg
    json.dump(dict(summary=S, rows=allrows), open(os.path.join(HERE, 'p2b_raycast_decode.json'), 'w'), indent=1)
    for k, v in S.items():
        print(f"{k:34s} " + ' '.join(f"{kk} {vv:.4f}" if isinstance(vv, float) else f"{kk} {vv}" for kk, vv in v.items()))
