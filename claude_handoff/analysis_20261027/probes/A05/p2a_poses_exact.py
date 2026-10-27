"""A05 probe 2a (CPU, demo data): recover the exact sensor poses of the 6 demo houses and verify the ray-cast theorem bit for bit.
Poses: world_id = v420:VALIDATION:<house_id from the local v365 index>; rng = default_rng(sha256(world_id + ':poses:8')[:8]);
centres = v420.agent_positions(gt_occ, cell, 8, rng); jitter = rng.uniform(-.1, .1, (3, 8, 3)) with z = 0 (v420.make_fixture).
Checks per agent: (1) centres == the export's positions_m; (2) v420.observe(gt_occ, poses) == the export's own map;
(3) RayCast(H, poses) == own map with H = observed-occupied voxels only; (4) RayCast(G_s, poses) == own map with the structured
occluder set of p1b (slabs + extruded wall plan + lintels + furniture hits); (5) the vectorised ray caster used by p2b == v420.observe.
Writes poses.json (trajectories in metres) for p2b."""
import glob, hashlib, json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from rate_baselines_lib import load  # noqa: E402
from scipy import ndimage  # noqa: E402


class W:
    """verbatim copies of mapping_3d_storey_data_v420 constants and of fibonacci_sphere / observe / agent_positions (pure numpy)"""
    FLOOR_T = 0.10; SENSOR_RANGE = 5.0; POSES = 3; SENSOR_HEIGHT = 1.2; BEAMS = 65536

    @staticmethod
    def fibonacci_sphere(n):
        i = np.arange(n) + .5
        phi = np.arccos(1 - 2 * i / n); theta = math.pi * (1 + 5 ** .5) * i
        return np.stack((np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)), 1)

    @staticmethod
    def observe(occ, cell, poses, rays):
        shape = np.array(occ.shape); cell = np.asarray(cell, np.float64)
        out = np.full(occ.shape, -1, np.int8)
        ds = 0.5 * float(cell.min()); nsteps = int(math.ceil(W.SENSOR_RANGE / ds))
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

    @staticmethod
    def agent_positions(spec, occ_r, cell, agents, rng):
        cx, _, cz = cell
        zc = (np.arange(occ_r.shape[2]) + .5) * cz
        band = (zc >= W.FLOOR_T) & (zc < W.FLOOR_T + W.SENSOR_HEIGHT + 0.3)
        free2 = ~occ_r[:, :, band].any(2)
        lab, nlab = ndimage.label(free2)
        if nlab == 0:
            raise ValueError('no free floor')
        comp = lab == (1 + int(np.argmax(np.bincount(lab.ravel())[1:])))
        clear = ndimage.distance_transform_edt(free2) * cx
        cand = np.argwhere(comp & (clear >= 0.45))
        if len(cand) < agents:
            raise ValueError('not enough free floor with clearance')
        order = rng.permutation(len(cand))
        for min_sep in (1.0, 0.5):
            pts = []
            for i in order:
                p = np.array([(cand[i][0] + .5) * cx, (cand[i][1] + .5) * cx])
                if all(np.linalg.norm(p - q) >= min_sep for q in pts):
                    pts.append(p)
                if len(pts) == agents:
                    return np.array([[q[0], q[1], W.FLOOR_T + W.SENSOR_HEIGHT] for q in pts])
        raise ValueError('could not place agents')

INDEX = os.path.join(ROOT, 'claude_handoff', 'claude_reviews_20261021', 'houseexpo_index_v365.json')


def seed(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'little')


RAYS = W.fibonacci_sphere(W.BEAMS)


def raycast(occ, cell, poses, chunk=8192):
    """vectorised first-hit ray marching, same arithmetic as v420.observe; occ bool [X,Y,Z]; returns ternary int8"""
    shape = np.array(occ.shape); cell = np.asarray(cell, np.float64)
    out = np.full(occ.shape, -1, np.int8)
    ds = 0.5 * float(cell.min()); S = int(math.ceil(W.SENSOR_RANGE / ds))
    steps = np.arange(1, S + 1, dtype=np.float64) * ds
    flat = occ.reshape(-1)
    for p in poses:
        for c0 in range(0, len(RAYS), chunk):
            r = RAYS[c0:c0 + chunk]
            idx = np.floor((p[None, None, :] + r[:, None, :] * steps[None, :, None]) / cell).astype(np.int64)   # [R, S, 3]
            inb = np.all((idx >= 0) & (idx < shape), 2)
            lin = np.where(inb, (idx[..., 0] * shape[1] + idx[..., 1]) * shape[2] + idx[..., 2], 0)
            hit = inb & flat[lin]
            stop = ~inb | hit
            first = np.where(stop.any(1), stop.argmax(1), S)                                                       # first terminating step
            before = np.arange(S)[None, :] < first[:, None]
            fr = lin[before]
            o = out.reshape(-1); o[fr] = np.maximum(o[fr], 0)
            rows = np.nonzero(first < S)[0]
            hh = rows[hit[rows, first[rows]]]
            o[lin[hh, first[hh]]] = 1
    return out


def structured_G(H, gt):
    slab = gt['slab']; obj = gt['obj']; occ = gt['occ']; struct = occ & ~obj & ~slab
    full_col = struct[:, :, 1:20].all(2); part = struct.any(2) & ~full_col
    lk = int(np.nonzero(struct[part].any(0))[0].min()) if part.any() else 32
    Hs = H & struct; wall = Hs[:, :, :lk].any(2); lint = Hs.any(2) & ~wall
    G = slab.copy(); G |= wall[:, :, None]; L3 = np.zeros_like(G); L3[:, :, lk:] = lint[:, :, None]; G |= L3; G |= H & obj
    return G, lk


if __name__ == '__main__':
    idx = json.load(open(INDEX)); lists = {}
    for r in idx['houses']:
        lists.setdefault((r['split'], r['stratum']), []).append(r)
    out = {}; report = []
    only = sys.argv[1:]
    for path in sorted(glob.glob(os.path.join(ROOT, 'demo_web3d', 'data', 'house_*.json'))):
        if only and not any(h in path for h in only):
            continue
        t0 = time.time()
        m, own, pred, gt = load(path); n = own.shape[0]
        hid = lists[('VALIDATION', int(m['rooms']))][int(m['index'])]['id']
        world_id = f'v420:VALIDATION:{hid}'
        rng = np.random.default_rng(seed(world_id + f':poses:{n}'))
        L = float(m['chart_m']); cell = (L / 128, L / 128, 3.2 / 32)
        centres = W.agent_positions(None, gt['occ'].astype(bool), cell, n, rng)
        jitter = rng.uniform(-.1, .1, (W.POSES, n, 3)); jitter[:, :, 2] = 0.
        traj = centres[None] + jitter
        ok_c = bool(np.abs(centres - np.asarray(m['positions_m'])).max() < 2e-3)
        occ = gt['occ'].astype(np.uint8)
        res = dict(house=m['house'], house_id=hid, centres_match=ok_c, agents=[])
        for a in range(n):
            poses = traj[:, a]
            ref = W.observe(occ, cell, poses, RAYS)
            mine = raycast(gt['occ'].astype(bool), cell, poses)
            H = own[a] == 1
            fromH = raycast(H, cell, poses)
            G, lk = structured_G(H, gt)
            fromG = raycast(G, cell, poses)
            res['agents'].append(dict(agent=a, observe_eq_export=bool((ref == own[a]).all()), vectorised_eq_observe=bool((mine == ref).all()),
                                      rayH_eq_export=bool((fromH == own[a]).all()), rayGs_eq_export=bool((fromG == own[a]).all()),
                                      mismatch_observe=int((ref != own[a]).sum())))
        out[m['house']] = dict(house_id=hid, chart_m=L, cell=cell, trajectory_m=traj.tolist(), lintel_k=lk)
        report.append(res)
        print(m['house'], hid, 'centres', ok_c, [(r['observe_eq_export'], r['vectorised_eq_observe'], r['rayH_eq_export'], r['rayGs_eq_export']) for r in res['agents']], f'{time.time() - t0:.0f}s', flush=True)
    suffix = ('_' + '_'.join(only)) if only else ''
    if not only:
        json.dump(out, open(os.path.join(HERE, 'poses.json'), 'w'), indent=1)
    json.dump(report, open(os.path.join(HERE, f'p2a_poses_exact{suffix}.json'), 'w'), indent=1)
    allok = all(r['centres_match'] and all(a['observe_eq_export'] and a['vectorised_eq_observe'] and a['rayH_eq_export'] and a['rayGs_eq_export'] for a in r['agents']) for r in report)
    print('ALL EXACT:', allok)
