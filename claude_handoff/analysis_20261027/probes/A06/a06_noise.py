"""A06 debate probe (Claude A06, 2026-10-27): furniture-F1 ceiling when the sensor has RANGE NOISE and maps are scored against the clean
truth (the evaluation change several members ask for). World = gt_occ of demo houses; exact poses (a06_raycast jitters); range noise
eps ~ N(0, sigma) added to every first-hit distance (sigma = 1 / 2 / 4 cm), the noisy ray then marks free up to the noisy range and a hit
at the noisy end point. Per agent (3 poses) two integration rules: 'occ-wins' (the generator's rule: any hit -> occupied) and an
OctoMap-style log-odds rule (occupied if .85 hits > .40 passes, counted once per ray per voxel). Scored with the training metric against
the CLEAN own map (= what a perfect lossless codec of the noisy map can reach: the 'floor'). Structured ceiling (analytic, not a
decoder): target furniture surface voxels grouped into faces (26-component x outward direction); a decoder that pools the n_f clean hits
of a face places it on the right voxel plane with P = 2 Phi((cell/2) sqrt(n_f) / sigma) - 1; voxel-weighted mean over faces ~ furniture
F1 of an ideal structure-aware denoiser. One process, CPU. Usage: a06_noise.py [n_houses]"""
import json, math, os, sys, time
import numpy as np
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a06_common import houses, scene, metrics_rows, mean  # noqa: E402
from a06_raycast import fibonacci_sphere, BEAMS, POSES, SENSOR_RANGE  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
NH = int(sys.argv[1]) if len(sys.argv) > 1 else 2
SIGMAS = [0.01, 0.02, 0.04]


def march(occ, cell, p, rays):
    """first-hit step per ray (0 = no hit in range) and per-voxel clean hit counts, same marching as observe()"""
    shape = np.array(occ.shape); ds = 0.5 * float(cell.min()); nsteps = int(math.ceil(SENSOR_RANGE / ds))
    alive = np.ones(len(rays), bool); hit_step = np.zeros(len(rays), np.int64)
    for s in range(1, nsteps + 1):
        idx = np.floor((p[None] + rays * (s * ds)) / cell).astype(np.int64)
        inside = np.all((idx >= 0) & (idx < shape), 1) & alive
        if not inside.any():
            break
        ii = idx[inside]; hit = occ[ii[:, 0], ii[:, 1], ii[:, 2]] == 1
        ai = np.nonzero(inside)[0]; hit_step[ai[hit]] = s; alive[ai[hit]] = False
        alive &= inside | ~alive
    return hit_step, ds, nsteps


def noisy_counts(occ, cell, poses, rays, sigma, rng):
    shape = np.array(occ.shape); H = np.zeros(occ.shape, np.int32); Fr = np.zeros(occ.shape, np.int32); Hc = np.zeros(occ.shape, np.int32)
    for p in poses:
        hs, ds, nsteps = march(occ, cell, p, rays)
        has = hs > 0
        dist = np.where(has, hs * ds + rng.normal(0, sigma, len(rays)) if sigma > 0 else hs * ds, np.inf)
        dist = np.where(has, np.clip(dist, ds, SENSOR_RANGE), np.inf)
        # clean hit counts (for the face statistics)
        e = np.floor((p[None] + rays[has] * (hs[has, None] * ds)) / cell).astype(np.int64); ok = np.all((e >= 0) & (e < shape), 1)
        np.add.at(Hc, tuple(e[ok].T), 1)
        # noisy end points
        e = np.floor((p[None] + rays[has] * dist[has, None]) / cell).astype(np.int64); ok = np.all((e >= 0) & (e < shape), 1)
        hv = e[ok]; np.add.at(H, tuple(hv.T), 1)
        ec = np.minimum(np.maximum(e, 0), shape - 1)
        endkey = np.full(len(rays), -1, np.int64); hk = np.ravel_multi_index(tuple(ec.T), occ.shape); hk[~ok] = -1; endkey[np.nonzero(has)[0]] = hk
        prev = np.full(len(rays), -1, np.int64)
        for s in range(1, nsteps + 1):
            t = s * ds
            act = t < dist                                       # still in front of the (noisy) end point
            if not act.any():
                break
            idx = np.floor((p[None] + rays[act] * t) / cell).astype(np.int64)
            inside = np.all((idx >= 0) & (idx < shape), 1)
            ai = np.nonzero(act)[0][inside]; key = np.ravel_multi_index(tuple(idx[inside].T), occ.shape)
            new = (key != prev[ai]) & (key != endkey[ai])          # once per ray per voxel; the end voxel is not a pass
            np.add.at(Fr.reshape(-1), key[new], 1); prev[ai] = key
    return H, Fr, Hc


def face_ceiling(t, S, Hc, cell, sigmas):
    """voxel-weighted mean P(face on the right plane) over target furniture faces, per sigma"""
    region = S['region']; furn = (t == 1) & region; free = t == 0
    lab, nl = ndimage.label(S['obj'], structure=np.ones((3, 3, 3)))
    dirs = [((0, 0, 1), cell[2]), ((1, 0, 0), cell[0]), ((-1, 0, 0), cell[0]), ((0, 1, 0), cell[1]), ((0, -1, 0), cell[1]), ((0, 0, -1), cell[2])]
    assigned = np.zeros_like(furn); faces = []
    for k, ((dx, dy, dz), c) in enumerate(dirs):
        nb = np.roll(np.roll(np.roll(free, -dx, 0), -dy, 1), -dz, 2)
        sel = furn & nb & ~assigned; assigned |= sel
        if not sel.any():
            continue
        ids = lab[sel] * 6 + k; w = Hc[sel]
        u, inv = np.unique(ids, return_inverse=True)
        nvox = np.bincount(inv); nhit = np.bincount(inv, weights=w)
        faces += [(int(a), float(b), c) for a, b in zip(nvox, nhit)]
    rest = int((furn & ~assigned).sum())
    out = {}
    for sg in sigmas:
        num = sum(nv * (math.erf((c / 2) * math.sqrt(max(nh, 1)) / (sg * math.sqrt(2))) if nh > 0 else 0.) for nv, nh, c in faces)
        out[str(sg)] = num / max(sum(nv for nv, _, _ in faces) + rest, 1)
    nh = np.array([b for _, b, _ in faces]); nv = np.array([a for a, _, _ in faces])
    return out, dict(faces=len(faces), voxels=int(nv.sum()), hits_per_face_median=float(np.median(nh)) if len(nh) else 0, voxels_with_no_face=rest,
                     share_voxels_in_faces_with_le3_hits=float(nv[nh <= 3].sum() / max(nv.sum(), 1)))


rays = fibonacci_sphere(BEAMS)
res = {}
for path in houses()[:NH]:
    t0 = time.time()
    S = scene(path); m = S['m']; n = m['agents']; cell = np.asarray(m['cell_m'], np.float64); occ = S['occ'].astype(np.uint8)
    rng0 = np.random.default_rng(int.from_bytes(m['house'].encode(), 'little') % (2 ** 32))
    centres = np.asarray(m['positions_m'], np.float64); jit = rng0.uniform(-.1, .1, (POSES, n, 3)); jit[:, :, 2] = 0.; traj = centres[None] + jit
    rng = np.random.default_rng(342202)
    clean, preds = [], {}
    for a in range(n):
        H, Fr, Hc = noisy_counts(occ, cell, traj[:, a], rays, 0., rng)
        t = np.where(H > 0, 1, np.where(Fr > 0, 0, -1)).astype(np.int8); clean.append(t)
        if a == 0 and not res.get('checked'):                    # the sigma = 0 path must reproduce the generator's observe() exactly
            from a06_raycast import observe
            ref = observe(occ, cell, traj[:, a], rays); res['checked'] = True
            assert (ref == t).all(), int((ref != t).sum())
        ceil, stats = face_ceiling(t, S, Hc, cell, SIGMAS + [0.08])
        res.setdefault('ceiling', []).append(ceil); res.setdefault('face_stats', []).append(stats)
        for sg in SIGMAS:
            H, Fr, _ = noisy_counts(occ, cell, traj[:, a], rays, sg, rng)
            preds.setdefault(f'occwins_{sg}', []).append(np.where(H > 0, 1, np.where(Fr > 0, 0, -1)).astype(np.int8))
            lo = .85 * H - .40 * Fr
            preds.setdefault(f'logodds_{sg}', []).append(np.where((H > 0) & (lo > 0), 1, np.where(Fr > 0, 0, np.where(H > 0, 1, -1))).astype(np.int8))
    T = np.stack(clean)
    for k, P in preds.items():
        rows = metrics_rows(np.stack(P), T, S)
        d = res.setdefault(k, dict(furn=[], surf=[], fpr=[]))
        d['furn'] += [r['furn_f1'] for r in rows]; d['surf'] += [r['surface_f1'] for r in rows]
        d['fpr'] += [float(((np.stack(P)[a] >= 0) & (T[a] < 0)).sum() / max((T[a] < 0).sum(), 1)) for a in range(n)]
    print(m['house'], f'{time.time() - t0:.0f}s', flush=True)
summ = {k: {kk: mean(v) for kk, v in d.items()} for k, d in res.items() if k not in ('ceiling', 'face_stats', 'checked')}
summ['ceiling'] = {sg: mean([c[sg] for c in res['ceiling']]) for sg in res['ceiling'][0]}
summ['face_stats'] = {k: float(np.median([s[k] for s in res['face_stats']])) for k in res['face_stats'][0]}
json.dump(dict(houses=NH, summary=summ), open(os.path.join(OUT, 'noise.json'), 'w'), indent=1)
for k, v in summ.items():
    print(k, json.dumps(v))
