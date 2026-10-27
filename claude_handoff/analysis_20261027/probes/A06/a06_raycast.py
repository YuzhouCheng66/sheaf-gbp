"""A06 raycast probe (Claude A06, 2026-10-27). CPU only. Claim under test: a first-hit map is a deterministic function of (its occupied
voxel set O, the sensor poses): re-casting the same rays through O alone reproduces the map exactly (O is a subset of the world, and every
first hit of the world lies in O), and the R8 union map = re-cast of all reached agents' poses through the union's O. If true, the known
mask (77 % of the lossless ctx bits) costs only the poses, and a codec that sends O (+ 44 bit/agent of poses) is lossless.
The demo JSON has the world (gt_occ) and agent centres but not the pose jitters, so own maps are RE-SIMULATED with the generator's own
observe() (copied verbatim) at the centres + our own +-0.1 m jitters: statistically the same maps, not the same bits; the agreement with
the demo own maps is reported.
Measured per house: (1) exactness of re-cast (own, R8 union); (2) bytes: ctx of the ternary map vs ctx of the binary O (+ poses);
(3) per tree edge: cost of the sender-side O alone and conditioned on the receiver-side O (common knowledge after the reverse message);
(4) lossy O variants re-cast through the decoder-side ray caster: coarse 2x2x1 / 1x1x2 / 2x2x2 'any' blocks, O minus thin furniture parts,
scored with the training metrics (surface, var, furniture-region F1, unknown->known FPR) against the re-simulated targets."""
import json, math, os, sys, time
from multiprocessing import Pool
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a06_common import houses, scene, metrics_rows, mean, RB  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
SENSOR_RANGE = 5.0; BEAMS = 65536; POSES = 3


def fibonacci_sphere(n):
    i = np.arange(n) + .5
    phi = np.arccos(1 - 2 * i / n); theta = math.pi * (1 + 5 ** .5) * i
    return np.stack((np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)), 1)


def observe(occ, cell, poses, rays):
    """verbatim from mapping_3d_storey_data_v420.observe"""
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


def ctx_bin(mask, side=None):
    """exact adaptive KT code length (bytes, + 2) of a binary 3D grid, causal 9-voxel context (as rate_split.ctx_bytes_binary);
    with `side` (common-knowledge binary grid) two more context bits: side at the voxel, any side voxel in its 6-neighbourhood."""
    s = torch.from_numpy(mask.astype(np.int64)); X, Y, Z = s.shape; c = torch.zeros_like(s)
    for dx, dy, dz in RB.NB:
        sh = torch.zeros_like(s); sh[dx:, dy:, dz:] = s[:X - dx, :Y - dy, :Z - dz]; c = c * 2 + sh
    if side is not None:
        sd = torch.from_numpy(side.astype(np.int64)); n6 = torch.from_numpy(dil(side).astype(np.int64))
        c = (c * 2 + sd) * 2 + n6
    idx = (c * 2 + s).reshape(-1)
    cnt = torch.bincount(idx, minlength=2 * 2 ** len(RB.NB) * (4 if side is not None else 1)).reshape(-1, 2).double()
    N = cnt.sum(1); used = N > 0; cnt, N = cnt[used], N[used]
    bits = ((RB.LG(N + 1.) - RB.LG(1.)) - (RB.LG(cnt + 0.5) - RB.LG(0.5)).sum(1)).sum() / math.log(2)
    return float(bits) / 8 + 2


def dil(m):
    o = m.copy(); o[1:] |= m[:-1]; o[:-1] |= m[1:]; o[:, 1:] |= m[:, :-1]; o[:, :-1] |= m[:, 1:]; o[:, :, 1:] |= m[:, :, :-1]; o[:, :, :-1] |= m[:, :, 1:]
    return o


def coarse_any(O, f):
    X, Y, Z = O.shape; bx, by, bz = f
    c = O.reshape(X // bx, bx, Y // by, by, Z // bz, bz).any((1, 3, 5))
    return c, c.repeat(bx, 0).repeat(by, 1).repeat(bz, 2)


POSE_BYTES = 44 / 8


def run_house(path):
    t0 = time.time()
    S = scene(path); m = S['m']; n = m['agents']; W = S['occ'].astype(np.uint8); cell = np.asarray(m['cell_m'], np.float64)
    rays = fibonacci_sphere(BEAMS)
    rng = np.random.default_rng(int.from_bytes(m['house'].encode(), 'little') % (2 ** 32))
    centres = np.asarray(m['positions_m'], np.float64)
    jit = rng.uniform(-.1, .1, (POSES, n, 3)); jit[:, :, 2] = 0.
    traj = centres[None] + jit
    own = np.stack([observe(W, cell, traj[:, a], rays) for a in range(n)])
    agree = [float((own[a] == S['own'][a]).mean()) for a in range(n)]
    # NOTE: raycast.json of the 2026-10-27 run stores this field with a bug (numerator counted both-unknown voxels too, values ~4-9);
    # fixed here to the intended IoU of the known masks. The memo quotes only agree_all (.996).
    agree_known = [float(((own[a] >= 0) & (S['own'][a] >= 0)).sum() / max(((own[a] >= 0) | (S['own'][a] >= 0)).sum(), 1)) for a in range(n)]
    reached = np.asarray(m['reached'][m['hops'].index(8)], bool)
    T8 = RB.targets(own, reached)
    res = dict(house=m['house'], agree_all=agree, agree_known_iou=agree_known, own=[], union=[], edges=[], lossy={})
    # (1)+(2) own maps: exactness and bytes
    for a in range(n):
        O = own[a] == 1
        re = observe(O.astype(np.uint8), cell, traj[:, a], rays)
        res['own'].append(dict(exact=bool((re == own[a]).all()), mismatch=int((re != own[a]).sum()), ctx_ternary=RB.ctx_bytes(own[a]),
                               ctx_O=ctx_bin(O), n_occ=int(O.sum()), known=float((own[a] >= 0).mean())))
    # union maps (identical inside a connected component at R8): exactness and bytes
    comps = {}
    for a in range(n):
        comps.setdefault(tuple(np.nonzero(reached[a])[0]), []).append(a)
    recast_union = {}
    for members, rows in comps.items():
        a = rows[0]; O = T8[a] == 1
        re = np.stack([observe(O.astype(np.uint8), cell, traj[:, b], rays) for b in members]).max(0)
        recast_union[members] = re
        for a in rows:
            res['union'].append(dict(agent=a, reached=len(members), exact=bool((re == T8[a]).all()), mismatch=int((re != T8[a]).sum()),
                                     ctx_ternary=RB.ctx_bytes(T8[a]), ctx_O=ctx_bin(O), n_occ=int(O.sum()), poses_B=POSE_BYTES * len(members)))
    # (3) tree edges: sender-side O, alone and conditioned on the receiver-side O
    adj = {i: set() for i in range(n)}
    for i, j in m['edges']:
        adj[i].add(j); adj[j].add(i)
    def side_of(s, r):
        side, stack = {s}, [s]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v != r and v not in side:
                    side.add(v); stack.append(v)
        return sorted(side)
    for i, j in m['edges']:
        for s, r in ((i, j), (j, i)):
            Ss, Sr = side_of(s, r), side_of(r, s)
            Os = (own[Ss] == 1).any(0); Or = (own[Sr] == 1).any(0)
            res['edges'].append(dict(sender=s, side=len(Ss), recv_side=len(Sr), ctx_ternary=RB.ctx_bytes(own[Ss].max(0)), ctx_O=ctx_bin(Os),
                                     ctx_O_given_recv=ctx_bin(Os & ~Or, side=Or), new_occ=int((Os & ~Or).sum()), n_occ=int(Os.sum())))
    # (4) lossy O variants, decoded by re-casting (decoder knows the poses)
    thin = S['thin'] & S['region']
    variants = {'2x2x1_any': (2, 2, 1), '1x1x2_any': (1, 1, 2), '2x2x2_any': (2, 2, 2), 'minus_thin': None, 'minus_thin_xy': 'xy'}
    for name, f in variants.items():
        preds_own, preds_u, bytes_own, bytes_u = [], [], [], []
        for a in range(n):
            O = own[a] == 1
            if f is None:
                Od = O & ~thin; b = ctx_bin(Od)
            elif f == 'xy':
                Od = O & ~(S['thin_xy'] & S['region']); b = ctx_bin(Od)
            else:
                c, Od = coarse_any(O, f); b = ctx_bin(c)
            preds_own.append(observe(Od.astype(np.uint8), cell, traj[:, a], rays)); bytes_own.append(b + POSE_BYTES)
        for members, rows in comps.items():
            O = T8[rows[0]] == 1
            if f is None:
                Od = O & ~thin; b = ctx_bin(Od)
            elif f == 'xy':
                Od = O & ~(S['thin_xy'] & S['region']); b = ctx_bin(Od)
            else:
                c, Od = coarse_any(O, f); b = ctx_bin(c)
            re = np.stack([observe(Od.astype(np.uint8), cell, traj[:, bb], rays) for bb in members]).max(0)
            for a in rows:
                preds_u.append((a, re)); bytes_u.append(b + POSE_BYTES * len(members))
        preds_u = np.stack([p for _, p in sorted(preds_u, key=lambda x: x[0])])
        r0 = metrics_rows(np.stack(preds_own), own, S); r8 = metrics_rows(preds_u, T8, S)
        fpr0 = [float(((np.stack(preds_own)[a] >= 0) & (own[a] < 0)).sum() / max((own[a] < 0).sum(), 1)) for a in range(n)]
        fpr8 = [float(((preds_u[a] >= 0) & (T8[a] < 0)).sum() / max((T8[a] < 0).sum(), 1)) for a in range(n)]
        res['lossy'][name] = dict(K0={k: [r[k] for r in r0] for k in ('surface_f1', 'var_occ_f1', 'furn_f1')}, R8={k: [r[k] for r in r8] for k in ('surface_f1', 'var_occ_f1', 'furn_f1')},
                                  K0_bytes=bytes_own, R8_bytes=bytes_u, K0_fpr=fpr0, R8_fpr=fpr8)
    # reference: the demo's own maps' ctx and our model's furniture F1 on this house
    res['secs'] = time.time() - t0
    print(m['house'], f'{res["secs"]:.0f}s', flush=True)
    return res


if __name__ == '__main__':
    torch.set_num_threads(2)
    part = os.path.join(OUT, 'raycast_partial.json')
    R = json.load(open(part)) if os.path.exists(part) else []
    done = {r['house'] for r in R}
    for path in houses():                                   # sequential: the shared machine is near its commit limit
        if os.path.basename(path)[6:-5] in done:
            continue
        for attempt in range(40):                           # the shared machine hits its commit limit now and then: wait and retry
            try:
                R.append(run_house(path)); break
            except (MemoryError, RuntimeError) as e:
                if not isinstance(e, MemoryError) and 'memory' not in str(e).lower():
                    raise
                print('memory pressure, retry', attempt, flush=True); time.sleep(30)
        json.dump(R, open(part, 'w'))
    json.dump(R, open(os.path.join(OUT, 'raycast.json'), 'w'), indent=1)
    def med(v):
        return float(np.median(v))
    own = [r for h in R for r in h['own']]; uni = [r for h in R for r in h['union']]; edg = [r for h in R for r in h['edges']]
    print('agreement with demo own maps (all voxels / known IoU):', mean([x for h in R for x in h['agree_all']]), mean([x for h in R for x in h['agree_known_iou']]))
    print(f"own: exact {sum(r['exact'] for r in own)}/{len(own)}; ctx ternary med {med([r['ctx_ternary'] for r in own]):.0f} B; ctx O med {med([r['ctx_O'] for r in own]):.0f} B (+5.5 B poses); occ voxels med {med([r['n_occ'] for r in own]):.0f}")
    print(f"union8: exact {sum(r['exact'] for r in uni)}/{len(uni)}; ctx ternary med {med([r['ctx_ternary'] for r in uni]):.0f} B; ctx O med {med([r['ctx_O'] for r in uni]):.0f} B; poses med {med([r['poses_B'] for r in uni]):.0f} B")
    print(f"edges: ctx ternary med {med([r['ctx_ternary'] for r in edg]):.0f}; ctx O med {med([r['ctx_O'] for r in edg]):.0f}; ctx O | receiver med {med([r['ctx_O_given_recv'] for r in edg]):.0f} (mean {np.mean([r['ctx_O_given_recv'] for r in edg]):.0f}); new occ frac {np.sum([r['new_occ'] for r in edg]) / np.sum([r['n_occ'] for r in edg]):.3f}")
    for name in R[0]['lossy']:
        k0 = {k: mean([x for h in R for x in h['lossy'][name]['K0'][k]]) for k in ('surface_f1', 'var_occ_f1', 'furn_f1')}
        k8 = {k: mean([x for h in R for x in h['lossy'][name]['R8'][k]]) for k in ('surface_f1', 'var_occ_f1', 'furn_f1')}
        b0 = med([x for h in R for x in h['lossy'][name]['K0_bytes']]); b8 = med([x for h in R for x in h['lossy'][name]['R8_bytes']])
        f0 = mean([x for h in R for x in h['lossy'][name]['K0_fpr']]); f8 = mean([x for h in R for x in h['lossy'][name]['R8_fpr']])
        print(f"{name:14s} K0 {b0:6.0f} B surf {k0['surface_f1']:.3f} var {k0['var_occ_f1']:.3f} furn {k0['furn_f1']:.3f} fpr {f0:.4f} | R8 {b8:6.0f} B surf {k8['surface_f1']:.3f} var {k8['var_occ_f1']:.3f} furn {k8['furn_f1']:.3f} fpr {f8:.4f}")
