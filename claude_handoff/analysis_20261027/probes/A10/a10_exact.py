"""a10_exact.py (A10, 2026-10-27; uk-linux CPU only, nice 19, 2 threads, self-limited to 13 min). Verifies the central claim of the
A10 design 'exact visible-surface lattice sheaf': an agent's whole ternary map is EXACTLY re-rendered from
  O = floor / ceiling slab layers (house constants) + every column the agent saw occupied both below 2.0 m and above 2.25 m (extruded
      over the full height: such a column can only be a wall) + all other first hits (explicit hits)
by re-running the simulator's own first-hit marcher (mapping_3d_storey_data_v420.observe) from the agent's 3 poses; and the round-8
union likewise from all poses. Pose precision variants: exact float64, float32-rounded, 1 cm grid (the DL's 44-bit poses).
Also the structured code's bytes (exact adaptive KT, same coders as the local probes) on this panel. Writes only to its own dir.
Run:  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 19 python -B a10_exact.py --source SEALED --scripts SCRIPTS --per-stratum 4"""
import argparse, json, math, os, sys, time
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); ap.add_argument('--scripts', required=True)
ap.add_argument('--per-stratum', type=int, default=4); ap.add_argument('--budget-s', type=float, default=780.)
args = ap.parse_args()
sys.dont_write_bytecode = True
for p in (args.source, os.path.join(args.source, 'exploration_grid'), args.scripts):
    sys.path.insert(0, p)
import mapping_3d_storey_data_v420 as W  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
T0 = time.time()
NB = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0), (0, 0, 2), (0, 2, 0), (2, 0, 0)]
NB2 = [(0, 1), (1, 0), (1, 1), (1, -1), (0, 2), (2, 0), (2, 1), (1, 2)]


def kt_bits(sym, ctx, nsym):
    sym = np.asarray(sym).reshape(-1); ctx = np.asarray(ctx).reshape(-1)
    if sym.size == 0:
        return 0.
    u, inv = np.unique(ctx, return_inverse=True)
    cnt = np.bincount(inv * nsym + sym, minlength=len(u) * nsym).reshape(-1, nsym).astype(np.float64)
    lg = np.vectorize(math.lgamma, otypes=[np.float64]); N = cnt.sum(1)
    return float(((lg(N + .5 * nsym) - math.lgamma(.5 * nsym)).sum() - (lg(cnt + .5) - math.lgamma(.5)).sum()) / math.log(2))


def causal(g, base):
    X, Y, Z = g.shape; c = np.zeros(g.shape, np.int64)
    for dx, dy, dz in NB:
        sh = np.zeros(g.shape, np.int64); sh[dx:, dy:, dz:] = g[:X - dx, :Y - dy, :Z - dz]; c = c * base + sh
    return c


def ctx2d(g):
    X, Y = g.shape; c = np.zeros(g.shape, np.int64)
    for dx, dy in NB2:
        sh = np.zeros(g.shape, np.int64)
        if dy >= 0:
            sh[dx:, dy:] = g[:X - dx, :Y - dy]
        else:
            sh[dx:, :dy] = g[:X - dx, -dy:]
        c = c * 2 + sh
    return c


ZC = (np.arange(32) + .5) * 0.1 - W.FLOOR_T


def decompose(t, slab_layers):
    V = t == 1
    low = V[:, :, (ZC < 2.0) & ~slab_layers].any(2); high = V[:, :, (ZC > 2.25) & ~slab_layers].any(2)
    W2 = low & high
    Vx = V & ~W2[:, :, None] & ~slab_layers[None, None, :]
    return W2, Vx


def occupied_set(t, slab_layers):
    W2, Vx = decompose(t, slab_layers)
    O = np.zeros(t.shape, np.uint8); O[:, :, slab_layers] = 1; O[W2] = 1; O[Vx] = 1
    return O, W2, Vx


def code_bytes(t, slab_layers, n_pose):
    W2, Vx = decompose(t, slab_layers)
    sel = ~W2[:, :, None] & ~slab_layers[None, None, :] & np.ones(t.shape, bool)
    v = Vx.astype(np.int64)
    return (kt_bits(W2.astype(np.int64), ctx2d(W2.astype(np.int64)), 2) + kt_bits(v[sel], causal(v, 2)[sel], 2)) / 8 + 4 + 24. * n_pose


def ctx_bytes(m):
    s = m.astype(np.int64) + 1
    return kt_bits(s, causal(s, 3), 3) / 8 + 2


rays = W.fibonacci_sphere(W.BEAMS)
rows = []; done = False
for rooms in (1, 2, 3):
    taken = 0
    for index in range(40):
        if taken >= args.per_stratum or done:
            break
        try:
            fx = W.make_fixture('VALIDATION', index, rooms)
        except ValueError:
            continue
        taken += 1
        own = fx['private']; n = own.shape[0]; cell = fx['cell']; traj = fx['trajectory_m']; occ = fx['occ']
        slab_layers = fx['slab'].any((0, 1))
        R8 = fx['reached'][list(W.HOPS).index(8)]
        h = dict(house=f'r{rooms}_i{index}', agents=[], union=[])
        for a in range(n):
            O, W2, Vx = occupied_set(own[a], slab_layers)
            assert (occ[O.astype(bool)] == 1).all(), 'O contains a free voxel'
            r = dict(agent=a, ctx=ctx_bytes(own[a]), code=code_bytes(own[a], slab_layers, 1), w2_cols=int(W2.sum()), hits=int(Vx.sum()))
            for name, P in (('f64', traj[:, a]), ('f32', traj[:, a].astype(np.float32).astype(np.float64)), ('cm', np.round(traj[:, a] * 100) / 100)):
                rec = W.observe(O, cell, P, rays)
                r[f'mism_{name}'] = int((rec != own[a]).sum()); r[f'occ_mism_{name}'] = int(((rec == 1) != (own[a] == 1)).sum())
            h['agents'].append(r)
        for a in range(n):                                              # round-8 union of agent a's reached set
            idx = np.nonzero(R8[a])[0]
            if a > 0 and (R8[a] == R8[0]).all():
                h['union'].append(dict(agent=a, same_as=0)); continue
            U = own[idx].max(0)
            O, W2, Vx = occupied_set(U, slab_layers)
            assert (occ[O.astype(bool)] == 1).all()
            rec = np.full(U.shape, -1, np.int8)
            for b in idx:
                rec = np.maximum(rec, W.observe(O, cell, traj[:, b], rays))
            h['union'].append(dict(agent=a, reached=int(len(idx)), ctx=ctx_bytes(U), code=code_bytes(U, slab_layers, len(idx)),
                                   mism_f64=int((rec != U).sum()), occ_mism_f64=int(((rec == 1) != (U == 1)).sum())))
        rows.append(h)
        print(h['house'], f'{time.time() - T0:.0f}s', 'own mism f64/f32/cm', [sum(r[f'mism_{k}'] for r in h['agents']) for k in ('f64', 'f32', 'cm')],
              'occ mism cm', sum(r['occ_mism_cm'] for r in h['agents']), 'union mism', [u.get('mism_f64') for u in h['union'] if 'mism_f64' in u],
              'code med', round(float(np.median([r['code'] for r in h['agents']]))), 'ctx med', round(float(np.median([r['ctx'] for r in h['agents']]))), flush=True)
        json.dump(rows, open(os.path.join(HERE, 'a10_exact.json'), 'w'), indent=1)
        if time.time() - T0 > args.budget_s:
            done = True
ag = [r for h in rows for r in h['agents']]
print('houses', len(rows), 'agents', len(ag), 'voxel mismatches f64', sum(r['mism_f64'] for r in ag), 'f32', sum(r['mism_f32'] for r in ag), 'cm', sum(r['mism_cm'] for r in ag),
      'occupancy mismatches cm', sum(r['occ_mism_cm'] for r in ag), '| own code median', round(float(np.median([r['code'] for r in ag]))), 'ctx median', round(float(np.median([r['ctx'] for r in ag]))))
un = [u for h in rows for u in h['union'] if 'mism_f64' in u]
print('unions', len(un), 'mismatches f64', sum(u['mism_f64'] for u in un), 'code median', round(float(np.median([u['code'] for u in un]))), 'ctx median', round(float(np.median([u['ctx'] for u in un]))))
