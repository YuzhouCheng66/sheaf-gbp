"""probe_visorder.py (A01, 2026-10-27): exact lossless cost of a map when the receiver re-simulates the rays ('visibility for free').
Codec (exactly decodable): sender and receiver march the public ray set (65,536 Fibonacci directions x 3 poses per agent, the generator's
step ds = half the smallest cell) in a fixed interleaved order (step s, pose p, ray r). Each time a ray enters a voxel whose state is not
yet decided, ONE bit is coded (occupied -> the ray stops; free -> it continues). Voxels no ray enters stay unknown at zero cost. The
decoded ternary map equals the sensor's map bit for bit (same marching as mapping_3d_storey_data_v420.observe). For a union (round 8 or
a directed tree edge), the agents' ray sets are processed one agent after the other; a voxel decided by an earlier agent costs nothing.
Context of a coded voxel (decided-before-me neighbours only): states of its 6 face neighbours (undecided / free / occupied), the number
of occupied decided edge neighbours (0, 1, 2+), and the distance band from the agent's centre (<1, 1-3, >3 m): 6,561 contexts.
Probabilities: adaptive KT per map (honest, nothing pretrained) and LOO-pretrained (counts from the other 5 houses + adaptive).
Maps are SYNTHETIC re-observations of the demo houses' ground truth with the demo agent centres and our own +-0.1 m jitter (the demo
files do not store the jitter); their ternary ctx and O-only costs are measured too, so the comparison is apples to apples.
Bits are attributed to the class of the coded voxel: furniture (gt obj), structure (walls / lintels), slab, free space.
Usage: python probe_visorder.py DATA_DIR OUT_JSON [processes]"""
import glob, json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, HERE)

SENSOR_RANGE = 5.0
INF = np.iinfo(np.int64).max
FACE = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
EDGE = [(a, b, 0) for a in (-1, 1) for b in (-1, 1)] + [(a, 0, b) for a in (-1, 1) for b in (-1, 1)] + [(0, a, b) for a in (-1, 1) for b in (-1, 1)]


def fibonacci_sphere(n):
    i = np.arange(n) + .5
    phi = np.arccos(1 - 2 * i / n); theta = math.pi * (1 + 5 ** .5) * i
    return np.stack((np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)), 1)


def first_visits(occ, cell, poses, rays):
    """ternary map (as observe) and the first-visit event index of every voxel (INF if never entered)"""
    shape = np.array(occ.shape); cell = np.asarray(cell, np.float64); R = len(rays); P = len(poses)
    out = np.full(occ.shape, -1, np.int8); t = np.full(occ.size, INF, np.int64)
    ds = 0.5 * float(cell.min()); nsteps = int(math.ceil(SENSOR_RANGE / ds))
    alive = np.ones((P, R), bool); done = np.zeros(P, bool)
    for s in range(1, nsteps + 1):
        if done.all():
            break
        for p in range(P):
            if done[p]:
                continue
            idx_all = np.floor((poses[p][None] + rays * (s * ds)) / cell).astype(np.int64)
            inside = np.all((idx_all >= 0) & (idx_all < shape), 1) & alive[p]
            if not inside.any():
                done[p] = True; continue
            ai = np.nonzero(inside)[0]; ii = idx_all[ai]
            flat = np.ravel_multi_index(ii.T, occ.shape)
            u, first = np.unique(flat, return_index=True)
            new = t[u] == INF
            t[u[new]] = ((s - 1) * P + p) * R + ai[first[new]]
            hit = occ.reshape(-1)[flat] == 1
            fr = flat[~hit]; o = out.reshape(-1); o[fr] = np.maximum(o[fr], 0); o[flat[hit]] = 1
            alive[p, ai[hit]] = False
            alive[p] &= inside | ~alive[p]
    return out, t


def nb_flat(shape, d):
    """flat index of neighbour at offset d for every voxel, -1 outside"""
    X, Y, Z = shape
    x, y, z = np.meshgrid(np.arange(X), np.arange(Y), np.arange(Z), indexing='ij')
    nx, ny, nz = x + d[0], y + d[1], z + d[2]
    ok = (nx >= 0) & (nx < X) & (ny >= 0) & (ny < Y) & (nz >= 0) & (nz < Z)
    return np.where(ok, (nx * Y + ny) * Z + nz, -1).reshape(-1)


def event_contexts(t, state, dist_band, NBF, NBE):
    """for every visited voxel: context id from neighbours decided strictly before it"""
    v = np.nonzero(t < INF)[0]; tv = t[v]
    ctx = np.zeros(len(v), np.int64)
    for k, nf in enumerate(NBF):
        n = nf[v]; okn = n >= 0
        dec = np.zeros(len(v), bool); dec[okn] = t[n[okn]] < tv[okn]
        val = np.where(dec, 1 + state[np.where(okn, n, 0)], 0)
        ctx = ctx * 3 + val
    occe = np.zeros(len(v), np.int64)
    for nf in NBE:
        n = nf[v]; okn = n >= 0
        dec = np.zeros(len(v), bool); dec[okn] = t[n[okn]] < tv[okn]
        occe += (dec & (state[np.where(okn, n, 0)] == 1))
    ctx = (ctx * 3 + np.minimum(occe, 2)) * 3 + dist_band[v]
    return v, ctx, state[v], np.argsort(tv, kind='stable')


NCTX = 3 ** 8


def kt_bits(ctx_sorted, sym_sorted, prior=None):
    """adaptive KT in event order; prior = [NCTX, 2] pretrained counts (or None)"""
    ctx_sorted = ctx_sorted.astype(np.int64); sym_sorted = sym_sorted.astype(np.int64)
    order = np.argsort(ctx_sorted, kind='stable'); cs = ctx_sorted[order]; ss = sym_sorted[order]
    start = np.ones(len(cs), bool); start[1:] = cs[1:] != cs[:-1]
    grp = np.cumsum(start) - 1; first = np.nonzero(start)[0]
    one = ss.astype(np.float64); zero = 1 - one
    e1 = np.cumsum(one) - one; e0 = np.cumsum(zero) - zero
    n1 = e1 - e1[first][grp]; n0 = e0 - e0[first][grp]
    if prior is not None:
        n0 = n0 + prior[cs, 0]; n1 = n1 + prior[cs, 1]
    p1 = (n1 + 0.5) / (n0 + n1 + 1.)
    ps = np.where(ss == 1, p1, 1 - p1)
    bits = -np.log2(ps)
    out = np.empty_like(bits); out[order] = bits
    return out


NB13 = [(1, a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)] + [(0, 1, -1), (0, 1, 0), (0, 1, 1), (0, 0, 1)]


def ctx_binary(mask, nb):
    """adaptive KT code of a binary 3-D map in raster order with causal neighbour context nb (same as probe_raycast)"""
    import torch
    from rate_baselines_lib import LG
    s = torch.from_numpy(np.ascontiguousarray(mask).astype(np.int64)); c = torch.zeros_like(s); X, Y, Z = s.shape
    for dx, dy, dz in nb:
        sh = torch.zeros_like(s)
        sh[max(dx, 0):X + min(dx, 0), max(dy, 0):Y + min(dy, 0), max(dz, 0):Z + min(dz, 0)] = s[max(-dx, 0):X - max(dx, 0), max(-dy, 0):Y - max(dy, 0), max(-dz, 0):Z - max(dz, 0)]
        c = c * 2 + sh
    idx = (c * 2 + s).reshape(-1)
    cnt = torch.bincount(idx, minlength=2 ** (len(nb) + 1)).reshape(-1, 2).double()
    N = cnt.sum(1); used = N > 0; cnt, N = cnt[used], N[used]
    return float(((LG(N + 1.) - LG(1.)) - (LG(cnt + 0.5) - LG(0.5)).sum(1)).sum() / math.log(2)) / 8 + 2


def house_job(path):
    import hashlib, torch
    torch.set_num_threads(2)
    from rate_baselines_lib import load, ctx_bytes
    t0 = time.time()
    m, own_demo, pred, gt = load(path); n = m['agents']
    cell = tuple(m['cell_m']); occ = gt['occ'].astype(np.uint8); shape = occ.shape
    rays = fibonacci_sphere(65536); rng = np.random.default_rng(int.from_bytes(hashlib.sha256(m['house'].encode()).digest()[:4], 'little'))
    NBF = [nb_flat(shape, d) for d in FACE]; NBE = [nb_flat(shape, d) for d in EDGE]
    X, Y, Z = shape
    xc = (np.arange(X) + .5) * cell[0]; yc = (np.arange(Y) + .5) * cell[1]; zc = (np.arange(Z) + .5) * cell[2]
    cls = np.full(shape, 3, np.int8)                                   # 0 furniture, 1 structure, 2 slab, 3 free space
    cls[gt['slab']] = 2; cls[gt['occ'] & ~gt['obj'] & ~gt['slab']] = 1; cls[gt['obj']] = 0
    cls = cls.reshape(-1)
    state = occ.reshape(-1).astype(np.int64)
    maps, ts, bands = [], [], []
    for a in range(n):
        centre = np.asarray(m['positions_m'][a], np.float64)
        jit = rng.uniform(-.1, .1, (3, 3)); jit[:, 2] = 0.
        o, t = first_visits(occ, cell, centre[None] + jit, rays); maps.append(o); ts.append(t)
        d = np.sqrt((xc[:, None, None] - centre[0]) ** 2 + (yc[None, :, None] - centre[1]) ** 2 + (zc[None, None, :] - centre[2]) ** 2).reshape(-1)
        bands.append(np.digitize(d, [1.0, 3.0]).astype(np.int64))
    own = np.stack(maps)
    events = {}
    def record(key, t, band):
        v, ctx, sym, order = event_contexts(t, state, band, NBF, NBE)
        events[key] = dict(ctx=ctx[order].astype(np.int16), sym=sym[order].astype(np.int8), cls=cls[v][order].astype(np.int8))
    for a in range(n):
        record(('own', a), ts[a], bands[a])
    BIG = 10 ** 9
    R8 = np.asarray(m['reached'][m['hops'].index(8)], bool)
    for a in range(n):                                                 # union: agents processed in index order
        tu = np.full(occ.size, INF, np.int64); band = np.zeros(occ.size, np.int64)
        for rank, s_ in enumerate(np.nonzero(R8[a])[0]):
            ts_ = np.where(ts[s_] < INF, rank * BIG + ts[s_], INF)
            take = ts_ < tu; tu = np.where(take, ts_, tu); band = np.where(take, bands[s_], band)
        record(('union8', a), tu, band)
    adj = {i: set() for i in range(n)}
    for i, j in m['edges']:
        adj[i].add(j); adj[j].add(i)
    for i, j in m['edges']:
        for s0, r0 in ((i, j), (j, i)):
            side, stack = {s0}, [s0]
            while stack:
                u = stack.pop()
                for w in adj[u]:
                    if w != r0 and w not in side:
                        side.add(w); stack.append(w)
            tu = np.full(occ.size, INF, np.int64); band = np.zeros(occ.size, np.int64)
            for rank, s_ in enumerate(sorted(side)):
                ts_ = np.where(ts[s_] < INF, rank * BIG + ts[s_], INF)
                take = ts_ < tu; tu = np.where(take, ts_, tu); band = np.where(take, bands[s_], band)
            record(('edge', s0, r0, len(side)), tu, band)
    # apples-to-apples raster coders on the synthetic maps
    raster = {}
    for a in range(n):
        raster[('own', a)] = dict(full=ctx_bytes(own[a]), O13=ctx_binary(own[a] == 1, NB13), n_known=int((own[a] >= 0).sum()))
    T8 = np.where(R8[:, :, None, None, None], own[None], np.int8(-1)).max(1)
    for a in range(n):
        raster[('union8', a)] = dict(full=ctx_bytes(T8[a]), O13=ctx_binary(T8[a] == 1, NB13), n_known=int((T8[a] >= 0).sum()))
    # exactness of the synthetic maps vs the union decoding (the decoder's map = states at visited voxels)
    for a in range(n):
        tk = events[('union8', a)]
        assert len(tk['sym']) == int((T8[a] >= 0).sum()), 'union decode count mismatch'
    counts = np.zeros((NCTX, 2))
    for key, e in events.items():
        if key[0] == 'own':
            np.add.at(counts, (e['ctx'].astype(np.int64), e['sym'].astype(np.int64)), 1.)
    print(m['house'], f'{time.time()-t0:.0f}s', flush=True)
    return dict(house=m['house'], events={str(k): v for k, v in events.items()}, raster={str(k): v for k, v in raster.items()}, counts=counts)


if __name__ == '__main__':
    import multiprocessing
    DATA = sys.argv[1]; OUT = sys.argv[2]; procs = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    paths = sorted(glob.glob(os.path.join(DATA, 'house_*.json')))
    with multiprocessing.get_context('spawn').Pool(procs) as pool:
        H = pool.map(house_job, paths)
    rows = []
    CN = ('furniture', 'structure', 'slab', 'free')
    for hi, h in enumerate(H):
        prior = sum(g['counts'] for gi, g in enumerate(H) if gi != hi)        # LOO pretrained counts (own-map events of the other houses)
        for key, e in h['events'].items():
            kind = eval(key)[0]
            b_ad = kt_bits(e['ctx'], e['sym']); b_pre = kt_bits(e['ctx'], e['sym'], prior)
            r = dict(house=h['house'], key=key, kind=kind, events=int(len(e['sym'])), vis_adaptive=float(b_ad.sum() / 8 + 2), vis_pretrained=float(b_pre.sum() / 8 + 2))
            for c in range(4):
                r[f'bits_{CN[c]}_share'] = float(b_pre[e['cls'] == c].sum() / max(b_pre.sum(), 1e-9))
                r[f'occ_{CN[c]}_share'] = float(((e['cls'] == c) & (e['sym'] == 1)).sum() / max((e['sym'] == 1).sum(), 1))
            rk = h['raster'].get(key)
            if rk:
                r.update({f'raster_{k}': v for k, v in rk.items()})
            if kind == 'edge':
                r['side'] = eval(key)[3]
            rows.append(r)
    S = {}
    for kind in ('own', 'union8', 'edge'):
        R_ = [r for r in rows if r['kind'] == kind]
        S[kind] = {k: float(np.median([r[k] for r in R_])) for k in R_[0] if k not in ('house', 'key', 'kind')}
    json.dump(dict(summary=S, rows=rows), open(OUT, 'w'), indent=1)
    print(json.dumps(S, indent=1))
