"""3D occupancy world family v400 (Claude, 2026-10-24): synthetic multi-storey buildings for 3D Sheaf-GBP map fusion.

Geometry (metres, deterministic from (split, index)): a 12.8 m cube holding FLOORS = 4 storeys of 3.2 m. Each storey has a 0.3 m slab
(the lowest slab is the ground), a 1.6-2.0 m corridor along y through the middle, 2-3 rooms per side separated by 0.2 m partitions,
0.3 m exterior walls, one 0.9 x 2.1 m door per room onto the corridor, one 1.2 x 1.0 m window (sill 0.9 m) per room on the exterior
wall, a 2.0 x 2.4 m stair opening in each slab above the ground (at a corridor end), and 2-5 boxes (furniture, 0.4-1.4 m footprint,
0.4-1.0 m tall) per room. Occupancy = union of all solids; everything else (including outside the building, reached through windows)
is free. Rasterisation at SIZE in {64, 128, 256} (voxel 0.2 / 0.1 / 0.05 m), cell-centre inclusion, same convention as v293.
Sensing: each agent takes POSES = 3 poses jittered +-0.1 m around a free standing point (sensor 1.2 m above its storey's slab),
BEAMS = 16384 first-hit rays on a Fibonacci sphere (0.14 m spacing at 5 m), strict 5 m range; the private map is the ternary union over poses.
Radio: 3D distance <= 8 m (floors do not block radio); the communication forest is the minimum-distance spanning forest over radio edges.
Public blocks: 8 x 8 x 4 (1.6 m x 1.6 m x one storey) => 256 coordinates per agent.
Splits by index: TRAIN index < 1_000_000, VALIDATION index in [1_000_000, 2_000_000), TEST never generated.
"""
import hashlib, math
import numpy as np

VERSION = 400
EXTENT = 12.8
FLOORS = 4
STOREY = EXTENT / FLOORS
SIZES = (64, 128, 256)
SENSOR_RANGE = 5.0
RADIO_RADIUS = 8.0
BEAMS = 16384     # review v400: 4096 beams space 0.28 m apart at 5 m (> the 0.2 m voxel); 16384 -> 0.14 m
POSES = 3
BLOCKS = (8, 8, 4)
SENSOR_HEIGHT = 1.2


def _seed(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'little')


def world_spec(split, index):
    split = str(split).upper()
    if split not in ('TRAIN', 'VALIDATION'):
        raise ValueError('Only TRAIN/VALIDATION, never TEST')
    index = int(index)
    identity = f'v400:{split}:{index}:storeys:{FLOORS}'
    rng = np.random.default_rng(_seed(identity))
    E, W = EXTENT, 0.3
    solids = []                                                   # axis-aligned boxes [x0,y0,z0,x1,y1,z1]
    holes = []                                                    # subtracted boxes (doors, windows, stair openings)
    furniture = []
    rooms = []
    # exterior walls
    solids += [[0, 0, 0, W, E, E], [E - W, 0, 0, E, E, E], [0, 0, 0, E, W, E], [0, E - W, 0, E, E, E]]
    cw = float(rng.uniform(1.6, 2.0)); cx0 = E / 2 - cw / 2; cx1 = E / 2 + cw / 2
    for f in range(FLOORS):
        z0 = f * STOREY
        solids.append([0, 0, z0, E, E, z0 + W])                                   # slab (ground for f = 0)
        if f > 0:                                                                # stair opening in the slab at a corridor end
            yend = rng.choice([W, E - W - 2.4]); solids_hole = [cx0 + .1, yend, z0 - .01, cx0 + 2.1, yend + 2.4, z0 + W - .01]   # review M1: do not clip the wall row above the slab
            holes.append(solids_hole)
        zt = z0 + W; zc = z0 + STOREY                                            # floor top, ceiling (next slab bottom)
        for side, (rx0, rx1) in enumerate(((W, cx0), (cx1, E - W))):
            n_rooms = int(rng.integers(2, 4))
            cuts = np.sort(rng.uniform(W + 1.5, E - W - 1.5, n_rooms - 1)) if n_rooms > 1 else np.array([])
            edges = np.r_[W, cuts, E - W]
            for k in range(n_rooms):
                y0, y1 = float(edges[k]), float(edges[k + 1])
                if k > 0:
                    solids.append([rx0, y0 - .1, zt, rx1, y0 + .1, zc])          # partition
                wall_x = (cx0 - .2, cx0) if side == 0 else (cx1, cx1 + .2)
                solids.append([wall_x[0], y0, zt, wall_x[1], y1, zc])            # corridor wall
                dy = float(rng.uniform(y0 + .3, max(y0 + .3, y1 - 1.2)))
                holes.append([wall_x[0] - .01, dy, zt - .01, wall_x[1] + .01, dy + .9, zt + 2.1])    # door
                if y1 - y0 > 2.0:
                    wy = float(rng.uniform(y0 + .4, y1 - 1.6))
                    ex = (0 - .01, W + .01) if side == 0 else (E - W - .01, E + .01)
                    holes.append([ex[0], wy, zt + .9, ex[1], wy + 1.2, zt + 1.9])                      # window
                inner = [rx0 + (W if side == 0 else .2) + .05, y0 + .15, rx1 - (.2 if side == 0 else W) - .05, y1 - .15]
                rooms.append(dict(floor=f, side=side, rect=inner, z=(zt, zc)))
                for _ in range(int(rng.integers(2, 6))):
                    bw, bd = rng.uniform(.4, 1.4, 2); bh = float(rng.uniform(.4, 1.0))
                    if inner[2] - inner[0] < bw + .6 or inner[3] - inner[1] < bd + .6:
                        continue
                    bx = float(rng.uniform(inner[0] + .3, inner[2] - .3 - bw)); by = float(rng.uniform(inner[1] + .3, inner[3] - .3 - bd))
                    furniture.append([bx, by, zt, bx + bw, by + bd, zt + bh])
    return dict(version=VERSION, split=split, index=index, world_id=identity, extent_m=[E, E, E], floors=FLOORS, storey_m=STOREY,
                corridor_x=[cx0, cx1], solids=solids, holes=holes, furniture=furniture, rooms=rooms)


def _fill(grid, box, value, size):
    cell = EXTENT / size
    # cell-centre inclusion: voxel i is inside [b0, b1) iff b0 <= (i + .5) cell < b1  <=>  ceil(b0/cell - .5) <= i < ceil(b1/cell - .5)
    # (review v400 M1: the previous floor(x + 1) form skipped the voxel layer whose centre lies exactly on b0, opening 0.2 m gaps at 64^3)
    lo = np.clip(np.ceil(np.asarray(box[:3]) / cell - .5 - 1e-9), 0, size).astype(int)
    hi = np.clip(np.ceil(np.asarray(box[3:]) / cell - .5 - 1e-9), 0, size).astype(int)
    if np.all(hi > lo):
        grid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = value


def rasterize(spec, size):
    if size not in SIZES:
        raise ValueError('native size required')
    occ = np.zeros((size, size, size), np.uint8)                   # index order [x, y, z]
    for b in spec['solids']:
        _fill(occ, b, 1, size)
    for b in spec['holes']:
        _fill(occ, b, 0, size)
    for b in spec['furniture']:
        _fill(occ, b, 1, size)
    return occ


def skeleton_spec():
    """the world-independent skeleton: four exterior walls and FLOORS full slabs (without stair openings)."""
    E, W = EXTENT, 0.3
    solids = [[0, 0, 0, W, E, E], [E - W, 0, 0, E, E, E], [0, 0, 0, E, W, E], [0, E - W, 0, E, E, E]] + [[0, 0, f * STOREY, E, E, f * STOREY + W] for f in range(FLOORS)]
    return dict(solids=solids, holes=[], furniture=[])


def skeleton_mask(size):
    return rasterize(skeleton_spec(), size).astype(bool)


def surface_mask(occ):
    """occupied voxels with at least one free 6-neighbour (the informative part of thick solids)."""
    o = occ.astype(bool); free = ~o
    nb = np.zeros_like(o)
    nb[1:] |= free[:-1]; nb[:-1] |= free[1:]; nb[:, 1:] |= free[:, :-1]; nb[:, :-1] |= free[:, 1:]; nb[:, :, 1:] |= free[:, :, :-1]; nb[:, :, :-1] |= free[:, :, 1:]
    return o & nb


def fibonacci_sphere(n):
    i = np.arange(n) + .5
    phi = np.arccos(1 - 2 * i / n); theta = math.pi * (1 + 5 ** .5) * i
    return np.stack((np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)), 1)


def agent_positions(spec, agents, rng, occ_fine):
    """free standing points: inside a room or the corridor, sensor 1.2 m above the slab, >= 0.4 m clearance in the fine raster."""
    size = occ_fine.shape[0]; cell = EXTENT / size
    pts = []
    tries = 0
    while len(pts) < agents and tries < 10000:
        tries += 1
        f = int(rng.integers(0, FLOORS)); zt = f * STOREY + .3
        if rng.random() < .25:
            x = float(rng.uniform(spec['corridor_x'][0] + .3, spec['corridor_x'][1] - .3)); y = float(rng.uniform(.6, EXTENT - .6))
        else:
            room = spec['rooms'][int(rng.integers(0, len(spec['rooms'])))]
            r = room['rect']
            if room['floor'] != f or r[2] - r[0] < 1.0 or r[3] - r[1] < 1.0:        # skip rooms too narrow to stand in with 0.4 m clearance
                continue
            x = float(rng.uniform(r[0] + .4, r[2] - .4)); y = float(rng.uniform(r[1] + .4, r[3] - .4))
        p = np.array([x, y, zt + SENSOR_HEIGHT])
        c = np.floor(p / cell).astype(int); m = max(1, int(round(.4 / cell)))
        lo = np.clip(c - m, 0, size - 1); hi = np.clip(c + m + 1, 1, size)
        if occ_fine[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].any():
            continue
        if any(np.linalg.norm(p[:2] - q[:2]) < 1.0 and abs(p[2] - q[2]) < 1. for q in pts):
            continue
        pts.append(p)
    if len(pts) < agents:
        raise RuntimeError('could not place agents')
    return np.stack(pts)


def observe(occ, poses, rays=None, step_frac=0.5):
    """first-hit ray casting; poses [P,3] metres (same agent) -> ternary union map int8 (-1 unknown, 0 free, 1 occupied)."""
    size = occ.shape[0]; cell = EXTENT / size
    rays = fibonacci_sphere(BEAMS) if rays is None else rays
    out = np.full(occ.shape, -1, np.int8)
    ds = cell * step_frac; nsteps = int(math.ceil(SENSOR_RANGE / ds))
    for p in poses:
        alive = np.ones(len(rays), bool)
        for s in range(1, nsteps + 1):
            pos = p[None] + rays * (s * ds)
            idx = np.floor(pos / cell).astype(int)
            inside = np.all((idx >= 0) & (idx < size), 1) & alive
            if not inside.any():
                break
            ii = idx[inside]
            hit = occ[ii[:, 0], ii[:, 1], ii[:, 2]] == 1
            # free along the way, occupied at the hit; distance mask is exact because s*ds <= range
            free_idx = ii[~hit]; out[free_idx[:, 0], free_idx[:, 1], free_idx[:, 2]] = np.maximum(out[free_idx[:, 0], free_idx[:, 1], free_idx[:, 2]], 0)
            hit_idx = ii[hit]; out[hit_idx[:, 0], hit_idx[:, 1], hit_idx[:, 2]] = 1
            alive_idx = np.nonzero(inside)[0]; alive[alive_idx[hit]] = False
            alive &= inside | ~alive
    return out


def radio_forest(positions):
    """radio adjacency (3D distance <= RADIO_RADIUS) and its minimum-distance spanning forest (Kruskal). edges: (i, j, weight)."""
    n = len(positions); d = np.linalg.norm(positions[:, None] - positions[None], axis=-1)
    adj = (d <= RADIO_RADIUS); np.fill_diagonal(adj, False)
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    edges = []
    for i, j in sorted(((i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j]), key=lambda e: d[e[0], e[1]]):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj; edges.append((i, j, float(d[i, j])))
    return adj, edges


def forest_reached(edges, n, hops):
    """reached[i, s] = True if source s is within `hops` tree edges of node i (hops = 0 -> self only)."""
    import collections
    nb = collections.defaultdict(list)
    for i, j, _ in edges:
        nb[i].append(j); nb[j].append(i)
    reached = np.zeros((n, n), bool)
    for i in range(n):
        dist = {i: 0}; q = collections.deque([i])
        while q:
            u = q.popleft()
            if dist[u] >= hops:
                continue
            for v in nb[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1; q.append(v)
        for s in dist:
            reached[i, s] = True
    return reached


def make_fixture(split, index, size=64, agents=8):
    spec = world_spec(split, index)
    occ = rasterize(spec, size)
    rng = np.random.default_rng(_seed(spec['world_id'] + f':poses:{agents}'))
    centres = agent_positions(spec, agents, rng, occ)
    jitter = rng.uniform(-.1, .1, (POSES, agents, 3)); jitter[:, :, 2] = 0.
    trajectory = centres[None] + jitter                                                 # [P,N,3]
    rays = fibonacci_sphere(BEAMS)
    private = np.stack([observe(occ, trajectory[:, a], rays) for a in range(agents)])   # [N,S,S,S] int8
    adj, edges = radio_forest(centres)
    hops = (0, 1, 2, 4, 8)
    reached = np.stack([forest_reached(edges, agents, h) for h in hops])               # [H,N,N]
    return dict(spec=dict(split=split, index=index, size=size, agents=agents, version=VERSION), world_id=spec['world_id'], occ=occ,
                surface=surface_mask(occ), private=private, positions_m=centres, trajectory_m=trajectory, adjacency=adj, edges=edges,
                reached=reached, hops=hops)


def array_hash(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256(str((str(a.dtype), a.shape)).encode()); h.update(a.tobytes()); return h.hexdigest()


def scene_plan(split, count, agents=8):
    base = 0 if str(split).upper() == 'TRAIN' else 1_000_000
    return [dict(split=str(split).upper(), index=base + i, size=64, agents=agents) for i in range(count)]
