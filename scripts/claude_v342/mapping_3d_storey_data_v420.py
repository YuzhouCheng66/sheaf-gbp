"""3D single-storey world family v420 (Claude, 2026-10-25): furnished multi-room storeys built from HouseExpo floor plans.
Design: claude_handoff/DESIGN_3D_STOREY_V420.md.

Geometry (metres, deterministic from (split, index, rooms) through the HouseExpo v365 index; TEST is never generated):
  * chart: the v371 house-fitted square L = max(8, ceil(max bbox extent + 1 m)) horizontally (house centred), H = 3.2 m vertically;
    the raster is NX x NY x NZ voxels with index order [x, y, z] (R1 = 128 x 128 x 32: cells L/128 horizontally, 0.1 m vertically);
  * walls: the complement of the HouseExpo free contour (v371 conservative raster: a column is free only if all 4 x 4 supersamples
    are inside the contour), extruded over the whole height;
  * floor slab z < 0.1 m, ceiling slab from 0.1 + H_w (H_w ~ U(2.4, 3.0) per house) upwards (solid above the ceiling);
  * door lintels: the v365 opening mask (openings 0.4-1.5 m wide through walls <= 0.6 m thick) is solid from 0.1 + h_d upwards,
    h_d ~ U(2.0, 2.15);
  * furniture: a small parametric library of axis-aligned boxes and vertical cylinders chosen by HouseExpo room category (beds with
    headboards, wardrobes, nightstands, sofas with backs and arms, coffee and dining tables with thin legs, chairs with backs,
    kitchen counters with overhanging wall cabinets, fridges, toilets, bathtubs, sinks, desks, bookshelves with thin shelves, lamps,
    plants, cars, boxes), up to one object attempt per 1.5 m^2 of room box, placed against a room side or free-standing, rejected when it overlaps
    walls or other objects, comes within 0.6 m (Euclidean) of an opening of the 512 or 128 raster, or splits the remaining free floor
    (largest component < 95 % of it with a 0.3 m walking margin around the object); placement and the connectivity rule act inside the
    house's largest free component; a wall-backed object needs >= 50 % wall within 4 cm behind its back; houses without room boxes get
    one generic room; parts thinner than a cell keep the cell containing their centre (review v420 M3 / S1-S3 / S10). Agents are placed
    on the R1 raster at every resolution and are >= 0.5 m apart (else the scene is skipped).
  Occupancy = walls | slabs | lintels | furniture; everything else inside the chart is free. Cell-centre inclusion as in v400.
Sensing, agents and radio are v400 / v401 unchanged: N = 8 agents on free floor with >= 0.45 m clearance from walls and furniture
footprints (largest free component), 3 poses jittered +-0.1 m, sensor 1.2 m above the floor top, first-hit rays on a Fibonacci sphere
(65,536 beams: 0.069 m apart at the 5 m range), strict 5 m range, ray marching at half the smallest cell; 8 m radio, Kruskal forest.
Public blocks 8 x 8 x 4 (L/8 x L/8 x 0.8 m) => D = 256 coordinates per agent.
"""
import hashlib, json, math, os
import numpy as np
from scipy import ndimage

import mapping_houseexpo_data_v365 as v365
import mapping_houseexpo_data_v371 as v371

VERSION = 420
HEIGHT_M = 3.2
FLOOR_T = 0.10
SENSOR_RANGE = 5.0
RADIO_RADIUS = 8.0
POSES = 3
SENSOR_HEIGHT = 1.2
AGENTS = 8
BLOCKS = (8, 8, 4)
BEAMS = 65536
DIMS = {'R1': (128, 128, 32), 'R2': (256, 256, 64)}
HOPS = (0, 1, 2, 4, 8)


def _seed(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'little')


# ----------------------------------------------------------------------------- furniture library (primitives in metres)
# a primitive is ('box', x0, y0, z0, x1, y1, z1) or ('cyl', cx, cy, r, z0, z1); z is height above the floor top
def _legs(x0, y0, x1, y1, h, t=0.05):
    return [('box', x0, y0, 0., x0 + t, y0 + t, h), ('box', x1 - t, y0, 0., x1, y0 + t, h), ('box', x0, y1 - t, 0., x0 + t, y1, h), ('box', x1 - t, y1 - t, 0., x1, y1, h)]


def _table(x0, y0, x1, y1, h, top=0.03):
    return [('box', x0, y0, h - top, x1, y1, h)] + _legs(x0, y0, x1, y1, h - top)


def _chair(x0, y0, x1, y1, back_side, rng):
    seat = float(rng.uniform(0.42, 0.48)); back = float(rng.uniform(0.80, 0.95)); t = 0.05
    out = [('box', x0, y0, seat - 0.04, x1, y1, seat)] + _legs(x0, y0, x1, y1, seat - 0.04, 0.04)
    if back_side == 'x0': out.append(('box', x0, y0, seat, x0 + t, y1, back))
    elif back_side == 'x1': out.append(('box', x1 - t, y0, seat, x1, y1, back))
    elif back_side == 'y0': out.append(('box', x0, y0, seat, x1, y0 + t, back))
    else: out.append(('box', x0, y1 - t, seat, x1, y1, back))
    return out


def _against(side, rect, depth, width, rng):
    """footprint of an object of (depth, width) standing against `side` of the room rect, slid to a random position along it."""
    rx0, ry0, rx1, ry1 = rect
    if side in ('y0', 'y1'):
        if rx1 - rx0 < width: return None
        a = float(rng.uniform(rx0, rx1 - width))
        return (a, ry0, a + width, ry0 + depth) if side == 'y0' else (a, ry1 - depth, a + width, ry1)
    if ry1 - ry0 < width: return None
    a = float(rng.uniform(ry0, ry1 - width))
    return (rx0, a, rx0 + depth, a + width) if side == 'x0' else (rx1 - depth, a, rx1, a + width)


def _wall_side_of(fp, side):
    return {'y0': 'y0', 'y1': 'y1', 'x0': 'x0', 'x1': 'x1'}[side]


def _make(kind, fp, side, rng):
    """primitives of one object of `kind` with footprint fp = (x0, y0, x1, y1), backed against `side` (or None if free-standing)."""
    x0, y0, x1, y1 = fp
    if kind == 'bed':
        h = float(rng.uniform(0.45, 0.6)); out = [('box', x0, y0, 0., x1, y1, h)]
        hb = {'y0': ('box', x0, y0, 0., x1, y0 + 0.08, 1.0), 'y1': ('box', x0, y1 - 0.08, 0., x1, y1, 1.0), 'x0': ('box', x0, y0, 0., x0 + 0.08, y1, 1.0), 'x1': ('box', x1 - 0.08, y0, 0., x1, y1, 1.0)}
        return out + [hb[side]]
    if kind in ('wardrobe', 'fridge', 'cabinet_tall'):
        return [('box', x0, y0, 0., x1, y1, float(rng.uniform(1.8, 2.1)))]
    if kind in ('nightstand', 'cabinet', 'tv_cabinet', 'sink', 'box', 'shoe_cabinet'):
        hmap = dict(nightstand=(0.45, 0.6), cabinet=(0.75, 0.95), tv_cabinet=(0.4, 0.55), sink=(0.8, 0.9), box=(0.3, 0.8), shoe_cabinet=(0.9, 1.1))
        return [('box', x0, y0, 0., x1, y1, float(rng.uniform(*hmap[kind])))]
    if kind == 'sofa':
        seat = float(rng.uniform(0.40, 0.48)); back = float(rng.uniform(0.80, 0.95)); arm = float(rng.uniform(0.55, 0.65)); bt = 0.2; at = 0.15
        out = [('box', x0, y0, 0., x1, y1, seat)]
        if side in ('y0', 'y1'):
            out.append(('box', x0, y0, 0., x1, y0 + bt, back) if side == 'y0' else ('box', x0, y1 - bt, 0., x1, y1, back))
            out += [('box', x0, y0, 0., x0 + at, y1, arm), ('box', x1 - at, y0, 0., x1, y1, arm)]
        else:
            out.append(('box', x0, y0, 0., x0 + bt, y1, back) if side == 'x0' else ('box', x1 - bt, y0, 0., x1, y1, back))
            out += [('box', x0, y0, 0., x1, y0 + at, arm), ('box', x0, y1 - at, 0., x1, y1, arm)]
        return out
    if kind == 'coffee_table':
        return _table(x0, y0, x1, y1, float(rng.uniform(0.40, 0.48)), 0.04)
    if kind in ('dining_table', 'desk'):
        return _table(x0, y0, x1, y1, float(rng.uniform(0.72, 0.78)))
    if kind == 'counter':                                                       # base cabinets + overhanging wall cabinets
        out = [('box', x0, y0, 0., x1, y1, 0.9)]
        d = 0.35
        uc = {'y0': ('box', x0, y0, 1.45, x1, y0 + d, 2.2), 'y1': ('box', x0, y1 - d, 1.45, x1, y1, 2.2), 'x0': ('box', x0, y0, 1.45, x0 + d, y1, 2.2), 'x1': ('box', x1 - d, y0, 1.45, x1, y1, 2.2)}
        return out + [uc[side]]
    if kind == 'toilet':
        out = [('box', x0, y0, 0., x1, y1, 0.42)]
        tk = {'y0': ('box', x0, y0, 0., x1, y0 + 0.2, 0.8), 'y1': ('box', x0, y1 - 0.2, 0., x1, y1, 0.8), 'x0': ('box', x0, y0, 0., x0 + 0.2, y1, 0.8), 'x1': ('box', x1 - 0.2, y0, 0., x1, y1, 0.8)}
        return out + [tk[side]]
    if kind == 'bathtub':
        return [('box', x0, y0, 0., x1, y1, float(rng.uniform(0.5, 0.6)))]
    if kind == 'bookshelf':                                                     # side panels + thin shelves every ~0.4 m
        h = float(rng.uniform(1.6, 2.0)); t = 0.03
        out = []
        if side in ('y0', 'y1'):
            out += [('box', x0, y0, 0., x0 + t, y1, h), ('box', x1 - t, y0, 0., x1, y1, h)]
            out.append(('box', x0, y0, 0., x1, y0 + t, h) if side == 'y0' else ('box', x0, y1 - t, 0., x1, y1, h))
        else:
            out += [('box', x0, y0, 0., x1, y0 + t, h), ('box', x0, y1 - t, 0., x1, y1, h)]
            out.append(('box', x0, y0, 0., x0 + t, y1, h) if side == 'x0' else ('box', x1 - t, y0, 0., x1, y1, h))
        out += [('box', x0, y0, z, x1, y1, z + t) for z in np.arange(0.0, h - 0.05, 0.4)]
        return out
    if kind == 'lamp':
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; top = float(rng.uniform(1.4, 1.7))
        return [('cyl', cx, cy, 0.15, 0., 0.03), ('cyl', cx, cy, 0.02, 0., top), ('cyl', cx, cy, 0.2, top, top + 0.3)]
    if kind == 'plant':
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        return [('cyl', cx, cy, 0.18, 0., 0.35), ('cyl', cx, cy, float(rng.uniform(0.2, 0.35)), 0.35, float(rng.uniform(0.8, 1.5)))]
    if kind == 'round_table':
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; r = (x1 - x0) / 2; h = float(rng.uniform(0.72, 0.76))
        return [('cyl', cx, cy, r, h - 0.03, h), ('cyl', cx, cy, 0.05, 0., h - 0.03), ('cyl', cx, cy, 0.25, 0., 0.03)]
    if kind == 'car':
        return [('box', x0, y0, 0.15, x1, y1, float(rng.uniform(1.35, 1.55))), ('box', x0 + 0.2, y0 + 0.2, 0., x0 + 0.5, y0 + 0.5, 0.15), ('box', x1 - 0.5, y1 - 0.5, 0., x1 - 0.2, y1 - 0.2, 0.15)]
    raise ValueError(kind)


# kind -> (depth, width) ranges and whether it stands against a side
SIZES_KIND = dict(bed=((1.9, 2.1), (1.2, 1.8), True), wardrobe=((0.55, 0.65), (0.9, 2.0), True), nightstand=((0.4, 0.5), (0.4, 0.5), True),
                  sofa=((0.85, 0.95), (1.6, 2.4), True), coffee_table=((0.5, 0.7), (0.9, 1.2), False), tv_cabinet=((0.4, 0.5), (1.2, 1.8), True),
                  lamp=((0.4, 0.4), (0.4, 0.4), False), plant=((0.5, 0.6), (0.5, 0.6), False), dining_table=((0.8, 1.0), (1.2, 1.8), False),
                  round_table=((0.9, 1.1), (0.9, 1.1), False), counter=((0.58, 0.62), (1.2, 3.0), True), fridge=((0.65, 0.75), (0.6, 0.75), True),
                  toilet=((0.65, 0.75), (0.38, 0.45), True), sink=((0.45, 0.55), (0.6, 0.9), True), bathtub=((0.7, 0.8), (1.5, 1.7), True),
                  desk=((0.6, 0.75), (1.0, 1.6), True), bookshelf=((0.3, 0.38), (0.8, 1.2), True), cabinet=((0.4, 0.6), (0.6, 1.2), True),
                  cabinet_tall=((0.4, 0.6), (0.6, 1.0), True), box=((0.3, 0.7), (0.3, 0.7), False), shoe_cabinet=((0.3, 0.4), (0.8, 1.2), True),
                  car=((1.7, 1.9), (4.2, 4.7), False))
ROOM_KINDS = {
    'Bedroom': ['bed', 'wardrobe', 'nightstand', 'desk', 'plant', 'lamp'], 'Child_Room': ['bed', 'wardrobe', 'desk', 'box', 'bookshelf'],
    'Guest_Room': ['bed', 'wardrobe', 'nightstand'], 'Living_Room': ['sofa', 'coffee_table', 'tv_cabinet', 'lamp', 'plant', 'bookshelf', 'round_table'],
    'Room': ['sofa', 'desk', 'bookshelf', 'cabinet', 'plant', 'box'], 'Hall': ['sofa', 'cabinet', 'plant', 'round_table'], 'Lobby': ['sofa', 'plant', 'cabinet'],
    'Dining_Room': ['dining_table', 'cabinet', 'plant', 'round_table'], 'Kitchen': ['counter', 'counter', 'fridge', 'round_table', 'cabinet_tall'],
    'Bathroom': ['bathtub', 'toilet', 'sink', 'cabinet'], 'Toilet': ['toilet', 'sink'], 'Office': ['desk', 'desk', 'bookshelf', 'cabinet', 'plant'],
    'Garage': ['car', 'cabinet_tall', 'box', 'box'], 'Storage': ['cabinet_tall', 'box', 'box', 'bookshelf'], 'Wardrobe': ['wardrobe', 'wardrobe', 'box'],
    'Gym': ['box', 'box', 'cabinet'], 'Hallway': ['shoe_cabinet', 'plant'], 'Entryway': ['shoe_cabinet', 'plant'],
    'Balcony': ['plant', 'box'], 'Terrace': ['plant', 'round_table'], 'Loggia': ['plant'], 'Boiler_room': ['cabinet_tall', 'box'],
}


DENSITY_M2 = 1.2                                                         # one object attempt per 1.2 m^2 of room box: footprint share median ~.30 = Replica (D8-1)
GENERIC_KINDS = ['cabinet', 'box', 'plant', 'bookshelf', 'sofa', 'coffee_table', 'lamp', 'cabinet_tall', 'desk', 'round_table']


def _shift(p, dx, dy):
    if p[0] == 'box':
        return ('box', p[1] + dx, p[2] + dy, p[3], p[4] + dx, p[5] + dy, p[6])
    return ('cyl', p[1] + dx, p[2] + dy, p[3], p[4], p[5])


def _footprint(prims):
    xs = [p[1] for p in prims if p[0] == 'box'] + [p[1] - p[3] for p in prims if p[0] == 'cyl']
    ys = [p[2] for p in prims if p[0] == 'box'] + [p[2] - p[3] for p in prims if p[0] == 'cyl']
    xe = [p[4] for p in prims if p[0] == 'box'] + [p[1] + p[3] for p in prims if p[0] == 'cyl']
    ye = [p[5] for p in prims if p[0] == 'box'] + [p[2] + p[3] for p in prims if p[0] == 'cyl']
    return min(xs), min(ys), max(xe), max(ye)


def world_spec(split, index, rooms=1):
    base = v371.world_spec(split, index, rooms)                       # house-fitted chart, verts in [0, L]^2, splits / TEST closure
    L = base['chart_m']
    d = json.load(open(os.path.join(os.environ['HOUSEEXPO_JSON_DIR'], base['house_id'] + '.json')))
    v = np.asarray(d['verts'], np.float64); centre = (v.min(0) + v.max(0)) / 2
    shift = -centre + L / 2
    room_rects = [(cat, (b[0] + shift[0], b[1] + shift[1], b[2] + shift[0], b[3] + shift[1])) for cat, bl in sorted(d.get('room_category', {}).items()) for b in bl]
    world_id = f"v420:{base['split']}:{base['house_id']}"
    rng = np.random.default_rng(_seed(world_id + ':geometry'))
    H_w = float(rng.uniform(2.4, 3.0)); h_d = float(rng.uniform(2.0, 2.15))
    # fine 2D raster (512) for placement: free floor, opening mask
    g512 = v371.rasterize(base, 512); free_all = g512['walls'] == 0; cell = L / 512
    # review v420 S2: Euclidean 0.6 m clearance from the openings of the 512 AND the 128 (R1 lintel) rasters
    door = g512['door_mask'] | np.kron(v371.rasterize(base, 128)['door_mask'], np.ones((4, 4), bool))
    near_door = ndimage.distance_transform_edt(~door) * cell < 0.6
    # review v420 M3 (ii): place furniture, and judge connectivity, inside the largest free component of the house only
    lab, nlab = ndimage.label(free_all)
    free = lab == (1 + int(np.argmax(np.bincount(lab.ravel())[1:]))) if nlab else free_all
    taken = np.zeros_like(free)
    objects = []
    if not room_rects:                                                  # review v420 S10: houses without room boxes get one generic room
        rr, cc = np.nonzero(free)
        if len(rr):
            room_rects = [('Generic', (cc.min() * cell, rr.min() * cell, (cc.max() + 1) * cell, (rr.max() + 1) * cell))]
    for cat, rect_box in room_rects:
        # HouseExpo room boxes include wall thickness: use the tight bounding box of the free fine cells inside the room box
        c0b, r0b = max(0, int(math.floor(rect_box[0] / cell))), max(0, int(math.floor(rect_box[1] / cell)))
        c1b, r1b = min(512, int(math.ceil(rect_box[2] / cell))), min(512, int(math.ceil(rect_box[3] / cell)))
        sub = free[r0b:r1b, c0b:c1b]
        if sub.size == 0 or not sub.any():
            continue
        rows = np.nonzero(sub.any(1))[0]; cols = np.nonzero(sub.any(0))[0]
        rect = ((c0b + cols[0]) * cell, (r0b + rows[0]) * cell, (c0b + cols[-1] + 1) * cell, (r0b + rows[-1] + 1) * cell)
        kinds = ROOM_KINDS.get(cat, GENERIC_KINDS if cat == 'Generic' else ['box', 'plant'])
        area = max(0., (rect[2] - rect[0]) * (rect[3] - rect[1]))
        n = int(min(len(kinds) + 14, max(1, rng.poisson(area / DENSITY_M2))))
        # the room's own kinds first, then a mix (30 % generic household objects: HouseExpo room labels are sometimes coarse)
        extra = [str(rng.choice(GENERIC_KINDS)) if rng.random() < 0.3 else str(rng.choice(kinds)) for _ in range(max(0, n - len(kinds)))]
        order = list(kinds) + extra
        for kind in order[:n]:
            (dlo, dhi), (wlo, whi), against = SIZES_KIND[kind]
            for _ in range(20):                                         # placement attempts
                depth, width = float(rng.uniform(dlo, dhi)), float(rng.uniform(wlo, whi))
                inner = (rect[0] + 0.02, rect[1] + 0.02, rect[2] - 0.02, rect[3] - 0.02)
                if against:
                    side = str(rng.choice(['x0', 'x1', 'y0', 'y1']))
                    fp = _against(side, inner, depth, width, rng)
                else:
                    side = None
                    if inner[2] - inner[0] < width + 0.6 or inner[3] - inner[1] < depth + 0.6: fp = None
                    else:
                        x = float(rng.uniform(inner[0] + 0.3, inner[2] - 0.3 - width)); y = float(rng.uniform(inner[1] + 0.3, inner[3] - 0.3 - depth))
                        fp = (x, y, x + width, y + depth)
                if fp is None:
                    continue
                prims = _make(kind, fp, side if side else 'y0', rng)
                if kind == 'dining_table':                              # chairs around the table, backs outwards
                    x0, y0, x1, y1 = fp; cw = 0.45
                    for k in range(int(rng.integers(2, 4))):
                        cx = x0 + (k + 0.5) * (x1 - x0) / 3 - cw / 2
                        prims += _chair(cx, y0 - 0.5, cx + cw, y0 - 0.05, 'y0', rng) + _chair(cx, y1 + 0.05, cx + cw, y1 + 0.5, 'y1', rng)
                if kind == 'desk' and against:
                    x0, y0, x1, y1 = fp; cw = 0.45
                    if side in ('y0', 'y1'):
                        cx = (x0 + x1) / 2 - cw / 2
                        prims += _chair(cx, y1 + 0.05, cx + cw, y1 + 0.5, 'y1', rng) if side == 'y0' else _chair(cx, y0 - 0.5, cx + cw, y0 - 0.05, 'y0', rng)
                    else:
                        cy = (y0 + y1) / 2 - cw / 2
                        prims += _chair(x1 + 0.05, cy, x1 + 0.5, cy + cw, 'x1', rng) if side == 'x0' else _chair(x0 - 0.5, cy, x0 - 0.05, cy + cw, 'x0', rng)
                # HouseExpo room boxes include the walls: a wall-backed object is pushed inwards (5 cm steps, up to 1.2 m) until its
                # whole footprint is free, i.e. until it actually stands against the wall
                push = {'x0': (1, 0), 'x1': (-1, 0), 'y0': (0, 1), 'y1': (0, -1)}.get(side, (0, 0)) if against else (0, 0)
                ok = False
                for k_push in range(0, 25 if against else 1):
                    dx, dy = push[0] * 0.05 * k_push, push[1] * 0.05 * k_push
                    fx0, fy0, fx1, fy1 = _footprint(prims); fx0 += dx; fx1 += dx; fy0 += dy; fy1 += dy
                    c0, r0 = int(math.floor(fx0 / cell)), int(math.floor(fy0 / cell)); c1, r1 = int(math.ceil(fx1 / cell)), int(math.ceil(fy1 / cell))
                    if c0 < 0 or r0 < 0 or c1 > 512 or r1 > 512:
                        continue
                    win = np.s_[r0:r1, c0:c1]
                    if free[win].all():
                        ok = True; break
                if not ok or taken[win].any() or near_door[win].any():
                    continue
                if against:                                             # review v420 M3 (iii): the back must actually touch a wall
                    back = {'x0': np.s_[r0:r1, max(0, c0 - 2):c0], 'x1': np.s_[r0:r1, c1:min(512, c1 + 2)],
                            'y0': np.s_[max(0, r0 - 2):r0, c0:c1], 'y1': np.s_[r1:min(512, r1 + 2), c0:c1]}[side]
                    strip = free_all[back]
                    if strip.size == 0 or (~strip).mean() < 0.5:
                        continue
                if dx or dy:
                    prims = [_shift(p, dx, dy) for p in prims]
                # connectivity: with this object (and a 0.3 m walking margin around it) the remaining free floor must stay one piece up
                # to 5 % in pockets (relative to the REMAINING floor, so the total furniture share is not capped)
                trial = taken.copy(); m = max(1, int(round(0.3 / cell))); trial[max(0, r0 - m):r1 + m, max(0, c0 - m):c1 + m] = True
                remaining = free & ~trial
                lab, _ = ndimage.label(remaining)
                if lab.max() == 0 or np.bincount(lab.ravel())[1:].max() < 0.95 * remaining.sum():
                    continue
                taken[win] = True; objects.append(dict(kind=kind, room=cat, prims=[list(p) for p in prims]))
                break
    return dict(version=VERSION, split=base['split'], index=int(index), rooms=int(rooms), house_id=base['house_id'], world_id=world_id,
                chart_m=L, height_m=HEIGHT_M, floor_t=FLOOR_T, wall_h=H_w, door_h=h_d, verts_m=base['verts_m'], room_rects=[(c, list(r)) for c, r in room_rects],
                objects=objects, base_spec=base)


def _cells(size_m, n):
    return size_m / n


def _cell_of(v, c, n):
    m = np.zeros(n, bool); m[min(n - 1, max(0, int(v // c)))] = True
    return m


def rasterize(spec, dims):
    """occupancy [NX, NY, NZ] uint8, slab mask (floor + ceiling and above), object mask; index order [x, y, z]."""
    NX, NY, NZ = dims
    L = spec['chart_m']; cx, cz = L / NX, HEIGHT_M / NZ
    g = v371.rasterize(spec['base_spec'], NX)                         # walls / openings at the chart raster, [y, x]
    walls2d = (g['walls'] != 0).T; door2d = g['door_mask'].T          # -> [x, y]
    zc = (np.arange(NZ) + .5) * cz
    floor_k = zc < FLOOR_T; ceil_k = zc >= FLOOR_T + spec['wall_h']; lintel_k = zc >= FLOOR_T + spec['door_h']
    occ = np.zeros((NX, NY, NZ), bool)
    occ |= walls2d[:, :, None]
    occ |= door2d[:, :, None] & lintel_k[None, None, :]
    slab = np.zeros_like(occ); slab[:, :, floor_k] = True; slab[:, :, ceil_k] = True
    occ |= slab
    obj = np.zeros_like(occ)
    xc = (np.arange(NX) + .5) * cx; yc = (np.arange(NY) + .5) * cx
    for o in spec['objects']:
        for p in o['prims']:
            if p[0] == 'box':
                _, x0, y0, z0, x1, y1, z1 = p
                ix = (xc >= x0) & (xc < x1); iy = (yc >= y0) & (yc < y1); iz = (zc >= FLOOR_T + z0) & (zc < FLOOR_T + z1)
                # parts thinner than a cell (legs, shelves, table tops): per axis, keep the cell that contains the part's centre
                ix = ix if ix.any() else _cell_of((x0 + x1) / 2, cx, NX)
                iy = iy if iy.any() else _cell_of((y0 + y1) / 2, cx, NY)
                iz = iz if iz.any() else _cell_of(FLOOR_T + (z0 + z1) / 2, cz, NZ)
                obj[np.ix_(ix, iy, iz)] = True
            else:
                _, ccx, ccy, r, z0, z1 = p
                m2 = (xc[:, None] - ccx) ** 2 + (yc[None, :] - ccy) ** 2 <= r ** 2
                if not m2.any():                                         # review v420 M3 (i): a pole thinner than a cell keeps its own cell
                    m2[_cell_of(ccx, cx, NX), _cell_of(ccy, cx, NY)] = True
                iz = (zc >= FLOOR_T + z0) & (zc < FLOOR_T + z1)
                iz = iz if iz.any() else _cell_of(FLOOR_T + (z0 + z1) / 2, cz, NZ)
                obj |= m2[:, :, None] & iz[None, None, :]
    obj &= ~walls2d[:, :, None]
    occ |= obj
    return dict(occ=occ.astype(np.uint8), slab=slab, obj=obj & ~slab, cell=(cx, cx, cz))


def fibonacci_sphere(n):
    i = np.arange(n) + .5
    phi = np.arccos(1 - 2 * i / n); theta = math.pi * (1 + 5 ** .5) * i
    return np.stack((np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)), 1)


def observe(occ, cell, poses, rays):
    """first-hit ray marching on the voxel grid with anisotropic cells; poses [P,3] metres -> ternary union int8 [NX,NY,NZ]."""
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


def agent_positions(spec, occ_r, cell, agents, rng):
    """free floor columns (no wall, no object below the sensor height) with >= 0.45 m clearance, largest component, >= 1 m apart."""
    cx, _, cz = cell
    zc = (np.arange(occ_r.shape[2]) + .5) * cz
    band = (zc >= FLOOR_T) & (zc < FLOOR_T + SENSOR_HEIGHT + 0.3)
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
    for min_sep in (1.0, 0.5):                                       # review v420 S1: never closer than 0.5 m (else the scene is skipped)
        pts = []
        for i in order:
            p = np.array([(cand[i][0] + .5) * cx, (cand[i][1] + .5) * cx])
            if all(np.linalg.norm(p - q) >= min_sep for q in pts):
                pts.append(p)
            if len(pts) == agents:
                return np.array([[q[0], q[1], FLOOR_T + SENSOR_HEIGHT] for q in pts])
    raise ValueError('could not place agents')


def radio_forest(positions):
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
            for w in nb[u]:
                if w not in dist:
                    dist[w] = dist[u] + 1; q.append(w)
        for s in dist:
            reached[i, s] = True
    return reached


def make_fixture(split, index, rooms=1, dims=DIMS['R1'], agents=AGENTS):
    spec = world_spec(split, index, rooms)
    r = rasterize(spec, tuple(dims)); occ = r['occ']; cell = r['cell']
    rng = np.random.default_rng(_seed(spec['world_id'] + f':poses:{agents}'))
    # review v420 S3: agents are placed on the R1 reference raster at every resolution (identical positions for paired R1 / R2 studies)
    ref = r if tuple(dims) == DIMS['R1'] else rasterize(spec, DIMS['R1'])
    centres = agent_positions(spec, ref['occ'].astype(bool), ref['cell'], agents, rng)
    jitter = rng.uniform(-.1, .1, (POSES, agents, 3)); jitter[:, :, 2] = 0.
    trajectory = centres[None] + jitter
    rays = fibonacci_sphere(BEAMS)
    private = np.stack([observe(occ, cell, trajectory[:, a], rays) for a in range(agents)])
    adj, edges = radio_forest(centres)
    reached = np.stack([forest_reached(edges, agents, h) for h in HOPS])
    return dict(spec=dict(split=split, index=int(index), rooms=int(rooms), dims=list(dims), agents=agents, version=VERSION), world_id=spec['world_id'],
                chart_m=spec['chart_m'], occ=occ, slab=r['slab'], obj=r['obj'], cell=cell, private=private, positions_m=centres, trajectory_m=trajectory,
                adjacency=adj, edges=edges, reached=reached, hops=HOPS, n_objects=len(spec['objects']), wall_h=spec['wall_h'])


def array_hash(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256(str((str(a.dtype), a.shape)).encode()); h.update(a.tobytes()); return h.hexdigest()


def scene_plan(split, per_stratum, dims=DIMS['R1']):
    """TRAIN: per_stratum houses of each HouseExpo stratum (rooms 1/2/3); VALIDATION: indices 0..per_stratum-1 of each stratum."""
    split = str(split).upper()
    if split not in ('TRAIN', 'VALIDATION'):
        raise ValueError('TEST is closed')
    return [dict(split=split, index=i, rooms=s, dims=list(dims), agents=AGENTS) for s in (1, 2, 3) for i in range(per_stratum)]
