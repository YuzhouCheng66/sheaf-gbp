"""HouseExpo world family v378 = v371 with ONE change: the house-fitted public chart is a RECTANGLE (Claude, 2026-10-23).

Motivation (D4-8/D4-10): code capacity per house area is the binding limit (house-fitted square chart: +10 pp; free-code ceiling .694 ->
.819; 16-bit block scalars do not help, the oracle-union fusion loses nothing). On the square chart a house's bounding box still
covers only ~52-63 % of the chart (median by stratum; HouseExpo aspect ratio median ~1.3), so ~40 % of the 64 blocks describe
exterior. Here each axis is fitted separately:
    Lx = max(8, ceil(extent_x + 1 m)),  Ly = max(8, ceil(extent_y + 1 m)),  then the shorter side is raised to >= half the longer
    (aspect <= 2), house centred; the same 8 x 8 block grid and native H x H raster cover the chart, so blocks are Lx/8 x Ly/8 metres
    and raster cells are anisotropic (Lx/H x Ly/H).
Everything metric is unchanged: 256 first-hit beams uniformly spaced in METRIC angle with a strict 5 m metric range (Amanatides-Woo
traversal on the anisotropic grid, the exact generalisation of v121 make_rays -- identical offsets when Lx = Ly, unit-tested), the
metric 5 m disk mask on cell centres, 8 m radio, opening mask with metric widths per axis. POSES ARE PAIRED with v371 (review M1):
the v371 square-chart poses of the same house / N translated into the rectangle (identical physical poses, radio graph and forest). Same splits / strata / index / TEST exclusion as v365/v371. With square=True every function reproduces v371
exactly (test_v378_rect.py).
Fixture contract identical to v371 except public_frame.extent_m = [Lx, Ly] and cell_size_m = [Lx/H, Ly/H].
"""
import math
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

import mapping_houseexpo_data_v365 as v365
import mapping_houseexpo_data_v371 as v371
from scripts import mapping_multiroom_data_v293 as d293
from scripts.prepare_houseexpo_pilot_v121 import raycast_observe, radius_adjacency

VERSION = 378
SIZES = v365.SIZES
AGENT_COUNTS = v365.AGENT_COUNTS
STRATA = v365.STRATA
INPUT_FIELDS = v365.INPUT_FIELDS
SENSOR_RANGE = d293.SENSOR_RANGE
RADIO_RADIUS = d293.RADIO_RADIUS
array_hash = v365.array_hash
scene_plan = v365.scene_plan


def chart_rule(ext_x, ext_y):
    lx = float(max(8., math.ceil(ext_x + 1.0))); ly = float(max(8., math.ceil(ext_y + 1.0)))
    if lx < ly / 2:
        lx = float(math.ceil(ly / 2))
    if ly < lx / 2:
        ly = float(math.ceil(lx / 2))
    return lx, ly


def world_spec(split, index, rooms=1, square=False):
    spec = v365.world_spec(split, index, rooms)                    # verts centred in a 20 m chart
    v = np.asarray(spec['verts_m']) - v365.EXTENT / 2              # centred at the origin
    ext = np.ptp(v, 0)
    if square:                                                     # v371 rule (equivalence tests only)
        L = float(max(8., math.ceil(float(ext.max()) + 1.0))); lx = ly = L
    else:
        lx, ly = chart_rule(float(ext[0]), float(ext[1]))
    spec.update(version=VERSION, world_id=spec['world_id'].replace('v365:', 'v378:' if not square else 'v371:'), chart_xy_m=[lx, ly],
                verts_m=(v + np.array([lx / 2, ly / 2])).tolist(),
                chart_rule='Lx = max(8, ceil(ext_x + 1)), Ly = max(8, ceil(ext_y + 1)), aspect <= 2; house centred; 8x8 blocks of Lx/8 x Ly/8 m' if not square else 'v371 square')
    return spec


def narrow_mask(walls, cx, cy, lo=.4, hi=1.5, max_thick=.6):
    """v365.narrow_mask with metric run widths per axis: runs along x (rows) use cx and thickness across them cy; runs along y use cy
    and thickness cx. Identical to v365.narrow_mask(walls, c) when cx == cy == c."""
    free = walls == 0
    out = np.zeros_like(free)
    for transpose in (False, True):
        arr = free.T if transpose else free
        run_cell, thick_cell = (cy, cx) if transpose else (cx, cy)
        m = np.zeros_like(arr)
        for r in range(arr.shape[0]):
            row = arr[r]; c = 0; n = len(row)
            while c < n:
                if not row[c]:
                    c += 1; continue
                start = c
                while c < n and row[c]:
                    c += 1
                if start > 0 and c < n and lo <= (c - start) * run_cell <= hi:
                    m[r, start:c] = True
        lab, k = ndimage.label(m)
        for sl_i, sl in enumerate(ndimage.find_objects(lab), 1):
            if sl is not None and (sl[0].stop - sl[0].start) * thick_cell <= max_thick:
                out |= ((lab == sl_i).T if transpose else (lab == sl_i))
    return out


def rasterize(spec, size, supersample=4):
    if size not in SIZES and size != 512:
        raise ValueError('native size required')
    lx, ly = spec['chart_xy_m']; cx, cy = lx / size, ly / size
    pix = np.asarray(spec['verts_m']) / np.array([cx, cy]) * supersample
    img = Image.new('L', (size * supersample, size * supersample), 0)
    ImageDraw.Draw(img).polygon([tuple(p) for p in pix], fill=1)
    s = np.asarray(img, dtype=np.uint8)
    free = s.reshape(size, supersample, size, supersample).min(axis=(1, 3)) != 0
    walls = (~free).astype(np.int8)
    return dict(walls=walls, door_mask=narrow_mask(walls, cx, cy))


def metric_trajectory(spec, agents=8):
    """PAIRED design (review code_review_v378_rect.md M1): the poses are the v371 square-chart poses of the same split / house / N,
    translated into the rectangular chart. The house is centred in both charts, so this is the identical physical pose; radio graph
    and forest are therefore identical to v371, and houses whose chart does not change (Lx = Ly = L) get bit-identical fixtures."""
    sq = v371.world_spec(spec['split'], spec['index'], spec['stratum'])
    L = sq['chart_m']; lx, ly = spec['chart_xy_m']
    return v371.metric_trajectory(sq, agents) + np.array([lx / 2 - L / 2, ly / 2 - L / 2])


def metric_trajectory_resampled(spec, agents=8):
    """(unpaired variant, NOT used: re-draws poses on the rectangular chart; kept for the record of the first review)"""
    if agents not in AGENT_COUNTS:
        raise ValueError('agents')
    rng = np.random.default_rng(v365._seed(spec['world_id'] + f':poses:{agents}'))
    lx, ly = spec['chart_xy_m']; cx, cy = lx / 512, ly / 512
    fine = rasterize(spec, 512)['walls'] == 0
    lab, nlab = ndimage.label(fine)
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, nlab + 1))
    comp = lab == (1 + int(np.argmax(sizes)))
    clearance = ndimage.distance_transform_edt(fine, sampling=(cy, cx)) if cx != cy else ndimage.distance_transform_edt(fine) * cx
    cand = np.argwhere(comp & (clearance >= .45))
    if len(cand) < agents:
        raise ValueError('not enough free space with clearance')
    order = rng.permutation(len(cand)); centres = []
    for min_sep in (1.0, .5, 0.):
        centres = []
        for i in order:
            rc = cand[i]; p = np.array([(rc[1] + .5) * cx, (rc[0] + .5) * cy])
            if all(np.linalg.norm(p - q) >= min_sep for q in centres):
                centres.append(p)
            if len(centres) == agents:
                break
        if len(centres) == agents:
            break
    centres = np.asarray(centres)
    deltas = rng.uniform(-.10, .10, (agents, 2))
    return centres[None] + np.array([-1., 0., 1.])[:, None, None] * deltas[None]


def make_rays_aniso(range_m, beams, cy, cx):
    """v121 make_rays generalised to anisotropic cells: beam angles uniform in METRIC space, traversal parameter t in metres
    (row boundaries every cy / |sin|, column boundaries every cx / |cos|), stop when t > range_m. For cx == cy == c it performs exactly
    the iterations of make_rays(range_m / c) (all parameters scaled by c, tie tolerance scaled by c)."""
    tol = 1e-10 * min(cx, cy)
    rays = []
    for angle in np.arange(beams) * (2 * np.pi / beams):
        dr, dc = float(np.sin(angle)), float(np.cos(angle))
        sr, sc = int(np.sign(dr)), int(np.sign(dc))
        ar, ac = abs(dr) / cy, abs(dc) / cx                       # cells per metre
        tr = .5 / ar if abs(dr) > 1e-12 else math.inf
        tc = .5 / ac if abs(dc) > 1e-12 else math.inf
        dtr = 1 / ar if abs(dr) > 1e-12 else math.inf
        dtc = 1 / ac if abs(dc) > 1e-12 else math.inf
        row = col = 0
        ray = [(0, 0)]
        while min(tr, tc) <= range_m:
            if abs(tr - tc) < tol:
                ray.extend([(row + sr, col), (row, col + sc)])
                row, col = row + sr, col + sc
                tr, tc = tr + dtr, tc + dtc
            elif tr < tc:
                row, tr = row + sr, tr + dtr
            else:
                col, tc = col + sc, tc + dtc
            ray.append((row, col))
        rays.append(ray)
    offsets = np.full((max(map(len, rays)), beams, 2), 32767, dtype=np.int64)
    for beam, ray in enumerate(rays):
        offsets[:len(ray), beam] = ray
    return offsets


def observe_trajectory(walls, trajectory_m, lx, ly):
    """v371.observe_trajectory on the anisotropic chart (first-hit traversal, strict metric 5 m)."""
    size = walls.shape[0]; cx, cy = lx / size, ly / size
    traj = np.asarray(trajectory_m)
    if (traj < 0).any() or (traj[..., 0] >= lx).any() or (traj[..., 1] >= ly).any():
        raise ValueError('pose outside chart')
    rc = np.stack([np.floor(traj[..., 1] / cy), np.floor(traj[..., 0] / cx)], -1).astype(np.int64)
    if np.any(walls[rc[..., 0], rc[..., 1]] != 0):
        raise ValueError('Sensor pose must be free; no resampling')
    rays = make_rays_aniso(SENSOR_RANGE, 256, cy, cx)
    private = np.full((traj.shape[1], size, size), -1, np.int8)
    ax = (np.arange(size, dtype=np.float64) + .5) * cx; ay = (np.arange(size, dtype=np.float64) + .5) * cy
    for step in range(len(traj)):
        observed = np.full_like(private, -1); raycast_observe(walls, rc[step], observed, rays)
        for node, (x, y) in enumerate(traj[step]):
            inside = (ax[None] - x) ** 2 + (ay[:, None] - y) ** 2 <= SENSOR_RANGE ** 2
            observed[node, ~inside] = -1
        private = np.maximum(private, observed)
    if not (private >= 0).any(axis=(1, 2)).all():
        raise AssertionError('All sources must have actual sensor evidence')
    return private


def make_fixture(split, index, size=128, rooms=1, agents=8, square=False):
    spec = world_spec(split, index, rooms, square=square)
    lx, ly = spec['chart_xy_m']
    geometry = rasterize(spec, size)
    trajectory = metric_trajectory(spec, agents) if not square else metric_trajectory_resampled(spec, agents)
    private = observe_trajectory(geometry['walls'], trajectory, lx, ly)
    adjacency = radius_adjacency(trajectory[-1], RADIO_RADIUS)
    frame = dict(origin_m=[0., 0.], extent_m=[lx, ly], shape=[size, size], cell_size_m=[lx / size, ly / size],
                 coordinate_convention='pose=(x,y); raster=[y,x]', sensor_range_m=SENSOR_RANGE, sensor_beams=256,
                 communication_radius_m=RADIO_RADIUS, static_snapshot=True, radio_occlusion=False)
    inputs = dict(private=private.copy(), positions_m=trajectory[-1].copy(), trajectory_m=trajectory.copy(), adjacency=adjacency.copy(), public_frame=frame)
    evaluation = dict(door_mask=geometry['door_mask'] & (private != 1).all(0), walls=geometry['walls'], geometry_spec=dict(spec, verts_m=None))
    manifest = dict(version=VERSION, world_id=spec['world_id'], house_id=spec['house_id'], split=split, index=int(index), stratum=int(rooms),
                    agents=agents, size=size, chart_xy_m=[lx, ly], private_sha256=array_hash(private), trajectory_sha256=array_hash(trajectory),
                    door_mask_semantics='opening 0.4-1.5 m wide (metric, per axis) through a wall <= 0.6 m thick (HouseExpo has no door labels)', TEST_opened=False)
    return dict(inputs=inputs, evaluation=evaluation, manifest=manifest)
