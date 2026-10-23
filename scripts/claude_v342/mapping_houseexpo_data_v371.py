"""HouseExpo world family v371 = v365 with ONE change: a HOUSE-FITTED public chart (Claude, 2026-10-23).

Motivation (D4-3/D4-4): on the fixed 20 m chart a typical HouseExpo house covers ~16 of the 64 public 2.5 m blocks, so ~75 % of the
64-dimensional owned code describes empty exterior. Here the square chart side is L = max(8, ceil(max_extent_m + 1.0)) metres per
house (public metadata: the building's bounding box), the house is centred in it, and the same 8 x 8 block grid / native H raster
covers the chart (block side L/8, cell L/H). Everything metric is unchanged: 256 first-hit beams with a strict 5 m range, 8 m radio,
three +-0.1 m poses, 0.45 m pose clearance, N in {8, 16}; same splits, strata, index and TEST exclusion as v365 (the index file and
the per-house contour hashes are reused unchanged).
Fixture contract identical to v365 except public_frame.extent_m = [L, L] and cell_size_m = L/H.
"""
import math
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

import mapping_houseexpo_data_v365 as v365
from scripts import mapping_multiroom_data_v293 as d293
from scripts.prepare_houseexpo_pilot_v121 import make_rays, raycast_observe, radius_adjacency

VERSION = 371
SIZES = v365.SIZES
AGENT_COUNTS = v365.AGENT_COUNTS
STRATA = v365.STRATA
INPUT_FIELDS = v365.INPUT_FIELDS
SENSOR_RANGE = d293.SENSOR_RANGE
RADIO_RADIUS = d293.RADIO_RADIUS
array_hash = v365.array_hash
scene_plan = v365.scene_plan
narrow_mask = v365.narrow_mask


def world_spec(split, index, rooms=1):
    spec = v365.world_spec(split, index, rooms)                    # verts centred in a 20 m chart
    v = np.asarray(spec['verts_m']) - v365.EXTENT / 2              # centred at the origin
    ext = float(np.ptp(v, 0).max())
    L = float(max(8., math.ceil(ext + 1.0)))
    spec.update(version=VERSION, world_id=spec['world_id'].replace('v365:', 'v371:'), chart_m=L, verts_m=(v + L / 2).tolist(),
                chart_rule='L = max(8, ceil(max bbox extent + 1 m)); house centred; 8x8 public blocks of L/8 m')
    return spec


def rasterize(spec, size, supersample=4):
    if size not in SIZES:
        raise ValueError('native size required')
    L = spec['chart_m']; cell = L / size
    pix = np.asarray(spec['verts_m']) / cell * supersample
    img = Image.new('L', (size * supersample, size * supersample), 0)
    ImageDraw.Draw(img).polygon([tuple(p) for p in pix], fill=1)
    s = np.asarray(img, dtype=np.uint8)
    free = s.reshape(size, supersample, size, supersample).min(axis=(1, 3)) != 0
    walls = (~free).astype(np.int8)
    return dict(walls=walls, door_mask=narrow_mask(walls, cell))


def metric_trajectory(spec, agents=8):
    if agents not in AGENT_COUNTS:
        raise ValueError('agents')
    rng = np.random.default_rng(v365._seed(spec['world_id'] + f':poses:{agents}'))
    L = spec['chart_m']; cell = L / 512
    fine = rasterize(spec, 512)['walls'] == 0
    lab, nlab = ndimage.label(fine)
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, nlab + 1))
    comp = lab == (1 + int(np.argmax(sizes)))
    clearance = ndimage.distance_transform_edt(fine) * cell
    cand = np.argwhere(comp & (clearance >= .45))
    if len(cand) < agents:
        raise ValueError('not enough free space with clearance')
    order = rng.permutation(len(cand)); centres = []
    for min_sep in (1.0, .5, 0.):
        centres = []
        for i in order:
            rc = cand[i]; p = np.array([(rc[1] + .5) * cell, (rc[0] + .5) * cell])
            if all(np.linalg.norm(p - q) >= min_sep for q in centres):
                centres.append(p)
            if len(centres) == agents:
                break
        if len(centres) == agents:
            break
    centres = np.asarray(centres)
    deltas = rng.uniform(-.10, .10, (agents, 2))
    return centres[None] + np.array([-1., 0., 1.])[:, None, None] * deltas[None]


def observe_trajectory(walls, trajectory_m, L):
    """v293.observe_trajectory with the chart side L as a parameter (first-hit DDA, strict metric 5 m)."""
    size = walls.shape[0]; cell = L / size
    traj = np.asarray(trajectory_m)
    if (traj < 0).any() or (traj >= L).any():
        raise ValueError('pose outside chart')
    rc = np.floor(traj[..., ::-1] / cell).astype(np.int64)
    if np.any(walls[rc[..., 0], rc[..., 1]] != 0):
        raise ValueError('Sensor pose must be free; no resampling')
    rays = make_rays(SENSOR_RANGE / cell, 256)
    private = np.full((traj.shape[1], size, size), -1, np.int8)
    axis = (np.arange(size, dtype=np.float64) + .5) * cell
    for step in range(len(traj)):
        observed = np.full_like(private, -1); raycast_observe(walls, rc[step], observed, rays)
        for node, (x, y) in enumerate(traj[step]):
            inside = (axis[None] - x) ** 2 + (axis[:, None] - y) ** 2 <= SENSOR_RANGE ** 2
            observed[node, ~inside] = -1
        private = np.maximum(private, observed)
    if not (private >= 0).any(axis=(1, 2)).all():
        raise AssertionError('All sources must have actual sensor evidence')
    return private


def make_fixture(split, index, size=128, rooms=1, agents=8):
    spec = world_spec(split, index, rooms)
    L = spec['chart_m']
    geometry = rasterize(spec, size)
    trajectory = metric_trajectory(spec, agents)
    private = observe_trajectory(geometry['walls'], trajectory, L)
    adjacency = radius_adjacency(trajectory[-1], RADIO_RADIUS)
    frame = dict(origin_m=[0., 0.], extent_m=[L, L], shape=[size, size], cell_size_m=L / size,
                 coordinate_convention='pose=(x,y); raster=[y,x]', sensor_range_m=SENSOR_RANGE, sensor_beams=256,
                 communication_radius_m=RADIO_RADIUS, static_snapshot=True, radio_occlusion=False)
    inputs = dict(private=private.copy(), positions_m=trajectory[-1].copy(), trajectory_m=trajectory.copy(), adjacency=adjacency.copy(), public_frame=frame)
    evaluation = dict(door_mask=geometry['door_mask'] & (private != 1).all(0), walls=geometry['walls'], geometry_spec=dict(spec, verts_m=None))
    manifest = dict(version=VERSION, world_id=spec['world_id'], house_id=spec['house_id'], split=split, index=int(index), stratum=int(rooms),
                    agents=agents, size=size, chart_m=L, private_sha256=array_hash(private), trajectory_sha256=array_hash(trajectory),
                    door_mask_semantics='opening 0.4-1.5 m wide through a wall <= 0.6 m thick (HouseExpo has no door labels)', TEST_opened=False)
    return dict(inputs=inputs, evaluation=evaluation, manifest=manifest)
