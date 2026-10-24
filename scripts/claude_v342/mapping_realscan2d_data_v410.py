"""Real-scan 2D world family v410 = HouseExpo v371 with ONE change: the geometry is a floor slice of a REAL 3D scan (Claude, 2026-10-24).

Everything else is v371, so the HouseExpo working baseline (train_v392 --decoder-width 2.0 --frame-cond, runs/v393-frame-cycles/c3-s*)
runs on it unchanged: the house-fitted square public chart L = max(8, ceil(max extent + 1 m)) with the floor centred and an 8 x 8 block
grid over an H = 128 raster (cell L/H); a cell is free only if all 4 x 4 supersamples are free; three +-0.1 m poses around a centre
with >= 0.45 m clearance in the largest free component (minimum separation 1.0 / 0.5 / 0 m); 256 first-hit beams, strict 5 m range;
8 m radio; N in {8, 16}; the same fixture contract (inputs / evaluation / manifest) and the same door-mask heuristic.
Geometry: the 2 cm base rasters of prep_realscan_base_v410.py (Replica now, HM3D later; gravity-aligned, upward floors only, review
M1): free = interior & no mesh surface within 0.15-1.60 m above the floor; everything else (obstacles, the scan boundary, the void
beyond exterior windows) is "wall", exactly as everything outside a HouseExpo polygon is wall; interior glass is not in the mesh and
is free in the slice. A fixture's raster resamples the base raster at a yaw that aligns the chart with the floor's main wall direction
(review M2; VALIDATION: aligned; TRAIN: aligned and turned by k x 90 deg, k from a hash of the index); a fine cell is free only if no
blocked base cell lies within its half-diagonal (conservative: thin obstacles are never lost by resampling; this erodes 3-11 % of the
free area at 128^2, review S3).
Splits by scene (never the same scene in two splits; the six FRL apartments are one flat restaged and stay together in TRAIN):
  VALIDATION = apartment_2, hotel_0, office_3, office_4 (one floor each; the 'rooms' field of a VALIDATION spec is the scene stratum
               1..4, so the trainer's equal-weight summary is equal weight per (scene, N)); index = agent placement (0-7 val, 8-15 fresh)
  TRAIN      = every other floor of the base index (15 floors of 14 scenes; about 10 distinct layouts, the six FRL apartments being one
               flat restaged, review S7); index i -> floor i mod 15, a quarter turn k(i) in {0, 1, 2, 3}
  TEST       = none in Replica (pilot); claims at scale come later from HM3D. Caveat: office_0-4 are rooms of one office building and
               room_0-2 may share a building with an apartment, so VALIDATION measures new rooms more than new buildings.
"""
import hashlib, json, math, os
from pathlib import Path
import numpy as np
from scipy import ndimage

import mapping_houseexpo_data_v365 as v365
import mapping_houseexpo_data_v371 as v371
from scripts.prepare_houseexpo_pilot_v121 import radius_adjacency

VERSION = 410
SIZES = (128, 256, 512)
AGENT_COUNTS = (8, 16)
SENSOR_RANGE = v371.SENSOR_RANGE
RADIO_RADIUS = v371.RADIO_RADIUS
array_hash = v365.array_hash
narrow_mask = v365.narrow_mask
BASE_DIR = Path(os.environ.get('REALSCAN_BASE_DIR', '/mnt/datasets/yuzhou-sheaf-gbp-research/realscan-v410/replica-base'))
INDEX_FILE = str(BASE_DIR / 'index.json')
VAL_SCENES = ('apartment_2', 'hotel_0', 'office_3', 'office_4')
_BASE = {}
_UNITS = None


def _seed(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'little')


def _units():
    global _UNITS
    if _UNITS is None:
        rows = json.load(open(INDEX_FILE))['rows']
        train = [r for r in rows if r['scene'] not in VAL_SCENES]
        val = [next(r for r in rows if r['scene'] == s and r['floor'] == 0) for s in VAL_SCENES]
        assert not {r['scene'] for r in train} & {r['scene'] for r in val}
        _UNITS = dict(TRAIN=train, VALIDATION=val)
    return _UNITS


def _base(row):
    key = row['file']
    if key not in _BASE:
        path = BASE_DIR / row['file']
        assert hashlib.sha256(open(path, 'rb').read()).hexdigest() == row['sha256'], f'base raster {path} changed'
        z = np.load(path)
        blocked = ~(z['interior'] & ~z['obstacle'])
        _BASE[key] = dict(blocked=blocked, cell=float(z['cell']), free_yx=np.argwhere(~blocked), dil={})
    return _BASE[key]


def world_spec(split, index, rooms=1):
    split = str(split).upper()
    if split not in ('TRAIN', 'VALIDATION'):
        raise ValueError('TEST is closed (Replica has no TEST split)')
    units = _units()[split]
    index = int(index)
    row = units[index % len(units)] if split == 'TRAIN' else units[int(rooms) - 1]
    # review v410 M2: the chart is aligned with the building's main wall direction (as v371 aligns HouseExpo houses), so TRAIN, VAL and
    # HouseExpo share one orientation distribution; TRAIN additionally turns the aligned floor by k x 90 deg (k from the index hash)
    k90 = (_seed(f'v410:TRAIN:{index}:k90') % 4) if split == 'TRAIN' else 0
    yaw = math.radians(-float(row['wall_angle_deg']) + 90. * k90)
    b = _base(row); c = b['cell']
    pts = (b['free_yx'][:, ::-1] + .5) * c                                          # free cell centres (x, y) in the scene frame
    R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    q = pts @ R.T
    lo, hi = q.min(0), q.max(0)
    L = float(max(8., math.ceil(float((hi - lo).max()) + 1.0)))
    centre_rot = (lo + hi) / 2                                                     # centre of the rotated free region
    world_id = f"v410:{split}:{index}:{row['scene']}:f{row['floor']}" + (f':g{int(rooms)}' if split == 'VALIDATION' else '')
    return dict(version=VERSION, split=split, index=index, rooms=int(rooms), world_id=world_id, house_id=f"{row['scene']}_f{row['floor']}",
                scene=row['scene'], floor=row['floor'], base_file=row['file'], yaw=yaw, k90=k90, wall_angle_deg=float(row['wall_angle_deg']), chart_m=L, centre_rot=centre_rot.tolist(),
                chart_rule='L = max(8, ceil(max extent of the rotated free region + 1 m)); floor centred; 8x8 public blocks of L/8 m')


def _blocked_fine(spec, n):
    """[n, n] bool raster (rows = y, cols = x) of the chart: True where a blocked base cell lies within the fine cell's half-diagonal."""
    row = next(r for r in _units()[spec['split']] if r['file'] == spec['base_file'])
    b = _base(row); c = b['cell']; L = spec['chart_m']; fc = L / n
    k = int(math.ceil((fc * math.sqrt(2) / 2) / c))
    if k not in b['dil']:
        b['dil'][k] = ndimage.binary_dilation(b['blocked'], structure=np.ones((3, 3), bool), iterations=k) if k > 0 else b['blocked']
    blocked = b['dil'][k]
    ax = (np.arange(n) + .5) * fc
    X, Y = np.meshgrid(ax, ax)                                                      # chart coordinates, rows = y
    yaw = spec['yaw']; R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    rot = np.stack([X.ravel() - L / 2, Y.ravel() - L / 2], 1) + np.asarray(spec['centre_rot'])
    s = rot @ R                                                                     # inverse rotation (R orthonormal): scene frame
    ix = np.floor(s[:, 0] / c).astype(np.int64); iy = np.floor(s[:, 1] / c).astype(np.int64)
    inside = (ix >= 0) & (ix < blocked.shape[1]) & (iy >= 0) & (iy < blocked.shape[0])
    out = np.ones(len(s), bool)
    out[inside] = blocked[iy[inside], ix[inside]]
    return out.reshape(n, n)


def rasterize(spec, size, supersample=4):
    if size not in SIZES:
        raise ValueError('native size required')
    fine_free = ~_blocked_fine(spec, size * supersample)
    free = fine_free.reshape(size, supersample, size, supersample).all(axis=(1, 3))   # v371 rule: free only if every supersample is free
    walls = (~free).astype(np.int8)
    return dict(walls=walls, door_mask=narrow_mask(walls, spec['chart_m'] / size))


def metric_trajectory(spec, agents=8):
    """v371.metric_trajectory on this world's 512 raster (same rule, same RNG construction from the world id)."""
    if agents not in AGENT_COUNTS:
        raise ValueError('agents')
    rng = np.random.default_rng(v365._seed(spec['world_id'] + f':poses:{agents}'))
    L = spec['chart_m']; cell = L / 512
    fine = rasterize(spec, 512)['walls'] == 0
    lab, nlab = ndimage.label(fine)
    if nlab == 0:
        raise ValueError('no free space')
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


def make_fixture(split, index, size=128, rooms=1, agents=8):
    spec = world_spec(split, index, rooms)
    L = spec['chart_m']
    geometry = rasterize(spec, size)
    trajectory = metric_trajectory(spec, agents)
    private = v371.observe_trajectory(geometry['walls'], trajectory, L)
    adjacency = radius_adjacency(trajectory[-1], RADIO_RADIUS)
    frame = dict(origin_m=[0., 0.], extent_m=[L, L], shape=[size, size], cell_size_m=L / size,
                 coordinate_convention='pose=(x,y); raster=[y,x]', sensor_range_m=SENSOR_RANGE, sensor_beams=256,
                 communication_radius_m=RADIO_RADIUS, static_snapshot=True, radio_occlusion=False)
    inputs = dict(private=private.copy(), positions_m=trajectory[-1].copy(), trajectory_m=trajectory.copy(), adjacency=adjacency.copy(), public_frame=frame)
    evaluation = dict(door_mask=geometry['door_mask'] & (private != 1).all(0), walls=geometry['walls'], geometry_spec=dict(spec))
    manifest = dict(version=VERSION, world_id=spec['world_id'], house_id=spec['house_id'], split=split, index=int(index), stratum=int(rooms),
                    agents=agents, size=size, chart_m=L, private_sha256=array_hash(private), trajectory_sha256=array_hash(trajectory),
                    door_mask_semantics='opening 0.4-1.5 m wide through a wall <= 0.6 m thick (heuristic, as v365)', TEST_opened=False)
    return dict(inputs=inputs, evaluation=evaluation, manifest=manifest)


def scene_plan(split, size=None, train_per_stratum=200):
    split = str(split).upper()
    if split not in ('TRAIN', 'VALIDATION'):
        raise ValueError('TEST is closed')
    sizes = (128,) if size is None else (size,)
    if split == 'TRAIN':
        return [dict(split=split, index=i, rooms=1, agents=(8 if i % 2 == 0 else 16), size=h) for h in sizes for i in range(train_per_stratum)]
    return [dict(split=split, index=i, rooms=g, agents=n, size=h) for h in sizes for g in range(1, len(VAL_SCENES) + 1) for n in AGENT_COUNTS for i in range(8)]
