"""Dynamic (multi-step exploration) fixtures on top of the sealed v293 corridor-spine worlds. Claude 2026-10-22 (item 3).

Same geometry/sensor/radio as v293 (20 m world, 0.3 m slabs, 0.8 m doors, 256 beams, 5 m range, 8 m radio), but each agent now
explores for T steps: room agents walk to their door and out into the corridor, corridor agents patrol along the corridor;
every step yields a new first-hit observation and the agent's map is the cumulative ternary max M_t. The radio graph is
re-evaluated at every step from the current positions (so the forest changes over time). Deterministic from the world id.
Splits follow v293 (TRAIN / VALIDATION by world index; TEST never generated)."""
import hashlib
import numpy as np

from scripts import mapping_multiroom_data_v293 as d293
from scripts.prepare_houseexpo_pilot_v121 import make_rays, raycast_observe, radius_adjacency

VERSION = 349
T_STEPS = 6


def _seed(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'little')


def _inside(rect, p, margin):
    x0, y0, x1, y1 = rect
    return (x0 + margin <= p[0] <= x1 - margin) and (y0 + margin <= p[1] <= y1 - margin)


def _clip(rect, p, margin):
    x0, y0, x1, y1 = rect
    return np.array([min(max(p[0], x0 + margin), x1 - margin), min(max(p[1], y0 + margin), y1 - margin)])


def dynamic_paths(spec, agents, T=T_STEPS):
    """[T, N, 2] metric poses. Room agents: start as v293, head to the door centre (~1 m/step), pass through, then patrol the
    corridor; corridor agents patrol along y. All poses stay in free space by construction (rect interiors with margins)."""
    rng = np.random.default_rng(_seed(spec['world_id'] + f':dyn{VERSION}:{agents}'))
    start, assignment = d293.metric_trajectory(spec, agents)        # [3,N,2] (+-0.1 m jitter), room id or -1
    pos = start[-1].copy()
    corridor = spec['corridor_rectangle_m']
    paths = [pos.copy()]
    phase = ['room' if a >= 0 else 'corr' for a in assignment]
    ydir = rng.choice([-1., 1.], size=agents)
    for t in range(1, T):
        new = pos.copy()
        for n in range(agents):
            step = rng.uniform(.8, 1.3)
            if phase[n] == 'room':
                room = spec['rooms'][assignment[n]]; rect = room['rectangle_m']; door = spec['door_rectangles_m'][assignment[n]]
                dc = np.array([(door[0] + door[2]) / 2, (door[1] + door[3]) / 2])
                # aim at a point just inside the room in front of the door, then at the corridor centre line beyond it
                side_x = rect[2] - .5 if room['side'] == 'left' else rect[0] + .5
                target = np.array([side_x, dc[1]]) if abs(pos[n, 0] - side_x) > .6 or abs(pos[n, 1] - dc[1]) > .3 else np.array([10., dc[1]])
                v = target - pos[n]; dist = np.linalg.norm(v)
                p = pos[n] + (v / max(dist, 1e-9)) * min(step, dist)
                if _inside(corridor, p, .3):
                    phase[n] = 'corr'; new[n] = p
                elif _inside(rect, p, .45) or _inside(door, p, .05):
                    new[n] = p
                else:
                    new[n] = _clip(rect, p, .45)
            else:
                p = np.array([10. + rng.uniform(-.25, .25), pos[n, 1] + ydir[n] * step])
                if not _inside(corridor, p, .3):
                    ydir[n] = -ydir[n]; p = np.array([10. + rng.uniform(-.25, .25), pos[n, 1] + ydir[n] * step])
                new[n] = _clip(corridor, p, .3)
        pos = new; paths.append(pos.copy())
    return np.stack(paths), assignment


def observe_steps(walls, paths):
    """Per-step first-hit observations with the v293 sensor; returns cumulative maps [T,N,H,H] int8 and per-step maps."""
    size = walls.shape[0]; cell = d293.EXTENT / size
    rays = make_rays(d293.SENSOR_RANGE / cell, 256)
    axis = (np.arange(size, dtype=np.float64) + .5) * cell
    T, n = paths.shape[:2]
    cumulative = np.full((T, n, size, size), -1, np.int8)
    current = np.full((n, size, size), -1, np.int8)
    for t in range(T):
        rc = np.floor(paths[t][:, ::-1] / cell).astype(np.int64)
        if np.any(walls[rc[:, 0], rc[:, 1]] != 0):
            raise ValueError('dynamic pose must be free')
        observed = np.full_like(current, -1); raycast_observe(walls, rc, observed, rays)
        for node, (x, y) in enumerate(paths[t]):
            inside = (axis[None] - x) ** 2 + (axis[:, None] - y) ** 2 <= d293.SENSOR_RANGE ** 2
            observed[node, ~inside] = -1
        current = np.maximum(current, observed)
        cumulative[t] = current
    return cumulative


def make_dynamic_fixture(split, index, size=128, rooms=6, agents=8, T=T_STEPS):
    spec = d293.world_spec(split, index, rooms)
    geometry = d293.rasterize(spec, size)
    paths, assignment = dynamic_paths(spec, agents, T)
    maps = observe_steps(geometry['walls'], paths)
    adjacency = np.stack([radius_adjacency(paths[t], d293.RADIO_RADIUS) for t in range(T)])
    return dict(spec=dict(split=split, index=index, rooms=rooms, agents=agents, size=size, T=T, version=VERSION),
                world_id=spec['world_id'], paths_m=paths, maps=maps, adjacency=adjacency, walls=geometry['walls'],
                door_mask=geometry['door_mask'], assignment=assignment)
