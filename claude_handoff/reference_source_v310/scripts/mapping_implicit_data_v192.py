"""CPU-only synthetic floorplans for the v192 implicit-code capacity experiment.

Coordinates are normalized (x, y); array indexing is [row=y, column=x]. Walls
and doors are continuous rectangles, not resized bitmaps. The same world ID
therefore has the same geometry at 128/256/512. Observation masks are algebraic
partitions, NOT simulated robot trajectories. Only TRAIN/VALIDATION exist.
"""
from __future__ import annotations

import hashlib
import json
import struct
import zlib
from functools import lru_cache

import numpy as np

SIZES = (128, 256, 512)
REGIMES = ("partition", "overlap", "unknown")
COMPLEXITY = {"base": 5, "medium": 11, "high": 19}
SPLIT_IDS = {"train": 0, "validation": 1}
MAX_INDEX = 1_000_000_000


def _identity(split, index, complexity):
    split = str(split).lower()
    if split not in SPLIT_IDS:
        raise ValueError("only TRAIN/VALIDATION allowed; TEST is not implemented")
    if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) or not 0 <= index < MAX_INDEX:
        raise ValueError("index must be an integer in [0, 1000000000)")
    if complexity not in COMPLEXITY:
        raise ValueError("complexity must be base, medium or high")
    return split, int(index), complexity


@lru_cache(maxsize=2048)
def _spec_json(split, index, complexity):
    split, index, complexity = _identity(split, index, complexity)
    # Disjoint integer ID ranges; size/regime never change a world's RNG.
    seed = 192_000_000_000 + SPLIT_IDS[split] * MAX_INDEX + index
    rng = np.random.default_rng(seed)
    walls = [(0., 0., .028, 1.), (.972, 0., 1., 1.),
             (0., 0., 1., .028), (0., .972, 1., 1.)]
    doors, rooms = [], [(.028, .028, .972, .972)]
    for _ in range(COMPLEXITY[complexity]):
        # Split larger rooms first, with random tie-breaking/aspect variation.
        scores = [(r[2]-r[0]) * (r[3]-r[1]) * rng.uniform(.85, 1.15) for r in rooms]
        k = int(np.argmax(scores))
        x0, y0, x1, y1 = rooms.pop(k)
        width, height = x1-x0, y1-y0
        vertical = width > height * rng.uniform(.80, 1.20)
        thickness = float(rng.uniform(.016, .024))  # >2 cells at size128.
        fraction = float(rng.uniform(.40, .60))
        if vertical:
            cut, length = x0 + fraction * width, height
            opening = min(.105, max(.052, length * .25))
            center = float(rng.uniform(y0 + .30*length, y0 + .70*length))
            walls.append((cut-thickness/2, y0, cut+thickness/2, y1))
            doors.append((cut-thickness/2-.002, center-opening/2, cut+thickness/2+.002, center+opening/2))
            rooms.extend([(x0, y0, cut-thickness/2, y1), (cut+thickness/2, y0, x1, y1)])
        else:
            cut, length = y0 + fraction * height, width
            opening = min(.105, max(.052, length * .25))
            center = float(rng.uniform(x0 + .30*length, x0 + .70*length))
            walls.append((x0, cut-thickness/2, x1, cut+thickness/2))
            doors.append((center-opening/2, cut-thickness/2-.002, center+opening/2, cut+thickness/2+.002))
            rooms.extend([(x0, y0, x1, cut-thickness/2), (x0, cut+thickness/2, x1, y1)])
    # Higher complexity adds genuine geometry, not merely more raster pixels.
    if complexity != "base":
        for room in rooms[::3]:
            x0, y0, x1, y1 = room
            w, h = x1-x0, y1-y0
            t = .016
            if min(w, h) > .10:
                # An L-shaped, wall-attached alcove/obstacle in selected rooms.
                walls.extend([(x0, y0, x0+.30*w, y0+t),
                              (x0+.30*w-t, y0, x0+.30*w, y0+.30*h)])
    spec = dict(version=192, split=split, index=index, seed=seed,
                world_id=f"{split}:{index}:{complexity}", complexity=complexity,
                wall_rectangles=walls, door_rectangles=doors,
                partition_x=float(rng.uniform(.38, .62)),
                overlap_width=float(rng.uniform(.12, .25)),
                unknown_rectangle=(float(rng.uniform(.58, .68)), 0., 1., float(rng.uniform(.35, .48))),
                nominal_rooms=len(rooms), minimum_wall_width=.016)
    return json.dumps(spec, sort_keys=True, separators=(",", ":"))


def world_spec(split, index, complexity="base"):
    """A fresh JSON-compatible copy; its geometry is independent of resolution."""
    return json.loads(_spec_json(*_identity(split, index, complexity)))


def query_coords(size, flat=True):
    """Cell centers in [-1,1], ordered (x,y), matching row-major flattening."""
    if size not in SIZES:
        raise ValueError(f"size must be one of {SIZES}")
    axis = (np.arange(size, dtype=np.float32) + .5) * (2. / size) - 1.
    x, y = np.meshgrid(axis, axis, indexing="xy")
    coords = np.stack((x, y), axis=-1)
    return coords.reshape(-1, 2) if flat else coords


def _rectangles(rectangles, x, y):
    result = np.zeros((len(y), len(x)), dtype=bool)
    # Vectorized pixels; only the small geometry-list dimension is a Python loop.
    for x0, y0, x1, y1 in rectangles:
        result |= ((y[:, None] >= y0) & (y[:, None] < y1)
                   & (x[None, :] >= x0) & (x[None, :] < x1))
    return result


def sample_world(split, index, size=128, complexity="base", regime=None):
    """Return walls, two private maps/masks, exact observed union and metadata.

    Private/union values: -1 unknown, 0 observed-free, 1 observed-wall. ``walls``
    is supervision only; it must not be supplied to a partial-observation model.
    """
    spec = world_spec(split, index, complexity)
    if size not in SIZES:
        raise ValueError(f"size must be one of {SIZES}")
    regime = REGIMES[int(index) % len(REGIMES)] if regime is None else regime
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}")
    axis = (np.arange(size, dtype=np.float64) + .5) / size
    walls = _rectangles(spec["wall_rectangles"], axis, axis)
    door_mask = _rectangles(spec["door_rectangles"], axis, axis)
    walls[door_mask] = False
    cut = spec["partition_x"]
    half_overlap = spec["overlap_width"] / 2 if regime != "partition" else 0.
    a = np.broadcast_to(axis[None, :] < cut+half_overlap, (size, size)).copy()
    b = np.broadcast_to(axis[None, :] >= cut-half_overlap, (size, size)).copy()
    if regime == "unknown":
        hidden = _rectangles([spec["unknown_rectangle"]], axis, axis)
        a &= ~hidden
        b &= ~hidden
    masks = np.stack((a, b))
    private = np.where(masks, walls[None, ...], -1).astype(np.int8)
    union = np.where(masks.any(axis=0), walls, -1).astype(np.int8)
    geometry_hash = hashlib.sha256(_spec_json(*_identity(split, index, complexity)).encode()).hexdigest()
    return dict(walls=walls.astype(np.uint8), masks=masks, private=private,
                union=union, door_mask=door_mask,
                metadata={**spec, "geometry_sha256": geometry_hash, "size": size,
                          "regime": regime, "observations": "algebraic partitions; not robot rollouts",
                          "coordinates": "query=(x,y) in [-1,1]; grid=[y,x]"})


def _ternary(value):
    a = np.asarray(value)
    if a.ndim != 2 or a.dtype.kind not in "iu" or not np.isin(a, (-1, 0, 1)).all():
        raise ValueError("a 2D integer {-1,0,1} map is required")
    return a


def metrics(prediction, sample, receiver=0):
    """Count-based metrics; None for an absent denominator, never fabricated 1.0.

    ``prediction`` is a ternary decoded map. Own means receiver-observed cells;
    peer means union-known but not receiver-observed. Wall recall counts unknown
    predictions as misses. Door blocking is predicted-wall on known door cells;
    door free recall additionally penalizes unknown predictions.
    """
    pred, target = _ternary(prediction), _ternary(sample["union"])
    if pred.shape != target.shape or receiver not in (0, 1):
        raise ValueError("prediction shape/receiver mismatch")
    own = np.asarray(sample["masks"][receiver], dtype=bool)
    union, known = target >= 0, pred >= 0
    peer, wall = union & ~own, target == 1
    door = np.asarray(sample.get("door_mask", np.zeros_like(union)), dtype=bool) & union
    definitions = {
        "own_accuracy": (pred == target, own),
        "fusion_accuracy": (pred == target, union),
        "peer_accuracy": (pred == target, peer),
        "peer_wall_recall": (pred == 1, peer & wall),
        "wall_recall": (pred == 1, wall),
        "wall_precision": (target == 1, (pred == 1) & union),
        "unknown_false_known": (known, ~union),
        "door_false_wall": (pred == 1, door),
        "door_free_recall": (pred == 0, door),
    }
    report = {}
    for name, (correct, region) in definitions.items():
        numerator, denominator = int((correct & region).sum()), int(region.sum())
        report[name] = numerator / denominator if denominator else None
        report[name + "_numerator"] = numerator
        report[name + "_denominator"] = denominator
    return report


def pack_2bit(private):
    """Actual 12-byte shape header plus 4 ternary symbols/byte; code3 reserved."""
    a = _ternary(private)
    codes = (a.astype(np.int16).ravel() + 1).astype(np.uint8)
    codes = np.pad(codes, (0, (-len(codes)) % 4))
    payload = (codes[0::4] | (codes[1::4] << 2) | (codes[2::4] << 4) | (codes[3::4] << 6)).tobytes()
    return struct.pack("<4sII", b"G192", *a.shape) + payload


def unpack_2bit(packet):
    if len(packet) < 12:
        raise ValueError("truncated packet")
    magic, height, width = struct.unpack("<4sII", packet[:12])
    if magic != b"G192" or not height or not width or len(packet) != 12 + (height*width+3)//4:
        raise ValueError("invalid packet header/length")
    packed = np.frombuffer(packet[12:], dtype=np.uint8)
    codes = np.stack([(packed >> k) & 3 for k in (0, 2, 4, 6)], axis=-1).ravel()
    if (codes == 3).any() or codes[height*width:].any():
        raise ValueError("reserved code or nonzero padding")
    return (codes[:height*width].astype(np.int8)-1).reshape(height, width)


def raw_packet_counts(private):
    """Real serialized sizes, not a claim about radio traffic or forwarding.

    Sparse format uses uint32 flat indices + uint8 values; zlib sees only a
    single source's payload. Shape/count headers are included, link headers not.
    The same overhead convention must be applied to learned-message comparisons.
    """
    a = _ternary(private)
    dense = pack_2bit(a)
    indices = np.flatnonzero(a.ravel() >= 0).astype("<u4")
    values = a.ravel()[indices].astype(np.uint8)
    sparse = struct.pack("<4sIII", b"S192", *a.shape, len(indices)) + indices.tobytes() + values.tobytes()
    return dict(dense_2bit_bytes=len(dense), dense_2bit_zlib_bytes=12+len(zlib.compress(dense[12:], 9)),
                sparse_index_value_bytes=len(sparse), sparse_zlib_bytes=16+len(zlib.compress(sparse[16:], 9)),
                observed_cells=len(indices), link_header_bytes_included=0,
                counting_scope="one source packet; no network forwarding assumption")
