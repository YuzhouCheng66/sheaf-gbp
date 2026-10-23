"""HouseExpo -> metric, ray-observed multi-agent occupancy graphs (CPU only).

The official ``verts`` contour encloses FREE floor space. Room-category boxes
are annotations, not obstacles. This adapter does not import PseudoSLAM/TF or
resize buildings. It crops a fixed metric chart, rasterizes conservatively,
and simulates reachable cell-by-cell motion with occluded range measurements.

Only private/positions/active/adjacency/seed/radius enter the model NPZ files.
Hidden raster geometry and packed observed masks are separate audit artifacts.
TRAIN/VALIDATION building IDs are partitioned before any crop or trajectory;
there is deliberately no TEST loader. Dataset generation uses known geometry
to sample feasible trajectories, NOT to fill private maps behind obstacles.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import tarfile
import time

import numpy as np
from PIL import Image, ImageDraw


SOURCE_URL = "https://raw.githubusercontent.com/TeaganLi/HouseExpo/master/HouseExpo/json.tar.gz"
SOURCE_SHA256 = "dad4a7d230937bdfe1d57c91330e697b625ec681eefd9cda3159082a939b4f2f"
SOURCE_READER = "https://github.com/TeaganLi/HouseExpo/blob/master/pseudoslam/envs/simulator/jsonReader.py"
INPUT_FIELDS = ("private", "positions", "active", "adjacency", "seed", "radius")
NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little") & ((1 << 63) - 1)


def partition_ids(ids, seed):
    """Partition untransformed source IDs, with deterministic 80/20 assignment."""
    result = {"train": [], "validation": []}
    for source_id in sorted(set(ids)):
        bucket = stable_seed(f"partition:{seed}:{source_id}") % 10
        split = "train" if bucket < 8 else "validation"
        result[split].append(source_id)
    for split in result:
        result[split].sort(key=lambda x: stable_seed(f"order:{seed}:{x}"))
    return result


def validate_vertices(record):
    verts = np.asarray(record.get("verts", []), dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1:] != (2,) or len(verts) < 3:
        raise ValueError("missing_or_invalid_polygon")
    if not np.isfinite(verts).all() or (np.abs(verts) > 10000).any():
        raise ValueError("nonfinite_or_unbounded_polygon")
    area = abs(np.dot(verts[:, 0], np.roll(verts[:, 1], -1))
               - np.dot(verts[:, 1], np.roll(verts[:, 0], -1))) / 2
    if area < 1.0 or np.min(np.ptp(verts, axis=0)) < .5:
        raise ValueError("degenerate_or_tiny_polygon")
    return verts


def geometry_hash(verts):
    """Translation/cyclic-order/reversal-invariant exact-contour deduplication.

    This is NOT a claim to detect every near-duplicate or rotated floor plan.
    """
    points = np.round((verts - verts.min(0)) * 10000).astype(np.int64)
    if len(points) > 1 and np.array_equal(points[0], points[-1]):
        points = points[:-1]
    candidates = []
    for oriented in (points, points[::-1]):
        index = np.lexsort((oriented[:, 1], oriented[:, 0]))[0]
        candidates.append(np.roll(oriented, -int(index), axis=0).tobytes())
    return hashlib.sha256(min(candidates)).hexdigest()


def rasterize_floor(verts, size, meters_per_cell, origin_xy, supersample=4):
    """Free iff every supersample is floor; internal thin walls are conservative.

    Raster truth is discretized geometry, not exact continuous-polygon truth.
    No synthetic walls are inserted at crop boundaries.
    """
    if not 8 <= size <= 128 or meters_per_cell <= 0 or supersample < 1:
        raise ValueError("invalid_raster_configuration")
    pixels = (np.asarray(verts) - np.asarray(origin_xy)) / meters_per_cell * supersample
    image = Image.new("L", (size * supersample, size * supersample), 0)
    ImageDraw.Draw(image).polygon([tuple(p) for p in pixels], fill=1)
    samples = np.asarray(image, dtype=np.uint8)
    free = samples.reshape(size, supersample, size, supersample).min(axis=(1, 3)) != 0
    return (~free).astype(np.int8)


def largest_free_component(truth):
    height, width = truth.shape
    visited = np.zeros_like(truth, dtype=bool)
    largest = []
    for r, c in np.argwhere(truth == 0):
        if visited[r, c]:
            continue
        queue = deque([(int(r), int(c))])
        visited[r, c] = True
        current = []
        while queue:
            p = queue.popleft()
            current.append(p)
            for dr, dc in NEIGHBORS:
                nr, nc = p[0] + dr, p[1] + dc
                if 0 <= nr < height and 0 <= nc < width and not visited[nr, nc] and truth[nr, nc] == 0:
                    visited[nr, nc] = True
                    queue.append((nr, nc))
        if len(current) > len(largest):
            largest = current
    return np.asarray(largest, dtype=np.int64).reshape(-1, 2)


def crop_house(verts, size, meters_per_cell, rng):
    center = (verts.min(0) + verts.max(0)) / 2
    extent = size * meters_per_cell
    # One source-house chart is fixed for all four trajectory variants.
    origin = np.floor((center - extent / 2) / meters_per_cell) * meters_per_cell
    truth = rasterize_floor(verts, size, meters_per_cell, origin)
    component = largest_free_component(truth)
    if len(component) < max(100, size * size * .015):
        raise ValueError("too_little_reachable_floor_in_center_crop")
    if (truth == 0).mean() > .98:
        raise ValueError("center_crop_has_almost_no_geometry")
    return truth, component, origin


def make_rays(range_cells, beams=256):
    """Cell-centered Amanatides-Woo traversal, conservative at corner ties.

    Returns packed integer offsets, with a sentinel for absent tail entries.
    Side cells at exact corner hits precede the diagonal, avoiding diagonal
    visibility through a zero-width crack. Beam angles/range are public.
    """
    if range_cells <= 0 or beams < 4:
        raise ValueError("invalid_sensor")
    rays = []
    for angle in np.arange(beams) * (2 * np.pi / beams):
        dr, dc = float(np.sin(angle)), float(np.cos(angle))
        sr, sc = int(np.sign(dr)), int(np.sign(dc))
        ar, ac = abs(dr), abs(dc)
        tr = .5 / ar if ar > 1e-12 else math.inf
        tc = .5 / ac if ac > 1e-12 else math.inf
        dtr = 1 / ar if ar > 1e-12 else math.inf
        dtc = 1 / ac if ac > 1e-12 else math.inf
        row = col = 0
        ray = [(0, 0)]
        while min(tr, tc) <= range_cells:
            if abs(tr - tc) < 1e-10:
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


def raycast_observe(truth, positions, private, ray_offsets):
    """Accumulate ONLY each robot's own line-of-sight sensor measurements."""
    agents = len(positions)
    if private.shape != (agents, *truth.shape) or np.any(truth[positions[:, 0], positions[:, 1]] != 0):
        raise ValueError("observer_must_occupy_free_cell")
    alive = np.ones((agents, ray_offsets.shape[1]), dtype=bool)
    for offset in ray_offsets:
        row = positions[:, 0, None] + offset[None, :, 0]
        col = positions[:, 1, None] + offset[None, :, 1]
        alive &= (row >= 0) & (row < truth.shape[0]) & (col >= 0) & (col < truth.shape[1])
        ai, bi = np.nonzero(alive)
        if not len(ai):
            break
        rr, cc = row[ai, bi], col[ai, bi]
        value = truth[rr, cc]
        private[ai, rr, cc] = value
        # The occupied hit is observed; nothing further on that beam is read.
        alive[ai[value == 1], bi[value == 1]] = False
    return private


def radius_adjacency(positions, radius_cells):
    positions = np.asarray(positions, dtype=np.float64)
    distance = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    adjacency = distance <= radius_cells
    np.fill_diagonal(adjacency, False)
    return adjacency


def shortest_cell_path(truth, start, goal):
    start, goal = tuple(start), tuple(goal)
    queue = deque([start])
    parent = {start: None}
    while queue:
        p = queue.popleft()
        if p == goal:
            path = []
            while p != start:
                path.append(p)
                p = parent[p]
            return path[::-1]
        for dr, dc in NEIGHBORS:
            q = p[0] + dr, p[1] + dc
            if 0 <= q[0] < truth.shape[0] and 0 <= q[1] < truth.shape[1] and truth[q] == 0 and q not in parent:
                parent[q] = p
                queue.append(q)
    raise ValueError("unreachable_sampled_goal")


def generate_trajectory(truth, component, rng, args, rays):
    # Spread reachable start poses, including variable overlap/connectivity.
    indices = rng.choice(len(component), size=args.agents, replace=False)
    positions = component[indices].copy()
    private = np.full((args.agents, args.size, args.size), -1, dtype=np.int8)
    paths = [[] for _ in range(args.agents)]
    snapshots = []
    pose_history = []
    total_steps = (args.snapshots - 1) * args.steps_per_snapshot + 1
    for step in range(total_steps):
        if step:
            for agent in range(args.agents):
                if not paths[agent]:
                    goal = component[int(rng.integers(len(component)))]
                    paths[agent] = shortest_cell_path(truth, positions[agent], goal)
                if paths[agent]:
                    positions[agent] = paths[agent].pop(0)
        pose_history.append(positions.copy())
        raycast_observe(truth, positions, private, rays)
        if step % args.steps_per_snapshot == 0:
            snapshots.append(dict(private=private.copy(), positions=positions.copy(),
                                  active=np.ones(args.agents, dtype=bool),
                                  adjacency=radius_adjacency(positions, args.comm_radius_m / args.meters_per_cell)))
    return snapshots, np.asarray(pose_history, dtype=np.int64)


def validate_local_data(data):
    private = data["private"]
    if private.dtype != np.int8 or private.ndim != 4 or not np.isin(private, (-1, 0, 1)).all():
        raise ValueError("bad_private_schema")
    batch, agents, height, width = private.shape
    if data["positions"].shape != (batch, agents, 2) or data["positions"].dtype != np.int64:
        raise ValueError("bad_position_schema")
    if data["active"].shape != (batch, agents) or data["active"].dtype != bool:
        raise ValueError("bad_active_schema")
    if data["adjacency"].shape != (batch, agents, agents) or data["adjacency"].dtype != bool:
        raise ValueError("bad_adjacency_schema")
    for i in range(batch):
        if not np.array_equal(data["adjacency"][i], radius_adjacency(data["positions"][i], data["radius"][i])):
            raise ValueError("radius_graph_mismatch")
    if data["seed"].shape != (batch,) or data["seed"].dtype != np.int64:
        raise ValueError("bad_seed_schema")


def write_npz(path, arrays):
    np.savez_compressed(path, **arrays)
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=sha256_file(path))


def build(args):
    start_time = time.monotonic()
    if args.archive.stat().st_size > 100_000_000:
        raise ValueError("archive_exceeds_authorized_100MB_limit")
    archive_hash = sha256_file(args.archive)
    if archive_hash != SOURCE_SHA256 and not args.allow_fixture_archive:
        raise ValueError("archive_hash_differs_from_verified_official_source")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists():
        raise FileExistsError("refusing_to_replace_existing_pilot_manifest")
    rays = make_rays(args.sensor_range_m / args.meters_per_cell, args.beams)
    manifest = dict(version="houseexpo-metric-raycast-v121", source_url=SOURCE_URL,
                    source_reader=SOURCE_READER, archive_path=str(args.archive.resolve()),
                    archive_sha256=archive_hash, source_kind="synthetic SUNCG-derived floor contours, not real sensor scans",
                    input_fields=list(INPUT_FIELDS), test_loaded=False,
                    configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    split_rule="source ID hash before rasterization; 80/20 TRAIN/VALIDATION; exact normalized contour duplicates excluded",
                    semantics=dict(private="-1 unknown, 0 observed free, 1 observed occupied",
                                   seen_mask="private >= 0; additionally packed in separate evidence artifact, little bit order",
                                   position="row, column in public metric chart; shared exact coordinate frame assumed",
                                   radius="Euclidean cell units; multiply by fixed meters_per_cell for meters",
                                   graph="rows senders, columns receivers; symmetric finite-radius edges; no self loops",
                                   sensor="360-degree discrete DDA beams; free before first hit, occupied first hit, occluded unknown",
                                   trajectory="geometry-assisted reachable data sampler; all motion is cardinal adjacent-cell; not a learned exploration policy",
                                   raster="4x supersampled filled free contour, conservative min-pooling; fixed physical scale, center crop, no resized house",
                                   agents="point sensors, no robot-body collision or mutual occlusion model",
                                   truth="separate audit NPZ only, never an inference input or observed-union target",
                                   limitation="no furniture augmentation, pose noise, sensor noise, full-building coverage guarantee, or arbitrary similarity deduplication"),
                    splits={}, rejected=[], source_count=0)
    used_geometry = set()
    used_seeds = set()
    with tarfile.open(args.archive, "r:gz") as archive:
        members = {Path(member.name).stem: member for member in archive.getmembers()
                   if member.isfile() and member.name.endswith(".json") and member.size <= 2_000_000}
        manifest["source_count"] = len(members)
        partitions = partition_ids(members, args.seed)
        for split, requested in (("train", args.train_houses), ("validation", args.validation_houses)):
            rows, source_rows, house_records, truths, trajectories = [], [], [], [], []
            for source_id in partitions[split]:
                if len(house_records) == requested:
                    break
                raw = archive.extractfile(members[source_id]).read()
                try:
                    record = json.loads(raw)
                    if record.get("id", source_id) != source_id:
                        raise ValueError("filename_record_id_mismatch")
                    verts = validate_vertices(record)
                    geom_hash = geometry_hash(verts)
                    if geom_hash in used_geometry:
                        raise ValueError("duplicate_normalized_contour")
                    source_seed = stable_seed(f"source:{source_id}")
                    if source_seed in used_seeds:
                        raise ValueError("source_seed_collision")
                    rng = np.random.default_rng(stable_seed(f"crop:{args.seed}:{source_id}"))
                    truth, component, origin = crop_house(verts, args.size, args.meters_per_cell, rng)
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    manifest["rejected"].append(dict(id=source_id, split=split, reason=str(exc)))
                    continue
                house_index = len(house_records)
                used_geometry.add(geom_hash)
                used_seeds.add(source_seed)
                truths.append(truth)
                house_records.append(dict(id=source_id, seed=source_seed, source_json_sha256=hashlib.sha256(raw).hexdigest(),
                                          normalized_contour_sha256=geom_hash, origin_xy_m=origin.tolist(),
                                          extent_m=args.size * args.meters_per_cell, room_num=record.get("room_num"),
                                          full_bbox_xy_m=[verts.min(0).tolist(), verts.max(0).tolist()],
                                          crop_truncated=bool((verts.min(0) < origin).any() or
                                                              (verts.max(0) > origin + args.size * args.meters_per_cell).any()),
                                          reachable_free_cells=len(component), free_fraction=float((truth == 0).mean())))
                for trajectory in range(args.trajectories):
                    rng = np.random.default_rng(stable_seed(f"trajectory:{args.seed}:{source_id}:{trajectory}"))
                    snapshots, history = generate_trajectory(truth, component, rng, args, rays)
                    trajectories.append(history)
                    for snapshot_index, snapshot in enumerate(snapshots):
                        snapshot.update(seed=np.int64(source_seed), radius=np.float64(args.comm_radius_m / args.meters_per_cell))
                        rows.append(snapshot)
                        source_rows.append((house_index, trajectory, snapshot_index))
                print(json.dumps(dict(event="house_complete", split=split, houses=len(house_records), requested=requested,
                                      graphs=len(rows), elapsed_s=round(time.monotonic() - start_time, 2))), flush=True)
            if len(house_records) != requested:
                raise RuntimeError(f"not_enough_eligible_{split}_houses:{len(house_records)}/{requested}")
            data = {key: np.stack([row[key] for row in rows]) for key in INPUT_FIELDS}
            validate_local_data(data)
            # Check all observed facts against geometry, but never expose it to model files.
            row_index = np.asarray(source_rows, dtype=np.int64)
            truth_array = np.asarray(truths, dtype=np.int8)
            for graph_index, house_index in enumerate(row_index[:, 0]):
                observed = data["private"][graph_index] >= 0
                if np.any(observed & (data["private"][graph_index] != truth_array[house_index][None])):
                    raise AssertionError("observed_fact_disagrees_with_raster_truth")
            known = data["private"] >= 0
            model_file = write_npz(args.output / f"{split}.npz", data)
            evidence_file = write_npz(args.output / f"{split}-evidence.npz", dict(
                row_house_trajectory_snapshot=row_index,
                seen_packed=np.packbits(known, axis=-1, bitorder="little"),
                pose_history=np.stack(trajectories),
                source_id=np.asarray([record["id"] for record in house_records])))
            geometry_file = write_npz(args.output / f"{split}-geometry-AUDIT-ONLY.npz", dict(
                truth=truth_array, source_id=np.asarray([record["id"] for record in house_records])))
            degrees = data["adjacency"].sum(-1)
            manifest["splits"][split] = dict(houses=house_records, graphs=len(rows), model_file=model_file,
                                           evidence_file=evidence_file, geometry_audit_only=geometry_file,
                                           known_fraction=float(known.mean()), mean_degree=float(degrees.mean()),
                                           isolated_node_fraction=float((degrees == 0).mean()),
                                           observed_wall_fraction=float((data["private"] == 1).sum() / max(1, known.sum())))
            del data, known, rows, truth_array
    manifest["elapsed_s"] = time.monotonic() - start_time
    with open(args.output / "manifest.json", "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    print(json.dumps(dict(event="complete", manifest=str((args.output / "manifest.json").resolve()),
                          elapsed_s=manifest["elapsed_s"])), flush=True)
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("D:/sheaf-gbp-artifacts/occupancy_codec_20261014/houseexpo-source/json.tar.gz"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-houses", type=int, default=200)
    parser.add_argument("--validation-houses", type=int, default=40)
    parser.add_argument("--trajectories", type=int, default=4)
    parser.add_argument("--snapshots", type=int, default=8)
    parser.add_argument("--steps-per-snapshot", type=int, default=6)
    parser.add_argument("--agents", type=int, default=6)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--meters-per-cell", type=float, default=.1)
    parser.add_argument("--sensor-range-m", type=float, default=3.)
    parser.add_argument("--comm-radius-m", type=float, default=5.)
    parser.add_argument("--beams", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261014121)
    parser.add_argument("--allow-fixture-archive", action="store_true", help="Tests only: permit an archive other than pinned official SHA256")
    args = parser.parse_args(argv)
    if min(args.train_houses, args.validation_houses, args.trajectories, args.snapshots, args.steps_per_snapshot, args.agents) < 1:
        parser.error("counts must be positive")
    if args.meters_per_cell not in (.1, .2) or not 8 <= args.size <= 128:
        parser.error("fixed scale must be .1 or .2 meters/cell, chart size 8..128")
    return args


if __name__ == "__main__":
    build(parse_args())
