#!/usr/bin/env python3
"""Authenticate rollouts and audit rigid-world replay, NOT closed-loop success.

Forces are archived policy outputs rotated into each world frame. Independent
world OBB-SAT checks both poses and their continuous sweeps. Existing dynamics
are re-evaluated at each recorded state, not inferred from rendered images.
No policy is rerun and no new success-rate numerator is created by this audit.
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from audit_local_skill_composition import sha256, write_json
from piano_movers.env import collision_clearance_margin, contact_points, step_dynamics
from piano_movers.local_skill_teacher import repeat_scene
from piano_movers.multi_waypoint_demo import WaypointCourse, Slit, _course_config, _full_collision_scenario
from piano_movers.payload_shapes import payload_rectangles_for_config
from piano_movers.rigid_scene_audit import RigidFrame, angle_wrap, certify_world_path, world_scene_from_legacy
from piano_movers.single_slit_0720_eval import load_single_slit_checkpoint


FRAMES = (
    ("identity", RigidFrame()),
    ("translated", RigidFrame(0., (.63, -.4))),
    ("rotated37_translated", RigidFrame(np.deg2rad(37.), (.63, -.4))),
    ("rotated90_translated", RigidFrame(np.pi / 2, (.63, -.4))),
    ("rotated180_translated", RigidFrame(np.pi, (.63, -.4))),
    ("rotated_minus90_translated", RigidFrame(-np.pi / 2, (.63, -.4))),
)


def state_error(a, b):
    delta = a - b
    delta[:, 2] = angle_wrap(delta[:, 2])
    return float(np.abs(delta).max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agents", type=int, nargs="+", default=[8, 12, 20])
    parser.add_argument("--shapes", nargs="+", default=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = json.loads(args.manifest.read_text())
    if source["status"] != "complete":
        raise ValueError("incomplete source manifest")
    root = Path(__file__).resolve().parents[1]
    checkpoint = Path(source["checkpoint"])
    if sha256(checkpoint) != source["checkpoint_sha256"]:
        raise ValueError("changed source checkpoint")
    torch.set_num_threads(1)
    base = load_single_slit_checkpoint(checkpoint, device=torch.device("cpu")).cfg
    selected = [r for r in source["results"] if r["agents"] in args.agents and r["shape"] in args.shapes]
    if not selected:
        raise ValueError("no selected cases")
    report = {
        "schema": "piano-movers.rigid-world-replay-audit.v1", "status": "running",
        "source_manifest": str(args.manifest.resolve()), "source_manifest_sha256": sha256(args.manifest),
        "checkpoint_sha256": source["checkpoint_sha256"], "policy_rerun": False,
        "audit_kind": "one_step_dynamics_covariance_and_continuous_replay_geometry",
        "not_evidence_for": ["new_closed_loop_success_rate", "learned_SE2_equivariance",
                             "changing_wall_normals", "frame_change_memory", "maze_navigation"],
        "frames": [{"name": n, **asdict(f)} for n, f in FRAMES],
        "selected_source_episodes": len(selected), "expected_frame_checks": len(selected) * len(FRAMES),
        "selection": {"shapes": args.shapes, "agents": args.agents, "outcome_filtered": False},
        "source_sha256": {str(p.relative_to(root)): sha256(p) for p in (
            Path(__file__), root / "src/piano_movers/rigid_scene_audit.py",
            root / "src/piano_movers/env.py", root / "src/piano_movers/multi_waypoint_demo.py")},
        "tolerances": {"world_clearance_m": 1e-10, "dynamics_state_max_abs": 1e-10,
                       "recorded_float32_attempted_state_max_abs": 1e-5},
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)  # Freeze selections before any result.
    started = time.monotonic()
    for index, row in enumerate(selected):
        h = row["header"]
        course = WaypointCourse(tuple(h["source"]), tuple(Slit(**w) for w in h["walls"]),
                                tuple(tuple(w) for w in h["waypoints"]), h["world"]["x"], h["world"]["y"],
                                tuple(h["source_velocity"]), h["morphology_scale"])
        cfg = replace(_course_config(base, row["agents"], course, row["shape"]), glyph_scale=h["glyph_scale"])
        scene = _full_collision_scenario(course, cfg)
        body = payload_rectangles_for_config(cfg, torch.device("cpu")).double().numpy()
        if not np.array_equal(body, h["body_rects"]):
            raise ValueError("source payload geometry mismatch")
        handles = contact_points(cfg, torch.device("cpu")).double().numpy()
        if not np.array_equal(handles, h["handles"]):
            raise ValueError("source handle geometry mismatch")
        geometry = world_scene_from_legacy(scene, cfg, body)
        trace_path = args.manifest.parent / row["trace"]
        if sha256(trace_path) != row["trace_sha256"]:
            raise ValueError("changed source trace")
        with np.load(trace_path) as trace:
            state = trace["state"].astype(np.float64)
            forces = trace["forces"].astype(np.float64)
            recorded_attempt = trace["attempted_state"].astype(np.float64)
            path = np.concatenate((state, trace["next_state"][-1:])).astype(np.float64)
            hits = int(trace["hit"].sum())
        state_tensor = torch.from_numpy(state)
        local_next = step_dynamics(state_tensor, torch.from_numpy(forces), cfg).numpy()
        attempt_error = state_error(local_next, recorded_attempt)
        local_clearance = geometry.clearance(path)
        pt = torch.from_numpy(path)
        legacy_clearance = collision_clearance_margin(pt, repeat_scene(scene, len(pt), pt), cfg).numpy()
        legacy_error = float(np.abs(local_clearance - legacy_clearance).max())
        checks = []
        for name, frame in FRAMES:
            world = geometry.transformed(frame)
            world_path = frame.states(path)
            certificate = certify_world_path(world_path, world)
            clearance_error = float(np.abs(world.clearance(world_path) - local_clearance).max())
            world_next = step_dynamics(torch.from_numpy(frame.states(state)),
                                       torch.from_numpy(frame.vectors(forces)), cfg).numpy()
            dynamics_error = state_error(world_next, frame.states(local_next))
            # Rigid transforms preserve actual communication distances; no
            # message bytes are discounted because the frame changed.
            transformed_handles = frame.points(handles)
            distance_error = float(np.abs(
                np.linalg.norm(transformed_handles[:, None] - transformed_handles[None], axis=-1)
                - np.linalg.norm(handles[:, None] - handles[None], axis=-1)).max())
            passed = (certificate.passed and clearance_error < 1e-10 and dynamics_error < 1e-10
                      and distance_error < 1e-10 and legacy_error < 1e-10 and attempt_error < 1e-5)
            checks.append({"frame": name, "passed": passed, "certificate": asdict(certificate),
                           "world_clearance_max_difference_m": clearance_error,
                           "dynamics_covariance_max_difference": dynamics_error,
                           "handle_pair_distance_max_difference_m": distance_error})
        report["results"].append({"source_case": row.get("case_id", f"{row['shape']}_n{row['agents']}"),
            "shape": row["shape"], "agents": row["agents"], "source_trace_sha256": row["trace_sha256"],
            "source_success": row["result"]["success"], "source_collision_attempts": hits,
            "recorded_state_count": len(state), "legacy_clearance_max_difference_m": legacy_error,
            "float32_recorded_attempt_max_difference": attempt_error, "checks": checks})
        write_json(args.output, report)
        print(json.dumps({"done": index + 1, "total": len(selected), "shape": row["shape"],
                          "agents": row["agents"], "all_frame_checks_passed": all(c["passed"] for c in checks)}), flush=True)
    report.update(status="complete", elapsed_seconds=time.monotonic() - started,
                  passed_frame_checks=sum(c["passed"] for r in report["results"] for c in r["checks"]))
    write_json(args.output, report)
    print(json.dumps({k: report[k] for k in ("status", "selected_source_episodes", "passed_frame_checks",
                                            "expected_frame_checks", "elapsed_seconds")}))
    if report["passed_frame_checks"] != report["expected_frame_checks"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
