#!/usr/bin/env python3
"""Query executable recovery teachers on authentic frozen-policy states."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from piano_movers.env import collision_mask, wrap_angle
from piano_movers.local_skill_teacher import (
    LocalTeacherConfig,
    certify_pose_path,
    executable_teacher_step,
    plan_local_target,
)
from piano_movers.multi_waypoint_demo import (
    DEFAULT_COURSE,
    _course_config,
    _full_collision_scenario,
)
from piano_movers.single_slit_0720_eval import load_single_slit_checkpoint
from audit_local_skill_composition import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rollout-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shapes", nargs="+", default=["L", "A", "C"])
    parser.add_argument("--x-step", type=float, default=0.01)
    parser.add_argument("--theta-bins", type=int, default=360)
    parser.add_argument("--horizon", type=int, default=512)
    parser.add_argument("--state-selection", choices=["last", "pre_contact"], default="last")
    args = parser.parse_args()
    torch.set_num_threads(1)
    parent = load_single_slit_checkpoint(args.checkpoint, device="cpu")
    source = json.loads(args.rollout_manifest.read_text())
    if source["checkpoint_sha256"] != parent.checkpoint_sha256:
        raise ValueError("checkpoint/rollout lineage mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    options = LocalTeacherConfig(x_step=args.x_step, theta_bins=args.theta_bins)
    manifest = {
        "schema": "piano-movers.p6-local-teacher-probe.v1",
        "status": "running",
        "checkpoint_sha256": parent.checkpoint_sha256,
        "rollout_manifest_sha256": sha256(args.rollout_manifest),
        "options": asdict(options),
        "state_selection": args.state_selection,
        "teacher_privileged": True,
        "split": "development_diagnostics",
        "results": [],
    }
    write_json(manifest_path, manifest)
    for shape in args.shapes:
        rows = [r for r in source["results"] if r["shape"] == shape]
        if len(rows) != 1:
            raise ValueError(f"expected one rollout for {shape}")
        row = rows[0]
        artifact = args.rollout_manifest.parent / row["trace"]
        if sha256(artifact) != row["trace_sha256"]:
            raise ValueError("rollout artifact SHA mismatch")
        trace = np.load(artifact)
        tick = len(trace["state"]) - 1
        if args.state_selection == "pre_contact" and np.any(trace["hit"]):
            tick = int(np.flatnonzero(trace["hit"])[0])
        state = torch.from_numpy(trace["state"][tick].copy()).view(1, 6)
        index = int(trace["waypoint_index"][tick])
        goal = state.new_tensor(DEFAULT_COURSE.waypoints[index])
        cfg = _course_config(parent.cfg, row["agents"], DEFAULT_COURSE, shape)
        if row["options"]["nominal_glyph_size"]:
            cfg = replace(cfg, glyph_scale=1.0)
        geometry = _full_collision_scenario(DEFAULT_COURSE, cfg)
        geometry = replace(geometry, state0=state, source=state[:, :3], goal=goal.view(1, 3))
        orientation = index == len(DEFAULT_COURSE.waypoints) - 1
        start_time = time.monotonic()
        path = plan_local_target(
            state, goal, geometry, cfg, orientation_required=orientation, options=options
        )
        record = {
            "shape": shape,
            "agents": row["agents"],
            "source_trace": str(artifact),
            "source_tick": tick,
            "state": state[0].tolist(),
            "goal": goal.tolist(),
            "orientation_required": orientation,
            "planning_status": path.status,
            "planning_seconds": time.monotonic() - start_time,
            "expansions": path.expansions,
            "lazy_replans": path.lazy_replans,
            "effective_planning_margin_m": path.planning_margin_m,
            "geometric_certificate": None if path.certificate is None else asdict(path.certificate),
            "executable_success": False,
        }
        if path.poses is not None:
            planned = torch.as_tensor(path.poses, dtype=state.dtype)
            trajectory, forces, modes = [state[0].clone()], [], []
            dwell = 0
            for _step in range(args.horizon):
                error = torch.linalg.vector_norm(state[0, :2] - goal[:2])
                angle_error = wrap_angle(state[0, 2] - goal[2]).abs()
                close = bool(error <= 0.04) and (not orientation or bool(angle_error <= 0.04))
                at_rest = bool(torch.linalg.vector_norm(state[0, 3:5]) < 0.02) and bool(
                    state[0, 5].abs() < 0.03
                )
                dwell = dwell + 1 if close and at_rest else 0
                if dwell >= 6:
                    record["executable_success"] = True
                    break
                # Once in the admitted goal region, explicitly settle rather
                # than chase a tolerance-boundary path point indefinitely.
                target_path = state[:, :3].clone() if close else planned
                f, next_state, mode = executable_teacher_step(
                    state, target_path, geometry, cfg, options
                )
                if f is None:
                    record["execution_failure"] = mode
                    break
                if bool(collision_mask(next_state, geometry, cfg)[0]):
                    raise RuntimeError("independent SAT rejected the certified teacher step")
                forces.append(f[0].clone())
                modes.append(mode)
                state = next_state
                trajectory.append(state[0].clone())
            trajectory = torch.stack(trajectory)
            result_path = args.output / f"{shape}_n{row['agents']}_teacher.npz"
            full_certificate = (
                certify_pose_path(trajectory, geometry, cfg, options)
                if len(trajectory) > 1
                else None
            )
            record.update(
                {
                    "steps": len(forces),
                    "goal_position_error_m": float(
                        torch.linalg.vector_norm(state[0, :2] - goal[:2])
                    ),
                    "brake_steps": modes.count("brake"),
                    "executed_certificate": None
                    if full_certificate is None
                    else asdict(full_certificate),
                    "elapsed_seconds": time.monotonic() - start_time,
                    "artifact": result_path.name,
                }
            )
            np.savez_compressed(
                result_path,
                planned_pose=path.poses,
                executed_state=trajectory.numpy(),
                forces=np.stack([f.numpy() for f in forces])
                if forces
                else np.empty((0, cfg.n_agents, 2)),
            )
            record["artifact_sha256"] = sha256(result_path)
        manifest["results"].append(record)
        write_json(manifest_path, manifest)
        print(json.dumps(record), flush=True)
    manifest["status"] = "complete"
    write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
