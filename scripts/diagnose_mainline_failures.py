#!/usr/bin/env python3
"""Describe completed failed episodes without inferring architecture limitations."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import torch

from audit_local_skill_composition import sha256, write_json
from piano_movers.local_skill_teacher import certify_pose_path
from piano_movers.multi_waypoint_demo import (
    Slit, WaypointCourse, _course_config, _full_collision_scenario,
)
from piano_movers.single_slit_0720_eval import load_single_slit_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    source = json.loads(args.rollouts.read_text())
    if source["status"] != "complete":
        raise ValueError("diagnostic report requires a complete audit snapshot")
    actor = load_single_slit_checkpoint(Path(source["checkpoint"]), device="cpu")
    if actor.checkpoint_sha256 != source["checkpoint_sha256"]:
        raise ValueError("actor checkpoint mismatch")
    rows = []
    for e in source["results"]:
        if e["result"]["success"] or "trace" not in e:
            continue
        path = args.rollouts.parent / e["trace"]
        if sha256(path) != e["trace_sha256"]:
            raise ValueError("authentic trajectory checksum mismatch")
        data = np.load(path)
        h = e["header"]
        course = WaypointCourse(
            tuple(h["source"]), tuple(Slit(**w) for w in h["walls"]),
            tuple(tuple(g) for g in h["waypoints"]), h["world"]["x"], h["world"]["y"],
            tuple(h.get("source_velocity", (0., 0., 0.))), h.get("morphology_scale", 1.),
        )
        cfg = _course_config(actor.cfg, e["agents"], course, e["shape"])
        if e["options"].get("nominal_glyph_size"):
            cfg = replace(cfg, glyph_scale=1.)
        scene = _full_collision_scenario(course, cfg)
        attempt = torch.from_numpy(np.concatenate((data["state"][-1:], data["attempted_state"][-1:])))
        certificates = []
        for i in range(len(course.walls)):
            single = replace(scene, wall_x=scene.wall_x[:, i:i + 1],
                             gap_y=scene.gap_y[:, i:i + 1], gap_half=scene.gap_half[:, i:i + 1])
            certificates.append(asdict(certify_pose_path(attempt, single, cfg)))
        tail = data["next_state"][-100:]
        stage = int(data["waypoint_index"][-1])
        rows.append({
            "case_id": e.get("case_id", f"{e['shape']}_{e['agents']}"),
            "shape": e["shape"], "agents": e["agents"], "profile": e.get("profile"),
            "trace_sha256": e["trace_sha256"], "result": e["result"],
            "stage": stage, "active_goal": h["waypoints"][stage],
            "last_state": tail[-1].tolist(), "attempted_state": attempt[-1].tolist(),
            "tail_xy_extent_m": np.ptp(tail[:, :2], axis=0).tolist(),
            "tail_heading_extent_rad": float(np.ptp(np.unwrap(tail[:, 2]))),
            "tail_mean_speed_mps": float(np.linalg.norm(tail[:, 3:5], axis=-1).mean()),
            "tail_rejection_fraction": float(data["hit"][-100:].mean()),
            "tail_geometric_plan_fraction": float(data["analytic_active"][-100:].mean()),
            "last_neural_authority_nodes": int((~data["analytic_active"][-1].astype(bool)).sum()),
            "last_plan_motion_spread": np.ptp(data["plan"][-1, :, :6], axis=0).tolist(),
            "last_attempt_certificate_by_wall": certificates,
        })
    report = {"schema": "piano-movers.mainline-failure-diagnostics.v1",
              "source_manifest": str(args.rollouts.resolve()),
              "source_manifest_sha256": sha256(args.rollouts),
              "checkpoint_sha256": actor.checkpoint_sha256,
              "interpretation": "descriptive diagnostics, not an architecture or memorization test",
              "results": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps({"diagnosed": len(rows)}))


if __name__ == "__main__":
    main()
