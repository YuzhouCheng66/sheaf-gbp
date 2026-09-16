#!/usr/bin/env python3
"""Check zero-input-bridge parity on genuine model-induced states and commands."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from audit_local_skill_composition import sha256, write_json
from piano_movers.env import radius_edges
from piano_movers.multi_waypoint_demo import Slit, WaypointCourse, _course_config, INTERMEDIATE_POSITION_TOLERANCE_M
from piano_movers.single_slit_0720_eval import load_single_slit_checkpoint
from train_mainline_skill_dagger import active_anchor_config


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--shapes", nargs="+", default=["A", "B", "T", "X"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    parent, child = [load_single_slit_checkpoint(p, device="cpu") for p in (args.parent, args.bridge)]
    source = json.loads(args.rollouts.read_text())
    if source["status"] != "complete" or source["checkpoint_sha256"] != parent.checkpoint_sha256:
        raise ValueError("requires complete, checksum-matched actor rollouts")
    if child.metadata["goal_context_bridge"]["parent_checkpoint_sha256"] != parent.checkpoint_sha256:
        raise ValueError("bridge source mismatch")
    results = []
    for row in source["results"]:
        if row["shape"] not in args.shapes or "trace" not in row:
            continue
        path = args.rollouts.parent / row["trace"]
        if sha256(path) != row["trace_sha256"]:
            raise ValueError("actor trace hash mismatch")
        arrays, header, options = np.load(path), row["header"], row["options"]
        course = WaypointCourse(tuple(header["source"]), tuple(Slit(**s) for s in header["walls"]),
                                tuple(tuple(g) for g in header["waypoints"]),
                                header["world"]["x"], header["world"]["y"],
                                tuple(header.get("source_velocity", (0., 0., 0.))),
                                float(header.get("morphology_scale", 1.)))
        cfg = _course_config(parent.cfg, row["agents"], course, row["shape"])
        if options["nominal_glyph_size"]:
            cfg = replace(cfg, glyph_scale=1.)
        cfg = replace(cfg, complete_local_corridor_exit=options["complete_corridor_exit"],
                      local_cspace_collision_substeps=options["planning_substeps"],
                      local_cspace_validate_all_lookaheads=options.get("validate_all_cspace_lookaheads", False),
                      local_support_check_alignment_sweep=options.get("check_support_alignment_sweep", False),
                      local_corridor_exit_goal_radius_m=(INTERMEDIATE_POSITION_TOLERANCE_M
                                                         if options.get("respect_handoff_goal_region", False) else 0.))
        edges = radius_edges(cfg, device="cpu")
        count = 0
        for stage in np.unique(arrays["waypoint_index"]):
            active = active_anchor_config(cfg, course, int(stage), options["position_only_control"])
            for p in (parent.policy, child.policy):
                p.bind_environment(active)
            indices = np.flatnonzero(arrays["waypoint_index"] == stage)
            for ix in sorted(set(np.linspace(0, len(indices) - 1, 3, dtype=int))):
                tick = int(indices[ix])
                obs = torch.from_numpy(arrays["observation"][tick:tick + 1])
                hidden = torch.from_numpy(arrays["private_hidden"][tick:tick + 1])
                before, after = [p(obs, edges, hidden) for p in (parent.policy, child.policy)]
                for key in ("local_se2_plan_mean", "local_se2_plan_used"):
                    torch.testing.assert_close(before.aux[key], after.aux[key], atol=0., rtol=0.)
                torch.testing.assert_close(before.forces, after.forces, atol=0., rtol=0.)
                torch.testing.assert_close(before.aux["local_se2_plan_used"],
                                           torch.from_numpy(arrays["plan"][tick:tick + 1]),
                                           atol=2e-5, rtol=2e-5)
                count += 1
        results.append({"case_id": row.get("case_id"), "shape": row["shape"],
                        "trace_sha256": row["trace_sha256"], "states": count,
                        "plan_and_force_bitwise_equal": True})
    if not results:
        raise ValueError("no actual states checked")
    result = {"parent_sha256": parent.checkpoint_sha256, "bridge_sha256": child.checkpoint_sha256,
              "actor_manifest_sha256": sha256(args.rollouts), "status": "complete",
              "results": results, "states_checked": sum(r["states"] for r in results),
              "scope": "sampled actual states, not a closed-loop success claim"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "results"}))


if __name__ == "__main__":
    main()
