#!/usr/bin/env python3
"""Verify force parity, real packets and live action gradients on actual states."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from audit_local_skill_composition import sha256, write_json
from train_mainline_skill_dagger import active_anchor_config
from piano_movers.actuation_training import replay_physical_actuation
from piano_movers.multi_waypoint_demo import (
    Slit, WaypointCourse, _course_config, SkillCompositionOptions, apply_solver_controls,
)
from piano_movers.single_slit_0720_eval import load_single_slit_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shapes", nargs="+", default=list("ABCDIOTX"))
    parser.add_argument("--states-per-case", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    source = json.loads(args.rollouts.read_text())
    if source["status"] != "complete":
        raise ValueError("source audit is incomplete")
    loaded = load_single_slit_checkpoint(Path(source["checkpoint"]), device="cpu")
    if loaded.checkpoint_sha256 != source["checkpoint_sha256"]:
        raise ValueError("checkpoint mismatch")
    rows = []
    for row in source["results"]:
        if row["shape"] not in args.shapes or "trace" not in row:
            continue
        path = args.rollouts.parent / row["trace"]
        if sha256(path) != row["trace_sha256"]:
            raise ValueError("trace mismatch")
        h = row["header"]
        course = WaypointCourse(tuple(h["source"]), tuple(Slit(**w) for w in h["walls"]),
                                tuple(tuple(g) for g in h["waypoints"]), h["world"]["x"], h["world"]["y"],
                                tuple(h.get("source_velocity", (0., 0., 0.))), h.get("morphology_scale", 1.))
        options = SkillCompositionOptions(**row["options"])
        cfg = _course_config(loaded.cfg, row["agents"], course, row["shape"])
        if options.nominal_glyph_size:
            cfg = replace(cfg, glyph_scale=1.)
        with np.load(path) as data:
            for tick in np.unique(np.linspace(0, len(data["state"]) - 1, args.states_per_case, dtype=int)):
                loaded.policy.bind_environment(active_anchor_config(cfg, course, int(data["waypoint_index"][tick]),
                                                                    options.position_only_control))
                apply_solver_controls(loaded.policy, options)
                obs = torch.from_numpy(data["observation"][tick:tick + 1].copy())
                private = torch.from_numpy(data["learned_plan"][tick:tick + 1].copy()).requires_grad_()
                geometry = torch.from_numpy(data["plan"][tick:tick + 1].copy())
                active = torch.from_numpy(data["analytic_active"][tick:tick + 1].copy()).bool()
                replay = replay_physical_actuation(loaded.policy, obs, private, geometry, active,
                                                   trace_messages=True)
                target = torch.from_numpy(data["forces"][tick:tick + 1].copy())
                torch.testing.assert_close(replay.forces, target, atol=2e-5, rtol=2e-5)
                packet_bytes = replay.message_trace.wire_float_count * replay.message_trace.natural.element_size()
                if packet_bytes != int(data["gbp_payload_bytes"][tick]):
                    raise ValueError("offline action path changed actual GBP payload")
                probe = torch.linspace(.1, 1., replay.forces.numel()).reshape_as(replay.forces)
                grad = torch.autograd.grad((replay.forces * probe).sum(), private)[0]
                if bool(grad[..., :3].count_nonzero()) or bool(grad[..., 6:].count_nonzero()):
                    raise ValueError("unexpected actuation dependency on unused plan coordinates")
                if bool(torch.where(active, grad, 0.).count_nonzero()):
                    raise ValueError("a geometry-overridden private plan acquired spurious action gradient")
                rows.append({"case_id": row.get("case_id"), "shape": row["shape"], "tick": int(tick),
                             "source_trace_sha256": row["trace_sha256"],
                             "force_max_abs_error_N": float((replay.forces.detach() - target).abs().max()),
                             "neural_authority_nodes": int((~active).sum()),
                             "max_action_gradient": float(grad.abs().max()),
                             "wire_payload_bytes": packet_bytes})
    if not rows or not any(r["max_action_gradient"] > 1e-5 for r in rows):
        raise ValueError("no authenticated live action gradient")
    report = {"status": "complete", "checkpoint_sha256": loaded.checkpoint_sha256,
              "rollout_manifest_sha256": sha256(args.rollouts), "results": rows,
              "states_checked": len(rows),
              "maximum_force_error_N": max(r["force_max_abs_error_N"] for r in rows),
              "states_with_live_action_gradient": sum(r["max_action_gradient"] > 1e-5 for r in rows),
              "scope": "offline exact actuation replay and gradient audit, not a new trained-policy success claim"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "results"}))


if __name__ == "__main__":
    main()
