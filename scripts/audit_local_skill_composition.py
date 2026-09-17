#!/usr/bin/env python3
"""Collect frozen-checkpoint composition controls and authentic visited states.

Outputs are diagnostic rollouts, not teacher-labelled DAgger training data.
Each ablation changes one declared composition contract. The baseline restores
the shipped p5 demo behavior exactly, including the historical orientation
constraint and early waypoint handoff.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from piano_movers.env import collision_mask, transition_clearance_margin
from piano_movers.multi_waypoint_demo import MultiWaypointEngine, SkillCompositionOptions
from piano_movers.local_skill_teacher import certify_pose_path


CONTROLS = {
    "baseline": SkillCompositionOptions(False, False),
    "position_only": SkillCompositionOptions(True, False),
    "clear_handoff": SkillCompositionOptions(False, True),
    "composed": SkillCompositionOptions(True, True),
    "preserve_memory": SkillCompositionOptions(True, True, True),
    "nominal_size": SkillCompositionOptions(False, False, False, True),
    "nearby_walls": SkillCompositionOptions(retain_nearby_wall=True),
    "contact_history": SkillCompositionOptions(preserve_contact_feedback=True),
    "previous_wall": SkillCompositionOptions(retain_previous_wall_only=True),
    "complete_exit": SkillCompositionOptions(complete_corridor_exit=True),
    "complete_exit_certified": SkillCompositionOptions(
        complete_corridor_exit=True, planning_substeps=32, continuous_contact_guard=True,
    ),
    "complete_exit_screened": SkillCompositionOptions(
        complete_corridor_exit=True, planning_substeps=32, continuous_contact_guard=True,
        validate_all_cspace_lookaheads=True,
    ),
    "complete_exit_swept": SkillCompositionOptions(
        complete_corridor_exit=True, planning_substeps=32, continuous_contact_guard=True,
        validate_all_cspace_lookaheads=True, check_support_alignment_sweep=True,
    ),
    "complete_exit_consistent": SkillCompositionOptions(
        complete_corridor_exit=True, planning_substeps=32, continuous_contact_guard=True,
        validate_all_cspace_lookaheads=True, check_support_alignment_sweep=True,
        respect_handoff_goal_region=True,
    ),
    "complete_exit_nearby": SkillCompositionOptions(
        complete_corridor_exit=True, retain_nearby_wall=True,
        preserve_private_memory=True, preserve_contact_feedback=True,
    ),
    "visible_position_handoff": SkillCompositionOptions(
        require_body_clearance=False, preserve_private_memory=True,
        retain_nearby_wall=True, preserve_contact_feedback=True,
    ),
    "coherent_handoff": SkillCompositionOptions(
        preserve_private_memory=True, retain_nearby_wall=True, preserve_contact_feedback=True,
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class Recorder:
    def __init__(self) -> None:
        self.arrays: dict[str, list[np.ndarray]] = {}
        self.first_hit = None
        self.minimum_clearance = float("inf")
        self.resolved_pose_collisions = 0
        self.path_length = 0.0
        self.payload_bytes = 0
        self.setup_bytes = 0
        self.analytic_active = 0
        self.agent_steps = 0

    def __call__(self, **values) -> None:
        output, transition = values["output"], values["transition"]
        state, scenario, cfg = values["state"], values["scenario"], values["cfg"]
        self.last_scenario, self.last_cfg = scenario, cfg
        arrays = {
            "state": state[0],
            "next_state": transition.resolved_state[0],
            "attempted_state": transition.attempted_state[0],
            "observation": values["observation"][0],
            "private_hidden": values["hidden"][0],
            "forces": output.forces[0],
            "plan": output.aux["local_se2_plan_used"][0],
            "learned_plan": output.aux["local_se2_plan_mean"][0],
            "analytic_active": output.aux["local_se2_terminal_attractor_active"][0],
            "hit": transition.attempted_hit[0],
            "resolved_transition_clearance": transition.resolved_clearance_m[0],
        }
        for name, tensor in arrays.items():
            self.arrays.setdefault(name, []).append(tensor.detach().cpu().numpy().copy())
        if "local_se2_contact_twist_residual" in output.aux:
            self.arrays.setdefault("contact_twist_residual", []).append(
                output.aux["local_se2_contact_twist_residual"][0].detach().cpu().numpy().copy())
        for key in ("tick", "waypoint_index"):
            self.arrays.setdefault(key, []).append(np.asarray(values[key], dtype=np.int32))
        self.minimum_clearance = min(
            self.minimum_clearance, float(transition.resolved_clearance_m[0])
        )
        self.resolved_pose_collisions += int(
            collision_mask(transition.resolved_state, scenario, cfg)[0]
        )
        self.path_length += float(torch.linalg.vector_norm(
            transition.resolved_state[0, :2] - state[0, :2]
        ))
        trace = output.aux["gbp_message_trace"]
        packet_bytes = trace.wire_float_count * trace.natural.element_size()
        self.payload_bytes += packet_bytes
        setup = output.aux.get("local_se2_distributed_grasp_message_trace")
        setup_bytes = 0 if setup is None else setup.numel() * setup.element_size()
        self.setup_bytes += setup_bytes
        self.arrays.setdefault("gbp_payload_bytes", []).append(np.asarray(packet_bytes))
        self.arrays.setdefault("setup_payload_bytes", []).append(np.asarray(setup_bytes))
        active = arrays["analytic_active"]
        self.analytic_active += int(active.sum())
        self.agent_steps += active.numel()
        if bool(transition.attempted_hit[0]) and self.first_hit is None:
            clearances = []
            for index in range(scenario.wall_x.shape[1]):
                part = replace(
                    scenario,
                    wall_x=scenario.wall_x[:, index:index + 1],
                    gap_y=scenario.gap_y[:, index:index + 1],
                    gap_half=scenario.gap_half[:, index:index + 1],
                )
                clearances.append(float(transition_clearance_margin(
                    state, transition.attempted_state, part, cfg
                )[0]))
            self.first_hit = {
                "step": values["tick"],
                "waypoint_index": values["waypoint_index"],
                "state": state[0].tolist(),
                "swept_clearance_per_wall_m": clearances,
            }

    def save(self, path: Path) -> None:
        np.savez_compressed(path, **{
            name: np.stack(items) for name, items in self.arrays.items()
        })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--recipe", type=Path,
                        help="Locked checkpoint and composition options; exclusive with --variants.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shapes", nargs="+", default=["A", "B", "L", "T", "X"])
    parser.add_argument("--agents", type=int, nargs="+", default=[12])
    parser.add_argument("--variants", nargs="+", choices=tuple(CONTROLS))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    recipe = None
    controls = CONTROLS
    if args.recipe is not None:
        if args.checkpoint is not None or args.variants is not None:
            parser.error("--recipe is exclusive with --checkpoint and --variants")
        recipe = json.loads(args.recipe.read_text())
        checkpoint = (root / recipe["checkpoint"]).resolve(strict=True)
        if sha256(checkpoint) != recipe["checkpoint_sha256"]:
            raise ValueError("recipe checkpoint hash mismatch")
        controls = {"recipe": SkillCompositionOptions(**recipe["composition_options"])}
        args.variants = ["recipe"]
    else:
        if args.checkpoint is None:
            parser.error("provide --checkpoint or --recipe")
        checkpoint = args.checkpoint.resolve(strict=True)
        args.variants = args.variants or list(CONTROLS)
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "manifest.json"
    if summary_path.exists():
        raise FileExistsError(f"Use a fresh output directory: {summary_path}")
    torch.set_num_threads(1)
    record = {
        "schema": "piano-movers.p6-composition-controls.v3",
        "status": "running",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "source_sha256": {
            str(path.relative_to(root)): sha256(path)
            for path in [Path(__file__).resolve(), *sorted((root / "src/piano_movers").glob("*.py"))]
        },
        "device": args.device,
        "teacher_labels_present": False,
        "split": "development_diagnostics",
        "requested": {"shapes": args.shapes, "agents": args.agents, "variants": args.variants},
        "results": [],
    }
    if recipe is not None:
        record.update(recipe=str(args.recipe.resolve()), recipe_sha256=sha256(args.recipe),
                      recipe_snapshot=recipe)
    write_json(summary_path, record)
    for variant in args.variants:
        engine = MultiWaypointEngine(checkpoint, args.device, options=controls[variant])
        for shape in args.shapes:
            for count in args.agents:
                started = time.monotonic()
                recorder = Recorder()
                engine.audit_callback = recorder
                handoffs = []
                for event in engine.events(count, shape):
                    if event["type"] == "header":
                        header = event
                    if event["type"] == "frame" and event["waypoint_reached"]:
                        handoffs.append({
                            "step": event["step"], "next_stage": event["waypoint_index"],
                            "state": event["next_state"],
                        })
                    if event["type"] == "done":
                        done = event
                trace_path = args.output / f"{variant}_{shape}_n{count}.npz"
                recorder.save(trace_path)
                trajectory = np.concatenate((
                    np.stack(recorder.arrays["state"]),
                    np.asarray(recorder.arrays["next_state"][-1])[None],
                ))
                certificate = certify_pose_path(
                    torch.from_numpy(trajectory), recorder.last_scenario,
                    replace(recorder.last_cfg, device="cpu"),
                )
                result = {
                    "variant": variant, "options": asdict(controls[variant]),
                    "shape": shape, "agents": count,
                    "header": header, "result": done, "handoffs": handoffs,
                    "first_rejection": recorder.first_hit,
                    "minimum_resolved_transition_clearance_m": recorder.minimum_clearance,
                    "independent_resolved_pose_collisions": recorder.resolved_pose_collisions,
                    "path_length_m": recorder.path_length,
                    "gbp_float_payload_bytes": recorder.payload_bytes,
                    "grasp_setup_payload_bytes": recorder.setup_bytes,
                    "wire_headers_included": False,
                    "continuous_certificate": asdict(certificate),
                    "analytic_plan_agent_step_fraction": recorder.analytic_active / recorder.agent_steps,
                    "elapsed_seconds": time.monotonic() - started,
                    "trace": trace_path.name, "trace_sha256": sha256(trace_path),
                }
                record["results"].append(result)
                write_json(summary_path, record)
                print(json.dumps({
                    "variant": variant, "shape": shape, "agents": count, **done,
                    "first_rejection": recorder.first_hit,
                    "analytic_fraction": result["analytic_plan_agent_step_fraction"],
                    "seconds": result["elapsed_seconds"],
                }), flush=True)
    record["status"] = "complete"
    write_json(summary_path, record)


if __name__ == "__main__":
    main()
