#!/usr/bin/env python3
"""Matched nominal / one-impulse / feedback-handoff recovery stress panel.

The environment applies a declared impulse, never changes the agent forces,
and uses the same continuously guarded contact physics thereafter. This is
not a new nominal success benchmark. Untriggered/non-contact cases remain in
the denominator and are not counted as demonstrated contact recovery.
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from audit_local_skill_composition import Recorder, sha256, write_json
from piano_movers.local_skill_disturbance import ContactImpulseOptions, SingleContactImpulse
from piano_movers.local_skill_teacher import certify_pose_path
from piano_movers.local_skill_validation import FAMILY_SEEDS, STATE_PROFILES, state_profile_course, validate_split_seeds
from piano_movers.multi_waypoint_demo import MultiWaypointEngine, SkillCompositionOptions
from piano_movers.contact_residual_control import load_contact_residual, contact_encoding_difference


VARIANTS = ("no_impulse", "impulse_reset_feedback", "impulse_preserve_feedback", "impulse_preserve_all_memory")


class ContactRecoveryEngine(MultiWaypointEngine):
    def __init__(self, *args, impulse_options, residual=None, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.options.continuous_contact_guard:
            raise ValueError("stress panel requires continuous contact guarding")
        self.impulse = SingleContactImpulse(impulse_options)
        self.residual = (load_contact_residual(residual, self.loaded.checkpoint_sha256)
                         if residual is not None else None)

    def _contact_step(self, state, forces, scenario, cfg):
        return self.impulse(state, forces, scenario, cfg)

    def _policy_step(self, target, observation, edges, hidden, topology):
        if self.residual is None:
            return super()._policy_step(target, observation, edges, hidden, topology)
        private = target.local_se2_base_planner_inputs(observation, hidden)
        features = target.local_se2_markov_shape_inputs_from_planner_inputs(private)
        delta = self.residual(contact_encoding_difference(target, features))
        return target(observation, edges, hidden, topology=topology, trace_messages=True,
                      local_se2_twist_residual=delta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=Path("configs/p6_force_position_heading_control.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=tuple(FAMILY_SEEDS), default="development_diagnostics")
    parser.add_argument("--seed", type=int, default=81201)
    parser.add_argument("--shapes", nargs="+", default=list("ABLTXZ"))
    parser.add_argument("--agents", nargs="+", type=int, default=[8, 12, 20])
    parser.add_argument("--profiles", nargs="+", choices=STATE_PROFILES, default=["entry_reference"])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS[:3])
    parser.add_argument("--trigger-clearance", type=float, default=.03)
    parser.add_argument("--delta-speed", type=float, default=.65)
    parser.add_argument("--delta-omega", type=float, default=0.)
    parser.add_argument("--resume", action="store_true", help="Authenticate and continue an interrupted prefix")
    parser.add_argument("--contact-residual", type=Path, help="Independent learned-contact control, not mainline")
    args = parser.parse_args()
    validate_split_seeds(args.split, [args.seed])
    for values in (args.shapes, args.agents, args.profiles, args.variants):
        if not values or len(values) != len(set(values)):
            raise ValueError("case dimensions must be nonempty and unique")
    if any(s not in list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for s in args.shapes):
        raise ValueError("uppercase letter shapes required")
    if any(n not in range(8, 21) for n in args.agents):
        raise ValueError("agent counts must be 8-20")
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[1]
    recipe = json.loads(args.recipe.read_text())
    checkpoint = (root / recipe["checkpoint"]).resolve(strict=True)
    if sha256(checkpoint) != recipe["checkpoint_sha256"]:
        raise ValueError("candidate checkpoint mismatch")
    options = SkillCompositionOptions(**recipe["composition_options"])
    impulse = ContactImpulseOptions(trigger_clearance_m=args.trigger_clearance,
                                    delta_speed_mps=args.delta_speed, delta_omega_radps=args.delta_omega)
    cases = [{"case_id": f"{v}_{s}_n{n}_{p}", "variant": v, "shape": s, "agents": n,
              "profile": p, "family": "combined", "seed": args.seed,
              "course": asdict(state_profile_course(args.seed, "combined", p))}
             for s in args.shapes for n in args.agents for p in args.profiles for v in args.variants]
    args.output.mkdir(parents=True, exist_ok=args.resume)
    catalog_path = args.output / "case_catalog.json"
    if not args.resume:
        write_json(catalog_path, cases)
    elif json.dumps(json.loads(catalog_path.read_text()), sort_keys=True) != json.dumps(cases, sort_keys=True):
        raise ValueError("resume case catalog differs")
    record = {
        "schema": "piano-movers.contact-recovery-stress.v1", "status": "running", "split": args.split,
        "recipe_snapshot": recipe, "recipe_sha256": sha256(args.recipe),
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "catalog_sha256": sha256(args.output / "case_catalog.json"), "requested_cases": len(cases),
        "source_sha256": {str(p.relative_to(root)): sha256(p) for p in (
            Path(__file__), root / "scripts/audit_local_skill_composition.py",
            *sorted((root / "src/piano_movers").glob("*.py")))},
        "impulse_options": asdict(impulse), "impulse_is_environment_stressor_not_agent_action": True,
        "nominal_benchmark": False, "teacher_labels_present": False, "outcome_based_admission": False,
        "results": [],
        "contact_residual": None if args.contact_residual is None else {
            "path": str(args.contact_residual.resolve()), "sha256": sha256(args.contact_residual),
            "release_status": "independent_control_not_promoted"},
    }
    manifest = args.output / "manifest.json"
    completed = 0
    if args.resume:
        previous = json.loads(manifest.read_text())
        if previous["status"] != "running":
            raise ValueError("only incomplete audits may resume")
        for key in ("schema", "split", "recipe_sha256", "checkpoint_sha256", "catalog_sha256", "impulse_options", "contact_residual"):
            if key == "contact_residual" and key not in previous and record[key] is None:
                continue
            if previous[key] != record[key]:
                raise ValueError(f"resume provenance changed: {key}")
        for path, digest in previous["source_sha256"].items():
            if path != str(Path(__file__).relative_to(root)) and record["source_sha256"].get(path) != digest:
                raise ValueError(f"resume source changed: {path}")
        for case, row in zip(cases, previous["results"], strict=False):
            if row["case_id"] != case["case_id"] or sha256(args.output / row["trace"]) != row["trace_sha256"]:
                raise ValueError("completed prefix or trace changed")
        completed = len(previous["results"])
        if completed > len(cases):
            raise ValueError("invalid completed prefix length")
        record = previous
        record.setdefault("resume_events", []).append({"completed_prefix": completed,
            "runner_sha256": sha256(Path(__file__)), "unchanged_simulator_and_policy_sources": True})
    write_json(manifest, record)
    torch.set_num_threads(1)
    started = time.monotonic()
    for case in cases[completed:]:
        course = state_profile_course(args.seed, "combined", case["profile"])
        candidate_options = replace(options,
            preserve_contact_feedback=case["variant"] in ("impulse_preserve_feedback", "impulse_preserve_all_memory"),
            preserve_private_memory=case["variant"] == "impulse_preserve_all_memory")
        recorder = Recorder()
        engine = ContactRecoveryEngine(checkpoint, "cpu", course, options=candidate_options,
                                       impulse_options=replace(impulse, enabled=case["variant"] != "no_impulse"),
                                       audit_callback=recorder, residual=args.contact_residual)
        row = dict(case, options=asdict(candidate_options), handoffs=[])
        for event in engine.events(case["agents"], case["shape"]):
            if event["type"] in ("header", "done"):
                row["header" if event["type"] == "header" else "result"] = event
            if event["type"] == "frame" and event["waypoint_reached"]:
                row["handoffs"].append({"tick": event["step"], "next_stage": event["waypoint_index"],
                                         "state": event["next_state"]})
        suffix = f"_resume{len(record['resume_events'])}" if args.resume else ""
        trace = args.output / (case["case_id"] + suffix + ".npz")
        if trace.exists():
            raise FileExistsError(trace)
        recorder.save(trace)
        path = np.concatenate((np.stack(recorder.arrays["state"]), recorder.arrays["next_state"][-1][None]))
        certificate = certify_pose_path(torch.from_numpy(path), recorder.last_scenario, recorder.last_cfg)
        observations = np.stack(recorder.arrays["observation"])
        hits = np.flatnonzero(np.stack(recorder.arrays["hit"]))
        hit_observation_ticks = np.flatnonzero(observations[:, :, 22].any(axis=1))
        velocities = path[:, 3:5]
        moving_after_contact = []
        for hit in hits:
            moved = np.flatnonzero(np.linalg.norm(path[hit + 1:, :2] - path[hit + 1, :2], axis=-1) > .02)
            moving_after_contact.append(None if not len(moved) else int(moved[0]))
        row.update(
            impulse_event=engine.impulse.event, first_rejection=recorder.first_hit,
            contact_ticks=hits.tolist(), contact_observed_ticks=hit_observation_ticks.tolist(),
            recovery_to_2cm_steps=moving_after_contact,
            stationary_step_fraction=float((np.linalg.norm(velocities, axis=-1) < .01).mean()),
            continuous_certificate=asdict(certificate), trace=trace.name, trace_sha256=sha256(trace),
            independent_resolved_pose_collisions=recorder.resolved_pose_collisions,
            minimum_resolved_transition_clearance_m=recorder.minimum_clearance,
            path_length_m=recorder.path_length, gbp_float_payload_bytes=recorder.payload_bytes,
            grasp_setup_payload_bytes=recorder.setup_bytes, wire_headers_included=False,
            analytic_plan_agent_step_fraction=recorder.analytic_active / recorder.agent_steps,
        )
        row["demonstrated_contact_recovery"] = bool(len(hits) and row["result"]["success"] and certificate.passed
                                                     and not recorder.resolved_pose_collisions)
        record["results"].append(row)
        write_json(manifest, record)
        print(json.dumps({"done": len(record["results"]), "total": len(cases), "case": case["case_id"],
                          "success": row["result"]["success"], "contact_attempts": len(hits),
                          "impulse_triggered": engine.impulse.event is not None,
                          "contact_recovery": row["demonstrated_contact_recovery"]}), flush=True)
    record.update(status="complete", elapsed_seconds=None if args.resume else time.monotonic() - started,
                  elapsed_seconds_since_resume=time.monotonic() - started if args.resume else None,
                  summaries={v: {
                      "cases": sum(r["variant"] == v for r in record["results"]),
                      "successes": sum(r["variant"] == v and r["result"]["success"] for r in record["results"]),
                      "impulses_triggered": sum(r["variant"] == v and r["impulse_event"] is not None for r in record["results"]),
                      "contact_episodes": sum(r["variant"] == v and bool(r["contact_ticks"]) for r in record["results"]),
                      "demonstrated_recoveries": sum(r["variant"] == v and r["demonstrated_contact_recovery"] for r in record["results"]),
                      "certificates_passed": sum(r["variant"] == v and r["continuous_certificate"]["passed"] for r in record["results"]),
                  } for v in args.variants})
    write_json(manifest, record)
    print(json.dumps(record["summaries"]))


if __name__ == "__main__":
    main()
