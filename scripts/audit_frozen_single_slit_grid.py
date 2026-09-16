#!/usr/bin/env python3
"""Re-run the actual frozen checkpoint on the nominal single-slit grid.

Legacy terminal success is recorded separately from a stricter no-rejected-
contact dwell and the independent continuous-motion certificate.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import json

import numpy as np
import torch

from audit_local_skill_composition import sha256, write_json
from piano_movers.config import HANDLE_LAYOUT_GLYPH_BOUNDARY_V4, HANDLE_LAYOUT_PARAMETRIC_T_V3
from piano_movers.env import collision_mask, radius_edges
from piano_movers.gbp import homogeneous_topology
from piano_movers.local_skill_teacher import certify_pose_path
from piano_movers.single_slit_0720_eval import (
    load_single_slit_checkpoint,
    strict_final_mask,
    STRICT_STABLE_STEPS,
)
from piano_movers.single_slit_feedback import (
    SOURCE_POSE,
    single_slit_env_config,
    single_slit_scenario,
)
from piano_movers.single_slit_contact_feedback import single_slit_contact_feedback_step_v1
from piano_movers.single_slit_contact_observation import (
    initial_single_slit_contact_observation_context_v1,
    single_slit_contact_observation_context_from_result_v1,
    single_slit_contact_observations_v1,
)


@torch.inference_mode()
def audit(policy, cfg, horizon):
    policy.bind_environment(cfg)
    state = torch.tensor([[*SOURCE_POSE, 0.0, 0.0, 0.0]])
    scene = single_slit_scenario(1, cfg, initial_state=state)
    context = initial_single_slit_contact_observation_context_v1(state, cfg)
    hidden = policy.init_hidden(1, "cpu")
    edges = radius_edges(cfg, device="cpu")
    topology = homogeneous_topology(edges, 1, cfg.n_agents, device="cpu")
    legacy_dwell = strict_dwell = contacts = penetrations = payload = setup = 0
    legacy_finish = None
    trajectory = [state[0].clone()]
    for tick in range(horizon):
        observation = single_slit_contact_observations_v1(state, scene, cfg, context)
        out = policy(observation, edges, hidden, topology=topology, trace_messages=True)
        transition = single_slit_contact_feedback_step_v1(state, out.forces, scene, cfg)
        state = transition.resolved_state
        hidden = out.hidden
        context = single_slit_contact_observation_context_from_result_v1(
            transition, out.forces, cfg
        )
        contacts += int(transition.attempted_hit[0])
        penetrations += int(collision_mask(state, scene, cfg)[0])
        trace = out.aux["gbp_message_trace"]
        payload += trace.wire_float_count * trace.natural.element_size()
        message = out.aux["local_se2_distributed_grasp_message_trace"]
        setup += message.numel() * message.element_size()
        passed = bool(strict_final_mask(state, scene.goal)[0])
        legacy_dwell = legacy_dwell + 1 if passed else 0
        strict_dwell = strict_dwell + 1 if passed and not bool(transition.attempted_hit[0]) else 0
        if legacy_dwell >= STRICT_STABLE_STEPS and legacy_finish is None:
            legacy_finish = tick + 1
        trajectory.append(state[0].clone())
        if strict_dwell >= STRICT_STABLE_STEPS:
            break
    trajectory = torch.stack(trajectory)
    certificate = certify_pose_path(trajectory, scene, cfg)
    return {
        "legacy_success": legacy_finish is not None,
        "legacy_finish_tick": legacy_finish,
        "strict_success": strict_dwell >= STRICT_STABLE_STEPS,
        "steps": tick + 1,
        "contacts": contacts,
        "resolved_pose_collisions": penetrations,
        "continuous_certificate": asdict(certificate),
        "gbp_payload_bytes": payload,
        "grasp_setup_payload_bytes": setup,
        "wire_headers_included": False,
        "env_cfg": asdict(cfg),
    }, trajectory.numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shapes", nargs="+", default=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    parser.add_argument("--agents", nargs="+", type=int, default=list(range(8, 21)))
    parser.add_argument("--horizon", type=int, default=384)
    parser.add_argument("--t-profile", choices=["benchmark", "demo"], default="benchmark")
    args = parser.parse_args()
    torch.set_num_threads(1)
    parent = load_single_slit_checkpoint(args.checkpoint, device="cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "manifest.json"
    if destination.exists():
        raise FileExistsError(destination)
    record = {
        "schema": "piano-movers.frozen-single-slit-grid.v1",
        "status": "running",
        "checkpoint_sha256": parent.checkpoint_sha256,
        "t_profile": args.t_profile,
        "horizon": args.horizon,
        "source_sha256": {
            str(p): sha256(p)
            for p in [
                Path(__file__),
                Path("src/piano_movers/models.py"),
                Path("src/piano_movers/single_slit_local_planner.py"),
            ]
        },
        "results": [],
    }
    write_json(destination, record)
    for shape in args.shapes:
        for count in args.agents:
            cfg = replace(
                single_slit_env_config(device="cpu", max_steps=args.horizon),
                n_agents=count,
                payload_shape=shape,
                comm_radius=0.40,
                handle_layout=HANDLE_LAYOUT_PARAMETRIC_T_V3
                if shape == "T"
                else HANDLE_LAYOUT_GLYPH_BOUNDARY_V4,
            )
            if shape == "T" and args.t_profile == "demo":
                cfg = replace(
                    cfg,
                    t_total_width_m=0.930,
                    t_total_height_m=0.920,
                    t_stem_width_m=0.160,
                    t_cap_thickness_m=0.152,
                )
            row, states = audit(parent.policy, cfg, args.horizon)
            case_id = f"{shape}_{count}"
            artifact = args.output / f"{case_id}.npz"
            np.savez_compressed(artifact, states=states)
            row.update(
                case_id=case_id,
                shape=shape,
                agents=count,
                artifact=artifact.name,
                artifact_sha256=sha256(artifact),
            )
            record["results"].append(row)
            write_json(destination, record)
            print(
                json.dumps(
                    {
                        k: row[k]
                        for k in (
                            "case_id",
                            "legacy_success",
                            "strict_success",
                            "steps",
                            "contacts",
                        )
                    }
                ),
                flush=True,
            )
    record["status"] = "complete"
    write_json(destination, record)


if __name__ == "__main__":
    main()
