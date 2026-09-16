#!/usr/bin/env python3
"""Query certified teachers at actual p6 visited states, retaining causal history.

Every admitted query supervises the real student observation, then an executable
teacher continuation. The preceding student observations are unlabelled recurrent
burn-in. Query failures and deduplication remain visible; this is development
DAgger, never held-out evaluation data.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from audit_local_skill_composition import sha256, write_json
from build_local_skill_dataset import generate_case
from piano_movers.config import EnvConfig
from piano_movers.env import PianoBatch
from piano_movers.local_skill_policy import (
    local_skill_features,
    LOCAL_SKILL_FEATURE_DIM,
    LOCAL_SKILL_GEOMETRY_FEATURE_DIM,
)
from piano_movers.local_skill_teacher import LocalTeacherConfig
from piano_movers.multi_waypoint_demo import (
    DEFAULT_COURSE,
    _course_config,
    _full_collision_scenario,
)
from piano_movers.single_slit_0720_eval import load_single_slit_checkpoint
from piano_movers.single_slit_contact_observation import SingleSlitContactObservationContextV1


def context_from_features(features, state):
    s, c = state[:, 2:3].sin(), state[:, 2:3].cos()

    def world(v):
        return torch.stack((c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]), -1)

    return SingleSlitContactObservationContextV1(
        features[:, 0, 73].bool(), world(features[..., 74:76]), world(features[..., 76:78])
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries-per-episode", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=512)
    args = parser.parse_args()
    torch.set_num_threads(1)
    source = json.loads(args.rollouts.read_text())
    if source["status"] != "complete" or source["split"] != "development":
        raise ValueError("requires a completed development student rollout")
    parent = load_single_slit_checkpoint(args.parent, device="cpu")
    if parent.checkpoint_sha256 != source["parent_sha256"]:
        raise ValueError("actor/parent lineage mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "manifest.json"
    if destination.exists():
        raise FileExistsError(destination)
    options = LocalTeacherConfig(x_step=0.0125)
    manifest = {
        "schema": "piano-movers.p6-local-training-set.v1",
        "status": "running",
        "parent_checkpoint_sha256": parent.checkpoint_sha256,
        "actor_planner_sha256": source["planner_sha256"],
        "actor_rollout_manifest_sha256": sha256(args.rollouts),
        "feature_dim": source.get("feature_dim", LOCAL_SKILL_FEATURE_DIM),
        "heldout_shapes": ["S", "Z"],
        "heldout_counts": [9, 11, 13, 15, 17, 19],
        "split": "train_development_dagger",
        "options": asdict(options),
        "source_sha256": {
            str(p): sha256(p)
            for p in [
                Path(__file__),
                Path("scripts/build_local_skill_dataset.py"),
                Path("src/piano_movers/local_skill_teacher.py"),
            ]
        },
        "results": [],
        "skipped_duplicate_queries": 0,
    }
    write_json(destination, manifest)
    for episode in source["results"]:
        if episode["shape"] in ("S", "Z") or episode["agents"] % 2:
            raise ValueError("held-out shape/count in training query request")
        path = args.rollouts.parent / episode["artifact"]
        if sha256(path) != episode["artifact_sha256"]:
            raise ValueError("actor trace checksum mismatch")
        data = np.load(path)
        if "env_cfg" in episode:
            cfg = replace(EnvConfig(**episode["env_cfg"]), device="cpu")
            goals, orientation, walls = (
                episode["goals"],
                episode["orientation_required"],
                episode["walls"],
            )
        else:
            # Backward-compatible provenance for the explicitly identified v1
            # five-letter default-course audit, not an arbitrary inferred layout.
            if episode["case_id"] != f"multiwall_{episode['shape']}_{episode['agents']}":
                raise ValueError("legacy trace lacks reconstructable scene metadata")
            cfg = _course_config(parent.cfg, episode["agents"], DEFAULT_COURSE, episode["shape"])
            original = _full_collision_scenario(DEFAULT_COURSE, cfg)
            goals, orientation = DEFAULT_COURSE.waypoints, [False, False, False, True]
            walls = {
                "x": original.wall_x[0].tolist(),
                "gap_y": original.gap_y[0].tolist(),
                "gap_half": original.gap_half[0].tolist(),
            }
        length = len(data["features"])
        ticks = list(np.linspace(0, length - 1, args.queries_per_episode, dtype=int))
        contact = np.flatnonzero(data["hit"])
        if len(contact):
            ticks.extend([max(0, int(contact[0]) - 10), min(length - 1, int(contact[0]) + 1)])
        seen = set()
        for tick in sorted(set(ticks)):
            tick = int(tick)
            state = torch.from_numpy(data["states"][tick].copy()).view(1, 6)
            stage = int(data["stages"][tick])
            key = (stage, *np.round(state.numpy().flatten(), 3).tolist())
            if key in seen:
                manifest["skipped_duplicate_queries"] += 1
                continue
            seen.add(key)
            goal = state.new_tensor(goals[stage])
            scene = PianoBatch(
                state0=state,
                source=state[:, :3],
                goal=goal.view(1, 3),
                wall_x=state.new_tensor([walls["x"]]),
                gap_y=state.new_tensor([walls["gap_y"]]),
                gap_half=state.new_tensor([walls["gap_half"]]),
            )
            feature = torch.from_numpy(data["features"][tick].copy()).unsqueeze(0)
            geometry_segments = feature.shape[-1] == LOCAL_SKILL_GEOMETRY_FEATURE_DIM
            context = context_from_features(feature, state)
            reconstructed = local_skill_features(
                state,
                scene,
                cfg,
                context,
                goal=goal,
                orientation_required=orientation[stage],
                geometry_segments=geometry_segments,
            )
            torch.testing.assert_close(reconstructed, feature, atol=2e-6, rtol=2e-6)
            started = time.monotonic()
            report, teacher = generate_case(
                state,
                goal,
                scene,
                cfg,
                context,
                orientation[stage],
                options,
                args.horizon,
                geometry_segments=geometry_segments,
            )
            case_id = f"{episode['case_id']}_query{tick}"
            row = {
                "case_id": case_id,
                "shape": cfg.payload_shape,
                "agents": cfg.n_agents,
                "origin": {
                    "kind": "p6_model_induced_dagger",
                    "actor_trace_sha256": episode["artifact_sha256"],
                    "tick": tick,
                    "stage_for_audit_only": stage,
                },
                "env_cfg": asdict(cfg),
                "start_state": state[0].tolist(),
                "goal": goal.tolist(),
                "walls": walls,
                "orientation_required": orientation[stage],
                **report,
                "elapsed_seconds": time.monotonic() - started,
            }
            if teacher is not None:
                torch.testing.assert_close(
                    torch.from_numpy(teacher["features"][0]), feature[0], atol=2e-6, rtol=2e-6
                )
                prefix = data["features"][max(0, tick - 31) : tick]
                state_prefix = data["states"][max(0, tick - 31) : tick]
                supervised = len(teacher["features"])
                teacher["features"] = np.concatenate((prefix, teacher["features"]))
                teacher["states"] = np.concatenate((state_prefix, teacher["states"]))
                teacher["target_twist"] = np.concatenate(
                    (np.zeros((len(prefix), 3), np.float32), teacher["target_twist"])
                )
                teacher["target_mask"] = np.r_[
                    np.zeros(len(prefix), bool), np.ones(supervised, bool)
                ]
                row["recurrent_burn_in_steps"] = len(prefix)
                row["student_observation_labelled"] = True
                artifact = args.output / f"{case_id}.npz"
                np.savez_compressed(artifact, **teacher)
                row.update(artifact=artifact.name, artifact_sha256=sha256(artifact))
            manifest["results"].append(row)
            write_json(destination, manifest)
            print(
                json.dumps(
                    {
                        k: row.get(k)
                        for k in (
                            "case_id",
                            "success",
                            "steps",
                            "execution_failure",
                            "elapsed_seconds",
                        )
                    }
                ),
                flush=True,
            )
    manifest["status"] = "complete"
    manifest["successes"] = sum(r["success"] for r in manifest["results"])
    write_json(destination, manifest)


if __name__ == "__main__":
    main()
