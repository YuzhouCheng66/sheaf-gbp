#!/usr/bin/env python3
"""Certified one-step teacher labels from stopped, feasible payload poses.

These are independent physical interventions, not trajectories and not DAgger.
They prevent one-step imitation from relying only on persistence of velocity.
All actor inputs are recomputed from the current local physical state; a stored
teacher path is used only to construct training labels, never for deployment.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from audit_local_skill_composition import sha256, write_json
from piano_movers.config import EnvConfig
from piano_movers.env import PianoBatch
from piano_movers.local_skill_policy import local_skill_features, LOCAL_SKILL_GEOMETRY_FEATURE_DIM
from piano_movers.local_skill_teacher import LocalTeacherConfig, executable_teacher_step
from piano_movers.single_slit_contact_observation import (
    initial_single_slit_contact_observation_context_v1,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries-per-episode", type=int, default=12)
    args = parser.parse_args()
    source = json.loads(args.dataset.read_text())
    if source["status"] != "complete" or source["split"] != "train":
        raise ValueError("requires complete training trajectories")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "manifest.json"
    if destination.exists():
        raise FileExistsError(destination)
    options = LocalTeacherConfig()
    record = {
        "schema": "piano-movers.independent-local-interventions.v1",
        "status": "running",
        "split": "train_interventions",
        "parent_checkpoint_sha256": source["parent_checkpoint_sha256"],
        "heldout_shapes": source["heldout_shapes"],
        "heldout_counts": source["heldout_counts"],
        "feature_dim": LOCAL_SKILL_GEOMETRY_FEATURE_DIM,
        "source_dataset_sha256": sha256(args.dataset),
        "success_semantics": "entry admitted for fitting; not episode success",
        "source_sha256": {
            str(p): sha256(p)
            for p in [
                Path(__file__),
                Path("src/piano_movers/local_skill_policy.py"),
                Path("src/piano_movers/local_skill_teacher.py"),
            ]
        },
        "results": [],
    }
    write_json(destination, record)
    for index, row in enumerate(source["results"]):
        if not row["success"]:
            continue
        path = args.dataset.parent / row["artifact"]
        if sha256(path) != row["artifact_sha256"]:
            raise ValueError("source checksum mismatch")
        data = np.load(path)
        cfg = replace(EnvConfig(**row["env_cfg"]), device="cpu")
        planned = torch.from_numpy(data["path"].copy()).float()
        goal = torch.tensor(row["goal"])
        walls = row["walls"]
        features, targets, forces, states, next_states, rejected = [], [], [], [], [], []
        ticks = np.unique(
            np.linspace(0, max(0, len(data["states"]) - 9), args.queries_per_episode, dtype=int)
        )
        for tick in ticks:
            state = torch.from_numpy(data["states"][tick : tick + 1].copy())
            state[:, 3:] = 0.0
            scene = PianoBatch(
                state0=state,
                source=state[:, :3],
                goal=goal.view(1, 3),
                wall_x=torch.tensor([walls["x"]]),
                gap_y=torch.tensor([walls["gap_y"]]),
                gap_half=torch.tensor([walls["gap_half"]]),
            )
            force, nxt, status = executable_teacher_step(state, planned, scene, cfg, options)
            if force is None:
                rejected.append({"source_tick": int(tick), "reason": status})
                continue
            context = initial_single_slit_contact_observation_context_v1(state, cfg)
            features.append(
                local_skill_features(
                    state,
                    scene,
                    cfg,
                    context,
                    goal=goal,
                    orientation_required=row["orientation_required"],
                    geometry_segments=True,
                )[0].numpy()
            )
            s, c = state[0, 2].sin(), state[0, 2].cos()
            target = torch.stack(
                (c * nxt[0, 3] + s * nxt[0, 4], -s * nxt[0, 3] + c * nxt[0, 4], nxt[0, 5])
            ) / state.new_tensor([0.65, 0.65, 1.4])
            targets.append(target.numpy())
            forces.append(force[0].numpy())
            states.append(state[0].numpy())
            next_states.append(nxt[0].numpy())
        result = {
            "case_id": f"stop_{row['case_id']}",
            "shape": row["shape"],
            "agents": row["agents"],
            "env_cfg": row["env_cfg"],
            "walls": walls,
            "goal": row["goal"],
            "orientation_required": row["orientation_required"],
            "independent_frames": True,
            "success": bool(features),
            "steps": len(features),
            "requested_queries": len(ticks),
            "rejected_queries": rejected,
            "origin": {
                "kind": "zero_velocity_physical_intervention",
                "source_artifact_sha256": row["artifact_sha256"],
            },
        }
        if features:
            artifact = args.output / f"{result['case_id']}.npz"
            np.savez_compressed(
                artifact,
                features=np.stack(features),
                target_twist=np.stack(targets),
                forces=np.stack(forces),
                states=np.stack(states),
                next_states=np.stack(next_states),
            )
            result.update(artifact=artifact.name, artifact_sha256=sha256(artifact))
        record["results"].append(result)
        if index % 20 == 0:
            write_json(destination, record)
            print(
                json.dumps(
                    {
                        "source_episodes": index + 1,
                        "labels": sum(r["steps"] for r in record["results"]),
                    }
                ),
                flush=True,
            )
    record["status"] = "complete"
    write_json(destination, record)


if __name__ == "__main__":
    main()
