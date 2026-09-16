#!/usr/bin/env python3
"""Add exact radius-clipped local boundary segments to saved teacher states.

No teacher is rerun and no label, body, layout, split, or physics is changed.
The source dataset and each source shard are hash checked and remain unchanged.
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
from piano_movers.env import PianoBatch, body_to_world, contact_points
from piano_movers.local_skill_policy import (
    LOCAL_SKILL_GEOMETRY_FEATURE_DIM,
    local_boundary_segments,
)
from piano_movers.local_skill_teacher import repeat_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.dataset.read_text())
    if source["status"] != "complete":
        raise ValueError("requires an immutable complete source dataset")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "manifest.json"
    if destination.exists():
        raise FileExistsError(destination)
    record = {
        **source,
        "status": "running",
        "results": [],
        "feature_dim": LOCAL_SKILL_GEOMETRY_FEATURE_DIM,
        "sensor_schema": "radius-local-clipped-solid-boundaries-v2-canonical-order",
        "source_dataset_sha256": sha256(args.dataset),
        "augmentation_source_sha256": {
            str(p): sha256(p)
            for p in [Path(__file__), Path("src/piano_movers/local_skill_policy.py")]
        },
    }
    write_json(destination, record)
    for index, row in enumerate(source["results"]):
        updated = dict(row)
        if row["success"]:
            path = args.dataset.parent / row["artifact"]
            if sha256(path) != row["artifact_sha256"]:
                raise ValueError("source shard checksum mismatch")
            data = dict(np.load(path))
            states = torch.from_numpy(data["states"][:-1].copy())
            if len(states) != len(data["features"]):
                raise ValueError(
                    "legacy DAgger prefix lacks physical states; re-collect with current recorder"
                )
            cfg = replace(EnvConfig(**row["env_cfg"]), device="cpu")
            walls = row["walls"]
            scene = PianoBatch(
                state0=states[:1],
                source=states[:1, :3],
                goal=torch.tensor([row["goal"]]),
                wall_x=torch.tensor([walls["x"]]),
                gap_y=torch.tensor([walls["gap_y"]]),
                gap_half=torch.tensor([walls["gap_half"]]),
            )
            portions = []
            with torch.inference_mode():
                for chunk in states.split(128):
                    handles = body_to_world(contact_points(cfg, "cpu"), chunk)
                    portions.append(
                        local_boundary_segments(
                            chunk, handles, repeat_scene(scene, len(chunk), chunk), cfg
                        ).numpy()
                    )
            data["features"] = np.concatenate((data["features"], np.concatenate(portions)), -1)
            if not np.isfinite(data["features"]).all():
                raise ValueError("non-finite boundary features")
            artifact = args.output / row["artifact"]
            np.savez_compressed(artifact, **data)
            updated.update(
                source_artifact_sha256=row["artifact_sha256"], artifact_sha256=sha256(artifact)
            )
        record["results"].append(updated)
        write_json(destination, record)
        if index % 50 == 0:
            print(json.dumps({"converted": index + 1, "total": len(source["results"])}), flush=True)
    record["status"] = "complete"
    write_json(destination, record)


if __name__ == "__main__":
    main()
