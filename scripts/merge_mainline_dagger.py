#!/usr/bin/env python3
"""Aggregate original-module DAgger rounds with authenticated actor provenance.

Inputs may come from different students with the same root parent. Teacher
failures remain in the merged manifest. Validation/test rollouts are never
admitted, and duplicate actor-state queries are rejected rather than reweighted.
"""

import argparse
import json
from pathlib import Path

from audit_local_skill_composition import sha256, write_json


def merge_datasets(paths, legacy_rollouts=()):
    record, rows, seen = None, [], set()
    sources = []
    actors = set()
    actor_identities = {}
    unqueried = []
    actor_proofs = {sha256(p): json.loads(p.read_text()) for p in legacy_rollouts}
    for path in paths:
        source = json.loads(path.read_text())
        if (source["schema"] != "piano-movers.original-architecture-dagger.v1"
                or source["status"] != "complete"
                or source["split"] != "train_development_dagger"):
            raise ValueError("requires complete, explicitly admitted development DAgger")
        if record is None:
            record = dict(source)
        for key in ("schema", "split", "parent_checkpoint_sha256", "observation_dim", "plan_dim",
                    "supervised_plan_coordinates", "heldout_shapes", "heldout_counts"):
            if source[key] != record[key]:
                raise ValueError(f"mixed dataset semantics: {key}")
        actor = source.get("actor_checkpoint_sha256")
        if not actor:
            proof = actor_proofs.get(source["rollout_manifest_sha256"])
            if not proof or proof["status"] != "complete":
                raise ValueError("legacy data requires its checksum-matched actor rollout manifest")
            actor = proof["checkpoint_sha256"]
            if actor != source["parent_checkpoint_sha256"]:
                raise ValueError("legacy collector required actor equal to root parent")
        actors.add(actor)
        extension = source.get("actor_contact_residual")
        identity = actor + ("+contact:" + extension["sha256"] if extension is not None else "")
        actor_identities[identity] = {"base_checkpoint_sha256": actor, "contact_residual": extension}
        unqueried.extend(dict(e, actor_checkpoint_sha256=actor) for e in source.get("unqueried_episodes", []))
        sources.append({"manifest": str(path.resolve()), "sha256": sha256(path),
                        "actor_checkpoint_sha256": actor,
                        "actor_contact_residual": extension,
                        "rollout_manifest_sha256": source["rollout_manifest_sha256"]})
        for row in source["results"]:
            key = (identity, row["actor_trace_sha256"], row["actor_tick"])
            if key in seen:
                raise ValueError("duplicate actor-state query")
            seen.add(key)
            if row["shape"] in source["heldout_shapes"] or row["agents"] in source["heldout_counts"]:
                raise ValueError("held-out shape/count in training manifest")
            merged = dict(row, actor_checkpoint_sha256=actor,
                          actor_contact_residual=extension,
                          source_manifest_sha256=sources[-1]["sha256"],
                          source_case_id=row["case_id"],
                          case_id=actor[:12] + "_" + row["case_id"])
            if "artifact" in row:
                artifact = (path.parent / row["artifact"]).resolve(strict=True)
                if sha256(artifact) != row["artifact_sha256"]:
                    raise ValueError("teacher artifact checksum mismatch")
                merged["artifact"] = str(artifact)
            elif row["admitted_for_training"]:
                raise ValueError("admitted teacher has no artifact")
            rows.append(merged)
    if not rows:
        raise ValueError("empty DAgger aggregation")
    record.update(results=rows, source_manifests=sources,
                  actor_checkpoint_sha256=None, actor_checkpoint_sha256s=sorted(actors),
                  actor_contact_residual=None, actor_identities=actor_identities,
                  rollout_manifest_sha256=None, variant="explicit_multi_round_aggregation",
                  source_sha256={str(Path(__file__)): sha256(Path(__file__))},
                  skipped_duplicate_queries=None)
    record["unqueried_episodes"] = unqueried
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-rollouts", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = merge_datasets(args.input, args.legacy_rollouts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps({"queries": len(result["results"]), "admitted": sum(
        row["admitted_for_training"] for row in result["results"]),
        "actors": result["actor_checkpoint_sha256s"]}))


if __name__ == "__main__":
    main()
