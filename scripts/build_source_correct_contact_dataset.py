#!/usr/bin/env python3
"""Preserve successful original-skill continuations in contact DAgger labels.

Original teacher forces/states remain immutable trajectory witnesses. Separate
training labels select the original student's first action only where its
full local continuation has been independently certified successful. No
teacher, rollout lookup or source-correct gate is added to deployment.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from audit_local_skill_composition import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data = json.loads(args.dataset.read_text())
    audit = json.loads(args.audit.read_text())
    if (data["status"] != "complete" or audit["status"] != "complete"
            or audit["dataset_sha256"] != sha256(args.dataset)
            or audit["prefix_selection"] != "all_active_contact_memory_frames"):
        raise ValueError("complete matching per-frame audit required")
    lookup = {(r["teacher_case"], r["prefix"]): r for r in audit["results"]}
    if len(lookup) != len(audit["results"]):
        raise ValueError("duplicate counterfactual query")
    args.output.mkdir(parents=True)
    result = dict(data, results=[], status="running",
        training_label_schema="piano-movers.source-correct-contact-labels.v1",
        source_dataset_sha256=sha256(args.dataset), source_correct_audit_sha256=sha256(args.audit),
        source_correct_horizon=audit["horizon"], original_teacher_witnesses_retained=True,
        labeling_source_sha256=sha256(Path(__file__)))
    destination = args.output / "manifest.json"
    write_json(destination, result)
    used = set()
    for row in data["results"]:
        new = dict(row)
        if not row["admitted_for_training"]:
            result["results"].append(new)
            continue
        path = args.dataset.parent / row["artifact"]
        if sha256(path) != row["artifact_sha256"]:
            raise ValueError("teacher witness changed")
        with np.load(path) as archive:
            arrays = {k: archive[k].copy() for k in archive.files}
        labels = arrays["target_force"].copy()
        preserve = np.zeros(len(labels), dtype=bool)
        proofs = []
        for tick in range(len(labels)):
            key = (row["case_id"], tick)
            proof = lookup.get(key)
            if proof is None:
                continue
            used.add(key)
            if proof["teacher_artifact_sha256"] != row["artifact_sha256"]:
                raise ValueError("counterfactual teacher identity mismatch")
            if proof["source_success"]:
                if not proof["continuous_certificate"]["passed"]:
                    raise ValueError("uncertified source preservation")
                trace = args.audit.parent / proof["trace"]
                if sha256(trace) != proof["trace_sha256"]:
                    raise ValueError("source continuation witness changed")
                with np.load(trace) as source:
                    if not np.array_equal(source["states"][0], arrays["states"][tick]):
                        raise ValueError("source continuation starts at another state")
                    labels[tick] = source["forces"][0]
                preserve[tick] = True
            proofs.append({"frame": tick, "preserve_source": bool(preserve[tick]),
                           "trace": str((args.audit.parent / proof["trace"]).resolve()),
                           "trace_sha256": proof["trace_sha256"]})
        arrays.update(training_target_force=labels, preserve_source_mask=preserve)
        artifact = args.output / (row["case_id"] + ".npz")
        np.savez_compressed(artifact, **arrays)
        new.update(artifact=artifact.name, artifact_sha256=sha256(artifact),
                   original_teacher_artifact_sha256=row["artifact_sha256"],
                   source_correct_frames=int(preserve.sum()), source_correct_proofs=proofs)
        result["results"].append(new)
        write_json(destination, result)
    if used != set(lookup):
        raise ValueError("counterfactual audit contains unmatched frames")
    result.update(status="complete", source_correct_frames=sum(r.get("source_correct_frames", 0) for r in result["results"]),
                  queried_contact_frames=len(lookup))
    write_json(destination, result)
    print(json.dumps({k: result[k] for k in ("status", "queried_contact_frames", "source_correct_frames")}))


if __name__ == "__main__":
    main()
