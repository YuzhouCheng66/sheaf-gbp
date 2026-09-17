#!/usr/bin/env python3
"""Paired audit of the single default-off crossing-interface option."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from audit_local_skill_composition import sha256, write_json
from piano_movers.multi_waypoint_demo import SkillCompositionOptions
from piano_movers.local_skill_disturbance import ContactImpulseOptions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source, candidate = (json.loads(p.read_text()) for p in (args.source, args.candidate))
    if source["status"] != "complete" or candidate["status"] != "complete":
        raise ValueError("completed panels required")
    for field in ("checkpoint_sha256", "split"):
        if source[field] != candidate[field]:
            raise ValueError(f"unmatched {field}")
    # The first near-contact manifest predates the optional zero angular
    # impulse. Normalize through the declared backward-compatible defaults.
    if asdict(ContactImpulseOptions(**source["impulse_options"])) != asdict(ContactImpulseOptions(**candidate["impulse_options"])):
        raise ValueError("unmatched impulse_options")
    if source.get("contact_residual") is not None or candidate.get("contact_residual") is not None:
        raise ValueError("not an interface-only comparison")
    originals = {r["case_id"]: r for r in source["results"]}
    if len(originals) != len(source["results"]):
        raise ValueError("duplicate source case")
    results = []
    for row in candidate["results"]:
        old = originals[row["case_id"]]
        if any(old["header"][k] != row["header"][k] for k in
               ("source", "source_velocity", "walls", "waypoints", "world", "morphology_scale")):
            raise ValueError("course mismatch")
        a, b = (asdict(SkillCompositionOptions(**r["options"])) for r in (old, row))
        old_flag, new_flag = a.pop("hold_feasible_crossing"), b.pop("hold_feasible_crossing")
        if a != b or old_flag:
            raise ValueError("additional interface changes or non-source baseline")
        paths = [args.source.parent / old["trace"], args.candidate.parent / row["trace"]]
        if any(sha256(p) != r["trace_sha256"] for p, r in zip(paths, (old, row))):
            raise ValueError("changed trajectory")
        with np.load(paths[0]) as x, np.load(paths[1]) as y:
            bitwise = set(x.files) == set(y.files) and all(np.array_equal(x[k], y[k]) for k in x.files)
        if not new_flag and not bitwise:
            raise ValueError("disabled-option default parity failed")
        results.append({"case_id": row["case_id"], "variant": row["variant"], "flag_enabled": new_flag,
            "source_success": old["result"]["success"], "candidate_success": row["result"]["success"],
            "arrays_bitwise_equal": bitwise,
            "source_steps": old["result"]["steps"], "candidate_steps": row["result"]["steps"],
            "source_contacts": len(old["contact_ticks"]), "candidate_contacts": len(row["contact_ticks"]),
            "source_path_m": old["path_length_m"], "candidate_path_m": row["path_length_m"],
            "source_payload_bytes": old["gbp_float_payload_bytes"], "candidate_payload_bytes": row["gbp_float_payload_bytes"],
            "candidate_certified": row["continuous_certificate"]["passed"],
            "candidate_penetration": row["result"]["exact_penetration"],
            "impulse_event_identical": old["impulse_event"] == row["impulse_event"]})
    summary = {"paired_cases": len(results),
        "source_successes": sum(r["source_success"] for r in results),
        "candidate_successes": sum(r["candidate_success"] for r in results),
        "repairs": sum(not r["source_success"] and r["candidate_success"] for r in results),
        "regressions": sum(r["source_success"] and not r["candidate_success"] for r in results),
        "bitwise_equal_cases": sum(r["arrays_bitwise_equal"] for r in results),
        "candidate_certified_cases": sum(r["candidate_certified"] for r in results),
        "candidate_penetrations": sum(r["candidate_penetration"] for r in results)}
    report = {"status": "complete", "source_manifest_sha256": sha256(args.source),
        "candidate_manifest_sha256": sha256(args.candidate), "script_sha256": sha256(Path(__file__)),
        "checkpoint_sha256": source["checkpoint_sha256"], "summary": summary, "results": results,
        "normalized_impulse_options": asdict(ContactImpulseOptions(**source["impulse_options"])),
        "statistical_scope": "paired correlated layout/shape/count panel, not independent Bernoulli trials",
        "communication_scope": "actual GBP float payload; wire headers excluded; setup reported in source manifests"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
