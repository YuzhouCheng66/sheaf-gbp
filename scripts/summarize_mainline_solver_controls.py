#!/usr/bin/env python3
"""Audit the predeclared 21-case precision/round-budget factorial.

This is a paired development comparison, not an independent OOD estimate.
Trace checks validate actual payload sizes; transport headers are not counted.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from audit_local_skill_composition import sha256, write_json


def load_panel(root, label):
    rows, manifests = {}, []
    for panel in ("eight", "D_counts"):
        path = root / f"p6_mainline_solver_{label}_{panel}_81201/manifest.json"
        document = json.loads(path.read_text())
        if document["status"] != "complete":
            raise ValueError(f"unfinished: {path}")
        catalog = path.parent / "case_catalog.json"
        if sha256(catalog) != document["catalog_sha256"]:
            raise ValueError("catalog mismatch")
        expected = {r["case_id"] for r in json.loads(catalog.read_text())["cases"]}
        if expected != {r["case_id"] for r in document["results"]}:
            raise ValueError("requested denominator mismatch")
        for row in document["results"]:
            key = row["case_id"]
            if key in rows:
                raise ValueError("duplicate case")
            trace = (path.parent / row["trace"]).resolve()
            if sha256(trace) != row["trace_sha256"]:
                raise ValueError("trace checksum mismatch")
            with np.load(trace) as arrays:
                controls = row["header"]["solver_controls"]
                expected_bytes = controls["rounds"] * len(row["header"]["sender"]) * 6 * 4
                if controls["wire_floats_per_directed_message"] != 6:
                    raise ValueError("unexpected message schema")
                if not np.all(arrays["gbp_payload_bytes"] == expected_bytes):
                    raise ValueError("actual per-step payload disagrees with topology/rounds")
                if int(arrays["gbp_payload_bytes"].sum()) != row["gbp_float_payload_bytes"]:
                    raise ValueError("episode payload mismatch")
                if int(arrays["setup_payload_bytes"].sum()) != row["grasp_setup_payload_bytes"]:
                    raise ValueError("grasp setup payload mismatch")
            rows[key] = dict(row, trace=str(trace))
        manifests.append({"path": str(path.resolve()), "sha256": sha256(path)})
    if len(rows) != 21:
        raise ValueError("the declared factorial has 21 cases per arm")
    return rows, manifests


def aggregate(rows):
    return {
        "episodes": len(rows),
        "successes": sum(r["result"]["success"] for r in rows),
        "continuous_certificates": sum(r["continuous_certificate"]["passed"] for r in rows),
        "control_steps": sum(r["result"]["steps"] for r in rows),
        "contact_attempts": sum(r["result"]["contact_events"] for r in rows),
        "path_length_m": sum(r["path_length_m"] for r in rows),
        "gbp_float_payload_bytes": sum(r["gbp_float_payload_bytes"] for r in rows),
        "grasp_setup_payload_bytes": sum(r["grasp_setup_payload_bytes"] for r in rows),
        "round_histogram": dict(Counter(r["header"]["solver_controls"]["rounds"] for r in rows)),
        "failures": [r["case_id"] for r in rows if not r["result"]["success"]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("output"))
    parser.add_argument("--old-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reference, provenance = load_panel(args.root, "reference")
    old = json.loads(args.old_reference.read_text())
    if old["status"] != "complete":
        raise ValueError("old reference incomplete")
    old_by_case = {r["case_id"]: r for r in old["results"]}
    parity = []
    for key, row in reference.items():
        if key not in old_by_case:
            continue
        old_row = old_by_case[key]
        if old_row["header"]["checkpoint_sha256"] != row["header"]["checkpoint_sha256"]:
            raise ValueError("reference checkpoint mismatch")
        old_path = args.old_reference.parent / old_row["trace"]
        if sha256(old_path) != old_row["trace_sha256"]:
            raise ValueError("old trace checksum mismatch")
        with np.load(old_path) as before, np.load(row["trace"]) as after:
            if set(before.files) != set(after.files):
                raise ValueError("trace array schema changed")
            if not all(np.array_equal(before[k], after[k]) for k in before.files):
                raise ValueError(f"default runtime controls changed the old reference: {key}")
        parity.append(key)
    if len(parity) != 9:
        raise ValueError("expected eight selected shapes plus D/12 in the old N=12 panel")
    summaries = {"reference": aggregate(list(reference.values()))}
    for label in ("equal", "diameter", "equal_diameter"):
        rows, sources = load_panel(args.root, label)
        provenance.extend(sources)
        if set(rows) != set(reference):
            raise ValueError("unpaired case sets")
        for key, row in rows.items():
            baseline = reference[key]
            if row["course"] != baseline["course"]:
                raise ValueError("course differs")
            if row["header"]["checkpoint_sha256"] != baseline["header"]["checkpoint_sha256"]:
                raise ValueError("weights differ")
            ignored = {"geometry_precision_boost_override", "cover_coordination_tree_diameter"}
            if ({k: v for k, v in row["options"].items() if k not in ignored}
                    != {k: v for k, v in baseline["options"].items() if k not in ignored}):
                raise ValueError("unmatched non-factorial controls")
        summary = aggregate(list(rows.values()))
        summary["paired_repairs"] = [k for k in rows if rows[k]["result"]["success"]
                                     and not reference[k]["result"]["success"]]
        summary["paired_regressions"] = [k for k in rows if reference[k]["result"]["success"]
                                         and not rows[k]["result"]["success"]]
        summaries[label] = summary
    report = {
        "status": "complete", "source_manifests": provenance,
        "old_reference_sha256": sha256(args.old_reference),
        "default_behavior_bitwise_parity_cases": parity,
        "actual_trace_payload_accounting_verified": True,
        "wire_headers_included": False, "summaries": summaries,
        "scope": "21 predeclared paired development cases; not full-distribution validation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
