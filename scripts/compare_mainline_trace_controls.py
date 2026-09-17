#!/usr/bin/env python3
"""Authenticate same-checkpoint paired rollouts, including unchanged traces."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from audit_local_skill_composition import sha256, write_json
from piano_movers.multi_waypoint_demo import SkillCompositionOptions


def case_key(row):
    return row["case_id"] if "case_id" in row else f"{row['shape']}_n{row['agents']}"


def stationary_fraction(states):
    """Low translation AND low rotation; this alone does not imply being stuck."""
    return float(((np.linalg.norm(states[:, 3:5], axis=-1) < .01) & (np.abs(states[:, 5]) < .01)).mean())


def full_case_index(rows):
    indexed = {case_key(r): r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError("duplicate case in paired audit")
    return indexed


def verify_options(old, new, only_option):
    before, after = (asdict(SkillCompositionOptions(**r["options"])) for r in (old, new))
    changed = [k for k in before if before[k] != after[k]]
    if only_option is not None and changed != [only_option]:
        raise ValueError(f"expected only {only_option}, observed {changed}")
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only-option", choices=tuple(SkillCompositionOptions.__dataclass_fields__),
                        help="Require exactly this one composition option to change")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    before, after = (json.loads(p.read_text()) for p in (args.before, args.after))
    if before["status"] != "complete" or after["status"] != "complete":
        raise ValueError("both controls must be complete")
    if before["checkpoint_sha256"] != after["checkpoint_sha256"]:
        raise ValueError("this audit isolates interfaces, not weight changes")
    source = full_case_index(before["results"])
    target = full_case_index(after["results"])
    if source.keys() != target.keys():
        raise ValueError("paired panels must cover exactly the same complete case set")
    rows = []
    for row in after["results"]:
        key = case_key(row)
        old = source[key]
        changed_options = verify_options(old, row, args.only_option)
        for field in ("shape", "agents", "profile", "seed", "family", "horizon"):
            if old.get(field) != row.get(field):
                raise ValueError(f"unmatched {field}")
        if "course" in row:
            if row["course"] != old.get("course"):
                raise ValueError("unmatched physical course")
        else:
            for field in ("source", "source_velocity", "walls", "waypoints", "world", "morphology_scale"):
                if row["header"][field] != old["header"][field]:
                    raise ValueError(f"unmatched course {field}")
        paths = (args.before.parent / old["trace"], args.after.parent / row["trace"])
        for path, metadata in zip(paths, (old, row), strict=True):
            if sha256(path) != metadata["trace_sha256"]:
                raise ValueError("changed trace")
        with np.load(paths[0]) as a, np.load(paths[1]) as b:
            if set(a.files) != set(b.files):
                raise ValueError("trace schemas differ")
            equal = all(np.array_equal(a[k], b[k]) for k in a.files)
            limit = min(len(a["forces"]), len(b["forces"]))
            same_tick = (a["forces"][:limit] == b["forces"][:limit]).all(axis=(1, 2))
            changed = np.flatnonzero(~same_tick)
            first = int(changed[0]) if len(changed) else None
            stalled = [stationary_fraction(z["state"]) for z in (a, b)]
        rows.append({"case_id": key, "shape": row["shape"], "agents": row["agents"],
                     "changed_options": changed_options, "before_trace_sha256": old["trace_sha256"],
                     "after_trace_sha256": row["trace_sha256"], "all_arrays_bitwise_equal": equal,
                     "first_changed_force_tick": first,
                     "before_success": old["result"]["success"], "after_success": row["result"]["success"],
                     "before_steps": old["result"]["steps"], "after_steps": row["result"]["steps"],
                     "before_contacts": old["result"]["contact_events"], "after_contacts": row["result"]["contact_events"],
                     "before_stationary_fraction": stalled[0], "after_stationary_fraction": stalled[1],
                     "before_path_m": old["path_length_m"], "after_path_m": row["path_length_m"],
                     "before_gbp_payload_bytes": old["gbp_float_payload_bytes"], "after_gbp_payload_bytes": row["gbp_float_payload_bytes"],
                     "before_setup_payload_bytes": old["grasp_setup_payload_bytes"], "after_setup_payload_bytes": row["grasp_setup_payload_bytes"],
                     "after_certified": row["continuous_certificate"]["passed"],
                     "after_penetration": row["result"]["exact_penetration"]})
    report = {"status": "complete", "checkpoint_sha256": before["checkpoint_sha256"],
              "before_manifest_sha256": sha256(args.before), "after_manifest_sha256": sha256(args.after),
              "cases": len(rows), "bitwise_unchanged": sum(r["all_arrays_bitwise_equal"] for r in rows),
              "repairs": sum(not r["before_success"] and r["after_success"] for r in rows),
              "regressions": sum(r["before_success"] and not r["after_success"] for r in rows),
              "after_certified": sum(r["after_certified"] for r in rows),
              "after_penetrations": sum(r["after_penetration"] for r in rows),
              "only_option": args.only_option, "source_sha256": sha256(Path(__file__)),
              "statistical_scope": "complete paired catalog; correlated cases, not independent Bernoulli trials",
              "communication_scope": "actual float payload plus attachment setup; wire headers excluded",
              "stationary_definition": "translation speed < 0.01 m/s AND |angular speed| < 0.01 rad/s; not a stuck classifier",
              "results": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "results"}))


if __name__ == "__main__":
    main()
