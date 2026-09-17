#!/usr/bin/env python3
"""Authenticate same-checkpoint paired rollouts, including unchanged traces."""

import argparse
import json
from pathlib import Path

import numpy as np

from audit_local_skill_composition import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    before, after = (json.loads(p.read_text()) for p in (args.before, args.after))
    if before["status"] != "complete" or after["status"] != "complete":
        raise ValueError("both controls must be complete")
    if before["checkpoint_sha256"] != after["checkpoint_sha256"]:
        raise ValueError("this audit isolates interfaces, not weight changes")
    source = {r["case_id"]: r for r in before["results"]}
    rows = []
    for row in after["results"]:
        old = source[row["case_id"]]
        if row["course"] != old["course"]:
            raise ValueError("unmatched physical course")
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
        rows.append({"case_id": row["case_id"], "before_trace_sha256": old["trace_sha256"],
                     "after_trace_sha256": row["trace_sha256"], "all_arrays_bitwise_equal": equal,
                     "first_changed_force_tick": first,
                     "before_success": old["result"]["success"], "after_success": row["result"]["success"],
                     "before_steps": old["result"]["steps"], "after_steps": row["result"]["steps"]})
    report = {"status": "complete", "checkpoint_sha256": before["checkpoint_sha256"],
              "before_manifest_sha256": sha256(args.before), "after_manifest_sha256": sha256(args.after),
              "cases": len(rows), "bitwise_unchanged": sum(r["all_arrays_bitwise_equal"] for r in rows),
              "repairs": sum(not r["before_success"] and r["after_success"] for r in rows),
              "regressions": sum(r["before_success"] and not r["after_success"] for r in rows),
              "results": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "results"}))


if __name__ == "__main__":
    main()
