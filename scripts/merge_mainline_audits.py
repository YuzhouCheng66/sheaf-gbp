#!/usr/bin/env python3
"""Merge complete disjoint audit shards without copying or relabelling traces."""

import argparse
import json
from pathlib import Path

from audit_local_skill_composition import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    record, keys, rows = None, set(), []
    for path in args.input:
        source = json.loads(path.read_text())
        if source["status"] != "complete":
            raise ValueError("cannot merge an unfinished audit")
        if record is None:
            record = dict(source)
        for key in ("schema", "split", "checkpoint_sha256", "source_sha256", "device"):
            if source[key] != record[key]:
                raise ValueError(f"mixed audit semantics: {key}")
        for row in source["results"]:
            key = row["variant"], row["shape"], row["agents"]
            if key in keys:
                raise ValueError(f"duplicate case: {key}")
            trace = (path.parent / row["trace"]).resolve()
            if sha256(trace) != row["trace_sha256"]:
                raise ValueError("trace checksum mismatch")
            keys.add(key)
            rows.append(dict(row, trace=str(trace)))
    if len(rows) != args.expected_cases:
        raise ValueError("case count differs from the declared panel")
    record.update(results=rows, requested={
        "shapes": sorted({r["shape"] for r in rows}),
        "agents": sorted({r["agents"] for r in rows}),
        "variants": sorted({r["variant"] for r in rows}),
    }, source_manifests=[{"path": str(p.resolve()), "sha256": sha256(p)} for p in args.input])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, record)
    print(json.dumps({"cases": len(rows), "successes": sum(r["result"]["success"] for r in rows)}))


if __name__ == "__main__":
    main()
