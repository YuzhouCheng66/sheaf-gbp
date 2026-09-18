#!/usr/bin/env python3
"""Compare actual HTTP-streamed demo states with authenticated evaluated traces."""

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from audit_local_skill_composition import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agents", type=int, default=12)
    parser.add_argument("--shapes", default="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.shapes or len(set(args.shapes)) != len(args.shapes):
        raise ValueError("require unique nonempty shapes")
    source = json.loads(args.reference.read_text())
    if source["status"] != "complete":
        raise ValueError("reference evaluation incomplete")
    with urlopen(args.url + "/api/model", timeout=10) as response:
        model = json.load(response)
    if model["checkpoint_sha256"] != source["checkpoint_sha256"] or model["mode"] != "authenticated_recipe":
        raise ValueError("served model does not match the authenticated candidate")
    report = {"status": "running", "reference_sha256": sha256(args.reference),
              "server": args.url, "model": model, "source_sha256": sha256(Path(__file__)),
              "requested_shapes": args.shapes, "agents": args.agents, "results": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    for shape in args.shapes:
        matches = [r for r in source["results"] if r["shape"] == shape and r["agents"] == args.agents]
        if len(matches) != 1:
            raise ValueError("require exactly one corresponding reference episode")
        old = matches[0]
        if model["composition_options"] != old["options"] or not old["continuous_certificate"]["passed"]:
            raise ValueError("unmatched options or uncertified reference")
        trace = args.reference.parent / old["trace"]
        if sha256(trace) != old["trace_sha256"]:
            raise ValueError("reference trace hash mismatch")
        with urlopen(f"{args.url}/api/stream?shape={shape}&agents={args.agents}", timeout=60) as response:
            events = [json.loads(line) for line in response if line.strip()]
        if events[0] != old["header"] or events[-1] != old["result"]:
            raise ValueError(f"{shape}: served geometry/configuration or outcome differs")
        frames = events[1:-1]
        if not frames or any(f["type"] != "frame" for f in frames):
            raise ValueError("malformed or failed HTTP stream")
        with np.load(trace) as saved:
            for key in ("state", "next_state"):
                actual = np.asarray([f[key] for f in frames], dtype=saved[key].dtype)
                if not np.array_equal(actual, saved[key]):
                    raise ValueError(f"{shape}: streamed {key} differs from certified evaluation")
        row = {"shape": shape, "agents": args.agents, "trace_sha256": old["trace_sha256"],
               "states_bitwise_equal": True, "header_equal": True, "outcome_equal": True,
               "result": events[-1]}
        report["results"].append(row)
        write_json(args.output, report)
        print(json.dumps(row), flush=True)
    report["status"] = "complete"
    write_json(args.output, report)


if __name__ == "__main__":
    main()
