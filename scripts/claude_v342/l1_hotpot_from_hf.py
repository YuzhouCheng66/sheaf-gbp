"""Write HotpotQA (distractor setting) in the official JSON layout from the Hugging Face mirror.

Fallback for run_l1_data.sh when http://curtis.ml.cmu.edu is unreachable (it did not resolve from uk-linux on
2026-10-29). The mirror ``hotpotqa/hotpot_qa`` (config ``distractor``) holds the same examples with a columnar
schema; this script rewrites them as the list-of-dicts layout that l1_data.load_hotpot_json expects:
    {_id, question, answer, type, level, supporting_facts: [[title, sent_idx], ...], context: [[title, [sentence, ...]], ...]}
Output files: <out>/hotpot_train_v1.1.json (train split) and <out>/hotpot_dev_distractor_v1.json (validation
split), plus <out>/hf_source.json recording the mirror, the split sizes and the sha256 of each written file.
Data stays local (never in the repo).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os


def convert(split):
    out = []
    for ex in split:
        ctx = [[t, list(s)] for t, s in zip(ex["context"]["title"], ex["context"]["sentences"])]
        sup = [[t, int(i)] for t, i in zip(ex["supporting_facts"]["title"], ex["supporting_facts"]["sent_id"])]
        out.append({"_id": ex["id"], "question": ex["question"], "answer": ex["answer"], "type": ex["type"],
                    "level": ex["level"], "supporting_facts": sup, "context": ctx})
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="directory for the two JSON files (local data dir)")
    ap.add_argument("--repo", default="hotpotqa/hotpot_qa")
    ap.add_argument("--config", default="distractor")
    args = ap.parse_args()
    from datasets import load_dataset  # imported here so --help works without the package

    ds = load_dataset(args.repo, args.config)
    os.makedirs(args.out, exist_ok=True)
    meta = {"source": f"{args.repo}:{args.config}", "files": {}}
    for split, fname in (("train", "hotpot_train_v1.1.json"), ("validation", "hotpot_dev_distractor_v1.json")):
        rows = convert(ds[split])
        path = os.path.join(args.out, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False)
        meta["files"][fname] = {"split": split, "n": len(rows), "sha256": sha256(path), "bytes": os.path.getsize(path)}
        print(f"{fname}: {len(rows)} examples, sha256 {meta['files'][fname]['sha256']}")
    with open(os.path.join(args.out, "hf_source.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
