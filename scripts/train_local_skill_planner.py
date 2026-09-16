#!/usr/bin/env python3
"""Fit the recurrent local planner on certified trajectory shards.

All optimization is node local. Data contains body-frame observations and next
twist targets only. Reported fit loss is a training diagnostic, not a success
rate or held-out validation result. Closed-loop evaluation is a separate gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from audit_local_skill_composition import sha256, write_json
from piano_movers.local_skill_policy import LocalSkillPlanner, LOCAL_SKILL_FEATURE_DIM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--additional-dataset", type=Path, action="append", default=[])
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--seed", type=int, default=6101)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--collective-loss", action="store_true")
    parser.add_argument("--goal-residual", action="store_true")
    parser.add_argument("--private-loss-weight", type=float, default=0.1)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text())
    if dataset["status"] != "complete" and not args.allow_partial:
        raise ValueError("dataset generation is still running")
    if dataset["heldout_shapes"] != ["S", "Z"]:
        raise ValueError("dataset holdout contract changed")
    datasets = [(args.dataset, dataset)]
    for path in args.additional_dataset:
        extra = json.loads(path.read_text())
        if extra["status"] != "complete" or extra["heldout_shapes"] != ["S", "Z"]:
            raise ValueError("additional dataset incomplete or violates holdouts")
        if extra["parent_checkpoint_sha256"] != dataset["parent_checkpoint_sha256"]:
            raise ValueError("dataset parent lineage mismatch")
        datasets.append((path, extra))
    entries = [
        (path.parent, row) for path, data in datasets for row in data["results"] if row["success"]
    ]
    if not entries:
        raise ValueError("no admitted executable teachers")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "dataset_snapshot.json").exists():
        raise FileExistsError(args.output)
    write_json(args.output / "dataset_snapshot.json", dataset)
    for index, (_, extra) in enumerate(datasets[1:]):
        write_json(args.output / f"additional_dataset_{index}.json", extra)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    dev = torch.device(args.device)
    features, targets, masks, starts, lengths, counts, labelled = [], [], [], [], [], [], []
    offset = 0
    for directory, row in entries:
        path = directory / row["artifact"]
        if sha256(path) != row["artifact_sha256"]:
            raise ValueError(f"artifact hash mismatch: {path}")
        if row["shape"] in ("S", "Z") or row["agents"] % 2:
            raise ValueError("held-out data entered training")
        data = np.load(path)
        x, y = data["features"], data["target_twist"]
        if x.shape[-1] != LOCAL_SKILL_FEATURE_DIM or y.shape != (len(x), 3):
            raise ValueError("malformed local dataset shard")
        if np.max(np.abs(y)) > 1.001:
            raise ValueError("target twist outside model output range")
        padded = np.zeros((len(x), 20, LOCAL_SKILL_FEATURE_DIM), np.float32)
        padded[:, : x.shape[1]] = x
        starts.append(offset)
        lengths.append(len(x))
        counts.append(x.shape[1])
        offset += len(x)
        features.append(torch.from_numpy(padded))
        targets.append(torch.from_numpy(y.astype(np.float32)))
        mask = data["target_mask"] if "target_mask" in data else np.ones(len(x), dtype=bool)
        if mask.shape != (len(x),) or not mask.any():
            raise ValueError("shard has no supervised steps")
        masks.append(torch.from_numpy(mask))
        labelled.append(torch.from_numpy(np.flatnonzero(mask)))
    x = torch.cat(features).to(dev)
    y = torch.cat(targets).to(dev)
    target_mask = torch.cat(masks).to(dev)
    label_count = torch.tensor([len(v) for v in labelled], device=dev)
    label_index = torch.nn.utils.rnn.pad_sequence(labelled, batch_first=True).to(dev)
    starts = torch.tensor(starts, device=dev)
    lengths = torch.tensor(lengths, device=dev)
    counts = torch.tensor(counts, device=dev)
    model = LocalSkillPlanner(goal_residual=args.goal_residual).to(dev)
    if args.resume:
        saved = torch.load(args.resume, map_location=dev, weights_only=False)
        if saved["parent_checkpoint_sha256"] != dataset["parent_checkpoint_sha256"]:
            raise ValueError("resume parent lineage mismatch")
        if saved.get("goal_residual", False) != args.goal_residual:
            raise ValueError("resume would silently change planner output semantics")
        model.load_state_dict(saved["policy_state_dict"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4)
    offsets = torch.arange(args.sequence_length, device=dev)[None, :]
    record = {
        "schema": "piano-movers.p6-local-planner-training.v1",
        "status": "running",
        "parent_checkpoint_sha256": dataset["parent_checkpoint_sha256"],
        "dataset_snapshot_sha256": sha256(args.output / "dataset_snapshot.json"),
        "additional_dataset_sha256": [sha256(path) for path, _ in datasets[1:]],
        "resume_checkpoint_sha256": sha256(args.resume) if args.resume else None,
        "device": str(dev),
        "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
        "source_sha256": {
            str(p): sha256(p)
            for p in [Path(__file__), Path("src/piano_movers/local_skill_policy.py")]
        },
        "seed": args.seed,
        "episodes": len(entries),
        "frames": len(x),
        "agent_frames": sum(
            length * count for length, count in zip(lengths.tolist(), counts.tolist())
        ),
        "parameters": sum(p.numel() for p in model.parameters()),
        "steps_requested": args.steps,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "learning_rate": args.learning_rate,
        "collective_loss": args.collective_loss,
        "goal_residual": args.goal_residual,
        "private_loss_weight": args.private_loss_weight,
        "history": [],
        "closed_loop_success": None,
        "checkpoint": None,
    }
    write_json(args.output / "training.json", record)
    started = time.monotonic()
    losses = []
    for step in range(1, args.steps + 1):
        episode = torch.randint(len(entries), (args.batch_size,), device=dev)
        node = (torch.rand(args.batch_size, device=dev) * counts[episode]).long()
        choice = (torch.rand(args.batch_size, device=dev) * label_count[episode]).long()
        anchor = label_index[episode, choice]
        begin = (
            anchor - torch.randint(4, args.sequence_length, (args.batch_size,), device=dev)
        ).clamp_min(0)
        local = begin[:, None] + offsets
        valid = local < lengths[episode, None]
        idx = starts[episode, None] + torch.minimum(local, lengths[episode, None] - 1)
        target = y[idx]
        mask = valid & target_mask[idx] & ((offsets >= 4) | (begin[:, None] == 0))
        # Sample complete episode/handle sequences, then ignore burn-in loss;
        # memory cannot cross into another trajectory or a held-out episode.
        if args.collective_loss:
            batch = x[idx]
            b, t, n, f = batch.shape
            prediction, _ = model.forward_sequence(batch.permute(0, 2, 1, 3).reshape(b * n, t, f))
            prediction = prediction.view(b, n, t, 3).permute(0, 2, 1, 3)
            node_mask = (torch.arange(n, device=dev)[None, :] < counts[episode, None])[
                :, None, :, None
            ]
            collective = (prediction * node_mask).sum(2) / counts[episode, None, None]
            collective_loss = F.smooth_l1_loss(
                collective, target, reduction="none", beta=0.05
            ).mean(-1)
            private_loss = F.smooth_l1_loss(
                prediction, target[:, :, None, :].expand_as(prediction), reduction="none", beta=0.05
            )
            private_loss = (private_loss * node_mask).sum((2, 3)) / (3 * counts[episode, None])
            # Full-convergence physical GBP approaches this common-precision
            # mean. This is a collective imitation surrogate, not unrolled
            # GBP or a claim of environment return optimization.
            per_step = collective_loss + args.private_loss_weight * private_loss
        else:
            prediction, _ = model.forward_sequence(x[idx, node[:, None]])
            per_step = F.smooth_l1_loss(prediction, target, reduction="none", beta=0.05).mean(-1)
        loss = (per_step * mask).sum() / mask.sum().clamp_min(1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step == 1 or step % 100 == 0 or step == args.steps:
            point = {
                "step": step,
                "fit_loss": float(np.mean(losses)),
                "seconds": time.monotonic() - started,
            }
            losses = []
            record["history"].append(point)
            record["step"] = step
            print(json.dumps(point), flush=True)
            payload = {
                "schema": record["schema"],
                "parent_checkpoint_sha256": record["parent_checkpoint_sha256"],
                "feature_dim": LOCAL_SKILL_FEATURE_DIM,
                "hidden_dim": model.hidden_dim,
                "goal_residual": model.goal_residual,
                "policy_state_dict": model.state_dict(),
                "training": record,
                "step": step,
                "dataset_snapshot_sha256": record["dataset_snapshot_sha256"],
            }
            ckpt = args.output / "local_planner_latest.pt"
            temporary = args.output / "local_planner_latest.tmp"
            torch.save(payload, temporary)
            temporary.replace(ckpt)
            record["checkpoint"] = {"path": ckpt.name, "sha256": sha256(ckpt)}
            write_json(args.output / "training.json", record)
    record["status"] = "complete"
    write_json(args.output / "training.json", record)


if __name__ == "__main__":
    main()
