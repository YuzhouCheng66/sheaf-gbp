#!/usr/bin/env python3
"""Diagnostic-only common coasting command through the real GBP and decoder.

Simulator twist constructs the known-zero-wrench counterfactual only. No
rollout policy, checkpoint, demo, message dimension or safety gate is changed.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from audit_local_skill_composition import sha256, write_json
from audit_mainline_weighted_velocity_statistics import coasting_plan
from train_mainline_skill_dagger import active_anchor_config
from piano_movers.env import radius_edges
from piano_movers.gbp import homogeneous_topology
from piano_movers.multi_waypoint_demo import Slit, WaypointCourse, _course_config, INTERMEDIATE_POSITION_TOLERANCE_M
from piano_movers.single_slit_0720_eval import load_single_slit_checkpoint


def tree_diameter(edges, mask, nodes):
    adjacency = [[] for _ in range(nodes)]
    for i, j in edges[mask].tolist():
        adjacency[i].append(j)
        adjacency[j].append(i)
    diameter = 0
    for root in range(nodes):
        distances, queue = {root: 0}, [root]
        for node in queue:
            for peer in adjacency[node]:
                if peer not in distances:
                    distances[peer] = distances[node] + 1
                    queue.append(peer)
        if len(distances) != nodes:
            raise ValueError("source coordination tree is disconnected")
        diameter = max(diameter, max(distances.values()))
    return diameter


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--statistics-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    source = json.loads(args.rollouts.read_text())
    diagnostic = json.loads(args.statistics_audit.read_text())
    if source["status"] != "complete" or diagnostic["rollout_sha256"] != sha256(args.rollouts):
        raise ValueError("diagnostic provenance mismatch")
    loaded = load_single_slit_checkpoint(Path(source["checkpoint"]), device="cpu")
    if loaded.checkpoint_sha256 != source["checkpoint_sha256"]:
        raise ValueError("checkpoint mismatch")
    by_case = {r["case_id"]: r for r in source["results"]}
    chosen = {}
    for row in sorted(diagnostic["results"], key=lambda r: -r["weighted_max_abs_normalized_wrench"]):
        chosen.setdefault(row["shape"], row)
    results = []
    for probe in chosen.values():
        row = by_case[probe["case_id"]]
        path = args.rollouts.parent / row["trace"]
        if sha256(path) != row["trace_sha256"]:
            raise ValueError("trace mismatch")
        a, h, opt = np.load(path), row["header"], row["options"]
        tick = probe["worst_tick"]
        course = WaypointCourse(tuple(h["source"]), tuple(Slit(**w) for w in h["walls"]),
                                tuple(tuple(g) for g in h["waypoints"]), h["world"]["x"], h["world"]["y"],
                                tuple(h.get("source_velocity", (0., 0., 0.))), h.get("morphology_scale", 1.))
        cfg = _course_config(loaded.cfg, row["agents"], course, row["shape"])
        if opt["nominal_glyph_size"]:
            cfg = replace(cfg, glyph_scale=1.)
        cfg = replace(cfg, complete_local_corridor_exit=opt["complete_corridor_exit"],
                      local_cspace_collision_substeps=opt["planning_substeps"],
                      local_cspace_validate_all_lookaheads=opt.get("validate_all_cspace_lookaheads", False),
                      local_support_check_alignment_sweep=opt.get("check_support_alignment_sweep", False),
                      local_corridor_exit_goal_radius_m=INTERMEDIATE_POSITION_TOLERANCE_M
                      if opt.get("respect_handoff_goal_region", False) else 0.)
        cfg = active_anchor_config(cfg, course, int(a["waypoint_index"][tick]), opt["position_only_control"])
        loaded.policy.bind_environment(cfg)
        obs = torch.from_numpy(a["observation"][tick:tick + 1])
        hidden = torch.from_numpy(a["private_hidden"][tick:tick + 1])
        state = torch.from_numpy(a["state"][tick:tick + 1])
        edges = radius_edges(cfg, device="cpu")
        actual = loaded.policy(obs, edges, hidden)
        torch.testing.assert_close(actual.forces, torch.from_numpy(a["forces"][tick:tick + 1]), atol=2e-5, rtol=2e-5)
        command = coasting_plan(obs, state[..., 3:6], cfg)
        arm = loaded.policy.arm_cfg
        topology = homogeneous_topology(edges, 1, cfg.n_agents, device="cpu")
        node_r = loaded.policy.correction_analytic_restrictions(obs)
        directed_r = loaded.policy.correction_directed_restrictions(node_r, edges)
        tree = loaded.policy.local_se2_analytic_tree_edge_mask
        sigma2 = obs.new_full((1, 2 * len(edges), arm.correction_edge_dim), arm.local_se2_analytic_edge_sigma2)
        sigma2 = torch.where(tree.repeat_interleave(2)[None, :, None], sigma2, torch.full_like(sigma2, 1e6))
        diameter = tree_diameter(edges, tree, cfg.n_agents)
        original_steps = loaded.policy.local_se2_analytic_gbp.num_steps
        active = actual.aux["local_se2_terminal_attractor_active"].to(obs.dtype)

        def solve_from_local_proposals(proposal, boost):
            q = obs.new_ones((1, cfg.n_agents, arm.correction_stalk_dim))
            q[..., :3] = 1. + (boost - 1.) * active
            natural = torch.zeros_like(q)
            natural[..., :3] = q[..., :3] * proposal
            solved = loaded.policy.local_se2_analytic_gbp(q, natural, directed_r, sigma2, topology)
            section = torch.einsum("bnev,bnv->bne", node_r, solved.mean)
            raw_force = loaded.policy.distributed_force_from_local_grasp_statistics(
                section, obs, edges, trace_messages=False,
            )[0]
            return loaded.policy.clamp_force_norm(raw_force)

        # Authenticate this reduced public-coordinate replay against the
        # complete original policy before constructing any counterfactual.
        reproduced = solve_from_local_proposals(actual.aux["local_se2_analytic_total_wrench_proposal"],
                                                arm.local_se2_geometry_evidence_precision_boost)
        torch.testing.assert_close(reproduced, actual.forces, atol=2e-5, rtol=2e-5)
        from piano_movers.single_slit_local_planner import local_se2_inverse_dynamics_wrench_proposals

        proposal = local_se2_inverse_dynamics_wrench_proposals(obs, command, cfg)
        contrasts = []
        try:
            for steps in sorted({original_steps, max(original_steps, diameter)}):
                loaded.policy.local_se2_analytic_gbp.num_steps = steps
                for boost in (arm.local_se2_geometry_evidence_precision_boost, 1.):
                    force = solve_from_local_proposals(proposal, boost)
                    sine, cosine, rx, ry = (obs[..., j] for j in (0, 1, 2, 3))
                    x, y = cosine * rx - sine * ry, sine * rx + cosine * ry
                    net = torch.stack((force[..., 0].sum(1), force[..., 1].sum(1),
                                       (x * force[..., 1] - y * force[..., 0]).sum(1)), -1)
                    scale = net.new_tensor([cfg.n_agents * cfg.max_force] * 2 + [.75 * cfg.n_agents * cfg.max_force])
                    contrasts.append({"precision_boost": boost, "GBP_rounds": steps,
                                      "normalized_net_wrench": (net / scale)[0].tolist(),
                                      "physical_net_wrench_N_N_Nm": net[0].tolist(),
                                      "clipped_nodes": int((force.norm(dim=-1) >= cfg.max_force * (1 - 1e-6)).sum())})
        finally:
            loaded.policy.local_se2_analytic_gbp.num_steps = original_steps
        results.append({"shape": row["shape"], "case_id": row["case_id"], "tick": tick,
                        "trace_sha256": row["trace_sha256"], "tree_diameter": diameter,
                        "original_policy_force_replay_verified": True, "contrasts": contrasts})
    report = {"status": "complete", "checkpoint_sha256": loaded.checkpoint_sha256,
              "rollout_sha256": sha256(args.rollouts), "statistics_audit_sha256": sha256(args.statistics_audit),
              "weights_or_deployment_modified": False,
              "simulator_twist_used_only_to_construct_known_zero_wrench_test": True,
              "method": "direct public-coordinate replay retains authentic confidence masks; final_plan_override is NOT used because it resets those masks",
              "scope": "actual finite-round GBP, distributed decoder and clipping; no success-rate inference",
              "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
