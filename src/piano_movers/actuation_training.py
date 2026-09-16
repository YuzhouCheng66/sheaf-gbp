"""Differentiable replay of the EXISTING physical Sheaf-GBP actuation path.

An offline training utility, not a new deployment planner or solver. Geometry
proposals/masks are cached from the original local geometry routine; its first
six outputs and active mask do not depend on the private learned plan. Only
coordinates 3:6 of the executed plan affect the current inverse dynamics.
Every communicated message is produced by the same policy-owned GBP module.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from .gbp import GBPMessageTrace, homogeneous_topology
from .models import SheafGBPPolicy
from .single_slit_local_planner import local_se2_inverse_dynamics_wrench_proposals


@dataclass
class ActuationReplay:
    forces: Tensor
    executed_plan: Tensor
    wrench_section: Tensor
    message_trace: GBPMessageTrace | None


def replay_physical_actuation(
    policy: SheafGBPPolicy, obs: Tensor, private_plan: Tensor,
    geometry_plan: Tensor, geometry_active: Tensor, *, trace_messages: bool = False,
) -> ActuationReplay:
    """Retain original geometry authority and backpropagate through real forces.

Inputs are node-local proposed plans and the original geometry branch's cached
plan/mask. This does not make a teacher or a source policy available at runtime.
The caller must authenticate cached tensors against real forward passes.
"""
    arm, cfg = policy.arm_cfg, policy.env_cfg
    if not (arm.local_se2_distributed_wrench_control
            and arm.local_se2_distributed_next_se2_control
            and arm.local_se2_analytic_total_wrench
            and arm.local_se2_distributed_grasp_allreduce
            and arm.local_se2_route_free_geometry_planner
            and not arm.return_source_candidate):
        raise ValueError("requires the audited physical next-SE(2)/GBP/grasp policy")
    expected = (*obs.shape[:2], 8)
    if private_plan.shape != expected or geometry_plan.shape != expected:
        raise ValueError("plans must be [batch,node,8]")
    if geometry_active.shape != (*obs.shape[:2], 1) or geometry_active.dtype != torch.bool:
        raise ValueError("geometry authority must be [batch,node,1] boolean")
    executed = torch.where(geometry_active, geometry_plan, private_plan)
    proposal = local_se2_inverse_dynamics_wrench_proposals(obs, executed, cfg)
    # Public precision is overwritten by this fixed physical confidence in
    # the full policy. Private precisions cannot affect the public I3 section.
    public_q = torch.full_like(proposal, arm.local_se2_analytic_unary_precision)
    public_q = public_q * (1. + (arm.local_se2_geometry_evidence_precision_boost - 1.)
                           * geometry_active.to(obs.dtype))
    q = torch.cat((public_q, obs.new_ones((*obs.shape[:2], arm.correction_stalk_dim - 3))), -1)
    natural = torch.cat((public_q * proposal, torch.zeros_like(q[..., 3:])), -1)
    edges = policy.local_se2_analytic_reference_edges
    topology = homogeneous_topology(edges, obs.shape[0], cfg.n_agents, device=obs.device)
    node_r = policy.correction_analytic_restrictions(obs)
    identity = torch.eye(3, device=obs.device, dtype=obs.dtype).expand(*obs.shape[:2], 3, 3)
    if not torch.equal(node_r[..., :3], identity) or bool(node_r[..., 3:].count_nonzero()):
        raise ValueError("this replay only supports the existing physical [I3,0] restriction")
    directed_r = policy.correction_directed_restrictions(node_r, edges)
    sigma2 = obs.new_full((obs.shape[0], 2 * len(edges), 3), arm.local_se2_analytic_edge_sigma2)
    sigma2 = torch.where(policy.local_se2_analytic_tree_edge_mask.repeat_interleave(2)[None, :, None],
                         sigma2, torch.full_like(sigma2, 1e6))
    solved = policy.local_se2_analytic_gbp(q, natural, directed_r, sigma2, topology,
                                         trace_messages=trace_messages)
    section = torch.einsum("bnev,bnv->bne", node_r, solved.mean)
    forces = policy.distributed_force_from_local_grasp_statistics(section, obs, edges,
                                                                  trace_messages=False)[0]
    forces = policy.clamp_force_norm(forces)
    return ActuationReplay(forces, executed, section, solved.message_trace)
