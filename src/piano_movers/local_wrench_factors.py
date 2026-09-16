"""Experimental local mechanical information factors, not deployed by p5/p6.

This module is a mathematical control for the weighted-statistics audit. It
does not insert a planner, alter a checkpoint, or solve a centralized control
problem at deployment. The returned rank-two factor is local to one handle;
integration into the actual distributed GBP protocol requires separate work.
"""

import torch
from torch import Tensor

from .config import EnvConfig
from .single_slit_local_planner import (
    LOCAL_SE2_LINEAR_VELOCITY_SCALE_MPS,
    LOCAL_SE2_ANGULAR_VELOCITY_SCALE_RADPS,
)


def local_wrench_information_factor(
    obs: Tensor, plan: Tensor, cfg: EnvConfig, confidence: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return a local (precision3x3, natural3) over normalized BODY wrench.

    An attached handle measures c_i = A_i v, where
    A_i = [[1, 0, -r_y], [0, 1, r_x]]. A normalized wrench z changes body twist
    by D z before damping, D = dt diag(N*fmax/m, N*fmax/m, .75*N*fmax/I).
    Its own desired twist u_i therefore defines

        ||A_i D z - (A_i decay^-1 u_i - c_i)||^2_{confidence_i} / 2.

    Unlike reweighting already-inverted partial measurements, this retains
    each handle's actual information matrix. No peer observation, peer
    reduction, global simulator twist, or future target is read here.
    Confidence expresses only relative scalar weighting; this function does
    not assert that the choices are calibrated noise precisions.
    """
    if obs.ndim != 3 or obs.shape[-1] < 18 or plan.shape != (*obs.shape[:2], 8):
        raise ValueError("requires local [batch,node,observation] and plan8")
    if confidence.shape != obs.shape[:2] or bool((confidence <= 0).any()):
        raise ValueError("requires one positive confidence per local node")
    sine, cosine = obs[..., 0], obs[..., 1]
    rx, ry = obs[..., 2], obs[..., 3]
    own_velocity = 2. * obs[..., 6:8]
    own_body_velocity = torch.stack((
        cosine * own_velocity[..., 0] + sine * own_velocity[..., 1],
        -sine * own_velocity[..., 0] + cosine * own_velocity[..., 1],
    ), -1)
    desired_translation = (plan[..., 3:5] * LOCAL_SE2_LINEAR_VELOCITY_SCALE_MPS
                           / (1. - cfg.linear_damping * cfg.dt))
    desired_rotation = (plan[..., 5] * LOCAL_SE2_ANGULAR_VELOCITY_SCALE_RADPS
                        / (1. - cfg.angular_damping * cfg.dt))
    desired_handle = desired_translation + torch.stack((-ry * desired_rotation, rx * desired_rotation), -1)
    residual_target = desired_handle - own_body_velocity
    force_scale = cfg.n_agents * cfg.max_force
    linear = cfg.dt * force_scale / cfg.mass
    angular = cfg.dt * .75 * force_scale / cfg.inertia
    one, zero = torch.ones_like(rx), torch.zeros_like(rx)
    h = torch.stack((
        torch.stack((linear * one, zero, -angular * ry), -1),
        torch.stack((zero, linear * one, angular * rx), -1),
    ), -2)
    precision = confidence[..., None, None] * h.transpose(-1, -2) @ h
    natural = confidence[..., None] * (h.transpose(-1, -2) @ residual_target[..., None]).squeeze(-1)
    return precision, natural
