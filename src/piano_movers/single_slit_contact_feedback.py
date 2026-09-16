"""Versioned, penetration-free contact feedback for the single-slit skill.

Legacy evaluators terminate an episode when a force proposal intersects the
wall.  This module deliberately does not change that contract.  It provides a
separate transition primitive for experiments in which a rejected contact is
an observation: the attempted transition is audited with the existing swept
exact-SAT authority, while a blocked payload remains at its last safe pose and
loses all momentum before the controller receives its next observation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import EnvConfig
from .env import (
    PianoBatch,
    collision_clearance_margin,
    step_dynamics,
    transition_clearance_margin,
)


SINGLE_SLIT_CONTACT_FEEDBACK_SCHEMA_V1 = (
    "piano-movers-single-slit-contact-feedback-step-v1"
)
# CUDA evaluates a swept endpoint in a ``batch * substeps`` SAT call and the
# following current pose in a ``batch`` SAT call.  Their different reduction
# shapes can disagree by a few 1e-8 m at exact tangency.  Refuse proposals in
# this conservative micron-scale band so an accepted pose remains positive
# under either execution shape.
CONTACT_FEEDBACK_CLEARANCE_RESERVE_M = 1.0e-6


@dataclass(frozen=True)
class SingleSlitContactFeedbackResultV1:
    """Outcome of one force step under the V1 contact-feedback contract.

    ``attempted_state`` is always the ordinary unconstrained dynamics result.
    ``resolved_state`` is identical to it for safe batch elements.  For a hit
    or a proposal inside the numerical clearance reserve,
    the pose is copied bitwise from the input state and all three velocity
    coordinates are set to exact zero.  Clearances use the same swept SAT
    substeps and sign convention as the legacy collision authority.
    """

    resolved_state: Tensor
    attempted_state: Tensor
    attempted_clearance_m: Tensor
    attempted_hit: Tensor
    resolved_clearance_m: Tensor
    schema_version: str = SINGLE_SLIT_CONTACT_FEEDBACK_SCHEMA_V1

    @property
    def accepted(self) -> Tensor:
        """Boolean batch mask for proposals executed without contact."""

        return ~self.attempted_hit


def _validate_inputs(state: Tensor, forces: Tensor, cfg: EnvConfig) -> None:
    if state.ndim != 2 or state.shape[-1] != 6 or state.shape[0] <= 0:
        raise ValueError("contact-feedback state must have shape [batch,6]")
    expected_forces = (state.shape[0], cfg.n_agents, 2)
    if tuple(forces.shape) != expected_forces:
        raise ValueError(
            "contact-feedback forces must have shape "
            f"{expected_forces}, got {tuple(forces.shape)}"
        )
    if not state.dtype.is_floating_point or not forces.dtype.is_floating_point:
        raise ValueError("contact-feedback state and forces must be floating point")
    if state.device != forces.device or state.dtype != forces.dtype:
        raise ValueError("contact-feedback state and forces must share device/dtype")
    if not bool(torch.isfinite(state).all()) or not bool(torch.isfinite(forces).all()):
        raise ValueError("contact-feedback state and forces must be finite")


def single_slit_contact_feedback_step_v1(
    state: Tensor,
    forces: Tensor,
    scenario: PianoBatch,
    cfg: EnvConfig,
) -> SingleSlitContactFeedbackResultV1:
    """Execute one force proposal with rejection-and-stop contact feedback.

    The caller must supply a collision-free current pose.  This precondition
    makes the no-penetration guarantee explicit: a blocked proposal cannot be
    resolved by retaining an already invalid pose.  Attempted contact is the
    exact swept signed-SAT clearance with a one-micron conservative numerical
    reserve.  This prevents CUDA batch-shape rounding at exact tangency from
    accepting a pose that is negative by only a few tens of nanometres on the
    next control call.
    """

    _validate_inputs(state, forces, cfg)
    # The exact signed-SAT clearance is itself a collision authority: its
    # ``<= 0`` sign is exhaustively parity-tested against ``collision_mask``
    # (including wall contact and the strict world-boundary convention).
    # Deriving the bit from the already-required clearance avoids recomputing
    # identical rectangle projections in this per-control-tick hot path.
    current_clearance = collision_clearance_margin(state, scenario, cfg)
    current_hit = current_clearance <= 0.0
    if bool(current_hit.any()):
        indices = torch.nonzero(current_hit, as_tuple=False).flatten()
        raise ValueError(
            "contact feedback requires a collision-free current pose; "
            f"batch_indices={indices.detach().cpu().tolist()}, "
            "minimum_clearance_m="
            f"{float(current_clearance.amin().detach().cpu().item()):.12g}"
        )

    attempted_state = step_dynamics(state, forces, cfg)
    attempted_clearance = transition_clearance_margin(
        state,
        attempted_state,
        scenario,
        cfg,
    )
    attempted_hit = attempted_clearance <= CONTACT_FEEDBACK_CLEARANCE_RESERVE_M

    stopped_at_contact = torch.cat(
        (state[:, :3], torch.zeros_like(state[:, 3:])),
        dim=-1,
    )
    resolved_state = torch.where(
        attempted_hit[:, None],
        stopped_at_contact,
        attempted_state,
    )
    resolved_clearance = transition_clearance_margin(
        state,
        resolved_state,
        scenario,
        cfg,
    )
    resolved_hit = resolved_clearance <= 0.0
    if bool(resolved_hit.any()):
        raise RuntimeError("contact-feedback resolution produced penetration")

    return SingleSlitContactFeedbackResultV1(
        resolved_state=resolved_state,
        attempted_state=attempted_state,
        attempted_clearance_m=attempted_clearance,
        attempted_hit=attempted_hit,
        resolved_clearance_m=resolved_clearance,
    )


__all__ = [
    "CONTACT_FEEDBACK_CLEARANCE_RESERVE_M",
    "SINGLE_SLIT_CONTACT_FEEDBACK_SCHEMA_V1",
    "SingleSlitContactFeedbackResultV1",
    "single_slit_contact_feedback_step_v1",
]
