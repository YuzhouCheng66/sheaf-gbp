"""Versioned local observations for rejected-contact feedback.

The legacy single-slit policy observes a 22-D row at every rigidly attached
handle.  That row describes only the *resolved* payload state, so a rejected
force proposal is otherwise erased before the next policy call.  This module
adds a small, explicit mechanical aftereffect without changing the legacy
dispatcher, model, checkpoint loader, or evaluator.

For handle ``i``, the five appended channels are::

    [last_hit,
     previous_force_i.x / max_force,
     previous_force_i.y / max_force,
     rejected_handle_velocity_i.x / 2 m/s,
     rejected_handle_velocity_i.y / 2 m/s]

The rejected handle velocity is the exact finite rigid-body displacement from
the resolved pose to the ordinary attempted pose, divided by ``dt``.  It is
structurally zero on an accepted transition.  Different handles therefore see
different residuals when the rejected proposal contained rotation, while no
agent receives another agent's force or observation through this local input.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .config import EnvConfig
from .env import PianoBatch, body_to_world, contact_points
from .single_slit_contact_feedback import (
    CONTACT_FEEDBACK_CLEARANCE_RESERVE_M,
    SINGLE_SLIT_CONTACT_FEEDBACK_SCHEMA_V1,
    SingleSlitContactFeedbackResultV1,
)
from .single_slit_feedback import (
    SINGLE_SLIT_FEEDBACK_OBSERVATION_DIM,
    single_slit_feedback_observations,
)


SINGLE_SLIT_CONTACT_OBSERVATION_SCHEMA_V1 = (
    "piano-movers-single-slit-contact-observation27-v1"
)
SINGLE_SLIT_CONTACT_FEEDBACK_FEATURE_DIM_V1 = 5
SINGLE_SLIT_CONTACT_OBSERVATION_DIM_V1 = (
    SINGLE_SLIT_FEEDBACK_OBSERVATION_DIM
    + SINGLE_SLIT_CONTACT_FEEDBACK_FEATURE_DIM_V1
)
SINGLE_SLIT_CONTACT_LAST_HIT_INDEX_V1 = SINGLE_SLIT_FEEDBACK_OBSERVATION_DIM
SINGLE_SLIT_CONTACT_PREVIOUS_FORCE_SLICE_V1 = slice(
    SINGLE_SLIT_CONTACT_LAST_HIT_INDEX_V1 + 1,
    SINGLE_SLIT_CONTACT_LAST_HIT_INDEX_V1 + 3,
)
SINGLE_SLIT_CONTACT_REJECTED_MOTION_SLICE_V1 = slice(
    SINGLE_SLIT_CONTACT_LAST_HIT_INDEX_V1 + 3,
    SINGLE_SLIT_CONTACT_OBSERVATION_DIM_V1,
)
SINGLE_SLIT_CONTACT_MOTION_SCALE_MPS_V1 = 2.0


@dataclass(frozen=True)
class SingleSlitContactObservationContextV1:
    """One tick of local mechanical feedback to append to resolved sensing.

    ``last_hit`` is a per-episode boolean.  The two floating tensors are
    already normalized and have one row per rigid handle.  Keeping the context
    typed prevents an evaluator from accidentally feeding an attempted pose as
    the physical observation itself.
    """

    last_hit: Tensor
    previous_force_normalized: Tensor
    rejected_handle_motion_normalized: Tensor
    schema_version: str = SINGLE_SLIT_CONTACT_OBSERVATION_SCHEMA_V1


def _validate_reference_state(state: Tensor, cfg: EnvConfig) -> None:
    if state.ndim != 2 or state.shape[-1] != 6 or state.shape[0] <= 0:
        raise ValueError("contact-observation state must have shape [batch,6]")
    if not state.dtype.is_floating_point:
        raise ValueError("contact-observation state must be floating point")
    if not bool(torch.isfinite(state).all()):
        raise ValueError("contact-observation state must be finite")
    if (
        not math.isfinite(float(cfg.max_force))
        or float(cfg.max_force) <= 0.0
        or not math.isfinite(float(cfg.dt))
        or float(cfg.dt) <= 0.0
    ):
        raise ValueError("contact observation requires positive finite max_force and dt")


def _validate_context(
    context: SingleSlitContactObservationContextV1,
    state: Tensor,
    cfg: EnvConfig,
) -> None:
    if not isinstance(context, SingleSlitContactObservationContextV1):
        raise TypeError("contact observation requires a V1 context")
    if context.schema_version != SINGLE_SLIT_CONTACT_OBSERVATION_SCHEMA_V1:
        raise ValueError("contact-observation context schema mismatch")
    batch = state.shape[0]
    if context.last_hit.shape != (batch,) or context.last_hit.dtype != torch.bool:
        raise ValueError("last_hit must be a boolean tensor with shape [batch]")
    expected = (batch, cfg.n_agents, 2)
    for label, value in (
        ("previous_force_normalized", context.previous_force_normalized),
        (
            "rejected_handle_motion_normalized",
            context.rejected_handle_motion_normalized,
        ),
    ):
        if value.shape != expected:
            raise ValueError(f"{label} must have shape {expected}")
        if value.dtype != state.dtype or value.device != state.device:
            raise ValueError(f"{label} must share the state dtype/device")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{label} must be finite")
    if context.last_hit.device != state.device:
        raise ValueError("last_hit must share the state device")
    safe = ~context.last_hit
    if bool(
        torch.count_nonzero(context.rejected_handle_motion_normalized[safe]).item()
    ):
        raise ValueError("rejected handle motion must be exactly zero on safe rows")


def initial_single_slit_contact_observation_context_v1(
    state: Tensor,
    cfg: EnvConfig,
) -> SingleSlitContactObservationContextV1:
    """Return the exact all-zero context before the first control proposal."""

    _validate_reference_state(state, cfg)
    batch = state.shape[0]
    local_shape = (batch, cfg.n_agents, 2)
    return SingleSlitContactObservationContextV1(
        last_hit=torch.zeros(batch, dtype=torch.bool, device=state.device),
        previous_force_normalized=torch.zeros(
            local_shape,
            dtype=state.dtype,
            device=state.device,
        ),
        rejected_handle_motion_normalized=torch.zeros(
            local_shape,
            dtype=state.dtype,
            device=state.device,
        ),
    )


def single_slit_contact_observation_context_from_result_v1(
    result: SingleSlitContactFeedbackResultV1,
    applied_forces: Tensor,
    cfg: EnvConfig,
) -> SingleSlitContactObservationContextV1:
    """Encode an executed V1 contact step as next-tick local feedback."""

    if not isinstance(result, SingleSlitContactFeedbackResultV1):
        raise TypeError("contact observation requires a V1 transition result")
    if result.schema_version != SINGLE_SLIT_CONTACT_FEEDBACK_SCHEMA_V1:
        raise ValueError("contact-feedback transition schema mismatch")
    _validate_reference_state(result.resolved_state, cfg)
    batch = result.resolved_state.shape[0]
    expected_state = (batch, 6)
    if result.attempted_state.shape != expected_state:
        raise ValueError("attempted_state must have shape [batch,6]")
    if (
        result.attempted_state.dtype != result.resolved_state.dtype
        or result.attempted_state.device != result.resolved_state.device
        or not bool(torch.isfinite(result.attempted_state).all())
    ):
        raise ValueError("attempted_state must be finite and share resolved dtype/device")
    if (
        result.attempted_hit.shape != (batch,)
        or result.attempted_hit.dtype != torch.bool
    ):
        raise ValueError("attempted_hit must be a boolean tensor with shape [batch]")
    if result.attempted_hit.device != result.resolved_state.device:
        raise ValueError("attempted_hit must share the transition device")
    for label, clearance in (
        ("attempted_clearance_m", result.attempted_clearance_m),
        ("resolved_clearance_m", result.resolved_clearance_m),
    ):
        if (
            clearance.shape != (batch,)
            or clearance.dtype != result.resolved_state.dtype
            or clearance.device != result.resolved_state.device
            or not bool(torch.isfinite(clearance).all())
        ):
            raise ValueError(f"{label} must be finite [batch] with transition dtype/device")
    if not torch.equal(
        result.attempted_hit,
        result.attempted_clearance_m <= CONTACT_FEEDBACK_CLEARANCE_RESERVE_M,
    ):
        raise ValueError("attempted hit and reserved-clearance authority disagree")
    if bool((result.resolved_clearance_m <= 0.0).any()):
        raise ValueError("contact observation requires a penetration-free resolved transition")

    expected_forces = (batch, cfg.n_agents, 2)
    if applied_forces.shape != expected_forces:
        raise ValueError(f"applied_forces must have shape {expected_forces}")
    if (
        applied_forces.dtype != result.resolved_state.dtype
        or applied_forces.device != result.resolved_state.device
        or not bool(torch.isfinite(applied_forces).all())
    ):
        raise ValueError("applied_forces must be finite and share transition dtype/device")

    safe = ~result.attempted_hit
    if bool(safe.any()) and not torch.equal(
        result.attempted_state[safe],
        result.resolved_state[safe],
    ):
        raise ValueError("a safe V1 transition must resolve to its attempted state bitwise")

    handles = contact_points(cfg, result.resolved_state.device).to(
        dtype=result.resolved_state.dtype
    )
    attempted_world = body_to_world(handles, result.attempted_state)
    resolved_world = body_to_world(handles, result.resolved_state)
    scale = float(cfg.dt) * SINGLE_SLIT_CONTACT_MOTION_SCALE_MPS_V1
    rejected_motion = (attempted_world - resolved_world) / scale
    rejected_motion = torch.where(
        result.attempted_hit[:, None, None],
        rejected_motion,
        torch.zeros_like(rejected_motion),
    )
    context = SingleSlitContactObservationContextV1(
        last_hit=result.attempted_hit.clone(),
        previous_force_normalized=applied_forces / float(cfg.max_force),
        rejected_handle_motion_normalized=rejected_motion,
    )
    _validate_context(context, result.resolved_state, cfg)
    return context


def single_slit_contact_observations_v1(
    resolved_state: Tensor,
    scenario: PianoBatch,
    cfg: EnvConfig,
    context: SingleSlitContactObservationContextV1,
) -> Tensor:
    """Return exact-clearance22 plus five versioned contact-feedback channels."""

    _validate_reference_state(resolved_state, cfg)
    _validate_context(context, resolved_state, cfg)
    base = single_slit_feedback_observations(resolved_state, scenario, cfg)
    if base.shape != (resolved_state.shape[0], cfg.n_agents, 22):
        raise RuntimeError("single-slit exact-clearance observation width changed")
    hit = context.last_hit.to(dtype=resolved_state.dtype).view(-1, 1, 1)
    hit = hit.expand(-1, cfg.n_agents, -1)
    feedback = torch.cat(
        (
            hit,
            context.previous_force_normalized,
            context.rejected_handle_motion_normalized,
        ),
        dim=-1,
    )
    result = torch.cat((base, feedback), dim=-1)
    if result.shape != (
        resolved_state.shape[0],
        cfg.n_agents,
        SINGLE_SLIT_CONTACT_OBSERVATION_DIM_V1,
    ):
        raise RuntimeError("contact observation width changed")
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("contact observation contains non-finite values")
    return result


__all__ = [
    "SINGLE_SLIT_CONTACT_FEEDBACK_FEATURE_DIM_V1",
    "SINGLE_SLIT_CONTACT_LAST_HIT_INDEX_V1",
    "SINGLE_SLIT_CONTACT_MOTION_SCALE_MPS_V1",
    "SINGLE_SLIT_CONTACT_OBSERVATION_DIM_V1",
    "SINGLE_SLIT_CONTACT_OBSERVATION_SCHEMA_V1",
    "SINGLE_SLIT_CONTACT_PREVIOUS_FORCE_SLICE_V1",
    "SINGLE_SLIT_CONTACT_REJECTED_MOTION_SLICE_V1",
    "SingleSlitContactObservationContextV1",
    "initial_single_slit_contact_observation_context_v1",
    "single_slit_contact_observation_context_from_result_v1",
    "single_slit_contact_observations_v1",
]
