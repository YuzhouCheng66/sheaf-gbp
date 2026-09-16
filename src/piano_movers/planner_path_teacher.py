"""Track an ordered SE(2) planner path with a feasible force teacher.

The path is ordered by the planner, not by any world coordinate.  In
particular, a later waypoint may have a smaller x coordinate than the current
waypoint.  The teacher first projects the measured payload pose onto that
ordered path, advances by an SE(2) lookahead distance, and applies pose/twist
PD control toward the resulting reference.  A caller can feed the returned
``nearest_index`` back as ``minimum_path_index`` on the next tick to make path
progress fail-closed at self-intersections.

This is a centralized, privileged training oracle.  It is not a deployable
multi-agent policy and must not be counted as part of the learned policy's
communication budget.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .cem_teacher import feasible_forces_for_absolute_wrench
from .config import HANDLE_LAYOUT_BOUNDARY_ATTACHED_V2, EnvConfig
from .env import wrap_angle


@dataclass(frozen=True)
class PlannerPathTeacherConfig:
    """Controls for projection, ordered lookahead, and rigid-body PD."""

    lookahead_distance: float = 0.20
    orientation_distance_scale: float = 0.35
    maximum_linear_speed: float = 0.65
    maximum_angular_speed: float = 1.40
    slowdown_distance: float = 0.30
    stop_distance: float = 0.010
    linear_position_gain: float = 4.0
    linear_velocity_gain: float = 3.0
    angle_position_gain: float = 8.0
    angular_velocity_gain: float = 4.0

    def validate(self) -> None:
        positive = (
            self.lookahead_distance,
            self.orientation_distance_scale,
            self.maximum_linear_speed,
            self.maximum_angular_speed,
            self.slowdown_distance,
            self.linear_position_gain,
            self.linear_velocity_gain,
            self.angle_position_gain,
            self.angular_velocity_gain,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("planner-path teacher gains and scales must be positive")
        if self.stop_distance < 0.0:
            raise ValueError("stop_distance must be non-negative")
        if self.stop_distance >= self.slowdown_distance:
            raise ValueError("stop_distance must be smaller than slowdown_distance")


@dataclass(frozen=True)
class PlannerPathTeacherDiagnostics:
    """Per-row references and grasp-allocation evidence."""

    nearest_index: Tensor
    reference_index: Tensor
    reference_pose: Tensor
    desired_twist: Tensor
    requested_wrench: Tensor
    realized_wrench: Tensor
    allocator_scale: Tensor
    remaining_path_distance: Tensor


@dataclass(frozen=True)
class PlannerPathTeacherOutput:
    """Executable per-handle forces plus auditable teacher diagnostics."""

    forces: Tensor
    wrench: Tensor
    diagnostics: PlannerPathTeacherDiagnostics


def _normalize_and_validate_inputs(
    state: Tensor,
    path: Tensor,
    cfg: EnvConfig,
    controls: PlannerPathTeacherConfig,
    path_lengths: Tensor | None,
    minimum_path_index: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    controls.validate()
    if cfg.n_agents != 12:
        raise ValueError("planner-path teacher requires exactly 12 force handles")
    if cfg.handle_layout != HANDLE_LAYOUT_BOUNDARY_ATTACHED_V2:
        raise ValueError("planner-path teacher requires boundary-attached-v2 handles")
    if state.ndim != 2 or state.shape[-1] != 6:
        raise ValueError("state must have shape [batch, 6]")
    if not state.is_floating_point() or not bool(torch.isfinite(state).all()):
        raise ValueError("state must be finite floating-point values")

    # A two-dimensional path is a convenience API for the batch-one planner
    # scout.  The normalized representation is always [B, N, 3].
    if path.ndim == 2:
        if state.shape[0] != 1:
            raise ValueError("a [waypoints, 3] path is only valid for batch size one")
        path = path.unsqueeze(0)
    if path.ndim != 3 or path.shape[0] != state.shape[0] or path.shape[-1] != 3:
        raise ValueError("path must have shape [batch, waypoints, 3]")
    if path.shape[1] < 1:
        raise ValueError("every planner path must contain at least one waypoint")
    if path.device != state.device or path.dtype != state.dtype:
        raise ValueError("state and path must share device and dtype")
    if not bool(torch.isfinite(path).all()):
        raise ValueError("path must contain only finite values")

    batch, waypoint_count = path.shape[:2]
    if path_lengths is None:
        path_lengths = torch.full(
            (batch,),
            waypoint_count,
            dtype=torch.long,
            device=state.device,
        )
    if (
        path_lengths.shape != (batch,)
        or path_lengths.device != state.device
        or path_lengths.dtype != torch.long
    ):
        raise ValueError("path_lengths must be int64 [batch] on the state device")
    if bool(((path_lengths < 1) | (path_lengths > waypoint_count)).any()):
        raise ValueError("path_lengths entries must lie in [1, waypoints]")

    if minimum_path_index is None:
        minimum_path_index = torch.zeros(batch, dtype=torch.long, device=state.device)
    if (
        minimum_path_index.shape != (batch,)
        or minimum_path_index.device != state.device
        or minimum_path_index.dtype != torch.long
    ):
        raise ValueError("minimum_path_index must be int64 [batch] on the state device")
    if bool(((minimum_path_index < 0) | (minimum_path_index >= path_lengths)).any()):
        raise ValueError("minimum_path_index must identify a valid waypoint in each row")
    return path, path_lengths, minimum_path_index


def _ordered_path_references(
    state: Tensor,
    path: Tensor,
    path_lengths: Tensor,
    minimum_path_index: Tensor,
    controls: PlannerPathTeacherConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Project onto path order and return lookahead pose/twist information."""

    batch, waypoint_count = path.shape[:2]
    waypoint_axis = torch.arange(waypoint_count, device=state.device).view(1, -1)
    valid = waypoint_axis < path_lengths[:, None]
    forward_search = valid & (waypoint_axis >= minimum_path_index[:, None])

    pose_delta_xy = path[..., :2] - state[:, None, :2]
    pose_delta_theta = wrap_angle(path[..., 2] - state[:, None, 2])
    projection_distance2 = pose_delta_xy.square().sum(dim=-1) + (
        controls.orientation_distance_scale * pose_delta_theta
    ).square()
    projection_distance2 = projection_distance2.masked_fill(~forward_search, torch.inf)
    nearest_index = projection_distance2.argmin(dim=-1)

    if waypoint_count == 1:
        cumulative = state.new_zeros((batch, 1))
    else:
        segment_xy = path[:, 1:, :2] - path[:, :-1, :2]
        segment_theta = wrap_angle(path[:, 1:, 2] - path[:, :-1, 2])
        segment_distance = torch.sqrt(
            segment_xy.square().sum(dim=-1)
            + (controls.orientation_distance_scale * segment_theta).square()
        )
        # Padded segments are irrelevant and must contribute exactly zero to
        # the per-row cumulative arclength.
        segment_valid = waypoint_axis[:, 1:] < path_lengths[:, None]
        segment_distance = torch.where(
            segment_valid,
            segment_distance,
            torch.zeros_like(segment_distance),
        )
        cumulative = torch.cat(
            (state.new_zeros((batch, 1)), segment_distance.cumsum(dim=-1)),
            dim=-1,
        )

    nearest_s = cumulative.gather(1, nearest_index[:, None]).squeeze(1)
    lookahead_s = nearest_s + controls.lookahead_distance
    lookahead_candidates = (
        valid
        & (waypoint_axis >= nearest_index[:, None])
        & (cumulative >= lookahead_s[:, None])
    )
    has_candidate = lookahead_candidates.any(dim=-1)
    first_candidate = lookahead_candidates.to(torch.int64).argmax(dim=-1)
    last_index = path_lengths - 1
    reference_index = torch.where(has_candidate, first_candidate, last_index)

    batch_axis = torch.arange(batch, device=state.device)
    nearest_pose = path[batch_axis, nearest_index]
    reference_pose = path[batch_axis, reference_index]
    last_s = cumulative.gather(1, last_index[:, None]).squeeze(1)
    remaining_distance = (last_s - nearest_s).clamp_min(0.0)

    # The feed-forward twist follows planner order.  It is computed from the
    # path projection rather than from measured pose, leaving cross-track
    # recovery to the proportional term below.
    guide_xy = reference_pose[:, :2] - nearest_pose[:, :2]
    guide_theta = wrap_angle(reference_pose[:, 2] - nearest_pose[:, 2])
    linear_time = torch.linalg.vector_norm(guide_xy, dim=-1) / controls.maximum_linear_speed
    angular_time = guide_theta.abs() / controls.maximum_angular_speed
    travel_time = torch.maximum(linear_time, angular_time)
    nonzero_guide = travel_time > 1e-8
    safe_time = travel_time.clamp_min(1e-8)
    desired_twist = torch.cat(
        (guide_xy / safe_time[:, None], (guide_theta / safe_time).unsqueeze(-1)),
        dim=-1,
    )
    desired_twist = torch.where(
        nonzero_guide[:, None],
        desired_twist,
        torch.zeros_like(desired_twist),
    )
    slowdown = (remaining_distance / controls.slowdown_distance).clamp(0.0, 1.0)
    stopped = remaining_distance <= controls.stop_distance
    slowdown = torch.where(stopped, torch.zeros_like(slowdown), slowdown)
    desired_twist = desired_twist * slowdown[:, None]
    return (
        nearest_index,
        reference_index,
        reference_pose,
        desired_twist,
        remaining_distance,
    )


@torch.no_grad()
def planner_path_teacher_output(
    state: Tensor,
    path: Tensor,
    cfg: EnvConfig,
    teacher_cfg: PlannerPathTeacherConfig | None = None,
    *,
    path_lengths: Tensor | None = None,
    minimum_path_index: Tensor | None = None,
) -> PlannerPathTeacherOutput:
    """Return feasible 12-handle forces that track an ordered SE(2) path.

    ``path`` may be ``[N, 3]`` for a single state or padded ``[B, N, 3]``.
    For padded batches, ``path_lengths`` declares the valid prefix of each
    row.  ``minimum_path_index`` is an optional nondecreasing progress hint;
    it is particularly important for paths that revisit the same pose.
    """

    controls = PlannerPathTeacherConfig() if teacher_cfg is None else teacher_cfg
    path, path_lengths, minimum_path_index = _normalize_and_validate_inputs(
        state,
        path,
        cfg,
        controls,
        path_lengths,
        minimum_path_index,
    )
    (
        nearest_index,
        reference_index,
        reference_pose,
        desired_twist,
        remaining_distance,
    ) = _ordered_path_references(
        state,
        path,
        path_lengths,
        minimum_path_index,
        controls,
    )

    position_error = reference_pose[:, :2] - state[:, :2]
    angle_error = wrap_angle(reference_pose[:, 2] - state[:, 2])
    requested_wrench = torch.cat(
        (
            cfg.mass
            * (
                controls.linear_position_gain * position_error
                + controls.linear_velocity_gain * (desired_twist[:, :2] - state[:, 3:5])
            ),
            (
                cfg.inertia
                * (
                    controls.angle_position_gain * angle_error
                    + controls.angular_velocity_gain * (desired_twist[:, 2] - state[:, 5])
                )
            ).unsqueeze(-1),
        ),
        dim=-1,
    )
    forces, realized_wrench, allocator_scale = feasible_forces_for_absolute_wrench(
        state,
        requested_wrench,
        cfg,
    )
    diagnostics = PlannerPathTeacherDiagnostics(
        nearest_index=nearest_index,
        reference_index=reference_index,
        reference_pose=reference_pose,
        desired_twist=desired_twist,
        requested_wrench=requested_wrench,
        realized_wrench=realized_wrench,
        allocator_scale=allocator_scale,
        remaining_path_distance=remaining_distance,
    )
    return PlannerPathTeacherOutput(
        forces=forces,
        wrench=realized_wrench,
        diagnostics=diagnostics,
    )
