"""Multi-wall, hand-specified waypoint demo for the frozen Sheaf-GBP skill.

This module does not claim mapping or global planning.  A supervisor supplies
an ordered SE(2) route.  The frozen p5 policy receives only the current target
and the current locally relevant slit, while exact collision feedback is
evaluated against every wall in the course.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
import json
import math
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Iterator
import webbrowser

import torch
from torch import Tensor

from .config import (
    HANDLE_LAYOUT_GLYPH_BOUNDARY_V4,
    HANDLE_LAYOUT_PARAMETRIC_T_V3,
    EnvConfig,
)
from .env import PianoBatch, body_to_world, contact_points, radius_edges
from .gbp import GBPMessageTrace, homogeneous_topology, coordination_tree_diameter
from .models import SheafGBPPolicy
from .payload_shapes import (
    LETTER_PAYLOAD_SHAPES,
    payload_rectangles_for_config,
    payload_shape_rectangles,
)
from .single_slit_0720_eval import (
    STRICT_STABLE_STEPS,
    load_single_slit_checkpoint,
    strict_final_mask,
)
from .single_slit_contact_feedback import single_slit_contact_feedback_step_v1
from .local_skill_physics import certified_contact_step
from .single_slit_contact_observation import (
    initial_single_slit_contact_observation_context_v1,
    single_slit_contact_observation_context_from_result_v1,
    single_slit_contact_observations_v1,
)
from .single_slit_local_planner import LOCAL_SE2_TERMINAL_WALL_RESERVE_M


ROLLOUT_LOCK = Lock()
DUMMY_WALL_X_M = 8.0
DEFAULT_N_AGENTS = 12
MIN_N_AGENTS = 8
MAX_N_AGENTS = 20
HORIZON = 1_100
INTERMEDIATE_POSITION_TOLERANCE_M = 0.14
LARGE_GLYPH_PASSAGE_SPAN_M = 0.648


@dataclass(frozen=True)
class SkillCompositionOptions:
    """Independent development controls; none are learned-policy inputs."""

    position_only_control: bool = True
    require_body_clearance: bool = True
    preserve_private_memory: bool = False
    nominal_glyph_size: bool = False
    retain_nearby_wall: bool = False
    preserve_contact_feedback: bool = False
    retain_previous_wall_only: bool = False
    complete_corridor_exit: bool = False
    planning_substeps: int = 4
    continuous_contact_guard: bool = False
    validate_all_cspace_lookaheads: bool = False
    check_support_alignment_sweep: bool = False
    hold_feasible_crossing: bool = False
    respect_handoff_goal_region: bool = False
    cover_coordination_tree_diameter: bool = False
    geometry_precision_boost_override: float | None = None
    resolve_position_goal_heading: bool = False


@dataclass(frozen=True)
class Slit:
    x: float
    gap_y: float
    aperture_m: float = 0.720

    @property
    def gap_half(self) -> float:
        return 0.5 * self.aperture_m


@dataclass(frozen=True)
class WaypointCourse:
    source: tuple[float, float, float]
    walls: tuple[Slit, ...]
    waypoints: tuple[tuple[float, float, float], ...]
    world_x: float = 3.9
    world_y: float = 1.6
    source_velocity: tuple[float, float, float] = (0., 0., 0.)
    morphology_scale: float = 1.0

    def validate(self) -> None:
        if len(self.source) != 3 or len(self.source_velocity) != 3:
            raise ValueError("source requires pose3 and world-frame twist3")
        if not math.isfinite(self.morphology_scale) or self.morphology_scale <= 0:
            raise ValueError("morphology scale must be finite and positive")
        if not self.walls or len(self.waypoints) != len(self.walls) + 1:
            raise ValueError("a course requires one post-wall waypoint per wall and one final goal")
        if not all(self.walls[index].x < self.walls[index + 1].x for index in range(len(self.walls) - 1)):
            raise ValueError("walls must be strictly ordered from left to right")
        if not self.source[0] < self.walls[0].x:
            raise ValueError("source must precede the first wall")
        for index, wall in enumerate(self.walls):
            if not wall.x < self.waypoints[index][0]:
                raise ValueError("each wall waypoint must lie beyond its wall")
            if not math.isfinite(wall.aperture_m) or wall.aperture_m < 0.720:
                raise ValueError("every slit must retain the audited 0.720 m aperture")
        if not self.waypoints[-1][0] > self.walls[-1].x:
            raise ValueError("final goal must lie beyond the final wall")
        numbers = (*self.source, *self.source_velocity, self.world_x, self.world_y,
                   *(value for wall in self.walls for value in (wall.x, wall.gap_y, wall.aperture_m)),
                   *(value for waypoint in self.waypoints for value in waypoint))
        if not all(math.isfinite(value) for value in numbers):
            raise ValueError("course geometry must be finite")


DEFAULT_COURSE = WaypointCourse(
    source=(-3.20, 0.00, 0.00),
    walls=(
        Slit(-1.80, -0.15),
        Slit(0.00, 0.17),
        Slit(1.80, -0.11),
    ),
    waypoints=(
        (-1.17, -0.15, 0.00),
        (0.63, 0.17, 0.00),
        (2.43, -0.11, 0.00),
        (3.24, 0.00, 0.00),
    ),
)


def _tensor_list(value: Tensor) -> list:
    return value.detach().cpu().tolist()


def _traceable_policy(policy) -> SheafGBPPolicy:
    target = policy if isinstance(policy, SheafGBPPolicy) else getattr(policy, "policy", None)
    if not isinstance(target, SheafGBPPolicy):
        raise TypeError("waypoint demo requires a traceable SheafGBPPolicy")
    return target


def _large_glyph_scale(payload_shape: str) -> float:
    """Scale one letter to a tight orientation the online planner can represent.

    The planner searches 72 passage headings.  We evaluate those same
    headings on the exact union vertices and enlarge the glyph until its best
    represented projection is 0.648 m.  The exact wall model leaves 0.660 m
    of physical opening, so the witness retains 6 mm per side.
    """

    shape = payload_shape.strip().upper()
    if shape == "T":
        return 1.0
    rectangles = payload_shape_rectangles(shape, "cpu").to(torch.float64)
    vertices = rectangles.reshape(-1, 2)
    angles = torch.arange(72, dtype=torch.float64) * (2.0 * torch.pi / 72.0)
    sine = torch.sin(angles)
    cosine = torch.cos(angles)
    projected_y = (
        sine[:, None] * vertices[None, :, 0]
        + cosine[:, None] * vertices[None, :, 1]
    )
    minimum_span = (projected_y.amax(dim=-1) - projected_y.amin(dim=-1)).amin()
    return max(1.0, LARGE_GLYPH_PASSAGE_SPAN_M / float(minimum_span))


def _bind_geometry(policy, cfg: EnvConfig) -> None:
    target = _traceable_policy(policy)
    target.bind_environment(cfg)


def apply_solver_controls(policy, options: SkillCompositionOptions) -> dict[str, Any]:
    """Explicit experimental solver controls, with no checkpoint tensor edits."""
    target = _traceable_policy(policy)
    if options.geometry_precision_boost_override is not None:
        value = options.geometry_precision_boost_override
        if not math.isfinite(value) or value < 1.:
            raise ValueError("geometry precision boost override must be finite and >=1")
        target.arm_cfg = replace(target.arm_cfg, local_se2_geometry_evidence_precision_boost=value)
    diameter = coordination_tree_diameter(target.local_se2_analytic_reference_edges,
                                          target.local_se2_analytic_tree_edge_mask,
                                          target.env_cfg.n_agents)
    steps = target.arm_cfg.gbp_steps
    if options.cover_coordination_tree_diameter:
        steps = max(steps, diameter)
    target.local_se2_analytic_gbp.num_steps = steps
    return {"rounds": steps, "tree_diameter": diameter,
            "geometry_precision_boost": target.arm_cfg.local_se2_geometry_evidence_precision_boost,
            "coordinate_channels_per_message": target.arm_cfg.correction_edge_dim,
            "wire_floats_per_directed_message": target.communication_floats_per_directed_edge_round()}


def _set_active_goal_without_rebinding(policy, cfg: EnvConfig) -> None:
    """Update only mission fields, preserving graph and grasp setup caches."""

    target = _traceable_policy(policy)
    old = target.env_cfg
    geometry_fields = (
        "n_agents",
        "handle_layout",
        "payload_shape",
        "glyph_scale",
        "t_total_width_m",
        "t_total_height_m",
        "t_stem_width_m",
        "t_cap_thickness_m",
        "comm_radius",
    )
    if any(getattr(old, name) != getattr(cfg, name) for name in geometry_fields):
        raise ValueError("mission-only goal update attempted to change policy geometry")
    target.env_cfg = cfg


def _course_config(
    base: EnvConfig,
    n_agents: int,
    course: WaypointCourse,
    payload_shape: str = "T",
) -> EnvConfig:
    shape = payload_shape.strip().upper()
    if shape not in LETTER_PAYLOAD_SHAPES:
        raise ValueError("payload_shape must be one uppercase A-Z letter")
    first = course.waypoints[0]
    first_wall = course.walls[0]
    return replace(
        base,
        n_agents=n_agents,
        handle_layout=(
            HANDLE_LAYOUT_PARAMETRIC_T_V3
            if shape == "T"
            else HANDLE_LAYOUT_GLYPH_BOUNDARY_V4
        ),
        payload_shape=shape,
        glyph_scale=_large_glyph_scale(shape) * course.morphology_scale,
        t_total_width_m=0.930 * course.morphology_scale,
        t_total_height_m=0.920 * course.morphology_scale,
        t_stem_width_m=0.160 * course.morphology_scale,
        t_cap_thickness_m=0.152 * course.morphology_scale,
        source_x=course.source[0],
        # The frozen skill was trained with its locally relevant wall at
        # x=0.  Each waypoint leg is expressed in that same local frame; the
        # exact simulator below remains in the shared world frame.
        goal_x=first[0] - first_wall.x,
        goal_y=first[1],
        goal_theta=first[2],
        world_x=course.world_x,
        world_y=course.world_y,
        max_steps=HORIZON,
    )


def _full_collision_scenario(course: WaypointCourse, cfg: EnvConfig) -> PianoBatch:
    device = torch.device(cfg.device)
    state0 = torch.zeros(1, 6, dtype=torch.float32, device=device)
    state0[0, :3] = state0.new_tensor(course.source)
    state0[0, 3:] = state0.new_tensor(course.source_velocity)
    source = state0[:, :3].clone()
    goal = state0.new_tensor(course.waypoints[-1]).view(1, 3)
    return PianoBatch(
        state0=state0,
        goal=goal,
        wall_x=state0.new_tensor([[wall.x for wall in course.walls]]),
        gap_y=state0.new_tensor([[wall.gap_y for wall in course.walls]]),
        gap_half=state0.new_tensor([[wall.gap_half for wall in course.walls]]),
        source=source,
    )


def _active_observation_scenario(
    course: WaypointCourse,
    state: Tensor,
    waypoint_index: int,
) -> tuple[Tensor, PianoBatch]:
    wall_index = min(waypoint_index, len(course.walls) - 1)
    wall = course.walls[wall_index]
    local_state = state.clone()
    local_state[:, 0] -= wall.x
    local_goal = list(course.waypoints[waypoint_index])
    local_goal[0] -= wall.x
    local_source = list(course.source)
    local_source[0] -= wall.x
    goal = state.new_tensor(local_goal).view(1, 3)
    source = state.new_tensor(local_source).view(1, 3)
    scenario = PianoBatch(
        state0=local_state.clone(),
        goal=goal,
        wall_x=state.new_tensor([[0.0, DUMMY_WALL_X_M]]),
        gap_y=state.new_tensor([[wall.gap_y, 0.0]]),
        gap_half=state.new_tensor([[wall.gap_half, course.world_y]]),
        source=source,
    )
    return local_state, scenario


def _intermediate_reached(state: Tensor, waypoint: tuple[float, float, float]) -> bool:
    target = state.new_tensor(waypoint)
    position_error = torch.linalg.vector_norm(state[0, :2] - target[:2])
    # Intermediate route inputs are 2-D points, not poses.  A payload may
    # leave a narrow slit in any safe passage heading, and the next local
    # skill invocation replans from that observed heading.  Only the final
    # target below is checked as a stationary SE(2) pose.
    return bool(position_error <= INTERMEDIATE_POSITION_TOLERANCE_M)


def _composition_observation(
    course, state, waypoint_index, cfg, context, *, retain_nearby_wall=False,
    retain_previous_wall_only=False,
):
    """Keep the original 27-D interface; optionally fill its unused wall slot.

    Wall zero retains the active-wall chart expected by the frozen policy.
    Each handle chooses its nearest other wall using the existing horizontal
    visibility contract. The normal sensor then masks out-of-range records.
    This is not an all-obstacle sensor: if both non-active walls are visible,
    the two-slot architecture can expose only the nearer secondary wall.
    No future waypoint is supplied to the policy.
    """
    local_state, scene = _active_observation_scenario(course, state, waypoint_index)
    if not (retain_nearby_wall or retain_previous_wall_only) or len(course.walls) == 1:
        return single_slit_contact_observations_v1(local_state, scene, cfg, context)
    active_index = min(waypoint_index, len(course.walls) - 1)
    active = course.walls[active_index]
    others = [
        wall for index, wall in enumerate(course.walls)
        if index != active_index and (not retain_previous_wall_only or index < active_index)
    ]
    if not others:
        return single_slit_contact_observations_v1(local_state, scene, cfg, context)
    candidates = []
    for wall in others:
        alternative = replace(
            scene,
            wall_x=state.new_tensor([[0., wall.x - active.x]]),
            gap_y=state.new_tensor([[active.gap_y, wall.gap_y]]),
            gap_half=state.new_tensor([[active.gap_half, wall.gap_half]]),
        )
        candidates.append(single_slit_contact_observations_v1(local_state, alternative, cfg, context))
    handles = body_to_world(contact_points(cfg, state.device).to(state), state)
    distance = (handles[..., 0, None] - state.new_tensor([wall.x for wall in others])).abs()
    nearest = distance.argmin(-1)
    rows = torch.stack(candidates, -2)
    return rows.gather(-2, nearest[..., None, None].expand(*nearest.shape, 1, rows.shape[-1])).squeeze(-2)


def _body_cleared_wall(state: Tensor, rectangles: Tensor, wall: Slit, cfg: EnvConfig) -> bool:
    """A centre crossing alone must not discard a still-intersecting wall.

    This is a supervisor handoff check, not an observation or policy feature.
    The default course travels rightward. Its exact compound-body support must
    clear the right face by the same reserve used by the local terminal skill.
    """

    vertices = rectangles.reshape(-1, 2)
    theta = state[0, 2]
    leftmost = state[0, 0] + (
        theta.cos() * vertices[:, 0] - theta.sin() * vertices[:, 1]
    ).amin()
    return bool(leftmost > wall.x + cfg.wall_slab + LOCAL_SE2_TERMINAL_WALL_RESERVE_M)


class MultiWaypointEngine:
    def __init__(
        self,
        checkpoint: Path,
        device: str,
        course: WaypointCourse = DEFAULT_COURSE,
        *,
        options: SkillCompositionOptions = SkillCompositionOptions(),
        audit_callback: Callable[..., None] | None = None,
        horizon: int = HORIZON,
    ) -> None:
        course.validate()
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("control-step horizon must be a positive integer")
        self.device = torch.device(device)
        self.loaded = load_single_slit_checkpoint(checkpoint, device=self.device)
        self.loaded.policy.eval()
        self.course = course
        self.options = options
        self.audit_callback = audit_callback
        self.horizon = horizon
        self.cache: OrderedDict[
            tuple[str, int], tuple[dict[str, Any], ...]
        ] = OrderedDict()

    def _contact_step(self, state, forces, scenario, cfg):
        """Physics seam for independent frame audits; defaults are unchanged."""
        if self.options.continuous_contact_guard:
            return certified_contact_step(state, forces, scenario, cfg)
        return single_slit_contact_feedback_step_v1(state, forces, scenario, cfg), None

    def _policy_step(self, target, observation, edges, hidden, topology):
        """Unchanged mainline call; independent controls may override this seam."""
        return target(observation, edges, hidden, topology=topology, trace_messages=True)

    @torch.inference_mode()
    def events(
        self,
        n_agents: int = DEFAULT_N_AGENTS,
        payload_shape: str = "T",
    ) -> Iterator[dict[str, Any]]:
        if not MIN_N_AGENTS <= n_agents <= MAX_N_AGENTS:
            raise ValueError(f"n_agents must lie in [{MIN_N_AGENTS},{MAX_N_AGENTS}]")
        shape = payload_shape.strip().upper()
        if shape not in LETTER_PAYLOAD_SHAPES:
            raise ValueError("payload_shape must be one uppercase A-Z letter")
        cache_key = (shape, n_agents)
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.cache.move_to_end(cache_key)
            yield from cached
            return

        recorded: list[dict[str, Any]] = []
        cfg = _course_config(
            self.loaded.cfg,
            n_agents,
            self.course,
            payload_shape=shape,
        )
        if self.options.nominal_glyph_size:
            cfg = replace(cfg, glyph_scale=1.0)
        if self.options.complete_corridor_exit:
            cfg = replace(cfg, complete_local_corridor_exit=True)
        cfg = replace(
            cfg, local_cspace_collision_substeps=self.options.planning_substeps,
            max_steps=self.horizon,
            local_corridor_goal_heading_search=self.options.resolve_position_goal_heading,
            local_cspace_validate_all_lookaheads=self.options.validate_all_cspace_lookaheads,
            local_support_check_alignment_sweep=self.options.check_support_alignment_sweep,
            local_support_hold_feasible_crossing=self.options.hold_feasible_crossing,
            local_corridor_exit_goal_radius_m=(INTERMEDIATE_POSITION_TOLERANCE_M
                                               if self.options.respect_handoff_goal_region else 0.),
        )
        _bind_geometry(self.loaded.policy, cfg)
        solver_controls = apply_solver_controls(self.loaded.policy, self.options)
        full_scenario = _full_collision_scenario(self.course, cfg)
        state = full_scenario.state0.clone()
        hidden = self.loaded.policy.init_hidden(1, self.device)
        local_state, _ = _active_observation_scenario(
            self.course, state, waypoint_index=0
        )
        context = initial_single_slit_contact_observation_context_v1(local_state, cfg)
        edges = radius_edges(cfg, device=self.device)
        topology = homogeneous_topology(edges, 1, cfg.n_agents, device=self.device)
        handles = contact_points(cfg, self.device)
        body_rects = payload_rectangles_for_config(cfg, self.device)
        expected_sender = edges.reshape(-1)
        expected_receiver = edges.flip(-1).reshape(-1)

        header = {
            "type": "header",
            "schema": "piano-movers.multi-wall-waypoint-demo.v1",
            "composition_options": asdict(self.options),
            "solver_controls": solver_controls,
            "checkpoint_sha256": self.loaded.checkpoint_sha256,
            "payload_shape": shape,
            "glyph_scale": cfg.glyph_scale,
            "n_agents": n_agents,
            "dt": cfg.dt,
            "world": {"x": self.course.world_x, "y": self.course.world_y, "wall_half_width": cfg.wall_slab},
            "source": list(self.course.source),
            "source_velocity": list(self.course.source_velocity),
            "morphology_scale": self.course.morphology_scale,
            "walls": [wall.__dict__ for wall in self.course.walls],
            "waypoints": [list(value) for value in self.course.waypoints],
            "body_rects": _tensor_list(body_rects),
            "handles": _tensor_list(handles),
            "sender": _tensor_list(expected_sender),
            "receiver": _tensor_list(expected_receiver),
        }
        recorded.append(header)
        yield header

        waypoint_index = 0
        dwell = 0
        contacts = 0
        continuous_rejections = 0
        for tick in range(self.horizon):
            active_goal = self.course.waypoints[waypoint_index]
            active_wall = self.course.walls[min(waypoint_index, len(self.course.walls) - 1)]
            cfg = replace(
                cfg,
                goal_x=active_goal[0] - active_wall.x,
                goal_y=active_goal[1],
                goal_theta=active_goal[2],
                goal_orientation_required=(
                    waypoint_index == len(self.course.waypoints) - 1
                    or not self.options.position_only_control
                ),
            )
            _set_active_goal_without_rebinding(self.loaded.policy, cfg)
            observation = _composition_observation(
                self.course, state, waypoint_index, cfg, context,
                retain_nearby_wall=self.options.retain_nearby_wall,
                retain_previous_wall_only=self.options.retain_previous_wall_only,
            )
            target = _traceable_policy(self.loaded.policy)
            output = self._policy_step(target, observation, edges, hidden, topology)
            transition, certificate = self._contact_step(state, output.forces, full_scenario, cfg)
            continuous_rejections += int(certificate is not None and not certificate.passed)
            if self.audit_callback is not None:
                self.audit_callback(
                    tick=tick, waypoint_index=waypoint_index, cfg=cfg,
                    state=state, observation=observation, hidden=hidden,
                    output=output, transition=transition, scenario=full_scenario,
                )
            trace = output.aux.get("gbp_message_trace")
            if not isinstance(trace, GBPMessageTrace):
                raise RuntimeError("policy omitted the physical GBP trace")
            if not torch.equal(trace.sender, expected_sender) or not torch.equal(
                trace.receiver, expected_receiver
            ):
                raise RuntimeError("GBP trace changed from the physical radius graph")

            hit = bool(transition.attempted_hit[0])
            contacts += int(hit)
            next_state = transition.resolved_state
            reached = False
            if waypoint_index < len(self.course.waypoints) - 1:
                reached = _intermediate_reached(next_state, active_goal)
                if self.options.require_body_clearance:
                    reached = reached and _body_cleared_wall(
                        next_state, body_rects, active_wall, cfg
                    )
                if reached:
                    waypoint_index += 1
                    dwell = 0
            else:
                final_goal = next_state.new_tensor(self.course.waypoints[-1]).view(1, 3)
                dwell = (
                    dwell + 1
                    if not hit and bool(strict_final_mask(next_state, final_goal)[0])
                    else 0
                )
            success = waypoint_index == len(self.course.waypoints) - 1 and dwell >= STRICT_STABLE_STEPS
            frame = {
                "type": "frame",
                "step": tick,
                "state": _tensor_list(state[0]),
                "next_state": _tensor_list(next_state[0]),
                "waypoint_index": waypoint_index,
                "waypoint_reached": reached,
                "hit": hit,
                "clearance_m": float(transition.attempted_clearance_m[0].item()),
                "intensity": _tensor_list(trace.intensity[0]),
                "success": success,
            }
            recorded.append(frame)
            yield frame
            context = single_slit_contact_observation_context_from_result_v1(
                transition, output.forces, cfg
            )
            state = next_state
            hidden = output.hidden
            if reached:
                # A supplied waypoint starts a fresh invocation of the same
                # local manipulation skill.  Reset only private temporal
                # memory; learned weights, graph, grasp statistics, and all
                # physical communication semantics remain unchanged.
                if not self.options.preserve_private_memory:
                    hidden = self.loaded.policy.init_hidden(1, self.device)
                next_local_state, _ = _active_observation_scenario(
                    self.course, state, waypoint_index
                )
                if not self.options.preserve_contact_feedback:
                    context = initial_single_slit_contact_observation_context_v1(
                        next_local_state, cfg
                    )
            if success:
                break

        done = {
            "type": "done",
            "success": waypoint_index == len(self.course.waypoints) - 1 and dwell >= STRICT_STABLE_STEPS,
            "steps": tick + 1,
            "waypoints_completed": waypoint_index + int(dwell >= STRICT_STABLE_STEPS),
            "waypoint_count": len(self.course.waypoints),
            "contact_events": contacts,
            "continuous_guard_rejections": continuous_rejections,
            "exact_penetration": False,
        }
        recorded.append(done)
        self.cache[cache_key] = tuple(recorded)
        self.cache.move_to_end(cache_key)
        while len(self.cache) > 4:
            self.cache.popitem(last=False)
        yield done


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Sheaf-GBP · Waypoints</title>
  <style>
    *{box-sizing:border-box}html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#dedede}
    body{font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
    main,canvas{position:absolute;inset:0;width:100%;height:100%}canvas{display:block;touch-action:none}
    #dock{position:absolute;z-index:3;left:50%;bottom:18px;transform:translateX(-50%);display:flex;align-items:center;gap:12px;height:48px;padding:0 12px;border:1px solid rgba(30,30,30,.16);border-radius:24px;background:rgba(244,244,244,.75);box-shadow:0 12px 34px rgba(28,28,28,.13);backdrop-filter:blur(16px)}
    button{border:0;background:none;color:#252525;width:32px;height:32px;padding:6px;display:grid;place-items:center;border-radius:50%;cursor:pointer}button:hover{background:rgba(32,32,32,.08)}button svg{width:17px;height:17px;fill:none;stroke:currentColor;stroke-width:1.7}
    #play{width:38px;height:38px;color:#f4f4f4;background:#1b1b1b}.divider{width:1px;height:22px;background:rgba(28,28,28,.13)}
    label{display:flex;align-items:center;gap:6px;color:#252525;font:600 12px/1 ui-monospace,SFMono-Regular,Consolas,monospace}input[type=range]{width:78px;accent-color:#222}output{width:18px;text-align:right}
    select{appearance:none;border:0;background:transparent;color:#252525;padding:6px 14px 6px 7px;font:600 13px/1 ui-monospace,SFMono-Regular,Consolas,monospace;cursor:pointer}
    #status{position:absolute;z-index:2;top:20px;right:22px;color:rgba(30,30,30,.56);font:600 12px/1 ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:.04em}
  </style>
</head>
<body>
<main><canvas id="canvas" aria-label="Multi-wall waypoint Sheaf-GBP transport"></canvas><div id="status" aria-live="polite"></div>
  <div id="dock"><select id="shape" aria-label="Payload shape"><option selected>T</option><option>A</option><option>B</option><option>C</option><option>D</option><option>E</option><option>F</option><option>G</option><option>H</option><option>I</option><option>J</option><option>K</option><option>L</option><option>M</option><option>N</option><option>O</option><option>P</option><option>Q</option><option>R</option><option>S</option><option>U</option><option>V</option><option>W</option><option>X</option><option>Y</option><option>Z</option></select><div class="divider"></div><label>ants <input id="agents" type="range" min="8" max="20" step="1" value="12"><output id="agentsOut">12</output></label><div class="divider"></div>
    <button id="play" aria-label="Run"><svg id="playIcon" viewBox="0 0 24 24"><path d="M9 6.5 17 12l-8 5.5Z" fill="currentColor" stroke="none"/></svg></button>
    <button id="reset" aria-label="Reset"><svg viewBox="0 0 24 24"><path d="M5 8a8 8 0 1 1-1 5M5 8V3M5 8h5"/></svg></button>
  </div>
</main>
<script>
const canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d'),play=document.getElementById('play'),playIcon=document.getElementById('playIcon'),statusEl=document.getElementById('status'),shape=document.getElementById('shape'),agents=document.getElementById('agents'),agentsOut=document.getElementById('agentsOut');
let header=null,frames=[],done=null,timeline=0,playing=false,lastTime=0,generation=0,aborter=null;const STEP_MS=25;
function resize(){const d=devicePixelRatio||1;canvas.width=innerWidth*d;canvas.height=innerHeight*d;ctx.setTransform(d,0,0,d,0,0);draw()}addEventListener('resize',resize);
function scale(){if(!header)return 1;return Math.min(innerWidth/(2*header.world.x+.45),innerHeight/(2*header.world.y+.3))}
function S(x,y){const s=scale();return[innerWidth/2+x*s,innerHeight/2-y*s]}
function frameIndex(){if(!frames.length)return-1;const t=Number.isFinite(timeline)?timeline:0;return Math.max(0,Math.min(frames.length-1,Math.floor(t)))}
function pose(){if(!header||!frames.length||timeline<=0)return header?header.source:[0,0,0];const i=frameIndex(),f=frames[i],q=Math.min(1,timeline-i),m=q<.18?0:(q-.18)/.82;let da=Math.atan2(Math.sin(f.next_state[2]-f.state[2]),Math.cos(f.next_state[2]-f.state[2]));return[f.state[0]+(f.next_state[0]-f.state[0])*m,f.state[1]+(f.next_state[1]-f.state[1])*m,f.state[2]+da*m]}
function transform(p,s){const c=Math.cos(s[2]),q=Math.sin(s[2]);return[s[0]+c*p[0]-q*p[1],s[1]+q*p[0]+c*p[1]]}
function poly(points,state){ctx.beginPath();points.forEach((p,i)=>{const z=S(...transform(p,state));i?ctx.lineTo(...z):ctx.moveTo(...z)});ctx.closePath()}
function background(){const g=ctx.createLinearGradient(0,0,innerWidth,innerHeight);g.addColorStop(0,'#eeeeec');g.addColorStop(.48,'#d8d8d5');g.addColorStop(1,'#eeeeeb');ctx.fillStyle=g;ctx.fillRect(0,0,innerWidth,innerHeight);ctx.save();ctx.globalAlpha=.10;ctx.strokeStyle='#3c3c3c';ctx.lineWidth=1;for(let i=0;i<9;i++){ctx.beginPath();for(let x=0;x<=innerWidth;x+=18){let y=innerHeight*(.16+.085*i)+Math.sin(x*.006+i*.72)*20;if(!x)ctx.moveTo(x,y);else ctx.lineTo(x,y)}ctx.stroke()}ctx.restore()}
function walls(){if(!header)return;ctx.fillStyle='#2e2e2d';for(const w of header.walls){const x0=S(w.x-header.world.wall_half_width,0)[0],x1=S(w.x+header.world.wall_half_width,0)[0],top=S(0,w.gap_y+w.aperture_m/2)[1],bottom=S(0,w.gap_y-w.aperture_m/2)[1];ctx.fillRect(x0,0,x1-x0,top);ctx.fillRect(x0,bottom,x1-x0,innerHeight-bottom)}}
function route(){if(!header)return;const pts=[header.source,...header.waypoints];ctx.save();ctx.strokeStyle='rgba(45,45,45,.18)';ctx.setLineDash([3,8]);ctx.lineWidth=1.2;ctx.beginPath();pts.forEach((p,i)=>{const z=S(p[0],p[1]);i?ctx.lineTo(...z):ctx.moveTo(...z)});ctx.stroke();ctx.setLineDash([]);const j=frameIndex(),k=j>=0?frames[j].waypoint_index:0;header.waypoints.forEach((p,i)=>{const z=S(p[0],p[1]);ctx.beginPath();ctx.arc(z[0],z[1],i===k?7:4,0,Math.PI*2);ctx.fillStyle=i<k?'rgba(30,30,30,.20)':i===k?'rgba(25,25,25,.72)':'rgba(30,30,30,.24)';ctx.fill()});ctx.restore()}
function finalPose(){if(!header)return;const goal=header.waypoints[header.waypoints.length-1];ctx.save();ctx.setLineDash([4,6]);ctx.lineWidth=1.25;ctx.strokeStyle='rgba(35,35,35,.42)';ctx.shadowColor='rgba(250,250,246,.60)';ctx.shadowBlur=5;for(const r of header.body_rects){poly(r,goal);ctx.stroke()}ctx.restore()}
function body(state){if(!header)return;ctx.save();ctx.fillStyle='#555553';ctx.strokeStyle='#202020';ctx.lineWidth=1.1;ctx.shadowColor='rgba(0,0,0,.24)';ctx.shadowBlur=8;for(const r of header.body_rects){poly(r,state);ctx.fill();ctx.stroke()}ctx.restore()}
function ant(point,state,i){const p=S(...transform(point,state)),a=state[2]+(i%2?Math.PI/2:-Math.PI/2);ctx.save();ctx.translate(...p);ctx.rotate(a);ctx.fillStyle='#1e1e1d';ctx.strokeStyle='#1e1e1d';ctx.lineWidth=1;ctx.beginPath();ctx.ellipse(-3,0,3.5,2.5,0,0,Math.PI*2);ctx.ellipse(2,0,2.5,2,0,0,Math.PI*2);ctx.fill();for(const y of[-1,1])for(const x of[-1,2]){ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x-3,y*4);ctx.stroke()}ctx.restore()}
function packet(a,b,t,str,bend){const c=[(a[0]+b[0])/2+(bend?.08:-.08)*(b[1]-a[1]),(a[1]+b[1])/2+(bend?.08:-.08)*(a[0]-b[0])],u=1-Math.pow(1-t,2),x=(1-u)*(1-u)*a[0]+2*(1-u)*u*c[0]+u*u*b[0],y=(1-u)*(1-u)*a[1]+2*(1-u)*u*c[1]+u*u*b[1];ctx.save();ctx.globalAlpha=.16+.42*str;ctx.strokeStyle='#4b4b4a';ctx.shadowColor='#f4f4ef';ctx.shadowBlur=13;ctx.lineWidth=.7+1.5*str;ctx.beginPath();ctx.moveTo(...a);ctx.quadraticCurveTo(...c,...b);ctx.stroke();ctx.translate(x,y);ctx.rotate(Math.atan2(b[1]-c[1],b[0]-c[0]));ctx.fillStyle='#292928';ctx.beginPath();ctx.moveTo(6,0);ctx.lineTo(-3,-3);ctx.lineTo(-1,0);ctx.lineTo(-3,3);ctx.closePath();ctx.fill();ctx.restore()}
function messages(state){if(!header||!frames.length||timeline<=0)return;const i=frameIndex(),f=frames[i],phase=Math.max(0,timeline-i),s=Math.min(f.intensity.length-1,Math.floor(phase*f.intensity.length)),t=(phase*f.intensity.length)%1,vals=f.intensity[s],sorted=[...vals].sort((a,b)=>a-b),den=sorted[Math.floor(sorted.length*.86)]||1,pts=header.handles.map(p=>S(...transform(p,state)));for(let m=0;m<header.sender.length;m++)packet(pts[header.sender[m]],pts[header.receiver[m]],t,Math.max(.08,Math.min(1,Math.sqrt((vals[m]||0)/den))),m%2)}
function draw(){background();route();walls();if(!header)return;finalPose();const s=pose();body(s);header.handles.forEach((p,i)=>ant(p,s,i));messages(s)}
async function prepare(){generation++;const mine=generation;if(aborter)aborter.abort();aborter=new AbortController();header=null;frames=[];done=null;timeline=0;playing=false;setIcon();statusEl.textContent='';draw();try{const res=await fetch('/api/stream?shape='+encodeURIComponent(shape.value)+'&agents='+agents.value,{signal:aborter.signal});if(!res.ok)throw Error(await res.text());const reader=res.body.getReader(),decoder=new TextDecoder();let pending='';while(true){const q=await reader.read();if(q.done)break;pending+=decoder.decode(q.value,{stream:true});let cut;while((cut=pending.indexOf('\n'))>=0){const line=pending.slice(0,cut).trim();pending=pending.slice(cut+1);if(!line)continue;const e=JSON.parse(line);if(mine!==generation)return;if(e.type==='header'){header=e;draw()}else if(e.type==='frame')frames.push(e);else if(e.type==='done')done=e}}}catch(e){if(e.name!=='AbortError')statusEl.textContent='error'}}
function setIcon(){playIcon.innerHTML=playing?'<path d="M8 7v10M16 7v10"/>':'<path d="M9 6.5 17 12l-8 5.5Z" fill="currentColor" stroke="none"/>'}
function toggle(){if(!frames.length)return;playing=!playing;setIcon();lastTime=performance.now();if(playing)requestAnimationFrame(tick)}
function tick(now){if(!playing)return;const dt=Math.max(0,Math.min(50,now-lastTime));lastTime=now;const max=done?Math.max(0,frames.length-1):Math.max(0,frames.length-3);timeline=Math.min(max,timeline+dt/STEP_MS);const i=frameIndex(),f=i>=0?frames[i]:null;statusEl.textContent=f?`${Math.min(f.waypoint_index+1,header.waypoints.length)} / ${header.waypoints.length}`:'';draw();if(done&&timeline>=frames.length-1){playing=false;setIcon();statusEl.textContent=done.success?'complete':statusEl.textContent;return}requestAnimationFrame(tick)}
function reset(){playing=false;timeline=0;setIcon();statusEl.textContent='';draw()}
play.addEventListener('click',toggle);document.getElementById('reset').addEventListener('click',reset);shape.addEventListener('change',prepare);agents.addEventListener('input',()=>agentsOut.value=agents.value);agents.addEventListener('change',prepare);resize();prepare();
</script></body></html>"""


class MultiWaypointHandler(BaseHTTPRequestHandler):
    engine: MultiWaypointEngine

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_text(self, status: HTTPStatus, value: str, content_type: str) -> None:
        data = value.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/":
            self._send_text(HTTPStatus.OK, INDEX_HTML, "text/html; charset=utf-8")
            return
        if not self.path.startswith("/api/stream"):
            self._send_text(HTTPStatus.NOT_FOUND, "not found", "text/plain")
            return
        try:
            from urllib.parse import parse_qs, urlparse

            query = parse_qs(urlparse(self.path).query)
            n_agents = int(query.get("agents", [str(DEFAULT_N_AGENTS)])[0])
            payload_shape = query.get("shape", ["T"])[0]
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            with ROLLOUT_LOCK:
                for event in self.engine.events(n_agents, payload_shape):
                    self.wfile.write(json.dumps(event, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return
        except Exception as exc:
            try:
                self.wfile.write(json.dumps({"type": "error", "message": str(exc)}).encode("utf-8") + b"\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    MultiWaypointHandler.engine = MultiWaypointEngine(
        args.checkpoint.resolve(strict=True), args.device
    )
    server = ThreadingHTTPServer((args.host, args.port), MultiWaypointHandler)
    url = f"http://{args.host}:{args.port}/"
    print(url, flush=True)
    if args.open:
        webbrowser.open(url)
    server.serve_forever()


if __name__ == "__main__":
    main()
