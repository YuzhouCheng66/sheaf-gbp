"""Small recurrent local planner with body-frame sensing and no route input.

The trained module predicts a local desired payload twist. The existing
Sheaf-GBP action layer remains responsible for communicating wrench proposals
and decoding individual handle forces. Simulation state is used here only to
render each node's own sensor observation, as in the environment's old sensor.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .config import EnvConfig
from .env import PianoBatch, body_to_world, contact_points, radius_edges
from .gbp import homogeneous_topology
from .models import SheafGBPPolicy
from .payload_shapes import payload_rectangles_for_config
from .single_slit_contact_observation import (
    SingleSlitContactObservationContextV1,
    single_slit_contact_observations_v1,
)


LOCAL_SKILL_RAYS = 32
LOCAL_SKILL_FEATURE_DIM = 78
LOCAL_SKILL_SEGMENTS = 12
LOCAL_SKILL_GEOMETRY_FEATURE_DIM = 78 + 7 * LOCAL_SKILL_SEGMENTS


def _body_frame(vector: Tensor, sine: Tensor, cosine: Tensor) -> Tensor:
    return torch.stack(
        (
            cosine * vector[..., 0] + sine * vector[..., 1],
            -sine * vector[..., 0] + cosine * vector[..., 1],
        ),
        dim=-1,
    )


def obstacle_boxes(scene: PianoBatch, cfg: EnvConfig, reference: Tensor) -> Tensor:
    """Physical solid rectangles including world bounds, for local ray casting."""

    wx, gy, gh = (v.to(reference) for v in (scene.wall_x, scene.gap_y, scene.gap_half))
    half = (gh - 0.03).clamp_min(0.0)
    bottom = torch.stack(
        (wx - cfg.wall_slab, wx + cfg.wall_slab, torch.full_like(wx, -cfg.world_y), gy - half),
        dim=-1,
    )
    top = torch.stack(
        (wx - cfg.wall_slab, wx + cfg.wall_slab, gy + half, torch.full_like(wx, cfg.world_y)),
        dim=-1,
    )
    boxes = torch.cat((bottom, top), dim=1)
    if scene.obstacle_rects is not None:
        reserve = reference.new_tensor([-0.03, 0.03, -0.03, 0.03])
        boxes = torch.cat((boxes, scene.obstacle_rects.to(reference) + reserve), dim=1)
    x, y = cfg.world_x, cfg.world_y
    bounds = (
        reference.new_tensor(
            [
                [-x - 10.0, -x, -y - 10.0, y + 10.0],
                [x, x + 10.0, -y - 10.0, y + 10.0],
                [-x, x, -y - 10.0, -y],
                [-x, x, y, y + 10.0],
            ]
        )
        .unsqueeze(0)
        .expand(reference.shape[0], -1, -1)
    )
    return torch.cat((boxes, bounds), dim=1)


def local_lidar(state: Tensor, handles: Tensor, scene: PianoBatch, cfg: EnvConfig) -> Tensor:
    """32 body-frame rays per handle, clipped at the node's sensing radius."""

    boxes = obstacle_boxes(scene, cfg, state)
    valid = (boxes[..., 0] < boxes[..., 1]) & (boxes[..., 2] < boxes[..., 3])
    angle = torch.arange(LOCAL_SKILL_RAYS, device=state.device, dtype=state.dtype) * (
        2.0 * math.pi / LOCAL_SKILL_RAYS
    )
    direction_angle = angle[None, :] + state[:, 2:3]
    direction = torch.stack((direction_angle.cos(), direction_angle.sin()), dim=-1)
    lower = boxes[..., (0, 2)]
    upper = boxes[..., (1, 3)]
    origin = handles[:, :, None, None, :]
    d = direction[:, None, :, None, :]
    safe_d = torch.where(d.abs() < 1.0e-8, torch.ones_like(d), d)
    first = (lower[:, None, None, :, :] - origin) / safe_d
    second = (upper[:, None, None, :, :] - origin) / safe_d
    near = torch.minimum(first, second)
    far = torch.maximum(first, second)
    inside_axis = (origin >= lower[:, None, None, :, :]) & (origin <= upper[:, None, None, :, :])
    parallel = d.abs() < 1.0e-8
    near = torch.where(parallel & inside_axis, -torch.inf, near)
    far = torch.where(parallel & inside_axis, torch.inf, far)
    near = torch.where(parallel & ~inside_axis, torch.inf, near)
    far = torch.where(parallel & ~inside_axis, -torch.inf, far)
    enter, leave = near.amax(-1).clamp_min(0.0), far.amin(-1)
    admissible = valid[:, None, None, :] & (leave >= enter)
    ranges = enter.masked_fill(~admissible, torch.inf).amin(-1)
    return ranges.clamp_max(cfg.wall_sensor_range) / cfg.wall_sensor_range


def local_boundary_segments(
    state: Tensor, handles: Tensor, scene: PianoBatch, cfg: EnvConfig
) -> Tensor:
    """Nearest solid-boundary segments clipped to each handle's sensing disk.

    Each segment is two body-frame endpoints, an inward normal, and validity.
    A corner outside the sensing radius is never exposed. This is idealized
    local geometric sensing, not a globally supplied wall or passage identity.
    """
    boxes = obstacle_boxes(scene, cfg, state)
    valid_box = (boxes[..., 0] < boxes[..., 1]) & (boxes[..., 2] < boxes[..., 3])
    corners = torch.stack(
        (boxes[..., (0, 2)], boxes[..., (1, 2)], boxes[..., (1, 3)], boxes[..., (0, 3)]), -2
    )
    a = corners.flatten(1, 2)[:, None] - handles[:, :, None]
    b = corners.roll(-1, -2).flatten(1, 2)[:, None] - handles[:, :, None]
    delta = b - a
    length2 = delta.square().sum(-1).clamp_min(1e-12)
    center = -(a * delta).sum(-1) / length2
    line_distance2 = (a + center[..., None] * delta).square().sum(-1)
    half = ((cfg.wall_sensor_range**2 - line_distance2).clamp_min(0.0) / length2).sqrt()
    lower = (center - half).clamp_min(0.0)
    upper = (center + half).clamp_max(1.0)
    valid = (line_distance2 < cfg.wall_sensor_range**2) & (upper > lower)
    valid &= valid_box.repeat_interleave(4, 1)[:, None]
    left, right = a + lower[..., None] * delta, a + upper[..., None] * delta
    normal = torch.stack((-delta[..., 1], delta[..., 0]), -1) / length2.sqrt()[..., None]
    sine, cosine = state[:, 2:3, None].sin(), state[:, 2:3, None].cos()
    left = _body_frame(left, sine, cosine) / cfg.wall_sensor_range
    right = _body_frame(right, sine, cosine) / cfg.wall_sensor_range
    normal = _body_frame(normal, sine, cosine)
    distance2 = (a + center.clamp(0.0, 1.0)[..., None] * delta).square().sum(-1)
    key = distance2.masked_fill(~valid, torch.inf)
    count = min(LOCAL_SKILL_SEGMENTS, key.shape[-1])
    values = torch.cat((left, right, normal, valid[..., None].to(state.dtype)), -1)
    values = torch.where(valid[..., None], values, torch.zeros_like(values))
    # Canonical local ordering must not leak the world-AABB corner numbering.
    # Incident edges have equal nearest distance; break these ties using only
    # body-frame coordinates. Quantization stabilizes float32 roundoff at ties.
    selected = torch.arange(key.shape[-1], device=key.device).expand_as(key)
    ranking = [key.round(decimals=5)] + [values[..., i].round(decimals=5) for i in range(6)]
    for rank in reversed(ranking):
        order = rank.gather(-1, selected).argsort(dim=-1, stable=True)
        selected = selected.gather(-1, order)
    selected = selected[..., :count]
    chosen = values.gather(-2, selected[..., None].expand(*selected.shape, 7))
    if count < LOCAL_SKILL_SEGMENTS:
        chosen = torch.nn.functional.pad(chosen, (0, 0, 0, LOCAL_SKILL_SEGMENTS - count))
    return chosen.flatten(-2)


def local_skill_features(
    state: Tensor,
    scene: PianoBatch,
    cfg: EnvConfig,
    context: SingleSlitContactObservationContextV1,
    *,
    goal: Tensor,
    orientation_required: bool,
    geometry_segments: bool = False,
) -> Tensor:
    """Return [B,N,78] from own sensing, known attachment geometry, and goal.

    No world position, route index, letter identity, or future target is kept.
    Concavity-sensitive geometry descriptors are common attachment information;
    lidar and velocity remain private per-handle measurements.
    """

    contact = contact_points(cfg, state.device).to(state)
    world = body_to_world(contact, state)
    offset = world - state[:, None, :2]
    sine, cosine = state[:, 2:3].sin(), state[:, 2:3].cos()
    velocity = torch.stack(
        (
            state[:, 3:4] - state[:, 5:6] * offset[..., 1],
            state[:, 4:5] + state[:, 5:6] * offset[..., 0],
        ),
        dim=-1,
    )
    body_velocity = _body_frame(velocity, sine, cosine) / 2.0
    # Each agent obtains goal-minus-centre by adding its known rigid handle
    # offset to its own goal-minus-handle measurement.
    goal = goal.reshape(-1, 3)
    own_goal = _body_frame(goal[:, None, :2] - world, sine, cosine) + contact[None]
    heading = goal[:, 2:3] - state[:, 2:3]
    mask = torch.full_like(heading, float(orientation_required))
    goal_heading = torch.cat((mask * heading.sin(), mask * heading.cos(), mask), dim=-1)
    shape = (
        SheafGBPPolicy._local_se2_geometry_signature_from_rectangles(
            payload_rectangles_for_config(cfg, state.device).to(state),
            state,
            dimensions=32,
        )
        .view(1, 1, 32)
        .expand(state.shape[0], cfg.n_agents, -1)
    )
    previous_force = _body_frame(context.previous_force_normalized, sine, cosine)
    rejected_velocity = _body_frame(context.rejected_handle_motion_normalized, sine, cosine)
    hit = context.last_hit.to(state.dtype)[:, None, None].expand(-1, cfg.n_agents, 1)
    result = torch.cat(
        (
            contact[None].expand(state.shape[0], -1, -1),
            body_velocity,
            own_goal / 2.0,
            goal_heading[:, None, :].expand(-1, cfg.n_agents, -1),
            local_lidar(state, world, scene, cfg),
            shape,
            hit,
            previous_force,
            rejected_velocity,
        ),
        dim=-1,
    )
    if result.shape[-1] != LOCAL_SKILL_FEATURE_DIM or not bool(torch.isfinite(result).all()):
        raise RuntimeError("invalid local-skill features")
    if geometry_segments:
        result = torch.cat((result, local_boundary_segments(state, world, scene, cfg)), -1)
    return result


class LocalSkillPlanner(nn.Module):
    """Node-local recurrent planner; three independent body-twist outputs."""

    def __init__(
        self,
        hidden_dim: int = 96,
        *,
        goal_residual: bool = False,
        feature_dim: int = LOCAL_SKILL_FEATURE_DIM,
        goal_residual_scale: float = 1.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.goal_residual = goal_residual
        self.goal_residual_scale = float(goal_residual_scale)
        if feature_dim not in (LOCAL_SKILL_FEATURE_DIM, LOCAL_SKILL_GEOMETRY_FEATURE_DIM):
            raise ValueError("unsupported local sensor interface")
        self.feature_dim = feature_dim
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, max(128, hidden_dim)),
            nn.SiLU(),
            nn.Linear(max(128, hidden_dim), hidden_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.SiLU(), nn.Linear(64, 3))

    def forward(self, features: Tensor, hidden: Tensor | None = None) -> tuple[Tensor, Tensor]:
        prefix = features.shape[:-1]
        x = features.reshape(-1, 1, self.feature_dim)
        initial = None if hidden is None else hidden.reshape(1, -1, self.hidden_dim)
        twist, h = self.forward_sequence(x, initial)
        return twist.reshape(*prefix, 3), h.reshape(*prefix, self.hidden_dim)

    def forward_sequence(self, features: Tensor, hidden: Tensor | None = None):
        encoded = self.encoder(features)
        sequence, h = self.gru(encoded, hidden)
        proposal = torch.tanh(self.head(sequence))
        if self.goal_residual:
            # One generic goal field plus a learned recurrent detour. The
            # residual smoothly vanishes only when all required goal errors
            # vanish; a heading error can still induce translational retreat.
            # This uses no wall template, letter identity, or stored path.
            delta = 2.0 * features[..., 4:6]
            angle = features[..., 8] * torch.atan2(features[..., 6], features[..., 7])
            velocity = 4.0 * delta
            speed = torch.linalg.vector_norm(velocity, dim=-1, keepdim=True).clamp_min(1e-9)
            velocity = velocity * (0.35 / speed).clamp_max(1.0) / 0.65
            omega = (4.0 * angle).clamp(-0.8, 0.8) / 1.4
            nominal = torch.cat((velocity, omega[..., None]), dim=-1)
            error = torch.cat((delta / 0.65, angle[..., None] / 1.4), dim=-1)
            envelope = torch.tanh(2.0 * torch.linalg.vector_norm(error, dim=-1, keepdim=True))
            proposal = proposal / torch.linalg.vector_norm(
                proposal, dim=-1, keepdim=True
            ).clamp_min(1.0)
            proposal = (nominal + self.goal_residual_scale * envelope * proposal).clamp(-1.0, 1.0)
        return proposal, h


def twist_to_plan(twist: Tensor, cfg: EnvConfig) -> Tensor:
    """Consistent 8-D next-pose/twist coordinates for the frozen action layer."""

    scales = twist.new_tensor([0.65, 0.65, 1.40])
    step_scales = twist.new_tensor([0.05, 0.05, 0.10])
    delta = twist * scales * cfg.dt / step_scales
    return torch.cat((delta.clamp(-1.0, 1.0), twist, torch.zeros_like(twist[..., :2])), dim=-1)


def tree_diameter(edges: Tensor, n_nodes: int) -> int:
    """Topology-only setup calculation, never a reduction of online sensing."""
    adjacency = [[] for _ in range(n_nodes)]
    if len(edges) != n_nodes - 1:
        raise ValueError("expected a connected attachment tree")
    for a, b in edges.cpu().tolist():
        adjacency[a].append(b)
        adjacency[b].append(a)
    diameter = 0
    for root in range(n_nodes):
        distance = {root: 0}
        queue = [root]
        for a in queue:
            for b in adjacency[a]:
                if b not in distance:
                    distance[b] = distance[a] + 1
                    queue.append(b)
        if len(distance) != n_nodes:
            raise ValueError("attachment communication tree is disconnected")
        diameter = max(diameter, max(distance.values()))
    return diameter


class LocalSkillController:
    """Node-local learned plans feeding the parent's physical Sheaf-GBP.

    The parent sensor is used only for handle kinematics and causal feedback.
    Its origin is the current body centre and its obstacle slots are empty;
    real obstacle sensing belongs to the new local lidar input. The final plan
    override cannot inherit the parent's geometry-conditioned plan. Private
    temporal state continues across goal changes.
    """

    def __init__(
        self,
        parent: SheafGBPPolicy,
        planner: LocalSkillPlanner,
        *,
        diameter_rounds=False,
        reset_private_each_step=False,
    ):
        self.parent = parent.eval()
        self.planner = planner.eval()
        self.private_hidden = None
        self.diameter_rounds = diameter_rounds
        self.reset_private_each_step = reset_private_each_step

    def reset(self, cfg: EnvConfig) -> None:
        self.cfg = cfg
        self.parent.bind_environment(cfg)
        self.private_hidden = None
        self.legacy_hidden = self.parent.init_hidden(1, cfg.device)
        self.edges = radius_edges(cfg, device=cfg.device)
        self.topology = homogeneous_topology(self.edges, 1, cfg.n_agents, device=cfg.device)
        self.tree_diameter = tree_diameter(
            self.edges[self.parent.local_se2_analytic_tree_edge_mask], cfg.n_agents
        )
        self.rounds = (
            max(self.parent.arm_cfg.gbp_steps, self.tree_diameter)
            if self.diameter_rounds
            else self.parent.arm_cfg.gbp_steps
        )
        self.parent.local_se2_analytic_gbp.num_steps = self.rounds

    def step(self, state, scene, goal, orientation_required, context):
        if self.reset_private_each_step:
            self.private_hidden = None
        features = local_skill_features(
            state,
            scene,
            self.cfg,
            context,
            goal=goal,
            orientation_required=orientation_required,
            geometry_segments=self.planner.feature_dim == LOCAL_SKILL_GEOMETRY_FEATURE_DIM,
        )
        twist, self.private_hidden = self.planner(features, self.private_hidden)
        plan = twist_to_plan(twist, self.cfg)
        output = self.decode_plan(state, context, plan)
        return output, features, plan

    def decode_plan(self, state, context, plan):
        """Run physical GBP alone, also exposing a diagnostic teacher-plan probe."""
        local_state = state.clone()
        local_state[:, :2] = 0.0
        empty_scene = PianoBatch(
            state0=local_state,
            source=local_state[:, :3],
            goal=local_state.new_zeros(1, 3),
            wall_x=local_state.new_tensor([[8.0, 9.0]]),
            gap_y=local_state.new_zeros(1, 2),
            gap_half=local_state.new_full((1, 2), 10.0),
        )
        physical_obs = single_slit_contact_observations_v1(
            local_state, empty_scene, self.cfg, context
        )
        output = self.parent(
            physical_obs,
            self.edges,
            self.legacy_hidden,
            topology=self.topology,
            final_local_se2_plan_override=plan,
            trace_messages=True,
        )
        self.legacy_hidden = output.hidden
        return output
