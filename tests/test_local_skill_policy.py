from dataclasses import replace

import torch

from piano_movers.env import radius_edges, PianoBatch
from piano_movers.local_skill_policy import (
    LocalSkillPlanner,
    local_skill_features,
    twist_to_plan,
    tree_diameter,
)
from piano_movers.single_slit_0720_bc import build_policy
from piano_movers.single_slit_contact_observation import (
    SINGLE_SLIT_CONTACT_OBSERVATION_SCHEMA_V1,
    initial_single_slit_contact_observation_context_v1,
    single_slit_contact_observations_v1,
)
from piano_movers.single_slit_feedback import single_slit_env_config, single_slit_scenario


def test_local_features_are_translation_invariant_and_ignore_masked_heading():
    cfg = replace(single_slit_env_config(), world_x=10.0, world_y=10.0)
    scene = single_slit_scenario(1, cfg)
    state = scene.state0.clone()
    context = initial_single_slit_contact_observation_context_v1(state, cfg)
    f = local_skill_features(
        state, scene, cfg, context, goal=scene.goal, orientation_required=False
    )
    delta = torch.tensor([0.24, -0.18])
    moved = state.clone()
    moved[:, :2] += delta
    goal = scene.goal.clone()
    goal[:, :2] += delta
    goal[:, 2] = 1.7
    shifted = replace(
        scene, wall_x=scene.wall_x + delta[0], gap_y=scene.gap_y + delta[1], goal=goal
    )
    g = local_skill_features(moved, shifted, cfg, context, goal=goal, orientation_required=False)
    torch.testing.assert_close(f, g, atol=2.0e-6, rtol=1.0e-6)


def test_node_planner_has_private_recurrence_and_no_agent_order_dependence():
    torch.manual_seed(8)
    policy = LocalSkillPlanner()
    x = torch.randn(1, 12, 78)
    a, h = policy(x)
    second, _ = policy(x, h)
    assert not torch.equal(a, second)
    perm = torch.randperm(12)
    b, _ = policy(x[:, perm])
    torch.testing.assert_close(b, a[:, perm])


def test_local_features_are_quarter_turn_equivariant_for_generic_walls():
    cfg = replace(single_slit_env_config(), world_x=10.0, world_y=10.0)
    state = torch.tensor([[-0.70, 0.12, 0.31, 0.04, -0.08, 0.10]])
    goal = torch.tensor([[0.9, -0.15, 1.0]])
    boxes = torch.tensor([[[-0.06, 0.06, 0.40, 3.0], [-0.06, 0.06, -3.0, -0.40]]])
    empty = torch.empty(1, 0)
    scene = PianoBatch(
        state0=state,
        source=state[:, :3],
        goal=goal,
        wall_x=empty,
        gap_y=empty,
        gap_half=empty,
        obstacle_rects=boxes,
    )
    context = initial_single_slit_contact_observation_context_v1(state, cfg)
    first = local_skill_features(state, scene, cfg, context, goal=goal, orientation_required=True)
    moved = state.clone()
    moved[:, 0], moved[:, 1] = -state[:, 1], state[:, 0]
    moved[:, 2] += torch.pi / 2
    moved[:, 3], moved[:, 4] = -state[:, 4], state[:, 3]
    target = goal.clone()
    target[:, 0], target[:, 1] = -goal[:, 1], goal[:, 0]
    target[:, 2] += torch.pi / 2
    rotated = torch.stack((-boxes[..., 3], -boxes[..., 2], boxes[..., 0], boxes[..., 1]), -1)
    geometry = replace(
        scene, state0=moved, source=moved[:, :3], goal=target, obstacle_rects=rotated
    )
    second = local_skill_features(
        moved, geometry, cfg, context, goal=target, orientation_required=True
    )
    torch.testing.assert_close(first, second, atol=3e-6, rtol=3e-6)


def test_tree_round_requirement_depends_on_attachment_topology():
    chain = torch.tensor([[i, i + 1] for i in range(5)])
    star = torch.tensor([[0, i] for i in range(1, 6)])
    assert tree_diameter(chain, 6) == 5
    assert tree_diameter(star, 6) == 2


def test_goal_residual_vanishes_at_goal_but_can_retreat_for_required_rotation():
    model = LocalSkillPlanner(goal_residual=True)
    features = torch.randn(2, 12, 78)
    features[..., 4:9] = features.new_tensor([0, 0, 0, 1, 1])
    prediction, _ = model(features)
    torch.testing.assert_close(prediction, torch.zeros_like(prediction), rtol=0, atol=0)
    features[..., 6:9] = features.new_tensor([1, 0, 1])
    prediction, _ = model(features)
    assert bool((prediction[..., :2].abs() > 1e-5).any())
    features[..., 6:9] = features.new_tensor([0, 0, 0])
    prediction, _ = model(features)
    torch.testing.assert_close(prediction, torch.zeros_like(prediction), rtol=0, atol=0)


def test_final_local_plan_reaches_actions_without_analytic_override():
    cfg = single_slit_env_config()
    policy = build_policy(
        "sheaf_gbp",
        cfg,
        observation_schema=SINGLE_SLIT_CONTACT_OBSERVATION_SCHEMA_V1,
        correction_sheaf=True,
        persistent_recovery_state=True,
        local_se2_planner=True,
        local_se2_analytic_total_wrench=True,
        local_se2_distributed_wrench_control=True,
        local_se2_distributed_next_se2_control=True,
        local_se2_distributed_grasp_allreduce=True,
        local_se2_terminal_attractor=True,
    ).eval()
    scene = single_slit_scenario(1, cfg)
    state = scene.state0.clone()
    state[:, 0] = 1.05
    context = initial_single_slit_contact_observation_context_v1(state, cfg)
    obs = single_slit_contact_observations_v1(state, scene, cfg, context)
    twist = torch.zeros(1, cfg.n_agents, 3, requires_grad=True)
    plan = twist_to_plan(twist, cfg)
    out = policy(
        obs,
        radius_edges(cfg, device="cpu"),
        policy.init_hidden(1, "cpu"),
        final_local_se2_plan_override=plan,
        trace_messages=True,
    )
    torch.testing.assert_close(out.aux["local_se2_plan_used"], plan, rtol=0, atol=0)
    out.forces[..., 0].sum().backward()
    assert twist.grad is not None and float(twist.grad.abs().sum()) > 0.0
    assert out.aux["gbp_message_trace"].wire_floats_per_message == 6
