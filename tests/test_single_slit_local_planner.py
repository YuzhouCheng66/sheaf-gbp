from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest

from piano_movers.config import (
    HANDLE_LAYOUT_GLYPH_BOUNDARY_V4,
    HANDLE_LAYOUT_PARAMETRIC_T_V3,
)
from piano_movers.env import observations, radius_edges
from piano_movers.models import (
    LOCAL_SE2_RECOVERY_CANONICAL_DIM,
    LOCAL_SE2_PLAN_BASE_MEAN_AUX_KEY,
    LOCAL_SE2_PLAN_MEAN_AUX_KEY,
    LOCAL_SE2_RECOVERY_PLAN_DELTA_AUX_KEY,
    local_se2_recovery_canonical_features,
)
from piano_movers.single_slit_0720_bc import build_policy
from piano_movers.single_slit_contact_observation import (
    SINGLE_SLIT_CONTACT_LAST_HIT_INDEX_V1,
    SINGLE_SLIT_CONTACT_OBSERVATION_SCHEMA_V1,
    initial_single_slit_contact_observation_context_v1,
    single_slit_contact_observations_v1,
)
from piano_movers.single_slit_feedback import (
    single_slit_env_config,
    single_slit_scenario,
)
from piano_movers.single_slit_dagger import (
    query_exact_local_se2_planning_teacher,
)
from piano_movers.single_slit_dataset import load_executable_teacher_artifact
from piano_movers.single_slit_local_planner import (
    LOCAL_SE2_ANGULAR_VELOCITY_SCALE_RADPS,
    LOCAL_SE2_LINEAR_VELOCITY_SCALE_MPS,
    expand_plan_target_for_agents,
    local_se2_inverse_dynamics_wrench_proposals,
    local_se2_plan_target,
    local_se2_terminal_attractor_plan,
)


ROOT = Path(__file__).resolve().parents[1]
TEACHER_ARTIFACT = (
    ROOT
    / "tests/fixtures/single_slit_T_executable_teacher_0p720.rollout.npz"
)


def test_local_plan_target_uses_payload_body_frame() -> None:
    state = torch.tensor([[1.0, 2.0, torch.pi / 2.0, 0.0, 0.0, 0.0]])
    label = SimpleNamespace(
        next_state=torch.tensor(
            [1.0, 2.05, torch.pi / 2.0 + 0.05, 0.0, 0.65, 0.70]
        ),
        semantic8=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.75]),
    )

    target = local_se2_plan_target(state, label)

    expected = torch.tensor([1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 0.2, 0.75])
    torch.testing.assert_close(target, expected, atol=1.0e-6, rtol=0.0)


def test_local_plan_target_wraps_shortest_heading_change() -> None:
    state = torch.tensor([[0.0, 0.0, 3.10, 0.0, 0.0, 0.0]])
    label = SimpleNamespace(
        next_state=torch.tensor([0.0, 0.0, -3.13, 0.0, 0.0, 0.0]),
        semantic8=torch.zeros(8),
    )

    target = local_se2_plan_target(state, label)

    expected = torch.remainder(torch.tensor(-3.13 - 3.10) + torch.pi, 2 * torch.pi)
    expected = (expected - torch.pi) / 0.10
    torch.testing.assert_close(target[2], expected, atol=3.0e-6, rtol=0.0)


def test_local_plan_expansion_is_private_per_agent_shape() -> None:
    target = torch.arange(16, dtype=torch.float32).reshape(2, 8)

    expanded = expand_plan_target_for_agents(target, 12)

    assert expanded.shape == (2, 12, 8)
    torch.testing.assert_close(expanded[:, 0], target)
    torch.testing.assert_close(expanded[:, -1], target)


def test_canonical_recovery_features_reconstruct_payload_from_each_handle() -> None:
    cfg = single_slit_env_config()
    state = torch.tensor([[-0.24, 0.11, 0.73, 0.08, -0.03, 0.0]])
    scenario = single_slit_scenario(
        1, cfg, initial_state=state, visible_aperture_m=0.720
    )
    context = initial_single_slit_contact_observation_context_v1(state, cfg)
    obs = single_slit_contact_observations_v1(state, scenario, cfg, context)

    canonical = local_se2_recovery_canonical_features(obs, cfg)

    assert canonical.shape == (1, cfg.n_agents, LOCAL_SE2_RECOVERY_CANONICAL_DIM)
    expected = torch.tensor(
        [
            state[0, 0] / cfg.world_x,
            state[0, 1] / cfg.world_y,
            (scenario.goal[0, 0] - state[0, 0]) / cfg.world_x,
        ]
    )
    torch.testing.assert_close(
        canonical[0, :, 2:5],
        expected.expand(cfg.n_agents, -1),
        atol=8.0e-7,
        rtol=0.0,
    )


def _path_memory_policy_and_inputs(
    letter: str,
    *,
    ridge_decoder: bool = False,
    n_agents: int = 12,
    population_resampling: bool = False,
) -> tuple[object, torch.Tensor, torch.Tensor]:
    base_cfg = single_slit_env_config()
    cfg = replace(
        base_cfg,
        n_agents=n_agents,
        payload_shape=letter,
        handle_layout=(
            HANDLE_LAYOUT_PARAMETRIC_T_V3
            if letter == "T"
            else HANDLE_LAYOUT_GLYPH_BOUNDARY_V4
        ),
    )
    policy = build_policy(
        "sheaf_gbp",
        base_cfg,
        observation_schema=SINGLE_SLIT_CONTACT_OBSERVATION_SCHEMA_V1,
        correction_sheaf=True,
        persistent_recovery_state=True,
        local_se2_planner=True,
        local_se2_shape_signature=True,
        local_se2_shape_signature_dim=32,
        local_se2_geometry_refinement_adapter=True,
        local_se2_geometry_refinement_canonical_features=True,
        local_se2_path_memory=True,
        local_se2_path_memory_shape_count=1,
        local_se2_path_memory_max_steps=2,
        local_se2_path_memory_ridge_decoder=ridge_decoder,
        local_se2_path_memory_ridge_points=3,
        local_se2_path_memory_canonical_family_mse_threshold=0.10,
        local_se2_path_memory_population_resampling=population_resampling,
        local_se2_path_memory_population_resampling_points=3,
        local_se2_path_memory_population_preemptive_counts=(10,),
        local_se2_path_memory_canonical_preemptive_counts=(
            11,
            13,
            15,
            16,
            17,
            20,
        ),
        local_se2_path_memory_population_recovery_counts=(11, 13, 15, 17, 20),
        local_se2_analytic_total_wrench=True,
    )
    policy.bind_environment(cfg)
    state = torch.tensor([[-1.05, 0.0, 0.0, 0.0, 0.0, 0.0]])
    scenario = single_slit_scenario(
        1, cfg, initial_state=state, visible_aperture_m=0.720
    )
    context = initial_single_slit_contact_observation_context_v1(state, cfg)
    obs = single_slit_contact_observations_v1(state, scenario, cfg, context)
    inputs = policy.local_se2_base_planner_inputs(
        obs,
        policy.init_hidden(1, "cpu"),
    )
    signature = policy.local_se2_shape_geometry_signature(obs)[0, 0]
    policy.local_se2_path_memory_shape_signatures[0].copy_(signature)
    policy.local_se2_path_memory_reference_poses[0].copy_(
        torch.tensor(
            [
                [-1.05, 0.00, 0.00],
                [-1.04, 0.01, 0.02],
                [-1.03, 0.02, 0.04],
            ]
        )
    )
    policy.local_se2_path_memory_horizons[0] = 2
    policy.local_se2_path_memory_letter_codes[0] = ord(letter)
    if population_resampling:
        policy.local_se2_path_memory_population_resampling_poses[0].copy_(
            torch.tensor(
                [
                    [-1.05, 0.00, 0.00],
                    [-0.20, 0.30, 0.40],
                    [1.05, 0.00, 0.00],
                ]
            )
        )
    if ridge_decoder:
        policy.local_se2_path_memory_train_pose_min.copy_(
            torch.tensor([-1.05, -0.50, -1.50])
        )
        policy.local_se2_path_memory_train_pose_max.copy_(
            torch.tensor([1.05, 0.50, 1.50])
        )
    return policy, inputs, torch.zeros(1, cfg.n_agents, 8)


def test_path_memory_is_strict_local_and_letter_code_independent() -> None:
    policy, inputs, incumbent = _path_memory_policy_and_inputs("A")

    plan, active = policy.local_se2_path_memory_plan(inputs, incumbent)
    policy.local_se2_path_memory_letter_codes[0] = ord("Q")
    relabelled, relabelled_active = policy.local_se2_path_memory_plan(
        inputs, incumbent
    )

    assert bool(active.all()) and bool(relabelled_active.all())
    torch.testing.assert_close(
        plan,
        plan[:, :1].expand_as(plan),
        atol=2.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(relabelled, plan, atol=0.0, rtol=0.0)


def test_path_memory_is_structurally_inactive_for_canonical_t() -> None:
    policy, inputs, incumbent = _path_memory_policy_and_inputs("T")
    incumbent = torch.linspace(-0.6, 0.6, 8).view(1, 1, 8).expand_as(incumbent)

    plan, active = policy.local_se2_path_memory_plan(inputs, incumbent)

    assert not bool(active.any())
    selected = torch.where(active, plan, incumbent)
    torch.testing.assert_close(selected, incumbent, atol=0.0, rtol=0.0)


def test_path_memory_preserves_near_canonical_parametric_t_family() -> None:
    policy, inputs, incumbent = _path_memory_policy_and_inputs(
        "T", ridge_decoder=True
    )
    incumbent = torch.linspace(-0.6, 0.6, 8).view(1, 1, 8).expand_as(incumbent)
    nearby_inputs = inputs.clone()
    nearby_inputs[..., -32:] += 0.01

    plan, active = policy.local_se2_path_memory_plan(nearby_inputs, incumbent)

    assert not bool(active.any())
    selected = torch.where(active, plan, incumbent)
    torch.testing.assert_close(selected, incumbent, atol=0.0, rtol=0.0)


def test_geometry_refinement_preserves_near_canonical_parametric_t_family() -> None:
    policy, inputs, incumbent = _path_memory_policy_and_inputs("T")
    incumbent = torch.linspace(-0.6, 0.6, 8).view(1, 1, 8).expand_as(incumbent)
    nearby_inputs = inputs.clone()
    nearby_inputs[..., -32:] += 0.01
    with torch.no_grad():
        policy.local_se2_geometry_refinement_head.bias.fill_(1.0)

    refined, delta = policy.local_se2_geometry_refinement_from_planner_inputs(
        nearby_inputs, incumbent
    )

    torch.testing.assert_close(refined, incumbent, atol=0.0, rtol=0.0)
    torch.testing.assert_close(delta, torch.zeros_like(delta), atol=0.0, rtol=0.0)


def test_private_path_memory_does_not_change_sheaf_wire_contract() -> None:
    memory_policy, _inputs, _incumbent = _path_memory_policy_and_inputs("A")
    cfg = single_slit_env_config()
    base_policy = build_policy(
        "sheaf_gbp",
        cfg,
        observation_schema=SINGLE_SLIT_CONTACT_OBSERVATION_SCHEMA_V1,
        correction_sheaf=True,
        persistent_recovery_state=True,
        local_se2_planner=True,
        local_se2_shape_signature=True,
        local_se2_shape_signature_dim=32,
        local_se2_geometry_refinement_adapter=True,
        local_se2_geometry_refinement_canonical_features=True,
        local_se2_analytic_total_wrench=True,
    )

    assert (
        memory_policy.communication_floats_per_directed_edge_round()
        == base_policy.communication_floats_per_directed_edge_round()
    )
    assert memory_policy.communication_rounds() == base_policy.communication_rounds()


def test_path_memory_ridge_decoder_is_ood_only_and_train_envelope_gated() -> None:
    policy, inputs, incumbent = _path_memory_policy_and_inputs(
        "A", ridge_decoder=True
    )
    ood_inputs = inputs.clone()
    ood_inputs[..., -32] += 0.01
    weights = policy.local_se2_path_memory_ridge_weights
    weights.zero_()
    weights[-1].copy_(
        torch.tensor(
            [
                [-1.05, 0.00, 0.00],
                [-1.00, 0.00, 0.00],
                [1.05, 0.00, 0.00],
            ]
        )
    )

    exact_plan, _ = policy.local_se2_path_memory_plan(inputs, incumbent)
    ridge_plan, _ = policy.local_se2_path_memory_plan(ood_inputs, incumbent)
    assert bool((ridge_plan[..., 0] > exact_plan[..., 0] + 0.5).all())

    # A physically out-of-envelope extrapolation is rejected without needing
    # a held-out label; the nearest individually certified train path remains.
    weights[-1, 1, 2] = 8.0
    fallback_plan, _ = policy.local_se2_path_memory_plan(ood_inputs, incumbent)
    torch.testing.assert_close(fallback_plan, exact_plan, atol=2.0e-6, rtol=0.0)


def test_path_memory_population_resampling_is_hybrid_and_preserves_n12() -> None:
    def evaluate(n_agents: int, recovery: bool) -> tuple[torch.Tensor, torch.Tensor]:
        policy, inputs, incumbent = _path_memory_policy_and_inputs(
            "T",
            n_agents=n_agents,
            population_resampling=True,
        )
        recovery_active = torch.full(
            (1, n_agents, 1),
            recovery,
            dtype=torch.bool,
        )
        return policy.local_se2_path_memory_plan(
            inputs,
            incumbent,
            recovery_active=recovery_active,
        )

    n10_plan, n10_active = evaluate(10, recovery=False)
    n9_plan, n9_inactive = evaluate(9, recovery=True)
    n11_preemptive_plan, n11_preemptive_active = evaluate(11, recovery=False)
    n11_recovery_plan, n11_active = evaluate(11, recovery=True)
    n12_plan, n12_active = evaluate(12, recovery=True)

    # N=10 takes the smoother population path globally. Exact canonical T
    # also pre-empts at its admitted difficult populations, while a count
    # excluded by the development gate stays on the successful incumbent even
    # after a contact.
    assert bool(n10_active.all())
    assert bool(n11_preemptive_active.all())
    assert bool(n11_active.all())
    assert bool((n10_plan[..., 1] > 0.0).all())
    assert bool((n11_preemptive_plan[..., 1] > 0.0).all())
    assert bool((n11_recovery_plan[..., 1] > 0.0).all())
    assert not bool(n9_inactive.any())
    n9_selected = torch.where(n9_inactive, n9_plan, torch.zeros_like(n9_plan))
    torch.testing.assert_close(
        n9_selected, torch.zeros_like(n9_selected), atol=0.0, rtol=0.0
    )

    # Canonical 12-agent T is a strict compatibility branch: even an asserted
    # recovery signal cannot activate the new path-memory planner.
    assert not bool(n12_active.any())
    n12_selected = torch.where(n12_active, n12_plan, torch.zeros_like(n12_plan))
    torch.testing.assert_close(
        n12_selected, torch.zeros_like(n12_selected), atol=0.0, rtol=0.0
    )


def test_exact_local_planning_teacher_advances_one_certified_se2_waypoint() -> None:
    cfg = single_slit_env_config()
    artifact = load_executable_teacher_artifact(TEACHER_ARTIFACT, cfg)

    label, error = query_exact_local_se2_planning_teacher(
        artifact.states[:1],
        0,
        artifact,
        cfg,
    )

    assert error is None and label is not None
    assert label.mode == "local_se2_forward"
    assert label.nearest_index == label.reference_index == 1
    assert float(label.clearance) > 0.0
    torch.testing.assert_close(
        label.next_state[:3],
        artifact.reference_poses[1],
        atol=2.0e-6,
        rtol=0.0,
    )


def test_exact_local_planning_teacher_recenters_when_forward_step_is_unsafe() -> None:
    cfg = single_slit_env_config()
    artifact = load_executable_teacher_artifact(TEACHER_ARTIFACT, cfg)
    # Deterministic safe off-fiber state whose forward index-54 transition is
    # unsafe, while a dynamically executable recenter transition is exact-SAT
    # safe.  The ordered progress memory must not move backwards.
    state = torch.tensor(
        [
            [
                -0.0357787646,
                -0.1737780124,
                1.2283085585,
                0.1087903082,
                -0.2494311780,
                0.2467506677,
            ]
        ]
    )

    label, error = query_exact_local_se2_planning_teacher(
        state,
        53,
        artifact,
        cfg,
    )

    assert error is None and label is not None
    assert label.mode == "local_se2_recenter"
    assert label.nearest_index == label.reference_index == 53
    assert float(label.clearance) > 0.0


def test_exact_local_planning_teacher_brakes_a_fast_off_fiber_state() -> None:
    cfg = single_slit_env_config()
    artifact = load_executable_teacher_artifact(TEACHER_ARTIFACT, cfg)
    state = torch.tensor(
        [
            [
                -0.0959979743,
                0.1622170806,
                1.1103563309,
                0.2204036266,
                0.0095788781,
                0.1819614321,
            ]
        ]
    )

    label, error = query_exact_local_se2_planning_teacher(
        state,
        41,
        artifact,
        cfg,
    )

    assert error is None and label is not None
    assert label.mode == "local_se2_brake"
    assert label.nearest_index == label.reference_index == 41
    assert float(label.clearance) > 0.0
    assert float(torch.linalg.vector_norm(label.next_state[3:])) < 2.0e-6


def test_local_inverse_dynamics_proposals_average_to_global_wrench() -> None:
    cfg = single_slit_env_config()
    state = torch.tensor([[0.3, -0.1, 0.47, 0.08, -0.03, 0.22]])
    scenario = single_slit_scenario(1, cfg, initial_state=state)
    obs = observations(state, scenario, cfg)

    desired_world_velocity = torch.tensor([0.12, 0.04])
    cosine = torch.cos(state[0, 2])
    sine = torch.sin(state[0, 2])
    desired_body_velocity = torch.stack(
        (
            cosine * desired_world_velocity[0] + sine * desired_world_velocity[1],
            -sine * desired_world_velocity[0] + cosine * desired_world_velocity[1],
        )
    )
    desired_omega = torch.tensor(-0.16)
    plan = torch.zeros(1, cfg.n_agents, 8)
    plan[..., 3:5] = (
        desired_body_velocity / LOCAL_SE2_LINEAR_VELOCITY_SCALE_MPS
    )
    plan[..., 5] = desired_omega / LOCAL_SE2_ANGULAR_VELOCITY_SCALE_RADPS

    proposals = local_se2_inverse_dynamics_wrench_proposals(obs, plan, cfg)
    linear_decay = 1.0 - cfg.linear_damping * cfg.dt
    angular_decay = 1.0 - cfg.angular_damping * cfg.dt
    expected_force = (
        cfg.mass
        * (desired_world_velocity / linear_decay - state[0, 3:5])
        / cfg.dt
    )
    expected_torque = (
        cfg.inertia
        * (desired_omega / angular_decay - state[0, 5])
        / cfg.dt
    )
    scale = torch.tensor(
        [
            cfg.n_agents * cfg.max_force,
            cfg.n_agents * cfg.max_force,
            0.75 * cfg.n_agents * cfg.max_force,
        ]
    )
    expected = torch.cat((expected_force, expected_torque.view(1))) / scale

    torch.testing.assert_close(
        proposals.mean(dim=1)[0], expected, atol=2.0e-6, rtol=2.0e-6
    )


def test_terminal_attractor_is_local_common_and_only_after_full_clearance() -> None:
    cfg = single_slit_env_config()
    learned = torch.randn(1, cfg.n_agents, 8).tanh()

    source_state = torch.tensor([[-1.05, 0.0, 0.0, 0.0, 0.0, 0.0]])
    source_scenario = single_slit_scenario(1, cfg, initial_state=source_state)
    source_obs = observations(source_state, source_scenario, cfg)
    source_plan, source_active = local_se2_terminal_attractor_plan(
        source_obs,
        learned,
        cfg,
        position_gain=2.0,
        angle_gain=2.0,
        max_speed=0.35,
        max_omega=0.8,
    )
    assert not bool(source_active.any())
    torch.testing.assert_close(source_plan, learned)

    clear_state = torch.tensor([[0.90, 0.12, 0.10, 0.0, 0.0, 0.0]])
    clear_scenario = single_slit_scenario(1, cfg, initial_state=clear_state)
    clear_obs = observations(clear_state, clear_scenario, cfg)
    clear_plan, clear_active = local_se2_terminal_attractor_plan(
        clear_obs,
        learned,
        cfg,
        position_gain=2.0,
        angle_gain=2.0,
        max_speed=0.35,
        max_omega=0.8,
    )
    assert bool(clear_active.all())
    torch.testing.assert_close(
        clear_plan[:, 1:, :6],
        clear_plan[:, :1, :6].expand(-1, cfg.n_agents - 1, -1),
        atol=2.0e-6,
        rtol=0.0,
    )
    # Private cross-track/progress coordinates remain the learned values.
    torch.testing.assert_close(clear_plan[..., 6:8], learned[..., 6:8])


def test_terminal_attractor_obeys_shared_nonzero_mission_goal() -> None:
    cfg = replace(single_slit_env_config(), goal_y=0.24, goal_theta=0.30)
    state = torch.tensor([[0.90, 0.0, 0.0, 0.0, 0.0, 0.0]])
    scenario = single_slit_scenario(1, cfg, initial_state=state)
    obs = observations(state, scenario, cfg)
    learned = torch.zeros(1, cfg.n_agents, 8)
    selected, active = local_se2_terminal_attractor_plan(
        obs,
        learned,
        cfg,
        position_gain=2.0,
        angle_gain=2.0,
        max_speed=0.35,
        max_omega=0.8,
    )

    assert bool(active.all())
    assert bool((selected[..., 3] > 0.0).all())
    assert bool((selected[..., 4] > 0.0).all())
    assert bool((selected[..., 5] > 0.0).all())
    torch.testing.assert_close(
        scenario.goal[0],
        torch.tensor([cfg.goal_x, cfg.goal_y, cfg.goal_theta]),
        atol=0.0,
        rtol=0.0,
    )


def test_feasible_corridor_is_source_exact_below_physical_fit_threshold() -> None:
    cfg = single_slit_env_config()
    learned = torch.randn(1, cfg.n_agents, 8).tanh()
    state = torch.tensor([[-1.05, 0.0, 0.0, 0.0, 0.0, 0.0]])
    # 0.960 m rendered width has only 0.900 m physical free width after the
    # exact collision model's bilateral 0.030 m reserve, less than this T.
    for aperture in (0.720, 0.960):
        scenario = single_slit_scenario(
            1, cfg, initial_state=state, visible_aperture_m=aperture
        )
        obs = observations(state, scenario, cfg)

        selected, active = local_se2_terminal_attractor_plan(
            obs,
            learned,
            cfg,
            position_gain=2.0,
            angle_gain=2.0,
            max_speed=0.35,
            max_omega=0.8,
            feasible_corridor=True,
            corridor_reserve_m=0.015,
            corridor_max_speed=0.50,
        )

        assert not bool(active.any())
        assert torch.equal(selected, learned)


def test_route_free_geometry_removes_canonical_t_learned_route_exception() -> None:
    cfg = single_slit_env_config()
    learned = torch.randn(1, cfg.n_agents, 8).tanh()
    state = torch.tensor([[-1.05, 0.0, 0.0, 0.0, 0.0, 0.0]])
    scenario = single_slit_scenario(
        1, cfg, initial_state=state, visible_aperture_m=0.720
    )
    obs = observations(state, scenario, cfg)

    selected, active = local_se2_terminal_attractor_plan(
        obs,
        learned,
        cfg,
        position_gain=2.0,
        angle_gain=2.0,
        max_speed=0.35,
        max_omega=0.8,
        feasible_corridor=True,
        route_free_geometry=True,
        corridor_reserve_m=0.015,
        corridor_max_speed=0.50,
    )

    visible = obs[..., 16] > 0.5
    assert bool(visible.any())
    assert bool(active[..., 0][visible].all())
    assert not torch.equal(selected[..., :6][visible], learned[..., :6][visible])
    assert torch.equal(selected[~visible.unsqueeze(-1).expand_as(selected)], learned[~visible.unsqueeze(-1).expand_as(learned)])


def test_feasible_corridor_uses_only_visible_local_geometry() -> None:
    cfg = single_slit_env_config()
    learned = torch.randn(1, cfg.n_agents, 8).tanh()
    state = torch.tensor([[-1.05, 0.0, 0.0, 0.0, 0.0, 0.0]])
    scenario = single_slit_scenario(
        1, cfg, initial_state=state, visible_aperture_m=1.080
    )
    obs = observations(state, scenario, cfg)

    selected, active = local_se2_terminal_attractor_plan(
        obs,
        learned,
        cfg,
        position_gain=2.0,
        angle_gain=2.0,
        max_speed=0.35,
        max_omega=0.8,
        feasible_corridor=True,
        corridor_reserve_m=0.015,
        corridor_max_speed=0.50,
    )

    visible = obs[..., 16:17] > 0.5
    assert bool(visible.any())
    assert torch.equal(active, visible)
    assert torch.equal(selected[~visible.expand_as(selected)], learned[~visible.expand_as(learned)])
    # Every visible node independently reconstructs the same physical target.
    visible_plans = selected[..., :6][visible.expand(-1, -1, 6)].reshape(-1, 6)
    torch.testing.assert_close(
        visible_plans[1:],
        visible_plans[:1].expand(visible_plans.shape[0] - 1, -1),
        atol=2.0e-6,
        rtol=0.0,
    )
    # The canonical T origin is not at its vertical bounding-box centre, so a
    # centred 1.080 m slit still requires a small pre-alignment before x motion.
    torch.testing.assert_close(
        visible_plans[:, 3],
        torch.zeros_like(visible_plans[:, 3]),
        atol=1.0e-7,
        rtol=0.0,
    )
    assert bool((visible_plans[:, 4] > 0.0).all())
    torch.testing.assert_close(selected[..., 6:8], learned[..., 6:8])


def test_dynamic_corridor_reserve_keeps_marginal_fit_on_feedback_planner() -> None:
    cfg = single_slit_env_config()
    learned = torch.randn(1, cfg.n_agents, 8).tanh()
    state = torch.tensor([[-1.05, 0.0, 0.0, 0.0, 0.0, 0.0]])

    active_by_aperture: dict[float, torch.Tensor] = {}
    selected_by_aperture: dict[float, torch.Tensor] = {}
    for aperture in (1.020, 1.080):
        scenario = single_slit_scenario(
            1, cfg, initial_state=state, visible_aperture_m=aperture
        )
        obs = observations(state, scenario, cfg)
        selected, active = local_se2_terminal_attractor_plan(
            obs,
            learned,
            cfg,
            position_gain=2.0,
            angle_gain=2.0,
            max_speed=0.35,
            max_omega=0.8,
            feasible_corridor=True,
            corridor_reserve_m=0.050,
            corridor_max_speed=0.50,
        )
        active_by_aperture[aperture] = active
        selected_by_aperture[aperture] = selected

    # A 1.020 m rendered opening is statically feasible, but has insufficient
    # bilateral margin for the 0.50 m/s direct controller. It must retain the
    # learned feedback plan. At 1.080 m the same local certificate is active.
    assert not bool(active_by_aperture[1.020].any())
    assert torch.equal(selected_by_aperture[1.020], learned)
    assert bool(active_by_aperture[1.080].any())
    assert not torch.equal(selected_by_aperture[1.080], learned)


def test_letter_corridor_prefers_robust_support_passage_action() -> None:
    cfg = replace(
        single_slit_env_config(),
        payload_shape="A",
        handle_layout=HANDLE_LAYOUT_GLYPH_BOUNDARY_V4,
    )
    learned = torch.randn(1, cfg.n_agents, 8).tanh()
    state = torch.tensor([[-1.05, 0.0, 0.0, 0.0, 0.0, 0.0]])
    scenario = single_slit_scenario(
        1, cfg, initial_state=state, visible_aperture_m=0.720
    )
    obs = observations(state, scenario, cfg)

    selected, active = local_se2_terminal_attractor_plan(
        obs,
        learned,
        cfg,
        position_gain=2.0,
        angle_gain=2.0,
        max_speed=0.35,
        max_omega=0.8,
        feasible_corridor=True,
        corridor_reserve_m=0.050,
        corridor_max_speed=0.50,
    )

    visible = obs[..., 16] > 0.5
    assert bool(active[..., 0][visible].all())
    # The upright A is taller than the physical opening, but it has a robust
    # whole-body passage orientation.  Prefer that smooth support maneuver to
    # an arbitrary feasible lattice route and rotate before approaching.
    visible_plans = selected[..., :6][
        visible.unsqueeze(-1).expand(-1, -1, 6)
    ].reshape(-1, 6)
    torch.testing.assert_close(
        visible_plans[1:],
        visible_plans[:1].expand(visible_plans.shape[0] - 1, -1),
        atol=2.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        visible_plans[:, 3],
        torch.zeros_like(visible_plans[:, 3]),
        atol=1.0e-7,
        rtol=0.0,
    )
    assert bool((visible_plans[:, 5].abs() > 0.5).all())
    torch.testing.assert_close(selected[..., 6:8], learned[..., 6:8])


@pytest.mark.parametrize(
    ("gap_center_y", "aligned_y", "lateral_sign"),
    ((0.36, 0.27, 1.0), (-0.36, -0.15, -1.0)),
)
def test_one_sided_easy_corridor_aligns_before_advancing(
    gap_center_y: float,
    aligned_y: float,
    lateral_sign: float,
) -> None:
    cfg = single_slit_env_config()
    learned = torch.randn(1, cfg.n_agents, 8).tanh()
    scenario = single_slit_scenario(
        1,
        cfg,
        initial_state=torch.tensor([[-1.05, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        visible_aperture_m=1.440,
    )
    scenario.gap_y[:, 0] = gap_center_y

    initial = torch.tensor([[-1.05, 0.0, 0.0, 0.0, 0.0, 0.0]])
    initial_obs = observations(initial, scenario, cfg)
    aligning_plan, active = local_se2_terminal_attractor_plan(
        initial_obs,
        learned,
        cfg,
        position_gain=2.0,
        angle_gain=2.0,
        max_speed=0.35,
        max_omega=0.8,
        feasible_corridor=True,
        corridor_reserve_m=0.050,
        corridor_max_speed=0.50,
    )
    visible = initial_obs[..., 16] > 0.5
    assert bool(active[..., 0][visible].all())
    torch.testing.assert_close(
        aligning_plan[..., 3][visible],
        torch.zeros_like(aligning_plan[..., 3][visible]),
        atol=1.0e-7,
        rtol=0.0,
    )
    assert bool((lateral_sign * aligning_plan[..., 4][visible] > 0.0).all())

    aligned = torch.tensor(
        [[-1.05, aligned_y, 0.0, 0.0, 0.0, 0.0]], dtype=initial.dtype
    )
    aligned_obs = observations(aligned, scenario, cfg)
    crossing_plan, crossing_active = local_se2_terminal_attractor_plan(
        aligned_obs,
        learned,
        cfg,
        position_gain=2.0,
        angle_gain=2.0,
        max_speed=0.35,
        max_omega=0.8,
        feasible_corridor=True,
        corridor_reserve_m=0.050,
        corridor_max_speed=0.50,
    )
    crossing_visible = aligned_obs[..., 16] > 0.5
    assert bool(crossing_active[..., 0][crossing_visible].all())
    assert bool((crossing_plan[..., 3][crossing_visible] > 0.0).all())


def test_post_contact_plan_adapter_is_source_null_but_live_after_hit() -> None:
    cfg = single_slit_env_config()
    policy = build_policy(
        "sheaf_gbp",
        cfg,
        observation_schema=SINGLE_SLIT_CONTACT_OBSERVATION_SCHEMA_V1,
        correction_sheaf=True,
        persistent_recovery_state=True,
        local_se2_planner=True,
        local_se2_recovery_adapter=True,
    )
    policy.eval()
    scenario = single_slit_scenario(1, cfg)
    context = initial_single_slit_contact_observation_context_v1(
        scenario.state0, cfg
    )
    safe_obs = single_slit_contact_observations_v1(
        scenario.state0, scenario, cfg, context
    )
    edges = radius_edges(cfg, device="cpu")
    hidden = policy.init_hidden(1, "cpu")
    with torch.no_grad():
        policy.local_se2_recovery_adapter_head.bias.fill_(0.35)
        safe = policy(safe_obs, edges, hidden)
    torch.testing.assert_close(
        safe.aux[LOCAL_SE2_PLAN_MEAN_AUX_KEY],
        safe.aux[LOCAL_SE2_PLAN_BASE_MEAN_AUX_KEY],
        atol=0.0,
        rtol=0.0,
    )
    assert not bool(
        torch.count_nonzero(safe.aux[LOCAL_SE2_RECOVERY_PLAN_DELTA_AUX_KEY])
    )

    hit_obs = safe_obs.clone()
    hit_obs[..., SINGLE_SLIT_CONTACT_LAST_HIT_INDEX_V1] = 1.0
    hit_obs[..., SINGLE_SLIT_CONTACT_LAST_HIT_INDEX_V1 + 1] = 0.5
    with torch.no_grad():
        hit = policy(hit_obs, edges, hidden)
    assert bool(torch.count_nonzero(hit.aux[LOCAL_SE2_RECOVERY_PLAN_DELTA_AUX_KEY]))
    assert not torch.equal(
        hit.aux[LOCAL_SE2_PLAN_MEAN_AUX_KEY],
        hit.aux[LOCAL_SE2_PLAN_BASE_MEAN_AUX_KEY],
    )


def test_compact_base_planner_input_reproduces_forward_plan_bitwise() -> None:
    """Authenticated planner shards must encode the exact private MLP input."""

    cfg = single_slit_env_config()
    policy = build_policy(
        "sheaf_gbp",
        cfg,
        observation_schema=SINGLE_SLIT_CONTACT_OBSERVATION_SCHEMA_V1,
        correction_sheaf=True,
        persistent_recovery_state=True,
        local_se2_planner=True,
        local_se2_recovery_adapter=True,
    )
    policy.eval()
    scenario = single_slit_scenario(1, cfg)
    context = initial_single_slit_contact_observation_context_v1(
        scenario.state0, cfg
    )
    obs = single_slit_contact_observations_v1(
        scenario.state0, scenario, cfg, context
    )
    hidden = policy.init_hidden(1, "cpu")
    edges = radius_edges(cfg, device="cpu")

    with torch.no_grad():
        planner_input = policy.local_se2_base_planner_inputs(obs, hidden)
        reconstructed = torch.tanh(
            policy.local_se2_plan_mean_head(
                policy.local_se2_planner_encoder(planner_input)
            )
        )
        forward = policy(obs, edges, hidden)

    assert planner_input.shape == (1, cfg.n_agents, 131)
    torch.testing.assert_close(
        reconstructed,
        forward.aux[LOCAL_SE2_PLAN_BASE_MEAN_AUX_KEY],
        atol=0.0,
        rtol=0.0,
    )


def test_private_planner_continuation_matches_full_forward_across_time() -> None:
    """Fast dataset extraction must not alter any later planner input."""

    cfg = single_slit_env_config()
    policy = build_policy(
        "sheaf_gbp",
        cfg,
        observation_schema=SINGLE_SLIT_CONTACT_OBSERVATION_SCHEMA_V1,
        correction_sheaf=True,
        persistent_recovery_state=True,
        local_se2_planner=True,
        local_se2_recovery_adapter=True,
        local_se2_shape_signature=True,
    )
    policy.eval()
    scenario = single_slit_scenario(1, cfg)
    context = initial_single_slit_contact_observation_context_v1(
        scenario.state0, cfg
    )
    observation = single_slit_contact_observations_v1(
        scenario.state0, scenario, cfg, context
    )
    edges = radius_edges(cfg, device="cpu")
    full_hidden = policy.init_hidden(1, "cpu")
    private_hidden = full_hidden.clone()
    with torch.no_grad():
        for _ in range(5):
            full_input = policy.local_se2_base_planner_inputs(
                observation, full_hidden
            )
            private_input, private_hidden = (
                policy.local_se2_base_planner_inputs_and_next_private_hidden(
                    observation, private_hidden
                )
            )
            assert torch.equal(private_input, full_input)
            full_hidden = policy(observation, edges, full_hidden).hidden

def test_contact_rebel_plan_adapter_is_supported_only_on_current_hit() -> None:
    cfg = single_slit_env_config()
    policy = build_policy(
        "sheaf_gbp",
        cfg,
        observation_schema=SINGLE_SLIT_CONTACT_OBSERVATION_SCHEMA_V1,
        correction_sheaf=True,
        persistent_recovery_state=True,
        local_se2_planner=True,
        local_se2_recovery_adapter=True,
        local_se2_recovery_canonical_features=True,
        local_se2_contact_rebel_adapter=True,
    )
    policy.eval()
    scenario = single_slit_scenario(1, cfg)
    context = initial_single_slit_contact_observation_context_v1(
        scenario.state0, cfg
    )
    safe_obs = single_slit_contact_observations_v1(
        scenario.state0, scenario, cfg, context
    )
    approach_hit_obs = safe_obs.clone()
    approach_hit_obs[..., SINGLE_SLIT_CONTACT_LAST_HIT_INDEX_V1] = 1.0
    approach_hit_obs[..., SINGLE_SLIT_CONTACT_LAST_HIT_INDEX_V1 + 1] = 0.5
    exit_state = scenario.state0.clone()
    exit_state[:, 0] = 0.15
    exit_scenario = single_slit_scenario(
        1, cfg, initial_state=exit_state, visible_aperture_m=0.720
    )
    exit_context = initial_single_slit_contact_observation_context_v1(
        exit_state, cfg
    )
    exit_hit_obs = single_slit_contact_observations_v1(
        exit_state, exit_scenario, cfg, exit_context
    )
    exit_hit_obs[..., SINGLE_SLIT_CONTACT_LAST_HIT_INDEX_V1] = 1.0
    exit_hit_obs[..., SINGLE_SLIT_CONTACT_LAST_HIT_INDEX_V1 + 1] = 0.5
    exit_safe_obs = exit_hit_obs.clone()
    exit_safe_obs[..., SINGLE_SLIT_CONTACT_LAST_HIT_INDEX_V1 :] = 0.0
    edges = radius_edges(cfg, device="cpu")
    hidden = policy.init_hidden(1, "cpu")
    with torch.no_grad():
        policy.local_se2_contact_rebel_head.bias.fill_(0.35)
        safe = policy(safe_obs, edges, hidden)
        approach_hit = policy(approach_hit_obs, edges, hidden)
        exit_hit = policy(exit_hit_obs, edges, hidden)
        exit_persistent = policy(exit_safe_obs, edges, exit_hit.hidden)
    torch.testing.assert_close(
        safe.aux[LOCAL_SE2_PLAN_MEAN_AUX_KEY],
        safe.aux[LOCAL_SE2_PLAN_BASE_MEAN_AUX_KEY],
        atol=0.0,
        rtol=0.0,
    )
    assert not bool(
        torch.count_nonzero(safe.aux[LOCAL_SE2_RECOVERY_PLAN_DELTA_AUX_KEY])
    )
    assert not bool(
        torch.count_nonzero(
            approach_hit.aux[LOCAL_SE2_RECOVERY_PLAN_DELTA_AUX_KEY]
        )
    )
    assert bool(
        torch.count_nonzero(exit_hit.aux[LOCAL_SE2_RECOVERY_PLAN_DELTA_AUX_KEY])
    )
    assert bool(
        torch.count_nonzero(
            exit_persistent.aux[LOCAL_SE2_RECOVERY_PLAN_DELTA_AUX_KEY]
        )
    )
    assert not torch.equal(
        exit_hit.aux[LOCAL_SE2_PLAN_MEAN_AUX_KEY],
        exit_hit.aux[LOCAL_SE2_PLAN_BASE_MEAN_AUX_KEY],
    )


def test_exit_refinement_is_source_null_below_its_later_support() -> None:
    cfg = single_slit_env_config()
    policy = build_policy(
        "sheaf_gbp",
        cfg,
        observation_schema=SINGLE_SLIT_CONTACT_OBSERVATION_SCHEMA_V1,
        correction_sheaf=True,
        persistent_recovery_state=True,
        local_se2_planner=True,
        local_se2_recovery_adapter=True,
        local_se2_recovery_canonical_features=True,
        local_se2_contact_rebel_adapter=True,
        local_se2_contact_rebel_min_center_x_m=0.14,
        local_se2_exit_refinement_adapter=True,
        local_se2_exit_refinement_min_center_x_m=0.17,
    )
    policy.eval()
    edges = radius_edges(cfg, device="cpu")

    def hit_observation(centre_x: float) -> torch.Tensor:
        state = torch.tensor([[centre_x, -0.24, 1.20, 0.0, 0.0, 0.0]])
        scenario = single_slit_scenario(
            1, cfg, initial_state=state, visible_aperture_m=0.720
        )
        context = initial_single_slit_contact_observation_context_v1(state, cfg)
        result = single_slit_contact_observations_v1(
            state, scenario, cfg, context
        )
        result[..., SINGLE_SLIT_CONTACT_LAST_HIT_INDEX_V1] = 1.0
        return result

    with torch.no_grad():
        policy.local_se2_exit_refinement_head.bias.fill_(0.35)
        below = policy(
            hit_observation(0.16), edges, policy.init_hidden(1, "cpu")
        )
        inside = policy(
            hit_observation(0.19), edges, policy.init_hidden(1, "cpu")
        )
    assert not bool(
        torch.count_nonzero(below.aux[LOCAL_SE2_RECOVERY_PLAN_DELTA_AUX_KEY])
    )
    assert bool(
        torch.count_nonzero(inside.aux[LOCAL_SE2_RECOVERY_PLAN_DELTA_AUX_KEY])
    )


def test_approach_refinement_is_contact_gated_and_spatially_supported() -> None:
    cfg = single_slit_env_config()
    policy = build_policy(
        "sheaf_gbp",
        cfg,
        observation_schema=SINGLE_SLIT_CONTACT_OBSERVATION_SCHEMA_V1,
        correction_sheaf=True,
        persistent_recovery_state=True,
        local_se2_planner=True,
        local_se2_recovery_adapter=True,
        local_se2_recovery_canonical_features=True,
        local_se2_contact_rebel_adapter=True,
        local_se2_exit_refinement_adapter=True,
        local_se2_approach_refinement_adapter=True,
        local_se2_approach_refinement_min_center_x_m=-0.30,
    )
    policy.eval()
    edges = radius_edges(cfg, device="cpu")

    def observation(centre_x: float, hit: bool) -> torch.Tensor:
        state = torch.tensor([[centre_x, 0.11, 0.95, 0.0, 0.0, 0.0]])
        scenario = single_slit_scenario(
            1, cfg, initial_state=state, visible_aperture_m=0.720
        )
        context = initial_single_slit_contact_observation_context_v1(state, cfg)
        result = single_slit_contact_observations_v1(state, scenario, cfg, context)
        result[..., SINGLE_SLIT_CONTACT_LAST_HIT_INDEX_V1] = float(hit)
        return result

    with torch.no_grad():
        policy.local_se2_approach_refinement_head.bias.fill_(0.35)
        def repeated(centre_x: float, hit: bool) -> object:
            hidden = policy.init_hidden(1, "cpu")
            output = None
            for _ in range(9):
                output = policy(observation(centre_x, hit), edges, hidden)
                hidden = output.hidden
            assert output is not None
            return output

        inactive = repeated(-0.18, False)
        below = repeated(-0.32, True)
        inside = repeated(-0.18, True)
    assert not bool(
        torch.count_nonzero(inactive.aux[LOCAL_SE2_RECOVERY_PLAN_DELTA_AUX_KEY])
    )
    assert not bool(
        torch.count_nonzero(below.aux[LOCAL_SE2_RECOVERY_PLAN_DELTA_AUX_KEY])
    )
    assert bool(
        torch.count_nonzero(inside.aux[LOCAL_SE2_RECOVERY_PLAN_DELTA_AUX_KEY])
    )


def test_deep_stuck_refinement_is_structurally_inactive_through_nine_hits() -> None:
    cfg = single_slit_env_config()
    policy = build_policy(
        "sheaf_gbp",
        cfg,
        observation_schema=SINGLE_SLIT_CONTACT_OBSERVATION_SCHEMA_V1,
        correction_sheaf=True,
        persistent_recovery_state=True,
        local_se2_planner=True,
        local_se2_recovery_adapter=True,
        local_se2_recovery_canonical_features=True,
        local_se2_contact_rebel_adapter=True,
        local_se2_exit_refinement_adapter=True,
        local_se2_approach_refinement_adapter=True,
        local_se2_approach_refinement_stuck_steps=9,
        local_se2_deep_stuck_refinement_adapter=True,
        local_se2_deep_stuck_refinement_stuck_steps=10,
    )
    policy.eval()
    edges = radius_edges(cfg, device="cpu")
    state = torch.tensor([[-0.18, 0.11, 0.95, 0.0, 0.0, 0.0]])
    scenario = single_slit_scenario(
        1, cfg, initial_state=state, visible_aperture_m=0.720
    )
    context = initial_single_slit_contact_observation_context_v1(state, cfg)
    observation = single_slit_contact_observations_v1(
        state, scenario, cfg, context
    )
    observation[..., SINGLE_SLIT_CONTACT_LAST_HIT_INDEX_V1] = 1.0
    with torch.no_grad():
        policy.local_se2_deep_stuck_refinement_head.bias.fill_(0.35)
        hidden = policy.init_hidden(1, "cpu")
        ninth = None
        for tick in range(10):
            output = policy(observation, edges, hidden)
            hidden = output.hidden
            if tick == 8:
                ninth = output
        tenth = output
    assert ninth is not None
    assert not bool(
        torch.count_nonzero(ninth.aux[LOCAL_SE2_RECOVERY_PLAN_DELTA_AUX_KEY])
    )
    assert bool(
        torch.count_nonzero(tenth.aux[LOCAL_SE2_RECOVERY_PLAN_DELTA_AUX_KEY])
    )


def test_retry_refinement_is_structurally_inactive_through_ten_hits() -> None:
    cfg = single_slit_env_config()
    policy = build_policy(
        "sheaf_gbp",
        cfg,
        observation_schema=SINGLE_SLIT_CONTACT_OBSERVATION_SCHEMA_V1,
        correction_sheaf=True,
        persistent_recovery_state=True,
        local_se2_planner=True,
        local_se2_recovery_adapter=True,
        local_se2_recovery_canonical_features=True,
        local_se2_contact_rebel_adapter=True,
        local_se2_exit_refinement_adapter=True,
        local_se2_approach_refinement_adapter=True,
        local_se2_approach_refinement_stuck_steps=9,
        local_se2_deep_stuck_refinement_adapter=True,
        local_se2_deep_stuck_refinement_stuck_steps=10,
        local_se2_retry_refinement_thresholds=(11,),
    )
    policy.eval()
    edges = radius_edges(cfg, device="cpu")
    state = torch.tensor([[-0.18, 0.11, 0.95, 0.0, 0.0, 0.0]])
    scenario = single_slit_scenario(
        1, cfg, initial_state=state, visible_aperture_m=0.720
    )
    context = initial_single_slit_contact_observation_context_v1(state, cfg)
    observation = single_slit_contact_observations_v1(
        state, scenario, cfg, context
    )
    observation[..., SINGLE_SLIT_CONTACT_LAST_HIT_INDEX_V1] = 1.0
    with torch.no_grad():
        policy.local_se2_retry_refinement_heads[0].bias.fill_(0.35)
        hidden = policy.init_hidden(1, "cpu")
        tenth = None
        for tick in range(11):
            output = policy(observation, edges, hidden)
            hidden = output.hidden
            if tick == 9:
                tenth = output
        eleventh = output
    assert tenth is not None
    assert not bool(
        torch.count_nonzero(tenth.aux[LOCAL_SE2_RECOVERY_PLAN_DELTA_AUX_KEY])
    )
    assert bool(
        torch.count_nonzero(eleventh.aux[LOCAL_SE2_RECOVERY_PLAN_DELTA_AUX_KEY])
    )


def test_analytic_phase_latch_is_source_null_then_irreversible_reentry() -> None:
    cfg = single_slit_env_config()
    policy = build_policy(
        "sheaf_gbp",
        cfg,
        observation_schema=SINGLE_SLIT_CONTACT_OBSERVATION_SCHEMA_V1,
        correction_sheaf=True,
        persistent_recovery_state=True,
        local_se2_planner=True,
        local_se2_recovery_adapter=True,
        local_se2_analytic_phase_latch=True,
        local_se2_reentry_center_x_threshold_m=-0.16,
    )
    scenario = single_slit_scenario(1, cfg)
    context = initial_single_slit_contact_observation_context_v1(
        scenario.state0, cfg
    )
    obs = single_slit_contact_observations_v1(
        scenario.state0, scenario, cfg, context
    )
    previous = torch.zeros(1, cfg.n_agents, policy.arm_cfg.persistent_recovery_dim)

    inactive_state, inactive, inactive_logits = (
        policy.update_persistent_recovery_state(obs, previous)
    )
    assert not bool(inactive.any())
    assert not bool(torch.count_nonzero(inactive_state))
    assert not bool(torch.count_nonzero(inactive_logits))

    def with_centre_x(value: float, *, hit: bool) -> torch.Tensor:
        result = obs.clone()
        sine = result[..., 0]
        cosine = result[..., 1]
        rotated_x = cosine * result[..., 2] - sine * result[..., 3]
        result[..., 4] = (value + rotated_x) / 2.6
        result[..., SINGLE_SLIT_CONTACT_LAST_HIT_INDEX_V1] = float(hit)
        return result

    retreat_state, retreat_active, retreat_logits = (
        policy.update_persistent_recovery_state(
            with_centre_x(-0.10, hit=True), previous
        )
    )
    assert bool(retreat_active.all())
    assert bool((retreat_state[..., 0] == -1.0).all())
    assert bool((retreat_logits.argmax(dim=-1) == 2).all())

    reentry_state, reentry_active, reentry_logits = (
        policy.update_persistent_recovery_state(
            with_centre_x(-0.17, hit=False), retreat_state
        )
    )
    assert bool(reentry_active.all())
    assert bool((reentry_state[..., 0] == 1.0).all())
    assert bool((reentry_logits.argmax(dim=-1) == 3).all())

    latched_state, _, latched_logits = policy.update_persistent_recovery_state(
        with_centre_x(0.20, hit=False), reentry_state
    )
    assert bool((latched_state[..., 0] == 1.0).all())
    assert bool((latched_logits.argmax(dim=-1) == 3).all())
