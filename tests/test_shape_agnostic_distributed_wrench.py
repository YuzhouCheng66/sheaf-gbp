from __future__ import annotations

from dataclasses import replace

import torch

import piano_movers.models as models_module
from piano_movers.config import HANDLE_LAYOUT_PARAMETRIC_T_V3
from piano_movers.env import radius_edges
from piano_movers.models import (
    CORRECTION_SECTION_AUX_KEY,
    LOCAL_SE2_DISTRIBUTED_GRASP_MESSAGE_AUX_KEY,
    LOCAL_SE2_DISTRIBUTED_GRASP_SETUP_TRANSMITTED_AUX_KEY,
    LOCAL_SE2_DISTRIBUTED_GRASP_STATS_AUX_KEY,
    LOCAL_SE2_DISTRIBUTED_RECOVERY_PRE_MEAN_AUX_KEY,
    LOCAL_SE2_ANALYTIC_WRENCH_PROPOSAL_AUX_KEY,
    LOCAL_SE2_PLAN_MEAN_AUX_KEY,
    LOCAL_SE2_RECOVERY_MODE_LOGITS_AUX_KEY,
    LOCAL_SE2_RECOVERY_SELECTED_MODE_AUX_KEY,
)
from piano_movers.single_slit_0720_bc import build_policy
from piano_movers.single_slit_0720_eval import _communication_dimensions
from piano_movers.single_slit_contact_observation import (
    SINGLE_SLIT_CONTACT_OBSERVATION_SCHEMA_V1,
    initial_single_slit_contact_observation_context_v1,
    single_slit_contact_observations_v1,
)
from piano_movers.single_slit_feedback import (
    single_slit_env_config,
    single_slit_scenario,
)
from piano_movers.single_slit_local_planner import (
    local_se2_inverse_dynamics_wrench_proposals,
)


def _policy_and_input():
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
    )
    state = torch.tensor([[-1.05, 0.03, 0.08, 0.02, -0.01, 0.01]])
    scenario = single_slit_scenario(
        1, cfg, initial_state=state, visible_aperture_m=0.720
    )
    context = initial_single_slit_contact_observation_context_v1(state, cfg)
    obs = single_slit_contact_observations_v1(state, scenario, cfg, context)
    edges = radius_edges(cfg, device=obs.device)
    return policy, obs, edges


def test_distributed_wrench_action_path_never_queries_global_grasp(monkeypatch) -> None:
    policy, obs, edges = _policy_and_input()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("deployment action queried the global handle layout")

    monkeypatch.setattr(models_module, "contact_points", forbidden)
    output = policy(
        obs,
        edges,
        policy.init_hidden(1, obs.device),
        trace_messages=True,
    )

    assert output.forces.shape == (1, policy.env_cfg.n_agents, 2)
    assert policy.communication_floats_per_directed_edge_round() == 6
    trace = output.aux["gbp_message_trace"]
    assert trace.natural.shape[-1] == 3
    assert trace.precision.shape[-1] == 3
    assert output.aux["communication_bundle_wire_trace"].shape[-1] == 6
    torch.testing.assert_close(
        output.aux[LOCAL_SE2_ANALYTIC_WRENCH_PROPOSAL_AUX_KEY],
        output.aux["local_se2_plan_used"][..., :3],
        atol=0.0,
        rtol=0.0,
    )


def test_post_gbp_force_loss_reaches_private_encoder_and_local_decoder() -> None:
    policy, obs, edges = _policy_and_input()
    output = policy(obs, edges, policy.init_hidden(1, obs.device))
    section = output.aux[CORRECTION_SECTION_AUX_KEY]
    # Both terms are post-GBP.  The section term guarantees a direct signal to
    # the private proposal encoder; force imitation additionally trains the
    # shared local decoder.
    loss = output.forces.square().mean() + section.square().mean()
    loss.backward()

    planner_grad = policy.local_se2_plan_mean_head.weight.grad
    decoder_grad = policy.local_se2_distributed_force_decoder[-1].weight.grad
    assert planner_grad is not None and bool(torch.isfinite(planner_grad).all())
    assert decoder_grad is not None and bool(torch.isfinite(decoder_grad).all())
    assert float(planner_grad.abs().sum()) > 0.0
    assert float(decoder_grad.abs().sum()) > 0.0


def test_distributed_next_se2_preserves_plan_semantics_before_gbp() -> None:
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
    ).eval()
    state = torch.tensor([[-1.05, 0.03, 0.08, 0.02, -0.01, 0.01]])
    scenario = single_slit_scenario(
        1, cfg, initial_state=state, visible_aperture_m=0.720
    )
    context = initial_single_slit_contact_observation_context_v1(state, cfg)
    obs = single_slit_contact_observations_v1(state, scenario, cfg, context)
    edges = radius_edges(cfg, device=obs.device)
    output = policy(obs, edges, policy.init_hidden(1, obs.device))
    plan = output.aux["local_se2_plan_used"]
    expected = local_se2_inverse_dynamics_wrench_proposals(obs, plan, cfg)

    torch.testing.assert_close(
        output.aux[LOCAL_SE2_ANALYTIC_WRENCH_PROPOSAL_AUX_KEY],
        expected,
        atol=0.0,
        rtol=0.0,
    )
    assert not torch.equal(expected, plan[..., :3])
    assert policy.communication_floats_per_directed_edge_round() == 6


def test_geometry_evidence_precision_is_exactly_null_on_canonical_t() -> None:
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
        local_se2_terminal_attractor=True,
        local_se2_feasible_corridor_attractor=True,
    ).eval()
    state = torch.tensor([[-0.55, 0.01, 0.04, 0.0, 0.0, 0.0]])
    scenario = single_slit_scenario(
        1, cfg, initial_state=state, visible_aperture_m=0.720
    )
    context = initial_single_slit_contact_observation_context_v1(state, cfg)
    obs = single_slit_contact_observations_v1(state, scenario, cfg, context)
    edges = radius_edges(cfg, device=obs.device)
    hidden = policy.init_hidden(1, obs.device)

    unboosted = policy(obs, edges, hidden)
    policy.arm_cfg = replace(
        policy.arm_cfg,
        local_se2_geometry_evidence_precision_boost=4.0,
    )
    boosted = policy(obs, edges, hidden)

    torch.testing.assert_close(boosted.forces, unboosted.forces, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        boosted.aux["correction_stalk_precision"],
        unboosted.aux["correction_stalk_precision"],
        atol=0.0,
        rtol=0.0,
    )


def test_geometry_evidence_precision_activates_for_a_rescaled_t() -> None:
    cfg = replace(
        single_slit_env_config(),
        handle_layout=HANDLE_LAYOUT_PARAMETRIC_T_V3,
        t_total_width_m=0.930,
        t_total_height_m=0.920,
        t_stem_width_m=0.160,
        t_cap_thickness_m=0.152,
    )
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
        local_se2_terminal_attractor=True,
        local_se2_feasible_corridor_attractor=True,
    ).eval()
    state = torch.tensor([[-0.55, 0.0, 0.0, 0.0, 0.0, 0.0]])
    scenario = single_slit_scenario(
        1, cfg, initial_state=state, visible_aperture_m=0.720
    )
    context = initial_single_slit_contact_observation_context_v1(state, cfg)
    obs = single_slit_contact_observations_v1(state, scenario, cfg, context)
    edges = radius_edges(cfg, device=obs.device)
    hidden = policy.init_hidden(1, obs.device)
    unboosted = policy(obs, edges, hidden)
    policy.arm_cfg = replace(
        policy.arm_cfg,
        local_se2_geometry_evidence_precision_boost=4.0,
    )
    boosted = policy(obs, edges, hidden)

    active = boosted.aux["local_se2_terminal_attractor_active"]
    assert bool(active.all())
    torch.testing.assert_close(
        boosted.aux["correction_stalk_precision"][..., :3],
        4.0 * unboosted.aux["correction_stalk_precision"][..., :3],
        atol=0.0,
        # This is the post-GBP posterior, so fixed pair-factor precision adds
        # a tiny non-scaling term on top of the exactly 4x unary evidence.
        rtol=1.0e-3,
    )


def test_route_free_canonical_t_geometry_is_allowed_to_raise_its_precision() -> None:
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
        local_se2_terminal_attractor=True,
        local_se2_feasible_corridor_attractor=True,
    ).eval()
    policy.arm_cfg = replace(
        policy.arm_cfg,
        local_se2_route_free_geometry_planner=True,
    )
    state = torch.tensor([[-0.55, 0.01, 0.04, 0.0, 0.0, 0.0]])
    scenario = single_slit_scenario(
        1, cfg, initial_state=state, visible_aperture_m=0.720
    )
    context = initial_single_slit_contact_observation_context_v1(state, cfg)
    obs = single_slit_contact_observations_v1(state, scenario, cfg, context)
    edges = radius_edges(cfg, device=obs.device)
    hidden = policy.init_hidden(1, obs.device)
    unboosted = policy(obs, edges, hidden)
    policy.arm_cfg = replace(
        policy.arm_cfg,
        local_se2_geometry_evidence_precision_boost=4.0,
    )
    output = policy(obs, edges, hidden)

    assert bool(output.aux["local_se2_terminal_attractor_active"].any())
    torch.testing.assert_close(
        output.aux["correction_stalk_precision"][..., :3],
        4.0 * unboosted.aux["correction_stalk_precision"][..., :3],
        atol=0.0,
        rtol=1.0e-3,
    )


def test_distributed_wrench_config_rejects_shape_lookup_modules() -> None:
    cfg = single_slit_env_config()
    try:
        build_policy(
            "sheaf_gbp",
            cfg,
            observation_schema=SINGLE_SLIT_CONTACT_OBSERVATION_SCHEMA_V1,
            correction_sheaf=True,
            persistent_recovery_state=True,
            local_se2_planner=True,
            local_se2_shape_signature=True,
            local_se2_analytic_total_wrench=True,
            local_se2_distributed_wrench_control=True,
        )
    except ValueError as error:
        assert "forbids global geometry" in str(error)
    else:
        raise AssertionError("shape-conditioned distributed policy was accepted")


def test_distributed_recovery_role_exits_outside_local_wall_support() -> None:
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
        local_se2_distributed_recovery_adapter=True,
        local_se2_distributed_recovery_local_support=True,
        local_se2_distributed_recovery_wall_range_m=0.45,
    )
    previous = torch.ones(1, cfg.n_agents, policy.arm_cfg.persistent_recovery_dim)
    obs = torch.zeros(1, cfg.n_agents, 27)
    # Agent zero remains 0.105 m from its own visible wall; agent one is
    # 0.840 m away.  No other handle has a wall record.
    obs[:, 0:2, 16] = 1.0
    obs[:, 0, 10] = 0.10
    obs[:, 1, 10] = 0.80

    state, active, logits = policy.update_persistent_recovery_state(obs, previous)

    assert bool(active[0, 0, 0])
    assert not bool(active[0, 1:, 0].any())
    assert not bool(torch.count_nonzero(state[0, 1:]))
    assert not bool(torch.count_nonzero(logits[0, 1:]))

    # A real rejected rigid-body transition is mechanical evidence shared by
    # all attached handles for that tick, so it reactivates every row.
    obs[..., 22] = 1.0
    _state, hit_active, _logits = policy.update_persistent_recovery_state(obs, previous)
    assert bool(hit_active.all())


def test_maneuver_recovery_exposes_exact_pre_residual_plan_for_warm_start() -> None:
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
        local_se2_distributed_recovery_adapter=True,
        local_se2_distributed_recovery_mode_count=4,
    )
    state = torch.tensor([[-1.05, 0.03, 0.08, 0.02, -0.01, 0.01]])
    scenario = single_slit_scenario(
        1, cfg, initial_state=state, visible_aperture_m=0.720
    )
    context = initial_single_slit_contact_observation_context_v1(state, cfg)
    obs = single_slit_contact_observations_v1(state, scenario, cfg, context)
    edges = radius_edges(cfg, device=obs.device)
    output = policy(obs, edges, policy.init_hidden(1, obs.device))

    pre = output.aux[LOCAL_SE2_DISTRIBUTED_RECOVERY_PRE_MEAN_AUX_KEY]
    torch.testing.assert_close(
        pre,
        output.aux[LOCAL_SE2_PLAN_MEAN_AUX_KEY],
        atol=0.0,
        rtol=0.0,
    )
    assert output.aux[LOCAL_SE2_RECOVERY_MODE_LOGITS_AUX_KEY].shape == (
        1,
        cfg.n_agents,
        4,
    )
    assert output.aux[LOCAL_SE2_RECOVERY_SELECTED_MODE_AUX_KEY].shape == (
        1,
        cfg.n_agents,
    )


def test_local_recovery_support_requires_distributed_adapter() -> None:
    cfg = single_slit_env_config()
    try:
        build_policy(
            "sheaf_gbp",
            cfg,
            observation_schema=SINGLE_SLIT_CONTACT_OBSERVATION_SCHEMA_V1,
            correction_sheaf=True,
            persistent_recovery_state=True,
            local_se2_planner=True,
            local_se2_analytic_total_wrench=True,
            local_se2_distributed_wrench_control=True,
            local_se2_distributed_recovery_local_support=True,
        )
    except ValueError as error:
        assert "requires the recovery adapter" in str(error)
    else:
        raise AssertionError("local recovery support without an adapter was accepted")


def test_distributed_grasp_allreduce_uses_only_local_handle_rows(monkeypatch) -> None:
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
        local_se2_distributed_grasp_allreduce=True,
    ).eval()
    state = torch.tensor([[-1.05, 0.03, 0.21, 0.02, -0.01, 0.01]])
    scenario = single_slit_scenario(
        1, cfg, initial_state=state, visible_aperture_m=0.720
    )
    context = initial_single_slit_contact_observation_context_v1(state, cfg)
    obs = single_slit_contact_observations_v1(state, scenario, cfg, context)
    edges = radius_edges(cfg, device=obs.device)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("distributed allocation queried global handles")

    monkeypatch.setattr(models_module, "contact_points", forbidden)
    output = policy(
        obs,
        edges,
        policy.init_hidden(1, obs.device),
        trace_messages=True,
    )

    stats = output.aux[LOCAL_SE2_DISTRIBUTED_GRASP_STATS_AUX_KEY]
    trace = output.aux[LOCAL_SE2_DISTRIBUTED_GRASP_MESSAGE_AUX_KEY]
    assert stats.shape == (1, cfg.n_agents, 4)
    assert trace.shape == (
        cfg.n_agents - 1,
        1,
        2 * (cfg.n_agents - 1),
        4,
    )
    torch.testing.assert_close(
        stats[..., 0],
        torch.full_like(stats[..., 0], float(cfg.n_agents)),
        atol=1.0e-6,
        rtol=0.0,
    )
    # Exact tree sum-product gives every node the same sufficient statistic.
    torch.testing.assert_close(
        stats[:, 1:],
        stats[:, :1].expand(-1, cfg.n_agents - 1, -1),
        atol=2.0e-6,
        rtol=0.0,
    )
    assert bool(torch.isfinite(output.forces).all())
    assert bool((torch.linalg.vector_norm(output.forces, dim=-1) <= cfg.max_force + 1e-6).all())

    # A second control tick reuses the rigid payload-frame sufficient
    # statistics.  It emits no hidden grasp packets and produces the same
    # allocation for the same instantaneous input.
    cached = policy(
        obs,
        edges,
        policy.init_hidden(1, obs.device),
        trace_messages=True,
    )
    assert bool(output.aux[LOCAL_SE2_DISTRIBUTED_GRASP_SETUP_TRANSMITTED_AUX_KEY])
    assert not bool(cached.aux[LOCAL_SE2_DISTRIBUTED_GRASP_SETUP_TRANSMITTED_AUX_KEY])
    assert cached.aux[LOCAL_SE2_DISTRIBUTED_GRASP_MESSAGE_AUX_KEY].shape == (
        0,
        1,
        2 * (cfg.n_agents - 1),
        4,
    )
    torch.testing.assert_close(cached.forces, output.forces, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        cached.aux[LOCAL_SE2_DISTRIBUTED_GRASP_STATS_AUX_KEY],
        stats,
        atol=0.0,
        rtol=0.0,
    )

    communication = _communication_dimensions(policy, edges)
    expected_grasp_packets = 2 * (cfg.n_agents - 1) ** 2
    assert communication["grasp_packets_per_setup"] == expected_grasp_packets
    assert communication["grasp_payload_floats_per_setup"] == (
        4 * expected_grasp_packets
    )
    assert communication["grasp_packets_per_control_tick"] == 0
    assert communication["grasp_payload_floats_per_control_tick"] == 0
    assert communication["payload_floats_per_control_tick"] == (
        communication["gbp_payload_floats_per_control_tick"]
    )


def test_cached_body_grasp_statistics_match_fresh_reduction_after_rotation() -> None:
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
    ).eval()
    edges = radius_edges(cfg, device="cpu")
    hidden = policy.init_hidden(1, "cpu")

    def observation(theta: float) -> torch.Tensor:
        state = torch.tensor([[-0.82, 0.03, theta, 0.0, 0.0, 0.0]])
        scenario = single_slit_scenario(
            1, cfg, initial_state=state, visible_aperture_m=0.720
        )
        context = initial_single_slit_contact_observation_context_v1(state, cfg)
        return single_slit_contact_observations_v1(state, scenario, cfg, context)

    # Prime the payload-frame cache at one orientation, then consume it at a
    # different orientation.
    policy(observation(0.19), edges, hidden)
    rotated_obs = observation(-0.23)
    cached = policy(rotated_obs, edges, hidden, trace_messages=True)
    assert not bool(
        cached.aux[LOCAL_SE2_DISTRIBUTED_GRASP_SETUP_TRANSMITTED_AUX_KEY]
    )

    # Rebinding represents a fresh team/attachment and forces a new exact tree
    # reduction at the same rotated pose.
    policy.bind_environment(cfg)
    fresh = policy(rotated_obs, edges, hidden, trace_messages=True)
    assert bool(fresh.aux[LOCAL_SE2_DISTRIBUTED_GRASP_SETUP_TRANSMITTED_AUX_KEY])
    torch.testing.assert_close(cached.forces, fresh.forces, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        cached.aux[LOCAL_SE2_DISTRIBUTED_GRASP_STATS_AUX_KEY],
        fresh.aux[LOCAL_SE2_DISTRIBUTED_GRASP_STATS_AUX_KEY],
        atol=0.0,
        rtol=0.0,
    )
    sine = rotated_obs[0, :, 0]
    cosine = rotated_obs[0, :, 1]
    body_x = rotated_obs[0, :, 2]
    body_y = rotated_obs[0, :, 3]
    world_x = cosine * body_x - sine * body_y
    world_y = sine * body_x + cosine * body_y
    expected = torch.stack(
        (
            torch.tensor(float(cfg.n_agents)),
            world_x.sum(),
            world_y.sum(),
            (world_x.square() + world_y.square()).sum(),
        )
    )
    torch.testing.assert_close(
        fresh.aux[LOCAL_SE2_DISTRIBUTED_GRASP_STATS_AUX_KEY][0, 0],
        expected,
        atol=2.0e-6,
        rtol=0.0,
    )


def test_distributed_grasp_allreduce_requires_distributed_wrench() -> None:
    cfg = single_slit_env_config()
    try:
        build_policy(
            "sheaf_gbp",
            cfg,
            observation_schema=SINGLE_SLIT_CONTACT_OBSERVATION_SCHEMA_V1,
            correction_sheaf=True,
            persistent_recovery_state=True,
            local_se2_planner=True,
            local_se2_analytic_total_wrench=True,
            local_se2_distributed_grasp_allreduce=True,
        )
    except ValueError as error:
        assert "requires analytic distributed wrench control" in str(error)
    else:
        raise AssertionError("grasp all-reduce without distributed control was accepted")
