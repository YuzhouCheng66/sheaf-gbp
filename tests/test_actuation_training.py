import torch

from piano_movers.actuation_training import replay_physical_actuation
from piano_movers.env import radius_edges
from piano_movers.single_slit_0720_bc import build_policy
from piano_movers.single_slit_contact_observation import (
    SINGLE_SLIT_CONTACT_OBSERVATION_SCHEMA_V1,
    initial_single_slit_contact_observation_context_v1,
    single_slit_contact_observations_v1,
)
from piano_movers.single_slit_feedback import single_slit_env_config, single_slit_scenario


def test_offline_actuation_matches_original_force_messages_and_authority_gradient():
    torch.manual_seed(73)
    cfg = single_slit_env_config()
    policy = build_policy(
        "sheaf_gbp", cfg, observation_schema=SINGLE_SLIT_CONTACT_OBSERVATION_SCHEMA_V1,
        correction_sheaf=True, persistent_recovery_state=True, local_se2_planner=True,
        local_se2_analytic_total_wrench=True, local_se2_distributed_wrench_control=True,
        local_se2_distributed_next_se2_control=True, local_se2_distributed_grasp_allreduce=True,
        local_se2_terminal_attractor=True, local_se2_feasible_corridor_attractor=True,
        local_se2_route_free_geometry_planner=True,
    ).eval()
    state = torch.tensor([[-1.1, .08, .3, .01, -.02, .03]])
    scene = single_slit_scenario(1, cfg, initial_state=state)
    context = initial_single_slit_contact_observation_context_v1(state, cfg)
    obs = single_slit_contact_observations_v1(state, scene, cfg, context)
    with torch.no_grad():
        source = policy(obs, radius_edges(cfg, device="cpu"), policy.init_hidden(1, "cpu"),
                        trace_messages=True)
    private = source.aux["local_se2_plan_mean"].detach().clone().requires_grad_()
    active = source.aux["local_se2_terminal_attractor_active"]
    replay = replay_physical_actuation(policy, obs, private, source.aux["local_se2_plan_used"],
                                       active, trace_messages=True)
    torch.testing.assert_close(replay.forces, source.forces, atol=2e-5, rtol=2e-5)
    assert replay.message_trace.wire_float_count == source.aux["gbp_message_trace"].wire_float_count
    torch.testing.assert_close(replay.message_trace.natural, source.aux["gbp_message_trace"].natural,
                               atol=2e-5, rtol=2e-5)
    weights = torch.linspace(.1, 1., replay.forces.numel()).reshape_as(replay.forces)
    gradient = torch.autograd.grad((replay.forces * weights).sum(), private)[0]
    assert gradient[..., :3].count_nonzero() == 0
    assert gradient[..., 6:].count_nonzero() == 0
    assert torch.where(active, gradient, 0.).count_nonzero() == 0
    assert (~active).any() and gradient[..., 3:6].abs().max() > 1e-6
    index = int(gradient.abs().flatten().argmax())
    numerical = []
    epsilon = 1e-3
    for direction in (-1., 1.):
        changed = private.detach().clone()
        changed.flatten()[index] += direction * epsilon
        with torch.no_grad():
            output = replay_physical_actuation(policy, obs, changed,
                                               source.aux["local_se2_plan_used"], active)
        numerical.append((output.forces * weights).sum())
    torch.testing.assert_close((numerical[1] - numerical[0]) / (2 * epsilon),
                               gradient.flatten()[index], atol=1e-3, rtol=2e-2)

    # Independent contact-authority control must preserve both mask and GBP
    # precision. It is NOT the old final-plan override (which clears the mask).
    delta = torch.tensor([.02, -.01, .04]).expand(1, cfg.n_agents, 3).clone().requires_grad_()
    actual = policy(obs, radius_edges(cfg, device="cpu"), policy.init_hidden(1, "cpu"),
                    trace_messages=True, local_se2_twist_residual=delta)
    replay_delta = replay_physical_actuation(policy, obs, private.detach(),
        source.aux["local_se2_plan_used"], active, trace_messages=True, twist_residual=delta)
    torch.testing.assert_close(actual.forces, replay_delta.forces, atol=2e-5, rtol=2e-5)
    assert torch.equal(actual.aux["local_se2_terminal_attractor_active"], active)
    assert actual.aux["gbp_message_trace"].wire_float_count == replay_delta.message_trace.wire_float_count
    residual_grad = torch.autograd.grad((replay_delta.forces * weights).sum(), delta)[0]
    assert torch.where(active, residual_grad, 0.).abs().max() > 1e-6
    zero = policy(obs, radius_edges(cfg, device="cpu"), policy.init_hidden(1, "cpu"),
                  local_se2_twist_residual=torch.zeros_like(delta))
    assert torch.equal(zero.forces, source.forces)
