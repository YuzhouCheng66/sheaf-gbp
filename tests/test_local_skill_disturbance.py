from dataclasses import replace

import pytest
import torch

from piano_movers.env import PianoBatch, collision_clearance_margin
from piano_movers.local_skill_disturbance import ContactImpulseOptions, SingleContactImpulse
from piano_movers.local_skill_physics import certified_contact_step
from piano_movers.single_slit_feedback import single_slit_env_config
from piano_movers.single_slit_contact_observation import single_slit_contact_observation_context_from_result_v1


def setup_case():
    cfg = replace(single_slit_env_config(), dt=.08, linear_damping=0.)
    state = torch.tensor([[-.54, .8, 0., 0., 0., 0.]])
    scene = PianoBatch(state0=state, source=state[:, :3], goal=state[:, :3],
                       wall_x=torch.tensor([[0.]]), gap_y=torch.tensor([[0.]]), gap_half=torch.tensor([[.36]]))
    return cfg, state, scene, torch.zeros(1, cfg.n_agents, 2)


def test_disabled_impulse_is_bitwise_original_physics():
    cfg, q, scene, force = setup_case()
    impulse = SingleContactImpulse(ContactImpulseOptions(enabled=False))
    a, ca = impulse(q, force, scene, cfg)
    b, cb = certified_contact_step(q, force, scene, cfg)
    assert ca == cb and impulse.event is None
    for name in ("attempted_state", "resolved_state", "attempted_hit", "attempted_clearance_m", "resolved_clearance_m"):
        assert torch.equal(getattr(a, name), getattr(b, name))


def test_near_wall_impulse_is_bounded_one_shot_and_feedback_is_real():
    cfg, q, scene, force = setup_case()
    # Choose a positive 0.015 m gap using actual compound-body support.
    current = collision_clearance_margin(q, scene, cfg)[0]
    q[:, 0] += current - .015
    impulse = SingleContactImpulse()
    result, _ = impulse(q, force, scene, cfg)
    assert impulse.event is not None
    delta = torch.tensor(impulse.event["velocity_impulse_world_mps"])
    assert torch.linalg.vector_norm(delta).item() == pytest.approx(.65)
    assert bool(result.attempted_hit[0])
    assert torch.equal(result.resolved_state[:, :3], q[:, :3])
    assert torch.equal(result.resolved_state[:, 3:], torch.zeros(1, 3))
    feedback = single_slit_contact_observation_context_from_result_v1(result, force, cfg)
    assert feedback.last_hit.item()
    assert feedback.rejected_handle_motion_normalized.abs().max().item() > 0
    event = dict(impulse.event)
    nxt, _ = impulse(result.resolved_state, force, scene, cfg)
    assert not nxt.attempted_hit.item()
    assert impulse.event == event
    assert torch.equal(force, torch.zeros_like(force))


def test_far_state_does_not_fabricate_contact_and_invalid_options_fail():
    cfg, q, scene, force = setup_case()
    q[:, 0] = -1.2
    impulse = SingleContactImpulse()
    result, _ = impulse(q, force, scene, cfg)
    assert impulse.event is None and not result.attempted_hit.item()
    with pytest.raises(ValueError):
        ContactImpulseOptions(delta_speed_mps=float("nan"))


def test_early_impulse_changes_real_pose_before_any_possible_rejection():
    cfg, q, scene, force = setup_case()
    current = collision_clearance_margin(q, scene, cfg)[0]
    q[:, 0] += current - .115
    impulse = SingleContactImpulse(ContactImpulseOptions(trigger_clearance_m=.12, delta_omega_radps=.8))
    result, cert = impulse(q, force, scene, cfg)
    assert impulse.event is not None and cert.passed
    assert not result.attempted_hit.item()
    assert float((result.resolved_state[:, :2] - q[:, :2]).norm()) > .04
    assert float((result.resolved_state[:, 2] - q[:, 2]).abs()) > .04
    assert float(impulse.event["angular_velocity_impulse_radps"]) == .8
