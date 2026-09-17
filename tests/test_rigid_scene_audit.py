from dataclasses import replace

import numpy as np
import pytest
import torch

from piano_movers.config import EnvConfig
from piano_movers.env import PianoBatch, collision_clearance_margin, step_dynamics
from piano_movers.local_skill_teacher import repeat_scene
from piano_movers.payload_shapes import payload_rectangles_for_config
from piano_movers.rigid_scene_audit import (
    RigidFrame, WorldRectangleScene, rectangle, world_scene_from_legacy, certify_world_path,
)


def test_frame_roundtrip_pose_twist_and_force_have_correct_semantics():
    f = RigidFrame(.731, (1.3, -2.7))
    states = np.array([[.8, -.4, .3, .17, -.24, .7]])
    np.testing.assert_allclose(f.inverse().states(f.states(states)), states, atol=2e-15)
    np.testing.assert_allclose(f.vectors([[0, 0]]), [[0, 0]], atol=0)
    np.testing.assert_allclose(f.states(states)[:, 5], states[:, 5], atol=0)
    with pytest.raises(ValueError):
        RigidFrame(float("nan"))


@pytest.mark.parametrize("letter", ["T", "A", "L", "S", "Z"])
def test_independent_obb_sat_matches_legacy_and_is_rigid_invariant(letter):
    cfg = replace(EnvConfig(), payload_shape=letter, world_x=4., world_y=1.6)
    q = torch.zeros(1, 6, dtype=torch.float64)
    scene = PianoBatch(state0=q, source=q[:, :3], goal=q[:, :3],
                       wall_x=torch.tensor([[-1.2, .4, 2.]]),
                       gap_y=torch.tensor([[.2, -.3, 0.]]),
                       gap_half=torch.tensor([[.36, .5, 3.]]),
                       obstacle_rects=torch.tensor([[[-2., -1.8, .8, 1.2]]]))
    body = payload_rectangles_for_config(cfg, torch.device("cpu")).double().numpy()
    independent = world_scene_from_legacy(scene, cfg, body)
    rng = np.random.default_rng(729)
    poses = rng.uniform([-4.5, -2., -np.pi], [4.5, 2., np.pi], (113, 3))
    states = torch.from_numpy(np.pad(poses, ((0, 0), (0, 3))))
    old = collision_clearance_margin(states, repeat_scene(scene, len(states), states), cfg).numpy()
    new = independent.clearance(poses)
    np.testing.assert_allclose(new, old, atol=3e-14, rtol=1e-12)
    frame = RigidFrame(.731, (3.1, -.83))
    rotated = independent.transformed(frame).clearance(frame.states(poses))
    np.testing.assert_allclose(rotated, new, atol=3e-14, rtol=1e-12)


def test_compound_not_convex_hull_and_recursive_guard_rejects_tunnelling():
    # Obstacle lies in the open notch of an L, inside its convex hull.
    body = np.stack((rectangle(-.5, -.3, -.5, .5), rectangle(-.5, .5, -.5, -.3)))
    scene = WorldRectangleScene(body, rectangle(-.1, .1, -.1, .1)[None],
                                rectangle(-4, 4, -4, 4))
    assert scene.clearance(np.zeros((1, 3)))[0] > 0
    small = rectangle(-.1, .1, -.1, .1)[None]
    wall = rectangle(-.05, .05, -2., 2.)[None]
    scene = WorldRectangleScene(small, wall, rectangle(-4, 4, -4, 4))
    path = np.array([[-1., 0., 0.], [1., 0., 0.]])
    assert (scene.clearance(path) > 0).all()
    for frame in (RigidFrame(), RigidFrame(1.17, (1.3, -2.1))):
        cert = certify_world_path(frame.states(path), scene.transformed(frame))
        assert not cert.passed and cert.reason == "interior_clearance_violation"
    safe = np.array([[-1., 0., 0.], [-1.2, .1, 1.]])
    assert certify_world_path(safe, scene).passed


def test_real_dynamics_rotates_forces_velocity_and_torque_consistently():
    cfg = replace(EnvConfig(), n_agents=12)
    rng = np.random.default_rng(291)
    states = rng.normal(0, .3, (37, 6))
    forces = rng.normal(0, 1., (37, 12, 2))
    frame = RigidFrame(2.731, (.63, -3.3))
    local = step_dynamics(torch.from_numpy(states), torch.from_numpy(forces), cfg).numpy()
    world = step_dynamics(torch.from_numpy(frame.states(states)),
                          torch.from_numpy(frame.vectors(forces)), cfg).numpy()
    np.testing.assert_allclose(world, frame.states(local), atol=3e-14, rtol=1e-12)


def test_empty_obstacles_oriented_arena_and_bad_rectangle():
    body = rectangle(-.1, .1, -.1, .1)[None]
    scene = WorldRectangleScene(body, np.empty((0, 4, 2)), rectangle(-1, 1, -2, 2))
    assert scene.clearance(np.zeros((1, 3)))[0] == pytest.approx(.9)
    frame = RigidFrame(.932, (4., -5.))
    assert scene.transformed(frame).clearance(frame.states(np.zeros((1, 3))))[0] == pytest.approx(.9)
    with pytest.raises(ValueError):
        WorldRectangleScene(body, np.array([[[0, 0], [1, 0], [2, 1], [0, 1]]]), scene.arena)


def test_persistent_world_physics_produces_consistent_local_contact_feedback():
    from piano_movers.framed_world_physics import FramedWorldPhysics
    from piano_movers.single_slit_contact_observation import single_slit_contact_observation_context_from_result_v1
    from piano_movers.single_slit_feedback import single_slit_env_config

    cfg = replace(single_slit_env_config(), dt=1., linear_damping=0., angular_damping=0.)
    state = torch.tensor([[-1., .9, 0., 2., 0., 0.]])
    scene = PianoBatch(state0=state, source=state[:, :3], goal=state[:, :3],
                       wall_x=torch.tensor([[0.]]), gap_y=torch.tensor([[0.]]), gap_half=torch.tensor([[.36]]))
    forces = torch.zeros(1, cfg.n_agents, 2)
    physics = FramedWorldPhysics(RigidFrame(.731, (3., -.5)))
    result, cert = physics(state, forces, scene, cfg)
    assert result.attempted_hit.item() and not cert.passed
    torch.testing.assert_close(result.resolved_state[:, :3], state[:, :3], atol=1e-6, rtol=0)
    assert result.resolved_state[:, 3:].abs().sum().item() == 0
    feedback = single_slit_contact_observation_context_from_result_v1(result, forces, cfg)
    assert feedback.last_hit.item()
    result2, cert2 = physics(result.resolved_state, forces, scene, cfg)
    assert cert2.passed and not result2.attempted_hit.item()
    assert len(physics.states) == 3


@pytest.mark.parametrize("guard", [False, True])
def test_default_engine_physics_seam_is_bitwise_legacy(guard):
    from piano_movers.local_skill_physics import certified_contact_step
    from piano_movers.multi_waypoint_demo import MultiWaypointEngine, SkillCompositionOptions
    from piano_movers.single_slit_contact_feedback import single_slit_contact_feedback_step_v1
    from piano_movers.single_slit_feedback import single_slit_env_config

    cfg = single_slit_env_config()
    state = torch.tensor([[-1., 0., .3, .1, -.02, .04]])
    scene = PianoBatch(state0=state, source=state[:, :3], goal=state[:, :3],
                       wall_x=torch.tensor([[0.]]), gap_y=torch.tensor([[0.]]), gap_half=torch.tensor([[.36]]))
    forces = torch.full((1, cfg.n_agents, 2), .03)
    engine = MultiWaypointEngine.__new__(MultiWaypointEngine)
    engine.options = SkillCompositionOptions(continuous_contact_guard=guard)
    expected, expected_cert = (certified_contact_step(state, forces, scene, cfg) if guard else
                              (single_slit_contact_feedback_step_v1(state, forces, scene, cfg), None))
    actual, actual_cert = engine._contact_step(state, forces, scene, cfg)
    assert actual_cert == expected_cert
    for name in ("resolved_state", "attempted_state", "attempted_hit",
                 "attempted_clearance_m", "resolved_clearance_m"):
        assert torch.equal(getattr(actual, name), getattr(expected, name))
