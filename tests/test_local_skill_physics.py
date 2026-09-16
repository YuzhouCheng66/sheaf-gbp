from dataclasses import replace

import torch

from piano_movers.cem_teacher import feasible_forces_for_absolute_wrench
from piano_movers.env import PianoBatch
from piano_movers.local_skill_physics import certified_contact_step
from piano_movers.single_slit_contact_feedback import single_slit_contact_feedback_step_v1
from piano_movers.single_slit_feedback import single_slit_env_config


def test_continuous_guard_rejects_tunnelling_that_sampled_physics_accepts():
    cfg = replace(
        single_slit_env_config(),
        dt=1.0,
        linear_damping=0.0,
        angular_damping=0.0,
        collision_substeps=1,
    )
    state = torch.tensor([[-1.0, 0.9, 0.0, 0.0, 0.0, 0.0]])
    scene = PianoBatch(
        state0=state,
        source=state[:, :3],
        goal=torch.tensor([[1.0, 0.9, 0.0]]),
        wall_x=torch.tensor([[0.0]]),
        gap_y=torch.tensor([[0.0]]),
        gap_half=torch.tensor([[0.36]]),
    )
    force, _, _ = feasible_forces_for_absolute_wrench(state, torch.tensor([[8.0, 0.0, 0.0]]), cfg)
    legacy = single_slit_contact_feedback_step_v1(state, force, scene, cfg)
    assert not bool(legacy.attempted_hit[0])
    guarded, certificate = certified_contact_step(state, force, scene, cfg)
    assert bool(guarded.attempted_hit[0])
    assert not certificate.passed
    assert certificate.reason == "interior_clearance_violation"
    torch.testing.assert_close(guarded.resolved_state, state, rtol=0, atol=0)
