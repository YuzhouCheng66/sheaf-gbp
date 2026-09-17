"""Opt-in audit simulator: persistent physical world state, fixed local chart.

World dynamics and world OBB-SAT run in float64. The policy receives float32
inverse-chart states and emits forces in that same chart. This interface
control requires a supplied, fixed scene frame; it does not infer a frame,
change wall normals between skills, or add learned equivariance. Identity
float64 is the necessary matched control for rotated float64 runs.
"""

import numpy as np
import torch

from .env import step_dynamics
from .payload_shapes import payload_rectangles_for_config
from .rigid_scene_audit import angle_wrap, certify_world_path, world_scene_from_legacy
from .single_slit_contact_feedback import SingleSlitContactFeedbackResultV1


class FramedWorldPhysics:
    def __init__(self, frame):
        self.frame = frame
        self.geometry = None
        self.world_state = None
        self.states = []
        self.forces = []
        self.hits = []
        self.minimum_clearance = np.inf

    def __call__(self, state, forces, scene, cfg):
        if state.shape != (1, 6) or forces.shape != (1, cfg.n_agents, 2):
            raise ValueError("single-episode state6 and per-agent force2 required")
        local = state.detach().cpu().double().numpy()
        if self.world_state is None:
            body = payload_rectangles_for_config(cfg, torch.device("cpu")).double().numpy()
            self.geometry = world_scene_from_legacy(scene, cfg, body).transformed(self.frame)
            self.world_state = self.frame.states(local)
            self.states.append(self.world_state[0].copy())
        recovered = self.frame.inverse().states(self.world_state)
        delta = local - recovered
        delta[:, 2] = angle_wrap(delta[:, 2])
        if not np.isfinite(delta).all() or np.abs(delta).max() > 2e-6:
            raise ValueError("controller state no longer corresponds to persistent world state")
        current_clearance = float(self.geometry.clearance(self.world_state)[0])
        if current_clearance <= 0:
            raise ValueError("frame audit requires collision-free current world pose")
        world_forces = self.frame.vectors(forces.detach().cpu().double().numpy())
        attempted = step_dynamics(torch.from_numpy(self.world_state),
                                  torch.from_numpy(world_forces), cfg).numpy()
        proposed_path = np.concatenate((self.world_state, attempted))
        cert = certify_world_path(proposed_path, self.geometry)
        hit = not cert.passed
        resolved = attempted.copy()
        if hit:
            resolved[:, :3] = self.world_state[:, :3]
            resolved[:, 3:] = 0.
        self.world_state = resolved
        self.states.append(resolved[0].copy())
        self.forces.append(world_forces[0].copy())
        self.hits.append(hit)
        resolved_clearance = current_clearance if hit else cert.minimum_sampled_clearance_m
        self.minimum_clearance = min(self.minimum_clearance, resolved_clearance)
        # Retain the feedback API, not its old sampled-collision algorithm.
        # An uncertified positive path is a rejection marker, NOT penetration.
        attempted_clearance = min(0., cert.minimum_sampled_clearance_m) if hit else cert.minimum_sampled_clearance_m
        result = SingleSlitContactFeedbackResultV1(
            resolved_state=state.new_tensor(self.frame.inverse().states(resolved)),
            attempted_state=state.new_tensor(self.frame.inverse().states(attempted)),
            attempted_hit=torch.tensor([hit], device=state.device),
            attempted_clearance_m=state.new_tensor([attempted_clearance]),
            resolved_clearance_m=state.new_tensor([resolved_clearance]),
        )
        return result, cert
