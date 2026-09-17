"""Declared environment-side impulse for contact recovery stress tests.

Not a policy input, action replacement, teacher, or new nominal benchmark.
The first near-obstacle state receives one bounded linear-velocity impulse
toward decreasing exact-SAT clearance. Subsequent actions and feedback are
the unchanged student loop. Trigger failures remain explicit in reports.
"""

from dataclasses import dataclass
import math

import torch

from .env import collision_clearance_margin
from .local_skill_physics import certified_contact_step
from .local_skill_teacher import repeat_scene


@dataclass(frozen=True)
class ContactImpulseOptions:
    enabled: bool = True
    trigger_clearance_m: float = .03
    delta_speed_mps: float = .65
    delta_omega_radps: float = 0.
    finite_difference_m: float = 1e-4

    def __post_init__(self):
        for value in (self.trigger_clearance_m, self.delta_speed_mps, self.finite_difference_m):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("impulse parameters must be positive and finite")
        if not math.isfinite(self.delta_omega_radps):
            raise ValueError("angular velocity impulse must be finite")


class SingleContactImpulse:
    def __init__(self, options=ContactImpulseOptions()):
        self.options = options
        self.tick = 0
        self.event = None

    @torch.no_grad()
    def __call__(self, state, forces, scene, cfg):
        if state.shape != (1, 6):
            raise ValueError("single-episode state6 required")
        physical_input = state
        if self.options.enabled and self.event is None:
            clearance = float(collision_clearance_margin(state, scene, cfg)[0])
            if 0 < clearance <= self.options.trigger_clearance_m:
                h = self.options.finite_difference_m
                probes = state.double().expand(4, -1).clone()
                probes[0, 0] += h
                probes[1, 0] -= h
                probes[2, 1] += h
                probes[3, 1] -= h
                values = collision_clearance_margin(probes, repeat_scene(scene, 4, probes), cfg)
                gradient = torch.stack((values[0] - values[1], values[2] - values[3])) / (2 * h)
                norm = torch.linalg.vector_norm(gradient)
                # Declared tie handling, never an outcome-dependent retry.
                degenerate = bool(norm < 1e-9)
                direction = gradient.new_tensor([0., 1.]) if degenerate else -gradient / norm
                delta = (direction * self.options.delta_speed_mps).to(state)
                physical_input = state.clone()
                physical_input[:, 3:5] += delta
                physical_input[:, 5] += self.options.delta_omega_radps
                self.event = {
                    "tick": self.tick, "clearance_before_m": clearance,
                    "velocity_impulse_world_mps": delta.tolist(),
                    "linear_momentum_impulse_kg_mps": (cfg.mass * delta).tolist(),
                    "angular_velocity_impulse_radps": self.options.delta_omega_radps,
                    "angular_momentum_impulse_kg_m2_radps": cfg.inertia * self.options.delta_omega_radps,
                    "clearance_gradient": gradient.tolist(), "degenerate_gradient_fallback": degenerate,
                    "state_before": state[0].tolist(), "physical_state_after_impulse": physical_input[0].tolist(),
                    "policy_saw_impulse_before_action": False, "agent_forces_modified": False,
                }
        result, cert = certified_contact_step(physical_input, forces, scene, cfg)
        if self.event is not None and self.event["tick"] == self.tick:
            self.event["attempt_rejected"] = bool(result.attempted_hit[0])
            self.event["resolved_state"] = result.resolved_state[0].tolist()
        self.tick += 1
        return result, cert
