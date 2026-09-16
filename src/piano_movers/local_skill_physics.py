"""Continuously certified contact rejection for p6, without changing public p5."""

from dataclasses import replace

import torch

from .local_skill_teacher import certify_pose_path
from .single_slit_contact_feedback import single_slit_contact_feedback_step_v1


def certified_contact_step(state, forces, scene, cfg):
    """Return mechanical feedback and a certificate for a newly accepted move.

    The V1 feedback *container* is retained for the existing local sensor API.
    This is a different transition rule: sampled-safe proposals must also pass
    the conservative continuous SAT motion-bound check. Uncertified movement
    is rejected with zero velocity, preserving the last safe pose. No action
    is planned or improved by this guard, and every rejection is observable.
    """
    if state.shape[0] != 1:
        raise ValueError("continuous p6 contact wrapper currently requires one episode")
    result = single_slit_contact_feedback_step_v1(state, forces, scene, cfg)
    if bool(result.attempted_hit[0]):
        return result, None
    certificate = certify_pose_path(torch.cat((state, result.attempted_state)), scene, cfg)
    if certificate.passed:
        return result, certificate
    stopped = torch.cat((state[:, :3], torch.zeros_like(state[:, 3:])), -1)
    from .env import collision_clearance_margin

    result = replace(
        result,
        resolved_state=stopped,
        attempted_hit=torch.ones_like(result.attempted_hit),
        attempted_clearance_m=torch.minimum(
            result.attempted_clearance_m,
            # The V1 container requires a nonpositive rejection marker. A
            # depth-limit rejection is uncertainty, not a measured overlap;
            # its marker must not be reported as a penetration measurement.
            state.new_tensor([min(0., certificate.minimum_sampled_clearance_m)]),
        ),
        resolved_clearance_m=collision_clearance_margin(stopped, scene, cfg),
    )
    return result, certificate
