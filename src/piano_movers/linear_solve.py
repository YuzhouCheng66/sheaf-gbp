"""Small nonsingular linear solves without an implicit CUDA status sync."""

from __future__ import annotations

import torch
from torch import Tensor


def solve_nonsingular(matrix: Tensor, rhs: Tensor) -> Tensor:
    """Solve a system whose caller has already guaranteed nonsingularity.

    Every use in PianoMovers adds a positive diagonal damping term to a grasp
    Gram matrix. ``torch.linalg.solve`` additionally checks the device-side
    solver status and synchronizes CUDA. ``solve_ex(check_errors=False)`` uses
    the same numerical solve and autograd rule but leaves the unused status on
    device, which matters inside long sequential rollouts.
    """

    return torch.linalg.solve_ex(matrix, rhs, check_errors=False).result
