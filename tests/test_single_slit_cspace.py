from __future__ import annotations

import math

import numpy as np

from piano_movers.payload_shapes import (
    CANONICAL_T_MORPHOLOGY,
    TMorphology,
    parametric_t_rectangles,
)
from piano_movers.single_slit_cspace import (
    cached_single_slit_cspace_solution,
    plan_single_slit_cspace,
)


def test_noncanonical_t_cspace_path_has_no_unrepresented_fiber_jump() -> None:
    morphology = TMorphology(
        total_width_m=0.930,
        total_height_m=0.920,
        stem_width_m=0.160,
        cap_thickness_m=0.152,
    )
    rectangles = parametric_t_rectangles(morphology).numpy()
    path = plan_single_slit_cspace(
        rectangles,
        effective_half_aperture=0.330,
        slab_half_thickness=0.060,
        world_y=1.6,
    )

    delta_xy = np.abs(np.diff(path[:, :2], axis=0))
    delta_theta = np.abs(
        np.arctan2(
            np.sin(np.diff(path[:, 2])),
            np.cos(np.diff(path[:, 2])),
        )
    )
    assert float(delta_xy.max()) <= 0.004001
    assert float(delta_theta.max()) <= 2.0 * math.pi / 720.0 + 1.0e-6
    np.testing.assert_allclose(path[0], (-0.70, 0.0, 0.0), atol=1.0e-7)
    np.testing.assert_allclose(path[-1], (0.70, 0.0, 0.0), atol=1.0e-7)


def test_canonical_boundary_case_refines_before_sacrificing_all_margin() -> None:
    rectangles = parametric_t_rectangles(CANONICAL_T_MORPHOLOGY).numpy()
    path, planning_safety = cached_single_slit_cspace_solution(
        tuple(float(value) for value in rectangles.reshape(-1)),
        rectangles.shape[0],
        0.330,
        0.060,
        1.600,
    )

    assert planning_safety >= 0.0005
    assert path.shape[0] > 1000
    assert float(np.abs(np.diff(path[:, :2], axis=0)).max()) <= 0.004001
