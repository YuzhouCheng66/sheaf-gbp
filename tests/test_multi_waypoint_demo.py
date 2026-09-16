from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from piano_movers.config import HANDLE_LAYOUT_GLYPH_BOUNDARY_V4
from piano_movers.multi_waypoint_demo import (
    DEFAULT_COURSE,
    DUMMY_WALL_X_M,
    INDEX_HTML,
    Slit,
    _active_observation_scenario,
    _course_config,
    _full_collision_scenario,
    _intermediate_reached,
    _large_glyph_scale,
)
from piano_movers.payload_shapes import payload_rectangles_for_config
from piano_movers.single_slit_feedback import single_slit_env_config


def test_default_course_has_one_post_wall_target_and_a_final_goal() -> None:
    DEFAULT_COURSE.validate()
    assert len(DEFAULT_COURSE.walls) == 3
    assert len(DEFAULT_COURSE.waypoints) == 4
    assert all(
        wall.x < waypoint[0]
        for wall, waypoint in zip(DEFAULT_COURSE.walls, DEFAULT_COURSE.waypoints)
    )


def test_course_validation_rejects_uncertified_narrower_slit() -> None:
    course = replace(
        DEFAULT_COURSE,
        walls=(Slit(-1.8, 0.0, aperture_m=0.719), *DEFAULT_COURSE.walls[1:]),
    )
    with pytest.raises(ValueError, match="0.720"):
        course.validate()


def test_full_collision_world_keeps_every_wall() -> None:
    cfg = _course_config(single_slit_env_config(), 12, DEFAULT_COURSE)
    scenario = _full_collision_scenario(DEFAULT_COURSE, cfg)

    torch.testing.assert_close(
        scenario.wall_x[0],
        torch.tensor([wall.x for wall in DEFAULT_COURSE.walls]),
    )
    torch.testing.assert_close(
        scenario.gap_y[0],
        torch.tensor([wall.gap_y for wall in DEFAULT_COURSE.walls]),
    )


def test_each_waypoint_leg_is_expressed_in_the_frozen_skill_wall_frame() -> None:
    state = torch.tensor([[0.61, 0.04, 0.20, 0.0, 0.0, 0.0]])
    local_state, scenario = _active_observation_scenario(
        DEFAULT_COURSE, state, waypoint_index=2
    )
    active_wall = DEFAULT_COURSE.walls[2]
    active_goal = DEFAULT_COURSE.waypoints[2]

    assert local_state[0, 0].item() == pytest.approx(state[0, 0].item() - active_wall.x)
    assert scenario.wall_x.tolist() == [[0.0, DUMMY_WALL_X_M]]
    assert scenario.goal[0, 0].item() == pytest.approx(active_goal[0] - active_wall.x)
    # Localization is observation-only; the shared exact world state is not mutated.
    assert state[0, 0].item() == pytest.approx(0.61)


def test_non_t_letter_uses_exact_glyph_geometry_and_boundary_handles() -> None:
    cfg = _course_config(
        single_slit_env_config(), 12, DEFAULT_COURSE, payload_shape="X"
    )
    assert cfg.payload_shape == "X"
    assert cfg.handle_layout == HANDLE_LAYOUT_GLYPH_BOUNDARY_V4
    assert cfg.glyph_scale > 1.0


def test_large_glyph_profile_has_a_tight_representable_passage_heading() -> None:
    cfg = _course_config(
        single_slit_env_config(), 12, DEFAULT_COURSE, payload_shape="X"
    )
    rectangles = payload_rectangles_for_config(cfg, "cpu").to(torch.float64)
    vertices = rectangles.reshape(-1, 2)
    angles = torch.arange(72, dtype=torch.float64) * (2.0 * torch.pi / 72.0)
    projected_y = (
        torch.sin(angles)[:, None] * vertices[None, :, 0]
        + torch.cos(angles)[:, None] * vertices[None, :, 1]
    )
    minimum_span = (projected_y.amax(dim=-1) - projected_y.amin(dim=-1)).amin()
    assert float(minimum_span) == pytest.approx(0.648, abs=1.0e-6)
    assert cfg.glyph_scale == pytest.approx(_large_glyph_scale("X"))


def test_intermediate_waypoint_ignores_heading() -> None:
    waypoint = (0.63, 0.17, 0.0)
    state = torch.tensor([[0.63, 0.17, 2.7, 0.0, 0.0, 0.0]])
    assert _intermediate_reached(state, waypoint)


def test_page_labels_route_as_waypoints_without_claiming_mapping() -> None:
    assert "header.waypoints" in INDEX_HTML
    assert "Multi-wall waypoint" in INDEX_HTML
    assert 'id="shape"' in INDEX_HTML
    assert "shape='+encodeURIComponent(shape.value)" in INDEX_HTML
    assert "function finalPose()" in INDEX_HTML
    assert "ctx.setLineDash([4,6])" in INDEX_HTML
    assert "mapping" not in INDEX_HTML.lower()
    assert "global planning" not in INDEX_HTML.lower()
