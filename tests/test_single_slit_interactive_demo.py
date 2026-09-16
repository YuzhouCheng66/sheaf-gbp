from __future__ import annotations

import re

import pytest

from piano_movers.single_slit_interactive_demo import (
    DemoParameters,
    INDEX_HTML,
    RANGES,
)


def test_default_interactive_geometry_is_admitted_and_canonical_slit_width():
    value = DemoParameters()
    value.validate()
    assert value.aperture_m == pytest.approx(0.720)
    assert value.gap_center_m == pytest.approx(0.0)
    for name, (lower, upper) in RANGES.items():
        assert lower <= getattr(value, name) <= upper


def test_interactive_geometry_fails_closed_below_certified_aperture():
    value = DemoParameters(upper_wall_length_m=0.73)
    with pytest.raises(ValueError, match="upper_wall_length_m"):
        value.validate()


def test_interactive_geometry_fails_closed_outside_the_admitted_t_family():
    value = DemoParameters(thickness_scale=1.5)
    with pytest.raises(ValueError, match="thickness_scale"):
        value.validate()


def test_zero_wall_lengths_are_a_physically_open_world():
    value = DemoParameters(upper_wall_length_m=0.0, lower_wall_length_m=0.0)
    value.validate()
    assert value.upper_bottom_m == pytest.approx(1.08)
    assert value.lower_top_m == pytest.approx(-1.08)
    assert value.aperture_m == pytest.approx(2.16)


@pytest.mark.parametrize("n_agents", [8, 13, 20])
def test_interactive_agent_count_covers_the_parametric_handle_family(n_agents):
    DemoParameters(n_agents=n_agents).validate()


def test_public_interactive_payload_is_the_formally_rerun_t_only():
    DemoParameters(payload_shape="T").validate()
    with pytest.raises(ValueError, match="restricted to the T payload"):
        DemoParameters(payload_shape="A").validate()


def test_hero_is_single_stage_minimal_and_renders_only_streamed_packets():
    lower = INDEX_HTML.lower()
    assert lower.count("<canvas") == 1
    assert "baseline" not in lower
    assert "teacher" not in lower
    assert "human-race" not in lower
    assert "<h1" not in lower
    assert "drawpacket" in lower
    assert "header.sender[m]" in lower
    assert "header.receiver[m]" in lower
    assert "f.intensity[sweep]" in lower
    assert "(!playing&&timeline<=0)" in lower
    assert "new eventsource" not in lower
    assert "fetch(paramsurl()" in lower


def test_hero_exposes_direct_wall_drag_agent_count_and_two_t_controls():
    assert INDEX_HTML.index('id="shape"') < INDEX_HTML.index('id="random"')
    assert "shape=${encodeURIComponent(p.shape)}" in INDEX_HTML
    assert "p.shape=inputs.shape.value" in INDEX_HTML
    for control in ("agents", "length", "thickness"):
        assert f'id="{control}" type="range"' in INDEX_HTML
    assert "drag='upper'" in INDEX_HTML
    assert "drag='lower'" in INDEX_HTML
    assert "MAX_WALL=.72" in INDEX_HTML
    assert "p.upperWall=Math.max(0" in INDEX_HTML
    assert "p.lowerWall=Math.max(0" in INDEX_HTML
    assert "background=null;prepare()" in INDEX_HTML


def test_random_keeps_the_original_narrow_slit_distribution():
    assert "RANDOM_APERTURE_MIN=.72" in INDEX_HTML
    assert "RANDOM_APERTURE_MAX=.80" in INDEX_HTML
    assert "aperture=RANDOM_APERTURE_MIN+r()" in INDEX_HTML
    assert "p.upperWall=WORLD_Y-center-aperture/2" in INDEX_HTML
    assert "p.lowerWall=WORLD_Y+center-aperture/2" in INDEX_HTML


def test_hero_palette_is_strictly_monochrome():
    for value in re.findall(r"#([0-9a-fA-F]{6})", INDEX_HTML):
        assert value[0:2] == value[2:4] == value[4:6], value
    for red, green, blue in re.findall(
        r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", INDEX_HTML
    ):
        assert red == green == blue, (red, green, blue)
