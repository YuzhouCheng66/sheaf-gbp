"""Exact compound geometry for morphology-honest demo payloads."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import torch
from torch import Tensor


PUBLIC_PAYLOAD_SHAPES = ("T", "I", "Y", "PI")
LETTER_PAYLOAD_SHAPES = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
SUPPORTED_PAYLOAD_SHAPES = tuple(
    dict.fromkeys((*PUBLIC_PAYLOAD_SHAPES, *LETTER_PAYLOAD_SHAPES))
)

CANONICAL_T_TOTAL_WIDTH_M = 0.935
CANONICAL_T_TOTAL_HEIGHT_M = 0.9185
CANONICAL_T_STEM_WIDTH_M = 0.220
CANONICAL_T_CAP_THICKNESS_M = 0.2035
CANONICAL_T_BODY_CENTER_Y_M = -0.06325
CANONICAL_T_CAP_STEM_OVERLAP_M = 0.110
_CANONICAL_T_OVERLAP_FRACTION = (
    CANONICAL_T_CAP_STEM_OVERLAP_M / CANONICAL_T_CAP_THICKNESS_M
)


@dataclass(frozen=True)
class TMorphology:
    """Four-dimensional physical T family used by the generalization study."""

    total_width_m: float = CANONICAL_T_TOTAL_WIDTH_M
    total_height_m: float = CANONICAL_T_TOTAL_HEIGHT_M
    stem_width_m: float = CANONICAL_T_STEM_WIDTH_M
    cap_thickness_m: float = CANONICAL_T_CAP_THICKNESS_M

    def validate(self) -> None:
        values = (
            self.total_width_m,
            self.total_height_m,
            self.stem_width_m,
            self.cap_thickness_m,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("T morphology dimensions must be positive and finite")
        if self.stem_width_m > self.total_width_m:
            raise ValueError("T stem width cannot exceed total width")
        if self.cap_thickness_m >= self.total_height_m:
            raise ValueError("T cap thickness must be smaller than total height")

    @property
    def is_canonical(self) -> bool:
        return self == CANONICAL_T_MORPHOLOGY


CANONICAL_T_MORPHOLOGY = TMorphology()


class _TMorphologyConfig(Protocol):
    payload_shape: str
    t_total_width_m: float
    t_total_height_m: float
    t_stem_width_m: float
    t_cap_thickness_m: float


def t_morphology_from_config(config: _TMorphologyConfig) -> TMorphology:
    morphology = TMorphology(
        total_width_m=float(config.t_total_width_m),
        total_height_m=float(config.t_total_height_m),
        stem_width_m=float(config.t_stem_width_m),
        cap_thickness_m=float(config.t_cap_thickness_m),
    )
    morphology.validate()
    return morphology


def parametric_t_rectangles(
    morphology: TMorphology,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return the exact two-rectangle union for one admitted T morphology.

    The body-frame vertical centre and the cap/stem overlap fraction remain
    fixed.  The canonical branch deliberately retains the historical literals
    so old collision arithmetic is bit-for-bit unchanged.
    """

    morphology.validate()
    if morphology.is_canonical:
        return torch.tensor(
            [
                _rectangle(0.0, -0.110, 0.220, 0.825),
                _rectangle(0.0, 0.29425, 0.935, 0.2035),
            ],
            dtype=torch.float32,
            device=device,
        )
    bottom = CANONICAL_T_BODY_CENTER_Y_M - morphology.total_height_m / 2.0
    top = CANONICAL_T_BODY_CENTER_Y_M + morphology.total_height_m / 2.0
    cap_bottom = top - morphology.cap_thickness_m
    overlap = _CANONICAL_T_OVERLAP_FRACTION * morphology.cap_thickness_m
    stem_top = cap_bottom + overlap
    stem_height = stem_top - bottom
    return torch.tensor(
        [
            _rectangle(
                0.0,
                (bottom + stem_top) / 2.0,
                morphology.stem_width_m,
                stem_height,
            ),
            _rectangle(
                0.0,
                (cap_bottom + top) / 2.0,
                morphology.total_width_m,
                morphology.cap_thickness_m,
            ),
        ],
        dtype=torch.float32,
        device=device,
    )


def payload_rectangles_for_config(
    config: _TMorphologyConfig,
    device: torch.device | str | None = None,
) -> Tensor:
    """Resolve collision/render geometry from a complete environment config."""

    normalized = normalize_payload_shape(config.payload_shape)
    if normalized == "T":
        return parametric_t_rectangles(t_morphology_from_config(config), device)
    # The early four-shape public demo deliberately used exaggerated I/Y
    # silhouettes with canonical-T force pads.  The A-Z experiment instead
    # uses one coherent, bounded sans-serif glyph family with boundary handles.
    # Keeping those domains separate avoids two outlier bodies whose feasible
    # 0.720 m passages have less than a millimetre of clearance, while still
    # preserving the old public geometry API and canonical T bit-for-bit.
    if (
        normalized in LETTER_PAYLOAD_SHAPES
        and getattr(config, "handle_layout", None) == "glyph-boundary-v4"
    ):
        return torch.tensor(
            _glyph_segment_rectangles(normalized),
            dtype=torch.float32,
            device=device,
        )
    return payload_shape_rectangles(normalized, device)


def normalize_payload_shape(shape: str) -> str:
    normalized = shape.strip().upper()
    if normalized == "Π":
        normalized = "PI"
    if normalized not in SUPPORTED_PAYLOAD_SHAPES:
        raise ValueError(
            f"payload_shape must be one of {SUPPORTED_PAYLOAD_SHAPES}, got {shape!r}"
        )
    return normalized


def _rectangle(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    angle: float = 0.0,
) -> list[list[float]]:
    half_width = width / 2.0
    half_height = height / 2.0
    local = (
        (-half_width, -half_height),
        (half_width, -half_height),
        (half_width, half_height),
        (-half_width, half_height),
    )
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return [
        [
            center_x + cosine * x - sine * y,
            center_y + sine * x + cosine * y,
        ]
        for x, y in local
    ]


def _bar_between(
    start: tuple[float, float],
    end: tuple[float, float],
    thickness: float,
) -> list[list[float]]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    return _rectangle(
        (start[0] + end[0]) / 2.0,
        (start[1] + end[1]) / 2.0,
        math.hypot(dx, dy),
        thickness,
        math.atan2(dy, dx),
    )


_GLYPH_LEFT = -0.27
_GLYPH_RIGHT = 0.27
_GLYPH_BOTTOM = -0.44
_GLYPH_TOP = 0.35
_GLYPH_MIDDLE = -0.045
_GLYPH_STROKE = 0.095


def _glyph_segment_rectangles(shape: str) -> list[list[list[float]]]:
    """Return a connected, legible sans-serif block glyph.

    The construction is intentionally geometric rather than font-derived:
    every visible stroke is one exact convex collision rectangle, all glyphs
    share a bounded physical scale, and joined stroke centre-lines overlap at
    their endpoints.  This makes the bodies reproducible on CPU/CUDA without
    introducing a font, rasterizer, polygon-union, or mesh dependency.
    """

    left = _GLYPH_LEFT
    right = _GLYPH_RIGHT
    bottom = _GLYPH_BOTTOM
    top = _GLYPH_TOP
    middle = _GLYPH_MIDDLE
    quarter_top = 0.5 * (top + middle)
    quarter_bottom = 0.5 * (bottom + middle)
    inner_left = -0.14
    inner_right = 0.14
    bowl_left = -0.24
    bowl_right = 0.24
    centre = 0.0
    segments: dict[str, tuple[tuple[tuple[float, float], tuple[float, float]], ...]] = {
        "A": (((left, bottom), (centre, top)), ((centre, top), (right, bottom)), ((-0.16, middle), (0.16, middle))),
        "B": (((left, bottom), (left, top)), ((left, top), (inner_right, top)), ((left, middle), (inner_right, middle)), ((left, bottom), (inner_right, bottom)), ((inner_right, top), (right, quarter_top)), ((right, quarter_top), (inner_right, middle)), ((inner_right, middle), (right, quarter_bottom)), ((right, quarter_bottom), (inner_right, bottom))),
        "C": (((right, top), (left, top)), ((left, top), (left, bottom)), ((left, bottom), (right, bottom))),
        "D": (((left, bottom), (left, top)), ((left, top), (inner_right, top)), ((inner_right, top), (right, quarter_top)), ((right, quarter_top), (right, quarter_bottom)), ((right, quarter_bottom), (inner_right, bottom)), ((inner_right, bottom), (left, bottom))),
        "E": (((left, bottom), (left, top)), ((left, top), (right, top)), ((left, middle), (0.20, middle)), ((left, bottom), (right, bottom))),
        "F": (((left, bottom), (left, top)), ((left, top), (right, top)), ((left, middle), (0.20, middle))),
        "G": (((right, top), (left, top)), ((left, top), (left, bottom)), ((left, bottom), (right, bottom)), ((right, bottom), (right, middle)), ((0.05, middle), (right, middle))),
        "H": (((left, bottom), (left, top)), ((right, bottom), (right, top)), ((left, middle), (right, middle))),
        "I": (((left, top), (right, top)), ((centre, top), (centre, bottom)), ((left, bottom), (right, bottom))),
        "J": (((left, top), (right, top)), ((right, top), (right, bottom)), ((right, bottom), (inner_left, bottom)), ((inner_left, bottom), (left, quarter_bottom))),
        "K": (((left, bottom), (left, top)), ((left, middle), (right, top)), ((left, middle), (right, bottom))),
        "L": (((left, top), (left, bottom)), ((left, bottom), (right, bottom))),
        "M": (((left, bottom), (left, top)), ((left, top), (centre, middle)), ((centre, middle), (right, top)), ((right, top), (right, bottom))),
        "N": (((left, bottom), (left, top)), ((left, top), (right, bottom)), ((right, bottom), (right, top))),
        "O": (((bowl_left, bottom), (bowl_left, top)), ((bowl_left, top), (bowl_right, top)), ((bowl_right, top), (bowl_right, bottom)), ((bowl_right, bottom), (bowl_left, bottom))),
        "P": (((left, bottom), (left, top)), ((left, top), (inner_right, top)), ((inner_right, top), (right, quarter_top)), ((right, quarter_top), (inner_right, middle)), ((inner_right, middle), (left, middle))),
        "Q": (((bowl_left, bottom), (bowl_left, top)), ((bowl_left, top), (bowl_right, top)), ((bowl_right, top), (bowl_right, bottom)), ((bowl_right, bottom), (bowl_left, bottom)), ((0.04, quarter_bottom), (0.28, -0.49))),
        "R": (((left, bottom), (left, top)), ((left, top), (inner_right, top)), ((inner_right, top), (right, quarter_top)), ((right, quarter_top), (inner_right, middle)), ((inner_right, middle), (left, middle)), ((centre, middle), (right, bottom))),
        "S": (((right, top), (left, top)), ((left, top), (left, middle)), ((left, middle), (right, middle)), ((right, middle), (right, bottom)), ((right, bottom), (left, bottom))),
        "T": (((left, top), (right, top)), ((centre, top), (centre, bottom))),
        "U": (((left, top), (left, bottom)), ((left, bottom), (right, bottom)), ((right, bottom), (right, top))),
        "V": (((left, top), (centre, bottom)), ((centre, bottom), (right, top))),
        "W": (((left, top), (-0.20, bottom)), ((-0.20, bottom), (centre, quarter_bottom)), ((centre, quarter_bottom), (0.20, bottom)), ((0.20, bottom), (right, top))),
        "X": (((left, top), (right, bottom)), ((right, top), (left, bottom))),
        "Y": (((left, top), (centre, middle)), ((right, top), (centre, middle)), ((centre, middle), (centre, bottom))),
        "Z": (((left, top), (right, top)), ((right, top), (left, bottom)), ((left, bottom), (right, bottom))),
    }
    return [
        _bar_between(start, end, _GLYPH_STROKE)
        for start, end in segments[shape]
    ]


def _points_in_rectangles(points: Tensor, rectangles: Tensor, tolerance: float) -> Tensor:
    """Return union occupancy for points and CCW convex quadrilaterals."""

    edges = torch.roll(rectangles, shifts=-1, dims=1) - rectangles
    offsets = points[:, None, None, :] - rectangles[None, :, :, :]
    cross = (
        edges[None, :, :, 0] * offsets[..., 1]
        - edges[None, :, :, 1] * offsets[..., 0]
    )
    return (cross >= -tolerance).all(dim=-1).any(dim=-1)


def glyph_boundary_handle_points(
    shape: str,
    n_agents: int,
    *,
    communication_radius_m: float,
) -> Tensor:
    """Generate deterministic, connected handles on the exact union boundary."""

    normalized = normalize_payload_shape(shape)
    if normalized not in LETTER_PAYLOAD_SHAPES:
        raise ValueError("glyph handles require an A-Z payload")
    if not 8 <= n_agents <= 20:
        raise ValueError("glyph handles support every integer N in [8,20]")
    if not math.isfinite(communication_radius_m) or communication_radius_m <= 0.0:
        raise ValueError("communication radius must be positive and finite")

    rectangles = payload_shape_rectangles(normalized, "cpu").to(torch.float64)
    samples_per_edge = 65
    fraction = torch.linspace(0.0, 1.0, samples_per_edge, dtype=torch.float64)
    candidates: list[Tensor] = []
    for rectangle in rectangles:
        for edge_index in range(4):
            start = rectangle[edge_index]
            end = rectangle[(edge_index + 1) % 4]
            edge = end - start
            points = start[None, :] + fraction[:, None] * edge[None, :]
            candidates.append(points)
    candidate = torch.cat(candidates, dim=0)
    # Rectangle edges hidden inside another stroke are not union boundaries.
    # Keep an edge sample only when a small radial neighborhood contains a
    # point outside the complete union.  Thirty-two fixed directions make the
    # test independent of which overlapping rectangle supplied the sample.
    angles = torch.linspace(0.0, 2.0 * math.pi, 33, dtype=torch.float64)[:-1]
    directions = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
    boundary_probe = candidate[:, None, :] + 2.0e-4 * directions[None, :, :]
    outside_probe = ~_points_in_rectangles(
        boundary_probe.reshape(-1, 2), rectangles, tolerance=1.0e-9
    ).reshape(candidate.shape[0], -1)
    candidate = candidate[outside_probe.any(dim=1)]
    quantized = torch.round(candidate * 1.0e6).to(torch.int64)
    keep: list[int] = []
    seen: set[tuple[int, int]] = set()
    for index, pair in enumerate(quantized.tolist()):
        key = (int(pair[0]), int(pair[1]))
        if key not in seen:
            seen.add(key)
            keep.append(index)
    candidate = candidate[torch.tensor(keep, dtype=torch.long)]
    if candidate.shape[0] < n_agents:
        raise RuntimeError("glyph boundary sampler produced too few candidates")

    centre = rectangles.reshape(-1, 2).mean(dim=0)
    selected = [int(torch.argmin(torch.linalg.vector_norm(candidate - centre, dim=-1)))]
    available = torch.ones(candidate.shape[0], dtype=torch.bool)
    available[selected[0]] = False
    connection_limit = 0.92 * communication_radius_m
    for _ in range(1, n_agents):
        selected_points = candidate[torch.tensor(selected, dtype=torch.long)]
        distance = torch.cdist(candidate, selected_points)
        nearest = distance.amin(dim=1)
        eligible = available & (nearest <= connection_limit) & (nearest >= 0.020)
        if not bool(eligible.any()):
            eligible = available & (nearest <= communication_radius_m)
        if not bool(eligible.any()):
            raise RuntimeError("glyph boundary cannot support a connected radius graph")
        radial = torch.linalg.vector_norm(candidate - centre, dim=-1)
        score = nearest + 0.08 * radial
        score = torch.where(eligible, score, torch.full_like(score, -torch.inf))
        chosen = int(torch.argmax(score))
        selected.append(chosen)
        available[chosen] = False
    result = candidate[torch.tensor(selected, dtype=torch.long)].to(torch.float32)
    if result.shape != (n_agents, 2) or not bool(torch.isfinite(result).all()):
        raise RuntimeError("glyph handle construction changed shape or finiteness")
    return result


def payload_shape_rectangles(
    shape: str,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return exact convex quads for one supported rigid payload glyph.

    All public morphologies retain the canonical twelve C2 force handles,
    mass, inertia, radius graph, and controller.  Their silhouettes deliberately
    exaggerate the distinguishing strokes of each glyph so the shapes remain
    legible while moving in the public demo.  Every stroke is physical collision
    geometry rather than decorative Canvas art.
    """

    normalized = normalize_payload_shape(shape)
    # A-Z is one coherent metric glyph family.  Earlier demo-only I/Y bodies
    # carried wide canonical-T handle pads and became sub-millimetre outliers
    # at the 0.720 m opening; they were visually distinct but not samples from
    # the alphabet distribution.  Boundary handles now remove that legacy
    # constraint, so use the same stroke grammar for every non-T letter.
    if normalized in LETTER_PAYLOAD_SHAPES and normalized != "T":
        return torch.tensor(
            _glyph_segment_rectangles(normalized),
            dtype=torch.float32,
            device=device,
        )
    shapes: dict[str, list[list[list[float]]]] = {
        "T": [
            _rectangle(0.0, -0.110, 0.220, 0.825),
            _rectangle(0.0, 0.29425, 0.935, 0.2035),
        ],
        "I": [
            # A skeletal centre spine with narrow physical handle tabs.  The
            # hooked top serif keeps all six upper handles attached without
            # recreating T's solid cap.
            _rectangle(0.0, -0.06325, 0.055, 0.9185),
            _rectangle(0.0, 0.090, 0.220, 0.018),
            _rectangle(0.0, -0.215, 0.220, 0.018),
            _rectangle(0.0, 0.386, 0.935, 0.020),
            _rectangle(-0.4525, 0.338, 0.030, 0.096),
            _rectangle(0.4525, 0.338, 0.030, 0.096),
            _rectangle(0.0, -0.4625, 0.540, 0.120),
        ],
        "Y": [
            # Deep negative space makes the fork legible in motion.  Thin
            # cross-tabs are structural force pads at the canonical handles;
            # the ants visually occupy them in the hero renderer.
            _rectangle(0.0, -0.16125, 0.060, 0.7225),
            _rectangle(0.0, 0.090, 0.220, 0.018),
            _rectangle(0.0, -0.215, 0.220, 0.018),
            _rectangle(0.0, -0.5135, 0.160, 0.018),
            _bar_between((-0.018, 0.205), (-0.290, 0.360), 0.070),
            _bar_between((0.018, 0.205), (0.290, 0.360), 0.070),
            _rectangle(-0.205, 0.386, 0.230, 0.020),
            _rectangle(0.205, 0.386, 0.230, 0.020),
            _bar_between((-0.420, 0.310), (-0.310, 0.365), 0.060),
            _bar_between((0.420, 0.310), (0.310, 0.365), 0.060),
            _rectangle(-0.4425, 0.300, 0.050, 0.040),
            _rectangle(0.4425, 0.300, 0.050, 0.040),
        ],
        "PI": [
            # A hairline hooked lintel and two widely separated full-height
            # legs maximize the Pi negative space while retaining all handles.
            _rectangle(0.0, 0.386, 0.935, 0.020),
            _rectangle(-0.4525, 0.338, 0.030, 0.096),
            _rectangle(0.4525, 0.338, 0.030, 0.096),
            _rectangle(-0.1275, -0.06825, 0.105, 0.9085),
            _rectangle(0.1275, -0.06825, 0.105, 0.9085),
        ],
    }
    if normalized in LETTER_PAYLOAD_SHAPES and normalized not in shapes:
        return torch.tensor(
            _glyph_segment_rectangles(normalized),
            dtype=torch.float32,
            device=device,
        )
    return torch.tensor(shapes[normalized], dtype=torch.float32, device=device)


def payload_shape_points(
    shape: str,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return vertices and rectangle centres for training-time risk proxies."""

    rectangles = payload_shape_rectangles(shape, device)
    centres = rectangles.mean(dim=1)
    return torch.cat((rectangles.reshape(-1, 2), centres), dim=0)
