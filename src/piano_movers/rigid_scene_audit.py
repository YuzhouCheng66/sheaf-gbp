"""Independent world-coordinate geometry for frame audits, not a planner.

Obstacles and arena are actual oriented rectangles in world coordinates.
SAT never inverse-transforms the scene to use the legacy axis-aligned checker.
Compound payloads remain unions of rectangles, NOT their convex hulls.
All arithmetic is float64; certificates are conservative numerical tests,
not interval-arithmetic proofs. This module does not change deployment.
"""

from dataclasses import dataclass
import math

import numpy as np

from .local_skill_teacher import ContinuousCertificate


def angle_wrap(value):
    return np.arctan2(np.sin(value), np.cos(value))


@dataclass(frozen=True)
class RigidFrame:
    """World = rotation(angle) * local + translation; vectors have no offset."""

    angle: float = 0.
    translation: tuple[float, float] = (0., 0.)

    def __post_init__(self):
        if len(self.translation) != 2 or not np.isfinite((self.angle, *self.translation)).all():
            raise ValueError("frame requires a finite angle and translation2")

    @property
    def rotation(self):
        c, s = math.cos(self.angle), math.sin(self.angle)
        return np.array([[c, -s], [s, c]], dtype=np.float64)

    def vectors(self, value):
        return np.asarray(value, dtype=np.float64) @ self.rotation.T

    def points(self, value):
        return self.vectors(value) + self.translation

    def states(self, value):
        result = np.asarray(value, dtype=np.float64).copy()
        if result.shape[-1] not in (3, 6):
            raise ValueError("state must be pose3 or state6")
        result[..., :2] = self.points(result[..., :2])
        result[..., 2] = angle_wrap(result[..., 2] + self.angle)
        if result.shape[-1] == 6:
            result[..., 3:5] = self.vectors(result[..., 3:5])
        return result

    def inverse(self):
        offset = -np.asarray(self.translation) @ self.rotation
        return RigidFrame(-self.angle, tuple(offset))


def rectangle(x0, x1, y0, y1):
    if not np.isfinite((x0, x1, y0, y1)).all() or x0 >= x1 or y0 >= y1:
        raise ValueError("rectangle needs finite increasing bounds")
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


def _rectangles(value, allow_empty=False):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 3 or result.shape[1:] != (4, 2) or not np.isfinite(result).all():
        raise ValueError("rectangles must have finite shape [R,4,2]")
    if not len(result):
        if allow_empty:
            return result
        raise ValueError("at least one rectangle required")
    edges = np.roll(result, -1, axis=1) - result
    lengths = np.linalg.norm(edges, axis=-1)
    if (lengths <= 0).any():
        raise ValueError("degenerate rectangle")
    # Source glyph vertices are stored in float32, so tolerate that rounding.
    unit = edges / lengths[..., None]
    if (np.abs((unit * np.roll(unit, -1, axis=1)).sum(-1)) > 2e-5).any():
        raise ValueError("quadrilateral is not a rectangle")
    if not np.allclose(edges[:, :2], -edges[:, 2:], atol=2e-7, rtol=2e-5):
        raise ValueError("rectangle edges must be in cyclic order")
    return result


def body_vertices(poses, body):
    poses = np.asarray(poses, dtype=np.float64)
    c, s = np.cos(poses[:, 2]), np.sin(poses[:, 2])
    rotation = np.stack((c, -s, s, c), -1).reshape(-1, 2, 2)
    return np.einsum("bde,rpe->brpd", rotation, body) + poses[:, None, None, :2]


def _axes(rects):
    edges = rects[..., 1:3, :] - rects[..., :2, :]
    normals = np.stack((-edges[..., 1], edges[..., 0]), -1)
    return normals / np.linalg.norm(normals, axis=-1, keepdims=True)


@dataclass(frozen=True)
class WorldRectangleScene:
    body: np.ndarray
    obstacles: np.ndarray
    arena: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "body", _rectangles(self.body))
        object.__setattr__(self, "obstacles", _rectangles(self.obstacles, allow_empty=True))
        arena = _rectangles(np.asarray(self.arena)[None])[0]
        object.__setattr__(self, "arena", arena)

    def transformed(self, frame):
        return WorldRectangleScene(self.body, frame.points(self.obstacles), frame.points(self.arena))

    def clearance(self, poses):
        """Minimum normalized SAT gap across every body/obstacle pair and arena."""
        poses = np.asarray(poses, dtype=np.float64)
        if poses.ndim != 2 or poses.shape[-1] not in (3, 6) or not np.isfinite(poses).all():
            raise ValueError("finite [batch,3|6] poses required")
        world = body_vertices(poses, self.body)
        ba = _axes(world)
        arena_axes = _axes(self.arena)
        arena_projection = self.arena @ arena_axes.T
        bp = np.einsum("brpd,ad->brpa", world, arena_axes)
        boundary = np.minimum(bp.min((1, 2)) - arena_projection.min(0),
                              arena_projection.max(0) - bp.max((1, 2))).min(-1)
        # Match legacy boundary-touch semantics; the certificate still rejects
        # this numerical zero because it requires a positive reserve.
        boundary = np.where(boundary == 0, np.finfo(np.float64).tiny, boundary)
        if not len(self.obstacles):
            return boundary
        oa = _axes(self.obstacles)
        shape = (len(poses), len(self.body), len(self.obstacles), 2, 2)
        axes = np.concatenate((np.broadcast_to(ba[:, :, None], shape),
                               np.broadcast_to(oa[None, None], shape)), axis=-2)
        bp = np.einsum("brpd,broad->bropa", world, axes)
        op = np.einsum("opd,broad->bropa", self.obstacles, axes)
        gaps = np.maximum(op.min(-2) - bp.max(-2), bp.min(-2) - op.max(-2))
        return np.minimum(boundary, gaps.max(-1).min((1, 2)))


def world_scene_from_legacy(scene, cfg, body):
    """Materialize legacy reserved walls/extra obstacles, including the arena."""
    obstacles = []
    for x, y, half in zip(scene.wall_x[0].tolist(), scene.gap_y[0].tolist(),
                          scene.gap_half[0].tolist(), strict=True):
        half = max(0., half - .03)
        if y - half > -cfg.world_y:
            obstacles.append(rectangle(x - cfg.wall_slab, x + cfg.wall_slab, -cfg.world_y, y - half))
        if y + half < cfg.world_y:
            obstacles.append(rectangle(x - cfg.wall_slab, x + cfg.wall_slab, y + half, cfg.world_y))
    if scene.obstacle_rects is not None:
        for x0, x1, y0, y1 in scene.obstacle_rects[0].tolist():
            obstacles.append(rectangle(x0 - .03, x1 + .03, y0 - .03, y1 + .03))
    return WorldRectangleScene(body, np.asarray(obstacles).reshape(-1, 4, 2),
                               rectangle(-cfg.world_x, cfg.world_x, -cfg.world_y, cfg.world_y))


def certify_world_path(path, scene, margin=1e-6, max_depth=18):
    """Independent recursive world-SAT test with the same rigid-motion bound."""
    path = np.asarray(path, dtype=np.float64)
    if path.ndim != 2 or path.shape[-1] not in (3, 6) or len(path) < 2 or not np.isfinite(path).all():
        raise ValueError("finite path with at least two pose3/state6 rows required")
    if not math.isfinite(margin) or margin <= 0 or not isinstance(max_depth, int) or max_depth < 1:
        raise ValueError("positive margin and integer depth required")
    q = path[:, :3]
    radius = np.linalg.norm(scene.body, axis=-1).max()
    margins = scene.clearance(q)
    queries, minimum = len(q), float(margins.min())
    unsafe = np.flatnonzero(margins <= margin)
    if len(unsafe):
        return ContinuousCertificate(False, "sampled_clearance_violation", max(0, int(unsafe[0]) - 1),
                                     minimum, queries, 0)
    left, right = q[:-1], q[1:]
    cl, cr = margins[:-1], margins[1:]
    indices = np.arange(len(left))
    for depth in range(max_depth + 1):
        motion = np.linalg.norm(right[:, :2] - left[:, :2], axis=-1)
        motion += radius * np.abs(angle_wrap(right[:, 2] - left[:, 2]))
        unresolved = np.maximum(cl, cr) <= motion + margin
        if not unresolved.any():
            return ContinuousCertificate(True, None, None, minimum, queries, depth)
        left, right, cl, cr, indices = (v[unresolved] for v in (left, right, cl, cr, indices))
        if depth == max_depth:
            return ContinuousCertificate(False, "subdivision_limit", int(indices[0]), minimum, queries, depth)
        middle = .5 * (left + right)
        middle[:, 2] = angle_wrap(left[:, 2] + .5 * angle_wrap(right[:, 2] - left[:, 2]))
        cm = scene.clearance(middle)
        queries += len(middle)
        minimum = min(minimum, float(cm.min()))
        unsafe = np.flatnonzero(cm <= margin)
        if len(unsafe):
            return ContinuousCertificate(False, "interior_clearance_violation", int(indices[unsafe[0]]),
                                         minimum, queries, depth + 1)
        left, right, cl, cr, indices = (np.concatenate(parts) for parts in (
            (left, middle), (middle, right), (cl, cm), (cm, cr), (indices, indices)))
    raise AssertionError("unreachable")
