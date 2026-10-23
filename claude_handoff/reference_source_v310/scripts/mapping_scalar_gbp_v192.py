"""CPU reference: dense local Gaussian factors, scalar radio-factor GBP.

This is an independent mathematical reference, not a training implementation.
For node i, E_i = .5 z_i.T Q_i z_i - h_i.T z_i. Every radio edge
(i,j) adds .5 sum_k omega[k] (z_i[k]-z_j[k])**2. All local off-diagonal
terms are retained. Scalarizing the radio interface creates factor-graph loops
even when the *physical* two-agent graph is a tree. Convergence is not general;
on convergence, compare means against the centralized quadratic problem.
GBP variances are not claimed to equal centralized marginal variances.

The solver is unquantized float64. PacketCodec64 is a SEPARATE, lossy 8-bit
protocol reference, not silently inserted into the solver. Neither component
imports torch, touches a GPU, performs I/O, or changes an existing experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class GaussianState:
    """Undefined coordinates have mean NaN, variance inf; no hidden ridge.

    minimum_norm_mean is a diagnostic gauge, NOT an estimate for an
    unidentified coordinate. covariance_pinv is likewise NOT its covariance.
    """

    mean: np.ndarray
    variance: np.ndarray
    minimum_norm_mean: np.ndarray
    covariance_pinv: np.ndarray
    identified: np.ndarray
    rank: int
    normal_residual: float


def _array(x, shape, name):
    a = np.asarray(x, dtype=np.float64)
    if a.shape != shape or not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: expected finite shape {shape}, got {a.shape}")
    return a.copy()


def gaussian_state(q, h, tol=1e-11):
    """Solve a PSD Gaussian factor, retaining explicit improper/flat status."""
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] != q.shape[1] or not q.shape[0]:
        raise ValueError("Q must be a nonempty square matrix")
    d = len(q)
    q = _array(q, (d, d), "Q")
    h = _array(h, (d,), "h")
    scale = max(1.0, float(np.max(np.abs(q))))
    if not np.allclose(q, q.T, rtol=0, atol=tol * scale):
        raise ValueError("Q must be symmetric")
    q = (q + q.T) / 2
    vals, vecs = np.linalg.eigh(q)
    threshold = tol * max(1.0, float(np.max(np.abs(vals))))
    if np.min(vals) < -threshold:
        raise ValueError("Q must be positive semidefinite")
    positive = vals > threshold
    null = vecs[:, ~positive]
    if null.shape[1] and np.linalg.norm(null.T @ h, ord=np.inf) > tol * max(1.0, np.linalg.norm(h)) * 10:
        raise ValueError("h outside range(Q): unbounded, not a flat Gaussian factor")
    pinv = (vecs[:, positive] / vals[positive]) @ vecs[:, positive].T
    gauge = pinv @ h
    identified = np.sum(null * null, axis=1) < tol if null.shape[1] else np.ones(d, dtype=bool)
    mean = np.where(identified, gauge, np.nan)
    variance = np.where(identified, np.diag(pinv), np.inf)
    return GaussianState(mean, variance, gauge, pinv, identified, int(positive.sum()),
                         float(np.max(np.abs(q @ gauge - h))))


def _clean_scalar_factor(precision, eta, tol=1e-10):
    precision = np.asarray(precision, dtype=np.float64).copy()
    eta = np.asarray(eta, dtype=np.float64).copy()
    if precision.shape != eta.shape or not np.all(np.isfinite(precision)) or not np.all(np.isfinite(eta)):
        raise ValueError("nonfinite or mismatched natural parameters")
    threshold = tol * max(1.0, float(np.max(np.abs(precision))))
    if np.min(precision) < -threshold:
        raise ValueError("negative scalar precision")
    precision[precision < 0] = 0
    empty = precision <= threshold
    if np.any(np.abs(eta[empty]) > tol * max(1.0, float(np.max(np.abs(eta)))) * 10):
        raise ValueError("zero precision with nonzero information")
    precision[empty] = 0
    eta[empty] = 0
    return precision, eta


def local_factor_messages(q, h, incoming_precision, incoming_eta):
    """Exact local dense-factor -> scalar messages via marginalization.

    With A=Q+diag(a), C=A^-1, m=C(h+b): alpha=1/diag(C)-a,
    beta=m/diag(C)-b. For an improper A, generalized Schur complements
    integrate only constrained subspaces; unconstrained volume is a constant.
    No regularizer is inserted to manufacture evidence.
    """
    d = len(h)
    a, b = _clean_scalar_factor(incoming_precision, incoming_eta)
    state = gaussian_state(q + np.diag(a), h + b)
    if state.rank == d:
        marginal_precision = 1 / np.diag(state.covariance_pinv)
        return _clean_scalar_factor(marginal_precision - a,
                                    state.minimum_norm_mean * marginal_precision - b)
    precision, eta = np.zeros(d), np.zeros(d)
    total_q, total_h = q + np.diag(a), h + b
    for k in range(d):
        rest = np.arange(d) != k
        if not rest.any():
            precision[k], eta[k] = q[k, k], h[k]
            continue
        sub = gaussian_state(total_q[np.ix_(rest, rest)], total_h[rest])
        row = total_q[k, rest]
        precision[k] = q[k, k] - row @ sub.covariance_pinv @ row
        eta[k] = h[k] - row @ sub.minimum_norm_mean
    return _clean_scalar_factor(precision, eta)


class ScalarFactorGBP:
    """Synchronous, damped scalar-factor GBP with replacement-only caches.

    edges contains unique undirected (i, j, positive scalar-or-D-vector omega).
    Source versions and packet versions are monotone. Repeated same-version
    identical data is a no-op; conflicting equal versions raise, stale data is
    ignored. Updating observations REPLACES a source factor, never adds it.
    """

    def __init__(self, q, h, edges: Iterable[tuple], damping=0.5):
        q = np.asarray(q, dtype=np.float64)
        if q.ndim != 3 or q.shape[1] != q.shape[2]:
            raise ValueError("Q must be N x D x D")
        self.n, self.d = q.shape[:2]
        if self.n < 1 or self.d < 1 or not 0 < damping <= 1:
            raise ValueError("invalid size or damping")
        self.q = _array(q, (self.n, self.d, self.d), "Q")
        self.h = _array(h, (self.n, self.d), "h")
        for i in range(self.n):
            gaussian_state(self.q[i], self.h[i])
        self.damping = float(damping)
        self.source_versions = np.zeros(self.n, dtype=np.int64)
        self.edges = {}
        self.neighbors = [[] for _ in range(self.n)]
        self.messages = {}
        self.message_versions = {}
        for i, j, weight in edges:
            i, j = int(i), int(j)
            if not 0 <= i < self.n or not 0 <= j < self.n or i == j:
                raise ValueError("invalid edge endpoints")
            key = tuple(sorted((i, j)))
            if key in self.edges:
                raise ValueError("duplicate undirected edge")
            w = np.broadcast_to(np.asarray(weight, dtype=np.float64), (self.d,)).copy()
            if not np.all(np.isfinite(w)) or np.any(w <= 0):
                raise ValueError("edge precision must be positive and finite")
            self.edges[key] = w
            self.neighbors[i].append(j)
            self.neighbors[j].append(i)
            for s, r in ((i, j), (j, i)):
                self.messages[(s, r)] = (np.zeros(self.d), np.zeros(self.d))
                self.message_versions[(s, r)] = -1
        self.iteration = 0

    def receive_message(self, sender, receiver, precision, eta, version):
        key = (int(sender), int(receiver))
        if key not in self.messages:
            raise ValueError("message is not on a registered radius edge")
        if not isinstance(version, (int, np.integer)) or version < 0:
            raise ValueError("version must be a nonnegative integer")
        precision = _array(precision, (self.d,), "precision")
        eta = _array(eta, (self.d,), "eta")
        precision, eta = _clean_scalar_factor(precision, eta)
        old_version = self.message_versions[key]
        if version < old_version:
            return False
        if version == old_version:
            old_p, old_h = self.messages[key]
            if not np.array_equal(precision, old_p) or not np.array_equal(eta, old_h):
                raise ValueError("same message version with conflicting payload")
            return False
        self.messages[key] = (precision.copy(), eta.copy())
        self.message_versions[key] = int(version)
        return True

    def replace_source(self, node, q, h, version):
        if not isinstance(node, (int, np.integer)) or not 0 <= node < self.n:
            raise ValueError("invalid source node")
        if not isinstance(version, (int, np.integer)) or version < 0:
            raise ValueError("source version must be a nonnegative integer")
        q, h = _array(q, (self.d, self.d), "Q"), _array(h, (self.d,), "h")
        gaussian_state(q, h)
        old_version = self.source_versions[node]
        if version < old_version:
            return False
        if version == old_version:
            if not np.array_equal(q, self.q[node]) or not np.array_equal(h, self.h[node]):
                raise ValueError("same source version with conflicting factor")
            return False
        self.q[node], self.h[node] = q, h
        self.source_versions[node] = version
        return True

    def incoming(self, node):
        a, b = np.zeros(self.d), np.zeros(self.d)
        for sender in self.neighbors[node]:
            p, h = self.messages[(sender, node)]
            a += p
            b += h
        return a, b

    def belief(self, node):
        a, b = self.incoming(node)
        return gaussian_state(self.q[node] + np.diag(a), self.h[node] + b)

    def means(self):
        return np.stack([self.belief(i).mean for i in range(self.n)])

    def minimum_norm_means(self):
        return np.stack([self.belief(i).minimum_norm_mean for i in range(self.n)])

    def normal_residual(self):
        z = self.minimum_norm_means()
        grad = np.einsum("nij,nj->ni", self.q, z) - self.h
        for (i, j), w in self.edges.items():
            r = w * (z[i] - z[j])
            grad[i] += r
            grad[j] -= r
        return float(np.max(np.abs(grad)))

    def energy(self, z=None):
        z = self.minimum_norm_means() if z is None else _array(z, (self.n, self.d), "z")
        value = .5 * np.einsum("ni,nij,nj->", z, self.q, z) - np.sum(self.h * z)
        for (i, j), w in self.edges.items():
            value += .5 * np.sum(w * (z[i] - z[j]) ** 2)
        return float(value)

    def step(self):
        proposals = {}
        delta = 0.0
        version = max(self.message_versions.values(), default=-1) + 1
        for i in range(self.n):
            a, b = self.incoming(i)
            alpha, beta = local_factor_messages(self.q[i], self.h[i], a, b)
            for j in self.neighbors[i]:
                reverse_p, reverse_h = self.messages[(j, i)]
                cp, ch = _clean_scalar_factor(alpha + a - reverse_p,
                                              beta + b - reverse_h)
                w = self.edges[tuple(sorted((i, j)))]
                p = w * cp / (w + cp)
                h = w * ch / (w + cp)
                old_p, old_h = self.messages[(i, j)]
                p = self.damping * p + (1 - self.damping) * old_p
                h = self.damping * h + (1 - self.damping) * old_h
                delta = max(delta, float(np.max(np.abs(p - old_p))), float(np.max(np.abs(h - old_h))))
                proposals[(i, j)] = p, h
        for (i, j), (p, h) in proposals.items():
            self.receive_message(i, j, p, h, version)
        self.iteration += 1
        return delta

    def run(self, max_steps=10000, atol=1e-9):
        if max_steps < 1 or atol <= 0:
            raise ValueError("invalid iteration limit or tolerance")
        delta = np.inf
        for steps in range(1, max_steps + 1):
            delta = self.step()
            residual = self.normal_residual()
            if delta < atol and residual < atol:
                return {"converged": True, "steps": steps, "message_delta": delta,
                        "normal_residual": residual, "quantized": False}
        return {"converged": False, "steps": max_steps, "message_delta": delta,
                "normal_residual": self.normal_residual(), "quantized": False}

    def centralized(self):
        """Exact dense reference to the SAME finite-edge-precision energy."""
        d = self.d
        q = np.zeros((self.n * d, self.n * d))
        for i in range(self.n):
            q[i*d:(i+1)*d, i*d:(i+1)*d] = self.q[i]
        for (i, j), w in self.edges.items():
            for k in range(d):
                a, b = i*d+k, j*d+k
                q[a, a] += w[k]
                q[b, b] += w[k]
                q[a, b] -= w[k]
                q[b, a] -= w[k]
        return gaussian_state(q, self.h.reshape(-1))


class PacketCodec64:
    """SEPARATE fixed-codebook lossy packet protocol, no solver integration.

    20-byte network-order header + 64 precision codes + 64 mean codes = 148 B.
    Means: 255 symmetric levels [-8,8], exact zero code127; code255 reserved.
    Precision: exact-empty code0, 255 log levels [1e-4,1e4]. Empty mean is
    canonical zero. Fixed tables are deployment constants, not free per-packet
    side information. Clipping is rejected unless explicitly requested.
    This test protocol has no checksum, authentication, or observation-origin
    lineage; valid-code bit flips are not detected. Sequence numbers provide
    cache idempotence, NOT general cyclic common-information correction.
    """

    HEADER = struct.Struct("!4sBBBBIII")
    MAGIC = b"SG92"
    DIM = 64
    MEAN_MIN, MEAN_MAX = -8.0, 8.0
    PRECISION_MIN, PRECISION_MAX = 1e-4, 1e4
    MEANS = np.linspace(MEAN_MIN, MEAN_MAX, 255, dtype=np.float64)
    PRECISIONS = np.concatenate(([0.0], np.geomspace(PRECISION_MIN, PRECISION_MAX, 255)))
    MEANS.setflags(write=False)
    PRECISIONS.setflags(write=False)

    @classmethod
    def encode(cls, mean, precision, sender, receiver, sequence, allow_clip=False):
        mean = _array(mean, (64,), "mean")
        precision = _array(precision, (64,), "precision")
        if np.any(precision < 0):
            raise ValueError("negative precision")
        ids = (sender, receiver, sequence)
        if any(not isinstance(x, (int, np.integer)) or not 0 <= x <= 0xffffffff for x in ids):
            raise ValueError("header ids must be uint32")
        if sender == receiver:
            raise ValueError("radio packet cannot be self-addressed")
        active = precision > 0
        out_of_range = ((mean[active] < cls.MEAN_MIN) | (mean[active] > cls.MEAN_MAX)).any()
        out_of_range |= ((precision[active] < cls.PRECISION_MIN) | (precision[active] > cls.PRECISION_MAX)).any()
        if out_of_range and not allow_clip:
            raise ValueError("quantizer saturation requires explicit allow_clip=True")
        mc = np.full(64, 127, dtype=np.uint8)
        pc = np.zeros(64, dtype=np.uint8)
        mc[active] = np.rint((np.clip(mean[active], cls.MEAN_MIN, cls.MEAN_MAX) - cls.MEAN_MIN)
                             * 254 / (cls.MEAN_MAX - cls.MEAN_MIN)).astype(np.uint8)
        log_ratio = (np.log(np.clip(precision[active], cls.PRECISION_MIN, cls.PRECISION_MAX))
                     - np.log(cls.PRECISION_MIN)) / np.log(cls.PRECISION_MAX / cls.PRECISION_MIN)
        pc[active] = (1 + np.rint(log_ratio * 254)).astype(np.uint8)
        return cls.HEADER.pack(cls.MAGIC, 1, 64, 8, 8, *map(int, ids)) + pc.tobytes() + mc.tobytes()

    @classmethod
    def decode(cls, packet):
        if not isinstance(packet, bytes) or len(packet) != cls.HEADER.size + 128:
            raise ValueError("packet must be exactly 148 bytes")
        magic, version, dim, mbits, pbits, sender, receiver, sequence = cls.HEADER.unpack_from(packet)
        if (magic, version, dim, mbits, pbits) != (cls.MAGIC, 1, 64, 8, 8) or sender == receiver:
            raise ValueError("incompatible packet header")
        pc = np.frombuffer(packet, dtype=np.uint8, count=64, offset=cls.HEADER.size)
        mc = np.frombuffer(packet, dtype=np.uint8, count=64, offset=cls.HEADER.size + 64)
        if np.any(mc == 255) or np.any((pc == 0) & (mc != 127)):
            raise ValueError("reserved mean code or noncanonical empty message")
        mean, precision = cls.MEANS[mc].copy(), cls.PRECISIONS[pc].copy()
        return {"sender": sender, "receiver": receiver, "sequence": sequence,
                "mean": mean, "precision": precision, "eta": mean * precision,
                "packet_bytes": len(packet), "quantized": True}

