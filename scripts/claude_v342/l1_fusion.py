"""Sheaf consensus fusion for LLM latent collaboration (batch L1; DESIGN_LLM_LATENT_SHEAF_GBP.md, report D8-82).

Energy per coordinate c of each agent's K x D_e slot tensor (F = I, diagonal precision lam > 0, edge weight w > 0):

    E = sum_i 1/2 * lam_i[c] * (x_i[c] - z_i[c])^2  +  sum_{(i,j) in edges} 1/2 * w * (x_i[c] - x_j[c])^2

Coordinates decouple, so every coordinate is an N x N linear system (N = number of agents).

* ``fuse_direct``: solves the normal equations (diag(lam) + w L) x = lam z. Exact and differentiable
  (used in training). On a tree it is identical to Gaussian belief propagation at convergence.
* ``gbp_sync``: synchronous GaBP on the edge list, really passing (precision, information) messages so that
  a lost or absent message can be simulated (``drop``) and the wire can be quantised (``wire_dtype``).
  Used at evaluation. The "C" check of the pre-registration asserts gbp_sync(rounds >= diameter, no drop,
  no wire) == fuse_direct.

All functions take ``z`` and ``lam`` of shape (N, *rest); the solve runs in float32 and returns z.dtype.
"""
from __future__ import annotations

import torch

WIRE_DTYPE = torch.float16


def _edge_weights(edges, w, device):
    w = torch.as_tensor(w, dtype=torch.float32, device=device)
    if w.ndim == 0:
        w = w.expand(len(edges))
    if w.shape[0] != len(edges):
        raise ValueError(f"w has {w.shape[0]} entries for {len(edges)} edges")
    return w


def laplacian(n: int, edges, w, device=None) -> torch.Tensor:
    """Weighted graph Laplacian (N, N) built from an incidence matrix so that gradients reach w."""
    w = _edge_weights(edges, w, device)
    inc = torch.zeros(len(edges), n, dtype=torch.float32, device=device)
    for e, (i, j) in enumerate(edges):
        if i == j or not (0 <= i < n and 0 <= j < n):
            raise ValueError(f"bad edge {(i, j)} for n={n}")
        inc[e, i] = 1.0
        inc[e, j] = -1.0
    return inc.T @ (w[:, None] * inc)


def fuse_direct(z: torch.Tensor, lam: torch.Tensor, edges, w) -> torch.Tensor:
    """Minimiser of the energy, shape like z. lam must be > 0 (the data term keeps A positive definite)."""
    if z.shape != lam.shape:
        raise ValueError(f"z {tuple(z.shape)} and lam {tuple(lam.shape)} differ")
    n = z.shape[0]
    zf = z.reshape(n, -1).to(torch.float32)      # (N, M)
    lf = lam.reshape(n, -1).to(torch.float32)
    L = laplacian(n, edges, w, device=z.device)   # (N, N)
    A = torch.diag_embed(lf.T) + L                # (M, N, N)
    b = (lf * zf).T.unsqueeze(-1)                 # (M, N, 1)
    x = torch.linalg.solve(A, b).squeeze(-1).T    # (N, M)
    return x.reshape(z.shape).to(z.dtype)


def directed_edges(edges, device=None):
    """Directed edge list: 2e = i->j, 2e+1 = j->i. Returns (src, dst) index tensors of length 2E."""
    src, dst = [], []
    for i, j in edges:
        src += [i, j]
        dst += [j, i]
    return (torch.tensor(src, dtype=torch.long, device=device),
            torch.tensor(dst, dtype=torch.long, device=device))


def gbp_sync(z: torch.Tensor, lam: torch.Tensor, edges, w, rounds: int,
             drop: torch.Tensor | None = None, wire_dtype=None):
    """Synchronous Gaussian BP with F = I.

    drop: optional bool tensor (2E, *rest) or broadcastable to it; True = the message on that directed edge
          is NOT received (absent agent, packet loss). A dropped message counts as nothing at the receiver,
          both in its belief and in the messages it sends on.
    wire_dtype: if given, the (precision, mean) pair on every message is rounded through that dtype
          (the design's fp16 wire: K x D_e means + K x D_e precisions).
    Returns (belief mean, belief precision), both shaped like z.
    """
    if z.shape != lam.shape:
        raise ValueError(f"z {tuple(z.shape)} and lam {tuple(lam.shape)} differ")
    n = z.shape[0]
    dev = z.device
    zf = z.reshape(n, -1).to(torch.float32)
    lf = lam.reshape(n, -1).to(torch.float32)
    m_cols = zf.shape[1]
    we = _edge_weights(edges, w, dev)
    wd = we.repeat_interleave(2)[:, None]        # (2E, 1)
    src, dst = directed_edges(edges, dev)
    e2 = src.shape[0]
    rev = torch.arange(e2, device=dev) ^ 1       # reverse direction of each directed edge
    eta = lf * zf                                # prior information vectors
    keep = None
    if drop is not None:
        keep = (~torch.as_tensor(drop, device=dev).bool()).to(torch.float32)
        keep = keep.reshape(e2, -1) if keep.numel() == e2 * m_cols else keep.reshape(e2, -1)
    p = torch.zeros(e2, m_cols, device=dev)
    h = torch.zeros(e2, m_cols, device=dev)
    for _ in range(rounds):
        p_in = torch.zeros(n, m_cols, device=dev).index_add_(0, dst, p)
        h_in = torch.zeros(n, m_cols, device=dev).index_add_(0, dst, h)
        p_ex = lf[src] + p_in[src] - p[rev]      # node src's belief without the message from dst
        h_ex = eta[src] + h_in[src] - h[rev]
        p_new = wd * p_ex / (wd + p_ex)          # marginalise x_src out of the pairwise factor
        m_new = h_ex / p_ex
        if wire_dtype is not None:
            p_new = p_new.to(wire_dtype).to(torch.float32)
            m_new = m_new.to(wire_dtype).to(torch.float32)
        h_new = p_new * m_new
        if keep is not None:
            p_new = p_new * keep
            h_new = h_new * keep
        p, h = p_new, h_new
    p_in = torch.zeros(n, m_cols, device=dev).index_add_(0, dst, p)
    h_in = torch.zeros(n, m_cols, device=dev).index_add_(0, dst, h)
    prec = lf + p_in
    # (lam z + H) / (lam + P) written as z + (H - P z) / (lam + P): with nothing received the belief is
    # bit-for-bit the agent's own z (the absence property the design asserts), not (lam z) / lam.
    mean = zf + (h_in - p_in * zf) / prec
    return mean.reshape(z.shape).to(z.dtype), prec.reshape(z.shape).to(z.dtype)


def tree_diameter(n: int, edges) -> int:
    """Longest shortest path in the (assumed connected) graph; the number of sync rounds GaBP needs on a tree."""
    adj = {i: [] for i in range(n)}
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    best = 0
    for s in range(n):
        dist = {s: 0}
        frontier = [s]
        while frontier:
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        nxt.append(v)
            frontier = nxt
        if len(dist) != n:
            raise ValueError("graph is not connected")
        best = max(best, max(dist.values()))
    return best


def edge_residual(x: torch.Tensor, edges) -> torch.Tensor:
    """Per-edge consensus residual ||x_i - x_j|| over the trailing (K, D) dims: shape (E, *batch)."""
    out = []
    for i, j in edges:
        d = (x[i] - x[j]).to(torch.float32)
        out.append(d.flatten(-2).norm(dim=-1) if d.ndim >= 2 else d.abs())
    return torch.stack(out)


def wire_bytes(k: int, d: int, dtype=WIRE_DTYPE) -> int:
    """Bytes of one message: K x D means + K x D precisions in the wire dtype."""
    return 2 * k * d * torch.empty(0, dtype=dtype).element_size()
