"""Hierarchical Nedelec (first kind) bases of order 1 and 2 on tetrahedra, with traces.

Local vertices are always sorted by global (periodic-merged) node id, so every edge is
oriented low -> high and every face has a canonical vertex order: no sign bookkeeping.

Order 1 (6 DOFs): Whitney  W_ab = l_a grad l_b - l_b grad l_a.
Order 2 (20 DOFs, Webb 1999 "LT/QN"):
    6 Whitney + 6 edge gradients  G_ab = grad(l_a l_b)
    + 2 per face (i<j<k):  g_k - g_i,  g_j - g_i   with  g_i = l_j l_k grad l_i.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.special import roots_jacobi

from .elements import TET_EDGES, TRI_EDGES

TET_FACES = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))


def _gj01(n: int, alpha: float):
    """Gauss-Jacobi on [0, 1] with weight (1 - u)^alpha."""
    x, w = roots_jacobi(n, alpha, 0.0)
    return (1.0 + x) / 2.0, w / 2.0 ** (alpha + 1.0)


def tet_rule(n: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Conical-product rule on the unit tetrahedron: barycentrics (Q, 4), weights summing to 1/6.
    Exact for polynomials of degree <= 2n - 1."""
    u, wu = _gj01(n, 2.0)
    v, wv = _gj01(n, 1.0)
    s, ws = _gj01(n, 0.0)
    U, V, S = np.meshgrid(u, v, s, indexing="ij")
    W = np.einsum("i,j,k->ijk", wu, wv, ws)
    x = U
    y = V * (1 - U)
    z = S * (1 - U) * (1 - V)
    bary = np.stack([1 - x - y - z, x, y, z], -1).reshape(-1, 4)
    return bary, W.reshape(-1)


def tri_rule(n: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Conical-product rule on the unit triangle: barycentrics (Q, 3), weights summing to 1/2."""
    u, wu = _gj01(n, 1.0)
    v, wv = _gj01(n, 0.0)
    U, V = np.meshgrid(u, v, indexing="ij")
    W = np.einsum("i,j->ij", wu, wv)
    x = U
    y = V * (1 - U)
    bary = np.stack([1 - x - y, x, y], -1).reshape(-1, 3)
    return bary, W.reshape(-1)


def n_local(order: int, simplex: str) -> int:
    if simplex == "tet":
        return 6 if order == 1 else 20
    return 3 if order == 1 else 8


def eval_basis(grad: torch.Tensor, lam: torch.Tensor, order: int, faces, edges, with_curl: bool):
    """Values (T, Q, n, 3) and curls (T, Q, n, 3) of the basis.

    ``grad`` (T, nv, 3) are barycentric gradients (in-plane ones for triangles: their
    values are then the tangential traces), ``lam`` (Q, nv) the quadrature barycentrics.
    """
    T = grad.shape[0]
    Q = lam.shape[0]
    L = lam[None, :, :, None]  # (1, Q, nv, 1)
    G = grad[:, None, :, :]  # (T, 1, nv, 3)
    vals, curls = [], []
    cross = lambda a, b: torch.cross(a.expand(T, Q, 3), b.expand(T, Q, 3), dim=-1)  # noqa: E731
    for a, b in edges:
        vals.append(L[:, :, a] * G[:, :, b] - L[:, :, b] * G[:, :, a])
        if with_curl:
            curls.append(2.0 * cross(G[:, :, a], G[:, :, b]))
    if order >= 2:
        zero = torch.zeros((T, Q, 3), dtype=grad.dtype, device=grad.device)
        for a, b in edges:
            vals.append(L[:, :, a] * G[:, :, b] + L[:, :, b] * G[:, :, a])
            if with_curl:
                curls.append(zero)
        for i, j, k in faces:
            def g(p, q, r):  # l_q l_r grad l_p and its curl
                v = L[:, :, q] * L[:, :, r] * G[:, :, p]
                c = None
                if with_curl:
                    c = L[:, :, q] * cross(G[:, :, r], G[:, :, p]) + L[:, :, r] * cross(G[:, :, q], G[:, :, p])
                return v, c

            gi, ci = g(i, j, k)
            gj, cj = g(j, i, k)
            gk, ck = g(k, i, j)
            vals += [gk - gi, gj - gi]
            if with_curl:
                curls += [ck - ci, cj - ci]
    v = torch.stack([x.expand(T, Q, 3) for x in vals], dim=2)
    c = torch.stack([x.expand(T, Q, 3) for x in curls], dim=2) if with_curl else None
    return v, c


def tet_local(x: torch.Tensor, order: int, nq: int = 3):
    """Curl-curl and mass matrices (T, n, n) for tets with sorted vertices x (T, 4, 3)."""
    from .elements import tet_gradients

    grad, vol = tet_gradients(x)
    bary, w = tet_rule(nq)
    lam = torch.as_tensor(bary, dtype=x.dtype, device=x.device)
    wq = torch.as_tensor(w, dtype=x.dtype, device=x.device) * 6.0  # unit-tet volume 1/6
    v, c = eval_basis(grad, lam, order, TET_FACES, TET_EDGES, with_curl=True)
    m = torch.einsum("tqid,tqjd,q->tij", v, v, wq) * vol[:, None, None]
    s = torch.einsum("tqid,tqjd,q->tij", c, c, wq) * vol[:, None, None]
    return s, m


def tri_local(x: torch.Tensor, order: int, nq: int = 3):
    """Tangential mass (T, n, n) and moments int N dS (T, n, 3) for sorted triangles."""
    from .elements import tri_gradients

    grad, area = tri_gradients(x)
    bary, w = tri_rule(nq)
    lam = torch.as_tensor(bary, dtype=x.dtype, device=x.device)
    wq = torch.as_tensor(w, dtype=x.dtype, device=x.device) * 2.0  # unit-triangle area 1/2
    v, _ = eval_basis(grad, lam, order, ((0, 1, 2),), TRI_EDGES, with_curl=False)
    m = torch.einsum("tqid,tqjd,q->tij", v, v, wq) * area[:, None, None]
    nint = torch.einsum("tqid,q->tid", v, wq) * area[:, None, None]
    return m, nint
