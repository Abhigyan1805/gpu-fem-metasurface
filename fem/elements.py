"""Batched first-order Nedelec (Whitney) element matrices in torch.

All functions take vertex coordinates of shape (T, 4, 3) for tetrahedra or (T, 3, 3)
for triangles and return dense per-element blocks, so they run unchanged on CPU or GPU.
Local edge e = (a, b) carries N_e = l_a grad(l_b) - l_b grad(l_a), oriented a -> b.
"""

from __future__ import annotations

import torch

TET_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
TRI_EDGES = ((0, 1), (0, 2), (1, 2))


def tet_gradients(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Barycentric gradients (T, 4, 3) and volumes (T,)."""
    ones = torch.ones(x.shape[:-1] + (1,), dtype=x.dtype, device=x.device)
    m = torch.cat([ones, x], dim=-1)  # rows [1, x, y, z] per vertex
    inv = torch.linalg.inv(m)  # lambda_i = sum_k inv[k, i] * [1, x, y, z]_k
    grad = inv[:, 1:, :].transpose(1, 2)
    vol = torch.abs(torch.linalg.det(m)) / 6.0
    return grad, vol


def tri_gradients(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """In-plane barycentric gradients (T, 3, 3) and areas (T,) of 3-D triangles."""
    e = torch.stack([x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]], dim=1)  # (T, 2, 3)
    g = e @ e.transpose(1, 2)
    g12 = torch.linalg.solve(g, e)  # rows: grad l1, grad l2
    grad = torch.cat([-(g12[:, :1] + g12[:, 1:]), g12], dim=1)
    area = 0.5 * torch.linalg.norm(torch.cross(e[:, 0], e[:, 1], dim=-1), dim=-1)
    return grad, area


def _whitney_mass(grad: torch.Tensor, measure: torch.Tensor, edges, denom: float) -> torch.Tensor:
    """int N_e . N_f with int l_i l_j = measure (1 + delta_ij) / denom."""
    dots = grad @ grad.transpose(1, 2)  # (T, n, n) grad l_i . grad l_j
    n = grad.shape[1]
    lam = (torch.ones(n, n, dtype=grad.dtype, device=grad.device) + torch.eye(n, dtype=grad.dtype, device=grad.device)) / denom
    ne = len(edges)
    out = torch.empty((grad.shape[0], ne, ne), dtype=grad.dtype, device=grad.device)
    for i, (a, b) in enumerate(edges):
        for j, (c, d) in enumerate(edges):
            out[:, i, j] = (
                lam[a, c] * dots[:, b, d]
                - lam[a, d] * dots[:, b, c]
                - lam[b, c] * dots[:, a, d]
                + lam[b, d] * dots[:, a, c]
            )
    return out * measure[:, None, None]


def tet_matrices(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Curl-curl S (T, 6, 6) and mass M (T, 6, 6) of Whitney tetrahedra."""
    grad, vol = tet_gradients(x)
    curls = torch.stack([2.0 * torch.cross(grad[:, a], grad[:, b], dim=-1) for a, b in TET_EDGES], dim=1)
    s = (curls @ curls.transpose(1, 2)) * vol[:, None, None]
    m = _whitney_mass(grad, vol, TET_EDGES, 20.0)
    return s, m


def tri_mass(x: torch.Tensor) -> torch.Tensor:
    """Tangential-trace mass int_tri N_e . N_f dS (T, 3, 3)."""
    grad, area = tri_gradients(x)
    return _whitney_mass(grad, area, TRI_EDGES, 12.0)


def tri_moment(x: torch.Tensor) -> torch.Tensor:
    """int_tri N_e dS = area/3 (grad l_b - grad l_a), shape (T, 3, 3) [edge, xyz]."""
    grad, area = tri_gradients(x)
    return torch.stack([(grad[:, b] - grad[:, a]) for a, b in TRI_EDGES], dim=1) * (area / 3.0)[:, None, None]


def tet_curl(x: torch.Tensor) -> torch.Tensor:
    """Constant curl of each local basis function (T, 6, 3)."""
    grad, _ = tet_gradients(x)
    return torch.stack([2.0 * torch.cross(grad[:, a], grad[:, b], dim=-1) for a, b in TET_EDGES], dim=1)


def tet_eval(x: torch.Tensor, bary: torch.Tensor) -> torch.Tensor:
    """Basis values at barycentric points bary (T, 4): returns (T, 6, 3)."""
    grad, _ = tet_gradients(x)
    return torch.stack(
        [bary[:, a, None] * grad[:, b] - bary[:, b, None] * grad[:, a] for a, b in TET_EDGES], dim=1
    )
