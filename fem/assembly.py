"""DOF numbering with periodic merging, and frequency-independent global matrices.

The system at wavenumber k0 (rad/mm) is

    A(k0) = S - k0^2 (eps_sub M_sub + eps_sup M_sup + M_air) + j k0 B_port + j k0 eta0 / Z_s B_sheet

All parts are symmetric and share the sparsity pattern of S, so only the upper triangle is
stored, and a new frequency only recombines value arrays. Element blocks are computed in
chunks with torch on any device; the pattern is built with torch.unique and every part is
scattered into it with torch.searchsorted + index_add.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch

from .basis import TET_FACES, n_local, tet_local, tri_local
from .elements import TET_EDGES, TRI_EDGES, tet_matrices, tri_mass, tri_moment
from .mesh import REGION_AIR, REGION_SUB, REGION_SUP, Mesh

CHUNK = {1: 8192, 2: 1024}


@dataclass
class Dofs:
    order: int
    n_dofs: int
    n_edges: int
    n_faces: int
    tet_perm: torch.Tensor  # (T, 4) local vertex order (sorted by merged id)
    tet_map: torch.Tensor  # (T, n_local) global dof ids
    edge_keys: torch.Tensor
    face_keys: torch.Tensor
    n_nodes: int


def _sorted(conn: torch.Tensor, per: torch.Tensor):
    g = per[conn]
    gs, perm = torch.sort(g, dim=1)
    if torch.any(gs[:, 1:] == gs[:, :-1]):
        raise RuntimeError("degenerate element after periodic merge (element spans the cell?)")
    return gs, perm


def _edge_keys(gs, local, n):
    return torch.stack([gs[:, a] * n + gs[:, b] for a, b in local], dim=1)


def _face_keys(gs, local, n):
    return torch.stack([(gs[:, i] * n + gs[:, j]) * n + gs[:, k] for i, j, k in local], dim=1)


def number_dofs(mesh: Mesh, order: int, device: torch.device) -> Dofs:
    per = torch.as_tensor(mesh.periodic, device=device)
    n = int(len(mesh.nodes))
    gs, perm = _sorted(torch.as_tensor(mesh.tets, device=device), per)
    ek = _edge_keys(gs, TET_EDGES, n)
    edge_keys, e_id = torch.unique(ek.reshape(-1), return_inverse=True)
    e_id = e_id.reshape(ek.shape)
    E = int(edge_keys.numel())
    if order == 1:
        return Dofs(1, E, E, 0, perm, e_id, edge_keys, torch.zeros(0, dtype=torch.long, device=device), n)
    fk = _face_keys(gs, TET_FACES, n)
    face_keys, f_id = torch.unique(fk.reshape(-1), return_inverse=True)
    f_id = f_id.reshape(fk.shape)
    F = int(face_keys.numel())
    fmap = torch.stack([2 * E + f_id, 2 * E + F + f_id], dim=2).reshape(len(f_id), -1)
    tmap = torch.cat([e_id, E + e_id, fmap], dim=1)
    return Dofs(2, 2 * E + 2 * F, E, F, perm, tmap, edge_keys, face_keys, n)


def _lookup(keys: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    idx = torch.searchsorted(keys, q.reshape(-1)).reshape(q.shape)
    if not torch.equal(keys[idx.clamp(max=len(keys) - 1)], q):
        raise RuntimeError("surface entity not found among volume entities")
    return idx


def tri_dofs(mesh: Mesh, dofs: Dofs, tris: np.ndarray, device: torch.device):
    """Sorted-vertex permutation (T, 3) and global dof map (T, n_local) of triangles."""
    per = torch.as_tensor(mesh.periodic, device=device)
    gs, perm = _sorted(torch.as_tensor(tris, device=device), per)
    e = _lookup(dofs.edge_keys, _edge_keys(gs, TRI_EDGES, dofs.n_nodes))
    if dofs.order == 1:
        return perm, e
    f = _lookup(dofs.face_keys, _face_keys(gs, ((0, 1, 2),), dofs.n_nodes))[:, 0]
    E, F = dofs.n_edges, dofs.n_faces
    return perm, torch.cat([e, E + e, (2 * E + f)[:, None], (2 * E + F + f)[:, None]], dim=1)


@dataclass
class System:
    """Upper-triangular shared-pattern CSR parts on the free DOFs (float64), port vectors."""

    indptr: np.ndarray
    indices: np.ndarray
    parts: dict[str, np.ndarray]
    port_moment: np.ndarray  # (n_free, 3): int_port N dS (load and mode average)
    port_area: float
    n_free: int
    n_dofs: int
    order: int
    free: np.ndarray

    def upper(self, coeffs: dict[str, complex]) -> sp.csr_matrix:
        data = np.zeros(len(self.indices), dtype=np.complex128)
        for name, c in coeffs.items():
            if c != 0 and name in self.parts:
                data += c * self.parts[name]
        return sp.csr_matrix((data, self.indices, self.indptr), shape=(self.n_free, self.n_free))

    @staticmethod
    def full(u: sp.csr_matrix) -> sp.csr_matrix:
        return (u + sp.triu(u, k=1, format="csr").T).tocsr()


def assemble(mesh: Mesh, *, pec_metal: bool, order: int = 1, device: str | torch.device = "cpu") -> System:
    device = torch.device(device)
    f64 = torch.float64
    nodes = torch.as_tensor(mesh.nodes, dtype=f64, device=device)
    dofs = number_dofs(mesh, order, device)
    tets = torch.as_tensor(mesh.tets, device=device)
    tets_sorted = torch.gather(tets, 1, dofs.tet_perm)

    # Dirichlet: ground plane, plus metal for PEC. All dofs of those triangles are fixed.
    fixed = torch.zeros(dofs.n_dofs, dtype=torch.bool, device=device)
    for t in [mesh.tris["ground"]] + ([mesh.tris["metal"]] if pec_metal else []):
        if len(t):
            fixed[tri_dofs(mesh, dofs, t, device)[1].reshape(-1)] = True
    free = torch.nonzero(~fixed).flatten()
    new_id = torch.full((dofs.n_dofs,), -1, dtype=torch.long, device=device)
    new_id[free] = torch.arange(free.numel(), device=device)
    nf = int(free.numel())

    nl = n_local(order, "tet")
    iu, ju = torch.triu_indices(nl, nl, device=device)

    def upper_entries(dmap, iu, ju):
        r = new_id[dmap[:, iu]]
        c = new_id[dmap[:, ju]]
        lo = torch.minimum(r, c)
        hi = torch.maximum(r, c)
        keep = lo >= 0
        return lo * nf + hi, keep

    # Pattern from all tets.
    keys, keep = upper_entries(dofs.tet_map, iu, ju)
    pattern = torch.unique(keys[keep])
    del keys, keep

    parts = {name: torch.zeros(pattern.numel(), dtype=f64, device=device) for name in ("S", "M_sub", "M_sup", "M_air")}
    reg = torch.as_tensor(mesh.tet_region, device=device)
    for start in range(0, len(tets), CHUNK[order]):
        sl = slice(start, start + CHUNK[order])
        x = nodes[tets_sorted[sl]]
        s_loc, m_loc = tet_matrices(x) if order == 1 else tet_local(x, order)
        k, kp = upper_entries(dofs.tet_map[sl], iu, ju)
        pos = torch.searchsorted(pattern, k.clamp(min=0))
        parts["S"].index_add_(0, pos[kp], s_loc[:, iu, ju][kp])
        for name, r in (("M_sub", REGION_SUB), ("M_sup", REGION_SUP), ("M_air", REGION_AIR)):
            sel = (reg[sl] == r)[:, None] & kp
            if sel.any():
                parts[name].index_add_(0, pos[sel], m_loc[:, iu, ju][sel])

    ntl = n_local(order, "tri")
    it, jt = torch.triu_indices(ntl, ntl, device=device)

    def surface(tris):
        perm, dmap = tri_dofs(mesh, dofs, tris, device)
        x = nodes[torch.gather(torch.as_tensor(tris, device=device), 1, perm)]
        if order == 1:
            m, nint = tri_mass(x), tri_moment(x)
        else:
            m, nint = tri_local(x, order)
        return dmap, m, nint

    def add_surface(name, tris):
        if len(tris) == 0:
            return
        dmap, m, _ = surface(tris)
        k, kp = upper_entries(dmap, it, jt)
        pos = torch.searchsorted(pattern, k.clamp(min=0))
        if not torch.equal(pattern[pos[kp]], k[kp]):
            raise RuntimeError(f"{name} entries outside the volume pattern")
        data = torch.zeros(pattern.numel(), dtype=f64, device=device)
        data.index_add_(0, pos[kp], m[:, it, jt][kp])
        parts[name] = data

    add_surface("B_port", mesh.tris["port"])
    if not pec_metal:
        add_surface("B_sheet", mesh.tris["metal"])

    dmap, _, nint = surface(mesh.tris["port"])
    pm = torch.zeros((dofs.n_dofs, 3), dtype=f64, device=device)
    pm.index_add_(0, dmap.reshape(-1), nint.reshape(-1, 3))
    xp = nodes[torch.as_tensor(mesh.tris["port"], device=device)]
    area = float(0.5 * torch.linalg.norm(torch.cross(xp[:, 1] - xp[:, 0], xp[:, 2] - xp[:, 0], dim=-1), dim=-1).sum())

    rows = pattern // nf
    indptr = torch.zeros(nf + 1, dtype=torch.long, device=device)
    indptr[1:] = torch.cumsum(torch.bincount(rows, minlength=nf), 0)
    return System(
        indptr=indptr.cpu().numpy(),
        indices=(pattern % nf).cpu().numpy(),
        parts={k: v.cpu().numpy() for k, v in parts.items()},
        port_moment=pm[free].cpu().numpy(),
        port_area=area,
        n_free=nf,
        n_dofs=dofs.n_dofs,
        order=order,
        free=free.cpu().numpy(),
    )
