"""Element-matrix unit tests and patch tests for the Whitney elements."""

import numpy as np
import pytest
import torch

from fem.elements import TET_EDGES, tet_eval, tet_gradients, tet_matrices, tri_mass, tri_moment

rng = np.random.default_rng(1)


def random_tets(n=5):
    x = rng.normal(size=(n, 4, 3))
    return torch.as_tensor(x)


def mc_points(n):
    """Uniform barycentric samples in a tetrahedron."""
    u = rng.exponential(size=(n, 4))
    return u / u.sum(1, keepdims=True)


def test_mass_matrix_matches_quadrature():
    x = random_tets(3)
    _, m = tet_matrices(x)
    _, vol = tet_gradients(x)
    # exact degree-2 quadrature on a tetrahedron (Keast 4-point rule)
    a, b = 0.5854101966249685, 0.1381966011250105
    pts = np.array([[a, b, b, b], [b, a, b, b], [b, b, a, b], [b, b, b, a]])
    for t in range(3):
        acc = np.zeros((6, 6))
        for p in pts:
            bary = torch.as_tensor(p)[None].repeat(1, 1)
            n = tet_eval(x[t : t + 1], bary)[0].numpy()
            acc += n @ n.T / 4
        np.testing.assert_allclose(m[t].numpy(), acc * vol[t].item(), rtol=1e-10, atol=1e-12)


def test_curl_by_finite_difference():
    x = random_tets(1)
    s, _ = tet_matrices(x)
    grad, vol = tet_gradients(x)
    # field of basis e at a point: N = l_a grad l_b - l_b grad l_a; curl via central differences
    inv = torch.linalg.inv(torch.cat([torch.ones(1, 4, 1, dtype=x.dtype), x], -1))[0].numpy()
    def basis(e, r):
        lam = np.array([1, *r]) @ inv
        a, b = TET_EDGES[e]
        g = grad[0].numpy()
        return lam[a] * g[b] - lam[b] * g[a]
    r0 = x[0].mean(0).numpy()
    curls = []
    for e in range(6):
        J = np.zeros((3, 3))
        for k in range(3):
            d = np.zeros(3); d[k] = 1e-6
            J[:, k] = (basis(e, r0 + d) - basis(e, r0 - d)) / 2e-6
        curls.append([J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]])
    c = np.array(curls)
    np.testing.assert_allclose(s[0].numpy(), c @ c.T * vol.item(), rtol=1e-6, atol=1e-8)


def test_patch_constant_field():
    """Edge DOFs E0.t_e reproduce a constant field exactly: zero curl energy, exact mass energy."""
    x = random_tets(4)
    s, m = tet_matrices(x)
    _, vol = tet_gradients(x)
    e0 = np.array([0.3, -1.1, 0.7])
    for t in range(4):
        xt = x[t].numpy()
        dof = np.array([e0 @ (xt[b] - xt[a]) for a, b in TET_EDGES])
        assert abs(dof @ s[t].numpy() @ dof) < 1e-10
        assert dof @ m[t].numpy() @ dof == pytest.approx(vol[t].item() * e0 @ e0, rel=1e-10)
        bary = torch.as_tensor(mc_points(3))
        vals = tet_eval(x[t : t + 1].repeat(3, 1, 1), bary).numpy()
        np.testing.assert_allclose(np.einsum("e,pek->pk", dof, vals), np.tile(e0, (3, 1)), atol=1e-10)


def test_gradient_in_kernel():
    """grad(phi) for linear phi has zero curl: S @ dofs(grad phi) = 0."""
    x = random_tets(3)
    s, _ = tet_matrices(x)
    for t in range(3):
        phi = rng.normal(size=4)
        dof = np.array([phi[b] - phi[a] for a, b in TET_EDGES])
        np.testing.assert_allclose(s[t].numpy() @ dof, 0, atol=1e-10)


def test_triangle_trace():
    tri = torch.as_tensor(rng.normal(size=(2, 3, 3)))
    m = tri_mass(tri)
    nint = tri_moment(tri)
    e0 = rng.normal(size=3)
    for t in range(2):
        p = tri[t].numpy()
        nrm = np.cross(p[1] - p[0], p[2] - p[0])
        area = 0.5 * np.linalg.norm(nrm)
        nrm /= np.linalg.norm(nrm)
        et = e0 - nrm * (nrm @ e0)
        dof = np.array([e0 @ (p[b] - p[a]) for a, b in ((0, 1), (0, 2), (1, 2))])
        assert dof @ m[t].numpy() @ dof == pytest.approx(area * et @ et, rel=1e-10)
        np.testing.assert_allclose(dof @ nint[t].numpy(), area * et, atol=1e-10)
