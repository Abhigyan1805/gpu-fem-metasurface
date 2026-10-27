import numpy as np
import pytest

from fem.designs import DESIGNS, get_design
from fem.geometry import load_pattern, metal_fraction
from fem.mesh import build_mesh


@pytest.mark.parametrize("name", DESIGNS)
def test_mesh_conforms_to_pattern(name):
    d = get_design(name)
    m = build_mesh(d, "coarse")
    x = m.nodes
    t = m.tris["metal"]
    area = 0.5 * np.linalg.norm(np.cross(x[t[:, 1]] - x[t[:, 0]], x[t[:, 2]] - x[t[:, 0]]), axis=1).sum()
    assert area == pytest.approx(metal_fraction(load_pattern(d.dxf)) * 100.0, rel=1e-9)
    # tet volumes positive-or-negative but nonzero, total = cell volume
    v = np.abs(np.einsum("ij,ij->i", np.cross(x[m.tets[:, 1]] - x[m.tets[:, 0]], x[m.tets[:, 2]] - x[m.tets[:, 0]]), x[m.tets[:, 3]] - x[m.tets[:, 0]])) / 6
    depth = d.h_mm * (2 if d.superstrate else 1) + m.info["air_mm"]
    assert v.sum() == pytest.approx(100.0 * depth, rel=1e-9)


def test_periodic_map_is_translation():
    m = build_mesh(get_design("class9"), "coarse")
    s = np.flatnonzero(m.periodic != np.arange(len(m.nodes)))
    d = m.nodes[s] - m.nodes[m.periodic[s]]
    assert np.allclose(np.mod(d[:, :2] + 5, 10) - 5, 0, atol=1e-9)
    assert np.allclose(d[:, 2], 0)
    assert np.all(m.periodic[m.periodic] == m.periodic)


def test_too_coarse_mesh_is_rejected():
    with pytest.raises(RuntimeError, match="a/2"):
        build_mesh(get_design("slab_class9"), "coarse", h_air=3.0, h_diel=1.5)
