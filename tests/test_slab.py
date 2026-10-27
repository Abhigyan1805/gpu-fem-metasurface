"""Analytic check: laterally uniform stacks vs the transmission-line result."""

import numpy as np
import pytest

from fem.analytic import stack_gamma
from fem.designs import get_design
from fem.solver import UnitCellFEM

F = np.array([2.0, 10.0, 18.0])


@pytest.mark.parametrize(
    "name,order,h_diel",
    [("slab_class5", 1, 0.3), ("slab_class9", 1, 0.3), ("slab_class5", 2, 0.6), ("slab_class9", 2, 1.0)],
)
def test_grounded_slab(name, order, h_diel):
    d = get_design(name)
    h_air = 1.0 if order == 1 else 2.0
    r = UnitCellFEM(d, "coarse", order=order, h_air=h_air, h_diel=h_diel).sweep(F, progress=False)
    g = stack_gamma(F, d.eps_c, d.h_mm, d.superstrate)
    assert np.max(np.abs(20 * np.log10(np.abs(r.s11) / np.abs(g)))) < 0.01
    assert np.max(np.abs(np.angle(r.s11 / g, deg=True))) < 1.0


@pytest.mark.parametrize("rs", [50.0, 377.0])
def test_resistive_sheet(rs):
    """Uniform R_s sheet on a grounded slab (Salisbury screen): tests the sheet term."""
    from dataclasses import replace

    from fem.mesh import build_mesh

    d = replace(get_design("class8"), r_sheet=rs)
    # a full-cell metal square: pattern covering the whole sheet plane
    mesh = build_mesh(replace(d, dxf=None), "coarse", h_diel=1.0)
    mesh.tris["metal"] = mesh.tris["sheet"]
    r = UnitCellFEM(d, mesh=mesh, order=2).sweep(F, progress=False)
    g = stack_gamma(F, d.eps_c, d.h_mm, d.superstrate, z_sheet=rs)
    assert np.max(np.abs(np.abs(r.s11) - np.abs(g))) < 2e-3
    assert np.max(np.abs(np.angle(r.s11 / g, deg=True))) < 1.0
