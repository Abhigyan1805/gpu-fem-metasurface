"""Transmission-line reference for a laterally uniform stack at normal incidence (e^{jwt}).

Model: PEC ground at z = -h, substrate eps_c,
optional uniform sheet Z_s at z = 0, optional identical superstrate. The reflection
coefficient is referenced to the top air interface.
"""

from __future__ import annotations

import numpy as np

C0 = 299_792_458.0
MU0 = 4e-7 * np.pi * 1.00000000055  # CODATA 2018 mu0 (close enough to 4 pi e-7)
ETA0 = MU0 * C0


def k0_per_mm(f_ghz):
    return 2.0 * np.pi * np.asarray(f_ghz, dtype=float) * 1e9 / C0 * 1e-3


def copper_zs(f_ghz, sigma: float = 5.8e7):
    w = 2.0 * np.pi * np.asarray(f_ghz, dtype=float) * 1e9
    return (1.0 + 1.0j) * np.sqrt(w * MU0 / (2.0 * sigma))


def stack_gamma(f_ghz, eps_c: complex, h_mm: float, superstrate: bool = False, z_sheet=None):
    """Gamma at the top air interface. ``z_sheet``: uniform sheet impedance at z = 0 (ohm/sq)."""
    k0 = k0_per_mm(f_ghz)
    n = np.sqrt(complex(eps_c))
    if n.imag > 0:
        n = -n
    beta = k0 * n
    eta1 = ETA0 / n
    z = 1j * eta1 * np.tan(beta * h_mm)
    if z_sheet is not None:
        z = 1.0 / (1.0 / z + 1.0 / np.asarray(z_sheet))
    if superstrate:
        t = np.tan(beta * h_mm)
        z = eta1 * (z + 1j * eta1 * t) / (eta1 + 1j * z * t)
    return (z - ETA0) / (z + ETA0)
