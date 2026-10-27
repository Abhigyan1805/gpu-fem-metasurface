"""Mesh-convergence and port-distance studies (class 5 by default), saved to results/convergence/.

Notch frequency is located on a fine 0.01 GHz grid around the resonance, so the result is
not limited by the 0.08 GHz sweep spacing.

    python scripts/convergence.py
"""

from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fem.analytic import stack_gamma  # noqa: E402
from fem.designs import get_design  # noqa: E402
from fem.solver import UnitCellFEM  # noqa: E402

OUT = ROOT / "results" / "convergence"

CASES = [
    # (label, order, mesh overrides)
    ("o1 h_edge0.20 h_diel0.60", 1, dict(h_edge=0.20, h_diel=0.60, h_air=2.0, grow=1.0)),
    ("o1 h_edge0.12 h_diel0.40", 1, dict(h_edge=0.12, h_diel=0.40, h_air=2.0, grow=1.0)),
    ("o1 h_edge0.12 h_diel0.25", 1, dict(h_edge=0.12, h_diel=0.25, h_air=2.0, grow=1.0)),
    ("o2 coarse", 2, dict(h_edge=0.30, h_diel=0.60, h_air=2.0, grow=1.0)),
    ("o2 h_edge0.20 h_diel0.50", 2, dict(h_edge=0.20, h_diel=0.50, h_air=2.0, grow=1.0)),
    ("o2 medium", 2, dict(h_edge=0.10, h_diel=0.60, h_air=2.0, grow=0.5)),
    ("o2 fine", 2, dict(h_edge=0.07, h_diel=0.40, h_air=2.0, grow=0.5)),
    ("o2 h_edge0.05 h_diel0.30", 2, dict(h_edge=0.05, h_diel=0.30, h_air=2.0, grow=0.5)),
]
# Notch search window per element order (order 1 resonates lower on these meshes).
WINDOW = {1: (8.6, 10.0), 2: (9.3, 10.5)}


def notch(model, lo, hi):
    f = np.arange(lo, hi + 1e-9, 0.05)
    db = np.array([20 * np.log10(abs(model.solve_one(x)[0])) for x in f])
    i = int(np.argmin(db))
    f2 = np.arange(f[max(i - 1, 0)], f[min(i + 1, len(f) - 1)] + 1e-9, 0.01)
    db2 = np.array([20 * np.log10(abs(model.solve_one(x)[0])) for x in f2])
    j = int(np.argmin(db2))
    return float(f2[j]), float(db2[j])


def main(design="class5", lo=9.3, hi=10.5, parts=("mesh", "port", "slab")):
    OUT.mkdir(parents=True, exist_ok=True)
    d = get_design(design)
    rows = []
    for label, order, kw in CASES if "mesh" in parts else []:
        for copper in ("pec", "sibc"):
            t0 = time.perf_counter()
            m = UnitCellFEM(d, "coarse", order=order, copper=copper, **kw)
            fn, dn = notch(m, *WINDOW[order])
            rows.append(dict(label=label, order=order, copper=copper, mesh=kw, n_dof=m.info["n_dof"], n_tets=m.info["n_tets"],
                             notch_ghz=fn, notch_db=dn, solver_mb=m.lin.memory_mb, wall_s=time.perf_counter() - t0,
                             peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
            print(json.dumps(rows[-1]), flush=True)
            del m
    if rows:
        (OUT / f"{design}_mesh.json").write_text(json.dumps(rows, indent=2))
    if "port" not in parts:
        return

    # Port distance: grounded-slab analytic error and the class-5 notch vs air thickness.
    port = []
    for air in (5.0, 10.0, 15.0):
        m = UnitCellFEM(d, "medium", air_mm=air)
        fn, dn = notch(m, lo, hi)
        f = np.array([2.0, 10.0, 18.0])
        s = np.array([m.solve_one(x)[0] for x in f])
        port.append(dict(air_mm=air, notch_ghz=fn, notch_db=dn, s11_db_at=dict(zip(map(str, f), (20 * np.log10(np.abs(s))).tolist()))))
        print(json.dumps(port[-1]), flush=True)
        del m
    (OUT / f"{design}_port_distance.json").write_text(json.dumps(port, indent=2))

    # Slab analytic check (no pattern), order 1 vs order 2 on the medium preset.
    slab = []
    for base in ("class5", "class9"):
        ds = get_design("slab_" + base)
        f = np.linspace(2, 18, 17)
        g = stack_gamma(f, ds.eps_c, ds.h_mm, ds.superstrate)
        for order in (1, 2):
            m = UnitCellFEM(ds, "medium", order=order)
            s = np.array([m.solve_one(x)[0] for x in f])
            slab.append(dict(stack=base, order=order, n_dof=m.info["n_dof"],
                             max_abs_db_err=float(np.max(np.abs(20 * np.log10(np.abs(s) / np.abs(g))))),
                             max_phase_err_deg=float(np.max(np.abs(np.angle(s / g, deg=True)))),
                             f_ghz=f.tolist(), fem_re=s.real.tolist(), fem_im=s.imag.tolist(), tl_re=g.real.tolist(), tl_im=g.imag.tolist()))
            print(json.dumps({k: v for k, v in slab[-1].items() if not isinstance(v, list)}), flush=True)
    (OUT / "slab_analytic.json").write_text(json.dumps(slab, indent=2))


if __name__ == "__main__":
    main(parts=tuple(sys.argv[1:]) or ("mesh", "port", "slab"))
