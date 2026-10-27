"""Measure factorisation work (PARDISO MFLOP) and split timing per design, for the GPU projection.

    python scripts/factor_cost.py

Writes results/factor_cost.json: for each design (medium preset, 10 GHz) the factor flop count,
the numeric-factorisation time, the solve time and the achieved CPU rate.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fem.designs import get_design  # noqa: E402
from fem.solver import UnitCellFEM  # noqa: E402


def main():
    out = []
    for name in ["class5", "class8", "class9", "class19"]:
        m = UnitCellFEM(get_design(name), "medium")
        m.solve_one(10.0)  # symbolic analysis happens here
        u = m.sys.upper(m.coeffs(10.0))
        b = 2j * m.sys.port_moment[:, 1]
        t0 = time.perf_counter()
        m.lin.factorize(u)
        t1 = time.perf_counter()
        m.lin.solve(b)
        t2 = time.perf_counter()
        r = dict(design=name, n_dof=m.info["n_dof"], nnz_factor=m.lin.nnz_factor, factor_gflop=m.lin.factor_mflop / 1e3,
                 t_factor_s=t1 - t0, t_solve_s=t2 - t1, cpu_gflops=m.lin.factor_mflop / 1e3 / (t1 - t0),
                 factor_memory_mb=m.lin.memory_mb)
        out.append(r)
        print(json.dumps(r), flush=True)
        del m
    (ROOT / "results" / "factor_cost.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
