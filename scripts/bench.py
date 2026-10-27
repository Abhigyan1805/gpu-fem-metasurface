"""Timing benchmark: assembly per device and per-frequency solve per backend.

    python scripts/bench.py --designs class5 class19 --devices cpu cuda --solvers pardiso cudss torch

Writes results/bench_<host>.json. Every backend solves the same 5 frequencies; the
S11 values are stored too, so backends can be checked against each other.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fem.assembly import assemble  # noqa: E402
from fem.designs import get_design  # noqa: E402
from fem.mesh import build_mesh  # noqa: E402
from fem.solver import UnitCellFEM  # noqa: E402

F = [4.0, 8.0, 10.0, 14.0, 18.0]


def sync(dev):
    if dev.startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--designs", nargs="+", default=["class5", "class8", "class9", "class19"])
    p.add_argument("--preset", default="medium")
    p.add_argument("--devices", nargs="+", default=["cpu"])
    p.add_argument("--solvers", nargs="+", default=["pardiso"])
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    host = platform.node() or "host"
    out = {"host": host, "platform": platform.platform(), "freqs_ghz": F, "runs": []}
    try:
        import torch

        out["torch"] = torch.__version__
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    for name in a.designs:
        d = get_design(name)
        t0 = time.perf_counter()
        mesh = build_mesh(d, a.preset)
        t_mesh = time.perf_counter() - t0
        for dev in a.devices:
            assemble(mesh, pec_metal=False, order=2, device=dev)  # warm-up
            sync(dev)
            t0 = time.perf_counter()
            assemble(mesh, pec_metal=False, order=2, device=dev)
            sync(dev)
            out["runs"].append(dict(design=name, kind="assembly", device=dev, seconds=time.perf_counter() - t0))
        for solver in a.solvers:
            m = None
            try:
                dev = "cuda" if solver in ("torch", "cudss", "cudss32", "cudss32ir") else "cpu"
                m = UnitCellFEM(d, a.preset, mesh=mesh, solver=solver, device=dev)
                m.solve_one(F[0])  # warm-up (symbolic analysis, CUDA context)
                ts, s11 = [], []
                for f in F:
                    t0 = time.perf_counter()
                    s, _, res, _ = m.solve_one(f)
                    ts.append(time.perf_counter() - t0)
                    s11.append([s.real, s.imag])
                out["runs"].append(dict(design=name, kind="solve", solver=solver, n_dof=m.info["n_dof"], t_mesh_s=t_mesh,
                                        seconds_per_freq=ts, mean_s=float(np.mean(ts)), s11=s11, residual=res))
            except Exception as exc:  # report and continue with the next backend
                out["runs"].append(dict(design=name, kind="solve", solver=solver, error=repr(exc)))
            if m is not None and dev == "cuda":
                out["runs"][-1]["device_memory_mb"] = float(m.lin.memory_mb)
            if m is not None and hasattr(m.lin, "release"):
                m.lin.release()  # free the factor before the next backend (4 GB cards run out otherwise)
            m = None
            gc.collect()
            if dev == "cuda":
                import cupy

                cupy.get_default_memory_pool().free_all_blocks()
            print(json.dumps(out["runs"][-1])[:300], flush=True)
    path = a.out or ROOT / "results" / f"bench_{host}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
