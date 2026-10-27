"""Frequency sweep CLI.

    python -m fem.sweep --design class5
    python -m fem.sweep --design class19_table1 --preset medium --n-freq 201
    python -m fem.sweep --design class5 --copper pec --tag pec

Writes results/<design>[_<tag>]/: s11_fem.csv, run.json, metrics.json, s11_compare.png.
"""

from __future__ import annotations

import argparse
import csv
import json
import resource
import time
from pathlib import Path

import numpy as np

from .designs import PAPER_NAME, ROOT, freq_grid_ghz, get_design, read_cst_csv
from .metrics import compare
from .plotting import s11_plot
from .solver import UnitCellFEM

RESULTS = ROOT / "results"


def db(x):
    return 20 * np.log10(np.clip(np.abs(x), 1e-12, None))


def machine_info() -> dict:
    import os
    import platform

    cpu = platform.processor()
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:
        pass
    info = dict(cpu=cpu, logical_cpus=os.cpu_count(), platform=platform.platform())
    try:
        from .linsolve import _load_mkl

        info["mkl_threads"] = int(_load_mkl().MKL_Get_Max_Threads())
    except Exception:
        pass
    try:
        import torch

        info["torch_threads"] = torch.get_num_threads()
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def plot_compare(path: Path, design, f, s11_db, info: dict) -> None:
    """CST (when the design has a CST curve on the same grid) and the FEM S11 in dB."""
    base = design.name.split("_")[0] if not design.name.startswith("slab_") else None
    curves = []
    if base and design.cst_csv is not None and len(f) == 201:
        _, cst = read_cst_csv(design.cst_csv)
        curves.append(("CST (from CSV)", db(cst), "cst"))
    curves.append((f"FEM order {info['order']} ({info['preset']}, {info['copper']})", s11_db, "fem"))
    title = f"{design.name} ({PAPER_NAME.get(base, '')})  eps_r={design.eps_r:g}  h={design.h_mm:g} mm  " \
            f"superstrate={'yes' if design.superstrate else 'no'}  {design.conductor}"
    s11_plot(path, title, f, curves)


def write_outputs(out: Path, design, res, args, wall_s: float) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    f = res.f_ghz
    with (out / "s11_fem.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Frequency_GHz", "S11_real", "S11_imag", "S11_mag", "S11_dB", "cross_mag", "residual", "t_solve_s"])
        for row in zip(f, res.s11.real, res.s11.imag, np.abs(res.s11), db(res.s11), np.abs(res.s11_cross), res.residual, res.t_solve):
            w.writerow([f"{v:.6g}" if isinstance(v, float) else v for v in row])
    run = dict(res.info)
    run.update(
        n_freq=int(f.size),
        wall_s=wall_s,
        t_solve_total_s=float(res.t_solve.sum()),
        t_solve_mean_s=float(res.t_solve.mean()),
        t_solve_median_s=float(np.median(res.t_solve)),
        max_residual=float(res.residual.max()),
        max_abs_s11=float(np.abs(res.s11).max()),
        max_cross=float(np.abs(res.s11_cross).max()),
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        argv=vars(args),
        machine=machine_info(),
        load_avg_end=list(__import__("os").getloadavg()),
        design_params=dict(eps_r=design.eps_r, tand=design.tand, h_mm=design.h_mm, superstrate=design.superstrate, conductor=design.conductor),
    )
    (out / "run.json").write_text(json.dumps(run, indent=2, default=str))

    metrics = {}
    if not design.name.startswith("slab_") and design.cst_csv is not None and f.size == 201:
        _, cst = read_cst_csv(design.cst_csv)
        metrics["fem_vs_cst"] = compare(f, db(cst), db(res.s11))
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_compare(out / "s11_compare.png", design, f, db(res.s11), res.info)
    return metrics


def main(argv=None):
    p = argparse.ArgumentParser(description="Periodic unit-cell FEM S11 sweep")
    p.add_argument("--design", required=True, help="class5|class8|class9|class19|class19_table1|slab_<base>")
    p.add_argument("--preset", default="medium", choices=["coarse", "medium", "fine"])
    p.add_argument("--order", type=int, default=2, choices=[1, 2])
    p.add_argument("--copper", default="sibc", choices=["sibc", "pec"])
    p.add_argument("--pol", default="y", choices=["x", "y"])
    p.add_argument("--n-freq", type=int, default=201)
    p.add_argument("--air-mm", type=float, default=10.0)
    p.add_argument("--device", default="cpu", help="torch device for assembly (cpu, cuda)")
    p.add_argument("--solver", default="pardiso", choices=["pardiso", "splu", "torch", "cudss", "cudss32", "cudss32ir"])
    p.add_argument("--tag", default="")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    design = get_design(args.design)
    out = args.out or RESULTS / (design.name + (f"_{args.tag}" if args.tag else ""))
    load0 = list(__import__("os").getloadavg())
    t0 = time.perf_counter()
    model = UnitCellFEM(
        design, args.preset, order=args.order, copper=args.copper, pol=args.pol,
        device=args.device, solver=args.solver, air_mm=args.air_mm,
    )
    print(json.dumps({k: model.info[k] for k in ("design", "order", "preset", "n_tets", "n_dof", "nnz")}), flush=True)
    res = model.sweep(freq_grid_ghz(args.n_freq), progress=not args.quiet)
    res.info["load_avg_start"] = load0
    metrics = write_outputs(out, design, res, args, time.perf_counter() - t0)
    if "fem_vs_cst" in metrics:
        m = metrics["fem_vs_cst"]
        print(f"vs CST: RMSE {m['rmse_db_clip40']:.2f} dB, notches {[(n['f_ref'], n['f_test'], None if n['df_pct'] is None else round(n['df_pct'], 2)) for n in m['notches']]}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
