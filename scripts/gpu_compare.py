"""Compare GPU sweeps (results/<design>_<tag>/) with the CPU PARDISO complex128 sweep of the same design.

    python scripts/gpu_compare.py class5 --tags gpu_cudss gpu_cudss32 gpu_cudss32ir

Also tabulates results/gpu_gtx1650/bench*.json (scripts/bench.py) into bench.md; a later file
(sorted by name) replaces the runs of the designs it contains.
Writes results/gpu_gtx1650/compare.{json,md}: per run the per-point solve time, max |dB difference|
against the CPU curve (dB clipped at -40 dB, as in the accuracy tables), max |S11| complex difference,
notch frequencies (CPU deepest minimum vs GPU deepest minimum) and the largest relative residual.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "gpu_gtx1650"


def load(d: Path):
    a = np.genfromtxt(d / "s11_fem.csv", delimiter=",", names=True)
    return a, json.loads((d / "run.json").read_text())


def db(a):
    return 20 * np.log10(np.clip(np.hypot(a["S11_real"], a["S11_imag"]), 1e-12, None))


def bench_table(paths: list[Path]) -> list[str]:
    b = None
    for path in paths:
        nb = json.loads(path.read_text())
        if b is None:
            b = nb
            continue
        names = {r["design"] for r in nb["runs"]}
        b["runs"] = [r for r in b["runs"] if r["design"] not in names] + nb["runs"]
    asm = {(r["design"], r["device"]): r["seconds"] for r in b["runs"] if r["kind"] == "assembly"}
    sol = [r for r in b["runs"] if r["kind"] == "solve" and "error" not in r]
    lines = [f"GPU: {b.get('gpu')}; torch {b.get('torch')}; frequencies {b['freqs_ghz']} GHz.", "",
             "| Design | Unknowns | Assembly CPU (s) | Assembly CUDA (s) | Solver | Precision | Mean per point (s) | Speed vs PARDISO | Max abs(ΔS11) vs PARDISO | Device memory (MB) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    prec = {"pardiso": "complex128", "cudss": "complex128", "cudss32": "complex64", "cudss32ir": "complex64 + IR (complex128 residual)"}
    for r in sol:
        ref = next(x for x in sol if x["design"] == r["design"] and x["solver"] == "pardiso")
        d = np.abs(np.array([complex(*v) for v in r["s11"]]) - np.array([complex(*v) for v in ref["s11"]])).max()
        lines.append(f"| {r['design']} | {r['n_dof']:,} | {asm.get((r['design'], 'cpu'), float('nan')):.2f} | "
                     f"{asm.get((r['design'], 'cuda'), float('nan')):.2f} | {r['solver']} | {prec.get(r['solver'], '')} | "
                     f"{r['mean_s']:.3f} | {ref['mean_s'] / r['mean_s']:.2f}× | {d:.1e} | "
                     f"{'' if 'device_memory_mb' not in r else round(r['device_memory_mb'])} |")
    errs = [r for r in b["runs"] if "error" in r]
    for r in errs:
        lines.append(f"\n{r['design']} {r['solver']}: {r['error']}")
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("designs", nargs="+")
    p.add_argument("--tags", nargs="+", default=["gpu_cudss", "gpu_cudss32", "gpu_cudss32ir"])
    a = p.parse_args()
    rows = []
    for name in a.designs:
        ref, rref = load(ROOT / "results" / name)
        ref_db = np.maximum(db(ref), -40)
        f = ref["Frequency_GHz"]
        s_ref = ref["S11_real"] + 1j * ref["S11_imag"]
        rows.append(dict(design=name, run="cpu_pardiso_c128", solver="pardiso", t_point_s=rref["t_solve_mean_s"],
                         t_total_s=rref["wall_s"], max_db_diff=0.0, max_abs_diff=0.0, notch_ghz=float(f[np.argmin(ref_db)]),
                         notch_shift_ghz=0.0, max_residual=rref["max_residual"]))
        for tag in a.tags:
            d = ROOT / "results" / f"{name}_{tag}"
            if not (d / "run.json").is_file():
                continue
            g, run = load(d)
            g_db = np.maximum(db(g), -40)
            s = g["S11_real"] + 1j * g["S11_imag"]
            rows.append(dict(
                design=name, run=tag, solver=run["solver"], t_point_s=run["t_solve_mean_s"], t_total_s=run["wall_s"],
                max_db_diff=float(np.abs(g_db - ref_db).max()), max_abs_diff=float(np.abs(s - s_ref).max()),
                notch_ghz=float(f[np.argmin(g_db)]), notch_shift_ghz=float(f[np.argmin(g_db)] - f[np.argmin(ref_db)]),
                max_residual=run["max_residual"], ir_steps_mean=run.get("ir_steps_mean"),
                gpu=run["machine"].get("gpu"), peak_rss_mb=run["peak_rss_mb"]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "compare.json").write_text(json.dumps(rows, indent=2))
    lines = ["| Design | Run | Per point (s) | Total 201 pts (s) | Speed vs CPU | Max dB diff vs CPU | Max abs(ΔS11) | Notch (GHz) | Notch shift (GHz) | Max residual | IR steps (mean) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        cpu = next(x for x in rows if x["design"] == r["design"] and x["run"] == "cpu_pardiso_c128")
        ir = r.get("ir_steps_mean")
        lines.append(f"| {r['design']} | {r['run']} | {r['t_point_s']:.3f} | {r['t_total_s']:.0f} | {cpu['t_point_s'] / r['t_point_s']:.2f}× | "
                     f"{r['max_db_diff']:.2e} | {r['max_abs_diff']:.1e} | {r['notch_ghz']:.2f} | {r['notch_shift_ghz']:+.2f} | "
                     f"{r['max_residual']:.1e} | {'' if ir is None else f'{ir:.1f}'} |")
    benches = sorted(OUT.glob("bench*.json"))
    if benches:
        (OUT / "bench.md").write_text("\n".join(bench_table(benches)) + "\n")
        print((OUT / "bench.md").read_text())
    (OUT / "compare.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
