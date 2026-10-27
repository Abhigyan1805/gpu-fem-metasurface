"""GPU per-point time PROJECTION (not a measurement) from the measured factorisation work.

    python scripts/gpu_projection.py

Assumption: a sparse direct factorisation reaches the same fraction of FP64 peak on a GPU
(cuDSS) as PARDISO reaches on this CPU. Peaks are vendor numbers. Writes results/gpu_projection.json.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

# Vendor FP64 peaks (GFLOP/s). CPU: 4 cores x 3.1 GHz base x 16 FP64 flop/cycle (one AVX-512 FMA unit).
CPU_PEAK = 4 * 3.1 * 16
GPUS = {"T4": 250.0, "A100 (FP64, non-tensor)": 9700.0}


def main():
    rows = json.loads((RES / "factor_cost.json").read_text())
    out = dict(
        assumption="same fraction of FP64 peak as PARDISO achieves on the i5-11300H; complex128 throughout; "
        "solve phase and host-device transfers ignored; projection only",
        cpu_peak_gflops=CPU_PEAK,
        gpu_peak_gflops=GPUS,
        rows=[],
    )
    for r in rows:
        eff = r["cpu_gflops"] / CPU_PEAK
        row = dict(design=r["design"], n_dof=r["n_dof"], factor_gflop=r["factor_gflop"], cpu_factor_s=r["t_factor_s"],
                   cpu_efficiency=eff)
        for name, peak in GPUS.items():
            row[f"proj_factor_s_{name}"] = r["factor_gflop"] / (eff * peak)
        out["rows"].append(row)
    (RES / "gpu_projection.json").write_text(json.dumps(out, indent=2))
    lines = ["| Design | Unknowns | Factor work (GFLOP) | CPU factor (s) | CPU % of peak | T4 projected (s) | A100 projected (s) |", "|---|---|---|---|---|---|---|"]
    for r in out["rows"]:
        lines.append(f"| {r['design']} | {r['n_dof']:,} | {r['factor_gflop']:.0f} | {r['cpu_factor_s']:.2f} | {100 * r['cpu_efficiency']:.0f} | "
                     f"{r['proj_factor_s_T4']:.2f} | {r['proj_factor_s_A100 (FP64, non-tensor)']:.3f} |")
    (RES / "gpu_projection.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
