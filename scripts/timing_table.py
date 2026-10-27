"""Collect per-design sweep timings from results/<design>/run.json into one summary table.

    python scripts/timing_table.py

Writes results/timing_summary.json and results/timing_summary.md.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
DESIGNS = ["class5", "class8", "class9", "class19", "class19_table1"]
PAPER = {"class5": "G1", "class8": "G2", "class9": "G3", "class19": "G4", "class19_table1": "G4"}


def main():
    rows = []
    for d in DESIGNS:
        p = RES / d / "run.json"
        if not p.is_file():
            continue
        r = json.loads(p.read_text())
        m = r.get("machine", {})
        row = dict(
            design=d,
            paper=PAPER[d],
            order=r["order"],
            preset=r["preset"],
            conductor=r["copper"],
            n_tets=r["n_tets"],
            n_dof=r["n_dof"],
            nnz=r["nnz"],
            n_freq=r["n_freq"],
            wall_s=r["wall_s"],
            mesh_s=r["t_mesh_s"],
            assembly_s=r["t_assemble_s"],
            solve_total_s=r.get("t_solve_total_s", r["t_solve_mean_s"] * r["n_freq"]),
            per_freq_mean_s=r["t_solve_mean_s"],
            per_freq_median_s=r["t_solve_median_s"],
            factor_memory_mb=r["solver_memory_mb"],
            peak_rss_mb=r["peak_rss_mb"],
            solver=r["solver"],
            device=r["device"],
            cpu=m.get("cpu"),
            threads=m.get("mkl_threads"),
            load_avg_start=r.get("load_avg_start"),
        )
        rows.append(row)
    out = dict(rows=rows)
    (RES / "timing_summary.json").write_text(json.dumps(out, indent=2))

    lines = [
        "| Design | Order | Tets | Unknowns | Mesh (s) | Assembly (s) | Solves (s) | Per point (s) | Total 201 pts (s) | Peak RSS (MB) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['design']} ({r['paper']}) | {r['order']} | {r['n_tets']:,} | {r['n_dof']:,} | {r['mesh_s']:.1f} | {r['assembly_s']:.1f} | "
            f"{r['solve_total_s']:.0f} | {r['per_freq_mean_s']:.2f} | {r['wall_s']:.0f} | {r['peak_rss_mb']:.0f} |"
        )
    cpus = sorted({f"{r['cpu']} ({r['threads']} MKL threads)" for r in rows if r["cpu"]})
    lines += ["", f"FEM hardware: {', '.join(cpus) or 'see run.json'}; PARDISO complex-symmetric, float64, one frequency at a time; machine shared with other jobs."]
    (RES / "timing_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
