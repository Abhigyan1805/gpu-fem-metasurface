"""Accuracy summary tables and the combined figure, built only from saved result files.

    python scripts/summary.py

Reads results/<design>/{s11_fem.csv,metrics.json,run.json} and the CST curves in data/.
Writes results/accuracy_summary.{json,md} and results/s11_all_designs.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fem.designs import PAPER_NAME, freq_grid_ghz, get_design, read_cst_csv  # noqa: E402
from fem.plotting import CST, FEM_COLORS, INK, MUTED  # noqa: E402

RES = ROOT / "results"
DESIGNS = ["class5", "class8", "class9", "class19", "class19_table1"]


def main():
    rows = []
    lines = [
        "| Design | CST notch (GHz, dB) | FEM notch (GHz, dB) | FEM Δf vs CST | RMSE FEM–CST (dB) |",
        "|---|---|---|---|---|",
    ]
    for d in DESIGNS:
        p = RES / d / "metrics.json"
        if not p.is_file():
            continue
        m = json.loads(p.read_text())
        fc = m["fem_vs_cst"]
        base = d.split("_")[0]
        fmt = lambda f, dbv: "–" if f is None else f"{f:.2f}, {dbv:.1f}"  # noqa: E731
        pct = lambda v: "–" if v is None else f"{v:+.1f} %"  # noqa: E731
        for i, n in enumerate(fc["notches"]):
            label = f"{d} ({PAPER_NAME[base]})" if i == 0 else ""
            rmse = f"{fc['rmse_db_clip40']:.2f}" if i == 0 else ""
            lines.append(
                f"| {label} | {fmt(n['f_ref'], n['depth_ref'])} | {fmt(n['f_test'], n['depth_test'])} | {pct(n['df_pct'])} | "
                f"{rmse} |"
            )
        rows.append(dict(design=d, fem_vs_cst=fc))
    lines += ["", "Notches: CST local minima deeper than −6 dB with ≥ 3 dB prominence, matched to the nearest FEM local minimum; "
              "frequencies on the 201-point grid (0.08 GHz step). RMSE over 2–18 GHz with dB clipped at −40 dB."]
    (RES / "accuracy_summary.json").write_text(json.dumps(rows, indent=2))
    (RES / "accuracy_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = freq_grid_ghz()
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), dpi=150, sharex=True)
    for ax, d in zip(axes.flat, ["class5", "class8", "class9", "class19"]):
        des = get_design(d)
        _, cst = read_cst_csv(des.cst_csv)
        fem = np.loadtxt(RES / d / "s11_fem.csv", delimiter=",", skiprows=1)
        ax.plot(f, 20 * np.log10(cst), color=CST, lw=1.8, label="CST")
        ax.plot(fem[:, 0], fem[:, 4], color=FEM_COLORS[0], lw=1.4, label="FEM (this work)")
        ax.set_title(f"{d} ({PAPER_NAME[d]})", loc="left", fontsize=9, color=INK)
        ax.set_ylim(-40, 1)
        ax.set_xlim(2, 18)
        ax.grid(True, color="#e4e3dc", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
    for ax in axes[1]:
        ax.set_xlabel("Frequency (GHz)", color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("|S11| (dB)", color=INK)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(RES / "s11_all_designs.png")


if __name__ == "__main__":
    main()


def convergence():
    """Plot class-5 notch vs unknowns and tabulate the slab analytic check."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conv = RES / "convergence"
    rows = json.loads((conv / "class5_mesh.json").read_text())
    _, cst = read_cst_csv(get_design("class5").cst_csv)
    f = freq_grid_ghz()
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
    styles = {(1, "sibc"): (FEM_COLORS[2], "s", "order 1, SIBC copper"), (2, "sibc"): (FEM_COLORS[0], "o", "order 2, SIBC copper"),
              (2, "pec"): (FEM_COLORS[1], "^", "order 2, PEC")}
    for key, (c, mk, lab) in styles.items():
        r = [x for x in rows if (x["order"], x["copper"]) == key]
        ax.plot([x["n_dof"] for x in r], [x["notch_ghz"] for x in r], color=c, marker=mk, ms=6, lw=1.4, label=lab)
    ax.axhline(f[np.argmin(cst)], color=CST, lw=1.6, label="CST notch (CSV)")
    ax.set_xscale("log")
    ax.set_xlabel("Unknowns", color=INK)
    ax.set_ylabel("Notch frequency (GHz)", color=INK)
    ax.set_title("class5 (G1): notch vs mesh and element order", loc="left", fontsize=9, color=INK)
    ax.grid(True, color="#e4e3dc", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(conv / "class5_convergence.png")

    lines = ["| Stack | Order | Unknowns | max abs(S11) error (dB) | max phase error (deg) |", "|---|---|---|---|---|"]
    for r in json.loads((conv / "slab_analytic.json").read_text()):
        lines.append(f"| {r['stack']} | {r['order']} | {r['n_dof']:,} | {r['max_abs_db_err']:.1e} | {r['max_phase_err_deg']:.3f} |")
    lines += ["", "Grounded slab (class5 stack) and slab + superstrate (class9 stack) without a pattern, 17 points 2–18 GHz, "
              "against the transmission-line result; medium preset."]
    (conv / "slab_analytic.md").write_text("\n".join(lines) + "\n")
    lines = ["| Case | Copper | Unknowns | Notch (GHz) | Depth (dB) | Factor memory (MB) | Peak RSS (MB) |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['label']} | {r['copper']} | {r['n_dof']:,} | {r['notch_ghz']:.2f} | {r['notch_db']:.1f} | {r['solver_mb']:.0f} | {r['peak_rss_mb']:.0f} |")
    port = json.loads((conv / "class5_port_distance.json").read_text())
    lines += ["", "| Air above stack (mm) | Notch (GHz) | Depth (dB) |", "|---|---|---|"]
    lines += [f"| {p['air_mm']:g} | {p['notch_ghz']:.2f} | {p['notch_db']:.1f} |" for p in port]
    (conv / "class5_convergence.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    convergence()
