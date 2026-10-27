"""Comparison plots (PNG). CST is the neutral reference; FEM curves use the categorical colours."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#2b2b2b"
MUTED = "#6b6b66"
CST = "#3d3d3a"
FEM_COLORS = ["#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def s11_plot(path: Path, title: str, f, curves: list[tuple[str, object, str]], ylim=(-40, 1)):
    """``curves``: (label, dB array, role) with role in {'cst', 'fem'}."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    k = 0
    for label, db, role in curves:
        if role == "cst":
            ax.plot(f, db, color=CST, lw=2.0, label=label, zorder=2)
        else:
            ax.plot(f, db, color=FEM_COLORS[k % len(FEM_COLORS)], lw=1.6, label=label, zorder=4)
            k += 1
    ax.set_xlim(f[0], f[-1])
    ax.set_ylim(*ylim)
    ax.set_xlabel("Frequency (GHz)", color=INK)
    ax.set_ylabel("|S11| (dB)", color=INK)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.grid(True, color="#e4e3dc", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
