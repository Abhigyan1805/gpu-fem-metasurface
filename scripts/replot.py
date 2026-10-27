"""Redraw results/<run>/s11_compare.png from the saved s11_fem.csv and run.json (no new solves).

    python scripts/replot.py                # every run directory under results/
    python scripts/replot.py class5 class9  # selected run directories
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fem.designs import get_design  # noqa: E402
from fem.sweep import plot_compare  # noqa: E402

RES = ROOT / "results"


def main(names: list[str]):
    dirs = [RES / n for n in names] if names else sorted(p.parent for p in RES.glob("*/run.json"))
    for d in dirs:
        run = json.loads((d / "run.json").read_text())
        a = np.genfromtxt(d / "s11_fem.csv", delimiter=",", names=True)
        plot_compare(d / "s11_compare.png", get_design(run["argv"]["design"]), a["Frequency_GHz"], a["S11_dB"], run)
        print(f"wrote {d / 's11_compare.png'}")


if __name__ == "__main__":
    main(sys.argv[1:])
