"""The four paper unit cells: stack parameters from the CST CSVs and the DXF patterns."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

PERIOD_MM = 10.0
SIGMA_CU = 5.8e7
F_START_GHZ = 2.0
F_STOP_GHZ = 18.0
N_FREQ = 201

# Paper Table I names for the CSV classes.
PAPER_NAME = {"class5": "G1", "class8": "G2", "class9": "G3", "class19": "G4"}


@dataclass(frozen=True)
class Design:
    name: str
    eps_r: float
    tand: float
    h_mm: float
    superstrate: bool
    r_sheet: float  # ohm/sq; 0 means copper
    dxf: Path | None
    cst_csv: Path | None = None
    period_mm: float = PERIOD_MM

    @property
    def eps_c(self) -> complex:
        return self.eps_r * (1.0 - 1j * self.tand)

    @property
    def z_top_mm(self) -> float:
        """Top air interface of the stack (S11 phase reference)."""
        return self.h_mm if self.superstrate else 0.0

    @property
    def conductor(self) -> str:
        return f"{self.r_sheet:g} ohm/sq" if self.r_sheet > 0 else "copper"


def read_cst_csv(path: Path) -> tuple[list[str], np.ndarray]:
    """Return (the 9 parameter columns, 201 linear abs(S11) values)."""
    with Path(path).open(newline="", encoding="utf-8") as f:
        row = next(csv.reader(f))
    return row[:9], np.asarray([float(v) for v in row[9:]], dtype=float)


def freq_grid_ghz(n: int = N_FREQ) -> np.ndarray:
    return np.linspace(F_START_GHZ, F_STOP_GHZ, n)


def _from_csv(name: str) -> Design:
    folder = DATA / name
    params, _ = read_cst_csv(folder / "cst_s11.csv")
    # Columns B, D, F, H, I = eps_r, tan d, h (mm), superstrate, R_s (run_batch.py).
    return Design(
        name=name,
        eps_r=float(params[1]),
        tand=float(params[3]),
        h_mm=float(params[5]),
        superstrate=bool(int(float(params[7]))),
        r_sheet=float(params[8]),
        dxf=folder / "pattern.dxf",
        cst_csv=folder / "cst_s11.csv",
    )


def get_design(name: str) -> Design:
    """``class5|class8|class9|class19`` (CSV parameters), ``class19_table1``
    (paper Table I: superstrate + 100 ohm/sq), or ``slab_<base>`` (no pattern)."""
    if name.startswith("slab_"):
        return replace(get_design(name[5:]), name=name, dxf=None)
    if name == "class19_table1":
        return replace(_from_csv("class19"), name=name, superstrate=True, r_sheet=100.0)
    if name in PAPER_NAME:
        return _from_csv(name)
    raise KeyError(f"unknown design {name!r}")


DESIGNS = ["class5", "class8", "class9", "class19"]
