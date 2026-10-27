"""Unit-cell FEM: mesh -> assembly -> per-frequency solve -> S11."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .analytic import ETA0, copper_zs, k0_per_mm
from .assembly import System, assemble
from .designs import SIGMA_CU, Design
from .linsolve import make_solver
from .mesh import Mesh, build_mesh


@dataclass
class Result:
    f_ghz: np.ndarray
    s11: np.ndarray  # co-polar (y), referenced to the top air interface
    s11_cross: np.ndarray  # x-polarised reflection (diagnostic)
    residual: np.ndarray
    t_solve: np.ndarray  # seconds per frequency (matrix combine + factor + solve)
    info: dict = field(default_factory=dict)


class UnitCellFEM:
    def __init__(
        self,
        design: Design,
        preset: str = "coarse",
        *,
        copper: str = "sibc",
        device: str = "cpu",
        solver: str = "pardiso",
        air_mm: float = 10.0,
        pol: str = "y",
        order: int = 2,
        mesh: Mesh | None = None,
        **mesh_kw,
    ):
        self.design = design
        self.copper = copper
        self.pol = pol
        t0 = time.perf_counter()
        self.mesh = mesh if mesh is not None else build_mesh(design, preset, air_mm=air_mm, **mesh_kw)
        t1 = time.perf_counter()
        self.pec_metal = design.r_sheet == 0 and copper == "pec"
        self.sys: System = assemble(self.mesh, pec_metal=self.pec_metal, order=order, device=device)
        t2 = time.perf_counter()
        self.solver_name = solver
        self.lin = make_solver(solver, **({"device": device} if solver in ("torch", "cudss", "cudss32", "cudss32ir") else {}))
        self.info = dict(
            design=design.name,
            preset=preset,
            copper=copper if design.r_sheet == 0 else f"R_s={design.r_sheet:g}",
            pol=pol,
            order=order,
            device=str(device),
            solver=solver,
            n_nodes=int(len(self.mesh.nodes)),
            n_tets=int(len(self.mesh.tets)),
            n_dofs_total=int(self.sys.n_dofs),
            n_dof=int(self.sys.n_free),
            nnz=int(len(self.sys.indices)),
            t_mesh_s=t1 - t0,
            t_assemble_s=t2 - t1,
            mesh=self.mesh.info,
        )

    def coeffs(self, f_ghz: float) -> dict[str, complex]:
        k0 = float(k0_per_mm(f_ghz))
        eps = self.design.eps_c
        c = {"S": 1.0, "M_sub": -k0**2 * eps, "M_sup": -k0**2 * eps, "M_air": -(k0**2), "B_port": 1j * k0}
        if "B_sheet" in self.sys.parts:
            zs = self.design.r_sheet if self.design.r_sheet > 0 else complex(copper_zs(f_ghz, SIGMA_CU))
            c["B_sheet"] = 1j * k0 * ETA0 / zs
        return c

    def solve_one(self, f_ghz: float):
        k0 = float(k0_per_mm(f_ghz))
        zp = self.mesh.z_port
        u = self.sys.upper(self.coeffs(f_ghz))
        comp = 1 if self.pol == "y" else 0
        inc = np.exp(1j * k0 * zp)
        b = 2j * k0 * inc * self.sys.port_moment[:, comp]
        self.lin.factorize(u)
        x = self.lin.solve(b)
        ax = u @ x + u.T @ x - u.diagonal() * x  # symmetric product from the upper triangle
        res = float(np.linalg.norm(ax - b) / np.linalg.norm(b))
        e_avg = self.sys.port_moment.T @ x / self.sys.port_area  # <E_x>, <E_y>, <E_z>
        ref = np.exp(-2j * k0 * self.design.z_top_mm)
        co = (e_avg[comp] - inc) * inc * ref
        cross = e_avg[1 - comp] * inc * ref
        return co, cross, res, x

    def sweep(self, f_ghz, progress: bool = True) -> Result:
        f_ghz = np.asarray(f_ghz, dtype=float)
        s11 = np.zeros(f_ghz.size, dtype=complex)
        xs = np.zeros_like(s11)
        res = np.zeros(f_ghz.size)
        ts = np.zeros(f_ghz.size)
        for i, f in enumerate(f_ghz):
            t0 = time.perf_counter()
            s11[i], xs[i], res[i], _ = self.solve_one(float(f))
            ts[i] = time.perf_counter() - t0
            if progress:
                db = 20 * np.log10(max(abs(s11[i]), 1e-12))
                print(f"  {f:6.2f} GHz  |S11| {db:8.3f} dB  res {res[i]:.1e}  {ts[i]:.2f}s", flush=True)
        info = dict(self.info)
        info["solver_memory_mb"] = float(getattr(self.lin, "memory_mb", float("nan")))
        info["nnz_factor"] = int(getattr(self.lin, "nnz_factor", -1))
        if getattr(self.lin, "refine", 0):
            steps = self.lin.ir_steps[-f_ghz.size:]
            info["ir_steps_mean"] = float(np.mean(steps))
            info["ir_steps_max"] = int(np.max(steps))
        return Result(f_ghz, s11, xs, res, ts, info)
