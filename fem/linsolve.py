"""Sparse direct solvers for the complex-symmetric FEM system.

``pardiso``: MKL PARDISO, mtype 6 (complex symmetric), upper triangle only, through ctypes
on the ``mkl_rt`` shipped with the ``pypardiso`` wheel (pypardiso itself is real-only).
The symbolic analysis is done once per pattern and reused for every frequency.
``splu``: SciPy SuperLU fallback. ``torch``/``cudss``: GPU direct solvers (see ``gpu.py``).
"""

from __future__ import annotations

import ctypes
import glob
import os
import site
import sys
from ctypes.util import find_library

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _load_mkl():
    path = os.environ.get("FEM_MKL_RT") or find_library("mkl_rt")
    if path is None:
        cands = sorted(
            glob.glob(f"{sys.prefix}/lib*/**/libmkl_rt.so*", recursive=True)
            + glob.glob(f"{site.USER_BASE}/lib*/**/libmkl_rt.so*", recursive=True),
            key=len,
        )
        if not cands:
            raise OSError("libmkl_rt not found (pip install pypardiso)")
        path = cands[0]
    return ctypes.CDLL(path)


class PardisoComplexSymmetric:
    """Reusable PARDISO handle for one sparsity pattern."""

    def __init__(self, threads: int | None = None, verbose: bool = False):
        self.lib = _load_mkl()
        self.fn = self.lib.pardiso
        i32p = ctypes.POINTER(ctypes.c_int32)
        self.fn.argtypes = [
            ctypes.POINTER(ctypes.c_int64), i32p, i32p, i32p, i32p, i32p,
            ctypes.c_void_p, i32p, i32p, i32p, i32p, i32p, i32p,
            ctypes.c_void_p, ctypes.c_void_p, i32p,
        ]
        self.fn.restype = None
        if threads:
            self.lib.MKL_Set_Num_Threads(ctypes.c_int(int(threads)))
        self.pt = np.zeros(64, dtype=np.int64)
        self.iparm = np.zeros(64, dtype=np.int32)
        ip = self.iparm
        ip[0] = 1  # user values
        ip[1] = 2  # METIS nested dissection
        ip[7] = 2  # iterative refinement steps
        ip[9] = 8  # pivot perturbation 1e-8
        ip[10] = 1  # scaling
        ip[12] = 1  # weighted matching (recommended for symmetric indefinite)
        ip[17] = -1  # report nnz(L)
        ip[18] = -1  # report factorisation MFLOP
        ip[34] = 1  # zero-based indices
        self.msglvl = int(verbose)
        self.n = 0
        self.pattern = None
        self._ia = self._ja = None

    def _call(self, phase: int, a: np.ndarray, b: np.ndarray | None = None, nrhs: int = 1):
        err = ctypes.c_int32(0)
        x = np.zeros_like(b) if b is not None else np.zeros(1, dtype=np.complex128)
        bb = b if b is not None else np.zeros(1, dtype=np.complex128)
        c32 = lambda v: ctypes.byref(ctypes.c_int32(v))  # noqa: E731
        self.fn(
            self.pt.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            c32(1), c32(1), c32(6), c32(phase), c32(self.n),
            a.ctypes.data, self._ia.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            self._ja.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            None, c32(nrhs), self.iparm.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            c32(self.msglvl), bb.ctypes.data, x.ctypes.data, ctypes.byref(err),
        )
        if err.value != 0:
            raise RuntimeError(f"PARDISO phase {phase} error {err.value}")
        return x

    @staticmethod
    def upper(a: sp.csr_matrix) -> sp.csr_matrix:
        u = sp.triu(a, format="csr")
        u.sort_indices()
        return u

    def factorize(self, a: sp.csr_matrix) -> None:
        u = self.upper(a)
        key = (u.shape[0], u.nnz)
        if self.pattern != key:
            if self.pattern is not None:
                self.release()
            self.n = u.shape[0]
            self._ia = u.indptr.astype(np.int32)
            self._ja = u.indices.astype(np.int32)
            self._a = np.ascontiguousarray(u.data, dtype=np.complex128)
            self._call(11, self._a)
            self.pattern = key
        self._a = np.ascontiguousarray(u.data, dtype=np.complex128)
        self._call(22, self._a)

    def solve(self, b: np.ndarray) -> np.ndarray:
        b = np.ascontiguousarray(b, dtype=np.complex128)
        nrhs = 1 if b.ndim == 1 else b.shape[1]
        if b.ndim == 2:
            b = np.asfortranarray(b)
        return self._call(33, self._a, b, nrhs)

    @property
    def memory_mb(self) -> float:
        """Peak PARDISO memory: max(analysis, permanent + factor), MB."""
        ip = self.iparm
        return max(int(ip[14]), int(ip[15]) + int(ip[16])) / 1024.0

    @property
    def nnz_factor(self) -> int:
        return int(self.iparm[17])

    @property
    def factor_mflop(self) -> int:
        return int(self.iparm[18])

    def release(self) -> None:
        if self.pattern is not None:
            self._call(-1, np.zeros(1, dtype=np.complex128))
            self.pattern = None

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


class SuperLU:
    def __init__(self, **_):
        self.lu = None

    def factorize(self, a: sp.csr_matrix) -> None:
        full = (a + sp.triu(a, k=1).T).tocsc()  # a is the upper triangle
        self.lu = spla.splu(full, permc_spec="COLAMD")

    def solve(self, b: np.ndarray) -> np.ndarray:
        return self.lu.solve(np.asarray(b, dtype=np.complex128))

    @property
    def memory_mb(self) -> float:
        return (self.lu.L.nnz + self.lu.U.nnz) * 16 / 2**20

    @property
    def nnz_factor(self) -> int:
        return int(self.lu.L.nnz + self.lu.U.nnz)

    def release(self) -> None:
        self.lu = None


def make_solver(name: str, **kw):
    if name == "pardiso":
        return PardisoComplexSymmetric(**kw)
    if name == "splu":
        return SuperLU(**kw)
    if name in ("torch", "cudss", "cudss32", "cudss32ir"):
        from .gpu import make_gpu_solver

        return make_gpu_solver(name, **kw)
    raise ValueError(f"unknown solver {name!r}")
