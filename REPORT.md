# A periodic unit-cell FEM solver for metasurface absorbers: validation against CST and GPU timing

Every number in this report is copied from a file in `results/`, named in each section. The code is
in `fem/` and the formulation is in [docs/DESIGN.md](docs/DESIGN.md).

Hardware: one laptop CPU (Intel Core i5-11300H, 4 cores, 3 MKL threads, about 5 GB of RAM),
shared with other jobs during the runs. GPU: the same laptop's GeForce GTX 1650 (4 GB), section 6.

## 1. Summary

**The solver.** A frequency-domain FEM for one periodic unit cell:

- Nédélec edge elements, order 2 by default (order 1 available);
- periodic side walls;
- a plane-wave port 10 mm above the stack, whose (0,0)-mode average gives S11;
- the pattern as a zero-thickness PEC, copper surface impedance (SIBC) or resistive sheet;
- assembly in PyTorch (CPU or CUDA), direct solve with MKL PARDISO (complex symmetric) or cuDSS.

**Validation.** The element unit and patch tests pass. Uniform stacks (a grounded slab, and a slab
with a superstrate) match the transmission-line result to 1.8e-5 dB and 0.021° at order 2. A
uniform resistive sheet (Salisbury screen, 50 and 377 Ω/sq) matches the analytic result.

**The four designs.** The FEM reproduces every CST feature of the four designs. Against CST
(medium mesh, 201 points):

| Design | Notch error | RMSE |
|---|---|---|
| G1 (class5) | −3.9 % | 1.40 dB |
| G2 (class8) | +1.6 % | 1.10 dB |
| G3 (class9) | +1.1 % / −4.1 % / −1.4 % | 4.16 dB |
| G4 (class19) | −2.0 % / −2.2 % | 1.48 dB |

The acceptance targets were set before any result (docs/DESIGN.md section 8). The 2 % notch target
is met on 3 of 7 notches; G4's first notch misses it narrowly at −2.04 %. The 1.5 dB RMSE target is
met on 3 of 4 designs. G3's 4.16 dB is dominated by the depth of very sharp notches.

**Mesh refinement.** On the fine mesh both G4 notches come within 2 % of CST (−1.5 % and −1.3 %).
On G1 a finer mesh moves the notch from 9.76 to 9.88 GHz, where it levels off. The remaining
−2.8 % to CST is a model difference, not mesh error (section 4.3).

**G4 parameters.** The original design table gives G4 a superstrate and a 100 Ω/sq sheet. The CST data
file for G4 has neither. Only the structure in the data file (no superstrate, copper) reproduces
the G4 CST curve. The Table I structure is a broadband absorber with a −20 dB dip at 9.36 GHz and
an RMSE of 7.90 dB against CST (section 4.4).

**Polarisation conversion.** G1 and G3 convert a large part of the y-polarised wave into x
polarisation at their notches: 42 % of the power (0.651²) for G1, and 90 % (0.948²) for G3 at its
notch near 9.4 GHz. The co-polar notch is therefore not all absorption, and absorption is not
1 − |S11|² for these cells (section 4.5).

**CPU speed (section 5).** 0.64 to 10 s per frequency point, 131 s to 2019 s per 201-point sweep.
Almost all of the time is the sparse factorisation at each frequency.

**GPU (section 6).** On the GTX 1650, a complex64 cuDSS factorisation with complex128 iterative
refinement reproduces the CPU's S11 exactly (at CSV precision), 4.0 to 9.7 times faster: 35 s for
G1 and 210 s for G3 per 201-point sweep. A complex128 cuDSS factorisation on this FP64-weak card is
about as fast as the CPU (0.83 to 1.21 times).

## 2. Problem set-up

The unit cell is 10 × 10 mm. The pattern sits at z = 0 on a substrate (ε_r, tan δ, h) over a PEC
ground. The optional superstrate is identical to the substrate.

- **Excitation:** normal incidence, y-polarised (DXF y axis).
- **S11:** co-polar, with the phase referenced to the top air interface of the stack.
- **Sweep:** 2–18 GHz, 201 points.
- **Inputs:** stack parameters from the parameter columns of each CST data file
  (`data/<design>/cst_s11.csv`, read by `fem/designs.py`), patterns from the DXF files
  (`data/<design>/pattern.dxf`), and the CST |S11| curves from the same CSV files.

| Design | Name | ε_r | tan δ | h (mm) | Superstrate | Conductor |
|---|---|---|---|---|---|---|
| class5 | G1 | 3.45 | 0.0025 | 0.762 | no | copper |
| class8 | G2 | 3.54 | 0.0033 | 3.048 | no | 50 Ω/sq |
| class9 | G3 | 2.94 | 0.0021 | 3.05 | yes | copper |
| class19 | G4 | 10.2 | 0.0023 | 1.27 | no (CSV) | copper (CSV) |
| class19_table1 | G4 as in Table I | 10.2 | 0.0023 | 1.27 | yes | 100 Ω/sq |

Copper is a surface impedance Z_s = (1+j)√(ωμ0/2σ), σ = 5.8e7 S/m, on a zero-thickness sheet.

## 3. Method

docs/DESIGN.md has the derivation. In brief:

- **Equation:** ∇×∇×E − k0²ε_c E = 0, weak form with order-2 hierarchical Nédélec elements of the
  first kind (20 DOFs per tetrahedron: Whitney functions, edge gradients and 2 face functions per
  face).
- **Periodic walls:** slave nodes are merged into their gmsh periodic masters before DOF numbering,
  so the mesh is topologically a torus. Element matrices still use the true coordinates. Local
  vertices are sorted by merged id, so no orientation signs are needed.
- **Port:** a first-order absorbing condition with the incident plane wave, 10 mm above the stack.
  Only the (0,0) Floquet pair propagates in air below 30 GHz. S11 is the cell average of E_y on the
  port minus the incident field, phase-shifted to the top of the stack.
- **Sheet:** PEC is a Dirichlet condition on the metal triangles. A resistive sheet or copper adds
  the surface term j k0 η0/Z_s ∫ E_t·W_t.
- **Assembly:** the frequency-independent parts (S, M_sub, M_sup, M_air, B_port, B_sheet) share one
  upper-triangular sparsity pattern. Each frequency only recombines the value arrays. Element
  blocks are batched torch tensors on the selected device.
- **Solve:** MKL PARDISO, mtype 6 (complex symmetric), through ctypes. The symbolic analysis is
  reused over the sweep, with one numeric factorisation per point. The relative residual is below
  3e-12 at every point of every CPU sweep (`max_residual` in `run.json`).
- **Why order 2:** order 1 converges too slowly in the 0.762 mm G1 substrate. Its notch moves
  9.07 → 9.20 → 9.49 GHz as the substrate element shrinks from 0.6 to 0.25 mm, and the next step
  exceeded the 2 GB memory budget. Order 2 reaches 9.65 GHz on its coarsest mesh (section 4.3).
- **A guarded failure mode:** merging periodic nodes is valid only when no element spans half a
  period in x or y. A 3 mm air mesh broke this silently and gave a 47° phase error. `fem/mesh.py`
  now rejects such meshes (`tests/test_mesh.py`).

## 4. Validation

### 4.1 Unit tests

`tests/` holds 19 tests. 18 run on CPU and one needs CUDA; all 19 pass on the GTX 1650 (section 6).

- **Elements:** mass matrix against an exact quadrature; curl against finite differences; a
  constant-field patch test (zero curl energy, exact mass energy, exact interpolation); gradients
  in the curl kernel; triangle traces.
- **Mesh:** the metal area equals the DXF area exactly; the total volume is exact; the periodic map
  is a lattice translation; too-coarse meshes are rejected.
- **Analytic stacks:** a grounded slab and a slab with superstrate, at order 1 and order 2;
  resistive sheets at 50 and 377 Ω/sq.

### 4.2 Analytic slab check

`results/convergence/slab_analytic.md`. 17 frequencies from 2 to 18 GHz, medium mesh, no pattern:

| Stack | Order | Unknowns | max abs(S11) error (dB) | max phase error (deg) |
|---|---|---|---|---|
| class5 | 1 | 4,733 | 1.3e-02 | 2.084 |
| class5 | 2 | 26,628 | 1.6e-05 | 0.020 |
| class9 | 1 | 18,442 | 1.3e-02 | 2.122 |
| class9 | 2 | 101,306 | 1.8e-05 | 0.021 |

Order 2 meets the target (< 0.01 dB, < 1°). Order 1 on this mesh does not (2.1°).

### 4.3 Mesh, order and port convergence on G1

`results/convergence/class5_convergence.md` and `class5_convergence.png`. The notch is located on a
0.01 GHz grid.

| Case | Copper | Unknowns | Notch (GHz) | Depth (dB) | Peak RSS (MB) |
|---|---|---|---|---|---|
| order 1, h_edge 0.20, h_diel 0.60 | SIBC | 7,072 | 9.07 | −11.3 | 360 |
| order 1, h_edge 0.12, h_diel 0.40 | SIBC | 15,522 | 9.20 | −11.3 | 406 |
| order 1, h_edge 0.12, h_diel 0.25 | SIBC | 40,457 | 9.49 | −10.8 | 563 |
| order 2 coarse | SIBC | 34,418 | 9.65 | −9.9 | 642 |
| order 2 medium (used in section 4.4) | SIBC | 50,534 | 9.76 | −9.5 | 737 |
| order 2 fine | SIBC | 98,482 | 9.86 | −9.1 | 1230 |
| order 2, h_edge 0.05, h_diel 0.30 | SIBC | 165,158 | 9.88 | −9.0 | 1913 |
| order 2, h_edge 0.05, h_diel 0.30 | PEC | 157,258 | 9.94 | −21.2 | 1808 |

- **Convergence:** the order-2 notch rises by 0.10 GHz from medium to fine and by 0.02 GHz after
  that. The converged zero-thickness model sits at about 9.9 GHz, −2.8 % from CST (10.16 GHz).
- **Copper model:** PEC sits 0.06 GHz above SIBC and is about 12 dB deeper. The notch depth is set
  by the copper-loss model, and CST's −15.3 dB lies between the two.
- **Port distance:** 5, 10 and 15 mm of air give notches at 9.78, 9.76 and 9.78 GHz, so the port
  model is not a factor.
- **Possible causes of the remaining offset to CST (not tested):**
  - CST may model copper with a finite thickness; the thickness used is not known;
  - CST's own mesh and solver settings;
  - SIBC on a zero-thickness strip uses a single-face Z_s. Current on both faces would halve the
    effective Z_s.

No parameter was adjusted to fit CST.

### 4.4 The four designs

`results/<design>/`, `results/accuracy_summary.md`, `results/s11_all_designs.png`. Order 2, medium
preset, copper as SIBC, 201 points. Notches are CST minima deeper than −6 dB, matched to the
nearest FEM local minimum. RMSE is over the band, in dB clipped at −40 dB.

| Design | CST notch (GHz, dB) | FEM notch (GHz, dB) | FEM Δf vs CST | RMSE FEM–CST (dB) |
|---|---|---|---|---|
| class5 (G1) | 10.16, −15.3 | 9.76, −9.5 | −3.9 % | 1.40 |
| class8 (G2) | 10.08, −25.4 | 10.24, −21.8 | +1.6 % | 1.10 |
| class9 (G3) | 7.44, −35.6 | 7.52, −23.1 | +1.1 % | 4.16 |
| | 9.76, −34.2 | 9.36, −33.7 | −4.1 % | |
| | 16.56, −26.1 | 16.32, −24.5 | −1.4 % | |
| class19 (G4) | 15.68, −13.5 | 15.36, −6.6 | −2.0 % | 1.48 |
| | 17.92, −7.1 | 17.52, −7.1 | −2.2 % | |
| class19_table1 (G4) | 15.68, −13.5 | 9.36, −20.0 | −40.3 % | 7.90 |
| | 17.92, −7.1 | 9.36, −20.0 | −47.8 % | |

Fine mesh on G1 and G4 (`results/class5_fine/`, `results/class19_fine/`, order 2, `fine` preset):

| Design | Unknowns | Notch FEM (GHz) | Δf vs CST | RMSE vs CST (dB) | Per point (s) | Sweep (s) | Peak RSS (MB) |
|---|---|---|---|---|---|---|---|
| class5 fine | 98,482 | 9.84 | −3.1 % | 1.32 | 2.22 | 450 | 1281 |
| class19 fine | 154,604 | 15.44 / 17.68 | −1.5 % / −1.3 % | 1.46 | 5.34 | 1079 | 1920 |

Refining moves every notch toward CST, and on the fine mesh both G4 notches meet the 2 % target.
G2 and G3 already need 1.4–2.0 GB on the medium mesh, so they were not refined.

![S11 of the four designs: CST and FEM](results/s11_all_designs.png)

Reading the curves:

- **G1:** one sharp notch, 0.40 GHz below CST on the medium mesh (0.28 GHz when converged). The FEM
  notch is shallower than CST's (−9.5 against −15.3 dB); section 4.3 shows that depth depends on the
  copper model. The FEM also shows the onset of a second resonance at 18 GHz, which CST shows as a
  slight roll-off.
- **G2 (50 Ω/sq):** the broad absorption band is well reproduced: notch +1.6 %, RMSE 1.10 dB. This
  is the best case. The resistive sheet makes the result less sensitive to the field at the strip
  edges.
- **G3:** all three notches appear. The two low notches are 1.84 GHz apart against CST's 2.32 GHz,
  and the maximum between them is deeper (−11.8 dB at 8.48 GHz against CST's −8.6 dB at 8.64 GHz).
  The 4.16 dB RMSE is dominated by the depth of very sharp notches on a 0.08 GHz grid. G3 used 125k
  unknowns and 2.0 GB, so no finer mesh was run.
- **G4 (data-file parameters):** every weak CST dip (5.68, 11.20, 13.60 GHz; FEM 5.60, 11.04,
  13.44 GHz) and both notches are reproduced with a uniform shift of −1.2 to −2.2 %.
- **G4 with the Table I parameters** (superstrate and 100 Ω/sq): a completely different, broadband
  response. The G4 CST curve therefore belongs to the structure in the data file, not to the
  Table I row.

### 4.5 Polarisation conversion

`results/<design>/s11_fem.csv`, column `cross_mag`. The FEM also records the x-polarised (0,0)
reflection. Power balance holds: |co|² + |cross|² ≤ 1 at every point of every sweep.

| Design | Max cross-polar abs(S11) | At (GHz) |
|---|---|---|
| class5 (G1) | 0.651 | 9.76 (the notch) |
| class8 (G2) | 0.001 | 10.72 |
| class9 (G3) | 0.948 | 9.36 (the second notch) |
| class19 (G4) | 0.169 | 13.44 |

- **G1 and G3** are not mirror-symmetric, so a large part of their co-polar notch is conversion to
  x polarisation, not absorption. At G3's second notch, 0.948² = 90 % of the power is reflected
  cross-polarised.
- **G4:** the pattern should be symmetric. Its small 0.169 term sits on a high-Q feature at
  13.44 GHz. One trapezoid of the DXF has an arc and vertices off by 0.25 drawing units. This
  slight asymmetry may cause it (not verified).
- **Consequence:** absorption computed as 1 − |S11|² overstates the absorption of G1 and G3 unless
  the cross-polar term is subtracted. Whether CST's S11 is co-polar only depends on its
  Floquet-port mode settings, which are not known.

## 5. CPU speed

`results/timing_summary.md`, `results/<design>/run.json`. Order 2, medium mesh, PARDISO
complex-symmetric, float64, one frequency at a time. Run on an i5-11300H (4 cores, 3 MKL threads)
shared with other jobs (load average 2.4–3.3 at the start of each run).

| Design | Tets | Unknowns | Mesh (s) | Assembly (s) | Solves (s) | Per point (s) | Total 201 pts (s) | Peak RSS (MB) | GTX 1650, complex64 + refinement, 201 pts (s) |
|---|---|---|---|---|---|---|---|---|---|
| class5 (G1) | 8,187 | 50,534 | 0.8 | 1.3 | 128 | 0.64 | 131 | 810 | 35 |
| class8 (G2) | 15,525 | 97,218 | 1.3 | 2.3 | 704 | 3.50 | 708 | 1385 | 144 |
| class9 (G3) | 19,921 | 124,682 | 0.6 | 2.8 | 2015 | 10.03 | 2019 | 2019 | 210 |
| class19 (G4) | 13,340 | 83,490 | 1.5 | 2.2 | 305 | 1.52 | 309 | 1108 | 72 |
| class19_table1 (G4) | 17,827 | 111,768 | 1.3 | 2.6 | 886 | 4.41 | 890 | 1548 | – |

The last column is the same sweep on the GTX 1650 with cuDSS (section 6). Its S11 is identical to
the CPU run. class19_table1 was not run on the GPU.

Where the time goes:

- Meshing plus assembly takes under 5 s for every design. 98–99.8 % of the sweep is the per-point
  sparse factorisation and solve.
- Cost grows quickly with the number of unknowns: 0.64 s per point at 50k, 10 s at 125k.
- G3 is the largest because it has 6.1 mm of dielectric (substrate and superstrate) to mesh.

The volume mesh is the price of generality. The same formulation handles finite metal thickness,
vias, inhomogeneous or anisotropic layers, several patterned layers and oblique incidence (a
Floquet phase on the merged DOFs), none of which a sheet-only model of a uniform stack covers.

## 6. GPU

`results/gpu_gtx1650/`, `results/<design>_gpu_*/`.

**Set-up.** NVIDIA GeForce GTX 1650, 4 GB, driver 577.00 (CUDA 12.9), under WSL2; torch
2.14.1+cu126, nvmath-python 1.0.0 with cuDSS 0.8.0, cupy 14.2. The CPU numbers are the i5-11300H
runs of section 5.

**Hardware caveat.** The GTX 1650 is a consumer Turing card. Its FP64 rate is 1/32 of FP32: about
93 GFLOP/s FP64 nominal, against 198 GFLOP/s for this CPU (FP32 about 3 TFLOP/s). A complex128
factorisation was therefore expected to be no faster than the CPU, and it is about the same (0.83
to 1.21 times the CPU's speed). These numbers say nothing direct about A100-class cards.

**Tests.** With CUDA available all 19 tests pass, including
`tests/test_device.py::test_cuda_assembly_matches_cpu` (CUDA assembly equals CPU assembly to rtol
1e-10). See `results/gpu_gtx1650/pytest.txt`.

**Implementation notes.**

- cuDSS operands live in persistent GPU buffers that are updated in place each frequency, so the
  symbolic analysis is done once per sweep. Handing cuDSS new operand buffers each frequency
  (`reset_operands`) invalidates the plan: the first refactorisation then fails with
  "Factorization cannot be performed before plan() has been called".
- `--solver torch` (`torch.sparse.spsolve`) does not run with the stock PyTorch wheel, which is not
  built with cuDSS ("Calling linear solver with sparse tensors requires compiling PyTorch with CUDA
  cuDSS"). It stays in the code for a cuDSS-enabled torch build and is untested.
- `cudss32` factorises and solves in complex64. `cudss32ir` uses the complex64 factor with
  iterative refinement: the residual b − Ax is formed with A and x in complex128 on the GPU, and the
  correction is solved with the complex64 factor, until ‖r‖/‖b‖ < 1e-10 or 20 steps. The matrix is
  still combined in double precision and rounded once to complex64. Assembly is unchanged (float64).
- `scripts/bench.py` frees each backend's cuDSS handle (`CuDSS.release()`) before the next one.
  Without that, the class8 and class9 factorisations failed with `ALLOC_FAILED` because earlier
  factors were still held. Dropping the Python object alone did not free them.

**Benchmark.** `scripts/bench.py`, 5 frequencies from 4 to 18 GHz after one warm-up solve
(`results/gpu_gtx1650/bench.md`). Frequencies: 4, 8, 10, 14 and 18 GHz.

| Design | Unknowns | Assembly CPU (s) | Assembly CUDA (s) | Solver | Precision | Mean per point (s) | Speed vs PARDISO | Max abs(ΔS11) vs PARDISO | Device memory (MB) |
|---|---|---|---|---|---|---|---|---|---|
| class5 | 50,534 | 1.33 | 0.54 | pardiso | complex128 | 0.713 | 1.00× | 0.0e+00 |  |
| class5 | 50,534 | 1.33 | 0.54 | cudss | complex128 | 0.664 | 1.07× | 9.8e-14 | 1442 |
| class5 | 50,534 | 1.33 | 0.54 | cudss32 | complex64 | 0.134 | 5.31× | 7.6e-05 | 1570 |
| class5 | 50,534 | 1.33 | 0.54 | cudss32ir | complex64 + IR (complex128 residual) | 0.154 | 4.63× | 5.3e-12 | 1802 |
| class19 | 83,490 | 1.78 | 0.89 | pardiso | complex128 | 1.563 | 1.00× | 0.0e+00 |  |
| class19 | 83,490 | 1.78 | 0.89 | cudss | complex128 | 1.757 | 0.89× | 7.3e-14 | 2314 |
| class19 | 83,490 | 1.78 | 0.89 | cudss32 | complex64 | 0.312 | 5.02× | 8.7e-06 | 2564 |
| class19 | 83,490 | 1.78 | 0.89 | cudss32ir | complex64 + IR (complex128 residual) | 0.331 | 4.73× | 1.5e-12 | 2978 |
| class8 | 97,218 | 2.62 | 1.00 | pardiso | complex128 | 3.465 | 1.00× | 0.0e+00 |  |
| class8 | 97,218 | 2.62 | 1.00 | cudss | complex128 | 4.196 | 0.83× | 3.5e-14 | 2136 |
| class8 | 97,218 | 2.62 | 1.00 | cudss32 | complex64 | 0.654 | 5.30× | 9.7e-06 | 2118 |
| class8 | 97,218 | 2.62 | 1.00 | cudss32ir | complex64 + IR (complex128 residual) | 0.716 | 4.84× | 5.8e-12 | 2300 |
| class9 | 124,682 | 2.67 | 1.30 | pardiso | complex128 | 10.363 | 1.00× | 0.0e+00 |  |
| class9 | 124,682 | 2.67 | 1.30 | cudss | complex128 | 8.580 | 1.21× | 2.5e-13 | 2884 |
| class9 | 124,682 | 2.67 | 1.30 | cudss32 | complex64 | 0.943 | 10.99× | 2.2e-05 | 2862 |
| class9 | 124,682 | 2.67 | 1.30 | cudss32ir | complex64 + IR (complex128 residual) | 0.983 | 10.54× | 9.4e-12 | 3090 |

- **Assembly** on CUDA is 2–3 times faster than on the CPU, but it is a one-off 1–3 s, so it does
  not matter for a sweep.
- **Agreement:** cuDSS complex128 and complex64 with refinement agree with PARDISO to 1e-11 or
  better in S11. Plain complex64 differs by up to 8e-5 in S11 at these 5 frequencies.
- **Device memory** is the memory in use after factorisation (`cudaMemGetInfo`), for all processes,
  including about 0.4 GB held by the Windows desktop. In the benchmark several backends share one
  process, so every GPU row except each design's first includes memory left by earlier backends.
  Read them as upper bounds. The clean values come from the sweeps, one process each
  (`solver_memory_mb` in `run.json`): 1.4 GB (class5, complex128), 1.3 GB (class5, complex64),
  1.7 GB (class19), 1.9 GB (class8) and 2.4 GB (class9), the last three with complex64 and
  refinement. All four designs ran in 4 GB in both precisions.

**Full 201-point sweeps on the GPU.** `python -m fem.sweep --device cuda --solver <s> --tag gpu_<s>`;
`scripts/gpu_compare.py` writes `results/gpu_gtx1650/compare.md`. Each run is compared with the CPU
PARDISO complex128 sweep on the same mesh (section 5). dB is clipped at −40 dB as in section 4.4,
and the notch is the deepest minimum.

| Design | Run | Per point (s) | Total 201 pts (s) | Speed vs CPU | Max dB diff vs CPU | Max abs(ΔS11) | Notch (GHz) | Notch shift (GHz) | Max residual | IR steps (mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| class5 | cpu_pardiso_c128 | 0.638 | 131 | 1.00× | 0.00e+00 | 0.0e+00 | 9.76 | +0.00 | 2.4e-12 |  |
| class5 | gpu_cudss | 0.666 | 136 | 0.96× | 0.00e+00 | 0.0e+00 | 9.76 | +0.00 | 2.4e-12 |  |
| class5 | gpu_cudss32 | 0.139 | 30 | 4.59× | 3.39e-03 | 5.4e-04 | 9.76 | +0.00 | 8.5e-04 |  |
| class5 | gpu_cudss32ir | 0.161 | 35 | 3.97× | 0.00e+00 | 0.0e+00 | 9.76 | +0.00 | 9.5e-11 | 2.5 |
| class19 | cpu_pardiso_c128 | 1.516 | 309 | 1.00× | 0.00e+00 | 0.0e+00 | 17.52 | +0.00 | 2.6e-12 |  |
| class19 | gpu_cudss32ir | 0.341 | 72 | 4.44× | 0.00e+00 | 0.0e+00 | 17.52 | +0.00 | 8.7e-11 | 2.3 |
| class8 | cpu_pardiso_c128 | 3.502 | 708 | 1.00× | 0.00e+00 | 0.0e+00 | 10.24 | +0.00 | 5.1e-13 |  |
| class8 | gpu_cudss32ir | 0.699 | 144 | 5.01× | 0.00e+00 | 0.0e+00 | 10.24 | +0.00 | 1.0e-10 | 2.5 |
| class9 | cpu_pardiso_c128 | 10.026 | 2019 | 1.00× | 0.00e+00 | 0.0e+00 | 9.36 | +0.00 | 1.4e-12 |  |
| class9 | gpu_cudss32ir | 1.030 | 210 | 9.73× | 0.00e+00 | 0.0e+00 | 9.36 | +0.00 | 9.4e-11 | 2.5 |

"0" means identical in the 6 significant digits stored in `s11_fem.csv`.

**Single precision.**

- **Plain complex64 is already close.** On class5 it is 4.6 times faster than the CPU, the curve is
  within 0.0034 dB of the complex128 result everywhere (the target was 0.05 dB), and the notch does
  not move. Its relative residual, however, reaches 8.5e-4, against 2e-12 for the double-precision
  solvers.
- **Complex64 with refinement gives the double-precision answer.** On average 2.3–2.5 refinement
  steps (at most 5) bring the residual to 1e-10 at every one of the 201 points, including 2 GHz at
  the low end of the band. S11 is then identical to the CPU curve. The refinement costs 4–16 % over
  plain complex64 (benchmark and class5 sweep).
- **The speed-up over the CPU grows with size** (201-point sweeps): 4.0 times for class5 (50k
  unknowns), 4.4 for class19, 5.0 for class8 and 9.7 for class9 (125k unknowns; 2019 s → 210 s).
  The benchmark agrees: 4.6–4.8 times for class5 to class8 and 10.5 times for class9. Why the gain
  jumps for class9 was not investigated.
- **The low-frequency concern does not hold with refinement.** The curl-curl system is nearly
  singular at low k0, and the design originally kept the GPU path in complex128 for that reason
  (docs/DESIGN.md section 5). That concern holds for complex64 alone, whose residual stays large (up
  to 8.5e-4 on class5). With double-precision refinement the complex64 factor converges at every
  point from 2 to 18 GHz on all four designs.

**Against the projection.** `results/gpu_projection.md` was made before any GPU run. It assumed a
GPU reaches the same fraction of its FP64 peak as PARDISO reaches on this CPU.

- **Supported:** an FP64-weak card is about as fast as this CPU in complex128. cuDSS complex128 runs
  at 0.83–1.21 times the CPU's speed on the GTX 1650: slower on class8 and class19, faster on class9
  (8.6 against 10.4 s per point). The projection said "about as fast as the CPU" for a T4.
- **Better than projected in complex128:** by FP64 peak alone (93 against 198 GFLOP/s) the GTX 1650
  should be about 2 times slower than the CPU. It is between 17 % slower and 21 % faster, so the
  "same fraction of peak" assumption is pessimistic for cuDSS on this card. The reason was not
  profiled.
- **Not tested:** the A100 projection (0.012–0.197 s per point) cannot be checked on this card.
- **What changes:** the projection ignored complex64 with refinement. On a card where FP32 is 32
  times FP64 it is the faster route. The measured gain (4–11 times) is far below 32, probably
  because each point also spends time on work that does not scale with the FP32 rate: combining the
  matrix on the CPU, host-device copies, the triangular solves and the refinement products. This was
  not profiled.

**Memory and disk.** Peak host RSS was 1.8–2.0 GB for the GPU sweeps (loading the CUDA libraries
alone adds about 1 GB), within the 2 GB budget. The benchmark process reached about 2.6 GB on
class9, because it holds the CPU PARDISO factor and the CUDA libraries in one process. The CUDA
stack added about 6.5 GB to the virtualenv, more than the 4 GB budgeted, because torch's CUDA wheels
bring cuDNN, NCCL, cuFFT and other libraries the solver does not use.

## 7. Limitations

- Zero-thickness metal only. CST's metal thickness for these designs is not known.
- Notch frequencies are 1–4 % from CST. The medium mesh is not fully converged for thin-substrate
  cells (about 1 % on G1), and finer meshes for G2 and G3 do not fit in 2 GB with a direct solver.
- Normal incidence only. Oblique incidence needs Floquet phases on the merged periodic DOFs. This
  is straightforward but not done.
- Direct solves only. An iterative GPU solver needs a curl-curl preconditioner (AMS /
  Hiptmair–Xu); plain ILU or Jacobi stalls.
- One frequency at a time. There is no fast sweep or model-order reduction yet.

## 8. Next steps

1. **GPU:** measure complex64 with refinement on a data-centre GPU, and try a cuDSS-enabled torch
   build for `--solver torch`.
2. **Fast frequency sweep by model-order reduction.** A(k0) is a polynomial in k0 apart from the
   copper term, so a Krylov/Padé reduced model built from a few factorisations could replace most
   of the 201 solves. This is the largest remaining speed gain.
3. **Finite metal thickness** (35 µm or 17 µm copper as solid PEC, or a thin impedance volume), once
   CST's settings are known, to test the remaining 1–3 % shift.
4. **Mixed order** (order 2 near the pattern, order 1 in the air) or **prism layers** in thin
   substrates, to reach the converged mesh with less memory.
