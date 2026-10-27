# Design

This document describes the formulation and the main design decisions of the solver. It was
written as the plan before the code, and has been updated where the implementation departed from
it. Those places are marked "As built".

## 0. Problem definition

| Item | Value |
|---|---|
| Unit cell | a = b = 10 mm, square lattice |
| Stack | zero-thickness pattern at z = 0 on a substrate (ε_r, tan δ, h), PEC ground at z = −h; optional identical superstrate on z ∈ [0, h] |
| Material | ε_c = ε_r(1 − j tan δ), time convention e^{jωt} |
| Conductor | copper as a surface impedance Z_s = (1+j)√(ωμ0/2σ), σ = 5.8e7 S/m, when R_s = 0; a resistive sheet Z_s = R_s otherwise |
| Excitation | normal incidence, y-polarised plane wave from z > 0 |
| S11 | co-polar (y) specular reflection; magnitude in dB = 20 log10 abs(S11); phase referenced to the top air interface of the stack (z = 0, or z = h with a superstrate) |
| Sweep | 2–18 GHz, 201 points, linear spacing (0.08 GHz step) |
| CST data file | one CSV row: column A class id, B ε_r, C 1, D tan δ, E 0, F h (mm), G flag, H superstrate (0/1), I R_s (Ω/sq, 0 = copper), then 201 values of abs(S11) (linear) |
| Pattern | DXF: the first closed polyline is the cell boundary (0 to 500 units), the others are the metal polygons; 500 units = 10 mm, y up |

The four designs (names G1 to G4; parameters from the CST data files):

| Design | Name | ε_r | tan δ | h (mm) | Superstrate | Conductor | CST notch |
|---|---|---|---|---|---|---|---|
| class5 | G1 | 3.45 | 0.0025 | 0.762 | no | copper | 10.16 GHz, −15.31 dB |
| class8 | G2 | 3.54 | 0.0033 | 3.048 | no | 50 Ω/sq | 10.08 GHz, −25.41 dB |
| class9 | G3 | 2.94 | 0.0021 | 3.05 | yes | copper | 7.44 GHz, −35.65 dB |
| class19 | G4 | 10.2 | 0.0023 | 1.27 | data file: no; Table I: yes | data file: copper; Table I: 100 Ω/sq | 15.68 GHz, −13.46 dB |

For G4 the original design table and the CST data file disagree. The data file's own parameter columns
say "no superstrate, copper", and its curve is the G4 CST reference. The solver runs the data-file
parameters as the primary case (`class19`) and the Table I variant (superstrate and 100 Ω/sq) as a
secondary case (`class19_table1`). REPORT.md section 4.4 shows which one agrees with CST.

Higher Floquet orders: the first (±1, 0) and (0, ±1) harmonics cut on in air at c/a = 30 GHz. In
2–18 GHz only the (0,0) TE/TM pair (the x- and y-polarised plane waves) propagates above the stack.
Inside ε_r = 10.2 they cut on at 9.4 GHz, but they stay trapped (evanescent in air).

## 1. Formulation

Frequency-domain vector wave equation for the total electric field E in one unit cell Ω:

    ∇ × (μ_r⁻¹ ∇ × E) − k0² ε_c(r) E = 0   in Ω,   k0 = ω/c0.

Weak form with test functions W in H(curl):

    ∫_Ω (∇×E)·(∇×W) − k0² ε_c E·W dV
      + j k0 ∫_Γp E_t·W_t dS                         (port / first-order ABC)
      + j k0 η0 ∫_S (1/Z_s) E_t·W_t dS               (impedance sheet at z = 0)
      = 2 j k0 ∫_Γp E_inc,t·W_t dS                    (incident plane wave)

**Port terms.** With e^{jωt} and the outward normal +z on the port plane Γp at z = z_p, write
E = E_inc + E_sc, with E_inc = ŷ E0 e^{+jk0 z} (travelling in −z) and E_sc outgoing ∝ e^{−jk0 z}.
For these plane waves n × ∇ × E_sc = +jk0 E_sc,t and n × ∇ × E_inc = −jk0 E_inc,t, so
n × ∇ × E = jk0 E_t − 2jk0 E_inc,t. This is exact for the (0,0) Floquet pair, which is all that
propagates in air below 30 GHz. Evanescent harmonics see a mismatch, so the port sits about one
period (10 mm) above the stack. There the slowest one, (±1,0) at 18 GHz with |k_z| = 0.50 mm⁻¹,
has decayed by e^{−5} (round trip about 4e-5).

**Sheet term.** The pattern is a zero-thickness surface S at z = 0 carrying J_s = E_t / Z_s. From
∇×∇×E − k0²ε E = −jωμ0 J, the surface current adds jωμ0 ∫ J_s·W = j k0 η0 ∫ (1/Z_s) E_t·W_t.

- Resistive sheets (class8: 50 Ω/sq; class19_table1: 100 Ω/sq) use this term.
- Copper has two options. `pec` sets tangential E = 0 on the metal triangles (Dirichlet
  elimination). `sibc` uses this term with Z_s = (1+j)√(ωμ0/2σ), which includes conductor loss.
  The default is `sibc`, because the CST notch depths include conductor loss; the PEC run is a
  cross-check. With 1/Z_s ≈ 30 S the sheet term is a large penalty, which a double-precision direct
  solver handles.

**S11 extraction.** The y-polarised (0,0) mode amplitude is the cell average of E_y on the port
plane, since all other Floquet harmonics average to zero over a period:

    R0 = ( <E_y>_Γp − E0 e^{jk0 z_p} ) · e^{jk0 z_p}                 (reference z = 0)
    S11 = R0 · e^{−2jk0 z_ref},  z_ref = top air interface of the stack (0 or h)

The cross-polar (x) reflection comes from <E_x> in the same way and is saved with every sweep.

**Elements.** The plan started with first-order Nédélec (Whitney) edge elements on tetrahedra
(6 DOFs per tetrahedron), with second order (Nédélec first kind, 20 DOFs per tetrahedron,
hierarchical) as a later phase if order 1 could not reach the accuracy target within 2 GB.

As built: order 2 is the default. Order 1 converges too slowly in the 0.762 mm G1 substrate
(REPORT.md section 4.3). Order 1 remains available with `--order 1`, and the mesh presets in
`fem/mesh.py` are tuned for order 2.

## 2. Boundaries

| Boundary | Treatment |
|---|---|
| x = 0 / x = a, y = 0 / y = a | Periodic (Floquet with zero phase at normal incidence). gmsh `setPeriodic` gives a conforming slave-to-master node map. Before edges are numbered, slave node ids are replaced by master ids (transitively, for the corner lines), so slave edges become master edges. The mesh is topologically a torus and needs no constraint equations. Element matrices still use the true coordinates. Oblique incidence would need a phase factor on the merged DOFs. |
| z = −h ground | PEC: tangential E = 0 on its edges (eliminated). |
| z = z_p port | First-order absorbing condition and incident plane wave (above); S11 from the mode average. |
| z = 0 pattern | PEC (Dirichlet on metal-triangle edges) or the impedance sheet term. |
| Layer interfaces | Natural: tangential E continuity is built into edge elements. |

A PML is not needed, because only the (0,0) pair propagates. The port-distance study (REPORT.md
section 4.3) confirms that the result does not depend on the port position.

## 3. Geometry and meshing

- DXF to polygons with `ezdxf` and `shapely` (scale 0.02 mm per unit). Collinear and
  near-duplicate vertices are removed with a 0.01 mm simplification. One small arc in the class19
  pattern is approximated by its chord.
- gmsh Python API (OpenCASCADE kernel). The headless `nox` build is used because the standard wheel
  needs libGLU. Boxes for the substrate [−h, 0], the optional superstrate [0, h] and the air
  [z_top, z_p]; the pattern polygons are plane surfaces at z = 0. Everything is fragmented, so the
  z = 0 plane is split into metal and non-metal faces and the mesh conforms to the pattern.
  Periodic faces are paired with translation transforms.
- Sizes (presets `coarse`, `medium`, `fine`):
  - at the metal edges h_e (0.20 / 0.12 / 0.08 mm in the original plan), grown by a
    Distance+Threshold field over about 1 mm;
  - in dielectrics min(λ_d(18 GHz)/10, 0.6 mm), where λ_d = 5.2 mm for ε_r = 10.2;
  - in air λ0(18 GHz)/8 ≈ 2 mm.
- Mesh files (`.msh`) are not stored; they are rebuilt in seconds.

As built: the presets were retuned for order 2 elements; `fem/mesh.py` holds the current values.
Merging periodic nodes is only valid when no element spans half a period in x or y, so
`fem/mesh.py` rejects meshes that are too coarse.

## 4. Assembly

- `torch` tensors on the selected `--device` (cpu or cuda), batched over all tetrahedra:
  barycentric gradients from the 4×4 vertex matrix inverse, curls 2∇λ_i×∇λ_j, and the closed-form
  mass integrals ∫λ_iλ_j = V(1+δ_ij)/20. The result is (n_tet, 6, 6) curl-curl and mass blocks for
  order 1. Order 2 uses hierarchical basis functions and quadrature (`fem/basis.py`). Triangle
  traces for the port and the sheet are built the same way.
- Global DOF numbering uses the periodic-merged node ids. Local vertices are sorted by merged id,
  so no orientation signs are needed.
- Because ε_c and Z_s are constant per region, the system is

      A(f) = S − k0² (Σ_r ε_r,c M_r) + j k0 B_p + j k0 η0 / Z_s(f) B_s,

  a few matrices with one shared upper-triangular sparsity pattern. A new frequency only
  recombines the value arrays, with no reassembly. Dirichlet DOFs are dropped by index once.

## 5. Linear solve

- The matrix is complex symmetric (not Hermitian) and indefinite.
- CPU: MKL PARDISO with mtype 6 (complex symmetric), through a small `ctypes` wrapper on the
  `mkl_rt` library that the `pypardiso` wheel ships (pypardiso itself is real-only). Only the upper
  triangle is stored, which halves the memory. Fallback: `scipy.sparse.linalg.splu` (SuperLU).
- GPU (`--solver`):
  - `cudss`: NVIDIA cuDSS sparse direct solver through `nvmath-python`, the GPU analogue of
    PARDISO;
  - `torch`: `torch.sparse.spsolve`, where the CUDA build supports it;
  - an iterative solver (GMRES or BiCGStab) would need a curl-curl aware preconditioner. Plain ILU
    or Jacobi stalls because of the gradient null space of the curl operator.
- Low-frequency breakdown: S has the discrete gradients in its kernel, so A ≈ S at low k0 is nearly
  singular (here k0 h_e ≈ 4e-4 at 2 GHz). Direct solvers in float64 handle this range. The plan
  therefore kept the GPU path in complex128. An iterative GPU solver would need tree-cotree gauging
  or an auxiliary-space (Hiptmair–Xu / AMS) preconditioner.

As built: two more GPU solvers were added. `cudss32` factorises and solves in complex64.
`cudss32ir` uses the complex64 factor with iterative refinement, forming the residual in
complex128. Complex64 alone leaves a residual up to 8.5e-4; with refinement the residual reaches
1e-10 at every frequency from 2 to 18 GHz on all four designs (REPORT.md section 6).

- Memory: peak RAM was budgeted at about 2 GB. The sweep runs one frequency at a time, and the
  symbolic factorisation is computed once and reused across frequencies (same pattern).

## 6. Frequency sweep

- One direct solve per frequency. The mesh, the assembled parts and the symbolic analysis are
  reused, so each point costs one numeric factorisation and solve.
- 201 points by default (the CST grid). `--n-freq` allows coarse development runs.
- Not done: a model-order-reduction fast sweep (Krylov/Padé on A(k0) = S − k0²M + jk0B + …), which
  would need only a few factorisations for the whole band.

## 7. Validation plan

1. **Element unit tests:** Whitney functions reproduce constant fields (patch test: for a constant
   E0 the edge DOFs E0·t_e interpolate it exactly, the curl-curl energy is zero, and the mass energy
   equals ε|E0|²V); tangential continuity across shared faces; gradient fields lie in the kernel of
   S.
2. **Analytic slab checks:** no pattern, S11 compared with the transmission-line result
   (`fem/analytic.py`):
   - a grounded slab (class5 stack), magnitude and phase;
   - a grounded slab with superstrate (class9 stack);
   - a uniform resistive sheet on a grounded slab (Salisbury screen, 377 Ω/sq on 3 mm) to test the
     sheet term: Y_in = 1/R_s + Y_down.
3. **Port distance:** S11 for one design with z_p − z_top = 5, 10 and 15 mm.
4. **Mesh convergence** on class5 (thin strips, sharpest notch): notch frequency and depth for
   coarse, medium and fine meshes.
5. **The four designs** against the CST curves, with class19 in both parameter variants.

## 8. Metrics and acceptance targets

Computed on the 201-point grid in `fem/metrics.py` and written to `results/<design>/metrics.json`:

- the notch frequency error Δf/f for each CST notch deeper than −6 dB, matched to the nearest FEM
  local minimum, and the notch depth difference in dB;
- the RMSE of abs(S11) in dB over the band, with dB clipped at −40 dB so that the bottom of a deep
  notch does not dominate, and the RMSE of linear abs(S11).

Targets, set before any results:

| Metric | Target against CST |
|---|---|
| Analytic slab abs(S11) error | < 0.01 dB, phase < 1° |
| Notch frequency | abs(Δf/f) ≤ 2 % |
| S11 RMSE (dB, clipped at −40 dB) | ≤ 1.5 dB |
| Notch depth | reported, no target (depth at a sharp notch depends on the 0.08 GHz sampling) |

No parameter is tuned to fit CST. When an alternative model is run (PEC against SIBC, the two
class19 variants), all variants are reported.

## 9. Risks considered

| Risk | Mitigation |
|---|---|
| RAM above 2 GB for fine meshes (3-D fill-in) | complex-symmetric PARDISO, coarse presets first, peak RSS reported |
| Thin-strip notches (class5) sensitive to edge refinement | refinement field at polygon edges, convergence study |
| Sheet singularities and first-order dispersion shift resonances down | convergence study; second-order elements |
| Conditioning of the copper SIBC penalty | comparison with PEC runs |
| class19 parameter ambiguity | both variants run and reported |
| gmsh periodic and fragment face matching | faces matched by bounding box; node maps asserted complete |
| Polarisation and orientation conventions for asymmetric cells (class5, class9) | y = DXF y is the primary case; the cross-polar reflection is recorded |
| Run time (201 points × 4 designs on a shared CPU) | coarse meshes for development, fine runs only for final results |
