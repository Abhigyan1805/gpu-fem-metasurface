# Development

## Setup

See the README for the CPU and CUDA install. The virtualenv lives in `.venv/` and is not committed.

Run the tests:

```bash
.venv/bin/python -m pytest -q tests
```

On a machine without CUDA, 18 tests pass and the CUDA assembly test is skipped.

## Regenerating the results

Every file in `results/` can be rebuilt with the commands below. The full CPU sweeps take from
about 2 minutes (class5) to about 35 minutes (class9) each on a 4-core laptop.

CPU sweeps, medium mesh, one design at a time:

```bash
scripts/run_all.sh class5 class8 class9 class19 class19_table1
.venv/bin/python -m fem.sweep --design class5 --preset fine --tag fine
.venv/bin/python -m fem.sweep --design class19 --preset fine --tag fine
```

Studies and summary tables:

```bash
.venv/bin/python scripts/convergence.py     # results/convergence/*.json (class5 mesh, order and port distance; slab check)
.venv/bin/python scripts/summary.py         # accuracy_summary.{json,md}, s11_all_designs.png, convergence tables and plot
.venv/bin/python scripts/timing_table.py    # timing_summary.{json,md}
.venv/bin/python scripts/factor_cost.py     # factor_cost.json (PARDISO factorisation work)
.venv/bin/python scripts/gpu_projection.py  # gpu_projection.json (projection, not a measurement)
.venv/bin/python scripts/replot.py          # redraw every results/<run>/s11_compare.png from saved CSVs
```

GPU sweeps and benchmark (CUDA install required):

```bash
.venv/bin/python -m fem.sweep --design class5 --device cuda --solver cudss --tag gpu_cudss
.venv/bin/python -m fem.sweep --design class5 --device cuda --solver cudss32 --tag gpu_cudss32
for d in class5 class19 class8 class9; do
  .venv/bin/python -m fem.sweep --design $d --device cuda --solver cudss32ir --tag gpu_cudss32ir
done
.venv/bin/python scripts/bench.py --designs class5 class19 class8 class9 --devices cpu cuda \
    --solvers pardiso cudss cudss32 cudss32ir --out results/gpu_gtx1650/bench.json
.venv/bin/python scripts/gpu_compare.py class5 class19 class8 class9
```

Timings depend on the machine and its load. The S11 values do not, apart from round-off.

## History

The solver was built in a sequence of milestones:

1. Design document (docs/DESIGN.md) and repository skeleton; design inputs copied to `data/`.
2. DXF loader and gmsh layered periodic mesh. The metal area of the mesh equals the DXF area
   exactly.
3. Whitney (order 1) and hierarchical Nédélec order 2 elements, merge-based periodic DOFs, the
   PARDISO complex-symmetric solver, port S11 and the sheet impedance, with element and slab
   tests. Order 2 became the default, because order 1 converged too slowly in the 0.762 mm G1
   substrate within the 2 GB memory budget.
4. Sweep command line, metrics and comparison plots; the class5 convergence, port-distance and
   slab studies.
5. All four designs and the class19 Table I variant on the medium mesh; class5 and class19 on the
   fine mesh; accuracy and timing summaries.
6. GPU path: `fem/gpu.py` (cuDSS, torch spsolve), `scripts/bench.py`, a Kaggle T4 notebook
   (prepared, not run) and a projection of GPU time from the measured factorisation work.
7. GPU measurements on a local GeForce GTX 1650. The cuDSS wrapper was fixed to keep its operands
   in persistent buffers, and the `cudss32` (complex64) and `cudss32ir` (complex64 with complex128
   iterative refinement) solvers were added. All 19 tests pass with CUDA.

## Open items

- Run the T4 notebook (`notebooks/kaggle_gpu_benchmark.ipynb`). It predates `cudss32ir`, so add that
  solver first.
- Fast frequency sweep by model-order reduction.
- Finite metal thickness.
- Mixed element order or prism layers in thin substrates.
- Oblique incidence.
- An AMS-preconditioned iterative GPU solver.
