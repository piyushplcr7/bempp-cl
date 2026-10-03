# bempp_cl.api.forces — shape derivatives and forces on top of bempp-cl

Read this before touching or using the subpackage. It is written for humans and for
agents running numerical experiments.

## What it is

A pairwise panel-integration engine with a declarative term algebra, plus problem
modules that solve magnetostatic boundary integral equations and evaluate
shape-derivative (force/torque) formulas from Piyush Panchal's PhD thesis.

    Form = Σ coeff · ∫_Γ∫_Γ test_op(x) ⋆ K(x, y; V) ⋆ trial_op(y)

* `terms.py`   – `arg(space, ops, coeffs)`, `term(kernel, test, trial, coeff)`, `Form`, `DV`, `nx`.
  Operators per space: P0 `id`; P1 `id, grad, nxgrad, n`; RWG `id, div`; modifiers `nx`, `DV`.
  SNC (= n × RWG) is RWG with an implicit `nx`. Contraction is fixed by value types (see docstring).
* `kernels.py` – `SL, DL, ADL, A1, C3, INTEGRABLE, COMB` (+ aliases `KV, A2, C1, OLD, N`) and
  complex `HSL(k), HDL(k)`. Convention `z = y − x`, all with `1/(4π)`.
* `velocity.py` – `cos_family(kappa)`, `rotations(center)`, `translations()`; `DV[i,j] = ∂V_i/∂x_j`.
* `engine/` – three backends with identical semantics: `cuda` (primary, hand-written
  `cuda/skeleton.cu` + spliced term glue), `jax` (portable, differentiable w.r.t. vertices via
  `vertices=`), `numpy` (slow reference). `set_backend("cuda"|"jax"|"numpy")`.
  `assemble(form)` → dense matrix in grid dofs; `evaluate(form, velocity)` → one number per
  velocity field (terms must carry coefficient vectors).
* `single.py` – surface quadrature, mass matrices, L² projections, reconstruction.
* `problems/tp_vp.py` – transmission problem, vector potential: `solve`, `mst_*`, `bem_form`,
  `source_terms`, `shape_derivative`. `problems/sources.py` – `TorusCurrent`, `UniformField`.
* `problems/matlab_ref.py` – loads `Export/export_tpvp.m` outputs and maps gypsilab dofs.
* `fem/` – FEM layer for energy-density materials: `geometry` (gmsh shield +
  conductors + Gauss spheres), `materials` (w(x, E) presets; define ONLY the energy
  density – PDE, Jacobian, D = dw/dE, charge and stress all follow via ufl), `floating`
  (fixed-charge conductors at floating potential, outer Newton on the nonlinear Q(V)
  curve – the generalization of the thesis capacitance series), `force`
  (`energy_fd_force`: fixed-charge virtual-work force, valid for ANY material;
  `mst_force`: Maxwell stress on Gauss spheres, LINEAR homogeneous media ONLY).
  IMPORTANT (verified numerically): in a nonlinear dielectric the fixed-charge force
  has a nonlocal charge-redistribution term – no local surface traction reproduces
  it (all candidates miss by >10% in a Kerr medium).  The energy route is the
  reference; `mst_force` and `energy_fd_force` agree to ~discretization error for
  linear materials.

## Invariants to check in every experiment

1. Backend agreement: `cuda`, `jax`, `numpy` must agree to ~1e-12 on `assemble` and `evaluate`
   (`test/unit/forces/test_engine.py`). The numpy/jax path is validated against bempp-cl's own
   Laplace V/K/K'/W and Maxwell EFIE/MFIE assemblers.
2. Calderón identities (`test_calderon.py`): V, W, N symmetric, K' = K^T (same-space and P0/P1
   layouts), C symmetric, DL-form = −ADL-form exactly, EFIE symmetric. These hold only up to the
   singular-rule asymmetry (see pitfalls), not machine precision.
2. Physics: sphere in uniform field must converge to the closed form (Bn, Ht); BEM and MST
   shape derivatives must agree with a gap that shrinks like h² (`test_tp_vp.py`).
3. Quadrature orders are explicit arguments (`singular_order`, `regular_order`); the thesis used
   Sauter–Schwab order 5 and a 625-point far rule. Report them with every result.
4. Translations give force, `rotations(center)` give torque, `cos_family(3)` the 81 dual-norm fields
   ordered `idx = a + 3b + 9c + 27α` exactly as in MATLAB.
5. Dirichlet trace sign: gypsilab NED = ψ × n = −SNC. The kernels contract `td = −g_snc`.

## Conventions and pitfalls

* bempp RWG: `mult · len / (2|e|) · (x − v_{2−j})`, sign +1 on the lower element index.
  gypsilab RWG: `flux · (x − v_k)/(2|e|)`; mapping in `matlab_ref.py`.
* bempp-cl's Sauter–Schwab singular rules are NOT mirror-symmetric under test/trial swap (same
  pair lists and weights, different points). Assembled Galerkin matrices therefore inherit an
  x↔y asymmetry at the singular-rule error level: ~5e-7 (SL), ~6e-5 (K/K'), ~3e-4 (C) relative at
  order 4 on a coarse sphere, shrinking with `singular_order`. bempp's own assemblers show the
  same, and the engine matches each bempp matrix to ~1e-15. Exact discrete symmetry would need
  orientation-canonicalised singular rules in `pairs.py` (at the cost of the 1e-12 bempp match).
* Coefficient vectors live in *global* dofs; the engine maps to local layout itself.
* CUDA backend: double precision only, real kernels only (complex forms → use jax/numpy).
* Consumer GPUs run FP64 at 1/64 rate; the far-field kernel is compute-bound there
  (~0.3 TFLOP/s on an RTX 4070 Ti SUPER, 81 fields on 8192 elements ≈ 90 s).
* Never assume energy stationarity at the discrete solution: any AD/energy mode must
  differentiate through the solve.

## Running

    .venv-forces/bin/python -m pytest test/unit/forces -q
    FORCES_MATLAB_REF=/path/to/exports .venv-forces/bin/python -m pytest test/unit/forces/test_matlab_ref.py

FEM cross-validation (`test_fem.py`) needs dolfinx + gmsh (no pip wheels; skip-if-missing elsewhere):

    conda create -y -n forces-fem -c conda-forge python=3.12 fenics-dolfinx numpy scipy numba meshio pytest
    ~/miniconda3/envs/forces-fem/bin/pip install --no-deps -e . gmsh jax
    ~/miniconda3/envs/forces-fem/bin/python -m pytest test/unit/forces/test_fem.py

It solves the two-permeable-sphere problem independently with dolfinx (P2, Maxwell
stress on Gauss spheres in the homogeneous matrix) and checks the BEM MST force and
the body-restricted shape derivative (`body_translations`) against it.
`test_fem_floating.py` tests the FEM layer itself: concentric capacitance
(analytic), image-charge force (analytic), and the force-route agreement gates.
