"""FEM layer for energy-density materials: geometry, solvers and forces.

The FEM side of the forces package.  A material is defined ONLY by its
energy density ``w(x, E)`` (per unit volume, ``E = -grad phi``); everything
else -- the PDE, the Newton Jacobian, the displacement field ``D = dw/dE``,
the free charge and the generalized Maxwell stress -- is derived from it, so
nonlinear and inhomogeneous materials cost nothing extra.

Modules:
* ``geometry``  -- gmsh builders: shield + conductors + Gauss surfaces.
* ``materials`` -- energy-density presets (linear, Kerr, saturating, ...).
* ``floating``  -- fixed-charge conductors at floating potential.
* ``force``     -- energy finite-difference forces and generalized MST.
"""
