"""Fixed-charge conductors at floating potential in an energy-density material.

Each conductor carries a prescribed net free charge; its potential is an
unknown constant adjusted by an outer Newton iteration on the charge-potential
curve Q(V) (measured on the Gauss spheres).  For a linear material this
reduces to the classical capacitance relation; for a general energy density it
is the nonlinear generalization.  The inner Dirichlet problem

    find phi minimizing  integral of w(x, -grad phi) dx,
    phi = V_i on conductor i,  phi = 0 on the shield

is solved by Newton's method; the PDE and its Jacobian are obtained by
``ufl.derivative`` from the energy alone, so the material may be nonlinear and
inhomogeneous.
"""

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import NonlinearProblem


class FloatingSolution:
    """Solution of the floating-charge problem."""

    def __init__(self, geometry, material, phi, potentials, charges, energy, order):
        self.geometry = geometry
        self.material = material
        self.phi = phi  # dolfinx Function of the potential
        self.potentials = np.asarray(potentials, dtype=float)  # conductor potentials
        self.charges = np.asarray(charges, dtype=float)  # measured charges (== targets)
        self.energy = float(energy)
        self.order = order


class _Inner:
    """Dirichlet problem for given conductor potentials (reused across solves)."""

    def __init__(self, geometry, material, order=2):
        self.geo = geometry
        self.material = material
        self.V = fem.functionspace(geometry.domain, ("Lagrange", order))
        self.phi = fem.Function(self.V)
        self.phi.x.array[:] = 0.0

        x = ufl.SpatialCoordinate(geometry.domain)
        E = -ufl.grad(self.phi)
        self.Ev = ufl.variable(E)
        self.w = material(x, self.Ev)
        self.D = ufl.diff(self.w, self.Ev)

        v = ufl.TestFunction(self.V)
        self.energy_form = fem.form(self.w * ufl.dx)
        self.residual = ufl.derivative(self.w * ufl.dx, self.phi, v)

        # Dirichlet: conductor potentials (mutable Constants) + grounded shield.
        from dolfinx import mesh as dmesh

        outer_facets = dmesh.locate_entities_boundary(geometry.domain, 2, lambda p: np.isclose(np.linalg.norm(p, axis=0), geometry.shield_radius))
        self.bcs = [fem.dirichletbc(fem.Constant(geometry.domain, 0.0), fem.locate_dofs_topological(self.V, 2, outer_facets), self.V)]
        self.constants = []
        for i in range(geometry.n_conductors):
            dofs = fem.locate_dofs_topological(self.V, 2, geometry.facet_tags.find(2 + i))
            c = fem.Constant(geometry.domain, 0.0)
            self.constants.append(c)
            self.bcs.append(fem.dirichletbc(c, dofs, self.V))

        self.problem = NonlinearProblem(
            self.residual,
            self.phi,
            petsc_options_prefix="float_",
            bcs=self.bcs,
            petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-12},
        )

        # Charge integrals on the Gauss spheres (geometric normals).
        self.q_forms = []
        from dolfinx import mesh as _dm  # noqa: F401

        X = ufl.SpatialCoordinate(geometry.domain)
        Davg = 0.5 * (self.D("+") + self.D("-"))
        dS = ufl.Measure("dS", domain=geometry.domain, subdomain_data=geometry.facet_tags)
        for i, (c, r) in enumerate(geometry.conductors):
            rv = X - ufl.as_vector(c)
            nv = rv / ufl.sqrt(ufl.dot(rv, rv))
            self.q_forms.append(fem.form(ufl.dot(Davg, nv) * dS(10 + i)))

    def set_potentials(self, values):
        for c, val in zip(self.constants, values):
            c.value = float(val)

    def solve(self):
        """Newton-solve for phi with the current potentials."""
        self.problem.solve()

    def measure_charges(self):
        return np.array([fem.assemble_scalar(f) for f in self.q_forms])

    def energy(self):
        return float(fem.assemble_scalar(self.energy_form))


def solve(geometry, material, charges, order=2, rtol=1e-8, max_outer=12):
    """Solve the floating-charge problem; returns a FloatingSolution."""
    charges = np.asarray(charges, dtype=float)
    if charges.shape != (geometry.n_conductors,):
        raise ValueError("one charge per conductor required")
    inner = _Inner(geometry, material, order)

    # Initial guess: isolated-ball capacitance C ~ 4 pi r (material scale 1).
    potentials = np.array([q / (4.0 * np.pi * r) for (_, r), q in zip(geometry.conductors, charges)])
    if not np.isfinite(potentials).all() or np.abs(potentials).max(initial=0.0) == 0.0:
        potentials = np.ones(geometry.n_conductors)
    potentials = np.where(potentials == 0.0, 1.0, potentials)

    scale = np.maximum(np.abs(charges), 1e-12)
    for _ in range(max_outer):
        inner.set_potentials(potentials)
        inner.solve()
        q = inner.measure_charges()
        if np.abs(q - charges).max() < rtol * scale.max():
            return FloatingSolution(geometry, material, inner.phi, potentials, q, inner.energy(), order)
        # Finite-difference Jacobian dQ/dV (n + 1 warm-started solves).
        n = geometry.n_conductors
        J = np.zeros((n, n))
        for j in range(n):
            step = 1e-2 * max(1.0, abs(potentials[j]))
            trial = potentials.copy()
            trial[j] += step
            inner.set_potentials(trial)
            inner.solve()
            qp = inner.measure_charges()
            J[:, j] = (qp - q) / step
        potentials = potentials + np.linalg.solve(J, charges - q)
        inner.set_potentials(potentials)

    inner.set_potentials(potentials)
    inner.solve()
    q = inner.measure_charges()
    if np.abs(q - charges).max() > 100 * rtol * scale.max():
        raise RuntimeError(f"floating-potential iteration did not converge: Q={q}, target={charges}")
    return FloatingSolution(geometry, material, inner.phi, potentials, q, inner.energy(), order)
