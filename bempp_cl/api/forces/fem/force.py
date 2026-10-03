"""Forces on conductors: energy finite differences and generalized MST.

Two independent routes:

* ``energy_fd_force`` -- rigidly displace one conductor by +/- delta, re-solve
  the floating-charge problem (charges are held fixed, so the potentials
  re-adjust) and take F_k = -(E(delta) - E(-delta)) / (2 delta).  Valid for
  any energy density; the sign convention is the fixed-charge one.
* ``mst_force`` -- Maxwell stress sigma = D (x) E - w I with D = dw/dE,
  integrated over the Gauss sphere of a conductor with the geometric normal.
  LINEAR homogeneous media near the Gauss surface only: there it reduces to
  the classical Maxwell stress tensor.  In nonlinear materials the
  fixed-charge force acquires a nonlocal charge-redistribution term and NO
  local surface traction reproduces it (numerically verified: all candidate
  tractions miss by >10% in a Kerr medium) -- use ``energy_fd_force`` there.
"""

import numpy as np
import ufl
from dolfinx import fem

from . import floating
from .geometry import build_geometry


def energy_fd_force(conductors, charges, material, which=0, delta=0.02, directions=(0, 1, 2), **geo_kwargs):
    """Fixed-charge force on conductor ``which`` by energy finite differences.

    ``conductors`` is the geometry spec ``[((cx, cy, cz), r), ...]``; all other
    keyword arguments are forwarded to ``build_geometry``.  Returns the force
    components for the requested ``directions``.
    """
    conductors = [(np.asarray(c, dtype=float), float(r)) for c, r in conductors]
    charges = np.asarray(charges, dtype=float)
    order = geo_kwargs.pop("order", 2)
    out = []
    for k in directions:
        energies = []
        for sign in (+1.0, -1.0):
            shifted = [(c + sign * delta * np.eye(3)[k], r) for c, r in conductors]
            geo = build_geometry(shifted, **geo_kwargs)
            sol = floating.solve(geo, material, charges, order=order)
            energies.append(sol.energy)
        out.append(-(energies[0] - energies[1]) / (2 * delta))
    return np.array(out)


def mst_force(solution, which=0):
    """Generalized Maxwell-stress force on conductor ``which``."""
    geo = solution.geometry
    domain = geo.domain
    phi = solution.phi
    x = ufl.SpatialCoordinate(domain)
    E = -ufl.grad(phi)
    Ev = ufl.variable(E)
    w = solution.material(x, Ev)
    D = ufl.diff(w, Ev)

    Eavg = 0.5 * (E("+") + E("-"))
    Davg = 0.5 * (D("+") + D("-"))
    wavg = 0.5 * (w("+") + w("-"))
    sigma = ufl.outer(Davg, Eavg) - wavg * ufl.Identity(3)

    X = ufl.SpatialCoordinate(domain)
    c, _ = geo.conductors[which]
    rv = X - ufl.as_vector(c)
    nv = rv / ufl.sqrt(ufl.dot(rv, rv))
    dS = ufl.Measure("dS", domain=domain, subdomain_data=geo.facet_tags)
    return np.array([fem.assemble_scalar(fem.form(ufl.dot(sigma, nv)[i] * dS(10 + which))) for i in range(3)])
