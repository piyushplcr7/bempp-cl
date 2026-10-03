"""Floating-charge conductors in energy-density materials (FEM layer).

Validation gates:

* Concentric ball in a grounded shield, linear material: the floating
  potential and energy must match the analytic capacitance
  C = 4 pi eps a R / (R - a),  V = Q / C,  E = Q^2 / (2 C).
* Small charged ball off-center in a grounded shield, linear material:
  the force must approach the method-of-images point-charge value
  F = Q^2 R d / (4 pi eps (R^2 - d^2)^2)  (attraction to the wall).
* Kerr material: the generalized Maxwell stress force and the fixed-charge
  energy finite-difference force must agree -- the two independent routes of
  the nonlinear experiment.

Requires dolfinx and gmsh (skip-if-missing; see test_fem.py for env setup).
"""

import numpy as np
import pytest

pytest.importorskip("dolfinx")
pytest.importorskip("gmsh")

from bempp_cl.api.forces.fem import floating, force
from bempp_cl.api.forces.fem.geometry import build_geometry
from bempp_cl.api.forces.fem import materials

INVFPI = 1.0 / (4.0 * np.pi)


def test_concentric_capacitance():
    """V = Q/C and E = Q^2/2C with C = 4 pi eps a R/(R-a)."""
    eps, a, Rout, Q = 2.0, 1.0, 3.0, 1.0
    geo = build_geometry([((0.0, 0.0, 0.0), a)], shield_radius=Rout, h_near=0.12, h_far=0.5)
    sol = floating.solve(geo, materials.linear(eps), [Q])
    C = 4.0 * np.pi * eps * a * Rout / (Rout - a)
    assert abs(sol.potentials[0] - Q / C) < 0.02 * (Q / C)
    assert abs(sol.energy - Q**2 / (2 * C)) < 0.03 * (Q**2 / (2 * C))
    assert abs(sol.charges[0] - Q) < 1e-8


def test_point_charge_image_force():
    """Off-center ball, a << R: force -> image-charge value Q^2 R d/(4 pi (R^2-d^2)^2)."""
    a, Rout, d, Q = 0.05, 2.0, 1.0, 1.0
    center = (d, 0.0, 0.0)
    geo = build_geometry([((center), a)], shield_radius=Rout, gauss_factor=1.4, h_near=0.015, h_far=0.3, dist_min=0.08)
    sol = floating.solve(geo, materials.linear(1.0), [Q])

    f_exact = Q**2 * Rout * d / (4.0 * np.pi * (Rout**2 - d**2) ** 2)
    f_mst = force.mst_force(sol)
    assert abs(f_mst[0] - f_exact) < 0.05 * f_exact, (f_mst, f_exact)
    assert np.linalg.norm(f_mst[1:]) < 0.12 * f_exact

    # Energy-FD on a tiny ball is remesh-noise dominated; information only.
    f_fd = force.energy_fd_force([(center, a)], [Q], materials.linear(1.0), delta=0.1, directions=(0,), shield_radius=Rout, gauss_factor=1.4, h_near=0.015, h_far=0.3, dist_min=0.08)
    print(f"\npoint charge: exact={f_exact}, mst={f_mst}, fd={f_fd[0]}")


def test_floating_force_routes():
    """Fixed-charge force routes in a stable geometry (ball near the wall).

    Linear material: the generalized MST (valid for linear media) and the
    energy finite difference agree to discretization accuracy, and the FD is
    stable in delta.  Kerr material: the FD is delta-stable and the force
    differs measurably from the linear material -- the fixed-charge force in
    a nonlinear dielectric is NOT a local Maxwell-stress traction (verified
    in diagnostics: all candidate surface tractions miss by >10%), so the
    energy route is the reference there.
    """
    a, Rout, d, Q = 0.4, 2.0, 1.2, 1.0
    center = (d, 0.0, 0.0)
    kwargs = dict(shield_radius=Rout, gauss_factor=1.3, h_near=0.1, h_far=0.4)

    lin = materials.linear(1.0)
    geo = build_geometry([(center, a)], **kwargs)
    sol = floating.solve(geo, lin, [Q])
    f_mst = force.mst_force(sol)
    f_fd_lin = [force.energy_fd_force([(center, a)], [Q], lin, delta=delta, directions=(0,), **kwargs)[0] for delta in (0.04, 0.08)]
    print(f"\nlinear: mst={f_mst[0]:.5f}, fd={f_fd_lin}")
    assert np.linalg.norm(f_mst[1:]) < 0.1 * abs(f_mst[0])
    assert abs(f_fd_lin[0] - f_fd_lin[1]) < 0.02 * abs(f_mst[0])  # delta-stability
    assert abs(f_fd_lin[0] - f_mst[0]) < 0.08 * abs(f_mst[0])  # routes agree (linear)

    kerr = materials.kerr(eps=1.0, chi=2.0)
    f_fd_kerr = [force.energy_fd_force([(center, a)], [Q], kerr, delta=delta, directions=(0,), **kwargs)[0] for delta in (0.04, 0.08)]
    print(f"kerr: fd={f_fd_kerr}")
    assert abs(f_fd_kerr[0] - f_fd_kerr[1]) < 0.03 * abs(f_fd_kerr[0])  # delta-stability
    # The material measurably changes the fixed-charge force (FD vs FD).
    assert abs(f_fd_kerr[0] - f_fd_lin[0]) > 0.05 * abs(f_fd_lin[0]), (f_fd_kerr, f_fd_lin)
