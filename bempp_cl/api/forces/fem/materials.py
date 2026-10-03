"""Energy-density materials: define w(x, E); everything else follows.

A material is any callable ``w(x, E) -> ufl scalar`` where ``x`` is the
ufl.SpatialCoordinate and ``E = -grad(phi)``.  Inhomogeneous materials use
``x``; nonlinear materials use ``E``.  With ``w(0) = 0`` the interior of the
conductors contributes nothing to the energy.
"""

import ufl


def linear(eps=1.0):
    """w = eps/2 |E|^2 (eps may be a constant or a ufl expression of x)."""

    def w(x, E):
        return 0.5 * eps * ufl.dot(E, E)

    return w


def kerr(eps=1.0, chi=1.0):
    """w = eps/2 |E|^2 + chi/4 |E|^4."""

    def w(x, E):
        e2 = ufl.dot(E, E)
        return 0.5 * eps * e2 + 0.25 * chi * e2**2

    return w


def saturating(eps=1.0, E0=1.0):
    """w = eps E0^2/2 (sqrt(1 + |E|^2/E0^2) - 1): D = eps E / sqrt(1 + |E|^2/E0^2)."""

    def w(x, E):
        e2 = ufl.dot(E, E)
        return 0.5 * eps * E0**2 * (ufl.sqrt(1.0 + e2 / E0**2) - 1.0)

    return w


def anisotropic(eps=(1.0, 1.0, 1.0)):
    """w = 1/2 (eps_x E_x^2 + eps_y E_y^2 + eps_z E_z^2)."""

    def w(x, E):
        return 0.5 * (eps[0] * E[0] ** 2 + eps[1] * E[1] ** 2 + eps[2] * E[2] ** 2)

    return w
