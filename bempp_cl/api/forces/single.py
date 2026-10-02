"""Single surface integrals: quadrature, mass matrices, projections, reconstruction.

Companion to the pairwise engine for everything that is not a double
integral: L² projections of traces onto spaces, evaluation of finite element
functions at quadrature points, mass matrices between spaces and simple
surface integrals of point-wise data.
"""

import numpy as _np
import scipy.sparse as _sp

from bempp_cl.api.integration import triangle_gauss as _triangle_gauss

from .basis import ElementData, SpaceData, evaluate_basis
from .terms import BasisArg, arg as _arg


class SurfaceQuadrature:
    """Gauss points on every element of a grid."""

    def __init__(self, grid, order=4):
        """Create the rule (bempp-cl symmetric triangle Gauss rule of the given order)."""
        self.grid = grid
        self.order = order
        pts, w = _triangle_gauss.rule(order)
        self.ref_points = _np.ascontiguousarray(pts.T)  # (nq, 2)
        self.ref_weights = _np.ascontiguousarray(w)  # (nq,) sums to 1/2
        self.elem = ElementData.from_grid(grid, _np)
        ne = grid.number_of_elements
        self.nq = len(w)
        self.element_indices = _np.arange(ne)
        # points [ne, nq, 3], weights [ne, nq] (Jacobian included), normals [ne, 3]
        self.points = self.elem.global_points(self.element_indices[:, None], _np.broadcast_to(self.ref_points, (ne, self.nq, 2)))
        self.weights = self.elem.integration_elements[:, None] * self.ref_weights[None, :]
        self.normals = self.elem.normals
        self._space_cache = {}

    @property
    def flat_points(self):
        """All points as (ne*nq, 3)."""
        return self.points.reshape(-1, 3)

    @property
    def flat_weights(self):
        return self.weights.reshape(-1)

    @property
    def flat_normals(self):
        return _np.repeat(self.normals, self.nq, axis=0)

    def space_data(self, space):
        if space.id not in self._space_cache:
            self._space_cache[space.id] = SpaceData(space, self.elem, _np)
        return self._space_cache[space.id]

    def basis(self, barg):
        """Basis operator values at all points: [ne, nq, nshape] or [ne, nq, nshape, 3]."""
        if not isinstance(barg, BasisArg):
            raise TypeError("expected a BasisArg")
        if barg.uses_velocity:
            raise ValueError("single integrals do not support the DV modifier")
        sd = self.space_data(barg.space)
        ne = self.grid.number_of_elements
        pts = _np.broadcast_to(self.ref_points, (ne, self.nq, 2))
        vals, _ = evaluate_basis(_np, self.elem, sd, barg.ops, self.element_indices, pts)
        return _np.asarray(vals)

    def integrate(self, values):
        """Integrate point data [ne, nq, ...] over the surface."""
        w = self.weights.reshape(self.weights.shape + (1,) * (values.ndim - 2))
        return _np.sum(w * values, axis=(0, 1))


def mass_matrix(test, trial, quad):
    """Sparse mass matrix ∫ test_i ⋆ trial_j between two basis arguments on the same grid."""
    test = _as_arg(test)
    trial = _as_arg(trial)
    bt = quad.basis(test)  # [ne, nq, i(,3)]
    br = quad.basis(trial)
    if test.value_type != trial.value_type:
        raise ValueError("mass_matrix needs operands of the same value type")
    if test.value_type == "scalar":
        loc = _np.einsum("eqi,eqj,eq->eij", bt, br, quad.weights)
    else:
        loc = _np.einsum("eqic,eqjc,eq->eij", bt, br, quad.weights)
    sdt, sdr = quad.space_data(test.space), quad.space_data(trial.space)
    rows = _np.repeat(sdt.local2global[:, :, None], sdr.nshape, axis=2)
    cols = _np.repeat(sdr.local2global[:, None, :], sdt.nshape, axis=1)
    M = _sp.coo_matrix((loc.ravel(), (rows.ravel(), cols.ravel())), shape=(sdt.grid_dof_count, sdr.grid_dof_count)).tocsr()
    return M


def load_vector(test, values, quad):
    """∫ test_i ⋆ f for point data f [ne, nq] or [ne, nq, 3]."""
    test = _as_arg(test)
    bt = quad.basis(test)
    if test.value_type == "scalar":
        loc = _np.einsum("eqi,eq,eq->ei", bt, values, quad.weights)
    else:
        loc = _np.einsum("eqic,eqc,eq->ei", bt, values, quad.weights)
    sdt = quad.space_data(test.space)
    out = _np.zeros(sdt.grid_dof_count)
    _np.add.at(out, sdt.local2global.ravel(), loc.ravel())
    return out


def project(values, space_or_arg, quad):
    """L² projection of point data onto a space (identity operator): coefficients."""
    barg = _as_arg(space_or_arg)
    M = mass_matrix(barg, barg, quad)
    rhs = load_vector(barg, values, quad)
    return _sp.linalg.spsolve(M.tocsc(), rhs)


def reconstruct(coeffs, space_or_arg, quad):
    """Evaluate a finite element function (with operator) at the quadrature points."""
    barg = _as_arg(space_or_arg)
    b = quad.basis(barg)
    sd = quad.space_data(barg.space)
    c = _np.asarray(coeffs)[sd.local2global]  # [ne, nshape]
    if barg.value_type == "scalar":
        return _np.einsum("eqi,ei->eq", b, c)
    return _np.einsum("eqic,ei->eqc", b, c)


def integral_of_basis(space_or_arg, quad):
    """∫ φ_i for every basis function (scalar spaces)."""
    barg = _as_arg(space_or_arg)
    return load_vector(barg, _np.ones((quad.grid.number_of_elements, quad.nq)), quad)


def _as_arg(x):
    return x if isinstance(x, BasisArg) else _arg(x)
