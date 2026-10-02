"""Per-element geometry and basis data derived from bempp-cl grids and spaces.

Reference element: bempp-cl's triangle with vertices (0,0), (1,0), (0,1) and
map ``x = v0 + s (v1 - v0) + t (v2 - v0)`` for ``p = (s, t)``.

Everything here is written against an array module ``xp`` so that, with
``xp = jax.numpy``, the geometry (and hence every integral) is differentiable
with respect to the vertex coordinates.
"""

import numpy as _np

from .terms import space_kind as _space_kind

# Reference gradients of the P1 hat functions [1-s-t, s, t]: shape (2, 3), rows = d/ds, d/dt.
_P1_GRAD_REF = _np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
# Vertex opposite to RWG shape function k (see shapesets.py: fn0 = (s, t-1) -> x - v2, ...).
_RWG_OPPOSITE = _np.array([2, 1, 0])


def _cross(xp, a, b):
    return xp.cross(a, b, axis=-1)


class ElementData:
    """Geometry of every element of a grid."""

    def __init__(self, xp, vertices, elements):
        """Build from ``vertices`` (3, nv) and ``elements`` (3, ne) arrays."""
        self.xp = xp
        verts = xp.asarray(vertices).T  # (nv, 3)
        elems = _np.asarray(elements)
        self.elements = elems  # (3, ne) int, numpy
        self.number_of_elements = elems.shape[1]
        corners = xp.stack([verts[elems[0]], verts[elems[1]], verts[elems[2]]], axis=1)  # (ne, 3, 3)
        self.corners = corners
        e1 = corners[:, 1] - corners[:, 0]
        e2 = corners[:, 2] - corners[:, 0]
        self.jac = xp.stack([e1, e2], axis=-1)  # (ne, 3, 2)
        n = _cross(xp, e1, e2)
        nrm = xp.sqrt(xp.sum(n * n, axis=-1))
        self.integration_elements = nrm  # = 2 * area
        self.normals = n / nrm[:, None]
        # Jacobian inverse transposed: J (JᵀJ)⁻¹, (ne, 3, 2)
        g11 = xp.sum(e1 * e1, axis=-1)
        g12 = xp.sum(e1 * e2, axis=-1)
        g22 = xp.sum(e2 * e2, axis=-1)
        det = g11 * g22 - g12 * g12
        inv = xp.stack(
            [xp.stack([g22, -g12], axis=-1), xp.stack([-g12, g11], axis=-1)],
            axis=-2,
        ) / det[:, None, None]  # (ne, 2, 2)
        self.jac_inv_t = xp.einsum("eij,ejk->eik", self.jac, inv)
        # Tangential gradients of the three hat functions: (ne, 3 shape, 3 comp)
        self.p1_grad = xp.einsum("eik,kj->eji", self.jac_inv_t, xp.asarray(_P1_GRAD_REF))
        # Edge lengths in bempp's local edge order (opposite vertex 2, 1, 0).
        self.edge_lengths = xp.stack(
            [
                xp.sqrt(xp.sum((corners[:, 0] - corners[:, 1]) ** 2, axis=-1)),
                xp.sqrt(xp.sum((corners[:, 2] - corners[:, 0]) ** 2, axis=-1)),
                xp.sqrt(xp.sum((corners[:, 1] - corners[:, 2]) ** 2, axis=-1)),
            ],
            axis=-1,
        )

    @classmethod
    def from_grid(cls, grid, xp=_np):
        """Build from a bempp-cl grid."""
        return cls(xp, grid.vertices, grid.elements)

    def global_points(self, elem_idx, pts):
        """Map reference points ``pts`` [..., 2] on elements ``elem_idx`` [...] to R³."""
        xp = self.xp
        c0 = self.corners[elem_idx, 0]  # [..., 3]
        J = self.jac[elem_idx]  # [..., 3, 2]
        return c0 + xp.einsum("...ij,...j->...i", J, pts)

    def adjacent(self, test_idx, trial_idx):
        """Boolean array: do elements test_idx[:, None] and trial_idx[None, :] share a vertex?"""
        et = self.elements[:, test_idx]  # (3, T)
        er = self.elements[:, trial_idx]  # (3, R)
        adj = _np.zeros((len(test_idx), len(trial_idx)), dtype=bool)
        for i in range(3):
            for j in range(3):
                adj |= et[i][:, None] == er[j][None, :]
        return adj


class SpaceData:
    """Dof data of a bempp-cl space in the layout used by the engine."""

    def __init__(self, space, elem, xp=_np):
        """Extract local-to-global maps, multipliers and RWG coefficients."""
        if space.requires_dof_transformation:
            raise NotImplementedError("Spaces with dof transformations (BC, RBC, dual) are not supported.")
        self.space = space
        self.xp = xp
        kind = _space_kind(space)
        self.kind = "RWG" if kind == "SNC" else kind
        self.is_snc = kind == "SNC"
        self.nshape = space.number_of_shape_functions
        self.local2global = _np.asarray(space.local2global, dtype=_np.int64)  # (ne, nshape)
        self.multipliers = xp.asarray(space.local_multipliers)  # (ne, nshape)
        self.normal_multipliers = xp.asarray(space.normal_multipliers, dtype="float64")  # (ne,)
        self.support = _np.asarray(space.support, dtype=bool)
        self.grid_dof_count = space.grid_dof_count
        self.global_dof_count = space.global_dof_count
        if self.kind == "RWG":
            self.rwg_coeff = self.multipliers * elem.edge_lengths / elem.integration_elements[:, None]
        else:
            self.rwg_coeff = None

    def local_coefficients(self, coeffs):
        """Map global coefficients (global_dof_count,) to local layout (ne, nshape)."""
        c = self.xp.asarray(coeffs)
        return c[self.local2global]


def evaluate_basis(xp, elem, sdata, ops, elem_idx, pts, DV=None):
    """Evaluate a basis operator chain on elements.

    Parameters
    ----------
    elem : ElementData
    sdata : SpaceData
    ops : tuple of operator names (see terms.py)
    elem_idx : int array [...]
    pts : reference points [..., nq, 2]
    DV : velocity Jacobians at the points, [..., nq, F, 3, 3], required for 'DV'

    Returns
    -------
    values with shape [..., nq, nshape] (scalar) or [..., nq, nshape, 3] (vector);
    if the chain contains 'DV' the result has an extra field axis:
    [..., nq, F, nshape, 3].
    """
    base = ops[0]
    kind = sdata.kind
    nq = pts.shape[-2]
    lead = pts.shape[:-1]  # [..., nq]
    normals = elem.normals[elem_idx] * sdata.normal_multipliers[elem_idx][..., None]  # [..., 3]
    mult = sdata.multipliers[elem_idx]  # [..., nshape]

    if kind == "P0":
        vals = xp.broadcast_to(mult[..., None, :], lead + (1,))
    elif kind == "P1":
        s, t = pts[..., 0], pts[..., 1]
        hats = xp.stack([1.0 - s - t, s, t], axis=-1)  # [..., nq, 3]
        if base == "id":
            vals = hats * mult[..., None, :]
        elif base == "n":
            vals = (hats * mult[..., None, :])[..., None] * normals[..., None, None, :]
        elif base in ("grad", "nxgrad"):
            g = elem.p1_grad[elem_idx] * mult[..., :, None]  # [..., 3 shape, 3]
            if base == "nxgrad":
                g = _cross(xp, xp.broadcast_to(normals[..., None, :], g.shape), g)
            vals = xp.broadcast_to(g[..., None, :, :], lead + (3, 3))
        else:
            raise ValueError(base)
    elif kind == "RWG":
        coeff = sdata.rwg_coeff[elem_idx]  # [..., 3]
        if base == "div":
            vals = xp.broadcast_to((2.0 * coeff)[..., None, :], lead + (3,))
        elif base == "id":
            x = elem.global_points(elem_idx[..., None], pts)  # [..., nq, 3]
            opp = elem.corners[elem_idx][..., _RWG_OPPOSITE, :]  # [..., 3 shape, 3]
            vals = coeff[..., None, :, None] * (x[..., None, :] - opp[..., None, :, :])  # [..., nq, 3, 3]
        else:
            raise ValueError(base)
    else:
        raise ValueError(kind)

    has_field_axis = False
    for m in ops[1:]:
        if m == "nx":
            n = normals[..., None, :]  # [..., 1, 3]
            if has_field_axis:
                n = n[..., None, None, :]
            else:
                n = n[..., None, :]
            vals = _cross(xp, xp.broadcast_to(n, vals.shape), vals)
        elif m == "DV":
            if DV is None:
                raise ValueError("Operator 'DV' requires velocity Jacobians.")
            if has_field_axis:
                raise ValueError("'DV' may appear only once in an operator chain.")
            # DV: [..., nq, F, 3, 3]; vals: [..., nq, nshape, 3] -> [..., nq, F, nshape, 3]
            vals = xp.einsum("...fij,...kj->...fki", DV, vals)
            has_field_axis = True
        else:
            raise ValueError(m)
    return vals, has_field_axis
