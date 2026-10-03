"""Velocity field families for shape derivatives.

A velocity family is a finite list of smooth fields ``V_f : R³ → R³`` with
Jacobians ``DV_f`` (``DV[i, j] = ∂V_i/∂x_j``).  Families are described by a
kind and a parameter table so that the CUDA backend can evaluate them on the
device; the array implementations here serve the numpy/jax backends.

Kinds:

    cos    V = e_α cos(a x₁) cos(b x₂) cos(c x₃),   params = (a, b, c, α)
    rot    V = e_d × (x − x_cg),                     params = (d, cg₁, cg₂, cg₃)
    trans  V = e_α,                                  params = (α,)
"""

import numpy as _np


class VelocityFamily:
    """A finite family of velocity fields."""

    def __init__(self, kind, params):
        """Create a family of the given kind with a (nfields, nparams) parameter table."""
        if kind not in ("cos", "rot", "trans"):
            raise ValueError(f"Unknown velocity family kind '{kind}'.")
        self.kind = kind
        self.params = _np.ascontiguousarray(_np.atleast_2d(_np.asarray(params, dtype="float64")))

    @property
    def nfields(self):
        """Number of fields in the family."""
        return self.params.shape[0]

    def V(self, xp, x):
        """Evaluate all fields at points ``x`` [..., 3]; returns [..., nfields, 3]."""
        p = xp.asarray(self.params)
        x = xp.asarray(x)
        if self.kind == "cos":
            a, b, c = p[:, 0], p[:, 1], p[:, 2]
            alpha = self.params[:, 3].astype(int)
            xe = x[..., None, :]  # [..., 1, 3]
            amp = xp.cos(a * xe[..., 0]) * xp.cos(b * xe[..., 1]) * xp.cos(c * xe[..., 2])  # [..., F]
            e = xp.asarray(_np.eye(3)[alpha])  # [F, 3]
            return amp[..., None] * e
        if self.kind == "rot":
            d = self.params[:, 0].astype(int)
            axis = xp.asarray(_np.eye(3)[d])  # [F, 3]
            cg = p[:, 1:4]  # [F, 3]
            rel = x[..., None, :] - cg  # [..., F, 3]
            return xp.cross(xp.broadcast_to(axis, rel.shape), rel)
        if self.kind == "trans":
            alpha = self.params[:, 0].astype(int)
            e = xp.asarray(_np.eye(3)[alpha])
            return xp.broadcast_to(e, x.shape[:-1] + e.shape)
        raise AssertionError

    def DV(self, xp, x):
        """Evaluate all Jacobians at ``x`` [..., 3]; returns [..., nfields, 3, 3]."""
        p = xp.asarray(self.params)
        x = xp.asarray(x)
        F = self.nfields
        if self.kind == "cos":
            a, b, c = p[:, 0], p[:, 1], p[:, 2]
            alpha = self.params[:, 3].astype(int)
            xe = x[..., None, :]
            ca, cb, cc = xp.cos(a * xe[..., 0]), xp.cos(b * xe[..., 1]), xp.cos(c * xe[..., 2])
            sa, sb, sc = xp.sin(a * xe[..., 0]), xp.sin(b * xe[..., 1]), xp.sin(c * xe[..., 2])
            row = xp.stack([-a * sa * cb * cc, -b * ca * sb * cc, -c * ca * cb * sc], axis=-1)  # [..., F, 3]
            e = xp.asarray(_np.eye(3)[alpha])  # [F, 3]
            return e[..., :, None] * row[..., None, :]
        if self.kind == "rot":
            d = self.params[:, 0].astype(int)
            axis = _np.eye(3)[d]
            mats = _np.zeros((F, 3, 3))
            for f in range(F):
                ax = axis[f]
                mats[f] = [[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]]
            m = xp.asarray(mats)
            return xp.broadcast_to(m, x.shape[:-1] + m.shape)
        if self.kind == "trans":
            return xp.zeros(x.shape[:-1] + (F, 3, 3))
        raise AssertionError

    def __repr__(self):
        return f"VelocityFamily(kind={self.kind!r}, nfields={self.nfields})"


def cos_family(kappa=3):
    """The cosine family of the thesis, ordered as ``idx = a + κ b + κ² c + κ³ α``."""
    params = []
    for alpha in range(3):
        for c in range(kappa):
            for b in range(kappa):
                for a in range(kappa):
                    params.append((a, b, c, alpha))
    return VelocityFamily("cos", params)


def rotations(center=(0.0, 0.0, 0.0)):
    """Rigid rotations about the three axes through ``center`` (torque)."""
    cx, cy, cz = center
    return VelocityFamily("rot", [(d, cx, cy, cz) for d in range(3)])


def translations():
    """The three constant unit fields (force)."""
    return VelocityFamily("trans", [(0,), (1,), (2,)])


class BodyTranslations(VelocityFamily):
    """Rigid translations restricted to a ball: the force on one body.

    V_f = e_alpha on {‖x − center‖ < radius} and 0 elsewhere; DV = 0.  This is
    the velocity of a multi-body configuration in which only one body is
    displaced (the two-sphere force tests of the thesis code).  Array backends
    only: the CUDA codegen knows the analytic cos/rot/trans kinds and cannot
    splice the indicator mask.
    """

    def __init__(self, center, radius):
        super().__init__("trans", [(0,), (1,), (2,)])
        self.center = _np.asarray(center, dtype="float64")
        self.radius = float(radius)

    def _mask(self, xp, x):
        d = xp.asarray(x)[..., None, :] - xp.asarray(self.center)
        return xp.linalg.norm(d, axis=-1) < self.radius  # [..., 1]

    def V(self, xp, x):
        e = xp.asarray(_np.eye(3)[self.params[:, 0].astype(int)])  # [F, 3]
        return self._mask(xp, x)[..., None] * e  # [..., F, 3]

    def DV(self, xp, x):
        return xp.zeros(x.shape[:-1] + (self.nfields, 3, 3))


def body_translations(center, radius):
    """Translations of the body occupying the ball ``(center, radius)``."""
    return BodyTranslations(center, radius)


def single_cos(a=1, b=1, c=1, alpha=0):
    """The single hard-coded cosine field of the plain CUDA variants."""
    return VelocityFamily("cos", [(a, b, c, alpha)])
