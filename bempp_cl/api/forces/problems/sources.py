"""Source fields: vector potentials A_J and their derivatives.

``TorusCurrent`` reproduces the thesis' torus current loop (surface current
J = (-sin θ, cos θ, 0) on the torus with major radius R and minor radius r,
integrated with a 40x40 Gauss-Legendre rule in (θ, φ)).  ``UniformField`` is
the analytic vector potential of a constant magnetic induction B0.
"""

import numpy as _np

_INV4PI = 1.0 / (4.0 * _np.pi)


class UniformField:
    """A_J = ½ B0 × x, curl A_J = B0."""

    def __init__(self, B0):
        self.B0 = _np.asarray(B0, dtype="float64")

    def A(self, X):
        return 0.5 * _np.cross(self.B0[None, :], X)

    def curlA(self, X):
        return _np.broadcast_to(self.B0, X.shape).copy()

    def jacA(self, X):
        """∂A_k/∂x_i as [n, 3 (i), 3 (k)]: A = ½ B0 × x -> ∂_i A_k = ½ ε_{k m i} B0_m."""
        J = _np.zeros((X.shape[0], 3, 3))
        eps = _np.zeros((3, 3, 3))
        eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1
        eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1
        M = 0.5 * _np.einsum("kmi,m->ik", eps, self.B0)
        J[:] = M
        return J

    def jac_curlA(self, X):
        """∂_i (curl A_J)_k as [n, 3 (i), 3 (k)]: zero for a uniform field."""
        return _np.zeros((X.shape[0], 3, 3))


class TorusCurrent:
    """Surface current loop on a torus, as in the thesis (unit current density)."""

    def __init__(self, R=2.0, r=0.5, J=1.0, npoints=40, center=(0.0, 0.0, 0.0)):
        self.R, self.r, self.J = float(R), float(r), float(J)
        self.center = _np.asarray(center, dtype="float64")
        x, w = _np.polynomial.legendre.leggauss(npoints)
        x = _np.pi * (x + 1.0)  # map [-1, 1] -> [0, 2π]
        w = _np.pi * w
        theta = _np.tile(x, npoints)
        phi = _np.repeat(x, npoints)
        self.WW = _np.tile(w, npoints) * _np.repeat(w, npoints)
        self.Y = _np.stack(
            [(R + r * _np.cos(phi)) * _np.cos(theta), (R + r * _np.cos(phi)) * _np.sin(theta), r * _np.sin(phi)], axis=1
        ) + self.center
        JJ = _np.stack([-_np.sin(theta), _np.cos(theta), 0 * theta], axis=1)
        surf = r * (R + r * _np.cos(phi))
        self.JW = JJ * (surf * self.WW)[:, None] * self.J  # current × surface element × weight

    _CHUNK = 1024

    def _chunked(self, fn, X, out_shape):
        out = _np.empty((X.shape[0],) + out_shape)
        for s in range(0, X.shape[0], self._CHUNK):
            out[s : s + self._CHUNK] = fn(X[s : s + self._CHUNK])
        return out

    def _diff(self, X):
        """x - y for all evaluation points and source points: [n, N, 3]."""
        return X[:, None, :] - self.Y[None, :, :]

    def A(self, X):
        def f(X):
            d = self._diff(X)
            r = _np.linalg.norm(d, axis=-1)
            return _INV4PI * _np.einsum("nN,Nk->nk", 1.0 / r, self.JW)

        return self._chunked(f, X, (3,))

    def curlA(self, X):
        """curl_x A_J = (1/4π) ∫ ∇_x(1/|x-y|) × J dS(y)."""

        def f(X):
            d = self._diff(X)
            r = _np.linalg.norm(d, axis=-1)
            gradG = -d / r[..., None] ** 3  # ∇_x (1/|x-y|) = (y - x)/|x-y|³
            return _INV4PI * _np.einsum("nNk->nk", _np.cross(gradG, self.JW[None, :, :]))

        return self._chunked(f, X, (3,))

    def jacA(self, X):
        """∂A_k/∂x_i as [n, 3 (i), 3 (k)]."""

        def f(X):
            d = self._diff(X)
            r = _np.linalg.norm(d, axis=-1)
            gradG = -d / r[..., None] ** 3  # [n, N, 3(i)]
            return _INV4PI * _np.einsum("nNi,Nk->nik", gradG, self.JW)

        return self._chunked(f, X, (3, 3))

    def jac_curlA(self, X):
        """∂_i (curl A_J)_k as [n, 3 (i), 3 (k)]  (``computeVecpotDCurlTorus``)."""

        def f(X):
            d = self._diff(X)  # x - y
            r = _np.linalg.norm(d, axis=-1)
            r3 = r**3
            r5 = r**5
            # ∂_i ∇_x G = 3 d d_i / r⁵ - e_i / r³
            out = _np.empty((X.shape[0], 3, 3))
            for i in range(3):
                g = 3.0 * d * (d[..., i] / r5)[..., None]
                g[..., i] -= 1.0 / r3
                out[:, i, :] = _np.einsum("nNk->nk", _np.cross(g, self.JW[None, :, :]))
            return _INV4PI * out

        return self._chunked(f, X, (3, 3))
