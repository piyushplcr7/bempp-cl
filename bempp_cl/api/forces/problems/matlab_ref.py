"""Load gypsilab/MATLAB reference exports (``Export/export_tpvp.m``) and map them to bempp-cl.

Dof conventions (see ``problems/tp_vp.py`` docstring and gypsilab's
``femRaoWiltonGlisson.m`` / ``femNedelec.m``):

* gypsilab RWG on element e, local k (1-based in MATLAB): flux_k (x - v_k) / (2 |e|),
  flux_k = +1 if elt(e, k+1) < elt(e, k+2) (cyclic, global vertex ids); dof = edge opposite v_k.
* bempp RWG local j: mult_j · len_j / (2 |e|) · (x - v_{2-j}); so gypsilab local k ↔ bempp local 2-k
  and c_bempp[D] = c_gyp[d] · flux_gyp / (mult · len), a per-edge constant.
* gypsilab NED_k = RWG_k × n = -SNC_k; a NED coefficient vector maps to minus the SNC vector.
"""

import numpy as _np

import bempp_cl.api as _bempp
from bempp_cl.api.grid.grid import _EDGE_LOCAL


class MatlabReference:
    """A loaded export with the grid rebuilt in bempp-cl and dof maps."""

    def __init__(self, path):
        from scipy.io import loadmat

        d = loadmat(path, squeeze_me=True)
        self.raw = d
        vtx = _np.asarray(d["vtx"], dtype="float64")  # (nv, 3)
        elt = _np.asarray(d["elt"], dtype="int64") - 1  # (ne, 3), 0-based
        self.grid = _bempp.Grid(_np.ascontiguousarray(vtx.T), _np.ascontiguousarray(elt.T.astype(_np.uint32)))
        # normals must agree (bempp: cross(v1-v0, v2-v0)); gypsilab: cross(tgt1, tgt2)
        nrm = _np.asarray(d["nrm"], dtype="float64")
        dots = _np.sum(nrm * self.grid.normals, axis=1)
        if not _np.all(dots > 0.99):
            raise ValueError("gypsilab and bempp normals disagree; element orientation differs.")
        self.elt = elt
        self.P1 = _bempp.function_space(self.grid, "P", 1)
        self.RWG = _bempp.function_space(self.grid, "RWG", 0)
        self.SNC = _bempp.function_space(self.grid, "SNC", 0)
        self.elt2dof_rwg = _np.asarray(d["elt2dof_RWG"], dtype="int64") - 1  # (ne, 3) gypsilab edge dofs
        self._build_rwg_map()
        self.mu = float(d["mu"])
        self.mu0 = float(d["mu0"])
        self.abc_alpha = _np.asarray(d["abc_alpha"], dtype="int64")

    def _build_rwg_map(self):
        """Per gypsilab RWG dof: bempp dof index and coefficient factor c_bempp = factor · c_gyp."""
        elt = self.elt
        ne = elt.shape[0]
        ngyp = int(self.elt2dof_rwg.max()) + 1
        l2g = self.RWG.local2global
        mult = self.RWG.local_multipliers
        verts = self.grid.vertices.T
        to_bempp = -_np.ones(ngyp, dtype="int64")
        factor = _np.zeros(ngyp)
        for e in range(ne):
            for k in range(3):
                kp1, kp2 = (k + 1) % 3, (k + 2) % 3
                flux = 1.0 if elt[e, kp1] < elt[e, kp2] else -1.0
                j = 2 - k
                D = l2g[e, j]
                m = mult[e, j]
                if m == 0:
                    continue
                length = _np.linalg.norm(verts[elt[e, kp1]] - verts[elt[e, kp2]])
                d = self.elt2dof_rwg[e, k]
                f = flux / (m * length)
                if to_bempp[d] >= 0:
                    assert to_bempp[d] == D and abs(factor[d] - f) < 1e-12 * abs(f), "inconsistent dof map"
                to_bempp[d] = D
                factor[d] = f
        assert _np.all(to_bempp >= 0)
        self.rwg_to_bempp = to_bempp
        self.rwg_factor = factor

    # ---- coefficient conversions (gypsilab -> bempp)
    def rwg_coeffs(self, c_gyp):
        out = _np.zeros(self.RWG.global_dof_count)
        out[self.rwg_to_bempp] = _np.asarray(c_gyp) * self.rwg_factor
        return out

    def ned_to_snc_coeffs(self, g_gyp):
        return -self.rwg_coeffs(g_gyp)

    def p1_coeffs(self, c):
        """gypsilab P1 dofs are the vertices in order; bempp P1 dofs likewise (vertex index)."""
        return _np.asarray(c, dtype="float64")

    # ---- matrix conversions: gypsilab (rows/cols in gypsilab dofs) -> bempp dof ordering
    def rwg_matrix(self, M, rows="rwg", cols="rwg"):
        """Permute/scale a gypsilab matrix into bempp dofs. Basis relation: b_gyp = b_bempp · factor."""
        M = _np.asarray(M, dtype="float64")
        if rows == "rwg":
            P = _np.zeros((self.RWG.global_dof_count, M.shape[0]))
            P[self.rwg_to_bempp, _np.arange(M.shape[0])] = self.rwg_factor
            M = P @ M
        if cols == "rwg":
            P = _np.zeros((M.shape[1], self.RWG.global_dof_count))
            P[_np.arange(M.shape[1]), self.rwg_to_bempp] = self.rwg_factor
            M = M @ P
        return M

    @property
    def traces(self):
        """Traces object in bempp conventions built from the MATLAB solution."""
        from .tp_vp import Spaces, Traces
        from ..single import SurfaceQuadrature

        sp = Spaces(self.grid)
        sp.P1, sp.RWG, sp.SNC = self.P1, self.RWG, self.SNC
        quad = SurfaceQuadrature(self.grid, 4)
        tr = Traces(sp, self.p1_coeffs(self.raw["Psi"]), self.ned_to_snc_coeffs(self.raw["g"]), quad)
        # use MATLAB's own RWG projection of the Neumann trace for exact comparability
        tr.tn = self.rwg_coeffs(self.raw["Psi_RWG"])
        return tr
