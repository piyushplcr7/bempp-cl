"""Magnetostatic transmission problem, vector potential formulation (TP-VP).

Port of ``solveTransmissionProblem.m`` / ``SdBemTPVP_dualnorm.m`` from the
thesis code.  A linear magnetic body Ω (permeability μ_i) sits in vacuum
(μ_e) in the field of a source current with vector potential A_J.

Unknowns (exterior traces of the total field, bempp-cl bases):

    psi   Neumann-type trace in nxgrad(P1)          (P1 dof vector)
    g     Dirichlet trace in SNC = n × RWG          (RWG dof vector)

Block system (Lagrange multiplier λ ∈ P1, scalars c1, c2 for mean-zero constraints)::

    [(1+μi/μe) A   -2 C          0      vec  0 ] [psi]   [μe ∫ nxgradφ · T_D A_J]
    [   2 Cᵀ   -(1+μe/μi) N    ortho    0    0 ] [ g ]   [μe ∫ ζ · T_N A_J      ]
    [    0        orthoᵀ        0       0   vec] [ λ ] = [0                     ]
    [  vecᵀ        0            0       0    0 ] [c1 ]   [0                     ]
    [    0         0           vecᵀ     0    0 ] [c2 ]   [0                     ]

with A = SL(nxgrad P1, nxgrad P1), C = ∫∫ nxgradφ_i · (∇_y G × ψ_j),
N = -SL(div ψ, div ψ), ortho = SL(ζ, grad φ), ζ the SNC basis.

Note on bases: gypsilab's NED basis is ψ × n = -SNC, so a MATLAB NED
coefficient vector equals minus the SNC coefficient vector (after the RWG
dof mapping).  The shape-derivative kernels contract the Dirichlet trace
against plain RWG functions; the vector to use is ``td = n × g`` expressed in
RWG coefficients, i.e. ``td = -g_snc``.
"""

import numpy as _np
import scipy.sparse as _sp
import scipy.sparse.linalg as _spla

import bempp_cl.api as _bempp

from ..terms import arg, term, Form, DV
from ..single import SurfaceQuadrature, mass_matrix, load_vector, project, reconstruct, integral_of_basis
from .. import get_engine


class Spaces:
    """Function spaces on a grid used by the TP-VP formulation."""

    def __init__(self, grid):
        self.grid = grid
        self.P1 = _bempp.function_space(grid, "P", 1)
        self.RWG = _bempp.function_space(grid, "RWG", 0)
        self.SNC = _bempp.function_space(grid, "SNC", 0)


class Traces:
    """Solution traces of the TP-VP problem."""

    def __init__(self, spaces, psi, g_snc, quad):
        self.spaces = spaces
        self.psi = psi  # nxgrad(P1) coefficients (P1 dofs)
        self.g_snc = g_snc  # SNC coefficients of the Dirichlet trace g
        self.quad = quad
        # Neumann trace as an RWG function (L² projection of Σ psi_i nxgrad φ_i), as in MATLAB
        psi_vals = reconstruct(psi, arg(spaces.P1, "nxgrad"), quad)
        self.tn = project(psi_vals, spaces.RWG, quad)
        # RWG coefficients of n × g (see module docstring)
        self.td = -g_snc

    def Bn(self, quad=None):
        """B·n = curl_Γ g = div_Γ (n × g)... evaluated at quadrature points [ne, nq]."""
        q = quad or self.quad
        return reconstruct(self.g_snc, arg(self.spaces.RWG, "div"), q)

    def Ht_vec(self, mu_e, quad=None):
        """Tangential H = (n × psi)/μ_e at quadrature points [ne, nq, 3]."""
        q = quad or self.quad
        psi_vals = reconstruct(self.psi, arg(self.spaces.P1, "nxgrad"), q)
        return _np.cross(q.normals[:, None, :], psi_vals) / mu_e

    def g_vec(self, quad=None):
        """The Dirichlet trace g at quadrature points [ne, nq, 3]."""
        q = quad or self.quad
        return reconstruct(self.g_snc, arg(self.spaces.SNC), q)

    def tn_vec(self, quad=None):
        q = quad or self.quad
        return reconstruct(self.tn, arg(self.spaces.RWG), q)


def operators(spaces, engine=None, singular_order=4, regular_order=4):
    """Assemble A, C, N, ortho (dense numpy) with the pairwise engine."""
    eng = engine or get_engine()
    P1, RWG, SNC = spaces.P1, spaces.RWG, spaces.SNC
    kw = dict(singular_order=singular_order, regular_order=regular_order)
    A = _np.asarray(eng.assemble(Form([term("SL", arg(P1, "nxgrad"), arg(P1, "nxgrad"))]), **kw))
    C = _np.asarray(eng.assemble(Form([term("DL", arg(P1, "nxgrad"), arg(RWG))]), **kw))
    N = _np.asarray(eng.assemble(Form([-1.0 * term("SL", arg(RWG, "div"), arg(RWG, "div"))]), **kw))
    ortho = _np.asarray(eng.assemble(Form([term("SL", arg(SNC), arg(P1, "grad"))]), **kw))
    return A, C, N, ortho


def solve(grid, source, mu_i, mu_e, engine=None, quad_order=4, singular_order=4, regular_order=4, project_traces=True):
    """Solve the TP-VP problem; returns a Traces object."""
    spaces = Spaces(grid)
    quad = SurfaceQuadrature(grid, quad_order)
    A, C, N, ortho = operators(spaces, engine, singular_order, regular_order)
    P1, RWG, SNC = spaces.P1, spaces.RWG, spaces.SNC
    n1, nn = P1.global_dof_count, RWG.global_dof_count
    vec = integral_of_basis(P1, quad)

    # source traces at quadrature points
    X = quad.flat_points
    nrm = quad.flat_normals
    AJ = source.A(X)
    curlAJ = source.curlA(X)
    TDAJ = AJ - _np.sum(AJ * nrm, axis=1, keepdims=True) * nrm  # tangential trace
    TNAJ = _np.cross(curlAJ, nrm)  # curl A × n
    shape = (grid.number_of_elements, quad.nq, 3)
    if project_traces:
        td_coeffs = project(TDAJ.reshape(shape), SNC, quad)
        tn_coeffs = project(TNAJ.reshape(shape), RWG, quad)
        M_nxg_snc = mass_matrix(arg(P1, "nxgrad"), arg(SNC), quad)
        M_rwg_snc = mass_matrix(arg(RWG), arg(SNC), quad)
        rhs1 = mu_e * (M_nxg_snc @ td_coeffs)
        rhs2 = mu_e * (M_rwg_snc.T @ tn_coeffs)
    else:
        rhs1 = mu_e * load_vector(arg(P1, "nxgrad"), TDAJ.reshape(shape), quad)
        rhs2 = mu_e * load_vector(arg(SNC), TNAJ.reshape(shape), quad)

    Z11 = _np.zeros((n1, n1))
    z1 = _np.zeros((n1, 1))
    zn = _np.zeros((nn, 1))
    v = vec[:, None]
    block = _np.block(
        [
            [(1 + mu_i / mu_e) * A, -2 * C, Z11, v, z1],
            [2 * C.T, -(1 + mu_e / mu_i) * N, ortho, zn, zn],
            [Z11, ortho.T, Z11, z1, v],
            [v.T, zn.T, z1.T, _np.zeros((1, 1)), _np.zeros((1, 1))],
            [z1.T, zn.T, v.T, _np.zeros((1, 1)), _np.zeros((1, 1))],
        ]
    )
    rhs = _np.concatenate([rhs1, rhs2, _np.zeros(n1), [0.0, 0.0]])
    sol = _np.linalg.solve(block, rhs)
    psi = sol[:n1]
    g = sol[n1 : n1 + nn]
    tr = Traces(spaces, psi, g, quad)
    tr.lam = sol[n1 + nn : 2 * n1 + nn]
    tr.operators = (A, C, N, ortho)
    return tr


# ----------------------------------------------------------------------------- references
def mst_density(traces, mu_i, mu_e, quad=None):
    """Scalar MST density ½ (Bn² (1/μe − 1/μi) − |Ht|² (μe − μi)) at quadrature points [ne, nq]."""
    q = quad or traces.quad
    Bn = traces.Bn(q)
    Ht = _np.linalg.norm(traces.Ht_vec(mu_e, q), axis=-1)
    return 0.5 * (Bn**2 * (1.0 / mu_e - 1.0 / mu_i) - Ht**2 * (mu_e - mu_i))


def mst_shape_derivative(traces, mu_i, mu_e, velocity, quad=None):
    """∫ density · (V·n) for every field of a velocity family: [nfields]."""
    q = quad or traces.quad
    dens = mst_density(traces, mu_i, mu_e, q)
    V = velocity.V(_np, q.points)  # [ne, nq, F, 3]
    Vn = _np.einsum("eqfk,ek->eqf", V, q.normals)
    return _np.einsum("eq,eqf,eq->f", dens, Vn, q.weights)


def mst_force(traces, mu_i, mu_e, quad=None):
    q = quad or traces.quad
    dens = mst_density(traces, mu_i, mu_e, q)
    return _np.einsum("eq,ek,eq->k", dens, q.normals, q.weights)


def mst_torque(traces, mu_i, mu_e, center, quad=None):
    q = quad or traces.quad
    dens = mst_density(traces, mu_i, mu_e, q)
    rvec = q.points - _np.asarray(center)
    return _np.einsum("eq,eqk,eq->k", dens, _np.cross(rvec, q.normals[:, None, :]), q.weights)


# ----------------------------------------------------------------------------- shape derivative
def bem_form(traces, mu_i, mu_e):
    """The pairwise part of the BEM shape derivative (before the 1/(2 μe) factor).

    Mirrors ``SdBemTPVP_dualnorm_GPU.cu``:
        (1+μi/μe)(TnA A1 TnA + 2 TnA A2 TnA) − 4(TnA C1 TdA + TdA C1 TnA + TnA C3 TdA) − (1+μe/μi) TdA N TdA
    """
    RWG = traces.spaces.RWG
    u = arg(RWG, coeffs=traces.tn)
    v = arg(RWG, coeffs=traces.td)
    ud = arg(RWG, "div", traces.td)
    r = mu_i / mu_e
    return Form(
        [
            (1 + r) * term("A1", u, u),
            2 * (1 + r) * term("SL", u, DV(u)),
            -4.0 * term("ADL", u, DV(v)),
            -4.0 * term("ADL", v, DV(u)),
            -4.0 * term("C3", u, v),
            -(1 + 1 / r) * term("A1", ud, ud),
        ]
    )


def source_terms(traces, source, velocity, quad=None):
    """Host-side terms l1, l21, l22 of ``SdBemTPVP_dualnorm.m``: returns (l1, l21, l22) each [nfields]."""
    q = quad or traces.quad
    X = q.flat_points
    W = q.flat_weights
    nrm = q.flat_normals
    Psi = traces.tn_vec(q).reshape(-1, 3)
    g = traces.g_vec(q).reshape(-1, 3)
    nxg = _np.cross(nrm, g)
    AJ = source.A(X)
    curlAJ = source.curlA(X)
    jac = source.jacA(X)  # [n, i, k] = ∂_i A_k
    V = velocity.V(_np, X)  # [n, F, 3]
    DVm = velocity.DV(_np, X)  # [n, F, 3, 3], DV[i, j] = ∂V_i/∂x_j
    F = velocity.nfields
    # T3 = ∫ Σ_i V_i (∂_i A · Psi)
    t3 = _np.einsum("n,nfi,nik,nk->f", W, V, jac, Psi)
    # T4 = ∫ Σ_i (DV_i · Psi) A_i
    t4 = _np.einsum("n,nfik,nk,ni->f", W, DVm, Psi, AJ)
    l1 = t3 + t4
    # l21 = ∫ (DV (n×g)) · curl A_J
    l21 = _np.einsum("n,nfik,nk,ni->f", W, DVm, nxg, curlAJ)
    # l22 = ∫ (n×g) · (V·∇) curl A_J = ∫ Σ_{i,k} (n×g)_k V_i ∂_i (curl A_J)_k
    jc = source.jac_curlA(X)  # [n, i, k]
    l22 = _np.einsum("n,nk,nfi,nik->f", W, nxg, V, jc)
    return l1, l21, l22


def shape_derivative(traces, source, mu_i, mu_e, velocity, engine=None, singular_order=4, regular_order=4, quad=None):
    """BEM shape derivative for every field: 1/(2 μe) · bem_form − l1 + l21 + l22."""
    eng = engine or get_engine()
    bem = _np.asarray(eng.evaluate(bem_form(traces, mu_i, mu_e), velocity=velocity, singular_order=singular_order, regular_order=regular_order))
    l1, l21, l22 = source_terms(traces, source, velocity, quad)
    return bem / (2 * mu_e) - l1 + l21 + l22
