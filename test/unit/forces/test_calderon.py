"""Calderón identity tests on the assembled Galerkin matrices.

The Laplace boundary integral operators form the Calderón projector

    P = [[ 1/2 I - K,     V    ],
         [     -W    , 1/2 I + K']]

with P^2 = P.  Expanding the blocks gives the operator identities
V = V', W = W' and <K phi, psi> = <phi, K' psi>.  For Galerkin
discretisation these hold for the assembled matrices up to the
x <-> y asymmetry of the singular quadrature: bempp-cl's
Sauter-Schwab rules are not mirror symmetric under the test/trial
swap (same pair lists and weights, different points), so matrices
inherited from them are asymmetric at the singular-rule error level
(~5e-7 relative at order 4, ~4e-9 at order 6; bempp's own
assemblers show the same asymmetry).  The identities therefore
validate, on every backend including CUDA, at that error level
(~5e-7 for SL forms, up to ~3e-4 for gradient-kernel forms at
singular order 4; all levels shrink with the singular order):

* kernel sign conventions (DL = -z/r^3 vs ADL = +z/r^3),
* the x/y placement of the surface-normal operators,
* the consistency of the singular and regular quadrature,
* the cross-product contraction of vector kernels with RWG bases.

The composition identities (V W = (K - 1/2 I)(K + 1/2 I) and
K V = V K') are deliberately not tested: composing Galerkin matrices
requires L2 transfers between the Dirichlet and Neumann spaces, and
the identities then hold only up to the discretisation error.
"""

import numpy as np
import pytest

import bempp_cl.api as bempp
from bempp_cl.api.forces import arg, term, Form, kernels, make_engine, available_backends

bempp.DEFAULT_DEVICE_INTERFACE = "numba"

SING = bempp.GLOBAL_PARAMETERS.quadrature.singular
REG = bempp.GLOBAL_PARAMETERS.quadrature.regular

BACKENDS = [b for b in ("numpy", "jax", "cuda") if b in available_backends()]


@pytest.fixture(scope="module")
def grid():
    return bempp.shapes.regular_sphere(2)


@pytest.fixture(scope="module")
def spaces(grid):
    return {
        "P0": bempp.function_space(grid, "DP", 0),
        "P1": bempp.function_space(grid, "P", 1),
        "RWG": bempp.function_space(grid, "RWG", 0),
        "SNC": bempp.function_space(grid, "SNC", 0),
    }


@pytest.fixture(scope="module", params=BACKENDS)
def engine(request):
    if request.param == "cuda":
        return make_engine("cuda")
    return make_engine(request.param, pair_chunk=2048)


def _assemble(engine, form):
    return np.asarray(engine.assemble(form, singular_order=SING, regular_order=REG))


def _rel(a, b):
    return np.abs(a - b).max() / np.abs(b).max()


def _skip_if_no_complex(engine):
    if type(engine).__name__ == "CudaEngine":
        pytest.skip("complex kernels are not implemented in the CUDA backend")


def test_single_layer_is_symmetric(engine, spaces):
    """V = V': scalar kernel, scalar (P0) and vector (RWG) bases."""
    P0, RWG = spaces["P0"], spaces["RWG"]
    for space in (P0, RWG):
        M = _assemble(engine, Form([term("SL", arg(space), arg(space))]))
        assert _rel(M, M.T) < 1e-5
        assert np.abs(M).max() > 0


def test_hypersingular_is_symmetric(engine, spaces):
    """W = W': hypersingular form written with n x grad on both sides."""
    P1 = spaces["P1"]
    M = _assemble(engine, Form([term("SL", arg(P1, "nxgrad"), arg(P1, "nxgrad"))]))
    assert _rel(M, M.T) < 1e-5


def test_divergence_form_is_symmetric(engine, spaces):
    """The magnetostatic N operator (div-div) is symmetric."""
    RWG = spaces["RWG"]
    M = _assemble(engine, Form([term("SL", arg(RWG, "div"), arg(RWG, "div"))]))
    assert _rel(M, M.T) < 1e-5


def test_adjoint_double_layer_is_double_layer_transpose(engine, spaces):
    """K' = adj(K): <eta, K phi> = <K' eta, phi> for Galerkin

    Same space (P1) and the classical P0/P1 layout of the Laplace
    double layer; the trial-side normal of DL meets the test-side
    normal of ADL under the transpose.
    """
    P0, P1 = spaces["P0"], spaces["P1"]
    K = _assemble(engine, Form([term("DL", arg(P1), arg(P1, "n"))]))
    Kt = _assemble(engine, Form([term("ADL", arg(P1, "n"), arg(P1))]))
    # ~6e-5 at singular order 4 (bempp's own K/K' pair shows the same), ~1e-6 at 6.
    assert _rel(K, Kt.T) < 1e-3

    K = _assemble(engine, Form([term("DL", arg(P0), arg(P1, "n"))]))
    Kt = _assemble(engine, Form([term("ADL", arg(P1, "n"), arg(P0))]))
    assert K.shape == (P0.global_dof_count, P1.global_dof_count)
    assert _rel(K, Kt.T) < 1e-3


def test_cross_contraction_of_gradient_kernels(engine, spaces):
    """The C operator (ADL, RWG x RWG, cross contraction) is symmetric.

    Under x <-> y the antisymmetry of the gradient kernel and the
    cross product in the vvv contraction contribute one sign flip
    each and cancel.  DL = -ADL pointwise, so its matrix is the
    exact negative.
    """
    RWG = spaces["RWG"]
    M = _assemble(engine, Form([term("ADL", arg(RWG), arg(RWG))]))
    # ~3e-4 at singular order 4, ~6e-6 at 6: gradient kernels amplify the
    # singular-rule orientation asymmetry.
    assert _rel(M, M.T) < 1e-3
    M2 = _assemble(engine, Form([term("DL", arg(RWG), arg(RWG))]))
    assert _rel(M2, M2.T) < 1e-3
    # DL = -ADL pointwise on identical quadrature: exact to rounding.
    assert np.abs(M + M2).max() < 1e-12 * np.abs(M).max()


def test_efie_is_symmetric(engine, spaces):
    """Maxwell EFIE (the analogue of V = V' for the electric field
    operator): symmetric, not Hermitian, for real wavenumbers."""
    _skip_if_no_complex(engine)
    RWG = spaces["RWG"]
    k = 1.3
    hsl = kernels.get("HSL")(k, 0.0)
    form = Form(
        [
            -1j * k * term(hsl, arg(RWG), arg(RWG)),
            -(1.0 / (1j * k)) * term(hsl, arg(RWG, "div"), arg(RWG, "div")),
        ]
    )
    M = _assemble(engine, form)
    assert _rel(M, M.T) < 1e-5
