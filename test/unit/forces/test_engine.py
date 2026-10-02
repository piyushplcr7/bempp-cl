"""Tests of the forces engine against bempp-cl's own assemblers."""

import numpy as np
import pytest

import bempp_cl.api as bempp
from bempp_cl.api.forces import arg, term, Form, DV, make_engine, available_backends
from bempp_cl.api.forces import kernels, cos_family, rotations
from bempp_cl.api.forces.velocity import VelocityFamily

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


def _dense(op):
    return op.weak_form().A


def _assemble(engine, form):
    return np.asarray(engine.assemble(form, singular_order=SING, regular_order=REG))


def _rel(a, b):
    return np.abs(a - b).max() / np.abs(b).max()


def test_laplace_single_layer_p0(engine, spaces):
    P0 = spaces["P0"]
    ref = _dense(bempp.operators.boundary.laplace.single_layer(P0, P0, P0))
    M = _assemble(engine, Form([term("SL", arg(P0), arg(P0))]))
    assert _rel(M, ref) < 1e-12


def test_laplace_single_layer_p1(engine, spaces):
    P1 = spaces["P1"]
    ref = _dense(bempp.operators.boundary.laplace.single_layer(P1, P1, P1))
    M = _assemble(engine, Form([term("SL", arg(P1), arg(P1))]))
    assert _rel(M, ref) < 1e-12


def test_laplace_double_layers(engine, spaces):
    P0, P1 = spaces["P0"], spaces["P1"]
    ref = _dense(bempp.operators.boundary.laplace.double_layer(P1, P1, P0))
    M = _assemble(engine, Form([term("DL", arg(P0), arg(P1, "n"))]))
    assert _rel(M, ref) < 1e-12
    ref = _dense(bempp.operators.boundary.laplace.adjoint_double_layer(P0, P0, P1))
    M = _assemble(engine, Form([term("ADL", arg(P1, "n"), arg(P0))]))
    assert _rel(M, ref) < 1e-12


def test_laplace_hypersingular(engine, spaces):
    P1 = spaces["P1"]
    ref = _dense(bempp.operators.boundary.laplace.hypersingular(P1, P1, P1))
    M = _assemble(engine, Form([term("SL", arg(P1, "nxgrad"), arg(P1, "nxgrad"))]))
    assert _rel(M, ref) < 1e-12


def _skip_if_no_complex(engine):
    if type(engine).__name__ == "CudaEngine":
        pytest.skip("complex kernels are not implemented in the CUDA backend")


def test_maxwell_efie_rwg(engine, spaces):
    """Validates the RWG basis (Piola, edge lengths, signs) and the div operator."""
    _skip_if_no_complex(engine)
    RWG, SNC = spaces["RWG"], spaces["SNC"]
    k = 1.3
    ref = _dense(bempp.operators.boundary.maxwell.electric_field(RWG, RWG, SNC, k))
    hsl = kernels.get("HSL")(k, 0.0)
    form = Form([-1j * k * term(hsl, arg(RWG), arg(RWG)), -(1.0 / (1j * k)) * term(hsl, arg(RWG, "div"), arg(RWG, "div"))])
    M = _assemble(engine, form)
    assert _rel(M, ref) < 1e-12


def test_maxwell_mfie_rwg(engine, spaces):
    """Validates the vector-kernel cross contraction with vector bases."""
    _skip_if_no_complex(engine)
    RWG, SNC = spaces["RWG"], spaces["SNC"]
    k = 1.3
    ref = _dense(bempp.operators.boundary.maxwell.magnetic_field(RWG, RWG, SNC, k))
    hdl = kernels.get("HDL")(k, 0.0)
    M = _assemble(engine, Form([-1.0 * term(hdl, arg(RWG), arg(RWG))]))
    assert _rel(M, ref) < 1e-12


@pytest.mark.skipif("cuda" not in BACKENDS, reason="no CUDA device")
def test_cuda_matches_numpy_on_rwg_laplace_forms(spaces):
    """The magnetostatic operators (A, N, C-type) on CUDA vs the validated numpy engine."""
    RWG, P1 = spaces["RWG"], spaces["P1"]
    cuda = make_engine("cuda")
    ref = make_engine("numpy", pair_chunk=2048)
    forms = {
        "A": Form([term("SL", arg(RWG), arg(RWG))]),
        "N": Form([term("SL", arg(RWG, "div"), arg(RWG, "div"))]),
        "C": Form([term("ADL", arg(RWG), arg(RWG))]),
        "C_nx": Form([term("ADL", arg(RWG, "id.nx"), arg(RWG))]),
        "ortho": Form([term("SL", arg(RWG, "id.nx"), arg(P1, "grad"))]),
    }
    for name, form in forms.items():
        a = _assemble(cuda, form)
        b = _assemble(ref, form)
        assert _rel(a, b) < 1e-12, name


def test_snc_is_n_cross_rwg(engine, spaces):
    """SNC test functions equal n x RWG: compare (SL, SNC, RWG) with (SL, nx RWG, RWG)."""
    RWG, SNC = spaces["RWG"], spaces["SNC"]
    A = _assemble(engine, Form([term("SL", arg(SNC), arg(RWG))]))
    B = _assemble(engine, Form([term("SL", arg(RWG, "id.nx"), arg(RWG))]))
    assert _rel(A, B) < 1e-13
    assert np.abs(A).max() > 0


def test_evaluate_matches_assemble(engine, spaces):
    P1, RWG = spaces["P1"], spaces["RWG"]
    rng = np.random.default_rng(0)
    u = rng.standard_normal(P1.global_dof_count)
    v = rng.standard_normal(P1.global_dof_count)
    a = rng.standard_normal(RWG.global_dof_count)
    b = rng.standard_normal(RWG.global_dof_count)
    form = Form([term("SL", arg(P1, "nxgrad"), arg(P1, "nxgrad")), 2.5 * term("ADL", arg(RWG), arg(RWG))])
    W = _assemble(engine, Form([form[0]]))
    C = _assemble(engine, Form([form[1]]))
    expected = u @ W @ v + a @ C @ b
    form_c = Form([term("SL", arg(P1, "nxgrad", u), arg(P1, "nxgrad", v)), 2.5 * term("ADL", arg(RWG, coeffs=a), arg(RWG, coeffs=b))])
    val = np.asarray(engine.evaluate(form_c, singular_order=SING, regular_order=REG))
    assert val.shape == (1,)
    assert abs(val[0] - expected) < 1e-12 * abs(expected)


def test_velocity_fields_batched(engine, spaces):
    """Batched evaluation over velocity fields equals per-field assembly + contraction."""
    RWG = spaces["RWG"]
    rng = np.random.default_rng(1)
    a = rng.standard_normal(RWG.global_dof_count)
    b = rng.standard_normal(RWG.global_dof_count)
    fam = VelocityFamily("cos", [(1, 0, 2, 0), (0, 1, 1, 2), (2, 2, 0, 1)])
    form = Form([term("A1", arg(RWG, coeffs=a), arg(RWG, coeffs=b)), term("C3", arg(RWG, coeffs=a), arg(RWG, coeffs=b)), term("SL", arg(RWG, coeffs=a), DV(arg(RWG, coeffs=b)))])
    val = np.asarray(engine.evaluate(form, velocity=fam, singular_order=SING, regular_order=REG))
    assert val.shape == (3,)
    for f in range(3):
        sub = VelocityFamily("cos", fam.params[f : f + 1])
        mats = np.asarray(
            engine.assemble(
                Form([term("A1", arg(RWG), arg(RWG)), term("C3", arg(RWG), arg(RWG)), term("SL", arg(RWG), DV(arg(RWG)))]),
                velocity=sub,
                singular_order=SING,
                regular_order=REG,
            )
        )
        expected = a @ mats[0] @ b
        assert abs(val[f] - expected) < 1e-11 * max(1.0, abs(expected))


def test_velocity_jacobian_finite_difference():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((5, 3))
    for fam in (cos_family(2), rotations((0.3, -0.2, 0.1))):
        DVa = fam.DV(np, x)
        h = 1e-6
        for j in range(3):
            e = np.zeros(3)
            e[j] = h
            fd = (fam.V(np, x + e) - fam.V(np, x - e)) / (2 * h)
            assert np.abs(fd - DVa[..., j]).max() < 1e-7


def test_translation_velocity_gives_force_like_kernel_zero(engine, spaces):
    """For constant V, A1 = z·(V(x)-V(y))/r³ vanishes identically."""
    from bempp_cl.api.forces import translations

    P0 = spaces["P0"]
    c = np.ones(P0.global_dof_count)
    val = np.asarray(engine.evaluate(Form([term("A1", arg(P0, coeffs=c), arg(P0, coeffs=c))]), velocity=translations(), singular_order=SING, regular_order=REG))
    assert np.abs(val).max() < 1e-14
