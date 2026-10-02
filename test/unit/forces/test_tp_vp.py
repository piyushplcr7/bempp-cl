"""End-to-end tests of the TP-VP problem (solver, MST references, BEM shape derivative)."""

import numpy as np
import pytest

import bempp_cl.api as bempp
from bempp_cl.api.forces import make_engine, cos_family, translations, rotations, available_backends
from bempp_cl.api.forces.problems import tp_vp
from bempp_cl.api.forces.problems.sources import UniformField, TorusCurrent

bempp.DEFAULT_DEVICE_INTERFACE = "numba"
MU_I, MU_E = 100.0, 1.0


@pytest.fixture(scope="module")
def engine():
    name = "cuda" if "cuda" in available_backends() else "jax" if "jax" in available_backends() else "numpy"
    return make_engine(name)


def _sphere_uniform(engine, level):
    grid = bempp.shapes.regular_sphere(level)
    src = UniformField([0.0, 0.0, 1.0])
    tr = tp_vp.solve(grid, src, MU_I, MU_E, engine=engine)
    q = tr.quad
    n = np.broadcast_to(q.normals[:, None, :], q.points.shape)
    B0 = src.B0
    Bn_ex = 3 * MU_I / (MU_I + 2 * MU_E) * np.einsum("k,eqk->eq", B0, n)
    B0t = B0 - np.einsum("k,eqk->eq", B0, n)[..., None] * n
    Ht_ex = 3 / (MU_I + 2 * MU_E) * np.linalg.norm(B0t, axis=-1)
    l2 = lambda a: np.sqrt(np.sum(q.weights * a**2))
    eBn = l2(tr.Bn() - Bn_ex) / l2(Bn_ex)
    eHt = l2(np.linalg.norm(tr.Ht_vec(MU_E), axis=-1) - Ht_ex) / l2(Ht_ex)
    return tr, src, eBn, eHt


def test_sphere_uniform_field_converges(engine):
    _, _, e2, h2 = _sphere_uniform(engine, 2)
    _, _, e3, h3 = _sphere_uniform(engine, 3)
    assert e2 < 0.06 and h2 < 0.04
    assert e3 < 0.6 * e2 and h3 < 0.5 * h2


def test_sphere_uniform_field_zero_force(engine):
    tr, src, _, _ = _sphere_uniform(engine, 2)
    assert np.abs(tp_vp.mst_force(tr, MU_I, MU_E)).max() < 1e-5
    sd = tp_vp.shape_derivative(tr, src, MU_I, MU_E, translations(), engine=engine)
    assert np.abs(sd).max() < 1e-12


def test_torus_bem_matches_mst(engine):
    """BEM and MST shape derivatives agree to discretisation accuracy and the gap shrinks with h."""
    src = TorusCurrent(R=2.0, r=0.5)
    fam = cos_family(3)
    rel = []
    for level in (2, 3):
        grid = bempp.shapes.regular_sphere(level)
        grid = bempp.Grid(grid.vertices + np.array([[5.0], [5.0], [3.0]]), grid.elements)
        tr = tp_vp.solve(grid, src, MU_I, MU_E, engine=engine)
        sd_mst = tp_vp.mst_shape_derivative(tr, MU_I, MU_E, fam)
        sd_bem = tp_vp.shape_derivative(tr, src, MU_I, MU_E, fam, engine=engine)
        rel.append(np.linalg.norm(sd_bem - sd_mst) / np.linalg.norm(sd_mst))
    assert rel[0] < 0.03
    assert rel[1] < 0.4 * rel[0]


def test_torus_torque_consistency(engine):
    """Rotational velocity fields: BEM shape derivative vs MST torque about the centre."""
    src = TorusCurrent(R=2.0, r=0.5)
    center = (5.0, 5.0, 3.0)
    grid = bempp.shapes.regular_sphere(3)
    grid = bempp.Grid(grid.vertices + np.array([[5.0], [5.0], [3.0]]), grid.elements)
    tr = tp_vp.solve(grid, src, MU_I, MU_E, engine=engine)
    torque_mst = tp_vp.mst_torque(tr, MU_I, MU_E, center)
    torque_bem = tp_vp.shape_derivative(tr, src, MU_I, MU_E, rotations(center), engine=engine)
    force_mst = tp_vp.mst_force(tr, MU_I, MU_E)
    force_bem = tp_vp.shape_derivative(tr, src, MU_I, MU_E, translations(), engine=engine)
    scale = np.abs(force_mst).max()
    assert np.abs(force_bem - force_mst).max() < 0.05 * scale
    assert np.abs(torque_bem - torque_mst).max() < 0.05 * scale
