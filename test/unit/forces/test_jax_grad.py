"""Differentiability of the jax backend with respect to vertex coordinates (basis of the AD/energy mode)."""

import numpy as np
import pytest

import bempp_cl.api as bempp
from bempp_cl.api.forces import arg, term, Form, make_engine, available_backends

bempp.DEFAULT_DEVICE_INTERFACE = "numba"

pytestmark = pytest.mark.skipif("jax" not in available_backends(), reason="jax not installed")


def test_vertex_gradient_matches_finite_differences():
    import jax
    import jax.numpy as jnp

    grid = bempp.shapes.regular_sphere(1)
    P0 = bempp.function_space(grid, "DP", 0)
    P1 = bempp.function_space(grid, "P", 1)
    eng = make_engine("jax", pair_chunk=512)
    rng = np.random.default_rng(3)
    u = rng.standard_normal(P0.global_dof_count)
    v = rng.standard_normal(P1.global_dof_count)
    form = Form([term("SL", arg(P0, coeffs=u), arg(P0, coeffs=u)), term("DL", arg(P0, coeffs=u), arg(P1, "n", v))])

    def f(vertices):
        return eng.evaluate(form, vertices=vertices, singular_order=3, regular_order=3)[0]

    V0 = jnp.asarray(grid.vertices)
    val = float(f(V0))
    grad = np.asarray(jax.grad(f)(V0))
    assert grad.shape == grid.vertices.shape
    # central finite differences on a few random vertex coordinates
    h = 1e-5
    for (i, j) in [(0, 3), (2, 7), (1, 11)]:
        E = np.zeros_like(grad)
        E[i, j] = h
        fd = (float(f(V0 + E)) - float(f(V0 - E))) / (2 * h)
        assert abs(fd - grad[i, j]) < 1e-6 * max(1.0, abs(val)), (i, j, fd, grad[i, j])
