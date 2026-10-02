"""Comparison against gypsilab/MATLAB exports produced by ``Export/export_tpvp.m``.

Set ``FORCES_MATLAB_REF`` to a directory (or a .mat file); tests are skipped otherwise.
"""

import glob
import os

import numpy as np
import pytest

import bempp_cl.api as bempp
from bempp_cl.api.forces import make_engine, available_backends, arg, term, Form
from bempp_cl.api.forces.problems import tp_vp
from bempp_cl.api.forces.problems.matlab_ref import MatlabReference
from bempp_cl.api.forces.problems.sources import TorusCurrent
from bempp_cl.api.forces.velocity import VelocityFamily

bempp.DEFAULT_DEVICE_INTERFACE = "numba"

_REF = os.environ.get("FORCES_MATLAB_REF", "")
if os.path.isdir(_REF):
    FILES = sorted(glob.glob(os.path.join(_REF, "tpvp_*.mat")))
elif os.path.isfile(_REF):
    FILES = [_REF]
else:
    FILES = []

pytestmark = pytest.mark.skipif(not FILES, reason="no MATLAB reference exports (set FORCES_MATLAB_REF)")


@pytest.fixture(scope="module")
def engine():
    name = "cuda" if "cuda" in available_backends() else "numpy"
    return make_engine(name)


@pytest.fixture(scope="module", params=FILES, ids=[os.path.basename(f) for f in FILES])
def ref(request):
    return MatlabReference(request.param)


def _rel(a, b):
    return np.abs(a - b).max() / np.abs(b).max()


def test_operators_match(ref, engine):
    """A, C, N, ortho from getVPTPBIOs (Sauter-Schwab order 5, 625-pt far rule) vs the engine."""
    sp = tp_vp.Spaces(ref.grid)
    A, C, N, ortho = tp_vp.operators(sp, engine, singular_order=5, regular_order=8)
    # A: nxgrad P1 x nxgrad P1 (vertex dofs, same numbering)
    assert _rel(A, ref.raw["Amat"]) < 1e-6
    # C: nxgrad P1 x RWG ; N: RWG x RWG ; ortho: NED x grad P1 = -SNC x grad P1
    assert _rel(C, ref.rwg_matrix(ref.raw["Cmat"], rows=None, cols="rwg")) < 1e-6
    assert _rel(N, ref.rwg_matrix(ref.raw["Nmat"])) < 1e-6
    assert _rel(ortho, -ref.rwg_matrix(ref.raw["ortho"], rows="rwg", cols=None)) < 1e-6


def test_solution_matches(ref, engine):
    src = TorusCurrent(R=float(ref.raw["R0"]), r=float(ref.raw["r0"]))
    tr = tp_vp.solve(ref.grid, src, ref.mu, ref.mu0, engine=engine, quad_order=2, singular_order=5, regular_order=8)
    mref = ref.traces
    assert _rel(tr.psi, mref.psi) < 1e-4
    assert _rel(tr.g_snc, mref.g_snc) < 1e-4


def test_mst_shape_derivatives_match(ref):
    fam = VelocityFamily("cos", ref.abc_alpha)
    mref = ref.traces
    sd = tp_vp.mst_shape_derivative(mref, ref.mu, ref.mu0, fam)
    assert _rel(sd, ref.raw["shape_derivatives_mst"]) < 1e-6


def test_bem_shape_derivatives_match(ref, engine):
    """The thesis GPU kernel (order-5 rules) vs the new engine on the same traces."""
    fam = VelocityFamily("cos", ref.abc_alpha)
    mref = ref.traces
    src = TorusCurrent(R=float(ref.raw["R0"]), r=float(ref.raw["r0"]))
    sd = tp_vp.shape_derivative(mref, src, ref.mu, ref.mu0, fam, engine=engine, singular_order=5, regular_order=8)
    assert _rel(sd, ref.raw["shape_derivatives_bem"]) < 1e-5
