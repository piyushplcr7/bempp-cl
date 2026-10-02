"""Shape derivatives, forces and magnetostatic operators on top of bempp-cl.

Quick start::

    import bempp_cl.api as bempp
    from bempp_cl.api.forces import arg, term, Form, cos_family, evaluate, assemble

    grid = bempp.shapes.sphere(h=0.2)
    P0 = bempp.function_space(grid, "DP", 0)
    V = assemble(Form([term("SL", arg(P0), arg(P0))]))          # Laplace single layer matrix
    sd = evaluate(Form([term("A1", arg(P0, coeffs=psi), arg(P0, coeffs=psi))]), velocity=cos_family(3))
"""

from .terms import BasisArg, Term, Form, arg, term, DV, nx  # noqa: F401
from .velocity import VelocityFamily, cos_family, rotations, translations, single_cos  # noqa: F401
from .pairs import PairSet  # noqa: F401
from .basis import ElementData, SpaceData  # noqa: F401
from . import kernels  # noqa: F401
from .engine import set_backend, get_engine, make_engine, available_backends, backend_name  # noqa: F401


def assemble(form, **kwargs):
    """Assemble a form with the default engine."""
    return get_engine().assemble(form, **kwargs)


def evaluate(form, velocity=None, **kwargs):
    """Evaluate a form with coefficients with the default engine."""
    return get_engine().evaluate(form, velocity=velocity, **kwargs)
