"""Kernel registry for the forces engine.

Every kernel is a function of the pair of quadrature points ``x`` (test) and
``y`` (trial), ``z = y - x``, and, for shape-derivative kernels, the velocity
field ``V`` and its Jacobian ``DV`` at ``x`` and ``y``.  All kernels include
the Laplace normalisation ``1 / (4 π)``.

Scalar kernels return one number per point pair, vector kernels a 3-vector.
The ``array`` implementations operate on an array module ``xp`` (numpy or
jax.numpy) with a trailing component axis; the ``cuda`` entry names the device
function implemented in ``engine/cuda/skeleton.cu``.  Kernels may carry a
tuple of real parameters (e.g. a wavenumber) passed to both implementations.

Conventions (identical to the PhD CUDA code, ``z := y - x``):

    SL          1 / (4π r)                                       (V, A2, KV)
    DL          -z / (4π r³)          = ∇_y G                    (double layer, with n_y on trial side)
    ADL         +z / (4π r³)          = ∇_x G                    (adjoint double layer / C1)
    A1          z·(V(x) − V(y)) / (4π r³)                        (KernelOld, KernelA1, KernelN)
    C3          [−3 z (z·(V(y) − V(x))) / r² + (V(y) − V(x))] / (4π r³)
    INTEGRABLE  3 z (z·(V(y) − V(x))) / (4π r⁵)
    COMB        (−DV(y) z + V(y) − V(x)) / (4π r³)
    HSL(k)      exp(i k r) / (4π r)                              (Helmholtz single layer, complex)
    HDL(k)      ∇_y exp(i k r)/(4π r) = (i k r − 1) exp(i k r) / (4π r³) · (−z)
"""

from dataclasses import dataclass
import math as _math

_INV4PI = 1.0 / (4.0 * _math.pi)


@dataclass(frozen=True)
class Kernel:
    """A registered kernel."""

    name: str
    value_type: str  # 'scalar' or 'vector'
    needs_velocity: bool
    array: object  # callable(xp, ctx, params) -> array
    cuda: str  # device function name
    params: tuple = ()
    is_complex: bool = False

    def __call__(self, *params):
        """Return a copy of this kernel bound to parameter values."""
        return Kernel(self.name, self.value_type, self.needs_velocity, self.array, self.cuda, tuple(float(p) for p in params), self.is_complex)


class Ctx:
    """Point-pair context passed to array kernels.

    Attributes are arrays with a trailing component axis where applicable:
    ``x, y, z`` [..., 3]; ``r, rinv`` [...]; ``Vx, Vy`` [..., 3];
    ``DVx, DVy`` [..., 3, 3] (``DV[i, j] = ∂V_i/∂x_j``).  Velocity related
    attributes are ``None`` when no velocity family is given.  ``rinv`` is
    zero where ``r`` is zero (masked pairs).
    """

    __slots__ = ("x", "y", "z", "r", "rinv", "Vx", "Vy", "DVx", "DVy")

    def __init__(self, x, y, z, r, rinv, Vx=None, Vy=None, DVx=None, DVy=None):
        self.x, self.y, self.z, self.r, self.rinv = x, y, z, r, rinv
        self.Vx, self.Vy, self.DVx, self.DVy = Vx, Vy, DVx, DVy


def _dot(xp, a, b):
    return xp.sum(a * b, axis=-1)


def _sl(xp, c, p):
    return _INV4PI * c.rinv


def _dl(xp, c, p):
    return (-_INV4PI * c.rinv**3)[..., None] * c.z


def _adl(xp, c, p):
    return (_INV4PI * c.rinv**3)[..., None] * c.z


def _a1(xp, c, p):
    return _INV4PI * _dot(xp, c.z, c.Vx - c.Vy) * c.rinv**3


def _c3(xp, c, p):
    dv = c.Vy - c.Vx
    zdv = _dot(xp, c.z, dv)
    r3 = c.rinv**3
    return _INV4PI * (-3.0 * (zdv * r3 * c.rinv**2)[..., None] * c.z + r3[..., None] * dv)


def _integrable(xp, c, p):
    dv = c.Vy - c.Vx
    zdv = _dot(xp, c.z, dv)
    return (3.0 * _INV4PI * zdv * c.rinv**5)[..., None] * c.z


def _comb(xp, c, p):
    dvz = xp.einsum("...ij,...j->...i", c.DVy, c.z)
    return (_INV4PI * c.rinv**3)[..., None] * (-dvz + c.Vy - c.Vx)


def _hsl(xp, c, p):
    k = p[0] + 1j * p[1]
    return _INV4PI * c.rinv * xp.exp(1j * k * c.r)


def _hdl(xp, c, p):
    k = p[0] + 1j * p[1]
    fac = _INV4PI * (1j * k * c.r - 1.0) * xp.exp(1j * k * c.r) * c.rinv**3
    return -fac[..., None] * c.z


_REGISTRY = {}


def register(name, value_type, needs_velocity, array, cuda, is_complex=False):
    """Register a kernel under a name."""
    if value_type not in ("scalar", "vector"):
        raise ValueError("value_type must be 'scalar' or 'vector'.")
    _REGISTRY[name] = Kernel(name, value_type, bool(needs_velocity), array, cuda, (), bool(is_complex))
    return _REGISTRY[name]


def alias(new_name, existing):
    """Register an alias for an existing kernel."""
    k = get(existing)
    _REGISTRY[new_name] = Kernel(new_name, k.value_type, k.needs_velocity, k.array, k.cuda, k.params, k.is_complex)


def get(name):
    """Return the kernel registered under ``name`` (a Kernel instance is returned as is)."""
    if isinstance(name, Kernel):
        return name
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown kernel '{name}'. Known kernels: {sorted(_REGISTRY)}")


def names():
    """Return the registered kernel names."""
    return sorted(_REGISTRY)


register("SL", "scalar", False, _sl, "kernel_SL")
register("DL", "vector", False, _dl, "kernel_DL")
register("ADL", "vector", False, _adl, "kernel_ADL")
register("A1", "scalar", True, _a1, "kernel_A1")
register("C3", "vector", True, _c3, "kernel_C3")
register("INTEGRABLE", "vector", True, _integrable, "kernel_INTEGRABLE")
register("COMB", "vector", True, _comb, "kernel_COMB")
register("HSL", "scalar", False, _hsl, "kernel_HSL", is_complex=True)
register("HDL", "vector", False, _hdl, "kernel_HDL", is_complex=True)

# Names used in the PhD code.
alias("KV", "SL")
alias("A2", "SL")
alias("C1", "ADL")
alias("OLD", "A1")
alias("N", "A1")
