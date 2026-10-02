"""Declarative term algebra for pairwise boundary integrals.

A :class:`Form` is a list of :class:`Term` objects. Each term describes one
double integral over pairs of panels

    coeff * ∫_Γ ∫_Γ  test_op(x) ⋆ K(x, y; V) ⋆ trial_op(y)  dS(y) dS(x)

where ``test_op`` / ``trial_op`` are basis operators applied to the shape
functions of a bempp-cl function space (identity, surface gradient, n × grad,
n·, div, n ×, DV·), ``K`` is a named kernel from :mod:`kernels` (scalar or
vector valued, possibly depending on a velocity field V), and ``⋆`` is the
contraction fixed by the value types:

    scalar K, scalar test, scalar trial :  K φ_i φ_j
    scalar K, vector test, vector trial :  K ψ_i · ψ_j
    vector K, scalar test, vector trial :  φ_i (K · ψ_j)
    vector K, vector test, scalar trial :  (ψ_i · K) φ_j
    vector K, vector test, vector trial :  ψ_i · (K × ψ_j)

A term with coefficient vectors attached to both sides is evaluated as a
number (or one number per velocity field); a term without coefficients is
assembled as a matrix.  This is the single contract every backend implements.
"""

from dataclasses import dataclass, field, replace
import numpy as _np

from . import kernels as _kernels

# Base operators available per space kind and the value type they produce.
_BASE_OPS = {
    "P0": {"id": "scalar"},
    "P1": {"id": "scalar", "grad": "vector", "nxgrad": "vector", "n": "vector"},
    "RWG": {"id": "vector", "div": "scalar"},
}

# Modifiers: vector -> vector maps that may be chained after a base operator.
_MODIFIERS = ("nx", "DV")

_IDENTIFIER_TO_KIND = {
    "p0_discontinuous": "P0",
    "p1_discontinuous": "P1",
    "p1_continuous": "P1",
    "rwg0": "RWG",
    "snc0": "SNC",
}


def space_kind(space):
    """Return the kind ('P0', 'P1', 'RWG', 'SNC') of a bempp-cl space."""
    try:
        return _IDENTIFIER_TO_KIND[space.identifier]
    except KeyError:
        raise ValueError(f"Space '{space.identifier}' is not supported by the forces engine.")


def _normalize_ops(kind, ops):
    """Normalize an operator specification into a tuple of op names."""
    if isinstance(ops, str):
        ops = tuple(ops.split("."))
    ops = tuple(ops)
    if len(ops) == 0:
        ops = ("id",)
    if kind == "SNC":
        # SNC is n x RWG on the same dofs.
        kind = "RWG"
        if ops[0] != "id":
            raise ValueError("Only the identity base operator is supported on SNC spaces.")
        ops = ("id", "nx") + ops[1:]
    base = ops[0]
    if base not in _BASE_OPS[kind]:
        raise ValueError(f"Operator '{base}' is not defined for {kind} spaces.")
    vtype = _BASE_OPS[kind][base]
    for m in ops[1:]:
        if m not in _MODIFIERS:
            raise ValueError(f"Unknown modifier '{m}'.")
        if vtype != "vector":
            raise ValueError(f"Modifier '{m}' requires a vector valued operand.")
    return kind, ops, vtype


@dataclass(frozen=True)
class BasisArg:
    """A function space together with an operator and optional coefficients."""

    space: object
    ops: tuple = ("id",)
    coeffs: object = None
    kind: str = field(init=False)
    value_type: str = field(init=False)

    def __post_init__(self):
        kind, ops, vtype = _normalize_ops(space_kind(self.space), self.ops)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "ops", ops)
        object.__setattr__(self, "value_type", vtype)
        if self.coeffs is not None:
            coeffs = _np.asarray(self.coeffs)
            if coeffs.ndim != 1 or coeffs.shape[0] != self.space.global_dof_count:
                raise ValueError(
                    f"Coefficient vector has shape {coeffs.shape}, expected ({self.space.global_dof_count},)."
                )
            object.__setattr__(self, "coeffs", coeffs)

    @property
    def uses_velocity(self):
        """True if the operator chain involves the velocity Jacobian."""
        return "DV" in self.ops

    def with_coeffs(self, coeffs):
        """Return a copy carrying the given coefficient vector."""
        return replace(self, coeffs=coeffs)


def arg(space, ops="id", coeffs=None):
    """Create a :class:`BasisArg`."""
    return BasisArg(space, ops, coeffs)


def DV(basis_arg):
    """Apply the velocity Jacobian to a vector valued basis argument."""
    return replace(basis_arg, ops=basis_arg.ops + ("DV",))


def nx(basis_arg):
    """Apply n × to a vector valued basis argument."""
    return replace(basis_arg, ops=basis_arg.ops + ("nx",))


_CONTRACTIONS = {
    ("scalar", "scalar", "scalar"): "sss",
    ("scalar", "vector", "vector"): "svv",
    ("vector", "scalar", "vector"): "vsv",
    ("vector", "vector", "scalar"): "vvs",
    ("vector", "vector", "vector"): "vvv",
}


@dataclass(frozen=True)
class Term:
    """One weighted pairwise integral."""

    coeff: float
    kernel: str
    test: BasisArg
    trial: BasisArg
    contraction: str = field(init=False)

    def __post_init__(self):
        kern = _kernels.get(self.kernel)
        key = (kern.value_type, self.test.value_type, self.trial.value_type)
        if key not in _CONTRACTIONS:
            raise ValueError(
                f"Unsupported combination kernel={kern.value_type}, test={self.test.value_type}, "
                f"trial={self.trial.value_type} in term '{self.kernel}'."
            )
        object.__setattr__(self, "contraction", _CONTRACTIONS[key])
        c = complex(self.coeff)
        object.__setattr__(self, "coeff", c.real if c.imag == 0.0 else c)

    @property
    def is_complex(self):
        """True if the kernel or the coefficient is complex."""
        return _kernels.get(self.kernel).is_complex or isinstance(self.coeff, complex)

    @property
    def uses_velocity(self):
        """True if the term depends on a velocity field."""
        return _kernels.get(self.kernel).needs_velocity or self.test.uses_velocity or self.trial.uses_velocity

    @property
    def has_coefficients(self):
        """True if both sides carry coefficient vectors."""
        return self.test.coeffs is not None and self.trial.coeffs is not None

    def __mul__(self, scalar):
        return replace(self, coeff=self.coeff * complex(scalar))

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    def __add__(self, other):
        return Form([self]) + other


def term(kernel, test, trial, coeff=1.0):
    """Create a :class:`Term`."""
    return Term(coeff, kernel, test, trial)


class Form:
    """A sum of terms."""

    def __init__(self, terms=()):
        """Create a form from an iterable of terms."""
        self.terms = []
        for t in terms:
            if isinstance(t, Form):
                self.terms.extend(t.terms)
            elif isinstance(t, Term):
                self.terms.append(t)
            else:
                raise TypeError(f"Cannot add object of type {type(t)} to a Form.")

    def __add__(self, other):
        if isinstance(other, (Term, Form)):
            return Form(self.terms + Form([other]).terms)
        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return Form([-t for t in self.terms])

    def __mul__(self, scalar):
        return Form([t * scalar for t in self.terms])

    __rmul__ = __mul__

    def __iter__(self):
        return iter(self.terms)

    def __len__(self):
        return len(self.terms)

    def __getitem__(self, i):
        return self.terms[i]

    @property
    def uses_velocity(self):
        """True if any term depends on a velocity field."""
        return any(t.uses_velocity for t in self.terms)

    @property
    def has_coefficients(self):
        """True if every term carries coefficients on both sides."""
        return all(t.has_coefficients for t in self.terms)

    @property
    def is_complex(self):
        """True if any term is complex valued."""
        return any(t.is_complex for t in self.terms)

    def check_single_space_pair(self):
        """Return (test_space, trial_space) if all terms share the same spaces."""
        spaces = {(t.test.space.id, t.trial.space.id) for t in self.terms}
        if len(spaces) != 1:
            raise ValueError("All terms of a form to be assembled must share the same test and trial spaces.")
        t = self.terms[0]
        return t.test.space, t.trial.space
