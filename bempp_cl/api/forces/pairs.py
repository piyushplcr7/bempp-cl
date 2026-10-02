"""Pair enumeration and quadrature rules for pairwise panel integration.

Panel pairs fall into four classes by the number of shared vertices:
far (0), vertex adjacent (1), edge adjacent (2) and coincident (3).  Far
pairs use a tensor product of a symmetric triangle Gauss rule; the adjacent
classes use bempp-cl's Sauter–Schwab (Duffy) rules, reusing
:class:`bempp_cl.core.singular_assembler._SingularQuadratureRuleInterfaceGalerkin`
which also provides the remapping for the shared local vertices.
"""

import numpy as _np

from bempp_cl.api.integration import triangle_gauss as _triangle_gauss
from bempp_cl.core.singular_assembler import (
    _SingularQuadratureRuleInterfaceGalerkin as _SingularRule,
)

CLASS_NAMES = ("coincident", "edge_adjacent", "vertex_adjacent")


class SingularClass:
    """Pairs of one adjacency class with per-pair point offsets."""

    def __init__(self, name, test_elements, trial_elements, test_offsets, trial_offsets, weights_offset, npoints):
        self.name = name
        self.test_elements = _np.asarray(test_elements, dtype=_np.int64)
        self.trial_elements = _np.asarray(trial_elements, dtype=_np.int64)
        self.test_offsets = _np.asarray(test_offsets, dtype=_np.int64)
        self.trial_offsets = _np.asarray(trial_offsets, dtype=_np.int64)
        self.weights_offset = int(weights_offset)
        self.npoints = int(npoints)

    @property
    def npairs(self):
        return len(self.test_elements)


class PairSet:
    """All panel pairs of a grid with their quadrature rules."""

    def __init__(self, grid, test_support=None, trial_support=None, singular_order=4, regular_order=4):
        """Create pairs and rules for the elements in the supports (default: all)."""
        self.grid = grid
        ne = grid.number_of_elements
        if test_support is None:
            test_support = _np.ones(ne, dtype=bool)
        if trial_support is None:
            trial_support = _np.ones(ne, dtype=bool)
        self.test_support = _np.asarray(test_support, dtype=bool)
        self.trial_support = _np.asarray(trial_support, dtype=bool)
        self.singular_order = singular_order
        self.regular_order = regular_order

        # Regular (far) tensor rule.
        pts, w = _triangle_gauss.rule(regular_order)
        self.regular_points = _np.ascontiguousarray(pts.T)  # (nq1, 2)
        self.regular_weights = _np.ascontiguousarray(w)  # (nq1,)
        self.regular_points_test = _np.repeat(self.regular_points, len(w), axis=0)  # (nq1*nq1, 2)
        self.regular_points_trial = _np.tile(self.regular_points, (len(w), 1))
        self.regular_weights_pairs = _np.repeat(w, len(w)) * _np.tile(w, len(w))

        # Singular classes from bempp-cl.
        rule = _SingularRule(grid, singular_order, self.test_support, self.trial_support)
        (
            test_points,
            trial_points,
            weights,
            test_indices,
            trial_indices,
            test_offsets,
            trial_offsets,
            weights_offsets,
            number_of_quad_points,
        ) = rule.get_arrays()
        self.singular_test_points = _np.ascontiguousarray(test_points.T)  # (Ntot, 2)
        self.singular_trial_points = _np.ascontiguousarray(trial_points.T)
        self.singular_weights = _np.ascontiguousarray(weights)
        self.singular_test_elements = test_indices.astype(_np.int64)
        self.singular_trial_elements = trial_indices.astype(_np.int64)
        self.singular_test_offsets = test_offsets.astype(_np.int64)
        self.singular_trial_offsets = trial_offsets.astype(_np.int64)
        self.singular_weights_offsets = weights_offsets.astype(_np.int64)
        self.singular_npoints = number_of_quad_points.astype(_np.int64)

        counts = rule.index_count
        starts = {
            "coincident": 0,
            "edge_adjacent": counts["coincident"],
            "vertex_adjacent": counts["coincident"] + counts["edge_adjacent"],
        }
        self.classes = {}
        for name in CLASS_NAMES:
            s = starts[name]
            n = counts[name]
            sl = slice(s, s + n)
            self.classes[name] = SingularClass(
                name,
                test_indices[sl],
                trial_indices[sl],
                test_offsets[sl],
                trial_offsets[sl],
                weights_offsets[s] if n > 0 else 0,
                rule.number_of_points(name),
            )

    @property
    def test_elements(self):
        """Indices of supported test elements."""
        return _np.flatnonzero(self.test_support)

    @property
    def trial_elements(self):
        """Indices of supported trial elements."""
        return _np.flatnonzero(self.trial_support)

    @property
    def number_of_regular_points(self):
        """Number of quadrature points per far pair."""
        return len(self.regular_weights_pairs)

    def singular_points(self, cls):
        """Return (test_pts, trial_pts, weights) for a class: [npairs, nq, 2], [npairs, nq, 2], [nq]."""
        c = self.classes[cls]
        q = _np.arange(c.npoints)
        tp = self.singular_test_points[c.test_offsets[:, None] + q[None, :]]
        rp = self.singular_trial_points[c.trial_offsets[:, None] + q[None, :]]
        w = self.singular_weights[c.weights_offset : c.weights_offset + c.npoints]
        return tp, rp, w

    def summary(self):
        """Return a short description of pair counts."""
        nt, nr = len(self.test_elements), len(self.trial_elements)
        nsing = sum(c.npairs for c in self.classes.values())
        return {
            "test_elements": nt,
            "trial_elements": nr,
            "far_pairs": nt * nr - nsing,
            **{k: v.npairs for k, v in self.classes.items()},
            "regular_points_per_pair": self.number_of_regular_points,
            **{f"{k}_points": v.npoints for k, v in self.classes.items()},
        }
