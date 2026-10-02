"""Array backend: one implementation for numpy and jax.numpy.

This is the portable reference implementation of the term contract in
:mod:`bempp_cl.api.forces.terms`.  All work is done on chunks of panel pairs
with a leading pair axis; with ``xp = jax.numpy`` every chunk computation is
jitted and differentiable with respect to the vertex coordinates.
"""

import numpy as _np

from .. import kernels as _kernels
from ..basis import ElementData, SpaceData, evaluate_basis
from ..pairs import PairSet, CLASS_NAMES


def _is_jax(xp):
    return xp.__name__.startswith("jax")


class ArrayEngine:
    """Reference engine running on numpy or jax.numpy."""

    def __init__(self, xp=_np, pair_chunk=2048, field_chunk=9, jit=True):
        """Create an engine for array module ``xp``."""
        self.xp = xp
        self.pair_chunk = int(pair_chunk)
        self.field_chunk = int(field_chunk)
        self.jit = bool(jit) and _is_jax(xp)
        self._pairsets = {}
        self._jit_cache = {}

    # ------------------------------------------------------------------ setup
    def pairset(self, grid, singular_order, regular_order):
        """Return (cached) pairs and rules for a grid."""
        key = (id(grid), singular_order, regular_order)
        if key not in self._pairsets:
            self._pairsets[key] = PairSet(grid, singular_order=singular_order, regular_order=regular_order)
        return self._pairsets[key]

    def _setup(self, form, vertices=None):
        """Build element and space data for a form (single grid)."""
        xp = self.xp
        grids = {t.test.space.grid.id: t.test.space.grid for t in form}
        grids.update({t.trial.space.grid.id: t.trial.space.grid for t in form})
        if len(grids) != 1:
            raise NotImplementedError("Cross-grid forms are not supported yet.")
        grid = next(iter(grids.values()))
        if vertices is None:
            elem = ElementData.from_grid(grid, xp)
        else:
            elem = ElementData(xp, vertices, grid.elements)
        sdatas = {}
        for t in form:
            for side in (t.test, t.trial):
                if side.space.id not in sdatas:
                    sdatas[side.space.id] = SpaceData(side.space, elem, xp)
        return grid, elem, sdatas

    # --------------------------------------------------------------- kernels
    def _pair_context(self, elem, te, tr, pts_t, pts_r, velocity):
        """Geometry (and velocity) at all point pairs of a chunk."""
        xp = self.xp
        x = elem.global_points(te[:, None], pts_t)  # [P, nq, 3]
        y = elem.global_points(tr[:, None], pts_r)
        z = y - x
        r2 = xp.sum(z * z, axis=-1)
        # Masked pairs (adjacent pairs in the far loop, padding) have r = 0; make them exact zeros.
        nonzero = r2 > 0
        r = xp.sqrt(xp.where(nonzero, r2, 1.0))
        rinv = xp.where(nonzero, 1.0 / r, 0.0)
        # All context arrays carry a field axis at position 2 (size 1 without velocity):
        # x, y, z [P, nq, 1, 3]; r, rinv [P, nq, 1]; V [P, nq, F, 3]; DV [P, nq, F, 3, 3].
        ctx = _kernels.Ctx(x[:, :, None], y[:, :, None], z[:, :, None], r[:, :, None], rinv[:, :, None])
        if velocity is not None:
            ctx.Vx = velocity.V(xp, x)
            ctx.Vy = velocity.V(xp, y)
            ctx.DVx = velocity.DV(xp, x)
            ctx.DVy = velocity.DV(xp, y)
        return ctx

    def _term_integrand(self, term, elem, sdatas, te, tr, pts_t, pts_r, ctx, nfields, mode):
        """Integrand values of one term at all point pairs of a chunk.

        Returns [P, nq, F'] (mode 'evaluate') or [P, nq, F', nsh_t, nsh_r]
        (mode 'assemble'), with F' = nfields if the term uses a velocity
        field, else 1.
        """
        xp = self.xp
        kern = _kernels.get(term.kernel)
        P, nq = pts_t.shape[0], pts_t.shape[1]
        F = nfields if term.uses_velocity else 1

        # Kernel values normalised to [P, nq, F, (3)].
        K = kern.array(xp, ctx, kern.params)
        if K.shape[2] != F:
            K = xp.broadcast_to(K, (P, nq, F) + K.shape[3:])

        def side_values(barg, e_idx, pts, DV):
            sd = sdatas[barg.space.id]
            vals, has_f = evaluate_basis(xp, elem, sd, barg.ops, e_idx, pts, DV=DV)
            if not has_f:
                vals = vals[:, :, None]  # [P, nq, 1, nsh(,3)]
            if vals.shape[2] != F:
                vals = xp.broadcast_to(vals, (P, nq, F) + vals.shape[3:])
            if mode == "evaluate":
                c = sd.local_coefficients(barg.coeffs)[e_idx]  # [P, nsh]
                if barg.value_type == "scalar":
                    vals = xp.einsum("pqfk,pk->pqf", vals, c)
                else:
                    vals = xp.einsum("pqfkc,pk->pqfc", vals, c)
            return vals

        DVx = ctx.DVx if term.test.uses_velocity else None
        DVy = ctx.DVy if term.trial.uses_velocity else None
        ut = side_values(term.test, te, pts_t, DVx)
        vr = side_values(term.trial, tr, pts_r, DVy)
        rule = term.contraction

        if mode == "evaluate":
            # ut/vr: [P, nq, F] or [P, nq, F, 3]
            if rule == "sss":
                val = K * ut * vr
            elif rule == "svv":
                val = K * xp.sum(ut * vr, axis=-1)
            elif rule == "vsv":
                val = ut * xp.sum(K * vr, axis=-1)
            elif rule == "vvs":
                val = xp.sum(ut * K, axis=-1) * vr
            elif rule == "vvv":
                val = xp.sum(ut * xp.cross(K, vr, axis=-1), axis=-1)
            else:
                raise AssertionError(rule)
            return val

        # assemble: ut [P,nq,F,i(,3)], vr [P,nq,F,j(,3)] -> [P,nq,F,i,j]
        if rule == "sss":
            val = K[..., None, None] * ut[..., :, None] * vr[..., None, :]
        elif rule == "svv":
            val = K[..., None, None] * xp.einsum("pqfic,pqfjc->pqfij", ut, vr)
        elif rule == "vsv":
            val = ut[..., :, None] * xp.einsum("pqfc,pqfjc->pqfj", K, vr)[..., None, :]
        elif rule == "vvs":
            val = xp.einsum("pqfic,pqfc->pqfi", ut, K)[..., :, None] * vr[..., None, :]
        elif rule == "vvv":
            Kx = xp.cross(K[..., None, :], vr, axis=-1)  # [P,nq,F,j,3]
            val = xp.einsum("pqfic,pqfjc->pqfij", ut, Kx)
        else:
            raise AssertionError(rule)
        return val

    def _chunk(self, form, elem, sdatas, te, tr, pts_t, pts_r, w, velocity, mode):
        """Weighted contributions of all terms for one chunk of pairs.

        ``w`` is [P, nq] and already includes the Jacobian factors and any
        masking.  Returns a list (per term) of [P, F'] or [P, F', i, j].
        """
        xp = self.xp
        nfields = velocity.nfields if velocity is not None else 1
        ctx = self._pair_context(elem, te, tr, pts_t, pts_r, velocity if form.uses_velocity else None)
        out = []
        for term in form:
            val = self._term_integrand(term, elem, sdatas, te, tr, pts_t, pts_r, ctx, nfields, mode)
            if mode == "evaluate":
                out.append(term.coeff * xp.einsum("pqf,pq->pf", val, w))
            else:
                out.append(term.coeff * xp.einsum("pqfij,pq->pfij", val, w))
        return out

    def _chunk_fn(self, form, mode, velocity):
        """Return the (possibly jitted) chunk function."""
        if not self.jit:

            def call_eager(elem, sdatas, te, tr, pts_t, pts_r, w):
                return self._chunk(form, elem, sdatas, te, tr, pts_t, pts_r, w, velocity, mode)

            return call_eager
        import jax

        key = (id(form), mode, id(velocity))
        entry = self._jit_cache.get(key)
        if entry is None or entry[0] is not form or entry[1] is not velocity:

            def fn(elem_arrays, sd_arrays, te, tr, pts_t, pts_r, w):
                elem_o, sd_o = self._rebind(form, elem_arrays, sd_arrays)
                return self._chunk(form, elem_o, sd_o, te, tr, pts_t, pts_r, w, velocity, mode)

            entry = (form, velocity, jax.jit(fn))
            self._jit_cache[key] = entry
        jitted = entry[2]

        def call(elem, sdatas, te, tr, pts_t, pts_r, w):
            return jitted(self._unbind_elem(elem), self._unbind_sd(sdatas), te, tr, pts_t, pts_r, w)

        return call

    # ---- helpers to pass ElementData/SpaceData through jax.jit as pytrees of arrays
    _ELEM_FIELDS = ("corners", "jac", "integration_elements", "normals", "jac_inv_t", "p1_grad", "edge_lengths")
    _SD_FIELDS = ("multipliers", "normal_multipliers", "rwg_coeff")

    def _unbind_elem(self, elem):
        return {k: getattr(elem, k) for k in self._ELEM_FIELDS}

    def _unbind_sd(self, sdatas):
        return {sid: {k: getattr(sd, k) for k in self._SD_FIELDS if getattr(sd, k) is not None} for sid, sd in sdatas.items()}

    def _rebind(self, form, elem_arrays, sd_arrays):
        elem = self._template_elem
        sdatas = self._template_sd
        e = _ShallowCopy(elem)
        for k, v in elem_arrays.items():
            setattr(e, k, v)
        s = {}
        for sid, sd in sdatas.items():
            c = _ShallowCopy(sd)
            for k, v in sd_arrays[sid].items():
                setattr(c, k, v)
            s[sid] = c
        return e, s

    # ---------------------------------------------------------------- driver
    def _run(self, form, mode, velocity=None, singular_order=4, regular_order=4, vertices=None):
        xp = self.xp
        grid, elem, sdatas = self._setup(form, vertices)
        self._template_elem, self._template_sd = elem, sdatas
        pairs = self.pairset(grid, singular_order, regular_order)
        chunk_fn = self._chunk_fn(form, mode, velocity)
        nfields = velocity.nfields if velocity is not None else 1

        dtypes = [("complex128" if t.is_complex else "float64") for t in form]
        if mode == "evaluate":
            acc = [xp.zeros((nfields if t.uses_velocity else 1,), dtype=d) for t, d in zip(form, dtypes)]
        else:
            test_space, trial_space = form.check_single_space_pair()
            sd_t, sd_r = sdatas[test_space.id], sdatas[trial_space.id]
            acc = [
                xp.zeros(((nfields if t.uses_velocity else 1), sd_t.grid_dof_count, sd_r.grid_dof_count), dtype=d)
                for t, d in zip(form, dtypes)
            ]

        def accumulate(vals, te, tr):
            for i, v in enumerate(vals):
                if mode == "evaluate":
                    acc[i] = acc[i] + xp.sum(v, axis=0)
                else:
                    rows = sd_t.local2global[te]  # [P, i]
                    cols = sd_r.local2global[tr]  # [P, j]
                    v = xp.moveaxis(v, 1, 0)  # [F, P, i, j]
                    if _is_jax(xp):
                        acc[i] = acc[i].at[:, rows[:, :, None], cols[:, None, :]].add(v)
                    else:
                        for f in range(v.shape[0]):
                            _np.add.at(acc[i][f], (rows[:, :, None], cols[:, None, :]), v[f])

        ie = elem.integration_elements

        # ---- singular classes
        for name in CLASS_NAMES:
            cls = pairs.classes[name]
            if cls.npairs == 0:
                continue
            tp_all, rp_all, w_cls = pairs.singular_points(name)
            w_cls = xp.asarray(w_cls)
            P = max(1, min(self.pair_chunk, cls.npairs))
            for s in range(0, cls.npairs, P):
                te = cls.test_elements[s : s + P]
                tr = cls.trial_elements[s : s + P]
                n = len(te)
                pad = P - n
                te_p = _np.concatenate([te, _np.zeros(pad, dtype=te.dtype)])
                tr_p = _np.concatenate([tr, _np.zeros(pad, dtype=tr.dtype)])
                pts_t = xp.asarray(_np.concatenate([tp_all[s : s + n], _np.zeros((pad,) + tp_all.shape[1:])]))
                pts_r = xp.asarray(_np.concatenate([rp_all[s : s + n], _np.zeros((pad,) + rp_all.shape[1:])]))
                mask = xp.asarray(_np.concatenate([_np.ones(n), _np.zeros(pad)]))
                w = (ie[te_p] * ie[tr_p] * mask)[:, None] * w_cls[None, :]
                vals = chunk_fn(elem, sdatas, xp.asarray(te_p), xp.asarray(tr_p), pts_t, pts_r, w)
                accumulate([v[:n] for v in vals], te, tr)

        # ---- far pairs: blocks of test elements x all trial elements, adjacency masked
        test_el = pairs.test_elements
        trial_el = pairs.trial_elements
        nr = len(trial_el)
        T = max(1, self.pair_chunk // nr)
        pts_t1 = xp.asarray(pairs.regular_points_test)  # [nq, 2]
        pts_r1 = xp.asarray(pairs.regular_points_trial)
        w_reg = xp.asarray(pairs.regular_weights_pairs)
        nq = len(pairs.regular_weights_pairs)
        P = T * nr
        for s in range(0, len(test_el), T):
            tb = test_el[s : s + T]
            nt = len(tb)
            te = _np.repeat(tb, nr)
            tr = _np.tile(trial_el, nt)
            adj = elem.adjacent(tb, trial_el).reshape(-1)
            n = len(te)
            pad = P - n
            te_p = _np.concatenate([te, _np.zeros(pad, dtype=te.dtype)])
            tr_p = _np.concatenate([tr, _np.zeros(pad, dtype=tr.dtype)])
            mask = _np.concatenate([(~adj).astype("float64"), _np.zeros(pad)])
            w = (ie[te_p] * ie[tr_p] * xp.asarray(mask))[:, None] * w_reg[None, :]
            pts_t = xp.broadcast_to(pts_t1[None], (P, nq, 2))
            pts_r = xp.broadcast_to(pts_r1[None], (P, nq, 2))
            vals = chunk_fn(elem, sdatas, xp.asarray(te_p), xp.asarray(tr_p), pts_t, pts_r, w)
            accumulate([v[:n] for v in vals], te, tr)

        return acc

    # ------------------------------------------------------------------ API
    def evaluate(self, form, velocity=None, singular_order=4, regular_order=4, vertices=None, per_term=False):
        """Evaluate a form with coefficients: returns [nfields] (or per-term [nterms, nfields])."""
        if not form.has_coefficients:
            raise ValueError("evaluate() requires coefficient vectors on both sides of every term.")
        if velocity is None and form.uses_velocity:
            raise ValueError("The form depends on a velocity field but none was given.")
        nfields = velocity.nfields if velocity is not None else 1
        xp = self.xp
        res = self._run(form, "evaluate", velocity, singular_order, regular_order, vertices)
        res = [xp.broadcast_to(r, (nfields,)) for r in res]
        stacked = xp.stack(res)
        return stacked if per_term else xp.sum(stacked, axis=0)

    def assemble(self, form, velocity=None, singular_order=4, regular_order=4, vertices=None):
        """Assemble a form as a dense matrix (grid dofs); [nfields, rows, cols] if velocity given."""
        if velocity is None and form.uses_velocity:
            raise ValueError("The form depends on a velocity field but none was given.")
        xp = self.xp
        nfields = velocity.nfields if velocity is not None else 1
        res = self._run(form, "assemble", velocity, singular_order, regular_order, vertices)
        total = None
        for r in res:
            r = xp.broadcast_to(r, (nfields,) + r.shape[1:])
            total = r if total is None else total + r
        return total if velocity is not None else total[0]


def _ShallowCopy(obj):
    """Shallow copy of an object (used to rebind arrays under jit)."""
    import copy

    return copy.copy(obj)
