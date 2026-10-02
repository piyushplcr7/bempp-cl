"""CUDA backend: hand-written kernel skeleton with per-form term glue.

The numerics live in ``cuda/skeleton.cu``.  For a given form this module
generates only (a) a handful of ``#define`` switches, (b) the ``Args`` struct
that unpacks a pointer table, (c) one small device function per term side
(basis function or trace value) and (d) the per-point ``point_contrib`` body
that calls the skeleton's kernels and contractions.  The generated text is
spliced at the ``@@GENERATED@@`` marker and compiled with NVRTC through CuPy.
"""

import hashlib as _hashlib
import os as _os

import numpy as _np

from .. import kernels as _kernels
from ..basis import ElementData, SpaceData
from ..pairs import PairSet

_SKELETON_PATH = _os.path.join(_os.path.dirname(__file__), "cuda", "skeleton.cu")
_FIELD_CHUNK = 27
_BLOCK = 128
_MAX_BLOCKS = 4096
_VEL_KIND = {"cos": 0, "rot": 1, "trans": 2}


# ----------------------------------------------------------------------------- codegen
def _basis_expr(kind, base, S, e, k, s, t, x):
    """C expression for basis function ``k`` of a base operator."""
    if kind == "P0":
        return f"basis_P0({S}, {e}, {k})"
    if kind == "P1":
        return {
            "id": f"basis_P1({S}, {e}, {k}, {s}, {t})",
            "grad": f"basis_P1_grad(G, {S}, {e}, {k})",
            "nxgrad": f"basis_P1_nxgrad(G, {S}, {e}, {k})",
            "n": f"basis_P1_n(G, {S}, {e}, {k}, {s}, {t})",
        }[base]
    if kind == "RWG":
        return {
            "id": f"basis_RWG(G, {S}, {e}, {k}, {x})",
            "div": f"basis_RWG_div({S}, {e}, {k})",
        }[base]
    raise ValueError(kind)


def _side_functions(ti, side_name, barg, space_slot, mode):
    """Generate the device function(s) for one term side.

    Returns (code, ctype, pre_dv_ops_applied) where ctype is 'REAL' or 'R3'.
    The 'DV' modifier is not applied here (it depends on the field) - the
    caller applies ``matvec`` in the field loop.
    """
    kind, ops = barg.kind, barg.ops
    base = ops[0]
    mods = [m for m in ops[1:]]
    ctype = "REAL" if barg.value_type == "scalar" else "R3"
    S = f"A.S{space_slot}"
    expr = _basis_expr(kind, base, S, "e", "k", "s", "t", "x")
    # modifiers before DV are field independent; anything after DV is applied in the field loop
    pre, post = [], []
    seen_dv = False
    for m in mods:
        if m == "DV":
            seen_dv = True
            continue
        (post if seen_dv else pre).append(m)
    for m in pre:
        if m == "nx":
            expr = f"cross(normal(G, {S}, e), {expr})"
    fn = f"side_{ti}_{side_name}"
    nsh = barg.space.number_of_shape_functions
    code = [
        f"__device__ __forceinline__ {ctype} {fn}_k(const Args& A, const Geo& G, int e, int k, REAL s, REAL t, R3 x) {{",
        f"    return {expr};",
        "}",
    ]
    if mode == "evaluate":
        cslot = f"A.c{ti}_{side_name}"
        zero = "(REAL)0" if ctype == "REAL" else "r3(0, 0, 0)"
        code += [
            f"__device__ __forceinline__ {ctype} {fn}(const Args& A, const Geo& G, int e, REAL s, REAL t, R3 x) {{",
            f"    {ctype} acc = {zero};",
            f"    for (int k = 0; k < {nsh}; ++k) acc += __ldg({cslot} + {nsh} * e + k) * {fn}_k(A, G, e, k, s, t, x);",
            "    return acc;",
            "}",
        ]
    return "\n".join(code), ctype, post


def _apply_post(expr, post_mods, side, S):
    """Apply field dependent modifiers (DV and anything after it)."""
    dv = "c.DVx" if side == "test" else "c.DVy"
    e = "te" if side == "test" else "tr"
    expr = f"matvec({dv}, {expr})"
    for m in post_mods:
        if m == "nx":
            expr = f"cross(normal(G, {S}, {e}), {expr})"
    return expr


def generate(form, mode, space_slots, velocity_kind, has_velocity):
    """Generate the spliced source section for a form."""
    out = []
    out.append("#define MODE_EVAL" if mode == "evaluate" else "#define MODE_ASSEMBLE")
    test_space, trial_space = form[0].test.space, form[0].trial.space
    nsh_t = test_space.number_of_shape_functions
    nsh_r = trial_space.number_of_shape_functions
    out.append(f"#define NSH_T {nsh_t}")
    out.append(f"#define NSH_R {nsh_r}")
    out.append(f"#define VEL_KIND {velocity_kind}")
    out.append(f"#define NEED_VEL {1 if has_velocity else 0}")

    # ---- Args struct and unpack
    ptr_names = ["G.corners", "G.normals", "G.intelem", "G.p1grad", "G.elements", "G.ne"]
    fields = ["Geo G;"]
    for slot in range(len(space_slots)):
        fields.append(f"SpaceD S{slot};")
        ptr_names += [f"S{slot}.mult", f"S{slot}.nmult", f"S{slot}.rwg", f"S{slot}.l2g", f"S{slot}.nshape"]
    if mode == "evaluate":
        for ti, t in enumerate(form):
            fields.append(f"const REAL* c{ti}_test; const REAL* c{ti}_trial;")
            ptr_names += [f"c{ti}_test", f"c{ti}_trial"]
    fields += [
        "const int* test_el; const int* trial_el;",
        "const REAL* reg_pt; const REAL* reg_pr; const REAL* reg_w;",
        "const int* s_te; const int* s_tr; const int* s_ot; const int* s_or; const int* s_ow; const int* s_np;",
        "const REAL* s_pt; const REAL* s_pr; const REAL* s_w;",
        "const REAL* velparams;",
        "REAL* partials;",
        "REAL* out; int ld;",
    ]
    ptr_names += [
        "test_el", "trial_el", "reg_pt", "reg_pr", "reg_w",
        "s_te", "s_tr", "s_ot", "s_or", "s_ow", "s_np", "s_pt", "s_pr", "s_w",
        "velparams", "partials", "out", "ld",
    ]
    types = {
        "G.corners": "const REAL*", "G.normals": "const REAL*", "G.intelem": "const REAL*", "G.p1grad": "const REAL*",
        "G.elements": "const int*", "G.ne": "int",
        "test_el": "const int*", "trial_el": "const int*",
        "reg_pt": "const REAL*", "reg_pr": "const REAL*", "reg_w": "const REAL*",
        "s_te": "const int*", "s_tr": "const int*", "s_ot": "const int*", "s_or": "const int*", "s_ow": "const int*",
        "s_np": "const int*", "s_pt": "const REAL*", "s_pr": "const REAL*", "s_w": "const REAL*",
        "velparams": "const REAL*", "partials": "REAL*", "out": "REAL*", "ld": "int",
    }
    out.append("struct Args {\n    " + "\n    ".join(fields) + "\n};")
    lines = ["__device__ __forceinline__ Args unpack(const unsigned long long* p) {", "    Args A;"]
    for i, name in enumerate(ptr_names):
        if name.endswith(".mult") or name.endswith(".nmult") or name.endswith(".rwg"):
            ty = "const REAL*"
        elif name.endswith(".l2g"):
            ty = "const int*"
        elif name.endswith(".nshape"):
            ty = "int"
        elif name.startswith("c") and ("_test" in name or "_trial" in name):
            ty = "const REAL*"
        else:
            ty = types[name]
        if ty == "int":
            lines.append(f"    A.{name} = (int)p[{i}];")
        else:
            lines.append(f"    A.{name} = ({ty})p[{i}];")
    lines += ["    return A;", "}"]
    out.append("\n".join(lines))

    # ---- per-term side functions
    sides = []
    for ti, t in enumerate(form):
        code_t, ct_t, post_t = _side_functions(ti, "test", t.test, space_slots[t.test.space.id], mode)
        code_r, ct_r, post_r = _side_functions(ti, "trial", t.trial, space_slots[t.trial.space.id], mode)
        out += [code_t, code_r]
        sides.append((ct_t, post_t, ct_r, post_r))

    # ---- point body
    body = []
    if mode == "evaluate":
        body.append(
            "__device__ __forceinline__ void point_contrib(const Args& A, const Geo& G, int te, int tr, REAL st, REAL tt, "
            "REAL sr, REAL trr, REAL w, const REAL* velparams, int f0, int nf, REAL* acc) {"
        )
    else:
        body.append(
            "__device__ __forceinline__ void point_contrib(const Args& A, const Geo& G, int te, int tr, REAL st, REAL tt, "
            "REAL sr, REAL trr, REAL w, const REAL* velparams, int f0, int nf, REAL* block) {"
        )
    body += [
        "    Ctx c;",
        "    c.x = map_point(G, te, st, tt);",
        "    c.y = map_point(G, tr, sr, trr);",
        "    c.z = c.y - c.x;",
        "    REAL r2 = dot(c.z, c.z);",
        "    c.r = sqrt(r2);",
        "    c.rinv = r2 > 0 ? (REAL)1 / c.r : (REAL)0;",
    ]
    const_terms = [ti for ti, t in enumerate(form) if not t.uses_velocity]
    vel_terms = [ti for ti, t in enumerate(form) if t.uses_velocity]

    def kexpr(t):
        return f"{_kernels.get(t.kernel).cuda}(c)"

    if mode == "evaluate":
        # field independent trace values
        for ti, t in enumerate(form):
            ct_t, _, ct_r, _ = sides[ti]
            body.append(f"    {ct_t} u{ti} = side_{ti}_test(A, G, te, st, tt, c.x);")
            body.append(f"    {ct_r} v{ti} = side_{ti}_trial(A, G, tr, sr, trr, c.y);")
        if const_terms:
            body.append("    if (f0 == 0) {")
            body.append("        REAL s = 0;")
            for ti in const_terms:
                t = form[ti]
                body.append(f"        s += ({t.coeff!r}) * contract({kexpr(t)}, u{ti}, v{ti});")
            body.append("        acc[FIELD_CHUNK] += w * s;")
            body.append("    }")
        if vel_terms:
            body += [
                "#if NEED_VEL",
                "    VelTab Tx, Ty;",
                "    vel_table(VEL_KIND, c.x, Tx);",
                "    vel_table(VEL_KIND, c.y, Ty);",
                "    for (int f = 0; f < nf; ++f) {",
                "        vel_eval(VEL_KIND, velparams + 4 * (f0 + f), c.x, Tx, c.Vx, c.DVx);",
                "        vel_eval(VEL_KIND, velparams + 4 * (f0 + f), c.y, Ty, c.Vy, c.DVy);",
                "        REAL s = 0;",
            ]
            for ti in vel_terms:
                t = form[ti]
                _, post_t, _, post_r = sides[ti]
                ue = f"u{ti}"
                ve = f"v{ti}"
                if t.test.uses_velocity:
                    ue = _apply_post(ue, post_t, "test", f"A.S{space_slots[t.test.space.id]}")
                if t.trial.uses_velocity:
                    ve = _apply_post(ve, post_r, "trial", f"A.S{space_slots[t.trial.space.id]}")
                body.append(f"        s += ({t.coeff!r}) * contract({kexpr(t)}, {ue}, {ve});")
            body += ["        acc[f] += w * s;", "    }", "#endif"]
        body.append("}")
    else:
        if vel_terms:
            body += [
                "#if NEED_VEL",
                "    VelTab Tx, Ty;",
                "    vel_table(VEL_KIND, c.x, Tx);",
                "    vel_table(VEL_KIND, c.y, Ty);",
                "    vel_eval(VEL_KIND, velparams + 4 * f0, c.x, Tx, c.Vx, c.DVx);",
                "    vel_eval(VEL_KIND, velparams + 4 * f0, c.y, Ty, c.Vy, c.DVy);",
                "#endif",
            ]
        for ti, t in enumerate(form):
            ct_t, post_t, ct_r, post_r = sides[ti]
            body.append(f"    {{ // term {ti}: {t.kernel if isinstance(t.kernel, str) else t.kernel.name}")
            body.append(f"        auto K = {kexpr(t)};")
            body.append(f"        {ct_t} ut[NSH_T]; {ct_r} vr[NSH_R];")
            body.append(f"        for (int k = 0; k < NSH_T; ++k) ut[k] = side_{ti}_test_k(A, G, te, k, st, tt, c.x);")
            body.append(f"        for (int k = 0; k < NSH_R; ++k) vr[k] = side_{ti}_trial_k(A, G, tr, k, sr, trr, c.y);")
            ue, ve = "ut[i]", "vr[j]"
            if t.test.uses_velocity:
                ue = _apply_post(ue, post_t, "test", f"A.S{space_slots[t.test.space.id]}")
            if t.trial.uses_velocity:
                ve = _apply_post(ve, post_r, "trial", f"A.S{space_slots[t.trial.space.id]}")
            body.append("        for (int i = 0; i < NSH_T; ++i)")
            body.append("            for (int j = 0; j < NSH_R; ++j)")
            body.append(f"                block[i * NSH_R + j] += w * ({t.coeff!r}) * contract(K, {ue}, {ve});")
            body.append("    }")
        body.append("}")
        st_slot, sr_slot = space_slots[test_space.id], space_slots[trial_space.id]
        body += [
            "__device__ __forceinline__ void scatter_block(const Args& A, int te, int tr, const REAL* block) {",
            "    for (int i = 0; i < NSH_T; ++i) {",
            f"        int row = __ldg(A.S{st_slot}.l2g + NSH_T * te + i);",
            "        for (int j = 0; j < NSH_R; ++j) {",
            f"            int col = __ldg(A.S{sr_slot}.l2g + NSH_R * tr + j);",
            "            atomicAdd(A.out + (size_t)row * A.ld + col, block[i * NSH_R + j]);",
            "        }",
            "    }",
            "}",
        ]
    out.append("\n".join(body))
    return "\n".join(out), ptr_names


# ----------------------------------------------------------------------------- engine
class CudaEngine:
    """Engine running the hand-written CUDA skeleton."""

    def __init__(self, field_chunk=_FIELD_CHUNK, block=_BLOCK, max_blocks=_MAX_BLOCKS, precision="double"):
        import cupy

        self.cp = cupy
        self.field_chunk = int(field_chunk)
        self.block = int(block)
        self.max_blocks = int(max_blocks)
        if precision != "double":
            raise NotImplementedError("Only double precision is implemented so far.")
        self.precision = precision
        self._modules = {}
        self._pairsets = {}
        self._device_data = {}
        with open(_SKELETON_PATH) as f:
            self._skeleton = f.read()

    # ---- compilation
    def _module(self, generated):
        head, tail = self._skeleton.split("// @@GENERATED@@")
        tail = tail.split("// @@END_GENERATED@@")[1]
        src = head + generated + tail
        key = _hashlib.sha1(src.encode()).hexdigest()
        if key not in self._modules:
            opts = (
                "-std=c++17",
                f"-DREAL=double",
                f"-DFIELD_CHUNK={self.field_chunk}",
                f"-DBLOCK={self.block}",
                "--use_fast_math" if False else "-lineinfo",
            )
            self._modules[key] = self.cp.RawModule(code=src, options=opts, backend="nvrtc")
        return self._modules[key]

    # ---- data
    def pairset(self, grid, singular_order, regular_order):
        key = (id(grid), singular_order, regular_order)
        if key not in self._pairsets:
            self._pairsets[key] = PairSet(grid, singular_order=singular_order, regular_order=regular_order)
        return self._pairsets[key]

    def _grid_device(self, grid):
        cp = self.cp
        key = id(grid)
        if key not in self._device_data:
            elem = ElementData.from_grid(grid, _np)
            d = {
                "elem": elem,
                "corners": cp.asarray(_np.ascontiguousarray(elem.corners)),
                "normals": cp.asarray(_np.ascontiguousarray(elem.normals)),
                "intelem": cp.asarray(_np.ascontiguousarray(elem.integration_elements)),
                "p1grad": cp.asarray(_np.ascontiguousarray(elem.p1_grad)),
                "elements": cp.asarray(_np.ascontiguousarray(elem.elements.astype(_np.int32))),
                "spaces": {},
            }
            self._device_data[key] = d
        return self._device_data[key]

    def _space_device(self, gdev, space):
        cp = self.cp
        if space.id not in gdev["spaces"]:
            sd = SpaceData(space, gdev["elem"], _np)
            rwg = sd.rwg_coeff if sd.rwg_coeff is not None else _np.zeros((1, 3))
            gdev["spaces"][space.id] = {
                "sd": sd,
                "mult": cp.asarray(_np.ascontiguousarray(sd.multipliers)),
                "nmult": cp.asarray(_np.ascontiguousarray(sd.normal_multipliers)),
                "rwg": cp.asarray(_np.ascontiguousarray(rwg)),
                "l2g": cp.asarray(_np.ascontiguousarray(sd.local2global.astype(_np.int32))),
                "nshape": sd.nshape,
            }
        return gdev["spaces"][space.id]

    def _pairs_device(self, pairs):
        cp = self.cp
        if not hasattr(pairs, "_cuda"):
            pairs._cuda = {
                "test_el": cp.asarray(pairs.test_elements.astype(_np.int32)),
                "trial_el": cp.asarray(pairs.trial_elements.astype(_np.int32)),
                "reg_pt": cp.asarray(_np.ascontiguousarray(pairs.regular_points_test)),
                "reg_pr": cp.asarray(_np.ascontiguousarray(pairs.regular_points_trial)),
                "reg_w": cp.asarray(_np.ascontiguousarray(pairs.regular_weights_pairs)),
                "s_te": cp.asarray(pairs.singular_test_elements.astype(_np.int32)),
                "s_tr": cp.asarray(pairs.singular_trial_elements.astype(_np.int32)),
                "s_ot": cp.asarray(pairs.singular_test_offsets.astype(_np.int32)),
                "s_or": cp.asarray(pairs.singular_trial_offsets.astype(_np.int32)),
                "s_ow": cp.asarray(pairs.singular_weights_offsets.astype(_np.int32)),
                "s_np": cp.asarray(pairs.singular_npoints.astype(_np.int32)),
                "s_pt": cp.asarray(_np.ascontiguousarray(pairs.singular_test_points)),
                "s_pr": cp.asarray(_np.ascontiguousarray(pairs.singular_trial_points)),
                "s_w": cp.asarray(_np.ascontiguousarray(pairs.singular_weights)),
            }
        return pairs._cuda

    # ---- driver
    def _prepare(self, form, velocity, singular_order, regular_order, mode):
        cp = self.cp
        if form.is_complex:
            raise NotImplementedError("Complex kernels are not implemented in the CUDA backend.")
        grids = {t.test.space.grid.id: t.test.space.grid for t in form}
        grids.update({t.trial.space.grid.id: t.trial.space.grid for t in form})
        if len(grids) != 1:
            raise NotImplementedError("Cross-grid forms are not supported yet.")
        grid = next(iter(grids.values()))
        gdev = self._grid_device(grid)
        pairs = self.pairset(grid, singular_order, regular_order)
        pdev = self._pairs_device(pairs)

        space_slots = {}
        space_devs = []
        for t in form:
            for side in (t.test, t.trial):
                if side.space.id not in space_slots:
                    space_slots[side.space.id] = len(space_slots)
                    space_devs.append(self._space_device(gdev, side.space))

        has_velocity = form.uses_velocity
        vel_kind = _VEL_KIND[velocity.kind] if velocity is not None else 0
        generated, ptr_names = generate(form, mode, space_slots, vel_kind, has_velocity)
        module = self._module(generated)

        # velocity parameters padded to width 4
        if velocity is not None:
            vp = _np.zeros((velocity.nfields, 4))
            vp[:, : velocity.params.shape[1]] = velocity.params
        else:
            vp = _np.zeros((1, 4))
        velparams = cp.asarray(vp)

        coeff_devs = []
        if mode == "evaluate":
            for t in form:
                for side in (t.test, t.trial):
                    sd = gdev["spaces"][side.space.id]["sd"]
                    coeff_devs.append(cp.asarray(_np.ascontiguousarray(sd.local_coefficients(side.coeffs))))

        return dict(
            grid=grid, gdev=gdev, pairs=pairs, pdev=pdev, space_slots=space_slots, space_devs=space_devs,
            module=module, ptr_names=ptr_names, velparams=velparams, coeff_devs=coeff_devs, keep=[],
        )

    def _pointer_table(self, P, partials, out, ld):
        cp = self.cp
        gdev, pdev = P["gdev"], P["pdev"]

        def ptr(a):
            return int(a.data.ptr)

        vals = [ptr(gdev["corners"]), ptr(gdev["normals"]), ptr(gdev["intelem"]), ptr(gdev["p1grad"]), ptr(gdev["elements"]), gdev["elem"].number_of_elements]
        for s in P["space_devs"]:
            vals += [ptr(s["mult"]), ptr(s["nmult"]), ptr(s["rwg"]), ptr(s["l2g"]), s["nshape"]]
        for c in P["coeff_devs"]:
            vals.append(ptr(c))
        vals += [ptr(pdev[k]) for k in ("test_el", "trial_el", "reg_pt", "reg_pr", "reg_w", "s_te", "s_tr", "s_ot", "s_or", "s_ow", "s_np", "s_pt", "s_pr", "s_w")]
        vals += [ptr(P["velparams"]), ptr(partials) if partials is not None else 0, ptr(out) if out is not None else 0, int(ld)]
        assert len(vals) == len(P["ptr_names"]), (len(vals), len(P["ptr_names"]))
        return cp.asarray(_np.array(vals, dtype=_np.uint64))

    def _launch(self, P, kernel_name, iprm, f0, nf, ptrs, npairs):
        cp = self.cp
        kern = P["module"].get_function(kernel_name)
        blocks = int(max(1, min(self.max_blocks, (npairs + self.block - 1) // self.block)))
        iprm_dev = cp.asarray(_np.array(iprm, dtype=_np.int32))
        kern((blocks,), (self.block,), (ptrs, iprm_dev, _np.int32(f0), _np.int32(nf)))
        return blocks

    def evaluate(self, form, velocity=None, singular_order=4, regular_order=4, per_term=False, vertices=None):
        """Evaluate a form with coefficients: returns numpy array [nfields]."""
        if vertices is not None:
            raise NotImplementedError("The CUDA backend does not support overriding vertices.")
        if per_term:
            raise NotImplementedError("per_term is not supported by the CUDA backend.")
        if not form.has_coefficients:
            raise ValueError("evaluate() requires coefficient vectors on both sides of every term.")
        if velocity is None and form.uses_velocity:
            raise ValueError("The form depends on a velocity field but none was given.")
        cp = self.cp
        P = self._prepare(form, velocity, singular_order, regular_order, "evaluate")
        pairs, pdev = P["pairs"], P["pdev"]
        nfields = velocity.nfields if (velocity is not None and form.uses_velocity) else 1
        nt, nr = len(pairs.test_elements), len(pairs.trial_elements)
        nfar = nt * nr
        nsing = len(pairs.singular_test_elements)
        result = _np.zeros(nfields)
        const = 0.0
        FC = self.field_chunk
        for f0 in range(0, nfields, FC):
            nf = min(FC, nfields - f0)
            for name, iprm, npairs in (("far_kernel", [nt, nr, pairs.number_of_regular_points], nfar), ("singular_kernel", [nsing, 0, 0], nsing)):
                if npairs == 0:
                    continue
                blocks = int(max(1, min(self.max_blocks, (npairs + self.block - 1) // self.block)))
                partials = cp.zeros((blocks, FC + 1), dtype=cp.float64)
                ptrs = self._pointer_table(P, partials, None, 0)
                self._launch(P, name, iprm, f0, nf, ptrs, npairs)
                sums = cp.asnumpy(partials.sum(axis=0))
                result[f0 : f0 + nf] += sums[:nf]
                if f0 == 0:
                    const += sums[FC]
        cp.cuda.Stream.null.synchronize()
        return result + const

    def assemble(self, form, velocity=None, singular_order=4, regular_order=4, vertices=None):
        """Assemble a dense matrix (numpy); [nfields, rows, cols] if velocity given."""
        if vertices is not None:
            raise NotImplementedError("The CUDA backend does not support overriding vertices.")
        if velocity is None and form.uses_velocity:
            raise ValueError("The form depends on a velocity field but none was given.")
        cp = self.cp
        test_space, trial_space = form.check_single_space_pair()
        P = self._prepare(form, velocity, singular_order, regular_order, "assemble")
        pairs = P["pairs"]
        nfields = velocity.nfields if (velocity is not None and form.uses_velocity) else 1
        rows, cols = test_space.grid_dof_count, trial_space.grid_dof_count
        nt, nr = len(pairs.test_elements), len(pairs.trial_elements)
        nsing = len(pairs.singular_test_elements)
        mats = []
        for f in range(nfields):
            out = cp.zeros((rows, cols), dtype=cp.float64)
            ptrs = self._pointer_table(P, None, out, cols)
            self._launch(P, "far_kernel", [nt, nr, pairs.number_of_regular_points], f, 1, ptrs, nt * nr)
            if nsing:
                self._launch(P, "singular_kernel", [nsing, 0, 0], f, 1, ptrs, nsing)
            mats.append(cp.asnumpy(out))
        cp.cuda.Stream.null.synchronize()
        return mats[0] if velocity is None else _np.stack(mats)
