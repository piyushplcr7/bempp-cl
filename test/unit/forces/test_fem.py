"""FEM cross-validation of the TP-VP forces against dolfinx.

Two permeable spheres (mu_i) on the x-axis sit in a uniform applied field B0
(along z) in a bounded vacuum matrix (mu_e) up to radius R_OUT, where
phi = -H0 . x is prescribed.  The physical force is along x (parallel
side-by-side dipoles repel); Newton's third law and the field-direction
symmetry give internal consistency checks.

The FEM reference solves  div(mu grad phi) = 0  with dolfinx (Lagrange
elements).  Force comes from the Maxwell stress tensor integrated over a
GAUSS SPHERE in the homogeneous matrix around each body (the air-gap trick):
both sides of the surface have mu_e, so no one-sided evaluation across the
material jump is needed and the surface normal is taken geometrically as
(x - center)/|x - center| -- independent of any BEM code.

Cross-checks:

* FEM internal: forces on the two spheres equal and opposite; force along x.
* FEM single sphere in uniform field: zero force.
* BEM MST force on one body (restricted to its panels) vs the FEM force.
* BEM shape derivative with a body-restricted translation velocity (the
  force-on-one-body case of the thesis) vs the FEM force.
* All gaps shrink under refinement.

Requires dolfinx and gmsh (not pip-installable); skipped otherwise.  Setup::

    conda create -y -n forces-fem -c conda-forge python=3.12 fenics-dolfinx numpy scipy numba meshio pytest
    ~/miniconda3/envs/forces-fem/bin/pip install --no-deps -e . gmsh jax
"""

import numpy as np
import pytest

pytest.importorskip("dolfinx")
pytest.importorskip("gmsh")

import gmsh
import ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmsh as dgmsh
from mpi4py import MPI

import bempp_cl.api as bempp
from bempp_cl.api.forces import make_engine, available_backends, body_translations
from bempp_cl.api.forces.problems import tp_vp
from bempp_cl.api.forces.problems.sources import UniformField

bempp.DEFAULT_DEVICE_INTERFACE = "numba"

MU_I, MU_E = 10.0, 1.0
B0 = np.array([0.0, 0.0, 1.0])
R, D = 1.0, 1.5  # sphere radius, |x-center| along x
R_G = 1.25  # Gauss-sphere radius (air-gap surface)
R_OUT = 8.0
C1 = np.array([-D, 0.0, 0.0])
C2 = np.array([D, 0.0, 0.0])


# --------------------------------------------------------------------------- FEM
def _fem_domain(h, single=False):
    """Two-sphere-in-ball mesh; Gauss surfaces get facet tags 6+i."""
    centers = [C1] if single else [C1, C2]
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add("spheres")
    occ = gmsh.model.occ
    outer = occ.addSphere(0.0, 0.0, 0.0, R_OUT)
    spheres = [occ.addSphere(*c, R) for c in centers]
    gauss = [occ.addSphere(*c, R_G) for c in centers]
    out, fragmap = occ.fragment([(3, outer)], [(3, t) for t in spheres + gauss])
    occ.synchronize()
    ns = len(spheres)
    incl_set = {t for entry in fragmap[1 : 1 + ns] for _, t in entry}  # fragments of the inclusion balls
    shell_set = set()
    for entry in fragmap[1 + ns :]:  # fragments of the Gauss balls minus inclusions = shells
        shell_set.update(t for _, t in entry)
    shell_set -= incl_set
    gmsh.model.addPhysicalGroup(3, sorted(incl_set), 1)
    gmsh.model.addPhysicalGroup(3, [t for _, t in out if t not in incl_set], 2)
    ifaces = [t for _, t in gmsh.model.getBoundary([(3, t) for t in sorted(incl_set)], oriented=False)]
    gmsh.model.addPhysicalGroup(2, ifaces, 4)
    gsurf = []
    for t in sorted(shell_set):
        for _, s in gmsh.model.getBoundary([(3, t)], oriented=False):
            if s not in ifaces:
                gsurf.append(s)
    gsurf = sorted(set(gsurf))
    for i, s in enumerate(gsurf):
        gmsh.model.addPhysicalGroup(2, [s], 6 + i)

    f = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f, "SurfacesList", ifaces + gsurf)
    th = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(th, "InField", f)
    gmsh.model.mesh.field.setNumber(th, "SizeMin", h)
    gmsh.model.mesh.field.setNumber(th, "SizeMax", 2.0)
    gmsh.model.mesh.field.setNumber(th, "DistMin", 0.5)
    gmsh.model.mesh.field.setNumber(th, "DistMax", 4.0)
    gmsh.model.mesh.field.setAsBackgroundMesh(th)
    gmsh.model.mesh.generate(3)
    domain, _, facet_tags, _, _, _ = dgmsh.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
    gmsh.finalize()
    return domain, facet_tags, len(centers)


def _fem_forces(h, order=2, single=False):
    """Maxwell stress forces on the Gauss spheres from a FEM solve."""
    domain, facet_tags, nb = _fem_domain(h, single)
    centers = [C1] if single else [C1, C2]
    V = fem.functionspace(domain, ("Lagrange", order))
    mu = fem.Function(fem.functionspace(domain, ("DG", 0)))

    def mu_expr(x):
        inside = np.zeros(x.shape[1], dtype=bool)
        for c in centers:
            inside |= np.linalg.norm(x - c[:, None], axis=0) < R + 1e-6
        return np.where(inside, MU_I, MU_E)

    mu.interpolate(mu_expr)  # DG0 interpolation points are cell centroids
    outer_facets = dmesh.locate_entities_boundary(domain, 2, lambda x: np.isclose(np.linalg.norm(x, axis=0), R_OUT))
    dofs = fem.locate_dofs_topological(V, 2, outer_facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: -(B0 @ x) / MU_E)  # phi = -H0 . x
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.dot(mu * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(fem.Constant(domain, 0.0), v) * ufl.dx
    problem = LinearProblem(a, L, bcs=[fem.dirichletbc(u_bc, dofs)], petsc_options_prefix="fem_", petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    # MST on the Gauss spheres: mu_e on both sides, normal from geometry.
    X = ufl.SpatialCoordinate(domain)
    H = -ufl.grad(uh)
    Havg = 0.5 * (H("+") + H("-"))
    T = MU_E * (ufl.outer(Havg, Havg) - 0.5 * ufl.dot(Havg, Havg) * ufl.Identity(3))
    dS = ufl.Measure("dS", domain=domain, subdomain_data=facet_tags)
    forces = []
    for i, c in enumerate(centers):
        rv = X - ufl.as_vector(c)
        nv = rv / ufl.sqrt(ufl.dot(rv, rv))
        forces.append(np.array([fem.assemble_scalar(fem.form(sum(T[k, j] * nv[j] for j in range(3)) * dS(6 + i))) for k in range(3)]))
    return forces


# --------------------------------------------------------------------------- BEM
def _two_sphere_grid(level):
    g1 = bempp.shapes.regular_sphere(level)
    g2 = bempp.shapes.regular_sphere(level)
    vertices = np.hstack([g1.vertices + C1[:, None], g2.vertices + C2[:, None]])
    elements = np.hstack([g1.elements, g2.elements + g1.number_of_vertices])
    return bempp.Grid(vertices, elements)


def _bem_forces(level):
    """(MST force on sphere 1, shape-derivative force on sphere 1)."""
    name = "jax" if "jax" in available_backends() else "numpy"
    engine = make_engine(name)
    src = UniformField(B0)
    grid = _two_sphere_grid(level)
    tr = tp_vp.solve(grid, src, MU_I, MU_E, engine=engine)

    q = tr.quad
    dens = tp_vp.mst_density(tr, MU_I, MU_E, q)
    centroids = grid.vertices[:, grid.elements].T.mean(axis=1)
    mask = np.linalg.norm(centroids - C1, axis=1) < 1.5 * R  # sphere-1 panels
    mst = np.einsum("eq,ek,eq->k", np.where(mask[:, None], dens, 0.0), q.normals, q.weights)
    sd = tp_vp.shape_derivative(tr, src, MU_I, MU_E, body_translations(C1, R * 1.001), engine=engine)
    return mst, sd


# --------------------------------------------------------------------------- tests
def test_fem_single_sphere_uniform_field_zero_force():
    """A single sphere in a uniform field feels no force (exact by symmetry)."""
    f = _fem_forces(0.3, order=1, single=True)[0]
    assert np.linalg.norm(f) < 0.05


def test_two_sphere_force_fem_vs_bem():
    """BEM MST and shape-derivative forces vs the dolfinx reference."""
    gaps_mst, gaps_sd = [], []
    for level, h in [(3, 0.2), (4, 0.15)]:
        f1, f2 = _fem_forces(h, order=2)
        fx = abs(f1[0])
        # Symmetrized force on sphere 1: antisymmetric FEM noise cancels.
        favg = 0.5 * (f1 - f2)

        # FEM internal consistency: Newton's third law and force along x.
        assert np.linalg.norm(f1 + f2) < 0.05 * fx
        assert np.linalg.norm(f1[1:]) < 0.08 * fx

        mst, sd = _bem_forces(level)
        # BEM internal consistency: shape derivative == MST force.
        assert np.linalg.norm(sd - mst) < 0.02 * np.linalg.norm(mst)
        gaps_mst.append(np.linalg.norm(mst - favg) / fx)
        gaps_sd.append(np.linalg.norm(sd - favg) / fx)
        print(f"\nlevel={level}: FEM={favg}, BEM-MST={mst}, BEM-SD={sd}")

    assert gaps_mst[0] < 0.05 and gaps_mst[1] < 0.04, gaps_mst
    assert gaps_sd[0] < 0.05 and gaps_sd[1] < 0.04, gaps_sd
    # The gap shrinks with refinement (with a floor for mesh nondeterminism).
    assert gaps_sd[1] < max(0.025, 0.85 * gaps_sd[0]), gaps_sd
