"""Geometry builder: electrostatic shield with conductors inside a dielectric.

A grounded spherical shield (Dirichlet phi = 0) contains conducting balls
carrying net charge, embedded in a dielectric with an arbitrary energy
density.  Each conductor is surrounded by a Gauss sphere of radius
``gauss_factor`` times the conductor radius, lying entirely in the dielectric:
charge, field and stress integrals are evaluated there with the geometric
normal ``(x - center)/|x - center|``, independent of facet orientation and of
the material jump at the conductor surface.

Facet tags: conductor ``i`` surface -> ``2 + i``; Gauss sphere ``i`` ->
``10 + i``.  Volume tags: dielectric pieces -> 1, conductor interiors -> 2
(contributions there are trivial since E = 0 for w with w(0) = 0).
"""

import gmsh
import numpy as np
from dolfinx.io import gmsh as dgmsh
from mpi4py import MPI


class Geometry:
    """A meshed shield + conductors configuration."""

    def __init__(self, domain, facet_tags, conductors, gauss_factor, shield_radius):
        self.domain = domain
        self.facet_tags = facet_tags
        self.conductors = [(np.asarray(c, dtype=float), float(r)) for c, r in conductors]
        self.gauss_factor = gauss_factor
        self.shield_radius = shield_radius

    @property
    def n_conductors(self):
        return len(self.conductors)


def build_geometry(conductors, shield_radius=3.0, gauss_factor=1.3, h_near=0.15, h_far=1.0, dist_min=None, dist_max=None):
    """Build a Geometry: spherical shield + balls ``((center, radius), ...)``.

    ``h_near`` is the element size on conductor/Gauss surfaces, ``h_far`` the
    far-field size.  The size field is h_near within ``dist_min`` (default
    ``0.25 * shield_radius``) of the surfaces and grades to ``h_far`` at
    ``dist_max`` (default ``0.8 * shield_radius``).
    """
    conductors = [((np.asarray(c, dtype=float)), float(r)) for c, r in conductors]
    if dist_min is None:
        dist_min = 0.25 * shield_radius
    if dist_max is None:
        dist_max = 0.8 * shield_radius
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", 1)
        gmsh.model.add("shield")
        occ = gmsh.model.occ
        outer = occ.addSphere(0.0, 0.0, 0.0, shield_radius)
        balls = [occ.addSphere(*c, r) for c, r in conductors]
        gauss = [occ.addSphere(*c, gauss_factor * r) for c, r in conductors]
        out, fragmap = occ.fragment([(3, outer)], [(3, t) for t in balls + gauss])
        occ.synchronize()
        cond_set = set()
        for entry in fragmap[1 : 1 + len(balls)]:  # fragments of the conductor balls
            cond_set.update(t for _, t in entry)
        shell_set = set()
        for entry in fragmap[1 + len(balls) :]:  # fragments of the Gauss balls
            shell_set.update(t for _, t in entry)
        shell_set -= cond_set

        gmsh.model.addPhysicalGroup(3, sorted(cond_set), 2)
        gmsh.model.addPhysicalGroup(3, [t for _, t in out if t not in cond_set], 1)
        ifaces = [t for _, t in gmsh.model.getBoundary([(3, t) for t in sorted(cond_set)], oriented=False)]
        for i, s in enumerate(ifaces):
            gmsh.model.addPhysicalGroup(2, [s], 2 + i)
        gsurf = []
        for t in sorted(shell_set):
            for _, s in gmsh.model.getBoundary([(3, t)], oriented=False):
                if s not in ifaces:
                    gsurf.append(s)
        gsurf = sorted(set(gsurf))
        for i, s in enumerate(gsurf):
            gmsh.model.addPhysicalGroup(2, [s], 10 + i)

        f = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(f, "SurfacesList", ifaces + gsurf)
        th = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(th, "InField", f)
        gmsh.model.mesh.field.setNumber(th, "SizeMin", h_near)
        gmsh.model.mesh.field.setNumber(th, "SizeMax", h_far)
        gmsh.model.mesh.field.setNumber(th, "DistMin", dist_min)
        gmsh.model.mesh.field.setNumber(th, "DistMax", dist_max)
        gmsh.model.mesh.field.setAsBackgroundMesh(th)
        gmsh.model.mesh.generate(3)
        domain, _, facet_tags, _, _, _ = dgmsh.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
    finally:
        gmsh.finalize()
    return Geometry(domain, facet_tags, conductors, gauss_factor, shield_radius)
