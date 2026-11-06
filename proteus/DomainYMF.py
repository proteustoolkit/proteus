"""A proteus ``Domain`` <-> a YMF domain block.

A YMF domain says what *kind* of region it is -- a box, a boundary
representation, an already-discretized mesh, or a reference to a file in
another language -- and Proteus's fourteen Domain classes fall into exactly
those four. See ``ymf.domain`` for the format and the reasoning.

What is translated is the region and how its boundary is named. Two things
Proteus's Domain classes also carry are deliberately not:

* **Meshing policy** -- everything on the object
  ``MeshTools.getMeshOptions(domain)`` returns: ``he``, ``use_gmsh``,
  ``use_plex``, ``regionConstraints``, partitioning, refinement levels. How a
  region becomes a mesh is not what the region is, and a YMF spec states it
  under ``discretizations[].mesh``.
* **Runtime state** -- ``shape_list``, ``auxiliaryVariables``,
  ``useSpatialTools``, ``BC``/``BCbyFlag``. Not data, and not round-trippable.

So ``to_ymf`` is lossy by design with respect to the *object*, and lossless
with respect to the *domain*. Where a class carries region state the format
cannot hold, it raises rather than quietly dropping it.
"""

from proteus import Domain

__all__ = ["DomainTranslationError", "to_ymf", "from_ymf"]


class DomainTranslationError(ValueError):
    """A domain cannot be translated without losing part of the region."""


def _plain(value):
    """numpy arrays and tuples -> lists of plain numbers, for YAML."""
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "item"):           # numpy scalar
        return value.item()
    return value


def _shared(domain):
    block = {"name": domain.name, "units": domain.units}
    if getattr(domain, "boundaryTags", None):
        block["boundary_tags"] = dict(domain.boundaryTags)
    return block


def _regions(domain):
    points = _plain(getattr(domain, "regions", None)) or []
    flags = _plain(getattr(domain, "regionFlags", None)) or []
    out = []
    for i, point in enumerate(points):
        entry = {"point": point}
        if i < len(flags):
            entry["flag"] = int(flags[i])
        out.append(entry)
    return out


def _uses_gmsh(domain):
    from proteus import MeshTools
    return bool(MeshTools.getMeshOptions(domain).use_gmsh)


def _piecewise_linear(domain, dimension):
    block = dict(_shared(domain), kind="piecewise_linear", dimension=dimension)
    block["vertices"] = _plain(domain.vertices)
    for attr, key in (("segments", "segments"), ("facets", "facets"),
                      ("volumes", "volumes"), ("holes", "holes")):
        value = _plain(getattr(domain, attr, None))
        if value:
            block[key] = value
    for attr, key in (("vertexFlags", "vertex_flags"),
                      ("segmentFlags", "segment_flags"),
                      ("facetFlags", "facet_flags")):
        value = _plain(getattr(domain, attr, None))
        if value:
            block[key] = [int(v) for v in value]
    regions = _regions(domain)
    if regions:
        block["regions"] = regions
    return block


#: Classes that are a path and a format rather than geometry, by format name.
_FILE_FORMATS = {
    "MeshHexDomain": "hex",
    "MeshTetgenDomain": "tetgen",
    "MeshYmfDomain": "ymf",
}

#: Classes holding a live object, which no document can carry.
_NOT_DATA = {
    "PUMIDomain": "an AdaptManager and checkpoint state",
    "DMPlexDomain": "a PETSc DMPlex object",
}


def to_ymf(domain):
    """Translate a proteus ``Domain`` into a YMF domain block."""
    kind = type(domain).__name__

    if kind in _NOT_DATA:
        raise DomainTranslationError(
            "%s holds %s, which is a live object rather than a description of "
            "a region; there is nothing to write. Give the domain it was built "
            "from instead." % (kind, _NOT_DATA[kind]))

    if isinstance(domain, Domain.InterpolatedBathymetryDomain):
        raise DomainTranslationError(
            "InterpolatedBathymetryDomain is a boundary representation with a "
            "field sampled over its interior -- a PSLG and an overlapping "
            "discrete field, both of which are the domain. A YMF domain is "
            "one kind at a time and has nowhere to put the second, so writing "
            "this as kind: piecewise_linear would drop the bathymetry, which "
            "is the whole of what the class adds.")

    if isinstance(domain, Domain.RectangularDomain):
        lower = [float(v) for v in domain.x[:domain.nd]]
        upper = [float(a + b) for a, b in zip(lower, domain.L[:domain.nd])]
        return dict(_shared(domain), kind="box", dimension=domain.nd,
                    lower=lower, upper=upper)

    if isinstance(domain, Domain.TriangulatedSurfaceDomain):
        # already a mesh: the archive's own vocabulary says it
        triangles = _plain(domain.triangles)
        return dict(_shared(domain), kind="discrete", dimension=3,
                    topology={"type": "Triangle",
                              "number_of_elements": len(triangles),
                              "values": triangles},
                    geometry={"type": "XYZ", "values": _plain(domain.nodes)})

    if isinstance(domain, Domain.GMSH_3D_Domain):
        if domain.length_scale != 1.0 or list(domain.permute_dims) != [0, 1, 2]:
            raise DomainTranslationError(
                "this GMSH_3D_Domain rescales by %r and permutes its axes to "
                "%r, which is how the .geo file's coordinates map into the "
                "domain -- description, not meshing. A YMF kind: file block "
                "carries only format and path, so it cannot say it."
                % (domain.length_scale, list(domain.permute_dims)))
        return dict(_shared(domain), kind="file", dimension=domain.nd,
                    format="gmsh_geo", path=str(domain.geofile))

    if isinstance(domain, Domain.MeshYmfDomain) and domain.grid is not None:
        # a mesh held in memory is a discrete domain: the block it came from
        from proteus.MeshTools import YMF_GRID_KEYS
        return dict(_shared(domain), kind="discrete", dimension=domain.nd,
                    **dict((k, domain.grid[k]) for k in YMF_GRID_KEYS
                           if k in domain.grid and k != "name"))

    if kind in _FILE_FORMATS:
        return dict(_shared(domain), kind="file", dimension=domain.nd,
                    format=_FILE_FORMATS[kind], path=str(domain.meshfile))

    # a PSLG/PLC that gmsh meshes from its .geo file and holds no geometry itself is that file
    if (isinstance(domain, (Domain.PiecewiseLinearComplexDomain, Domain.PlanarStraightLineGraphDomain))
            and getattr(domain, "geofile", None) and not domain.vertices
            and _uses_gmsh(domain)):
        return dict(_shared(domain), kind="file", dimension=domain.nd,
                    format="gmsh_geo", path=str(domain.geofile) + ".geo")

    # ExtrudedTriangulation is a PLC once built, so it writes as one; how it
    # was constructed is not part of the region.
    if isinstance(domain, Domain.PiecewiseLinearComplexDomain):
        return _piecewise_linear(domain, 3)
    if isinstance(domain, Domain.PlanarStraightLineGraphDomain):
        return _piecewise_linear(domain, 2)

    raise DomainTranslationError(
        "no YMF translation for %s; it is not a box, a boundary "
        "representation, a mesh or a file reference" % (kind,))


_FILE_CLASSES = {v: getattr(Domain, k) for k, v in _FILE_FORMATS.items()}


def _is_surface(block):
    """Whether a ``discrete`` block is a triangulated surface rather than a mesh to solve on.

    Triangles in three coordinates with ``dimension: 3`` are a surface (what
    :class:`Domain.TriangulatedSurfaceDomain` writes); triangles with
    ``dimension: 2`` or ``XY`` geometry are a planar mesh, as are tetrahedra.
    """
    if block["topology"].get("type") != "Triangle":
        return False
    if block.get("dimension") is not None:
        return int(block["dimension"]) == 3
    return block["geometry"].get("type", "XYZ") != "XY"


def from_ymf(block):
    """Build a proteus ``Domain`` from a YMF domain block."""
    from ymf.domain import validate_domain_block

    validate_domain_block(block)
    kind = block["kind"]
    name = block.get("name", "ymf_domain")
    units = block.get("units", "m")

    if kind == "box":
        lower, upper = block["lower"], block["upper"]
        domain = Domain.RectangularDomain(
            L=[float(b - a) for a, b in zip(lower, upper)],
            x=[float(a) for a in lower], name=name, units=units)
        if block.get("boundary_tags"):
            domain.boundaryTags = dict(block["boundary_tags"])
        return domain

    if kind == "file":
        fmt = block["format"]
        if fmt == "gmsh_geo":
            # the constructor appends the suffix it was given
            path = block["path"]
            base = path[:-4] if path.endswith(".geo") else path
            # a PSLG (2D) or PLC (3D) that holds no geometry of its own: its .geo file is the geometry and
            # gmsh meshes it (C7.1 step J; GMSH_3D_Domain, which needed gmsh2tetgen, is deprecated)
            from proteus import MeshTools
            nd = int(block.get("dimension", 3))
            if nd not in (2, 3):
                raise DomainTranslationError(
                    "a gmsh_geo block of dimension %r: gmsh geometries are 2D or 3D" % (nd,))
            domain = (Domain.PlanarStraightLineGraphDomain if nd == 2
                      else Domain.PiecewiseLinearComplexDomain)(name=name, units=units)
            domain.geofile = base
            MeshTools.getMeshOptions(domain).use_gmsh = True
            if block.get("boundary_tags"):
                domain.boundaryTags = dict(block["boundary_tags"])
            return domain
        if fmt not in _FILE_CLASSES:
            raise DomainTranslationError(
                "no proteus Domain reads format %r; known: %s"
                % (fmt, ", ".join(sorted(_FILE_CLASSES))))
        if fmt == "ymf":
            # the block says the dimension; the mesh file need not be at hand
            return Domain.MeshYmfDomain(block["path"], nd=block.get("dimension"))
        return _FILE_CLASSES[fmt](block["path"])

    if kind == "discrete":
        topology, geometry = block["topology"], block["geometry"]
        if _is_surface(block):
            return Domain.TriangulatedSurfaceDomain(
                geometry["values"], topology["values"], name=name, units=units)
        if topology.get("type") not in ("Triangle", "Tetrahedron"):
            raise DomainTranslationError(
                "proteus solves on a discrete mesh of triangles or tetrahedra; "
                "this one is %r" % (topology.get("type"),))
        # a mesh is a domain (D4): read in memory, as a ymf mesh file would be
        return Domain.MeshYmfDomain(
            name, nd=2 if topology["type"] == "Triangle" else 3, grid=block)

    # piecewise_linear
    common = dict(vertices=block["vertices"], name=name, units=units,
                  holes=block.get("holes"),
                  regions=[r["point"] for r in block.get("regions", [])] or None,
                  regionFlags=[r["flag"] for r in block.get("regions", [])
                               if "flag" in r] or None,
                  vertexFlags=block.get("vertex_flags"))
    if block.get("dimension", 2) == 3:
        domain = Domain.PiecewiseLinearComplexDomain(
            facets=block.get("facets"), facetFlags=block.get("facet_flags"),
            **common)
    else:
        domain = Domain.PlanarStraightLineGraphDomain(
            segments=block.get("segments"),
            segmentFlags=block.get("segment_flags"), **common)
    if block.get("boundary_tags"):
        domain.boundaryTags = dict(block["boundary_tags"])
    return domain
