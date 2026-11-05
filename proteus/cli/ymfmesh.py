"""Make for a YMF spec's meshes: generate one only when it is missing or stale.

A spec's outputs on a box (``mesh: {cells: ...}``) are meshed in memory,
in parallel, as part of each solve; there are no files to go stale. An
output on any other region (``mesh: {h: ...}``) is meshed by triangle (2-D)
or tetgen (3-D), which is serial, and for a large mesh needs far more
memory than one rank of the run. So before anything is solved, the YMF
runner (``pro run SPEC.ymf``) brings each such mesh up to date here, on
rank 0, and the solves read it:

* **One mesh per region and element size,** named
  ``<stem>_mesh_<digest>`` in the output directory, the digest taken over
  the region and the generator's options. Outputs that differ only in
  their elements, their solver, their levels (uniform refinement of the
  mesh read) or their time step share it, as does a rerun.
* **Current or not** is :mod:`proteus.cli.mesh`'s manifest
  (``<prefix>.mesh.ymf``), as for a model's own mesh: a key over the
  generator, its command and its ``.poly``, and a checksum of each file it
  wrote.
* ``pro run SPEC.ymf --no-mesh`` stops before solving anything when a mesh
  is missing or stale, and ``pro msh SPEC.ymf`` makes them as a job of its
  own -- one process, on a large-memory node if need be.
"""
import hashlib
import json
import os

__all__ = ["needed", "bring_up_to_date"]


def _mesh_of(doc, problem, name, realization, outdir, stem):
    """``(prefix, domain, meshOptions, plan)`` for one output, or None on a box."""
    region = problem["domain"].get("region")
    if region is None:
        return None
    if region.get("kind") == "file":
        # a mesh file is read, not made; a geometry file (gmsh_geo) is meshed by the solve itself, with the
        # element size setup sets, not through the triangle/tetgen PSLG path below
        return None
    from proteus import ADRProblem, MeshTools
    from .ymfrun import run_arguments
    _, p, n, _ = ADRProblem.setup(problem, **run_arguments(doc, problem, name, realization))
    options = MeshTools.getMeshOptions(p.domain)
    digest = hashlib.sha256(json.dumps(
        {"region": problem["domain"]["region"], "options": options.triangleOptions},
        sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
    prefix = os.path.join(outdir, "%s_mesh_%s" % (stem, digest))
    p.domain.polyfile = prefix
    return prefix, p.domain, options, MeshTools.meshFiles(p.domain, options)


def needed(doc, problem, todo, outdir, stem):
    """``{prefix: {domain, options, plan, outputs}}``: the meshes that the
    outputs in ``todo`` (``(discretization, key, realization)`` triples)
    are solved on. Empty when every one is on a box."""
    meshes = {}
    for name, key, realization in todo:
        found = _mesh_of(doc, problem, name, realization, outdir, stem)
        if found is None:
            continue
        prefix, domain, options, plan = found
        m = meshes.setdefault(prefix, {"domain": domain, "options": options, "plan": plan,
                                       "outputs": []})
        m["outputs"].append(key)
    return meshes


def bring_up_to_date(meshes, comm, no_mesh=False, force=False, remedy=None, log=print):
    """Make each mesh that is missing or stale, on rank 0; return an error or None.

    With ``no_mesh`` nothing is made: the error names what is missing and
    ``remedy``, the command that makes it. With ``force`` every mesh is made.
    """
    from . import mesh
    error = None
    if comm.isMaster():
        missing = []
        for prefix, m in sorted(meshes.items()):
            # the generator's input, written from the spec: part of the key
            m["domain"].writePoly(prefix)
            state, reason = mesh.state(m["plan"])
            name = os.path.basename(prefix)
            if state == "current" and not force:
                log("mesh %s: up to date" % (name,))
            elif no_mesh:
                missing.append("%s (%s; for %s)" % (name, reason or state,
                                                    ", ".join(o.split("/")[-1]
                                                              for o in m["outputs"])))
            else:
                log("mesh %s: %s; running %s" % (name, reason or "forced",
                                                 m["plan"]["command"]))
                mesh.make(m["plan"], m["domain"], m["options"])
        if missing:
            error = ("%s missing or stale, and --no-mesh was given:\n  %s"
                     % ("a mesh is" if len(missing) == 1 else "%d meshes are" % len(missing),
                        "\n  ".join(missing)))
            if remedy:
                error += "\nmake %s with: %s" % ("it" if len(missing) == 1 else "them", remedy)
    if comm.size() > 1:
        error = comm.comm.tompi4py().bcast(error, root=0)
    return error
