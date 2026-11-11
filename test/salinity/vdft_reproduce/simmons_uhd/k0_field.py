"""Random hydraulic-conductivity field for the Simmons benchmark.

The paper triggers plume fingering with a random K0-field (Cremer & Graf 2015),
one value per element, generated with a correlation length of 0.0021 m in both
x and z -- deliberately SHORTER than the element size at every discretisation
level, so that the sampled semivariogram has range 0 ("Conceptual model").

A field whose correlation length is below the grid spacing is, on that grid,
indistinguishable from independent per-element noise.  So no geostatistical
generator is required: an i.i.d. draw per element reproduces the paper's field
exactly in distribution, and does so reproducibly from a seed.  The seed, not a
field file, is the reproducibility contract for a realisation.

INTERPRETATION OF sigma
-----------------------
Table 2 / the calibration section quote "a variance of sigma = 10^-2 [m^2 s^-2]"
against a calibrated mean K0 = 0.00112 m/s.  Read literally that is a standard
deviation of 0.1 m/s, ~90x the mean, which would put most of the field at
negative conductivity -- impossible.  Cremer & Graf (2015), which this paper
follows for the perturbation method, perturb ln K.  We therefore read sigma as
the variance of ln K0 and draw

    ln K0_e ~ N(ln K0_bar, sigma)     =>    K0_e = K0_bar * exp(Z),  Z~N(0,sigma)

which for the four cases' sigma of 1e-3 .. 1e-1 gives coefficients of variation
of 3% .. 32% -- physically sensible for a homogeneous sand pack, and matching
the modest fingering perturbation the paper describes.

Set ``lognormal=False`` to get the literal reading (additive Gaussian noise on
K0 itself, clipped positive) if that ever needs to be tested.

PARALLEL REPRODUCIBILITY
------------------------
The field is indexed by a GEOMETRIC global element name -- which rectangle of
the structured grid the element lies in, and which of that rectangle's two
triangles it is -- so a realisation is identical under any partitioning and any
number of ranks.  See ``geometric_index_for_mesh``.

Do NOT go back to ``elementNumbering_subdomain2global`` for this.  It is a
consistent global name within one run, but ``partitionMesh`` renumbers the
elements per partition, so it names a different element on a different rank
count: at level 0, all 20000 elements change number between 1, 4 and 88 ranks.

The index is also the element's MATERIAL TYPE in flow_p.  That has to be a
global name for a second, separate reason: Richards ghost-scatters the per-node
material type between ranks (Richards.py:1659, so that two ranks sharing a cut
edge pick the same rock and fL_ij = -fL_ji still holds), and the scatter
transmits the INDEX.  A rank-local index arrives on the neighbour as some
unrelated element's index, or as none at all, and the kernel reads the soil
tables out of bounds.
"""

import os

import numpy as np

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def generate(n_elements, k0_mean, sigma, seed, lognormal=True,
             preserve_mean=False):
    """Draw one realisation of the per-element K0 field.

    Parameters
    ----------
    n_elements : int
        Number of elements in the GLOBAL mesh.
    k0_mean : float
        Mean (geometric mean, when ``lognormal``) conductivity, m/s.
    sigma : float
        Variance of ln K0 (or of K0 itself when ``lognormal=False``).
    seed : int
        Realisation index.  The benchmark's 30 UHD realisations are seeds
        0..29.
    lognormal : bool
        See the module docstring.
    preserve_mean : bool
        If True, shift by -sigma/2 so the ARITHMETIC mean is ``k0_mean``
        rather than the geometric mean.  Off by default: the paper calibrates
        K0 as the field's central value, not its arithmetic mean.

    Returns
    -------
    (n_elements,) float64 array of hydraulic conductivities, m/s.
    """
    rng = np.random.default_rng(seed)
    if lognormal:
        z = rng.normal(0.0, np.sqrt(sigma), n_elements)
        if preserve_mean:
            z -= 0.5 * sigma
        k = k0_mean * np.exp(z)
    else:
        k = k0_mean + rng.normal(0.0, np.sqrt(sigma), n_elements)
        k = np.maximum(k, 1.0e-12)
    return k


def cache_path(n_elements, sigma, seed, lognormal=True):
    return os.path.join(
        _CACHE_DIR,
        "k0_n%d_sig%g_seed%d_%s.npy"
        % (n_elements, sigma, seed, "ln" if lognormal else "add"))


def generate_cached(n_elements, k0_mean, sigma, seed, lognormal=True,
                    preserve_mean=False):
    """``generate`` with an on-disk cache, so repeated runs of one realisation
    are bit-identical even if numpy's generator ever changes."""
    path = cache_path(n_elements, sigma, seed, lognormal)
    if os.path.exists(path):
        k = np.load(path)
        if k.shape == (n_elements,):
            return k
    k = generate(n_elements, k0_mean, sigma, seed, lognormal, preserve_mean)
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        np.save(path, k)
    except OSError:
        pass          # a read-only run directory is not a reason to fail
    return k


def geometric_index_for_mesh(mesh, nx, nz, L, x0=(0.0, 0.0)):
    """``(idx, n_global)``: a PARTITION-INDEPENDENT global name per element.

    ``elementNumbering_subdomain2global`` is a perfectly good global name
    WITHIN a run, but it is not the same name in two runs: ``partitionMesh``
    renumbers the elements so each rank owns a contiguous block, so the global
    number of the element at a given point changes with the rank count.
    Measured on this deck at level 0, all 20000 elements get a different global
    number under 1, 4 and 88 ranks.  Indexing the field by it would make
    ``seed`` name a different realisation on every rank count -- the seed is
    supposed to BE the reproducibility contract (see the module docstring), and
    two runs of "seed 0" at different np would not be comparable.

    The mesh here is the structured union-jack triangulation that
    ``generateTriangularMeshFromRectangularGrid`` builds from (nnx, nny), so an
    element has a name that owes nothing to the partition: the rectangle it
    lies in, plus which of that rectangle's two triangles it is.  Both come
    straight from the centroid.  Whichever way the cell's diagonal runs, the
    two centroids sit at different thirds of the cell in z -- (1/3,1/3) and
    (2/3,2/3) for "/", (2/3,1/3) and (1/3,2/3) for "\\" -- so the z-third alone
    separates them.

        idx = 2 * (j * nx + i) + (third_z - 1)        in [0, 2*nx*nz)

    ``mesh`` is the rank-local subdomain mesh; ghost elements are included.
    """
    # Straight from nodeArray, not from elementBarycentersArray: the barycentre
    # array is filled by computeGeometricInfo, which has not necessarily run by
    # the time initializeMesh is called.
    n_local = len(mesh.elementMaterialTypes)
    c = mesh.nodeArray[mesh.elementNodesArray[:n_local], :2].mean(axis=1)
    dx = (L[0] - x0[0]) / float(nx)
    dz = (L[1] - x0[1]) / float(nz)

    fx = (c[:, 0] - x0[0]) / dx
    fz = (c[:, 1] - x0[1]) / dz
    i = np.clip(np.floor(fx).astype(np.int64), 0, nx - 1)
    j = np.clip(np.floor(fz).astype(np.int64), 0, nz - 1)
    # thirds of the cell: a triangle centroid is at 1/3 or 2/3, never near 0 or
    # 1, so this rounding has half a third of slack on either side and cannot be
    # flipped by the floating-point reconstruction of the grid coordinates.
    third_z = np.rint(3.0 * (fz - j)).astype(np.int64)
    if not np.all((third_z == 1) | (third_z == 2)):
        bad = int((~((third_z == 1) | (third_z == 2))).sum())
        raise RuntimeError(
            "k0_field: %d of %d element centroids are not on a third of their "
            "cell in z -- this is not the structured union-jack grid the "
            "geometric element index assumes (nx=%d nz=%d, L=%r)."
            % (bad, c.shape[0], nx, nz, L))

    idx = 2 * (j * nx + i) + (third_z - 1)

    # A collision would silently give two elements the same K and leave a draw
    # unused.  Cheap to rule out locally; a cross-rank collision is impossible
    # once the map is injective on each rank, because it is a pure function of
    # the centroid.
    if np.unique(idx).size != idx.size:
        raise RuntimeError(
            "k0_field: geometric element index is not injective (%d elements, "
            "%d distinct indices)" % (idx.size, np.unique(idx).size))

    return idx, 2 * int(nx) * int(nz)


def global_index_for_mesh(mesh):
    """``(idx, n_global)``: the element's global number in THIS run.

    Superseded for field lookup by ``geometric_index_for_mesh``, which is also
    stable across rank counts; kept because it is the only thing that can check
    that the local->global map is present and spans the mesh, which is a useful
    parallel sanity probe in its own right.

    ``mesh`` is a rank-local subdomain mesh.  ``elementNumbering_subdomain2global``
    maps its elements onto the global numbering, which is what the field is
    indexed by; on one rank that map is the identity.  Ghost elements are
    included, so ``idx`` covers ``mesh.nElements_global`` (the LOCAL count --
    Proteus keeps that confusing name on subdomain meshes).

    """
    n_local = len(mesh.elementMaterialTypes)
    global_mesh = getattr(mesh, "globalMesh", None)

    # The local->global element map lives on the GLOBAL mesh, not on the
    # subdomain mesh: MeshTools.partitionMesh sets subdomainMesh.globalMesh =
    # self and has the partitioner fill *self*.elementNumbering_subdomain2global,
    # indexed by subdomain-local element number.  Every other consumer in
    # Proteus reads it that way (FemTools.py:2323 and friends).
    #
    # A subdomain mesh also carries an attribute of the same name, from
    # Mesh.__init__ -- but that one is the identity arange() built before
    # partitioning, so reading it silently yields local == global.  In serial
    # that is correct and invisible; in parallel it makes every rank draw the
    # same prefix field[0:n_local], so the same physical element gets a
    # different K on its owner and on the ranks that ghost it, and the "random"
    # field acquires the periodicity of the partition.  Prefer the global mesh.
    idx = None
    which = None
    for name, source in (("globalMesh", global_mesh), ("mesh", mesh)):
        g2l = getattr(source, "elementNumbering_subdomain2global", None)
        if g2l is None:
            continue
        g2l = np.asarray(g2l)
        if g2l.size == n_local and n_local > 0:
            idx, which = g2l, name
            break

    if idx is None:
        # Never silently correct-looking: in serial the identity IS the answer,
        # but on more than one rank it is a wrong field, not a different one.
        from mpi4py import MPI
        from proteus import Comm
        if Comm.get().size() > 1:
            raise RuntimeError(
                "k0_field: no usable elementNumbering_subdomain2global "
                "(n_local=%d); falling back to local==global would give each "
                "rank the same prefix of the field and make owned and ghosted "
                "copies of an element disagree." % n_local)
        idx = np.arange(n_local)
        which = "arange (serial)"

    n_global = getattr(global_mesh, "nElements_global", None)
    if n_global is None:
        n_global = max(int(idx.max()) + 1 if idx.size else 0, n_local)
    n_global = max(int(n_global), int(idx.max()) + 1 if idx.size else 0)

    # Verify the map actually spans the global element numbering.  A per-rank
    # "is it the identity?" test cannot do this: under contiguous element
    # partitioning rank 0's map legitimately IS arange(n_local), so that test
    # false-positives on exactly the rank whose logEvent output you can see
    # (see proteus_parallel_rank0_logevent_trap).  If the lookup had degraded
    # to local == global everywhere, the union of the local maps would only
    # reach max(n_local) - 1; a correct map reaches n_global - 1.
    from proteus import Comm
    from proteus.Profiling import logEvent
    comm = Comm.get()
    if comm.size() > 1:
        from mpi4py import MPI
        span = comm.comm.tompi4py().allreduce(int(idx.max()), op=MPI.MAX)
        logEvent("k0_field: %s map, global span %d of %d elements%s"
                 % (which, span + 1, n_global,
                    "" if span == n_global - 1 else "  <-- WRONG, rank-local"),
                 level=1)
        if span != n_global - 1:
            raise RuntimeError(
                "k0_field: element map spans only %d of %d global elements on "
                "%d ranks, so the K0 field is rank-local: owned and ghosted "
                "copies of an element would get different K."
                % (span + 1, n_global, comm.size()))

    return idx, int(n_global)


def global_field_for_mesh(mesh, k0_mean, sigma, seed, nx, nz, L, x0=(0.0, 0.0),
                          per_cell=False, **kwargs):
    """``(field_global, idx, n_global)`` -- the WHOLE realisation, plus this
    rank's element -> global-index map.

    Every rank holds the full ``n_global`` field.  That is not a new cost --
    ``generate_cached`` has always built the global array before slicing it --
    and it is what lets the deck index the soil-property tables by the GLOBAL
    index, which is the only index that means the same thing on two ranks.  At
    the benchmark's largest grid (level 4.5, 605k elements) the field is 4.8 MB
    per rank.

    ``global_index_for_mesh`` is still run in parallel, purely as the sanity
    probe it has always been: it asserts that the local->global element map is
    present and spans the mesh.  Its answer is not used for the lookup.

    ``per_cell`` draws one value per grid rectangle and gives it to both of the
    rectangle's triangles, matching the paper's one value per HGS brick.  The
    geometric index is 2*cell + (triangle), so repeating each draw twice lines
    the field up with it.  The cell field is cached under its own size
    (nx*nz), so it never collides with a per-triangle field.
    """
    from proteus import Comm
    if Comm.get().size() > 1:
        global_index_for_mesh(mesh)
    idx, n_global = geometric_index_for_mesh(mesh, nx, nz, L, x0)
    if per_cell:
        field = np.repeat(generate_cached(n_global // 2, k0_mean, sigma, seed,
                                          **kwargs), 2)
    else:
        field = generate_cached(n_global, k0_mean, sigma, seed, **kwargs)
    return field, idx, n_global


def field_for_mesh(mesh, k0_mean, sigma, seed, nx, nz, L, x0=(0.0, 0.0),
                   **kwargs):
    """Per-element K0 for THIS rank's elements, drawn from the global field."""
    field, idx, _ = global_field_for_mesh(mesh, k0_mean, sigma, seed,
                                          nx, nz, L, x0, **kwargs)
    return field[idx]


if __name__ == "__main__":
    import cases

    for name, case in sorted(cases.CASES.items()):
        for level in (0, 4):
            _, _, nx, nz = cases.spatial_level(level)
            n_el = 2 * nx * nz
            k = generate(n_el, cases.K0, case.sigma_lnK, seed=0)
            print("%s level=%d  n_el=%7d  K0: min=%.4e mean=%.4e max=%.4e "
                  "CV=%.1f%%" % (name, level, n_el, k.min(), k.mean(), k.max(),
                                 100.0 * k.std() / k.mean()))
