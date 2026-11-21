"""C5 step 2: a split discretization runs from its dispatch through ymf_run, and the run is the deck's, bitwise.

Each case two ways, each in a directory of its own:

* the deck through parun: ``parun --TwoPhaseFlow damBreakLevelSet.py -C "final_time=0.03 dt_output=0.01 he=0.08
  genMesh=True"``, ``parun --SWEs dam3Bumps.py -C "final_time=0.5 dt_output=0.1 structured=False he=2.0 dam_x_location=16.05"``;
* the spec through ``pro run``: proteus.Dispatch maps level_set_five onto RANS2P, VOF, NCLS, RDLS and MCorr, and
  convex_limited onto SW2DCV, and proteus.DispatchRun builds the TwoPhaseFlowProblem or SWFlowProblem from that and
  from ymf's split plain data.

Every numeric array the two archives hold (every model's unknowns at every output time, the mesh, the material
types) must be bitwise equal, on one rank and on two (the partition pinned by ``-mat_partitioning_type current``).
The mesh records may differ only in the name of their HDF5 file.

dambreak_short.ymf writes the deck's phi_init with hypot, which ymf's notation has since cekees/ymf#12; with an
older ymf the dam-break case fails to parse, and should: it cannot match the deck without it.
"""
import os
import shutil
import subprocess
import sys

import pytest

pytest.importorskip("ymf.symbolic.split")
h5py = pytest.importorskip("h5py")
numpy = pytest.importorskip("numpy")

HERE = os.path.dirname(os.path.abspath(__file__))
DECKS = os.path.join(os.path.dirname(HERE), "TwoPhaseFlow")


def _mpiexec():
    beside = os.path.join(os.path.dirname(sys.executable), "mpiexec")
    return beside if os.path.exists(beside) else shutil.which("mpiexec")


MPIEXEC = _mpiexec()


def _run(command, cwd, ranks):
    env = dict(os.environ, OMP_NUM_THREADS="1", PETSC_OPTIONS="-mat_partitioning_type current")
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    if ranks > 1:
        command = [MPIEXEC, "-n", str(ranks)] + command
    ran = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True, timeout=600, env=env)
    return ran


def _arrays(path, stem):
    """{dataset: array} of an archive; a mesh record (text) with its file's name replaced by a placeholder."""
    out = {}
    with h5py.File(str(path), "r") as f:
        def visit(key, item):
            if not isinstance(item, h5py.Dataset):
                return
            value = item[()]
            if isinstance(value, numpy.ndarray) and value.dtype.kind in "SOU":
                value = numpy.array([v.replace(stem.encode() + b".h5", b"<archive>.h5") for v in value])
            elif isinstance(value, bytes):
                value = value.replace(stem.encode() + b".h5", b"<archive>.h5")
            out[key] = value
        f.visititems(visit)
    return out


def _parun():
    found = shutil.which("parun", path=os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", ""))
    return [found] if found else [sys.executable, os.path.join(os.path.dirname(os.path.dirname(HERE)), "scripts",
                                                               "parun")]


#: (deck directory, deck, parun's flag, its options, spec, what ymf_run says it dispatched to, a field that moves)
CASES = {
    "damBreakLevelSet": ("TwoPhaseFlow", "damBreakLevelSet.py", "--TwoPhaseFlow",
                         "final_time=0.03 dt_output=0.01 he=0.08 genMesh=True", "dambreak_short.ymf",
                         "level_set_five -- split, dispatched to flow RANS2P, vof VOF, ncls NCLS, rdls RDLS, "
                         "mcorr MCorr", "velocity_t4"),
    # the same deck with interface=1: RANS2P and CLSVOF. The spec's atmosphere (p = 0, no flux, φ signed + where flow
    # enters) is setAtmosphere, which sets no CLSVOF data; the deck has the same boundary
    "damBreakLevelSet_clsvof": ("TwoPhaseFlow", "damBreakLevelSet.py", "--TwoPhaseFlow",
                                "final_time=0.03 dt_output=0.01 he=0.08 genMesh=True interface=1",
                                "dambreak_short_clsvof.ymf",
                                "level_set_clsvof -- split, dispatched to flow RANS2P, clsvof CLSVOF", "velocity_t4"),
    # the projection scheme: CLSVOF, RANS3PF, PresInc, Pres, PresInit (damBreak.py), the spec's atmosphere again
    # setAtmosphere. The deck's initial level set is its characteristic function (disc_ICs), the spec's a distance
    "damBreak_projection": ("TwoPhaseFlow", "damBreak.py", "--TwoPhaseFlow",
                            "final_time=0.03 dt_output=0.01 he=0.08 genMesh=True", "dambreak_short_projection.ymf",
                            "clsvof_projection -- split, dispatched to clsvof CLSVOF, flow RANS3PF, pressureInc "
                            "PresInc, pressure Pres, pressureInit PresInit", "velocity_t4"),
    # the same dam break on the stored mesh of test/TwoPhaseFlow (meshDambreak: the mesh of he=0.08) read as the deck reads it
    # with genMesh=False; a mesh that is already made has no h, the run measures its longest edge (STORED_HE) and
    # the deck is given the same he
    "damBreakLevelSet_stored": ("TwoPhaseFlow", "damBreakLevelSet.py", "--TwoPhaseFlow",
                                "final_time=0.03 dt_output=0.01 he={he} genMesh=False", "dambreak_short_stored.ymf",
                                "level_set_five -- split, dispatched to flow RANS2P, vof VOF, ncls NCLS, rdls RDLS, "
                                "mcorr MCorr", "velocity_t4"),
    # the gauges damBreak.py attaches (level-set height along lines on CLSVOF, pressure at points on Pres), as `gauges:`
    "damBreak_projection_gauges": ("TwoPhaseFlow", "damBreak.py", "--TwoPhaseFlow",
                                   "final_time=0.03 dt_output=0.01 he=0.08 genMesh=True",
                                   "dambreak_short_projection_gauges.ymf",
                                   "clsvof_projection -- split, dispatched to clsvof CLSVOF, flow RANS3PF, pressureInc "
                                   "PresInc, pressure Pres, pressureInit PresInit", "velocity_t4"),
    # the 3-D tank: Tank3D, TetGen at h = 0.4 (volume h^3/6), gravity along -z, the top open; the deck's iterative defaults
    "damBreakLevelSet3D": ("TwoPhaseFlow", "damBreakLevelSet3D.py", "--TwoPhaseFlow",
                           "final_time=0.03 dt_output=0.01 he=0.4", "dambreak3d_short.ymf",
                           "level_set_five -- split, dispatched to flow RANS2P, vof VOF, ncls NCLS, rdls RDLS, "
                           "mcorr MCorr", "velocity_t4"),
    # the dispersive model (sw_model 1, GN_SW2DCV: h, hu, hv and the auxiliaries h eta, h w, h beta) on the same basin
    "circular_dam_dispersive": ("SWFlow", "circular_damBreak.py", "--SWEs",
                                "sw_model=1 final_time=2.0 dt_output=0.4", "circular_dam_dispersive_short.ymf",
                                "relaxed_convex_limited -- split, dispatched to gn GN_SW2DCV", "h_u_t6"),
    # structured meshes with a cell count per axis: the deck's refinement 4 is nnx = 81, nny = 41 nodes (80 x 40 cells)
    "dam3Bumps_structured": ("SWFlow", "dam3Bumps.py", "--SWEs",
                             "final_time=0.5 dt_output=0.1 structured=True dam_x_location=16.05",
                             "dam3bumps_structured.ymf", "convex_limited -- split, dispatched to swe SW2DCV",
                             "h_u_t6"),
    # the dam3Bumps spec on the mesh of the deck's own run (dam3bumps.node, .ele ... in the deck's directory): read with
    # genMesh off, it must give the run of the mesh it was made by
    "dam3Bumps_stored": ("SWFlow", "dam3Bumps.py", "--SWEs",
                         "final_time=0.5 dt_output=0.1 structured=False he=2.0 dam_x_location=16.05",
                         "dam3bumps_stored.ymf", "convex_limited -- split, dispatched to swe SW2DCV", "h_u_t6"),
    "dam3Bumps": ("SWFlow", "dam3Bumps.py", "--SWEs", "final_time=0.5 dt_output=0.1 structured=False he=2.0 dam_x_location=16.05",
                  "dam3bumps.ymf", "convex_limited -- split, dispatched to swe SW2DCV", "h_u_t6"),
    # the same dam break with open (outflow) boundaries: SWFlow's reflectingBCs off, no Dirichlet data; the
    # rarefaction leaves through x = 0 within the five seconds
    "dam3Bumps_open": ("SWFlow", "dam3Bumps.py", "--SWEs",
                       "final_time=5.0 dt_output=1.0 structured=False he=2.0 dam_x_location=16.05 "
                       "reflecting_BCs=False", "dam3bumps_open.ymf", "convex_limited -- split, dispatched to swe SW2DCV",
                       "h_u_t6"),
    # SWFlow's structured mesh (mesh: {cells: 49} is the deck's nnx = nny = 50) on a flat bed with walls
    "circular_dam_structured": ("SWFlow", "circular_damBreak.py", "--SWEs",
                                "sw_model=0 final_time=2.0 dt_output=0.4", "circular_dam.ymf",
                                "convex_limited -- split, dispatched to swe SW2DCV", "h_u_t6"),
}


def _params():
    return [pytest.param(case, ranks, id="%s-%d" % (case, ranks)) for case in sorted(CASES) for ranks in (1, 2)]


@pytest.mark.parametrize("case, ranks", _params())
def test_the_spec_runs_the_deck_bitwise(tmp_path, case, ranks):
    if ranks > 1 and MPIEXEC is None:
        pytest.skip("mpiexec not available")
    folder, script, flag, options, spec_file, banner, moving = CASES[case]
    deck, spec = tmp_path / "deck", tmp_path / "spec"
    deck.mkdir()
    spec.mkdir()
    # the deck meshes in (or beside) its own directory: a copy, so the stored mesh files stay as they are
    shutil.copy(os.path.join(os.path.dirname(HERE), folder, script), str(deck))
    spec_path = os.path.join(HERE, spec_file)
    if "{he}" in options:
        # a stored mesh: the deck reads meshDambreak beside itself, the spec names it by path (MESH_PREFIX), and the
        # deck's he is the element size the run measures on the mesh (its longest edge)
        from proteus.DispatchRun import stored_element_size
        stored = os.path.join(DECKS, "meshDambreak")
        for suffix in (".node", ".ele", ".edge", ".poly"):
            shutil.copy(stored + suffix, str(deck / ("meshDambreak" + suffix)))
        measured = stored_element_size(stored)
        he = measured / 3.22
        while 3.22 * he != measured and 3.22 * he < measured * (1 + 1e-9):      # the deck's he is 3.22 * opts.he
            he = numpy.nextafter(he, 1.0)
        options = options.replace("{he}", repr(float(he)))
        for name in os.listdir(HERE):
            if name.endswith(".ymf"):
                shutil.copy(os.path.join(HERE, name), str(spec / name))
        text = (spec / spec_file).read_text().replace("MESH_PREFIX", stored)
        (spec / spec_file).write_text(text)
        spec_path = str(spec / spec_file)
    ran = _run(_parun() + [flag, script, "-l1", "-C", options], deck, ranks)
    assert ran.returncode == 0, ran.stdout[-3000:] + ran.stderr[-3000:]
    if "{he}" not in CASES[case][3] and case.endswith("_stored"):
        # the mesh the deck just made, read by the spec
        for name in os.listdir(HERE):
            if name.endswith(".ymf"):
                shutil.copy(os.path.join(HERE, name), str(spec / name))
        text = (spec / spec_file).read_text().replace("MESH_PREFIX", str(deck / "dam3bumps"))
        (spec / spec_file).write_text(text)
        spec_path = str(spec / spec_file)
    ran = _run([sys.executable, "-m", "proteus.cli", "run", spec_path, "--outdir", str(spec)],
               spec, ranks)
    assert ran.returncode == 0, ran.stdout[-3000:] + ran.stderr[-3000:]
    assert banner in ran.stdout
    if case.endswith("_stored"):
        # the run's record says the mesh was read, not made (bitwise equal arrays cannot tell: it is the same mesh)
        (record,) = [f for f in os.listdir(str(spec)) if f.endswith(".archive.ymf")]
        assert "mesh: stored " in (spec / record).read_text()
    (archive,) = [f for f in os.listdir(str(spec)) if f.endswith(".h5")]
    expected = _arrays(deck / (script[:-3] + ".h5"), script[:-3])
    found = _arrays(spec / archive, archive[:-3])
    assert sorted(found) == sorted(expected)
    assert float(numpy.abs(found[moving]).max()) > 0.1          # the water moved
    if case.endswith("_gauges"):
        # the gauge files, written by the same Gauges objects: equal as text, and not empty
        for gauge_file in ("height1.csv", "height2.csv", "pressure.csv"):
            theirs, ours = (deck / gauge_file).read_text(), (spec / gauge_file).read_text()
            assert theirs.count("\n") > 3, gauge_file
            assert ours == theirs, gauge_file
    different = [key for key in expected if not numpy.array_equal(found[key], expected[key])]
    assert not different, "%d of %d datasets differ: %s" % (len(different), len(expected), different[:10])


def _stored_split(tmp_path, mutate=None):
    """ymf's split of the stored-mesh dam break (MESH_PREFIX replaced), for the realizer's data-only parts."""
    from ymf.closure import load, plain
    from ymf.symbolic.split import split_problem
    for name in os.listdir(HERE):
        if name.endswith(".ymf"):
            shutil.copy(os.path.join(HERE, name), str(tmp_path / name))
    spec = tmp_path / "dambreak_short_stored.ymf"
    spec.write_text(spec.read_text().replace("MESH_PREFIX", os.path.join(DECKS, "meshDambreak")))
    doc, _, _ = load(str(spec))
    doc = plain(doc)
    if mutate:
        mutate(doc)
    return doc, split_problem(doc, "level_set_five")


def test_a_stored_mesh_is_read_not_made(tmp_path):
    """The file domain's tank: genMesh off, the polyfile the stored prefix, the element size measured on the mesh,
    each side's setter chosen from the flags the mesh stores (walls = bottom, right, left; top = atmosphere)."""
    from proteus.DispatchRun import _tank, stored_element_size, tank_sides

    class Comm(object):
        def isMaster(self):
            raise AssertionError("a stored mesh writes no PSLG")

        def barrier(self):
            pass
    _, split = _stored_split(tmp_path)
    sides = tank_sides(split)
    assert {s: v[1] for s, v in sides.items()} == {"y+": "setAtmosphere", "x+": "setFreeSlip", "x-": "setFreeSlip",
                                                    "y-": "setFreeSlip"}
    assert {s: v[0] for s, v in sides.items()} == {"y+": "top", "x+": "walls", "x-": "walls", "y-": "walls"}
    domain, _ = _tank(split, stored_element_size(os.path.join(DECKS, "meshDambreak")), "x", str(tmp_path), Comm())
    assert domain.MeshOptions.genMesh is False
    assert domain.polyfile == os.path.join(DECKS, "meshDambreak")
    assert abs(stored_element_size(os.path.join(DECKS, "meshDambreak")) - 0.45) < 1e-12


def test_a_stored_mesh_whose_file_is_missing_is_refused(tmp_path):
    from proteus.DispatchRun import RealizeError, _tank

    def move(doc):
        doc["model"]["strong_form"]["domain"]["path"] = str(tmp_path / "nowhere")
    _, split = _stored_split(tmp_path, move)
    with pytest.raises(RealizeError, match="does not exist"):
        _tank(split, 0.1, "x", str(tmp_path), None)


def test_a_stored_mesh_in_another_format_is_refused(tmp_path):
    from proteus.DispatchRun import RealizeError, tank_sides

    def other(doc):
        doc["model"]["strong_form"]["domain"]["format"] = "hex"
    _, split = _stored_split(tmp_path, other)
    with pytest.raises(RealizeError, match="stored triangle/tetgen mesh"):
        tank_sides(split)


def test_a_relaxation_auxiliary_starts_from_the_depth_it_is_an_expression_of():
    from proteus.DispatchRun import RealizeError, _AuxiliaryInitial

    class Depth(object):
        def uOfXT(self, X, t):
            return 1.5 + X[0]
    assert _AuxiliaryInitial("h^2", "h", Depth(), 2).uOfXT((0.5, 0.0), 0.0) == 4.0
    assert _AuxiliaryInitial("0", "h", Depth(), 2).uOfXT((0.5, 0.0), 0.0) == 0.0
    assert _AuxiliaryInitial("h x", "h", Depth(), 2).uOfXT((0.5, 0.0), 0.0) == 1.0
    with pytest.raises(RealizeError, match="reads u"):
        _AuxiliaryInitial("u^2", "h", Depth(), 2)


def test_cells_per_axis_reach_the_solver_that_takes_them_and_are_refused_by_the_one_that_does_not():
    from proteus.cli.ymfrun import cells_along_x, one_count
    assert cells_along_x({"x": 80, "y": 40}) == 80 and cells_along_x(49) == 49
    assert one_count(49, "d") == 49 and one_count({"x": 8, "y": 8}, "d") == 8
    with pytest.raises(ValueError, match="unequal per axis"):
        one_count({"x": 80, "y": 40}, "adr_box")


def _split_of(name, entry, mutate=None):
    from ymf.closure import load, plain
    from ymf.symbolic.split import split_problem
    doc, _, _ = load(os.path.join(HERE, name))
    doc = plain(doc)
    if mutate:
        mutate(doc)
    return split_problem(doc, entry)


def test_a_squared_initial_value_is_the_product_the_deck_computes():
    """ymf's ``**2`` on one point is the C library's pow(u, 2.0), not u*u, for about one double in 800; the deck squares
    arrays, which multiplies. The initial level set of the 3-D dam break must equal the deck's at every point, including
    the ones where the two differ (the chewbacca nightly's TetGen mesh has such a node: 38 of 95 datasets differed)."""
    from proteus.DispatchRun import _field, exact_squares
    split = _split_of("dambreak3d_short.ymf", "level_set_five")
    code = split["initial"]["φ"]
    assert "**2" in code
    column = (1.2, 0.5, 0.6)

    def deck(x):
        d = numpy.array([x[0] - column[0], x[1] - column[1], x[2] - column[2]])
        if (d <= 0.0).all():
            return float(d.max())
        return float(numpy.sqrt((numpy.maximum(d, 0.0)**2).sum()))

    field = _field(code, 3)
    # points whose scalar pow(u, 2.0) is not u*u, found rather than listed: the C library decides
    hits = [i for i in range(1, 4000) if (numpy.float64(1.2 + i * 1e-3) - 1.2)**2 != numpy.square(numpy.float64(1.2 + i * 1e-3) - 1.2)]
    if not hits:
        pytest.skip("this C library's pow(u, 2.0) equals u*u on every probe point: nothing to tell the two apart")
    for i in hits[:200]:
        point = numpy.array([1.2 + i * 1e-3, 0.77, 0.9])
        assert float(field.uOfXT(point, 0.0)) == deck(point), point
    assert exact_squares("(x - 1)**2 + y**3 + (z)**2.0 + t**(2) + w**p") == \
        "numpy.square(x - 1) + y ** 3 + numpy.square(z) + numpy.square(t) + w ** p"


def test_a_3d_tank_sets_one_setter_per_face():
    from proteus.DispatchRun import _tank, tank_sides

    class Comm(object):
        def isMaster(self):
            return False

        def barrier(self):
            pass
    split = _split_of("dambreak3d_short.ymf", "level_set_five")
    sides = tank_sides(split)
    assert {s: v[1] for s, v in sides.items()} == {"x-": "setFreeSlip", "x+": "setFreeSlip", "y-": "setFreeSlip",
                                                    "y+": "setFreeSlip", "z-": "setFreeSlip", "z+": "setAtmosphere"}
    assert {s: v[0] for s, v in sides.items()} == {"x-": "walls", "x+": "walls", "y-": "walls", "y+": "walls",
                                                    "z-": "walls", "z+": "top"}
    domain, _ = _tank(split, 0.4, "x", "/tmp", Comm())
    assert domain.nd == 3 and domain.MeshOptions.triangleOptions == "VApq1.35q12feena1.066667e-02"


def test_a_3d_face_in_two_regions_or_none_is_refused():
    from proteus.DispatchRun import RealizeError, tank_sides

    def split_the_wall(doc):
        # y = 1 now belongs to a region of its own that is not a whole face of the sets: z = 0 loses its region
        doc["model"]["strong_form"]["boundary_regions"][0]["where"] = "x = 0 or x = 3.22 or y = 0 or y = 1"
    split = _split_of("dambreak3d_short.ymf", "level_set_five", split_the_wall)
    with pytest.raises(RealizeError, match="side z- of the tank is in no boundary region"):
        tank_sides(split)


def test_gauges_are_built_for_the_model_that_solves_their_fields_and_refused_otherwise():
    from proteus.DispatchRun import RealizeError, gauge_objects
    owners = {"clsvof": ({"unknowns": {"φ": ["φ"]}}, [("φ", "clsvof")]),
              "flow": ({"unknowns": {"v": ["v_0", "v_1"]}}, [("v_0", "u"), ("v_1", "v")])}

    def doc(*gauges):
        return {"model": {"gauges": list(gauges)}}
    point = {"name": "p", "kind": "point", "fields": ["v"], "points": [[1.0, 2.0]], "file": "p.csv"}
    built = gauge_objects(doc(point), owners)
    (gauge,) = built["flow"]
    assert gauge.fileName == "p.csv" and sorted(gauge.fields) == ["u", "v"]
    assert list(gauge.points) == [(1.0, 2.0, 0.0)]            # padded to three coordinates
    with pytest.raises(RealizeError, match="boundary_integral kind is not run through dispatch"):
        gauge_objects(doc({"name": "f", "kind": "boundary_integral", "tags": ["a"], "quantities": ["force"]}), owners)
    with pytest.raises(RealizeError, match="no sub-problem solves T"):
        gauge_objects(doc(dict(point, fields=["T"])), owners)
    with pytest.raises(RealizeError, match="one gauge is one file on one model"):
        gauge_objects(doc(dict(point, fields=["v", "φ"])), owners)


def test_a_shallow_water_gauge_samples_the_depth(tmp_path):
    """A point gauge on h goes on SWFlow's n module and writes its file: the dam's depth (1.8 m) at its centre at
    t = 0, with a row per step."""
    ran = _run([sys.executable, "-m", "proteus.cli", "run", os.path.join(HERE, "circular_dam_gauges.ymf"),
                "--outdir", str(tmp_path), "--cells", "9"], tmp_path, 1)
    assert ran.returncode == 0, ran.stdout[-3000:] + ran.stderr[-3000:]
    rows = (tmp_path / "depth.csv").read_text().splitlines()
    assert "h [       10        10         0]" in rows[0] and len(rows) > 10
    assert abs(float(rows[1].split(",")[1]) - 1.8) < 1e-12
