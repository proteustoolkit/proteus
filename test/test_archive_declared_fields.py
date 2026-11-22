"""Integration tests for the archiveFields hooks on the real coefficients.

test_archive_fields.py covers the ArchiveFields machinery with stubs. This
file checks that the eight fields migrated out of NumericalSolution.py are
declared by the classes that actually own them, and that the declarations
compose through super().

Importing the mprans modules needs Proteus's compiled extensions, so these
skip in a source tree that hasn't been built.
"""

import numpy as np
import pytest

pytest.importorskip("proteus.kernels.SW2DCV", reason="needs built extensions")

from proteus.ArchiveFields import (ArchiveField, ArchiveFieldError,  # noqa: E402
                                   archive_fields_for)
from proteus.TransportCoefficients import TC_base  # noqa: E402
from proteus.kernels import CLSVOF, GN_SW2DCV, RANS2P, RANS3PF, SW2DCV  # noqa: E402

N_DOF = 5


class StubFemSpace(object):
    def __init__(self):
        self.written = []

    def writeFunctionXdmf(self, ar, u, tCount=0, init=True):
        self.written.append((u.name, np.asarray(u.dof).copy(), tCount))


class StubU(object):
    def __init__(self, n):
        self.dof = np.arange(n, dtype="d")
        self.femSpace = StubFemSpace()


class StubLevelModel(object):
    """Enough of a OneLevelTransport for the hooks and the writer."""

    def __init__(self, coefficients, n=N_DOF, name="level", **attrs):
        self.coefficients = coefficients
        self.name = name
        self.u = {0: StubU(n)}
        for k, v in attrs.items():
            setattr(self, k, v)

    def archiveDeclaredFields(self, archive, t, tCount, model_name=None):
        # Mirrors OneLevelTransport.archiveDeclaredFields. Reproduced rather
        # than inherited because constructing a real OneLevelTransport needs
        # a mesh, a numerics module and a solve.
        from proteus.ArchiveFields import ArchiveFieldError

        for field in archive_fields_for(self, model_name or self.name):
            femSpace = field.resolve_fem_space(self)
            name = field.archive_name
            try:
                femSpace.writeFunctionXdmf(
                    archive, _Fn(field.value, name, femSpace), tCount
                )
            except Exception as exc:
                raise ArchiveFieldError(
                    "failed writing archive field %r at t=%s: %s" % (name, t, exc)
                ) from exc


class _Fn(object):
    def __init__(self, dof, name, femSpace):
        self.dof, self.name, self.femSpace = dof, name, femSpace


class StubBathymetry(object):
    def __init__(self, n=N_DOF):
        self.dof = np.ones(n)


def blank(cls):
    """A coefficients instance without running its __init__.

    The real __init__ signatures take a dozen physical parameters and, for
    some, a mesh. The hooks only touch the few attributes each test sets
    explicitly, so bypassing __init__ keeps these tests about the
    declarations rather than about constructor plumbing.
    """
    return cls.__new__(cls)


def names(lm, model_name="mymodel"):
    return [f.archive_name for f in archive_fields_for(lm, model_name)]


# --------------------------------------------------------------------------
# TC_base: the generic quantDOFs field, and nothing else
# --------------------------------------------------------------------------


def test_tc_base_declares_nothing_by_default():
    # Every coefficients class that doesn't override the hook must archive
    # exactly what it did before this change: nothing extra.
    c = blank(TC_base)
    assert names(StubLevelModel(c)) == []


def test_tc_base_declares_quant_dofs_when_the_flag_is_set():
    c = blank(TC_base)
    c.outputQuantDOFs = True
    lm = StubLevelModel(c, quantDOFs=np.zeros(N_DOF))
    assert names(lm) == ["quantDOFs_for_mymodel"]


def test_quant_dofs_is_skipped_when_the_model_has_none():
    # The flag can be set on a coefficients class whose level model never
    # allocated the array.
    c = blank(TC_base)
    c.outputQuantDOFs = True
    assert names(StubLevelModel(c)) == []


# --------------------------------------------------------------------------
# phi_s -- RANS2P and RANS3PF
# --------------------------------------------------------------------------


@pytest.mark.parametrize("module", [RANS2P, RANS3PF], ids=["RANS2P", "RANS3PF"])
def test_phi_s_is_declared_by_the_class_that_owns_it(module):
    c = blank(module.Coefficients)
    c.phi_s = np.ones(N_DOF) * 7.0
    lm = StubLevelModel(c)
    (field,) = list(archive_fields_for(lm, "m"))
    assert field.archive_name == "phi_s"
    np.testing.assert_array_equal(field.value, np.ones(N_DOF) * 7.0)


@pytest.mark.parametrize("module", [RANS2P, RANS3PF], ids=["RANS2P", "RANS3PF"])
def test_phi_s_before_initialize_mesh_is_skipped_not_raised(module):
    # phi_s is allocated in initializeMesh. If a model somehow archives
    # first, the hook must not blow up the run -- but unlike the code it
    # replaces it logs instead of passing silently.
    c = blank(module.Coefficients)
    assert names(StubLevelModel(c)) == []


# --------------------------------------------------------------------------
# bathymetry and eta -- SW2DCV and GN_SW2DCV
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module", [SW2DCV, GN_SW2DCV], ids=["SW2DCV", "GN_SW2DCV"]
)
def test_bathymetry_and_eta_are_declared(module):
    c = blank(module.Coefficients)
    c.b = StubBathymetry()
    assert names(StubLevelModel(c)) == ["bathymetry", "eta"]


@pytest.mark.parametrize(
    "module", [SW2DCV, GN_SW2DCV], ids=["SW2DCV", "GN_SW2DCV"]
)
def test_eta_is_water_depth_plus_bathymetry(module):
    c = blank(module.Coefficients)
    c.b = StubBathymetry()
    lm = StubLevelModel(c)
    fields = {f.archive_name: f for f in archive_fields_for(lm, "m")}
    np.testing.assert_array_equal(
        fields["eta"].value, c.b.dof + lm.u[0].dof
    )


@pytest.mark.parametrize(
    "module", [SW2DCV, GN_SW2DCV], ids=["SW2DCV", "GN_SW2DCV"]
)
def test_eta_is_recomputed_each_time_the_archive_is_written(module):
    # eta is derived, not stored. The old code recomputed it inline at each
    # archive write; a declaration that captured a stale array would be a
    # silent regression.
    c = blank(module.Coefficients)
    c.b = StubBathymetry()
    lm = StubLevelModel(c)
    lm.u[0].dof[:] = 3.0
    fields = {f.archive_name: f for f in archive_fields_for(lm, "m")}
    np.testing.assert_array_equal(fields["eta"].value, c.b.dof + 3.0)


# --------------------------------------------------------------------------
# vof -- CLSVOF, previously reached by sniffing the model name
# --------------------------------------------------------------------------


def test_clsvof_declares_vof_without_the_driver_checking_its_name():
    # Replaces `if 'clsvof' in model.name:` in NumericalSolution.py.
    c = blank(CLSVOF.Coefficients)
    lm = StubLevelModel(c, vofDOFs=np.full(N_DOF, 0.5))
    (field,) = list(archive_fields_for(lm, "m"))
    assert field.archive_name == "vof"
    np.testing.assert_array_equal(field.value, np.full(N_DOF, 0.5))


def test_vof_comes_from_the_level_model_not_the_coefficients():
    # CLSVOF computes vofDOFs on the level model; the declaration lives on
    # the coefficients. If the hook read self.vofDOFs it would fail here.
    c = blank(CLSVOF.Coefficients)
    lm = StubLevelModel(c, vofDOFs=np.full(N_DOF, 0.25))
    assert not hasattr(c, "vofDOFs")
    assert names(lm) == ["vof"]


# --------------------------------------------------------------------------
# composition through super()
# --------------------------------------------------------------------------


def test_a_subclass_hook_composes_with_the_base_hook():
    # SW2DCV declares bathymetry/eta *and* inherits quantDOFs from TC_base.
    # A hook that forgot `yield from super().archiveFields(lm)` would drop
    # the inherited field.
    c = blank(SW2DCV.Coefficients)
    c.b = StubBathymetry()
    c.outputQuantDOFs = True
    lm = StubLevelModel(c, quantDOFs=np.zeros(N_DOF))
    assert names(lm) == ["quantDOFs_for_mymodel", "bathymetry", "eta"]


@pytest.mark.parametrize(
    "module",
    [RANS2P, RANS3PF, SW2DCV, GN_SW2DCV, CLSVOF],
    ids=["RANS2P", "RANS3PF", "SW2DCV", "GN_SW2DCV", "CLSVOF"],
)
def test_every_migrated_hook_chains_to_super(module):
    c = blank(module.Coefficients)
    c.b = StubBathymetry()
    c.phi_s = np.zeros(N_DOF)
    c.outputQuantDOFs = True
    lm = StubLevelModel(c, quantDOFs=np.zeros(N_DOF), vofDOFs=np.zeros(N_DOF))
    assert "quantDOFs_for_mymodel" in names(lm), (
        "%s.Coefficients.archiveFields does not chain to super(), so it "
        "drops the inherited quantDOFs field" % module.__name__
    )


# --------------------------------------------------------------------------
# the write path
# --------------------------------------------------------------------------


def test_declared_fields_reach_write_function_xdmf():
    c = blank(SW2DCV.Coefficients)
    c.b = StubBathymetry()
    lm = StubLevelModel(c)
    lm.archiveDeclaredFields(archive=object(), t=0.5, tCount=3, model_name="sw")
    written = [name for name, _, _ in lm.u[0].femSpace.written]
    assert written == ["bathymetry", "eta"]
    assert all(tc == 3 for _, _, tc in lm.u[0].femSpace.written)


def test_a_failing_write_names_the_field_instead_of_passing_silently():
    from proteus.ArchiveFields import ArchiveFieldError

    class Exploding(StubFemSpace):
        def writeFunctionXdmf(self, ar, u, tCount=0, init=True):
            raise IOError("hdf5 write failed")

    c = blank(SW2DCV.Coefficients)
    c.b = StubBathymetry()
    lm = StubLevelModel(c)
    lm.u[0].femSpace = Exploding()
    with pytest.raises(ArchiveFieldError, match="bathymetry"):
        lm.archiveDeclaredFields(archive=object(), t=0.5, tCount=0, model_name="sw")


# --------------------------------------------------------------------------
# what was deliberately not migrated
# --------------------------------------------------------------------------


def test_phi_sp_is_not_declared_anywhere():
    """phi_sp was dead code; it must not come back.

    Commit eed483b3 replaced the nodal phi_sp field with quadrature-point
    q_phi_porous, after which `coefficients.phi_sp` no longer existed. The
    two archiving blocks that referenced it kept running inside
    `try/except: pass` and silently wrote nothing -- archives from before
    that change contain a phi_sp0 field and archives after it do not.
    Reviving the declaration would resurrect an AttributeError, this time a
    loud one.
    """
    for module in (RANS2P, RANS3PF, SW2DCV, GN_SW2DCV, CLSVOF):
        c = blank(module.Coefficients)
        c.b = StubBathymetry()
        c.phi_s = np.zeros(N_DOF)
        lm = StubLevelModel(c, quantDOFs=np.zeros(N_DOF), vofDOFs=np.zeros(N_DOF))
        assert "phi_sp" not in names(lm)


# --------------------------------------------------------------------------
# on_mesh_nodes -- fields defined on the mesh, not on a solution space
# --------------------------------------------------------------------------


def test_phi_s_is_declared_on_the_mesh_nodes_not_the_solution_space():
    """phi_s is a vertex field and must say so.

    It is allocated as numpy.ones(mesh.nodeArray.shape[0]) -- one value per
    mesh vertex. Routed through the model's own solution space it was
    written by writeFunctionXdmf_C0P2Lagrange, which declares that space's
    DOF count: for a C0P2 velocity that put 25 values into a DataItem
    claiming 81. Verified against a real archive from before this change:
    NS_convergence_ev.h5 holds phi_s0_t0 with shape (81,) because phi_s was
    then sized to the space; once it became vertex-sized the declaration
    stopped matching the data and nothing noticed.
    """
    for module in (RANS2P, RANS3PF):
        c = blank(module.Coefficients)
        c.phi_s = np.ones(N_DOF)
        lm = StubLevelModel(c)
        (field,) = list(archive_fields_for(lm, "m"))
        assert field.archive_name == "phi_s"
        assert field.on_mesh_nodes is True, (
            "%s must declare phi_s with on_mesh_nodes=True, or it is written "
            "against a solution space whose DOF count it does not match"
            % module.__name__
        )


def test_fields_default_to_the_solution_space():
    # on_mesh_nodes is opt-in; a normal DOF field keeps dispatching through
    # its finite element space.
    assert ArchiveField("u", np.zeros(3)).on_mesh_nodes is False


def test_bathymetry_and_eta_stay_on_the_solution_space():
    # These are genuine DOF vectors of the SWE solution space, unlike phi_s.
    c = blank(SW2DCV.Coefficients)
    c.b = StubBathymetry()
    for f in archive_fields_for(StubLevelModel(c), "m"):
        assert f.on_mesh_nodes is False, f.archive_name


# --------------------------------------------------------------------------
# archive_scalar_dofs -- derived nodal scalars a model hangs off coefficients
# --------------------------------------------------------------------------


def test_archive_scalar_dofs_are_declared():
    """Arrived on main (1.9.0) as another try/except loop in the driver.

    m_comp_co2 computes flash fields Sg/X/c_brine after each step and
    exposes them as coefficients.archive_scalar_dofs = {name: array}. That
    is a declaration, so it is served by the hook rather than by a loop in
    NumericalSolution.py.
    """
    c = blank(TC_base)
    c.archive_scalar_dofs = {"Sg": np.full(N_DOF, 0.25),
                             "X": np.full(N_DOF, 0.5),
                             "c_brine": np.full(N_DOF, 0.75)}
    lm = StubLevelModel(c)
    fields = {f.archive_name: f for f in archive_fields_for(lm, "m")}
    assert sorted(fields) == ["Sg", "X", "c_brine"]
    np.testing.assert_array_equal(fields["Sg"].value, np.full(N_DOF, 0.25))


def test_archive_scalar_dofs_go_on_component_zeros_space():
    """They are sized to u[0].dof and written through component 0.

    m_comp_co2 says so explicitly: under its node split the component 1
    DOFs are renumbered, so a split-sized array archived against the mesh
    nodes comes out speckled.
    """
    c = blank(TC_base)
    c.archive_scalar_dofs = {"Sg": np.zeros(N_DOF)}
    lm = StubLevelModel(c)
    (field,) = list(archive_fields_for(lm, "m"))
    assert field.component == 0
    assert field.on_mesh_nodes is False
    assert field.resolve_fem_space(lm) is lm.u[0].femSpace


def test_no_archive_scalar_dofs_is_the_normal_case():
    c = blank(TC_base)
    assert names(StubLevelModel(c)) == []
    c.archive_scalar_dofs = {}
    assert names(StubLevelModel(c)) == []


def test_archive_scalar_dofs_compose_with_quant_dofs():
    c = blank(TC_base)
    c.outputQuantDOFs = True
    c.archive_scalar_dofs = {"Sg": np.zeros(N_DOF)}
    lm = StubLevelModel(c, quantDOFs=np.zeros(N_DOF))
    assert names(lm) == ["quantDOFs_for_mymodel", "Sg"]


def test_a_broken_scalar_dof_declaration_is_reported_not_skipped():
    """The version on main wrapped this in a bare except, so a wrong-shaped
    or absent array was written as nothing at all, silently."""
    c = blank(TC_base)
    c.archive_scalar_dofs = {"Sg": None}
    with pytest.raises(ArchiveFieldError, match="Sg"):
        list(archive_fields_for(StubLevelModel(c), "m"))
