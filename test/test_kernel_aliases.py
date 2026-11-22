"""The old import paths of the kernels (C49 R1) are deprecated aliases of the new ones through 2.0.x.

proteus.mprans, proteus.richards, proteus.m_comp_co2 and proteus.elastoplastic became proteus.kernels; a few modules of
proteus.mprans went to proteus, proteus.mbd (see proteus/_kernel_aliases.py). Each case runs in a fresh interpreter:
the alias is installed by the first import of an old package, and what the import order does is part of the check.
"""
import subprocess
import sys
import textwrap

import pytest

#: old module -> new module (the same object)
SAME = [
    ("proteus.mprans.RANS2P", "proteus.kernels.RANS2P"),
    ("proteus.mprans.cRANS2P", "proteus.kernels.cRANS2P"),
    ("proteus.mprans.VOF", "proteus.kernels.VOF"),
    ("proteus.mprans.TADR", "proteus.kernels.TADR"),
    ("proteus.mprans.SpatialTools", "proteus.SpatialTools"),
    ("proteus.mprans.BoundaryConditions", "proteus.BoundaryConditions"),
    ("proteus.mprans.BodyDynamics", "proteus.mbd.BodyDynamics"),
    ("proteus.mprans.MoveMeshMonitor", "proteus.MoveMeshMonitor"),
    ("proteus.mprans.cMoveMeshMonitor", "proteus.cMoveMeshMonitor"),
    ("proteus.mprans.MeshSmoothing", "proteus.MeshSmoothing"),
    ("proteus.mprans.cArgumentsDict", "proteus.cArgumentsDict"),
    ("proteus.richards.Richards", "proteus.kernels.Richards"),
    ("proteus.richards.cRichards", "proteus.kernels.cRichards"),
    ("proteus.m_comp_co2.m_comp_co2", "proteus.kernels.m_comp_co2"),
    ("proteus.m_comp_co2.cm_comp_co2", "proteus.kernels.cm_comp_co2"),
    ("proteus.elastoplastic.ElastoPlastic", "proteus.kernels.ElastoPlastic"),
    ("proteus.elastoplastic.cElastoPlastic", "proteus.kernels.cElastoPlastic"),
    ("proteus.cADR", "proteus.kernels.cADR"),
]


def _run(code):
    ran = subprocess.run([sys.executable, "-W", "always", "-c", textwrap.dedent(code)], capture_output=True, text=True,
                         timeout=300)
    assert ran.returncode == 0, ran.stdout[-2000:] + ran.stderr[-3000:]
    return ran.stderr


@pytest.mark.parametrize("old, new", SAME)
def test_an_old_path_is_the_module_of_the_new_one(old, new):
    _run("""
        import importlib
        old = importlib.import_module(%r)
        new = importlib.import_module(%r)
        assert old is new, (old, new)
        assert old.__spec__.name == %r, old.__spec__.name
        """ % (old, new, new))


def test_from_imports_through_an_old_package_work_for_every_form_air_water_vv_and_the_decks_use():
    err = _run("""
        from proteus.mprans import RANS2P, cRANS2P, VOF, NCLS, RDLS, MCorr, SpatialTools, BoundaryConditions, BodyDynamics
        from proteus.mprans.SpatialTools import Tank2D, Rectangle, assembleDomain
        from proteus.mprans.BoundaryConditions import BC_RANS
        from proteus.richards import Richards
        from proteus.m_comp_co2 import m_comp_co2
        from proteus.elastoplastic import ElastoPlastic
        import proteus.mprans
        import proteus.kernels as k
        assert RANS2P is k.RANS2P and Richards is k.Richards and m_comp_co2 is k.m_comp_co2
        assert proteus.mprans.NCLS is NCLS            # a submodule as an attribute of the old package
        assert Tank2D.__module__ == "proteus.SpatialTools"
        assert BC_RANS.__mro__[1].__name__ == "BC_Base"
        # a class reports its new module path
        assert RANS2P.Coefficients.__module__ == "proteus.kernels.RANS2P"
        """)
    assert "proteus.mprans is deprecated" in err and "DeprecationWarning" in err


def test_a_star_import_of_an_old_package_gives_the_kernels():
    _run("""
        from proteus.mprans import *
        import proteus.kernels as k
        assert RANS2P is k.RANS2P and SpatialTools.Tank2D is not None and BodyDynamics.RigidBody is not None
        """)


def test_the_removed_richards_adr_is_not_silently_the_top_level_adr():
    _run("""
        import importlib
        for name in ("proteus.richards.ADR", "proteus.richards.cADR", "proteus.mprans.NoSuchModule", "proteus.gwvd"):
            try:
                importlib.import_module(name)
            except ImportError:
                continue
            raise AssertionError(name + " imported")
        """)


def test_importing_the_new_paths_warns_about_nothing():
    err = _run("""
        import proteus.kernels.RANS2P, proteus.kernels.cADR, proteus.SpatialTools, proteus.BoundaryConditions
        import proteus.mbd.BodyDynamics, proteus.MoveMeshMonitor
        from proteus import *
        """)
    assert "is deprecated" not in err, err
