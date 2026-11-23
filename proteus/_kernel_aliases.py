"""Deprecated import paths of the kernel packages (C49 R1).

proteus.mprans, proteus.richards, proteus.m_comp_co2 and proteus.elastoplastic became proteus.kernels, and a few modules
of proteus.mprans went elsewhere (SpatialTools and BoundaryConditions were merged into proteus.SpatialTools and
proteus.BoundaryConditions, BodyDynamics went to proteus.mbd, MeshSmoothing, MoveMeshMonitor, cMoveMeshMonitor and
cArgumentsDict to proteus). Each old package is a thin ``__init__`` that calls :func:`install`: ``proteus.mprans.X``
(``import``, ``from ... import``, a dotted use after an import) is then the very module object of its new home, so
``proteus.mprans.RANS2P is proteus.kernels.RANS2P`` and classes report their new module path. The old paths warn once
per package with a DeprecationWarning, stay through 2.0.x and are removed at 2.1.

What an alias cannot do: Cython ``cimport`` (update it), and ``proteus.richards.ADR`` / ``cADR`` (removed with the
Richards ADR model, no replacement: proteus.kernels.ADR is the unrelated top-level ADR model).
"""
import importlib
import importlib.abc
import importlib.util
import sys
import warnings

_HERE_TO_PROTEUS = ("MeshSmoothing", "MoveMeshMonitor", "cMoveMeshMonitor", "cArgumentsDict")

#: old package -> {submodule: new module path, or None when it was removed}; every other submodule of an old package
#: is the one of the same name in proteus.kernels
_SPECIAL = {
    "proteus.mprans": dict({"SpatialTools": "proteus.SpatialTools",
                            "BoundaryConditions": "proteus.BoundaryConditions",
                            "BodyDynamics": "proteus.mbd.BodyDynamics"},
                           **{name: "proteus." + name for name in _HERE_TO_PROTEUS}),
    "proteus.richards": {"ADR": None, "cADR": None},
    "proteus.m_comp_co2": {},
    "proteus.elastoplastic": {},
}


def _target(fullname):
    package, _, name = fullname.rpartition(".")
    table = _SPECIAL.get(package)
    if table is None:
        return None
    new = table.get(name, "proteus.kernels." + name)
    if new is None:
        return None
    try:
        return new if importlib.util.find_spec(new) is not None else None
    except ModuleNotFoundError:
        return None


class _AliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        new = _target(fullname)
        if new is None:
            return None
        spec = importlib.util.spec_from_loader(fullname, self)
        spec.loader_state = {"target": new}
        return spec

    def create_module(self, spec):
        module = importlib.import_module(spec.loader_state["target"])
        spec.loader_state["spec"] = module.__spec__   # the import system is about to replace it with this alias's
        return module

    def exec_module(self, module):
        module.__spec__ = module.__spec__.loader_state["spec"]


_FINDER = _AliasFinder()


def install(package):
    """Called by the ``__init__`` of an old package: warn, and make its submodules aliases (and attributes) of the new ones."""
    warnings.warn("%s is deprecated: the kernels are in proteus.kernels (SpatialTools and BoundaryConditions in "
                  "proteus.SpatialTools and proteus.BoundaryConditions, BodyDynamics in proteus.mbd); the old path "
                  "stays through 2.0.x and is removed at 2.1" % package, DeprecationWarning, stacklevel=3)
    if _FINDER not in sys.meta_path:
        sys.meta_path.insert(0, _FINDER)

    def __getattr__(name, _package=package):
        # proteus.mprans.RANS2P after only `import proteus.mprans`, as the old package's submodule attributes did
        # (not `__path__` and the other module attributes)
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return importlib.import_module("%s.%s" % (_package, name))
        except ModuleNotFoundError:
            raise AttributeError("module %r has no attribute %r" % (_package, name)) from None

    sys.modules[package].__getattr__ = __getattr__
