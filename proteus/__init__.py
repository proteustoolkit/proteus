"""
Modules for computing numerical solutions of differential equations

Importing within the package
----------------------------

**A module that writes ``proteus.X.Y`` must ``import proteus.X`` itself.**

``proteus.X`` is attribute access on the package, not an import. It only
resolves because importing ``proteus.X`` anywhere in the process sets ``X`` as
an attribute of ``proteus`` -- so a module that relies on it works for as long
as something else happens to be imported first, and raises ``AttributeError``
the day someone imports it on its own. That is not hypothetical: it is how
``mprans`` modules came to be unimportable except in a particular order.

The rule used to be repeated as a comment in the six files where somebody had
been bitten. It is stated here once instead, and
``test/test_import_hygiene.py`` enforces it across the package, which is the
only version of a convention that survives.

``from proteus import X`` binds ``X`` and is fine; it just is not what
``proteus.X.Y`` needs.
"""
import os as _os
import sys as _sys

# On macOS, libfabric (which conda-forge's MPICH runs over) picks its
# "sockets" provider, whose two connection threads busy-poll at 100% CPU
# each from MPI_Init on, in every process: each MPI rank holds three cores,
# half of it system time. Its "tcp" provider has no such threads. This has
# to be set before MPI starts, which importing proteus.Comm (mpi4py.MPI)
# does, and it only takes effect if MPI wasn't started before proteus was
# imported. A value already in the environment is left alone; libraries
# other than libfabric ignore the variable.
if _sys.platform == "darwin":
    _os.environ.setdefault("FI_PROVIDER", "tcp")

# Under MPI, one thread per rank for the BLAS and OpenMP runtimes numpy brings in,
# unless the user set the variables (proteus/_threads.py).  Without MPI nothing
# changes.  The variables are read when numpy is imported, so this comes first.
from . import _threads
_threads.apply_defaults(_threads.launched_size())

__version__ = '2.0.0.dev'

#: What ``from proteus import *`` binds. It is that surface, not an
#: inventory of submodules: several exist that are deliberately absent,
#: because a star-import should not drag in TwoPhaseFlow, SWFlow or
#: MeshAdaptPUMI (which needs PUMI built).
__all__ = ["Archiver",
           "Domain",
           "SplitOperator",
           "StepControl",
           "NumericalSolution",
           "Comm",
           "AnalyticalSolutions",
           "TransportCoefficients",
           "TwophaseDarcyCoefficients",
           "DiagUtils",
           "EGeometry",
           "ErrorEstimators",
           "FemTools",
           "LatexReport",
           "LinearAlgebraTools",
           "LinearSolvers",
           "MeshTools",
           "NonlinearSolvers",
           "Norms",
           "NumericalFlux",
           "PostProcessingTools",
           "Profiling",
           "Quadrature",
           "RefUtils",
           "ShockCapturing",
           "SimTools",
           "SubgridError",
           "SubsurfaceTransportCoefficients",
           "TimeIntegration",
           "Transport",
           "Viewers",
           "AuxiliaryVariables",
           "deim_utils",
           "default_p",
           "default_n",
           "default_s",
           "default_so",
           "cfemIntegrals",
           "cshockCapturing",
           "csmoothers",
           "csubgridError",
           "ctimeIntegration",
           "ctransportCoefficients",
           "clapack",
           "superluWrappers",
           "cmeshTools",
           "cnumericalFlux",
           "cTwophaseDarcyCoefficients",
           "cADR",
           "WaveTools",
           "Context",
           "BoundaryConditions",
           "SpatialTools",
           "defaults"]
