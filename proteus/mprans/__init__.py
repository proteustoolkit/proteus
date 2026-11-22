"""Deprecated: the free surface flow kernels are in proteus.kernels (C49 R1); see proteus._kernel_aliases."""
from proteus import _kernel_aliases

_kernel_aliases.install(__name__)

__all__ = ["AddedMass", "ArchiveBeams", "BodyDynamics", "BoundaryConditions", "CLSVOF", "Dissipation", "GN_SW2DCV",
           "Kappa", "MCorr", "MCorr3P", "MoveMesh", "MoveMeshMonitor", "NCLS", "NCLS3P", "Pres", "PresInc", "PresInit",
           "RANS2P", "RANS2P_IB", "RANS3PF", "RANS3PSed", "RDLS", "SW2D", "SW2DCV", "SpatialTools", "TADR", "VOF", "VOF3P",
           "VOS3P", "beamFEM"]
