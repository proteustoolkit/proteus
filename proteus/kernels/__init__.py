"""
The model kernels: free surface incompressible flow and fluid/structure interaction (the former proteus.mprans),
variably saturated flow (proteus.richards), compositional two-phase flow (proteus.m_comp_co2), elastic and plastic
solids (proteus.elastoplastic) and the advection-diffusion-reaction model with the unfitted (cut cell) terms
(proteus.ADR). One flat package since C49 R1; the old import paths are deprecated aliases (proteus/_kernel_aliases.py)
that stay through 2.0.x and are removed at 2.1.
"""

__all__ = ["ADR",
           "AddedMass",
           "ArchiveBeams",
           "CLSVOF",
           "Dissipation",
           "ElastoPlastic",
           "GN_SW2DCV",
           "Kappa",
           "MCorr",
           "MCorr3P",
           "MoveMesh",
           "NCLS",
           "NCLS3P",
           "Pres",
           "PresInc",
           "PresInit",
           "RANS2P",
           "RANS2P_IB",
           "RANS3PF",
           "RANS3PSed",
           "RDLS",
           "Richards",
           "SW2D",
           "SW2DCV",
           "TADR",
           "VOF",
           "VOF3P",
           "VOS3P",
           "beamFEM",
           "cADR",
           "cElastoPlastic",
           "cRichards",
           "cm_comp_co2",
           "m_comp_co2"]
