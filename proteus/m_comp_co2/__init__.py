"""Deprecated: the compositional CO2-brine kernel is in proteus.kernels (C49 R1); see proteus._kernel_aliases."""
from proteus import _kernel_aliases

_kernel_aliases.install(__name__)

__all__ = ["m_comp_co2", "cm_comp_co2"]
