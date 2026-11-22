"""Deprecated: the Richards kernel is in proteus.kernels (C49 R1); see proteus._kernel_aliases."""
from proteus import _kernel_aliases

_kernel_aliases.install(__name__)

__all__ = ["Richards", "cRichards"]
