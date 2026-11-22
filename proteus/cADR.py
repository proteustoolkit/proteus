"""Deprecated: the ADR extension is proteus.kernels.cADR (C49 R1). ``proteus.cADR`` is that module, through 2.0.x."""
import sys
import warnings

from proteus.kernels import cADR as _cADR

warnings.warn("proteus.cADR is deprecated: use proteus.kernels.cADR (removed at 2.1)", DeprecationWarning, stacklevel=2)
sys.modules[__name__] = _cADR
