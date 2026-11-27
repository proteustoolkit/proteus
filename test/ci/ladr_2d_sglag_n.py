"""ladr_2d with the subgrid error lagged (AdvectionDiffusionReaction_ASGS lag=True): the deck the state oracle measures
the LaggedSGEState owner on (test/test_state_oracle.py). ladr_2d_n.py itself runs lag=False."""
from proteus import *
from proteus.default_n import *
try:
    from .ladr_2d_n import *
except ImportError:
    from ladr_2d_n import *

subgridError = AdvectionDiffusionReaction_ASGS(coefficients, nd, lag=True)
