"""Split-operator driver for the Simmons benchmark.

pnList order fixes the model indices that the two decks refer to by number:
Richards is 0 (transport_p's V_model) and TADR is 1 (flow_p's density_model).
Changing the order silently breaks the coupling in both directions.
"""

import os

from proteus.default_so import *

from flow_p import *
from flow_n import *
from transport_p import *
from transport_n import *

pnList = [("flow_p", "flow_n"), ("transport_p", "transport_n")]

systemStepControllerType = Sequential_MinModelStep
systemStepExact = True

# The run name has to carry the whole configuration: archives from different
# stabilizations or from the negative control must not collide.  The level tag
# is "%g" with the decimal point spelled "p", so the paper's mixed-level
# validation grid is l4p5z4 and cannot be mistaken for l4; the z part appears
# only when the two levels differ.  A literal "." would survive Path.with_suffix
# but not every splitext in the plotting path, and this name ends up in
# filenames.
name = "simmons_%s_l%s%s_t%d_s%d_stab%d%s%s" % (
    CASE_NAME.lower(),
    ("%g" % SPATIAL_LEVEL).replace(".", "p"),
    "" if SPATIAL_LEVEL_Z == SPATIAL_LEVEL
    else "z" + ("%g" % SPATIAL_LEVEL_Z).replace(".", "p"),
    TEMPORAL_LEVEL, SEED, STAB,
    "_fct" if FCT else "",
    "" if RANDOM_K else "_uniformK")

needEBQ_GLOBAL = False
needEBQ = False

# Output times: the case's reporting times, in seconds, plus two tiny leading
# steps so the coupled fields are populated before the first real step (the
# same startup the working Elder/Liu decks use).
T = case.T
# SIMMONS_T_MINUTES truncates the run, for smoke tests that only need to prove a
# code path executes (STAB=5 + FCT, serial vs parallel) and would otherwise pay
# for the full 140 min of physical time.  It only ever SHORTENS -- a value above
# the case's own T is ignored -- so a stray setting cannot quietly turn a
# production run into something that is not the benchmark.
_T_minutes_env = os.environ.get("SIMMONS_T_MINUTES")
if _T_minutes_env:
    T = min(T, float(_T_minutes_env) * 60.0)
tnList = [0.0, 1.0e-8, 2.0e-8] + [t for t in case.t_out if t <= T]
if tnList[-1] < T:
    tnList.append(T)

from proteus.Profiling import logEvent as _logEvent
if DT_MAX is None:
    _logEvent("Simmons: dt UNCAPPED -- dt will run up to the reporting "
              "interval (%.0f s).  Expect no fingering; see domain_simmons."
              % (tnList[4] - tnList[3]))
else:
    _logEvent("Simmons: dt_max = %.3f s (Courant %.2f on U_buoyant = %.3e m/s, "
              "dz = %.3e m) -> >= %d steps for T = %.0f s"
              % (DT_MAX, TARGET_COURANT, U_BUOYANT, dz, T / DT_MAX, T))

archiveFlag = ArchiveFlags.EVERY_USER_STEP
useOneMesh = True
useOneArchive = True
