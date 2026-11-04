"""Split-operator driver for the coupled MMS.

Richards is model 0 (transport_p's V_model) and TADR is model 1 (flow_p's
density_model).  MMS_FLOW_ONLY=1 runs Richards alone (T0 and the T1 reference).
"""

from proteus.default_so import *

import mms_exact
from domain_mms import N, COURANT, FCT, tnList, T

if mms_exact.FLOW_ONLY:
    pnList = [("flow_p", "flow_n")]
else:
    pnList = [("flow_p", "flow_n"), ("transport_p", "transport_n")]

systemStepControllerType = Sequential_MinModelStep
systemStepExact = True

name = "mms_%s%s%s_N%d_co%g_eps%g_om%g_disp%d%s" % (
    mms_exact.C_FIELD,
    "_steady" if mms_exact.STEADY else "",
    "_flowonly" if mms_exact.FLOW_ONLY else "",
    N, COURANT, mms_exact.EPS, mms_exact.OMEGA_MAX, mms_exact.DISP,
    "_fct" if FCT else "_low")

needEBQ_GLOBAL = False
needEBQ = False

# tnList is the time grid itself, so this archives every step; the errors come
# from transport_p's monitor log, the archive is for inspection only.
archiveFlag = ArchiveFlags.EVERY_USER_STEP
useOneMesh = True
useOneArchive = True
