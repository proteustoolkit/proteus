from proteus.default_so import *

from flow_p import *
from flow_n import *
from tranport_p import *
from tranport_n import *

pnList = [("flow_p", "flow_n"), ("tranport_p", "tranport_n")]

systemStepControllerType = Sequential_MinModelStep
systemStepExact = True

name = "ls_CCS_so"

needEBQ_GLOBAL = False
needEBQ = False

# FULL 20 YEARS, so every Voss & Souza (1987) Fig 9a comparison time
# (1, 2, 4, 10, 15, 20 yr) falls in the window.  Matches ../run_optionC_20yr,
# which is the same deck at alpha=3.1 -- the pair isolates the retention-curve
# change.
T = 7305.0  # days = 20 years (20 * 365.25)
nDTout = 240  # same ~monthly cadence (DT ~30.4 d)
DT = T / nDTout
# Time list is specified directly in days.
tnList = [0.0, 1.0e-8, 2.0e-8] + [i * DT for i in range(1, nDTout + 2)]

archiveFlag = ArchiveFlags.EVERY_USER_STEP
useOneMesh = True
useOneArchive = True
