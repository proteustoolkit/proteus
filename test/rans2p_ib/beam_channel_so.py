"""The beam channel with a second model after RANS2P_IB: a dye (VOF, carried by the flow), so a failure of the dye
model comes after RANS2P_IB's postStep has moved a flexible beam (test_step_retry's multi-model RANS2P_IB cases)."""
from proteus.default_so import *
from proteus import Context
import beam_channel_p

pnList = [("beam_channel_p", "beam_channel_n"), ("dye_p", "dye_n")]
name = "beam_channel_dye"
systemStepControllerType = Sequential_MinModelStep
needEBQ_GLOBAL = False
needEBQ = False
tnList = [0.0, 0.01, 0.02, 0.03, 0.04]
