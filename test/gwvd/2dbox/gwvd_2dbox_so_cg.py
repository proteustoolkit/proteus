from proteus.default_so import *
from proteus.gwvd import Gwvd_consolidated_classes
from gwvd_2dbox_cg import *

name = "gwvd_2dbox_cv%d_pp%d_rho%d" % (consistentVelocityFlag, vel_pp, int(rho_sw))
pnList = [("gwvd_2dbox_flow_p_cg","gwvd_2dbox_flow_c0p1_n"),
          ("gwvd_2dbox_trans_p_cg","gwvd_2dbox_trans_c0p1_n")]

modelSpinUpList=[0]

#systemStepControllerType = Sequential_MinFLCBDFModelStep
#systemStepControllerType = Sequential_FixedStep
systemStepControllerType =Gwvd_consolidated_classes.ISO_fixed_MinModelStep # Sequential_MinAdaptiveModelStep
atol_iso=1e-8
needEBQ_GLOBAL  = True

needEBQ = True
 
#tnlist = [i*T/nDTout for i in range(nDTout+1)]
# 2012 deck set `tnlist` (ignored); the paper reports t = 1000 d
tnList = [0.0] + [86400.0*d for d in (10, 100, 500, 1000) if 86400.0*d <= T]
if tnList[-1] < T: tnList.append(T)  
  

