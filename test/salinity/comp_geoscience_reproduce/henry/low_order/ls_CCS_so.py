from proteus.default_so import *
#from proteus.SplitOperator import StrangSplit_time_step

from flow_p import *
from flow_n import *
from tranport_p import *
from tranport_n import *
#from thelper_tadr_n import *

pnList = [ ("flow_p","flow_n"), #,
         ("tranport_p","tranport_n")]

#systemStepControllerType = StrangSplit_time_step

systemStepControllerType = Sequential_MinModelStep

#systemStepControllerType = Sequential_MinAdaptiveModelStep

#systemStepControllerType = Sequential_MinFLCBDFModelStep
#Sequential_MinAdaptiveModelStep

systemStepExact = True

name="ls_CCS_so"

needEBQ_GLOBAL  = False
needEBQ = False

T=  30000.0  # seconds — past Henry steady state (~500 min). If your time unit is days, change back to 30.
nDTout = 201
DT = T/nDTout
tnList = [0.0,1.0e-8, 2.0e-8]+ [i*DT for i  in range(1,nDTout+2)]
#tnList= [i*DT for i  in range(1,nDTout+2)]


#nDTout= 201
archiveFlag = ArchiveFlags.EVERY_USER_STEP
useOneMesh = True
useOneArchive = True
