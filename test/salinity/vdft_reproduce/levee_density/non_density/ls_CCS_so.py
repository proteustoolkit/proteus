from proteus.default_so import *
#from proteus.SplitOperator import StrangSplit_time_step

from Raingarden_p import *
from Raingarden_n import *
from tranport_p import *
from tranport_n import *
#from thelper_tadr_n import *

pnList = [ ("Raingarden_p","Raingarden_n"), #,
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

T=  20000.0 #0003
nDTout = 801
DT = T/nDTout
tnList = [0.0,1.0e-8, 2.0e-8]+ [i*DT for i  in range(1,nDTout+2)]
#tnList= [i*DT for i  in range(1,nDTout+2)]


#nDTout= 201
archiveFlag = ArchiveFlags.EVERY_USER_STEP
useOneMesh = True
useOneArchive = True
