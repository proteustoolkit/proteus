from proteus.default_so import *

from Raingarden_p import *
from Raingarden_n import *
from tranport_p import *
from tranport_n import *
#from thelper_tadr_n import *

pnList = [ ("Raingarden_p","Raingarden_n"), #,
         ("tranport_p","tranport_n")]

systemStepControllerType = Sequential_MinModelStep

#systemStepControllerType = Sequential_MinAdaptiveModelStep

#systemStepControllerType = Sequential_MinFLCBDFModelStep
#Sequential_MinAdaptiveModelStep

systemStepExact = True

name="ls_CCS_so"

needEBQ_GLOBAL  = False
needEBQ = False

T=  0.5  #0003
nDTout = 201
DT = T/nDTout
tnList = [0.0,1.0e-8, 2.0e-8]+ [i*DT for i  in range(1,nDTout+2)]
#tnList= [i*DT for i  in range(1,nDTout+2)]


#nDTout= 201
archiveFlag = ArchiveFlags.EVERY_USER_STEP
useOneMesh = True
useOneArchive = True
