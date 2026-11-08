from proteus import StepControl
"""
Split operator module for two-phase flow
"""

from proteus.default_so import *
import os
from proteus import Context

# Create context from main module
name_so = os.path.basename(__file__)
if '_so.py' in name_so[-6:]:
    name = name_so[:-6]
elif '_so.pyc' in name_so[-7:]:
    name = name_so[:-7]
else:
   raise NameError('Split operator module must end with "_so.py"')

try:
    case = __import__(name)
    Context.setFromModule(case)
    ct = Context.get()
except ImportError:
    raise ImportError(str(name) + '.py not found')

from proteus import BoundaryConditions
for BC in ct.domain.bc:
    BC.getContext()

from proteus.SplitOperator import Sequential_FixedStep_Simple, defaultSystem
if ct.sedimentDynamics:
    pnList = [("vos_p",               "vos_n"),#0
              ("vof_p",               "vof_n"),#1
              ("ls_p",                "ls_n"),#2
              ("redist_p",            "redist_n"),#3
              ("ls_consrv_p",         "ls_consrv_n"),#4
              ("threep_navier_stokes_sed_p", "threep_navier_stokes_sed_n"),#5
              ("twp_navier_stokes_p", "twp_navier_stokes_n"),#6
              ("pressureincrement_p", "pressureincrement_n"),#7
              ("pressure_p", "pressure_n")]#8
    
else:
    pnList = [("vof_p",               "vof_n"),#0
              ("ls_p",                "ls_n"),#1
              ("redist_p",            "redist_n"),#2
              ("ls_consrv_p",         "ls_consrv_n"),#3
              ("twp_navier_stokes_p", "twp_navier_stokes_n"),#4
              ("pressureincrement_p", "pressureincrement_n"),#5
              ("pressure_p", "pressure_n")]#6

if ct.useRANS > 0:
    pnList.append(("kappa_p",
                   "kappa_n"))#7 or #9
    pnList.append(("dissipation_p",
                   "dissipation_n"))#8 or #10

pnList.append(("pressureInitial_p",
                   "pressureInitial_n"))#7 or #9

        
              
name = "tank"

#modelSpinUpList = [ct.VOF_model, ct.LS_model, ct.V_model, ct.PINIT_model]
modelSpinUpList = [ct.PI_model]

class Sequential_MinAdaptiveModelStepPS(Sequential_MinAdaptiveModelStep):
    def __init__(self,modelList,system=defaultSystem,stepExact=True):
        Sequential_MinAdaptiveModelStep.__init__(self,modelList,system,stepExact)
        self.modelList = modelList[:len(pnList)-1]

#class Sequential_MinAdaptiveModelStepPS(Sequential_FixedStep):
#    def __init__(self,modelList,system=defaultSystem,stepExact=True):
#        Sequential_FixedStep.__init__(self,modelList,system,stepExact)
#        self.modelList = modelList[:len(pnList)-1]
#dt_system_fixed = ct.dt_fixed

systemStepControllerType = Sequential_MinAdaptiveModelStepPS

needEBQ_GLOBAL = False
needEBQ = False

tnList = [0.0,ct.dt_init]+[ct.dt_init+i*ct.dt_fixed for i in range(1,ct.nDTout+1)]

info = open("TimeList.txt","w")
#archiveFlag = ArchiveFlags.EVERY_SEQUENCE_STEP
