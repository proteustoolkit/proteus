from proteus import FemTools as ft
from proteus import MeshTools as mt
import numpy as np

class SWFlowProblem:
    """ SWFlowProblem """

    def __init__(self,
                 sw_model=0,  # 0: shallow water, 1: disp shallow water
                 # TIME STEPPING #
                 cfl=0.33,
                 outputStepping=None,
                 # DOMAIN AND MESH #
                 structured=False,
                 he=None,
                 nnx=None,
                 nny=None,
                 domain=None,
                 AdH_file=None,
                 # BATHYMETRY #
                 bathymetry=None,
                 triangleFlag=1,
                 # INITIAL CONDITIONS #
                 initialConditions=None,
                 # BOUNDARY CONDITIONS #
                 boundaryConditions=None,
                 reflectingBCs=False,
                 # OTHERS #
                 useSuperlu=None,
                 analyticalSolution=None,
                 # AUXILIARY VARIABLES #
                 auxiliaryVariables=None,
                 genMesh=True,
                 # GENERATION/absorption #
                 gen_length=0.,
                 gen_start=0.,
                 abs_length=0.,
                 abs_start=0.,
                 # WAVE CONDITIONS #
                 waveConditions=None):
        """ Constructor for structured meshes  """
        # ***** SET OF ASSERTS ***** #
        assert sw_model in [
            0, 1], "sw_model={0,1} for shallow water equations or dispersive Serre-Green-Nagdhi respectively"
        assert cfl <= 1, "Choose cfl <= 1"
        assert isinstance(
            outputStepping, OutputStepping), "Provide an object from the OutputStepping class"
        assert type(
            he) == float, "Provide (float) he (characteristic mesh size)"
        if structured:
            assert type(nnx) == int and type(
                nny) == int, "Provide (int) nnx and (int) nny"
        if domain is None:
            assert AdH_file is not None, "If domain is None then provide an AdH File"
        elif bathymetry is not None:
            assert callable(bathymetry) or type(bathymetry)==np.ndarray, "bathymetry should be a callable or numpy array" #dict
            
        assert triangleFlag in [0, 1, 2], "triangleFlag must be 1, 2 or 3"
        assert type(
            initialConditions) == dict, "Provide dict of initial conditions"
        assert type(
            boundaryConditions) == dict, "Provide dict of boundary conditions"
        self.assert_initialConditions(sw_model, initialConditions)
        self.assert_boundaryConditions(sw_model, boundaryConditions)

        # ***** SAVE PARAMETERS ***** #
        self.genMesh = genMesh
        self.sw_model = sw_model
        self.cfl = cfl
        self.outputStepping = outputStepping.getOutputStepping()
        self.he = he
        self.nnx = nnx
        self.nny = nny
        self.nnz = 1
        self.domain = domain
        self.AdH_file = AdH_file
        self.bathymetry = bathymetry
        self.triangleFlag = triangleFlag
        self.initialConditions = initialConditions
        self.boundaryConditions = boundaryConditions
        self.reflectingBCs = reflectingBCs
        self.useSuperlu = useSuperlu
        self.analyticalSolution = analyticalSolution
        self.auxiliaryVariables = []
        self.waveConditions = waveConditions

        # ***** CREATE FINITE ELEMENT SPACES ***** #
        self.FESpace = FESpace().getFESpace()

        # ***** DEFINE PHYSICAL AND NUMERICAL PARAMETERS ***** #
        self.physical_parameters = default_physical_parameters
        self.swe_parameters = default_swe_parameters
        self.GN_swe_parameters = default_GN_swe_parameters

    def assert_initialConditions(self, sw_model, initialConditions):
        assert 'water_height' in initialConditions, 'Provide water_height in initialConditions'
        assert 'x_mom' in initialConditions, 'Provide x_mom in initialConditions'
        assert 'y_mom' in initialConditions, 'Provide y_mom in initialConditions'
        if sw_model == 1:  # dispersive SWEs
            assert 'h_times_eta' in initialConditions, 'Provide auxiliary function h_eta in initialConditions'
            assert 'h_times_w' in initialConditions, 'Provide auxiliary function h_w in initialConditions'
            assert 'h_times_beta' in initialConditions, 'Provide auxiliary function h_beta in initialConditions'
    #

    def assert_boundaryConditions(self, sw_model, boundaryConditions):
        # check dirichlet BCs
        assert 'water_height' in boundaryConditions, 'Provide water_height in BCs'
        assert 'x_mom' in boundaryConditions, 'Provide x_mom in BCs'
        assert 'y_mom' in boundaryConditions, 'Provide y_mom in BCs'
        if sw_model == 1:  # dispersive SWEs
            assert 'h_times_eta' in boundaryConditions, 'Provide auxiliary variable h_eta in BCs'
            assert 'h_times_w' in boundaryConditions, 'Provide auxiliary variable h_w in BCs'
            assert 'h_times_beta' in boundaryConditions, 'Provide auxiliary variable h_beta in BCs'


class OutputStepping:
    """
    OutputStepping handles how often the solution is outputted.
    """

    def __init__(self,
                 final_time,
                 dt_init=0.001,
                 dt_output=None,
                 nDTout=None):
        self.final_time = final_time
        self.dt_init = dt_init
        assert not (
            dt_output is None and nDTout is None), "Provide dt_output or nDTout"
        self.dt_output = dt_output
        self.nDTout = nDTout
    #

    def getOutputStepping(self):
        # COMPUTE dt_init #
        dt_init = min(0.1 * self.dt_output, self.dt_init)
        if self.nDTout is None:
            self.nDTout = int(round(self.final_time/self.dt_output))
        else:
            self.dt_output = float(self.final_time) / float(self.nDTout)
        #
        return {'final_time': self.final_time,
                'dt_init': dt_init,
                'dt_output': self.dt_output,
                'nDTout': self.nDTout}
#


class FESpace:
    """
    Create FE Spaces.
    """

    def __init__(self):
        pass

    def getFESpace(self):
        basis = ft.C0_AffineLinearOnSimplexWithNodalBasis  # p1 space
        # QUADRATURE RULE #
        elementQuadrature = ft.SimplexGaussQuadrature(2, 3)
        elementBoundaryQuadrature = ft.SimplexGaussQuadrature(1, 3)
        return {'basis': basis,
                'elementQuadrature': elementQuadrature,
                'elementBoundaryQuadrature': elementBoundaryQuadrature}


# ***************************************** #
# ********** PHYSICAL PARAMETERS ********** #
# ***************************************** #
default_physical_parameters = {'gravity': 9.81,
                               'LINEAR_FRICTION': 0,
                               'mannings': 0.0,
                               'gen_start': 0.0,
                               'gen_length': 0.0,
                               'abs_start': 0.0,
                               'abs_length': 0.0,}

# ****************************************** #
# ********** NUMERICAL PARAMETERS ********** #
# ****************************************** #
default_swe_parameters = {'LUMPED_MASS_MATRIX': 0,
                          'cfl': 0.33,
                          'SSPOrder': 3,
                          'cE' : 1.,
                          'INTEGRATE_BY_PARTS_WALLS': 1}
default_GN_swe_parameters = {'LUMPED_MASS_MATRIX': 0,
                             'cfl': 0.33,
                             'SSPOrder': 3,
                             'INTEGRATE_BY_PARTS_WALLS': 1}
