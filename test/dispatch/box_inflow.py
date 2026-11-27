"""A subcritical inflow in a 40 m x 10 m box of still water: test/SWFlow/obstacle_flow.py without its cylinder.

The hand-written deck of ``box_inflow.ymf`` (test_dispatch_run.py): the state h = 1, q = (1.5, 0) is the Dirichlet
data of the nodes with x = 0, nothing is imposed elsewhere (reflecting_BCs False: SWFlow's outflow), on the deck's
unstructured mesh path (a RectangularDomain's PSLG meshed by triangle at he). Run as
``parun --SWEs box_inflow.py -C "final_time=1.0 dt_output=0.2 he=4.0"``.
"""
from proteus import Context
from proteus.Domain import RectangularDomain, PlanarStraightLineGraphDomain
import proteus.SWFlow.SWFlowProblem as SWFlowProblem

opts = Context.Options([
    ("sw_model", 0, "sw_model = {0,1} for {SWEs, Disperisve SWEs}}"),
    ("final_time", 20.0, "Final time for simulation"),
    ("dt_output", 1.0, "Time interval to output solution"),
    ("cfl", 0.25, "Desired CFL restriction"),
    ("reflecting_BCs", False, "Use reflecting BCs for all boundaries"),
    ("he", 0.5, "Mesh size"),
    ("mannings", 0.02, "Mannings roughness coefficient"),
])

L = (40.0, 10.0)
rectangle = RectangularDomain(L=L)
rectangle.writePoly("box_inflow")
domain = PlanarStraightLineGraphDomain(fileprefix="box_inflow")
domain.MeshOptions.triangleOptions = "pAq30Dena%f" % (0.5 * opts.he**2,)

h_inflow = 1.0
q_inflow = h_inflow * 1.5


def bathymetry_function(X):
    return 0.0 * X[0]


class water_height_at_t0(object):
    def uOfXT(self, X, t):
        return max(h_inflow - bathymetry_function(X), 0.)


class Zero(object):
    def uOfXT(self, X, t):
        return 0.0


def h_DBC(X, flag):
    if X[0] == 0.0:
        return lambda x, t: h_inflow


def x_mom_DBC(X, flag):
    if X[0] == 0.0:
        return lambda x, t: q_inflow


def y_mom_DBC(X, flag):
    if X[0] == 0.0:
        return lambda x, t: 0.0


outputStepping = SWFlowProblem.OutputStepping(opts.final_time, dt_output=opts.dt_output)
initialConditions = {"water_height": water_height_at_t0(), "x_mom": Zero(), "y_mom": Zero(),
                     "h_times_eta": Zero(), "h_times_w": Zero(), "h_times_beta": Zero()}
boundaryConditions = {"water_height": h_DBC, "x_mom": x_mom_DBC, "y_mom": y_mom_DBC,
                      "h_times_eta": lambda x, flag: None, "h_times_w": lambda x, flag: None,
                      "h_times_beta": lambda x, flag: None}
mySWFlowProblem = SWFlowProblem.SWFlowProblem(sw_model=opts.sw_model, cfl=opts.cfl, outputStepping=outputStepping,
                                              structured=False, he=opts.he, nnx=None, nny=None, domain=domain,
                                              initialConditions=initialConditions,
                                              boundaryConditions=boundaryConditions,
                                              reflectingBCs=opts.reflecting_BCs, bathymetry=bathymetry_function)
mySWFlowProblem.physical_parameters["mannings"] = opts.mannings
