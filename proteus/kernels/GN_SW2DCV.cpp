#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"

#include "GN_SW2DCV.h"

namespace py = pybind11;
using proteus::GN_SW2DCV_base;
using pybind11::return_value_policy;

PYBIND11_MODULE(cGN_SW2DCV, m) {
  proteus::import_numpy();

  py::class_<GN_SW2DCV_base>(m, "cGN_SW2DCV_base")
      .def(py::init(&proteus::newGN_SW2DCV))
      .def("convexLimiting", &GN_SW2DCV_base::convexLimiting)
      .def("calculateEdgeBasedCFL", &GN_SW2DCV_base::calculateEdgeBasedCFL,
           return_value_policy::take_ownership)
      .def("calculatePreStep", &GN_SW2DCV_base::calculatePreStep)
      .def("calculateEV", &GN_SW2DCV_base::calculateEV)
      .def("calculateBoundsAndHighOrderRHS",
           &GN_SW2DCV_base::calculateBoundsAndHighOrderRHS)
      .def("calculateResidual", &GN_SW2DCV_base::calculateResidual)
      .def("calculateMassMatrix", &GN_SW2DCV_base::calculateMassMatrix)
      .def("calculateLumpedMassMatrix",
           &GN_SW2DCV_base::calculateLumpedMassMatrix);
}
