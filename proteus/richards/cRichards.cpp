#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"

#include "Richards.h"

namespace py = pybind11;
using proteus::richards::Richards_base;

PYBIND11_MODULE(cRichards, m)
{
  proteus::import_numpy();

  py::class_<Richards_base>(m, "cRichards_base")
    .def(py::init(&proteus::richards::newRichards))
    .def("calculateResidual", &Richards_base::calculateResidual)
    .def("calculateJacobian", &Richards_base::calculateJacobian)
    .def("invert", &Richards_base::invert)
    .def("FCTStep", &Richards_base::FCTStep)
    .def("ImplicitFCTStep", &Richards_base::ImplicitFCTStep)
    .def("kth_FCT_step", &Richards_base::kth_FCT_step)
    .def("calculateResidual_entropy_viscosity", &Richards_base::calculateResidual_entropy_viscosity)
    .def("calculateMassMatrix", &Richards_base::calculateMassMatrix)
    .def("calculateEdgeVelocity", &Richards_base::calculateEdgeVelocity);
}