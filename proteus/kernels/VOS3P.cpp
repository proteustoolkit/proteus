#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"

#include "VOS3P.h"

namespace py = pybind11;
using proteus::cppVOS3P_base;

PYBIND11_MODULE(cVOS3P, m)
{
    proteus::import_numpy();

    py::class_<cppVOS3P_base>(m, "cppVOS3P_base")
        .def(py::init(&proteus::newVOS3P))
        .def("calculateResidual", &cppVOS3P_base::calculateResidual)
        .def("calculateJacobian", &cppVOS3P_base::calculateJacobian)
        .def("FCTStep", &cppVOS3P_base::FCTStep)
        .def("kth_FCT_step", &cppVOS3P_base::kth_FCT_step)
        .def("calculateResidual_entropy_viscosity", &cppVOS3P_base::calculateResidual_entropy_viscosity)
        .def("calculateMassMatrix", &cppVOS3P_base::calculateMassMatrix);
}
