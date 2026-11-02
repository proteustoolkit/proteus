#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"

#include "TADR.h"

namespace py = pybind11;
using proteus::TADR_base;

PYBIND11_MODULE(cTADR, m)
{
    proteus::import_numpy();

    py::class_<TADR_base>(m, "cTADR_base")
        .def(py::init(&proteus::newTADR))
        .def("calculateResidual", &TADR_base::calculateResidual)
        .def("calculateJacobian", &TADR_base::calculateJacobian)
        .def("invert", &TADR_base::invert)
        .def("FCTStep", &TADR_base::FCTStep)
        .def("calculateResidual_compatible", &TADR_base::calculateResidual_compatible)
        .def("reconstructFlux_compatible", &TADR_base::reconstructFlux_compatible)
        .def("FCTStep_compatible", &TADR_base::FCTStep_compatible);
}
