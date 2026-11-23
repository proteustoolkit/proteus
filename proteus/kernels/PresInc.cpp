#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"

#include "PresInc.h"

namespace py = pybind11;
using proteus::cppPresInc_base;

PYBIND11_MODULE(cPresInc, m)
{
    proteus::import_numpy();

    py::class_<cppPresInc_base>(m, "PresInc")
        .def(py::init(&proteus::newPresInc))
        .def("calculateResidual", &cppPresInc_base::calculateResidual)
        .def("calculateJacobian", &cppPresInc_base::calculateJacobian);
}
