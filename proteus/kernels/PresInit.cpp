#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"

#include "PresInit.h"

namespace py = pybind11;
using proteus::cppPresInit_base;

PYBIND11_MODULE(cPresInit, m)
{
    proteus::import_numpy();

    py::class_<cppPresInit_base>(m, "PresInit")
        .def(py::init(&proteus::newPresInit))
        .def("calculateResidual", &cppPresInit_base::calculateResidual)
        .def("calculateJacobian", &cppPresInit_base::calculateJacobian);
}
