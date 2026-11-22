#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"

#include "MoveMesh2D.h"

namespace py = pybind11;
using proteus::MoveMesh2D_base;

PYBIND11_MODULE(cMoveMesh2D, m)
{
    proteus::import_numpy();

    py::class_<MoveMesh2D_base>(m, "cMoveMesh2D_base")
        .def(py::init(&proteus::newMoveMesh2D))
        .def("calculateResidual", &MoveMesh2D_base::calculateResidual)
        .def("calculateJacobian", &MoveMesh2D_base::calculateJacobian);
}
