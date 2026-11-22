#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"

#include "MoveMesh.h"

namespace py = pybind11;
using proteus::MoveMesh_base;

PYBIND11_MODULE(cMoveMesh, m)
{
    proteus::import_numpy();

    py::class_<MoveMesh_base>(m, "cMoveMesh_base")
        .def(py::init(&proteus::newMoveMesh))
        .def("calculateResidual", &MoveMesh_base::calculateResidual)
        .def("calculateJacobian", &MoveMesh_base::calculateJacobian);
}
