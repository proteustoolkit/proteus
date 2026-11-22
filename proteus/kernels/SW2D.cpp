#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"

#include "SW2D.h"

namespace py = pybind11;
using proteus::SW2D_base;

PYBIND11_MODULE(cSW2D, m)
{
    proteus::import_numpy();

    py::class_<SW2D_base>(m, "cSW2D_base")
        .def(py::init(&proteus::newSW2D))
        .def("calculateResidual"        , &SW2D_base::calculateResidual)
        .def("calculateJacobian"        , &SW2D_base::calculateJacobian)
        .def("calculateResidual_supg"   , &SW2D_base::calculateResidual_supg)
        .def("calculateJacobian_supg"   , &SW2D_base::calculateJacobian_supg);
}
