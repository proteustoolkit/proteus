#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"

#include "RANS3PF2D.h"

namespace py = pybind11;
using proteus::cppRANS3PF2D_base;

PYBIND11_MODULE(cRANS3PF2D, m)
{
    proteus::import_numpy();

    py::class_<cppRANS3PF2D_base>(m, "cppRANS3PF2D_base")
        .def(py::init(&proteus::newRANS3PF2D))
        .def("calculateResidual", &cppRANS3PF2D_base::calculateResidual)
        .def("calculateJacobian", &cppRANS3PF2D_base::calculateJacobian)
        .def("calculateVelocityAverage", &cppRANS3PF2D_base::calculateVelocityAverage)
        .def("getBoundaryDOFs", &cppRANS3PF2D_base::getBoundaryDOFs);
}

