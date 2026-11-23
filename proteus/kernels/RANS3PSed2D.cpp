#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"

#include "RANS3PSed2D.h"

namespace py = pybind11;
using proteus::cppRANS3PSed2D_base;

PYBIND11_MODULE(cRANS3PSed2D, m)
{
    proteus::import_numpy();

    py::class_<cppRANS3PSed2D_base>(m, "cppRANS3PSed2D_base")
        .def(py::init(&proteus::newRANS3PSed2D))
        .def("calculateResidual", &cppRANS3PSed2D_base::calculateResidual)
        .def("calculateJacobian", &cppRANS3PSed2D_base::calculateJacobian)
        .def("calculateVelocityAverage", &cppRANS3PSed2D_base::calculateVelocityAverage);
}

