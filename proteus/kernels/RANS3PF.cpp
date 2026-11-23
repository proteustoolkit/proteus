#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"

#include "RANS3PF.h"

namespace py = pybind11;
using proteus::cppRANS3PF_base;

PYBIND11_MODULE(cRANS3PF, m)
{
    proteus::import_numpy();

    py::class_<cppRANS3PF_base>(m, "cppRANS3PF_base")
        .def(py::init(&proteus::newRANS3PF))
        .def("calculateResidual", &cppRANS3PF_base::calculateResidual)
        .def("calculateJacobian", &cppRANS3PF_base::calculateJacobian)
        .def("calculateVelocityAverage", &cppRANS3PF_base::calculateVelocityAverage)
        .def("getBoundaryDOFs", &cppRANS3PF_base::getBoundaryDOFs);
}

