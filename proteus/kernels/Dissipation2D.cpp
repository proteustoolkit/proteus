#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"

#include "Dissipation2D.h"

namespace py = pybind11;
using proteus::Dissipation2D_base;

PYBIND11_MODULE(cDissipation2D, m)
{
    proteus::import_numpy();

    py::class_<Dissipation2D_base>(m, "cDissipation2D_base")
        .def(py::init(&proteus::newDissipation2D))
        .def("calculateResidual", &Dissipation2D_base::calculateResidual)
        .def("calculateJacobian", &Dissipation2D_base::calculateJacobian);
}
