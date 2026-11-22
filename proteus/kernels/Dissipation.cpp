#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"

#include "Dissipation.h"

namespace py = pybind11;
using proteus::Dissipation_base;

PYBIND11_MODULE(cDissipation, m)
{
    proteus::import_numpy();

    py::class_<Dissipation_base>(m, "cDissipation_base")
        .def(py::init(&proteus::newDissipation))
        .def("calculateResidual", &Dissipation_base::calculateResidual)
        .def("calculateJacobian", &Dissipation_base::calculateJacobian);
}
