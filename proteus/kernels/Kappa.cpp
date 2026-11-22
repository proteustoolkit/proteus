#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"

#include "Kappa.h"

namespace py = pybind11;
using proteus::Kappa_base;

PYBIND11_MODULE(cKappa, m)
{
    proteus::import_numpy();

    py::class_<Kappa_base>(m, "cKappa_base")
        .def(py::init(&proteus::newKappa))
        .def("calculateResidual", &Kappa_base::calculateResidual)
        .def("calculateJacobian", &Kappa_base::calculateJacobian);
}
