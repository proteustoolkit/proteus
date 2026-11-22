#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"

#include "ADR.h"

namespace py = pybind11;
using proteus::cADR_base;

PYBIND11_MODULE(cADR, m)
{
    proteus::import_numpy();

    py::class_<cADR_base>(m, "cADR_base")
        .def(py::init(&proteus::newADR))
        .def("calculateResidual", &cADR_base::calculateResidual)
        .def("calculateJacobian", &cADR_base::calculateJacobian);
}
