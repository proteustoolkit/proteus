#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"

#include "Pres.h"

namespace py = pybind11;
using proteus::cppPres_base;

PYBIND11_MODULE(cPres, m)
{
    proteus::import_numpy();

    py::class_<cppPres_base>(m, "Pres")
        .def(py::init(&proteus::newPres))
        .def("calculateResidual", &cppPres_base::calculateResidual)
        .def("calculateJacobian", &cppPres_base::calculateJacobian);
}
