#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"

#include "ElastoPlastic.h"

namespace py = pybind11;
using proteus::ElastoPlastic_base;

PYBIND11_MODULE(cElastoPlastic, m)
{
    proteus::import_numpy();

    py::class_<ElastoPlastic_base>(m, "cElastoPlastic_base")
        .def(py::init(&proteus::newElastoPlastic))
        .def("calculateResidual", &ElastoPlastic_base::calculateResidual)
        .def("calculateJacobian", &ElastoPlastic_base::calculateJacobian);
}
