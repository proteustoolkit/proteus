#include "VOF3P.h"

namespace py = pybind11;

PYBIND11_MODULE(cVOF3P, m)
{
    using proteus::cppVOF3P_base;
    using proteus::newVOF3P;

    proteus::import_numpy();

    py::class_<cppVOF3P_base>(m, "cppVOF3P_base")
        .def("calculateResidualElementBased", &cppVOF3P_base::calculateResidualElementBased)
        .def("calculateJacobian", &cppVOF3P_base::calculateJacobian)
        .def("FCTStep", &cppVOF3P_base::FCTStep)
        .def("calculateResidualEdgeBased", &cppVOF3P_base::calculateResidualEdgeBased);

    m.def("newVOF3P", newVOF3P);
}

