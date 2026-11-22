#include "AddedMass.h"

namespace py = pybind11;

PYBIND11_MODULE(cAddedMass, m)
{
    using proteus::cppAddedMass_base;
    using proteus::newAddedMass;

    proteus::import_numpy();

    py::class_<cppAddedMass_base>(m, "cppAddedMass_base")
        .def("calculateResidual", &cppAddedMass_base::calculateResidual)
        .def("calculateJacobian", &cppAddedMass_base::calculateJacobian);

    m.def("newAddedMass", newAddedMass);
}
