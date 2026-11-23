#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"

#include "VOF.h"

namespace py = pybind11;
using proteus::VOF_base;

PYBIND11_MODULE(cVOF, m)
{
    proteus::import_numpy();

    py::class_<VOF_base>(m, "cVOF_base")
        .def(py::init(&proteus::newVOF))
        .def("calculateResidualElementBased"    , &VOF_base::calculateResidualElementBased  )
        .def("calculateJacobian"                , &VOF_base::calculateJacobian              )
        .def("FCTStep"                          , &VOF_base::FCTStep                        )
        .def("calculateResidualEdgeBased"       , &VOF_base::calculateResidualEdgeBased     );
}
