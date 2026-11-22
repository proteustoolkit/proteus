#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"

#include "CLSVOF.h"

namespace py = pybind11;
using proteus::CLSVOF_base;
using py::return_value_policy;

PYBIND11_MODULE(cCLSVOF, m)
{
    proteus::import_numpy();

    py::class_<CLSVOF_base>(m, "cCLSVOF_base")
        .def(py::init(&proteus::newCLSVOF))
        .def("calculateResidual"        , &CLSVOF_base::calculateResidual        )
        .def("calculateJacobian"        , &CLSVOF_base::calculateJacobian        )
        .def("calculateMetricsAtEOS"    , &CLSVOF_base::calculateMetricsAtEOS    , return_value_policy::take_ownership)
        .def("calculateMetricsAtETS"    , &CLSVOF_base::calculateMetricsAtETS    , return_value_policy::take_ownership)
        .def("normalReconstruction"     , &CLSVOF_base::normalReconstruction     )
        .def("calculateRhsL2Proj"       , &CLSVOF_base::calculateRhsL2Proj       )
        .def("calculateLumpedMassMatrix", &CLSVOF_base::calculateLumpedMassMatrix)
        .def("assembleSpinUpSystem"     , &CLSVOF_base::assembleSpinUpSystem     )
        .def("FCTStep"                  , &CLSVOF_base::FCTStep                  );
}
