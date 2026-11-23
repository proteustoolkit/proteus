#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"

#include "RANS2P2D.h"

namespace py = pybind11;
using proteus::RANS2P2D_base;

PYBIND11_MODULE(cRANS2P2D, m)
{
    proteus::import_numpy();

    py::class_<RANS2P2D_base>(m, "cRANS2P2D_base")
      .def(py::init(&proteus::newRANS2P2D))
      .def("calculateResidual"                    , &RANS2P2D_base::calculateResidual                     )
      .def("calculateJacobian"                    , &RANS2P2D_base::calculateJacobian                     )
      .def("calculateVelocityAverage"             , &RANS2P2D_base::calculateVelocityAverage              )
      .def("getTwoPhaseAdvectionOperator"         , &RANS2P2D_base::getTwoPhaseAdvectionOperator          )
      .def("getTwoPhaseInvScaledLaplaceOperator"  , &RANS2P2D_base::getTwoPhaseInvScaledLaplaceOperator   )
      .def("getTwoPhaseScaledMassOperator"        , &RANS2P2D_base::getTwoPhaseScaledMassOperator         )
      .def("step6DOF"        , &RANS2P2D_base::step6DOF);
}
