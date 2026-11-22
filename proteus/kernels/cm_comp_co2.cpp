#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"

#include "m_comp_co2.h"

namespace py = pybind11;
using proteus::m_comp_co2::M_comp_co2_base;

PYBIND11_MODULE(cm_comp_co2, m)
{
  proteus::import_numpy();

  py::class_<M_comp_co2_base>(m, "cM_comp_co2_base")
    .def(py::init(&proteus::m_comp_co2::newm_comp_co2))
    .def("calculateResidual", &M_comp_co2_base::calculateResidual)
    .def("calculateJacobian", &M_comp_co2_base::calculateJacobian)
    .def("invert", &M_comp_co2_base::invert)
    .def("FCTStep", &M_comp_co2_base::FCTStep)
    //.def("kth_FCT_step", &M_comp_co2_base::kth_FCT_step)
    .def("calculateResidual_entropy_viscosity", &M_comp_co2_base::calculateResidual_entropy_viscosity)
    .def("calculateMassMatrix", &M_comp_co2_base::calculateMassMatrix)
    .def("dissolutionFlash", &M_comp_co2_base::dissolutionFlash)
    .def("calculateFlashFields", &M_comp_co2_base::calculateFlashFields);
}