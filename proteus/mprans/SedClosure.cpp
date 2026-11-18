#include "SedClosure.h"

namespace py = pybind11;

PYBIND11_MODULE(SedClosure, m)
{
    using proteus::cppHsuSedStress2D; 

    proteus::import_numpy();

    py::class_<cppHsuSedStress2D>(m, "HsuSedStress")
        .def(py::init<double, double, double, double, double, double, double, double,
                      double, double, double, double, double, double, double, double,
	     double>(),
	     py::arg("aDarcy"),
	     py::arg("betaForch"),
	     py::arg("grain"),
	     py::arg("packFraction"),
	     py::arg("packMargin"),
	     py::arg("maxFraction"),
	     py::arg("frFraction"),
	     py::arg("sigmaC"),
	     py::arg("C3e"),
	     py::arg("C4e"),
	     py::arg("eR"),
	     py::arg("fContact"),
	     py::arg("mContact"),
	     py::arg("nContact"),
	     py::arg("angFriction"),
	     py::arg("vos_limiter"),
	     py::arg("mu_fr_limiter"))
        // The closure is a parameter holder. Pickle (and so copy.deepcopy, which
        // MultilevelTransport.initialize applies to every model's coefficients)
        // by its seventeen constructor arguments.
        .def(py::pickle(
            [](const cppHsuSedStress2D& s) {
                return py::make_tuple(s.aDarcy_, s.betaForch_, s.grain_, s.packFraction_,
                                      s.packMargin_, s.maxFraction_, s.frFraction_,
                                      s.sigmaC_, s.C3e_, s.C4e_, s.eR_, s.fContact_,
                                      s.mContact_, s.nContact_, s.angFriction_,
                                      s.vos_limiter_, s.mu_fr_limiter_);
            },
            [](py::tuple t) {
                if (t.size() != 17)
                    throw std::runtime_error("Invalid state for HsuSedStress");
                return cppHsuSedStress2D(
                    t[0].cast<double>(), t[1].cast<double>(), t[2].cast<double>(),
                    t[3].cast<double>(), t[4].cast<double>(), t[5].cast<double>(),
                    t[6].cast<double>(), t[7].cast<double>(), t[8].cast<double>(),
                    t[9].cast<double>(), t[10].cast<double>(), t[11].cast<double>(),
                    t[12].cast<double>(), t[13].cast<double>(), t[14].cast<double>(),
                    t[15].cast<double>(), t[16].cast<double>());
            }))
        .def_readonly("aDarcy", &cppHsuSedStress2D::aDarcy_)
        .def_readonly("betaForch", &cppHsuSedStress2D::betaForch_)
        .def_readonly("grain", &cppHsuSedStress2D::grain_)
        .def_readonly("packFraction", &cppHsuSedStress2D::packFraction_)
        .def_readonly("packMargin", &cppHsuSedStress2D::packMargin_)
        .def_readonly("maxFraction", &cppHsuSedStress2D::maxFraction_)
        .def_readonly("frFraction", &cppHsuSedStress2D::frFraction_)
        .def_readonly("sigmaC", &cppHsuSedStress2D::sigmaC_)
        .def_readonly("C3e", &cppHsuSedStress2D::C3e_)
        .def_readonly("C4e", &cppHsuSedStress2D::C4e_)
        .def_readonly("eR", &cppHsuSedStress2D::eR_)
        .def_readonly("fContact", &cppHsuSedStress2D::fContact_)
        .def_readonly("mContact", &cppHsuSedStress2D::mContact_)
        .def_readonly("nContact", &cppHsuSedStress2D::nContact_)
        .def_readonly("angFriction", &cppHsuSedStress2D::angFriction_)
        .def_readonly("vos_limiter", &cppHsuSedStress2D::vos_limiter_)
        .def_readonly("mu_fr_limiter", &cppHsuSedStress2D::mu_fr_limiter_)
        .def("betaCoeff", &cppHsuSedStress2D::xt_betaCoeff)
        .def("gs0", &cppHsuSedStress2D::gs0)
        .def("kappa_sed1", &cppHsuSedStress2D::xt_kappa_sed1)
        .def("dkappa_sed1_dk", &cppHsuSedStress2D::xt_dkappa_sed1_dk)
        .def("deps_sed_deps", &cppHsuSedStress2D::xt_deps_sed_deps)
        .def("psc", &cppHsuSedStress2D::psc)
        .def("psc_term", &cppHsuSedStress2D::psc_term)
        .def("dpsc_term_dtheta", &cppHsuSedStress2D::dpsc_term_dtheta)
        .def("mu_sc", &cppHsuSedStress2D::mu_sc)
        .def("mu_fr", &cppHsuSedStress2D::mu_fr)
        .def("l_sc", &cppHsuSedStress2D::l_sc)
        .def("tausc_term_theta", &cppHsuSedStress2D::tausc_term_theta)
        .def("gamma_s", &cppHsuSedStress2D::gamma_s)
        .def("dgamma_s_dtheta", &cppHsuSedStress2D::dgamma_s_dtheta)
        .def("jint1", &cppHsuSedStress2D::xt_jint1)
        .def("jint2", &cppHsuSedStress2D::xt_jint2)
        .def("djint2_dtheta", &cppHsuSedStress2D::xt_djint2_dtheta)
        .def("k_diff", &cppHsuSedStress2D::k_diff)
        .def("p_friction", &cppHsuSedStress2D::p_friction)
        .def("gradp_friction", &cppHsuSedStress2D::gradp_friction)
        .def("mIntFluid", &cppHsuSedStress2D::xt_mIntFluid)
        .def("mIntSolid", &cppHsuSedStress2D::xt_mIntSolid)
        .def("mIntgradC", &cppHsuSedStress2D::xt_mIntgradC)
        .def("dmInt_duFluid", &cppHsuSedStress2D::xt_dmInt_duFluid)
        .def("dmInt_duSolid", &cppHsuSedStress2D::xt_dmInt_duSolid)
        .def("p_s", &cppHsuSedStress2D::p_s);
}
