// Standalone test of proteus/UnfittedKernels.h: needs only a C++ compiler (no proteus build, no Python).
// For each kernel: the Jacobian equals a finite difference of the residual, and the penalty block is symmetric
// positive semidefinite. Prints one line per check, "FAIL ..." and a non-zero exit on the first violation.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include "UnfittedKernels.h"

using namespace proteus::unfitted;
static int failures = 0;
static void check(const char *what, double err, double tol)
{
	std::printf("%s %-58s %.3e (tol %.1e)\n", err <= tol ? "ok  " : "FAIL", what, err, tol);
	if (!(err <= tol))
		failures++;
}

template <int n>
struct Data
{
	double jv_test[n], jv_trial[n], fv_test[n], fv_trial[n], x0[n];
	double w, c, s, g, ju0, fu0;
};

// residual of kernel `which` at dof vector x: ju = ju0 + sum_j jv_trial[j] x_j, avg_flux_u = fu0 + sum_j fv_trial[j] x_j
template <int n>
static void residual(int which, const Data<n> &d, const double *x, double *r)
{
	double ju = d.ju0, fu = d.fu0;
	for (int j = 0; j < n; j++)
	{
		ju += d.jv_trial[j] * x[j];
		fu += d.fv_trial[j] * x[j];
	}
	for (int i = 0; i < n; i++)
		r[i] = 0.0;
	if (which == 0)
		jumpPenalty<n>(d.w, d.c, ju, d.jv_test, d.jv_trial, r, nullptr);
	else if (which == 1)
		symmetricConsistency<n>(d.w, d.s, fu, ju, d.fv_test, d.jv_test, d.fv_trial, d.jv_trial, r, nullptr);
	else
		load<n>(d.w, d.g, d.jv_test, r);
}

template <int n>
static void run(const char *tag, bool petrov)
{
	std::mt19937 gen(12345 + n);
	std::uniform_real_distribution<double> U(-1.0, 1.0);
	for (int trial = 0; trial < 50; trial++)
	{
		Data<n> d;
		for (int i = 0; i < n; i++)
		{
			d.jv_test[i] = U(gen);
			d.fv_test[i] = U(gen);
			d.jv_trial[i] = petrov ? U(gen) : d.jv_test[i];
			d.fv_trial[i] = petrov ? U(gen) : d.fv_test[i];
			d.x0[i] = U(gen);
		}
		d.w = 0.3 + std::fabs(U(gen));
		d.c = 0.2 + std::fabs(U(gen));
		d.s = (trial % 2) ? 1.0 : -1.0;
		d.g = U(gen);
		d.ju0 = U(gen);
		d.fu0 = U(gen);
		for (int which = 0; which < 2; which++)
		{
			double J[n * n] = {};
			if (which == 0)
				jumpPenalty<n>(d.w, d.c, 0.0, d.jv_test, d.jv_trial, nullptr, J);
			else
				symmetricConsistency<n>(d.w, d.s, 0.0, 0.0, d.fv_test, d.jv_test, d.fv_trial, d.jv_trial, nullptr, J);
			// finite difference (the residual is linear in x, so the central difference is exact to round-off)
			double scale = 1e-300, err = 0.0;
			for (int j = 0; j < n; j++)
			{
				double xp[n], xm[n], rp[n], rm[n];
				for (int k = 0; k < n; k++)
					xp[k] = xm[k] = d.x0[k];
				const double h = 1e-3;
				xp[j] += h;
				xm[j] -= h;
				residual<n>(which, d, xp, rp);
				residual<n>(which, d, xm, rm);
				for (int i = 0; i < n; i++)
				{
					const double fd = (rp[i] - rm[i]) / (2 * h);
					scale = std::fmax(scale, std::fabs(J[i * n + j]));
					err = std::fmax(err, std::fabs(fd - J[i * n + j]));
				}
			}
			char what[128];
			std::snprintf(what, sizeof what, "%s n=%d %s: Jacobian vs finite difference of its residual", tag, n,
						  which == 0 ? "jumpPenalty" : "symmetricConsistency");
			if (trial == 0 || err / scale > 1e-9)
				check(what, err / scale, 1e-9);
		}
		// the penalty block is symmetric positive semidefinite when test = trial and c, w >= 0: x^T J x >= 0 and J = J^T
		if (!petrov)
		{
			double J[n * n] = {};
			jumpPenalty<n>(d.w, d.c, 0.0, d.jv_test, nullptr, J);
			double asym = 0.0, q = 0.0;
			for (int i = 0; i < n; i++)
				for (int j = 0; j < n; j++)
				{
					asym = std::fmax(asym, std::fabs(J[i * n + j] - J[j * n + i]));
					q += d.x0[i] * J[i * n + j] * d.x0[j];
				}
			if (asym > 1e-14 || q < -1e-14)
			{
				std::printf("FAIL %s n=%d jumpPenalty block not SPSD: asym %.3e, x^T J x %.3e\n", tag, n, asym, q);
				failures++;
			}
		}
		// the load is minus w g v
		{
			double r[n] = {};
			load<n>(d.w, d.g, d.jv_test, r);
			double err = 0.0;
			for (int i = 0; i < n; i++)
				err = std::fmax(err, std::fabs(r[i] + d.w * d.g * d.jv_test[i]));
			if (err > 1e-15)
			{
				std::printf("FAIL %s n=%d load: %.3e\n", tag, n, err);
				failures++;
			}
		}
	}
}

int main()
{
	run<3>("galerkin", false);	// P1 triangle
	run<6>("galerkin", false);	// P2 triangle
	run<4>("galerkin", false);	// P1 tetrahedron
	run<10>("galerkin", false); // P2 tetrahedron
	run<3>("petrov-galerkin", true);
	run<10>("petrov-galerkin", true);
	std::printf("%s\n", failures ? "FAILED" : "ALL PASSED");
	return failures ? 1 : 0;
}
