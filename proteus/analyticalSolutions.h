#ifndef ANALYTICALSOLUTIONS_H
#define ANALYTICALSOLUTIONS_H
#include "index_type.h"
/**
   \file analyticalSolutions.h
   \defgroup analyticalSolutions analyticalSolutions
   \brief A C library of analytical solutions to differential equations for use in verification
   @{
*/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef max
#define max(a,b) (((a)>(b))?(a):(b))
#endif
static const double PI = 3.1415926535897932384;

extern int PlaneCouetteFlow_u(
  int *iwork,
  double *rwork,
  int nPoints,
  double t,
  double *x,
  double *u
);
extern int diffusionSin1D(
  int *iwork,
  double *rwork,
  int nPoints,
  double t,
  double *x,
  double *u
);
extern int diffusionSin2D(
  int *iwork,
  double *rwork,
  int nPoints,
  double t,
  double *x,
  double *u
);
extern int diffusionSin3D(
  int *iwork,
  double *rwork,
  int nPoints,
  double t,
  double *x,
  double *u
);
extern int diffusionSin1D_r(
  int *iwork,
  double *rwork,
  int nPoints,
  double t,
  double *x,
  double *u,
  double *r
);
extern int diffusionSin2D_r(
  int *iwork,
  double *rwork,
  int nPoints,
  double t,
  double *x,
  double *u,
  double *r
);
extern int diffusionSin3D_r(
  int *iwork,
  double *rwork,
  int nPoints,
  double t,
  double *x,
  double *u,
  double *r
);
extern int LinearAD_DiracIC(
  int *iwork,
  double *rwork,
  int nPoints,
  double T,
  double *x,
  double *u
);
extern int LinearAD_DiracIC_advectiveVelocity(
  int *iwork,
  double *rwork,
  int nPoints,
  double T,
  double *x,
  double *f
);
extern int LinearAD_DiracIC_diffusiveVelocity(
  int *iwork,
  double *rwork,
  int nPoints,
  double T,
  double *x,
  double *f
);
extern int LinearAD_DiracIC_du(
  int *iwork,
  double *rwork,
  int nPoints,
  double T,
  double *x,
  double *f
);
extern int LinearAD_DiracIC_totalVelocity(
  int *iwork,
  double *rwork,
  int nPoints,
  double T,
  double *x,
  double *f
);
extern int LinearAD_SteadyState(
  int *iwork,
  double *rwork,
  int nPoints,
  double t,
  double *X,
  double *u
);
extern int LinearADR_Decay_DiracIC(
  int *iwork,
  double *rwork,
  int nPoints,
  double T,
  double *x,
  double *u
);
extern int LinearADR_Decay_DiracIC_dr(
  int *iwork,
  double *rwork,
  int nPoints,
  double T,
  double *x,
  double *u,
  double *dr
);
extern int LinearADR_Decay_DiracIC_r(
  int *iwork,
  double *rwork,
  int nPoints,
  double T,
  double *x,
  double *u,
  double *r
);
extern int LinearADR_Sine(
  int *iwork,
  double *rwork,
  int nPoints,
  double t,
  double *x,
  double *u
);
extern int LinearADR_Sine_advectiveVelocity(
  int *iwork,
  double *rwork,
  int nPoints,
  double t,
  double *x,
  double *f
);
extern int LinearADR_Sine_diffusiveVelocity(
  int *iwork,
  double *rwork,
  int nPoints,
  double t,
  double *x,
  double *f
);
extern int LinearADR_Sine_dr(
  int *iwork,
  double *rwork,
  int nPoints,
  double t,
  double *x,
  double *u,
  double *dr
);
extern int LinearADR_Sine_du(
  int *iwork,
  double *rwork,
  int nPoints,
  double t,
  double *x,
  double *f
);
extern int LinearADR_Sine_r(
  int *iwork,
  double *rwork,
  int nPoints,
  double t,
  double *x,
  double *u,
  double *r
);
extern int LinearADR_Sine_totalVelocity(
  int *iwork,
  double *rwork,
  int nPoints,
  double t,
  double *x,
  double *f
);
extern int NonlinearAD_SteadyState(
  int *iwork,
  double *rwork,
  int nPoints,
  double t,
  double *X,
  double *u
);
extern int NonlinearADR_Decay_DiracIC(
  int *iwork,
  double *rwork,
  int nPoints,
  double T,
  double *x,
  double *u
);
extern int NonlinearADR_Decay_DiracIC_dr(
  int *iwork,
  double *rwork,
  int nPoints,
  double T,
  double *x,
  double *u,
  double *dr
);
extern int NonlinearADR_Decay_DiracIC_r(
  int *iwork,
  double *rwork,
  int nPoints,
  double T,
  double *x,
  double *u,
  double *r
);
extern int NonlinearDAE(
  int *iwork,
  double *rwork,
  int nPoints,
  double T,
  double *x,
  double *u
);
extern int NonlinearDAE_f(
  int *iwork,
  double *rwork,
  int nPoints,
  double t,
  double *x,
  double *f
);
extern int PlanePoiseuilleFlow_u(
  int *iwork,
  double *rwork,
  int nPoints,
  double t,
  double *x,
  double *u
);
extern int PoiseuillePipeFlow(
  int *iwork,
  double *rwork,
  int nPoints,
  double t,
  double *x,
  double *u
);
extern int PoiseuillePipeFlow_P(
  int *iwork,
  double *rwork,
  int nPoints,
  double t,
  double *x,
  double *u
);
extern int poissonsEquationExp1D(
  int *iwork,
  double *rwork,
  int nPoints,
  double t,
  double *X,
  double *u
);
extern int poissonsEquationExp2D(
  int *iwork,
  double *rwork,
  int nPoints,
  double t,
  double *X,
  double *u
);
extern int poissonsEquationExp3D(
  int *iwork,
  double *rwork,
  int nPoints,
  double t,
  double *X,
  double *u
);
extern int poissonsEquationExp3D_dr(
  int *iwork,
  double *rwork,
  int nPoints,
  double t,
  double *X,
  double *u,
  double *dr
);
extern int poissonsEquationExp1D_r(
  int *iwork,
  double *rwork,
  int nPoints,
  double t,
  double *X,
  double *u,
  double *r
);
extern int poissonsEquationExp2D_r(
  int *iwork,
  double *rwork,
  int nPoints,
  double t,
  double *X,
  double *u,
  double *r
);
extern int poissonsEquationExp3D_r(
  int *iwork,
  double *rwork,
  int nPoints,
  double t,
  double *X,
  double *u,
  double *r
);
extern int STflowSphere_P(
  int *iwork,
  double *rwork,
  int nPoints,
  double t,
  double *x,
  double *u
);
extern int STflowSphere_Vx(
  int *iwork,
  double *rwork,
  int nPoints,
  double t,
  double *x,
  double *u
);
extern int STflowSphere_Vy(
  int *iwork,
  double *rwork,
  int nPoints,
  double t,
  double *x,
  double *u
);
extern int STflowSphere_Vz(
  int *iwork,
  double *rwork,
  int nPoints,
  double t,
  double *x,
  double *u
);
extern void coords(
  double vx,
  double vy,
  double vz,
  double xS,
  double yS,
  double zS,
  double *x,
  double *r,
  double *theta,
  double *norm_v,
  double *eR,
  double *eTHETA
);
extern void vel(
  double rS,
  double norm_v,
  double r,
  double theta,
  double *vR,
  double *vTHETA
);
extern double uOfX_df(
  double nlC,
  double lu
);
extern double uOfX_f(
  double a,
  double b,
  double nlC,
  double nlD,
  double x,
  double lu
);
extern double f(
  double C,
  double b,
  double a,
  int q,
  int r
);
extern double df(
  double C,
  double b,
  double a,
  int q,
  int r
);
/** @} */
#endif
