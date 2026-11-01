#ifndef PSKRELATIONS_H
#define PSKRELATIONS_H
#include <algorithm>
#include <cmath>
#include <iostream>
#include <cassert>
#include "densityRelations.h"
#include "SubsurfaceTransportCoefficients.h"
/** \file pskrelations.h
    \defgroup pskrelations pskrelations
    \brief A library of pressure-saturation-permeability relations.
    @{
*/

/** \todo Work out what needs to happen at Se=0,1 for psk relations */

/* jcc two phase flow, modified by cek and mwf*/ 
using namespace std;

class  PskRelation
{
public:
  double Se,
    dSe_dSw,
    Sw_min,
    Sw_max,
    krw,
    dkrw,
    krn,
    dkrn,
    psic,
    dpsic,
    //in case need psic--> Se form
    dSe_dpsic;
  PskRelation(const double* rwork, const int* iwork = 0):
  Se(1.0),dSe_dSw(1.0),
    Sw_min(rwork[0]),
    Sw_max(rwork[1]),
    krw(1.0),dkrw(0.0),
    krn(0.0),dkrn(0.0),psic(0.0),
    dpsic(0.0),dSe_dpsic(0.0)
  {}
  PskRelation():
    Se(1.0),dSe_dSw(1.0),
    Sw_min(0.0),
    Sw_max(1),
    krw(1.0),dkrw(0.0),
    krn(0.0),dkrn(0.0),psic(0.0),
    dpsic(0.0),dSe_dpsic(0.0)
  {}
  virtual inline void setParams(const double* rwork, const int* iwork = 0)
  {
    Sw_min = rwork[0];
    Sw_max = rwork[1];
  }
  /*for fudge factors aka tolerances in various models*/
  virtual inline void setTolerances(const double* rwork_tol)
  {
  }
  virtual ~PskRelation(){}

  /*linear coefficients*/
  inline void calc(const double& Sw)
  {  
    calc_Se(Sw);
    
    krw  = Se;
    dkrw = dSe_dSw;
    
    krn  = (1.0-Se);
    dkrn = -dSe_dSw;
    
    psic  = Se;
    dpsic = dSe_dSw;   
  }
  
  inline void calc_Se(const double& Sw)
  {
    if (Sw < Sw_min)
      {
        Se = 0.0;
        dSe_dSw = 0.0;
      }
    else if (Sw > Sw_max)
      {
        Se = 1.0;
        dSe_dSw =0.0;
      }
    else
      {
        Se = (Sw - Sw_min)/(Sw_max-Sw_min);
        dSe_dSw = 1.0/(Sw_max - Sw_min);
      }
  }
  //for Sw as a function of capillary pressure head
  //note dkrw,dkrn are set to be wrt psic
  virtual inline void calc_from_psic(const double& psicIn)
  {
    bool implemented = false;
    assert(implemented);
  }

};

/* quadratic kr */
class SimplePSK : public PskRelation
{
public:
  SimplePSK(const double* rwork, const int* iwork = 0):
    PskRelation(rwork)
  {}

  inline void calc(const double& Sw)
  {  
    calc_Se(Sw);
    
    krw  = Se*Se;
    dkrw = 2.0*Se*dSe_dSw;
    
    krn  = (1.0-Se)*(1.0-Se); 
    dkrn = 2.0*(Se-1.0)*dSe_dSw; 
    
    psic  =  (1.0-Se)/Sw_min; /*mwf change to test het -Se;*/
    dpsic = -dSe_dSw;   
  }
  //
  virtual inline void calc_from_psic(const double& psicIn)
  {
    psic = max(0.0,psicIn);
    dpsic= 1.0;
    Se = 1.0-psic*Sw_min;
    Se = max(0.0,min(1.0,Se));
    
    krw  = Se*Se;
    dkrw = 2.0*Se*dSe_dSw;
    
    krn  = (1.0-Se)*(1.0-Se); 
    dkrn = 2.0*(Se-1.0)*dSe_dSw; 
    
  }
};

/* Van Genuchten-Mualem */ 
class VGMorig: public PskRelation
{
public:
  double alpha,
    m,
    n,
    Se_eps;
  double Se_eps_const;
  VGMorig()
  {}
  VGMorig(const double* rwork, const int* iwork = 0):
    PskRelation(rwork),
    alpha(rwork[2]),
    m(rwork[3])
  {
    n = (1.0)/(1.0-m);
    Se_eps_const=1.0e-4;
    //mwf debug
    //std::cout<<"VGMorig ctor rwork[2]= "<<rwork[2]<<" alpha= "<<alpha<<" rwork[3]= "<<rwork[3]<<" m= "<<m<<" n= "<<n<<std::endl;
  }
  /*for fudge factors aka tolerances in various models*/
  virtual inline void setTolerances(const double* rwork_tol)
  {
    Se_eps_const = rwork_tol[0];
  }
  
  inline void setParams(const double* rwork, const int* iwork = 0)
  {
    Sw_min = rwork[0];
    Sw_max = rwork[1];
    alpha = rwork[2];
    m = rwork[3];
    n = (1.0)/(1.0-m);  
    //mwf debug
    //std::cout<<"VGMorig setParams rwork[2]= "<<rwork[2]<<" alpha= "<<alpha<<" rwork[3]= "<<rwork[3]<<" m= "<<m<<" n= "<<n<<std::endl;
  }
  inline void calc_Se_eps(const double& Se)
  {
    /* cek todo, work out when you need to stay away from Se=0 and 1 and what to assign in these cases */
    if(Se <= Se_eps_const)
      Se_eps=Se_eps_const;
    else if(Se >= (1.0-Se_eps_const))
      Se_eps=1.0-Se_eps_const;
    else
      Se_eps=Se;
  }

  inline void calc(const double& Sw)
  {
    double Seovmmo,Seovm,Seovmmh,S,Smmo,S2mmo,Sm,S2m,pcesub1,pcesub2; 
    calc_Se(Sw);
    calc_Se_eps(Se);

    Seovmmo  = pow(Se_eps,((1.0/m)-1.0));
    Seovm    = Seovmmo*Se_eps;
    Seovmmh  = pow(Se_eps,((1.0/m)-0.5)); 
    S       = 1.0 - Seovm;
    Smmo    = pow(S,m-1.0);
    S2mmo   = pow(S,2.0*m-1.0);
    Sm      = Smmo*S;
    S2m     = S2mmo*S; 
    
    pcesub1 = pow(((1.0/Seovm)-1.0),((1.0/n)-1.0));
    pcesub2 = pcesub1*((1.0/Seovm)-1.0);
	
    krw  = sqrt(Se)*(1.0-Sm)*(1.0-Sm);
    dkrw = (0.5*(1.0/sqrt(Se_eps))*(1.0-Sm)*(1.0-Sm) + 2.0*(1-Sm)*Smmo*Seovmmh)*dSe_dSw;

    krn  = sqrt(1.0-Se)*S2m;
    dkrn = (-0.5*(1.0/sqrt(1.0-Se_eps))*S2m - 2.0*sqrt(1.0-Se)*(S2mmo)*Seovmmo)*dSe_dSw; 
    
    psic  = pow((pow(Se_eps,(-1.0/m)) - 1.0),(1.0/n))/alpha;
    dpsic = ((-1.0/(alpha*n*m))*(pow((pow(Se_eps,-1.0/m)-1.0),-m))*(pow(Se_eps,((-1.0/m)-1.0))))*dSe_dSw;  
  }
};  

class VGM : public VGMorig
{
 public:
  double ns_del,eps_small,sqrt_eps_small;
  VGM()
  {}
  VGM(const double* rwork, const int* iwork = 0):
    VGMorig(rwork,iwork),
    ns_del(1.0e-8),
    eps_small(1.0e-16),
    sqrt_eps_small(1.0e-8)
  {}
  VGM(const VGM& r)
  {
    Se=r.Se;
    dSe_dSw=r.dSe_dSw;
    Sw_min=r.Sw_min;
    Sw_max=r.Sw_max;
    krw=r.krw;
    dkrw=r.dkrw;
    krn=r.krn;
    dkrn=r.dkrn;
    psic=r.psic;
    dpsic=r.dpsic;
    dSe_dpsic=r.dSe_dpsic;
    alpha=r.alpha;
    m=r.m;
    n=r.n;
    Se_eps_const=r.Se_eps_const;
    ns_del=r.ns_del;
    eps_small=r.eps_small;
    sqrt_eps_small=r.sqrt_eps_small;
  }
  /*for fudge factors aka tolerances in various models*/
  virtual inline void setTolerances(const double* rwork_tol)
  {
    eps_small = rwork_tol[0]; //mwf 072110 don't tie to eps_small? sqrt_eps_small = sqrt(eps_small);
    ns_del    = rwork_tol[1];
  }

  inline void calc_Se(const double& Sw)
  {
    Se = (Sw - Sw_min)/(Sw_max-Sw_min);
    dSe_dSw = 1.0/(Sw_max - Sw_min);
    Se = max(eps_small,min(Se,1.0-eps_small));
  }

  inline void calc(const double& Sw)
  {
    calc_Se(Sw);
    //taken from MualemVanGenuchten2p in pdetk
    double sBar,psiC,DsBar_DpC,DDsBar_DDpC,DkrW_DpC,DkrN_DpC;
    double vBar,uBar,
      alphaPsiC, alphaPsiC_n, alphaPsiC_nM1, alphaPsiC_nM2,
      onePlus_alphaPsiC_n,
      sqrt_sBar, sqrt_1minusSbar,
      sBarByOnePlus_alphaPsiC_n, sBarBy_onePlus_alphaPsiC_n_2;

    sBar = Se;
    //begin MualemVanGenuchten2p setVFraction
    onePlus_alphaPsiC_n = pow(sBar,1.0/-m);
    alphaPsiC_n = onePlus_alphaPsiC_n - 1.0;
    alphaPsiC = pow(alphaPsiC_n,1.0/n);
    psiC = alphaPsiC/alpha;

    alphaPsiC_nM1 = alphaPsiC_n/alphaPsiC;
    sBarByOnePlus_alphaPsiC_n = sBar/onePlus_alphaPsiC_n;
    sqrt_sBar = sqrt(sBar);
    sqrt_1minusSbar = sqrt(1.0 - sBar);
    
    DsBar_DpC = -alpha*(n-1.0)*alphaPsiC_nM1 
      *sBarByOnePlus_alphaPsiC_n;
    //DthetaW_DpC = thetaSR[i] * DsBar_DpC; 

    vBar = 1.0-alphaPsiC_nM1*sBar;
    uBar = alphaPsiC_nM1*sBar;

    //change names krW--> krw, krN--> krn
    krw = sqrt_sBar*vBar*vBar;
    krn = sqrt_1minusSbar*uBar*uBar;
    psic= psiC;
    if(psiC<=0.0) 
    {
      DsBar_DpC = 0.0;
      //DthetaW_DpC = 0.0;
      krw = 1.0;
      krn = 0.0;
    }

    //begin MualemVanGenuchten2p calculateDerivatives
    alphaPsiC_nM2 =   alphaPsiC_nM1/alphaPsiC;      
  
    sBarBy_onePlus_alphaPsiC_n_2 = sBarByOnePlus_alphaPsiC_n
      /onePlus_alphaPsiC_n;
    DDsBar_DDpC =  alpha*alpha*(n-1.)
      *((2*n-1.)*alphaPsiC_nM1*alphaPsiC_nM1
	*sBarBy_onePlus_alphaPsiC_n_2
      -
	(n-1.)*alphaPsiC_nM2
	*sBarByOnePlus_alphaPsiC_n);

    //DDthetaW_DDpC = thetaSR[i]*DDsBar_DDpC;

    DkrW_DpC = (0.5/sqrt_sBar)*DsBar_DpC*vBar*vBar
      -
      2.0*sqrt_sBar*vBar*
      (alpha*(n-1.0)*alphaPsiC_nM2*sBar
       + alphaPsiC_nM1 * DsBar_DpC);

    //DKW_DpC = KWs[i]*DkrW_DpC;

  
    //recalculate if necessary
    if (sqrt_1minusSbar >= sqrt_eps_small)//SQRT_MACHINE_EPSILON)
      {
	DkrN_DpC = -(0.5/sqrt_1minusSbar)*DsBar_DpC*uBar*uBar
	  +
	  2.0*sqrt_1minusSbar*uBar*
	  (alpha*(n-1.0)*alphaPsiC_nM2*sBar
	   + alphaPsiC_nM1 * DsBar_DpC);
      }
    else
      {
	DkrN_DpC =((1.0 - sBar)/eps_small)*2.0*sqrt_eps_small*uBar*
	  (alpha*(n-1.0)*alphaPsiC_nM2*sBar
	   + alphaPsiC_nM1 * DsBar_DpC)
	  - (DsBar_DpC/eps_small)*sqrt_eps_small*uBar*uBar;
      }
    
    //if we're in the nonsmooth regime
    if (psiC < ns_del && psiC > 0.0 )
      {
	DkrW_DpC = 0.0;
      }

    if (psiC <= 0.0)
      {
	DDsBar_DDpC = 0.0;
	//DDthetaW_DDpC = 0.0;
	DkrW_DpC = 0.0;
	DkrN_DpC = 0.0;
      }
    //end calculateDerivatives
    double DpC_Dse = 0.0;
    if (fabs(DsBar_DpC) > 0.0)
      DpC_Dse = 1.0/DsBar_DpC;
    double DpC_Dsw = DpC_Dse*dSe_dSw;
    dkrw = DkrW_DpC*DpC_Dsw;
    dkrn = DkrN_DpC*DpC_Dsw;
    dpsic= DpC_Dsw;

  }
  //TODO add this to other classes
  virtual inline void calc_from_psic(const double& psicIn)
  {
    //taken from MualemVanGenuchten2p in pdetk
    double sBar,psiC,DsBar_DpC,DDsBar_DDpC,DkrW_DpC,DkrN_DpC;
    double vBar,uBar,
      alphaPsiC, alphaPsiC_n, alphaPsiC_nM1, alphaPsiC_nM2,
      onePlus_alphaPsiC_n,
      sqrt_sBar, sqrt_1minusSbar,
      sBarByOnePlus_alphaPsiC_n, sBarBy_onePlus_alphaPsiC_n_2;
    

    psiC = max(0.0,psicIn);
    alphaPsiC = alpha*psiC;
    alphaPsiC_n = pow(alphaPsiC,n);
    alphaPsiC_nM1 = alphaPsiC_n/alphaPsiC;
    onePlus_alphaPsiC_n = 1.0 + alphaPsiC_n;
    sBar = pow(onePlus_alphaPsiC_n,-m);
    sBarByOnePlus_alphaPsiC_n = sBar/onePlus_alphaPsiC_n;
    sqrt_sBar = sqrt(sBar);
    sqrt_1minusSbar = sqrt(1.0 - sBar);
    //thetaW = thetaSR[i]*sBar + thetaR[i];
    DsBar_DpC = -alpha*(n-1.0)*alphaPsiC_nM1 
      *sBarByOnePlus_alphaPsiC_n;
    //DthetaW_DpC = thetaSR[i] * DsBar_DpC; 
    vBar = 1.0-alphaPsiC_nM1*sBar;
    uBar = alphaPsiC_nM1*sBar;

    //change names krW--> krw, krN--> krn
    krw = sqrt_sBar*vBar*vBar;
    krn = sqrt_1minusSbar*uBar*uBar;
    Se = sBar;
    psic=psiC;
    if(psiC<=0.0) 
      {
	sBar = 1.0;
	Se = sBar;
	//thetaW = thetaS[i];
	DsBar_DpC = 0.0;
	//DthetaW_DpC = 0.0;
	krw = 1.0;
	krn = 0.0;
      }    
    //mwf debug
    //std::cout<<"vgm_calc_from_psic alpha= "<<alpha<<" m= "<<m<<" psic= "<<psic<<" sBar= "<<sBar<<" krw= "<<krw<<" krn= "<<krn
    //     <<std::endl;
    //begin MualemVanGenuchten2p calculateDerivatives
    alphaPsiC_nM2 =   alphaPsiC_nM1/alphaPsiC;      
  
    sBarBy_onePlus_alphaPsiC_n_2 = sBarByOnePlus_alphaPsiC_n
      /onePlus_alphaPsiC_n;
    DDsBar_DDpC =  alpha*alpha*(n-1.)
      *((2*n-1.)*alphaPsiC_nM1*alphaPsiC_nM1
	*sBarBy_onePlus_alphaPsiC_n_2
      -
	(n-1.)*alphaPsiC_nM2
	*sBarByOnePlus_alphaPsiC_n);

    //DDthetaW_DDpC = thetaSR[i]*DDsBar_DDpC;

    DkrW_DpC = (0.5/sqrt_sBar)*DsBar_DpC*vBar*vBar
      -
      2.0*sqrt_sBar*vBar*
      (alpha*(n-1.0)*alphaPsiC_nM2*sBar
       + alphaPsiC_nM1 * DsBar_DpC);

    //DKW_DpC = KWs[i]*DkrW_DpC;

  
    //recalculate if necessary
    if (sqrt_1minusSbar >= sqrt_eps_small)//SQRT_MACHINE_EPSILON)
      {
	DkrN_DpC = -(0.5/sqrt_1minusSbar)*DsBar_DpC*uBar*uBar
	  +
	  2.0*sqrt_1minusSbar*uBar*
	  (alpha*(n-1.0)*alphaPsiC_nM2*sBar
	   + alphaPsiC_nM1 * DsBar_DpC);
      }
    else
      {
	DkrN_DpC =((1.0 - sBar)/eps_small)*2.0*sqrt_eps_small*uBar*
	  (alpha*(n-1.0)*alphaPsiC_nM2*sBar
	   + alphaPsiC_nM1 * DsBar_DpC)
	  - (DsBar_DpC/eps_small)*sqrt_eps_small*uBar*uBar;
      }
    
    //if we're in the nonsmooth regime
    if (psiC < ns_del && psiC > 0.0 )
      {
	DkrW_DpC = 0.0;
      }

    if (psiC <= 0.0)
      {
	DDsBar_DDpC = 0.0;
	//DDthetaW_DDpC = 0.0;
	DkrW_DpC = 0.0;
	DkrN_DpC = 0.0;
      }
    //end calculateDerivatives
    dkrw = DkrW_DpC; //note: \od{k_{rw}}{\psi_c} not \od{k_{rw}}{S_w}
    dkrn = DkrN_DpC;
    dpsic= 1.0;
    dSe_dpsic=DsBar_DpC;
  }
};
/* Van Genuchten-Burdine */
class VGB : public  VGM
{
public:
  VGB(const double* rwork, const int* iwork = 0):
    VGM(rwork,iwork)
  {}

  inline void calc(const double& Sw)
  {    
    double S,Smmo,Sm,alpha,Se1ovMmo; 
    calc_Se(Sw);
    calc_Se_eps(Se);

    Se1ovMmo = pow(Se_eps,((1.0/m)-1.0));
    S = 1.0-Se1ovMmo*Se_eps;
    Smmo = pow(S,(m-1.0)); 
    Sm = Smmo*S; 

    //cek from  matlab, needs optimizing
    
    krw  = Se_eps*Se_eps*(1.0-pow(1.0-pow(Se_eps,1.0/m),1.0/m));
    dkrw = (2.0*Se_eps*(1.0-pow(1.0-pow(Se_eps,1.0/m),1.0/m))+Se_eps*pow(1.0-pow(Se_eps,1.0/m),1.0/m)/(m*m)*pow(Se_eps,1.0/m)/(1.0-pow(Se_eps,1.0/m)))*dSe_dSw;

    krn  = pow(1.0-Se_eps,2.0)*pow(1.0-pow(Se_eps,1.0/m),1.0*m);
    dkrn = (-2.0*(1.0-Se_eps)*pow(1.0-pow(Se_eps,1.0/m),1.0*m)-pow(1.0-Se_eps,2.0)*pow(1.0-pow(Se_eps,1.0/m),1.0*m)*pow(Se_eps,1.0/m)/Se_eps/(1.0-pow(Se_eps,1.0/m)))*dSe_dSw;

//     krw  = (Se*Se)*(1.0-Sm); 
//     dkrw = ((2.0*Se)*(1.0-Sm)+(Se*Se)*(Se1ovMmo)*Smmo)*dSe_dSw;
    
//     krn  = (1.0-Se)*(1.0-Se)*(Sm);
//     dkrn = (2.0*(Se-1.0)*Sm-((1.0-Se)*(1.0-Se))*Smmo*Se1ovMmo)*dSe_dSw;
    psic  = pow((pow(Se_eps,(-1.0/m)) - 1.0),(1.0/n))/alpha;
    dpsic = ((-1.0/(alpha*n*m))*(pow((pow(Se_eps,-1.0/m)-1.0),-m))*(pow(Se_eps,((-1.0/m)-1.0))))*dSe_dSw;
  }
};

/* Brooks-Corey-Mualem */					  
class BCM : public PskRelation
{
 public:
  double pd,lambda;

  BCM(const double* rwork, const int* iwork = 0):
    PskRelation(rwork,iwork),
    pd(rwork[2]),
    lambda(rwork[3])
  {}
  
  inline void setParams(const double* rwork, const int* iwork = 0)
  {
    Sw_min = rwork[0];
    Sw_max = rwork[1];
    pd = rwork[2];
    lambda = rwork[3];
  }
  inline void calc(const double& Sw)
  {  
    double Value,Expon,krwovSe,Oovbclpo,Oovbcl,X,sqrt1mu;
    calc_Se(Sw);
    
    Oovbcl   = 1.0/lambda; 
    Oovbclpo = Oovbcl+1.0; 
    X        = pow(Se,Oovbcl);
    Value    = 1.0-X*Se;
    Expon    = ((4.0+5.0*lambda)/(2.0*lambda)); 
    krwovSe   = pow(Se,(Expon-1.0));
    sqrt1mu  = sqrt(1.0-Se); 
    
    krw  = krwovSe*Se; 
    dkrw = (Expon*krwovSe)*dSe_dSw;
    
    krn  = sqrt1mu*Value*Value;
    dkrn = (-0.5*(1.0/sqrt1mu)*Value*Value - 2.0*sqrt1mu*Value*Oovbclpo*X )*dSe_dSw;
    
    psic  = 1.0/X;
    dpsic = (-Oovbcl/(X*Se))*dSe_dSw;
  }
};

/* Brooks-Corey-Burdine */					  
class BCB : public BCM
{
public:
  BCB(const double* rwork, const int* iwork = 0):BCM(rwork,iwork)
  {}

  inline void calc(const double& Sw)
  {
    double Se2ovL,Se2,Se3,omSe,Expon,Semoovlmo,Se_cutOff; 
    calc_Se(Sw);

    Se2ovL     = pow(Se,(2.0/lambda));
    Se2        = Se*Se; 
    Se3        = Se*Se2;
    omSe       = 1.0-Se;   
    Expon     = (2.0+3.0*lambda)/lambda;		
    Se_cutOff = max(1.0e-4,Se);
    Semoovlmo  = pow(Se_cutOff,((-1.0/lambda)-1.0)); 					
    
    krw  = Se2ovL*Se3; 
    dkrw = Expon*Se2ovL*Se2*dSe_dSw;
  
    krn  = (omSe*omSe)*(1.0-Se2ovL*Se);
    dkrn = (2.*omSe*(1.0-Se2ovL*Se) - (omSe*omSe)*(Expon-2.0)*Se2ovL*dSe_dSw)*dSe_dSw;
    
    /* cek debug */
    krw  = pow(Se,(2.0+3.0*lambda)/lambda);
    dkrw = (((2.0+3.0*lambda)/lambda)*pow(Se,(2.0+3.0*lambda)/lambda - 1.0))*dSe_dSw;
    
    krn  = (1.0-Se)*(1.0-Se)*(1.0-pow(Se,(2.0+lambda)/lambda));
    dkrn  = (-2.0*(1.0-Se)*(1.0-pow(Se,(2.0+lambda)/lambda))
             -((2.0+lambda)/lambda)*(1.0-Se)*(1.0-Se)*pow(Se,(2.0+lambda)/lambda-1.0))*dSe_dSw;
    
    psic  = pd*Semoovlmo*Se;
    dpsic = pd*(-1.0/lambda)*Semoovlmo*dSe_dSw;
  }
};

class FractionalFlowVariables
{
public:
  double muw,
    mun,
    lambdaw,
    dlambdaw,
    lambdan,
    dlambdan,
    lambdat,
    dlambdat,
    fw,
    dfw,
    fn,
    dfn;
  
  FractionalFlowVariables(double muwIn,double munIn):
    muw(muwIn),
    mun(munIn)
  {}

  inline void calc(const PskRelation& psk,
                   const DensityRelation& density_w,
                   const DensityRelation& density_n)
  {		
    lambdaw  = density_w.rho*psk.krw/muw; 
    dlambdaw =(density_w.rho/muw)*psk.dkrw;
    
    lambdan  =(density_n.rho*psk.krn)/mun;
    dlambdan =(density_n.rho/mun)*psk.dkrn;
    
    lambdat  = lambdaw + lambdan;
    dlambdat = dlambdaw + dlambdan;
    
    fw      = lambdaw/lambdat;
    dfw     = (dlambdaw*lambdat - lambdaw*dlambdat)/(lambdat*lambdat);
    
    fn      = lambdan/lambdat;
    dfn     = (dlambdan*lambdat - lambdan*dlambdat)/(lambdat*lambdat);
  }
};

struct CompressibleN_FractionalFlowVariables : public FractionalFlowVariables
{
public:
  CompressibleN_FractionalFlowVariables(double muwIn,double munIn):
    FractionalFlowVariables(muwIn,munIn)
  {}
  
  double drhon,
    dlambdaw_psiw, 
    drhon_psiw,
    dlambdan_psiw,
    dlambdat_psiw,
    dfw_psiw,
    dfn_psiw;
  
  inline void calc(const PskRelation& psk, 
                   const DensityRelation& density_w,
                   const DensityRelation& density_n)
  {  
    lambdaw       = density_w.rho*psk.krw/muw; 
    dlambdaw      = (density_w.rho/muw)*psk.dkrw;
    dlambdaw_psiw = 0.0; 
    
    drhon         = density_n.drho*psk.dpsic;
    drhon_psiw    = density_n.drho;
    
    lambdan       = (density_n.rho*psk.krn)/mun;
    dlambdan      = (1.0/mun)*(psk.dkrn*density_n.rho + drhon*psk.krn);
    dlambdan_psiw = drhon_psiw*(psk.krn/mun);
    
    lambdat       = lambdaw + lambdan;
    dlambdat      = dlambdaw + dlambdan;
    dlambdat_psiw = dlambdaw_psiw + dlambdan_psiw;
    
    fw       = lambdaw/lambdat;
    dfw      = (dlambdaw*lambdat - lambdaw*dlambdat)/(lambdat*lambdat);
    dfw_psiw = (dlambdaw_psiw*lambdat - lambdaw*dlambdat_psiw)/(lambdat*lambdat);
    
    fn       = lambdan/lambdat;
    dfn      = (dlambdan*lambdat - lambdan*dlambdat)/(lambdat*lambdat);
    dfn_psiw = (dlambdan_psiw*lambdat - lambdan*dlambdat_psiw)/(lambdat*lambdat);
  }
};


class PskSpline: public PskRelation
{
  /*************************************************************
     spline psk relations. tables of values are held (externally) 
     in splineArray in the order
     
     u (sw, or psic), u^-1 (either sw or psic), krw,krn

     force dSe_dSw = 1.0, so that will be consistent with analytical evaluations
      where dkrw, etc are wrt to se

   ************************************************************/
public:
  PskSpline(const double* rworkIn, const int* iworkIn=0):
    PskRelation(),
    nknots(2),
    lastIndex(0),
    uinvOffset(1),
    krwOffset(2),
    krnOffset(3),
    splineArray(rworkIn)
  {
    assert(iworkIn);
    nknots = iworkIn[0];
  }
  virtual ~PskSpline() {}
  virtual inline void calc(const double& Sw)
  {
    assert(splineArray);
    //
    Se = Sw; //note force Se = Sw since splines are evaluated directly
    dSe_dSw = 1.0;
    piecewiseLinearTableLookup(Sw,
			       nknots,
			       &lastIndex,
			       &psic,
			       &dpsic,
			       splineArray,
			       splineArray+uinvOffset*nknots);
    
    //
    piecewiseLinearTableLookup(Sw,
			       nknots,
			       &lastIndex,
			       &krw,
			       &dkrw,
			       splineArray,
			       splineArray+krwOffset*nknots);
    //
    piecewiseLinearTableLookup(Sw,
			       nknots,
			       &lastIndex,
			       &krn,
			       &dkrn,
			       splineArray,
			       splineArray+krnOffset*nknots);

  }
  virtual inline void calc_from_psic(const double& psicIn)
  {
    assert(splineArray);
    //
    psic = psicIn;
    piecewiseLinearTableLookup(psic,
			       nknots,
			       &lastIndex,
			       &Se, //same as Sw
			       &dSe_dpsic, //same as dSw_dpsic
			       splineArray,
			       splineArray+uinvOffset*nknots);
    
    //
    piecewiseLinearTableLookup(psic,
			       nknots,
			       &lastIndex,
			       &krw,
			       &dkrw,
			       splineArray,
			       splineArray+krwOffset*nknots);
    //
    piecewiseLinearTableLookup(psic,
			       nknots,
			       &lastIndex,
			       &krn,
			       &dkrn,
			       splineArray,
			       splineArray+krnOffset*nknots);

  }
  virtual inline void setParams(const double* rwork, const int* iwork = 0)
  {
    splineArray = rwork;
    if (iwork)
      nknots = iwork[0];
  }
public:
  int nknots,lastIndex,uinvOffset,krwOffset,krnOffset;
  const double* splineArray; 
};

/** \file
    \brief Richards free-function PSK closures (moved here from
           proteus/richards/psk_models.h).

    The class hierarchy above (VGM/VGB/BCM/BCB/...) exposes the same
    constitutive models through a stateful calc(Sw) interface.  The routines
    below are the flat, psiC-based form the optimized Richards kernel calls
    from inside its element loops, where a virtual dispatch and a member-state
    write per quadrature point are not affordable.  They are duplicated closures
    on purpose for now; unifying them with the classes above is tracked as
    post-1.9.0 work.
*/


// =============================================================================
// Pore size distribution / relative permeability (PSK) closures for Richards.
//
// Richards.h owns the PDE coefficient assembly (mass, diffusion tensor,
// buoyancy flux, their Jacobians); this header owns *only* the constitutive
// relations theta_w(psiC) and k_rw(psiC) plus their inverses.  Selecting a
// model is a single branch on PSK_TYPE_member at each call site:
//
//     PSK_TYPE 0 -> vgm_*   van Genuchten retention + Mualem k_rw  (default)
//     PSK_TYPE 1 -> bc_*    Brooks-Corey retention + Burdine k_rw
//     PSK_TYPE 2 -> bc_*    Brooks-Corey retention + Mualem  k_rw
//     PSK_TYPE 3 -> gardner_* Gardner exponential retention + k_rw = S_e
//
// Codes 1 and 2 share one retention curve and differ only in the exponent of
// k_rw = S_e^eta, so only the forward closure branches on them; the inverses
// (which see theta_w alone) treat them alike.  Code 3 is a model of its own in
// both halves; it is the closure Tracy's analytical solutions assume.
//
// Conventions shared by every routine here:
//   psiC is the suction (-u).  psiC > 0 is unsaturated, psiC <= 0 saturated.
//   vgm_ and bc_ clamp that saturated side to (thetaS, 1) with zero slope;
//   gardner_ deliberately does NOT, and the block below says why -- a zero
//   dtheta/dpsiC is a lost Jacobian diagonal wherever beta = 0.
//   Derivatives are with respect to psiC, NOT with respect to u; the caller in
//   Richards.h owns the sign flip d(psiC)/du = -1.
//   The vgm_, bc_ and gardner_ routines take the same parameter slots in the
//   same order, so the branch is a one-line swap.  In the BC parameterisation
//   the second numeric parameter is the pore-size index lambda (taking n_vg's
//   slot) and alpha = 1/p_d is the inverse entry-pressure head; under Gardner
//   alpha is the exponential decay rate [1/m] and the n_vg slot is unused.
//
// The Brooks-Corey conductivity exponent eta is NOT fixed by lambda.  Two
// closures are in common use and they do not agree: Burdine gives
// eta = (2+3*lambda)/lambda, Mualem gives eta = 2.5 + 2/lambda.  At the
// lambda = 0.592 sand of Szymkiewicz [2009], WRR 45, W10403 (Table 1, soil 5)
// that is 6.378 against 5.878 -- not a rounding difference: Burdine is a factor
// ~4 drier in k_rw by psiC = 7.5 m and ~7 by 50 m, which moves a gravity-driven
// wetting front.  Which one applies is a property of the parameter set being
// reproduced, so it is selected by PSK_TYPE and the exponent is derived from
// lambda here, in bc_eta, rather than computed by the caller.
// =============================================================================

namespace proteus
{
namespace richards
{
namespace psk
{

// -----------------------------------------------------------------------------
// van Genuchten - Mualem (pore-connectivity exponent l = 1/2)
//
//   S_e     = (1 + (alpha*psiC)^n)^(-m),   m = 1 - 1/n
//   theta_w = thetaR + thetaSR * S_e
//   k_rw    = sqrt(S_e) * (1 - (alpha*psiC)^(n-1) * S_e)^2
//
// pcBarStar floors alpha*psiC at 1e-8 so pow(pcBar, n-2) stays finite as
// psiC -> 0 for n < 2; sqrt_sBarStar floors the same way inside the k_rw
// derivative only.
// -----------------------------------------------------------------------------
inline void vgm_wetting(const double psiC,
                        const double alpha,
                        const double n_vg,
                        const double thetaR,
                        const double thetaSR,
                        double &thetaW,
                        double &DthetaW_DpsiC,
                        double &KWr,
                        double &DKWr_DpsiC)
{
  const double m_vg   = 1.0 - 1.0 / n_vg;
  const double thetaS = thetaR + thetaSR;
  if (psiC > 0.0) {
    const double pcBar     = alpha * psiC;
    double       pcBarStar = pcBar;
    if (pcBar < 1.0e-8) pcBarStar = 1.0e-8;
    const double pcBar_nM2       = pow(pcBarStar, n_vg - 2);
    const double pcBar_nM1       = pcBar_nM2 * pcBar;
    const double pcBar_n         = pcBar_nM1 * pcBar;
    const double onePlus_pcBar_n = 1.0 + pcBar_n;

    const double sBar = pow(onePlus_pcBar_n, -m_vg);
    /* using -mn = 1-n */
    const double DsBar_DpsiC =
        alpha * (1.0 - n_vg) * (sBar / onePlus_pcBar_n) * pcBar_nM1;

    const double vBar  = 1.0 - pcBar_nM1 * sBar;
    const double vBar2 = vBar * vBar;
    const double DvBar_DpsiC =
        -alpha * (n_vg - 1.0) * pcBar_nM2 * sBar - pcBar_nM1 * DsBar_DpsiC;

    thetaW        = thetaSR * sBar + thetaR; //thetaS;//
    DthetaW_DpsiC = thetaSR * DsBar_DpsiC;   //0.0;//

    const double sqrt_sBar     = sqrt(sBar);
    double       sqrt_sBarStar = sqrt_sBar;
    if (sqrt_sBar < 1.0e-8) sqrt_sBarStar = 1.0e-8;
    KWr        = sqrt_sBar * vBar2;
    DKWr_DpsiC = ((0.5 / sqrt_sBarStar) * DsBar_DpsiC * vBar2
                  + 2.0 * sqrt_sBar * vBar * DvBar_DpsiC);
  } else {
    thetaW        = thetaS;
    DthetaW_DpsiC = 0.0;
    KWr           = 1.0;
    DKWr_DpsiC    = 0.0;
  }
}

// Analytic van Genuchten inverse: theta_w -> psiC.  Leaves u untouched (so the
// caller keeps its previous iterate) outside the strictly unsaturated range.
//
// The dry limit is a cap on psiC, not a fraction of thetaR -- same convention as
// bc_invert_analytic below.  1.01*thetaR looks harmless but is a band in theta,
// and theta -> psiC is exponentially steep in the tail, so it silently swallows
// a huge band in head: for a sand with alpha=14.5 1/m and n=2.68 it refuses to
// invert anything drier than psiC = 3.8 m, which for a 20 m column is the whole
// unwetted region.  Every FCT-limited mass landing there was discarded (u kept
// its pre-limiter value), which breaks the conservation chain theta_limited ->
// psi and shows up as spurious infiltration.  pcBar <= 1e4 puts the cut at
// psiC = 1e4/alpha instead, i.e. far outside any physical range.
inline void vgm_invert_analytic(const double m,
                                const double rho,
                                const double alpha,
                                const double n_vg,
                                const double thetaR,
                                const double thetaSR,
                                double &u)
{
  double psiC, pcBar, pcBar_n, sBar, thetaW, thetaS, m_vg;
  m_vg   = 1.0 - 1.0 / n_vg;
  thetaS = thetaR + thetaSR;
  thetaW = m / rho;
  const double pcBarMax = 1.0e4;
  const double SeMin    = pow(1.0 + pow(pcBarMax, n_vg), -m_vg);
  if (thetaW > thetaR + SeMin * thetaSR && thetaW < thetaS) {
    sBar    = (thetaW - thetaR) / thetaSR;
    pcBar_n = pow(sBar, -1.0 / m_vg) - 1.0;
    pcBar   = pow(pcBar_n, 1.0 / n_vg);
    psiC    = pcBar / alpha;
    u       = -psiC;
  } else if (thetaW >= thetaS) {
    if (u < 0.0) u = 0.0;
  } else {
    u = -pcBarMax / alpha;
  }
}

// Newton inverse of the FULL forward mass m = rho(u) * theta_w(u), so the
// exp(beta*u) factor the analytic inverse ignores is included.  The analytic
// inverse seeds the iteration.
//
// The step cap is one capillary length, not an absolute 5 cm.  dtheta/du spans
// ~4 decades over the retention curve, so a fixed step cap cannot reach the
// root at either end: with duMax = 5e-2 the 50 iterations here covered only
// 2.5 m of head, and any correction needing more than that fell through to the
// u = u_prev revert at the bottom of the loop.
inline void vgm_invert_newton(const double m,
                              const double rho,
                              const double beta,
                              const double alpha,
                              const double n_vg,
                              const double thetaR,
                              const double thetaSR,
                              double &u)
{
  const double u_prev = u;

  const double thetaS = thetaR + thetaSR;
  const double m_vg   = 1.0 - 1.0 / n_vg;

  const double psiC0 = -u;
  if (psiC0 <= 0.0) { return; } //saturated, no inversion
  const double rhom0 = rho * std::exp(beta * u);//first guess
  const double thetaW_imp = m / rhom0;
  // Dry limit stated the same way as in vgm_invert_analytic: a cap on psiC, not
  // a fraction of thetaR.  Same reason -- theta -> psiC is exponentially steep
  // in the tail, so 1.01*thetaR is a narrow band in theta but a huge one in
  // head (psiC = 3.8 m for alpha = 14.5 1/m, n = 2.68), and every FCT-limited
  // mass landing in it was silently discarded, breaking theta_limited -> psi.
  // The analytic seed below is exact at beta = 0 and lands within a couple of
  // Newton steps otherwise, so widening the range does not lengthen the solve.
  const double pcBarMax = 1.0e4;
  const double SeMin    = std::pow(1.0 + std::pow(pcBarMax, n_vg), -m_vg);
  if (thetaW_imp < thetaR + SeMin * thetaSR) { return; } //below the dry cut
  const double thetaEps = 1e-12;
  double m_target = m;
  if (thetaW_imp > 0.99 * thetaS) {
    const double thetaWc = std::min(thetaW_imp, thetaS - thetaEps);
    m_target = rhom0 * thetaWc;
  }

  // Use the analytic van Genuchten inverse as the initial guess for Newton.
  {
    const double thetaW_guess = std::min(m_target / rhom0, thetaS - thetaEps);
    if (thetaW_guess > thetaR + thetaEps && thetaW_guess < thetaS - thetaEps) {
      const double sBar = (thetaW_guess - thetaR) / thetaSR;
      if (sBar > 0.0 && sBar < 1.0) {
        const double pcBar_n = std::pow(sBar, -1.0 / m_vg) - 1.0;
        if (pcBar_n > 0.0) {
          const double pcBar = std::pow(pcBar_n, 1.0 / n_vg);
          const double u_guess = -pcBar / alpha;
          if (std::isfinite(u_guess) && u_guess < 0.0) {
            u = u_guess;
          }
        }
      }
    }
  }

  /*----------------------------------------------------
    Newton solve (UNSATURATED ONLY)
  ----------------------------------------------------*/
  const int    maxIts = 50;
  const double tol    = 1e-12 * std::max(1.0, std::fabs(m));
  const double duMax  = 1.0 / alpha;

  auto theta_and_dtheta_du = [&](double u,
                                 double &thetaW,
                                 double &dtheta_du) -> bool
  {
    const double psiC = -u;

    if (psiC <= 0.0) { return false; } //no inversion in saturation
    //van Genuchten relations
    const double pcBar     = alpha * psiC;
    const double pcBarStar = (pcBar < 1e-12) ? 1e-12 : pcBar;
    const double pcBar_nM2       = std::pow(pcBarStar, n_vg - 2.0);
    const double pcBar_nM1       = pcBar_nM2 * pcBar;
    const double pcBar_n         = pcBar_nM1 * pcBar;
    const double onePlus_pcBar_n = 1.0 + pcBar_n;
    const double sBar = std::pow(onePlus_pcBar_n, -m_vg);

    const double DsBar_DpsiC =
      alpha * (1.0 - n_vg) * (sBar / onePlus_pcBar_n) * pcBar_nM1;

    thetaW    = thetaR + thetaSR * sBar;
    dtheta_du = -thetaSR * DsBar_DpsiC;
    if (thetaW <= thetaR + thetaEps) return false;
    if (thetaW >= thetaS - thetaEps) return false;
    return true;
  };
  for (int it = 0; it < maxIts; ++it)
  {
    if (-u <= 0.0) { u = u_prev; return; }

    double thetaW, dtheta_du;
    if (!theta_and_dtheta_du(u, thetaW, dtheta_du)) {
      u = u_prev; return;
    }

    const double rhom = rho * std::exp(beta * u);
    const double g  = rhom * thetaW - m_target;
    if (std::fabs(g) < tol) return;

    const double gp = rhom * (beta * thetaW + dtheta_du);
    // guard against near-zero derivative
    const double gpTol = 1e-14 * std::max(1.0, std::fabs(rhom * thetaW));
    if (std::fabs(gp) < gpTol) { u = u_prev; return; }

    double du = -g / gp;
    if (du >  duMax) du =  duMax;
    if (du < -duMax) du = -duMax;

    u += du;
    if (-u <= 0.0) { u = u_prev; return; }
  }
  u = u_prev;
}

// -----------------------------------------------------------------------------
// Brooks-Corey
//
//   S_e     = (alpha*psiC)^(-lambda)          for alpha*psiC >= 1
//   S_e     = 1                               for alpha*psiC <  1  (saturated)
//   theta_w = thetaR + thetaSR * S_e
//   k_rw    = S_e^eta,  eta = bc_eta(lambda, kr_model)
//   psiC(S_e) = S_e^(-1/lambda) / alpha       (analytic inverse)
//
// BC has a derivative discontinuity in (theta_w, k_rw) at the entry pressure
// alpha*psiC = 1.  Newton typically needs a smoothed regularisation there in
// production; that is deliberately not added at this layer - callers can wrap
// these closures with their own smoothing if needed.
// -----------------------------------------------------------------------------

// Which k_rw closure supplies the Brooks-Corey exponent.  Maps onto PSK_TYPE:
// 1 -> burdine, 2 -> mualem.  burdine is first so it is the zero value and
// stays the default, matching what this header did before the choice existed.
enum class bc_kr { burdine = 0, mualem = 1 };

// eta(lambda) for the two closures.  Kept as a named function rather than
// inlined into bc_wetting so the two formulas sit side by side and the caller
// never has to restate either one.
inline double bc_eta(const double lam, const bc_kr kr_model)
{
  return (kr_model == bc_kr::mualem) ? (2.5 + 2.0 / lam)
                                     : ((2.0 + 3.0 * lam) / lam);
}

inline void bc_wetting(const double psiC,
                       const double alpha,
                       const double lam,
                       const double thetaR,
                       const double thetaSR,
                       double &thetaW,
                       double &DthetaW_DpsiC,
                       double &KWr,
                       double &DKWr_DpsiC,
                       const bc_kr kr_model = bc_kr::burdine)
{
  const double thetaS = thetaR + thetaSR;
  if (psiC <= 0.0) {
    // saturated (no suction)
    thetaW        = thetaS;
    DthetaW_DpsiC = 0.0;
    KWr           = 1.0;
    DKWr_DpsiC    = 0.0;
    return;
  }
  const double pcBar = alpha * psiC;
  if (pcBar <= 1.0) {
    // suction below entry pressure: still fully wetting-saturated
    thetaW        = thetaS;
    DthetaW_DpsiC = 0.0;
    KWr           = 1.0;
    DKWr_DpsiC    = 0.0;
    return;
  }
  // unsaturated branch (pcBar > 1)
  const double Se        = pow(pcBar, -lam);
  const double dSe_DpsiC = -lam * alpha * pow(pcBar, -lam - 1.0);
  thetaW        = thetaR + thetaSR * Se;
  DthetaW_DpsiC = thetaSR * dSe_DpsiC;
  // k_rw = Se^eta, exponent set by the selected closure
  const double exp_w = bc_eta(lam, kr_model);
  KWr        = pow(Se, exp_w);
  DKWr_DpsiC = exp_w * pow(Se, exp_w - 1.0) * dSe_DpsiC;
}

inline void bc_invert_analytic(const double m,
                               const double rho,
                               const double alpha,
                               const double lam,
                               const double thetaR,
                               const double thetaSR,
                               double &u)
{
  const double thetaS = thetaR + thetaSR;
  const double thetaW = m / rho;
  // Dry limit of the inversion, expressed the same way as in bc_invert_newton:
  // a multiple of the entry pressure, not a fraction of thetaR.  See the note
  // there for why 1.01*thetaR cannot be used with the BC tail.
  const double psiCMax_over_pd = 1.0e4;
  const double SeMin = pow(psiCMax_over_pd, -lam);
  if (thetaW > thetaR + SeMin * thetaSR && thetaW < thetaS) {
    const double Se = (thetaW - thetaR) / thetaSR;
    if (Se > 0.0 && Se < 1.0) {
      const double pcBar = pow(Se, -1.0 / lam);
      const double psiC  = pcBar / alpha;
      u = -psiC;
    }
  } else if (thetaW >= thetaS) {
    if (u < -1.0 / alpha) u = -1.0 / alpha;
  } else {
    u = -psiCMax_over_pd / alpha;
  }
}

// Inverse of the FULL forward mass m = rho(u)*theta_w(u) against the BC
// retention curve.  Solved as a bracketed (safeguarded) Newton rather than the
// plain Newton vgm_invert_newton uses, for three reasons specific to BC:
//
//  * Dead band.  Both routines cut the dry tail at a multiple of the entry
//    pressure rather than at a fraction of thetaR (see vgm_invert_newton for
//    why a band in theta is the wrong variable to state it in); BC needs that
//    the more, since its Se ~ psiC^-lambda tail is far fatter than van
//    Genuchten's.  With thetaR = 0.05, thetaSR = 0.35, p_d = 0.5 m, lambda = 2
//    a 1.01*thetaR band would fire for every node drier than psiC ~ 13 m, where
//    under VG (n = 1.8) it does not fire until psiC ~ 1800 m.  Here the limit is
//    enforced exactly rather than through a proxy in theta: if g at that head
//    has not yet changed sign the root is drier than the band and u is left
//    alone.
//
//  * Entry-pressure wall.  theta is flat (== thetaS) for psiC <= p_d, so the
//    root can lie inside a region the retention curve cannot resolve.  That is
//    detected up front (g(u_wall) <= 0) and answered with the entry pressure
//    itself -- the driest head consistent with full saturation, and the
//    correct inverse of the mass handed in -- rather than by discarding the
//    solve and reverting to a stale iterate carrying an unrelated theta.
//
//  * Conditioning.  In the dry tail the compressibility term beta*theta
//    dominates thetaSR*dSe/du (at Se = 1e-6, beta = 1e-5 it is ~350x larger),
//    so the root is set almost entirely by rho(u).  Any seed that evaluates
//    exp(beta*u) at a stale iterate is then wrong by orders of magnitude in
//    Se, and unguarded Newton walks off.  Bracketing removes the dependence on
//    the seed: g is monotone increasing in u, so [u_floor, u_wall] brackets the
//    root by construction and every iterate stays inside it.
inline void bc_invert_newton(const double m,
                             const double rho,
                             const double beta,
                             const double alpha,
                             const double lam,
                             const double thetaR,
                             const double thetaSR,
                             double &u)
{
  if (-u <= 0.0) { return; } //ponded / saturated head: no inversion

  // g(u) = rho(u)*theta_w(u) - m, evaluated with the BC closure (theta_w
  // clamped to thetaS inside the entry pressure).  Monotone increasing in u.
  auto g_of = [&](const double uu) -> double {
    const double pcBar = alpha * (-uu);
    const double Se    = (pcBar <= 1.0) ? 1.0 : pow(pcBar, -lam);
    return rho * std::exp(beta * uu) * (thetaR + thetaSR * Se) - m;
  };

  // Upper end of the bracket: one part in 1e9 outside the entry pressure, so
  // the BC derivative (which drops to zero across it) stays defined.
  const double u_entry = -1.0 / alpha;
  const double u_wall  = -(1.0 + 1.0e-9) / alpha;
  if (g_of(u_wall) <= 0.0) { u = u_entry; return; } //root at or inside the wall

  // Lower end: the driest head the inversion is defined down to, stated as a
  // multiple of the entry pressure.  A floor on Se instead would sit at
  // psiC = p_d*Se^(-1/lambda) and so swing with lambda -- Se = 1e-6 is
  // psiC = 1e3*p_d at lambda = 2 but 1e12*p_d at lambda = 0.5, far outside any
  // head the solver will ever see and wide enough to make the bracket useless.
  const double psiCMax_over_pd = 1.0e4;
  const double u_floor = -psiCMax_over_pd / alpha;
  if (g_of(u_floor) > 0.0) { return; } //root drier than the band: leave u alone

  double lo = u_floor, hi = u_wall; // g(lo) <= 0 < g(hi)

  // Analytic BC inverse (density-free) as the opening guess; the bracket
  // catches it if the neglected exp(beta*u) puts it in the wrong place.
  {
    const double Se = (m / rho - thetaR) / thetaSR;
    if (Se > 0.0 && Se < 1.0) {
      const double u_guess = -pow(Se, -1.0 / lam) / alpha;
      if (std::isfinite(u_guess) && u_guess > lo && u_guess < hi) u = u_guess;
      else u = 0.5 * (lo + hi);
    } else u = 0.5 * (lo + hi);
  }
  if (!(u > lo && u < hi)) u = 0.5 * (lo + hi);

  // maxIts covers the bisection worst case over the widest bracket; Newton
  // reaches the root in a handful of steps whenever the guess is sane.
  const int    maxIts = 100;
  const double tol    = 1e-12 * std::max(1.0, std::fabs(m));

  for (int it = 0; it < maxIts; ++it) {
    // u is strictly inside (u_floor, u_wall), so pcBar > 1 and the BC
    // derivative below is always the unsaturated-branch one.
    const double pcBar     = alpha * (-u);
    const double Se        = pow(pcBar, -lam);
    const double dSe_DpsiC = -lam * alpha * pow(pcBar, -lam - 1.0);
    const double thetaW    = thetaR + thetaSR * Se;
    const double dtheta_du = -thetaSR * dSe_DpsiC; // d(psiC)/d(u) = -1

    const double rhom = rho * std::exp(beta * u);
    const double g    = rhom * thetaW - m;
    if (std::fabs(g) < tol) return;
    if (g > 0.0) hi = u; else lo = u;

    const double gp    = rhom * (beta * thetaW + dtheta_du);
    const double gpTol = 1e-14 * std::max(1.0, std::fabs(rhom * thetaW));
    double u_next = (std::fabs(gp) > gpTol) ? (u - g / gp) : 0.5 * (lo + hi);
    // Bisect whenever Newton leaves the bracket.
    if (!(u_next > lo && u_next < hi)) u_next = 0.5 * (lo + hi);

    if (std::fabs(u_next - u) <= 1e-15 * std::fabs(u)) return;
    u = u_next;
  }
  // Bracketed throughout, so the last iterate is the best available answer.
}

// -----------------------------------------------------------------------------
// Gardner (quasi-linear) retention + conductivity, with the Irmay-style linear
// theta_w(k_rw) pairing:
//
//   S_e     = exp(-alpha*psiC) = exp(alpha*u)
//   theta_w = thetaR + thetaSR * S_e
//   k_rw    = S_e                       i.e. K(psi) = Ks*exp(alpha*psi)
//   psiC(S_e) = -ln(S_e)/alpha          (analytic inverse, exact)
//
// alpha here is Gardner's exponent [1/m]; it takes the same parameter slot as
// van Genuchten's alpha and the Brooks-Corey inverse entry pressure, and the
// second numeric parameter (n_vg / lambda) is unused.
//
// This is the pairing Tracy's analytical solutions are built on, and both
// halves of it matter: with k_rw = S_e and theta_w affine in S_e, the Kirchhoff
// transform hbar = exp(alpha*psi) turns Richards into a linear advection-
// diffusion equation for hbar, which is what makes the closed-form steady and
// transient solutions exist.  Substituting Mualem k_rw here would keep the
// retention curve but destroy the linearisation, so the two are one model, not
// two independent choices.
//
// There is no entry pressure and no dry-end kink: theta_w and k_rw are C-inf
// for psiC > 0 and both approach saturation smoothly as psiC -> 0.  What it
// does have is an unbounded dry tail -- S_e -> 0 only as psiC -> inf -- so the
// inversions cut at alpha*psiC = 7e2, stated as a cap on head like the vgm_/bc_
// routines rather than as a band in theta.  That is where exp() underflows,
// i.e. as far out as double precision can represent the curve at all, and
// orders of magnitude beyond any head a solve will see.
//
// UNLIKE vgm_ and bc_, there is NO saturated branch: the exponential is
// continued through psiC = 0 into psiC < 0 rather than clamped to
// (theta_w, k_rw) = (thetaS, 1).  That is deliberate and it is not cosmetic.
// Clamping sets DthetaW_DpsiC = 0, hence dm = drhom*thetaW, which at beta = 0
// is exactly zero -- and the low-order diagonal in Richards.h is
//
//     globalJacobian[ii] += bc_mask[i]*(MLi*dm/dt + J_ii) + (1-bc_mask[i])
//
// where MLi*dm/dt outweighs the graph term J_ii by ~3 orders at a small dt.  A
// free DOF that reaches psi >= 0 therefore keeps its residual row but loses its
// diagonal, takes a correction ~3 orders too large, falls back below zero where
// dm > 0 again, and overshoots once more: Newton locks into an exact period-2
// orbit that no tolerance or step cap will break.  vgm_/bc_ carry the same
// clamp harmlessly because the only node pinned at psi = 0 is normally a
// Dirichlet node, which bc_mask = 0 replaces with an identity row; Gardner
// reaches psi = 0 at *free* nodes because its diffusivity is
// Ks/(alpha*thetaSR), constant over the whole curve and typically orders above
// van Genuchten's in the unsaturated range, so a sharp boundary layer drives
// interior nodes into saturation within one step.
//
// The continuation is the same exponential, so the closure stays C-inf and
// every derivative used to build the diagonal stays strictly positive.  It does
// mean k_rw > 1 and theta_w > thetaS for psi > 0, which is unphysical -- but
// Gardner is a psi <= 0 model with no saturated branch to speak of, and these
// values are only ever visited by transient iterates on the way back down.  Any
// case that genuinely ponds wants vgm_ or bc_, which cap properly.
// -----------------------------------------------------------------------------

// Dry-end cut for the Gardner inversions, in units of alpha*psiC.  exp(-7e2) is
// ~1e-304, one decade off denormal, so this is the widest cut that keeps S_e a
// normal double.
constexpr double gardner_alphaPsiCMax = 7.0e2;

// Wet-end rail on the SAME exponent, so exp(alpha*u) cannot overflow if an
// iterate runs away.  This is an arithmetic guard, not a saturation limit: at a
// typical alpha it sits hundreds of metres above ground, far outside any head a
// solve can reach without having already failed, and it is placed on the
// exponent (not on theta) so it can never fire near psi = 0 where the clamp
// above would cost the diagonal.
constexpr double gardner_alphaPsiMax = 5.0e1;

inline void gardner_wetting(const double psiC,
                            const double alpha,
                            const double n_vg,
                            const double thetaR,
                            const double thetaSR,
                            double &thetaW,
                            double &DthetaW_DpsiC,
                            double &KWr,
                            double &DKWr_DpsiC)
{
  (void)n_vg; // Gardner is a one-parameter curve
  // One branch for every psiC.  exp(-alpha*psiC) underflows to 0 past the dry
  // cut (theta_w -> thetaR, k_rw -> 0 with zero slope, all correct); the wet
  // rail below only bounds a runaway iterate.
  const double x  = -alpha * psiC;
  const double Se = std::exp(x < gardner_alphaPsiMax ? x : gardner_alphaPsiMax);
  const double dSe_DpsiC = (x < gardner_alphaPsiMax) ? (-alpha * Se) : 0.0;
  thetaW        = thetaR + thetaSR * Se;
  DthetaW_DpsiC = thetaSR * dSe_DpsiC;
  KWr           = Se;
  DKWr_DpsiC    = dSe_DpsiC;
}

// Analytic Gardner inverse: theta_w -> psiC.  Exact (the retention curve is a
// plain exponential), so this is the whole inversion whenever beta = 0.
//
// The admissible band is the forward curve's own range, dry cut to wet rail --
// NOT thetaW < thetaS.  Since gardner_wetting continues the exponential above
// psi = 0, theta_w > thetaS is a value the forward model genuinely produces and
// the inverse has to be able to return the u > 0 that generated it; cutting at
// thetaS would leave those DOFs holding a stale iterate whose theta is
// unrelated to the mass handed in, which is the conservation break the vgm_/bc_
// dry-cut comments describe, at the other end of the curve.
inline void gardner_invert_analytic(const double m,
                                    const double rho,
                                    const double alpha,
                                    const double n_vg,
                                    const double thetaR,
                                    const double thetaSR,
                                    double &u)
{
  (void)n_vg;
  const double thetaW = m / rho;
  const double SeMin  = std::exp(-gardner_alphaPsiCMax);
  const double SeMax  = std::exp(gardner_alphaPsiMax);
  if (thetaW > thetaR + SeMin * thetaSR && thetaW < thetaR + SeMax * thetaSR) {
    const double Se = (thetaW - thetaR) / thetaSR;
    const double u_new = std::log(Se) / alpha; // = -psiC; sign follows Se vs 1
    if (std::isfinite(u_new)) u = u_new;
  }
}

// Inverse of the FULL forward mass m = rho(u)*theta_w(u) against the Gardner
// curve, i.e. including the exp(beta*u) the analytic inverse drops.
//
// Bracketed Newton, like bc_invert_newton and for the same conditioning
// reason: g(u) = rho*exp(beta*u)*(thetaR + thetaSR*exp(alpha*u)) - m is a sum
// of increasing exponentials, so it is monotone in u and [u_floor, u_rail]
// brackets the root by construction.  In the dry tail the beta*theta term
// dominates thetaSR*alpha*Se, so a plain Newton seeded from a stale iterate can
// walk off exactly as it does under BC; bracketing removes the dependence on
// the seed.
//
// Both ends of the bracket are the forward curve's own limits.  There is no
// wall at u = 0: gardner_wetting has no saturated branch, so psi > 0 is inside
// the model here and a mass above rho*thetaS inverts to the positive head that
// produced it rather than being flattened onto zero.  Nothing is refused for
// being "ponded" either -- an iterate that has overshot into psi > 0 is exactly
// the state the inversion has to be able to walk back down.
inline void gardner_invert_newton(const double m,
                                  const double rho,
                                  const double beta,
                                  const double alpha,
                                  const double n_vg,
                                  const double thetaR,
                                  const double thetaSR,
                                  double &u)
{
  (void)n_vg;

  auto g_of = [&](const double uu) -> double {
    const double x  = alpha * uu;
    const double Se = std::exp(x < gardner_alphaPsiMax ? x : gardner_alphaPsiMax);
    return rho * std::exp(beta * uu) * (thetaR + thetaSR * Se) - m;
  };

  const double u_rail = gardner_alphaPsiMax / alpha;
  if (g_of(u_rail) <= 0.0) { u = u_rail; return; } //root at or above the rail

  const double u_floor = -gardner_alphaPsiCMax / alpha;
  if (g_of(u_floor) > 0.0) { return; } //root drier than the band: leave u alone

  double lo = u_floor, hi = u_rail; // g(lo) <= 0 < g(hi)

  // Density-free analytic inverse as the opening guess; the bracket catches it
  // if the neglected exp(beta*u) puts it in the wrong place.
  {
    const double Se = (m / rho - thetaR) / thetaSR;
    double u_guess = 0.5 * (lo + hi);
    if (Se > 0.0) { // Se >= 1 is admissible now: it is the psi > 0 continuation
      const double u_analytic = std::log(Se) / alpha;
      if (std::isfinite(u_analytic) && u_analytic > lo && u_analytic < hi)
        u_guess = u_analytic;
    }
    u = u_guess;
  }
  if (!(u > lo && u < hi)) u = 0.5 * (lo + hi);

  // maxIts covers the bisection worst case over the widest bracket; from the
  // analytic seed Newton lands in a couple of steps for any sane beta.
  const int    maxIts = 100;
  const double tol    = 1e-12 * std::max(1.0, std::fabs(m));

  for (int it = 0; it < maxIts; ++it) {
    const double Se        = std::exp(alpha * u);
    const double thetaW    = thetaR + thetaSR * Se;
    const double dtheta_du = thetaSR * alpha * Se;

    const double rhom = rho * std::exp(beta * u);
    const double g    = rhom * thetaW - m;
    if (std::fabs(g) < tol) return;
    if (g > 0.0) hi = u; else lo = u;

    const double gp    = rhom * (beta * thetaW + dtheta_du);
    const double gpTol = 1e-14 * std::max(1.0, std::fabs(rhom * thetaW));
    double u_next = (std::fabs(gp) > gpTol) ? (u - g / gp) : 0.5 * (lo + hi);
    // Bisect whenever Newton leaves the bracket.
    if (!(u_next > lo && u_next < hi)) u_next = 0.5 * (lo + hi);

    if (std::fabs(u_next - u) <= 1e-15 * std::fabs(u)) return;
    u = u_next;
  }
  // Bracketed throughout, so the last iterate is the best available answer.
}

// -----------------------------------------------------------------------------
// Hornung-Messing manufactured law (Ait Hammou Oulhaj, Cances & Chainais-
// Hillairet 2018; Dielmann, Kuzmin & Kees, JNMA), written with u = -psiC:
//
//   S_e     = 1 - (4/pi^2) atan(u)^2,   k_rw = 1/(1+u^2)          for u < 0
//   S_e     = 1,                        k_rw = 1                  for u >= 0
//   theta_w = thetaR + thetaSR * S_e
//
// With thetaR = 0, thetaSR = pi^2/2 and K_s = 2 this is s(p) = pi^2/2 -
// 2 atan(p)^2 and eta(p) = 2/(1+p^2). S_e and k_rw are C^1 across u = 0 and
// S_e -> 0 only as u -> -infinity. alpha and n_vg are unused.
// -----------------------------------------------------------------------------
inline void hm_wetting(const double psiC,
                       const double alpha,
                       const double n_vg,
                       const double thetaR,
                       const double thetaSR,
                       double &thetaW,
                       double &DthetaW_DpsiC,
                       double &KWr,
                       double &DKWr_DpsiC)
{
  (void)alpha; (void)n_vg;
  if (psiC > 0.0) {
    const double u    = -psiC;
    const double at   = std::atan(u);
    const double opu2 = 1.0 + u * u;
    const double c    = 4.0 / (M_PI * M_PI);
    const double Se   = 1.0 - c * at * at;
    // d/dpsiC = -d/du
    thetaW        = thetaR + thetaSR * Se;
    DthetaW_DpsiC = thetaSR * (2.0 * c * at / opu2);
    KWr           = 1.0 / opu2;
    DKWr_DpsiC    = 2.0 * u / (opu2 * opu2);
  } else {
    thetaW        = thetaR + thetaSR;
    DthetaW_DpsiC = 0.0;
    KWr           = 1.0;
    DKWr_DpsiC    = 0.0;
  }
}

// Analytic inverse theta_w -> u on the unsaturated branch,
//   u = -tan((pi/2) sqrt(1 - S_e)).
// Same conventions as vgm_invert_analytic: at or above thetaS a negative u is
// moved to 0 (any u >= 0 has the same theta) and a nonnegative u is kept; below
// the dry cut u is set to the cut.
constexpr double hm_uMin = -1.0e8;
inline void hm_invert_analytic(const double m,
                               const double rho,
                               const double alpha,
                               const double n_vg,
                               const double thetaR,
                               const double thetaSR,
                               double &u)
{
  (void)alpha; (void)n_vg;
  const double thetaW = m / rho;
  const double thetaS = thetaR + thetaSR;
  const double c      = 4.0 / (M_PI * M_PI);
  const double SeMin  = 1.0 - c * std::atan(hm_uMin) * std::atan(hm_uMin);
  if (thetaW > thetaR + SeMin * thetaSR && thetaW < thetaS) {
    const double Se = (thetaW - thetaR) / thetaSR;
    const double u_new = -std::tan(0.5 * M_PI * std::sqrt(1.0 - Se));
    if (std::isfinite(u_new)) u = u_new;
  } else if (thetaW >= thetaS) {
    if (u < 0.0) u = 0.0;
  } else {
    u = hm_uMin;
  }
}

// Inverse of the full mass m = rho exp(beta u) theta_w(u). g(u) is increasing
// on the unsaturated branch, so a bracketed Newton/bisection on
// [hm_uMin, 0] is safe. If m is at or above the saturated mass the head is
// not determined by m; then u is left alone when u >= 0 and moved to 0 otherwise.
inline void hm_invert_newton(const double m,
                             const double rho,
                             const double beta,
                             const double alpha,
                             const double n_vg,
                             const double thetaR,
                             const double thetaSR,
                             double &u)
{
  auto g_of = [&](const double uu, double &gp) -> double {
    double thetaW, dth_dpsiC, kr, dkr;
    hm_wetting(-uu, alpha, n_vg, thetaR, thetaSR, thetaW, dth_dpsiC, kr, dkr);
    const double rhom = rho * std::exp(beta * uu);
    gp = rhom * (beta * thetaW - dth_dpsiC);
    return rhom * thetaW - m;
  };
  double gp;
  if (g_of(0.0, gp) <= 0.0) { if (u < 0.0) u = 0.0; return; } // saturated
  double lo = hm_uMin, hi = 0.0;
  if (g_of(lo, gp) > 0.0) { u = lo; return; }                  // drier than the cut
  {
    double ua = u;
    hm_invert_analytic(m * std::exp(-beta * u), rho, alpha, n_vg, thetaR, thetaSR, ua);
    u = (ua > lo && ua < hi) ? ua : 0.5 * (lo + hi);
  }
  const double tol = 1e-12 * std::max(1.0, std::fabs(m));
  for (int it = 0; it < 200; ++it) {
    const double g = g_of(u, gp);
    if (std::fabs(g) < tol) return;
    if (g > 0.0) hi = u; else lo = u;
    double u_next = (gp > 0.0) ? u - g / gp : 0.5 * (lo + hi);
    if (!(u_next > lo && u_next < hi)) u_next = 0.5 * (lo + hi);
    if (std::fabs(u_next - u) <= 1e-15 * std::max(1.0, std::fabs(u))) return;
    u = u_next;
  }
}

} // namespace psk
} // namespace richards
} // namespace proteus

namespace proteus
{
namespace m_comp_co2
{
namespace psk
{

// =============================================================================
// Two-phase PSK closures for m_comp_co2, parameterised by effective saturation
// S_e rather than by suction psiC.  These six routines are the only ones
// m_comp_co2.h calls.
//
//   vgm_pc_from_Se / bc_pc_from_Se                capillary curve p_c(S_e)
//   vgm_wetting_from_Se / bc_wetting_from_Se      theta_w(S_e), k_rw(S_e)
//   vgm_kr_nonwetting_from_Se / bc_..._from_Se    k_rn(S_e)
//
// The underlying constitutive models are NOT new here: the VGM branch is the
// same Mualem-van Genuchten curve as VGM::calc in proteus/pskRelations.h (the
// two forms are related by (1 - S_e^(1/m))^m = (alpha*psiC)^(n-1) * S_e), and
// the BC branch is Brooks-Corey-Burdine, the same as BCB::calc there.  What
// these routines add, and pskRelations.h does not have, is the regularisation
// the compositional Newton solve needs: second derivatives d2p_c/dS_e2, the C1
// cubic-Hermite ramp across gas appearance at S_e = 1, the linear cap on the
// diverging p_c tail below Se_min_pc, the residual-gas remap Se_trap, and the
// Hermite bridge over k_rn's triple zero at S_e = 1.  Unifying the two layers
// is post-1.9.0 work.
//
// The psiC-based closures (vgm_wetting, bc_wetting, vgm_kr_nonwetting,
// bc_kr_nonwetting and the four invert routines) used to be duplicated here as
// well.  They had no call site in m_comp_co2.h and were a stale fork of the
// Richards versions, so they have been removed; the maintained copies live in
// proteus/pskRelations.h under namespace proteus::richards::psk.
//
// vgm_pc_from_Se : Inverse capillary curve, given effective saturation
//                  S_e = (theta_w - theta_R) / theta_SR (= sBar).
//                  pcBar = (S_e^(-1/m) - 1)^(1/n);  pc = pcBar / alpha.
//                  Returns (pc, dpc/dSe) with dpc/dSe < 0. Both clamped to
//                  0 at S_e >= 1 (no capillary suction in the fully-wet
//                  branch); near S_e = 0 the value diverges - clamped via
//                  a small floor so derivatives remain finite.
// =============================================================================

inline void vgm_pc_from_Se(const double Se,
                           const double alpha,
                           const double n_vg,
                           double &pc,
                           double &Dpc_DSe,
                           double &D2pc_DSe2)
{
  const double m_vg = 1.0 - 1.0 / n_vg;
  // Floor Se from below to keep pcBar_n bounded for Se near 0 (gas-saturated).
  // Raised from 1e-12 to 1e-3 to bound pc at the saturation cap; otherwise
  // pc reaches ~1e12/alpha at the floor and floating-point cancellation in
  // the antisymmetric edge flux F_ij + F_ji = 0 leaks mass on the order of
  // |pc|*eps_machine per edge. With the 1e-3 floor, |pc_max| ~ 1e3/alpha and
  // the antisymmetric roundoff drops by ~9 orders of magnitude.  bc_pc_from_Se
  // bounds the same tail with its own cap (Se_min_pc), but the two are not
  // equivalent: below the floor VGM holds pc flat (Dpc_DSe = 0 via
  // dSeStar_dSe), whereas BC extrapolates linearly with a finite slope.
  const double SeStar  = (Se < 1.0e-3) ? 1.0e-3 : Se;
  const double SemInvM = pow(SeStar, -1.0 / m_vg);
  const double pcBar_n_raw = SemInvM - 1.0;
  // ------------------------------------------------------------------------
  // VGM regularization: SOFT FLOOR on pcBar_n.
  //
  //   pcBar_n = pcBar_n_raw + eps_pcBar
  //
  // This is C-infinity smooth across Se -> 1 (no kinks), bounds Dpc_DSe by
  // pcBar_n_soft^(1/n - 1) * (constant), and recovers the physical
  // pcBar_n_raw whenever raw >> eps_pcBar (i.e., away from full saturation).
  //
  // The earlier hard-cutoff regularization (pcBar_n = max(raw, eps)) made
  // Dpc_DSe C0-discontinuous at the threshold; Newton would stall when its
  // iterate crossed the threshold mid-step. The soft floor eliminates that.
  //
  // Cost: pc(Se = 1) = (eps_pcBar)^(1/n)/alpha ~ 0.001 instead of exactly 0.
  // Physically negligible (any DOF actually AT Se = 1 has zero gas flux
  // already because k_rn = 0 there).
  //
  // The legacy `if (Se >= 1.0)` and `pcBar_n_raw <= 0.0` early-returns are
  // dropped: with the soft floor, the formula is well-defined everywhere
  // including Se = 1 (pcBar_n_raw can dip slightly negative due to roundoff
  // when Se >= 1, but pcBar_n_soft remains >= eps_pcBar/2 > 0 as long as
  // |pcBar_n_raw| < eps_pcBar/2 which holds for Se in [1 - small, 1 + small]).
  // ------------------------------------------------------------------------
  const double eps_pcBar = 1.0e-3;
  // For Se >= 1 the raw value can be slightly negative due to roundoff;
  // clamp from below to keep the soft-floored pcBar_n strictly positive.
  const double pcBar_n_raw_clamped = (pcBar_n_raw > -0.5 * eps_pcBar)
                                       ? pcBar_n_raw
                                       : -0.5 * eps_pcBar;
  const double pcBar_n = pcBar_n_raw_clamped + eps_pcBar;
  const double pcBar = pow(pcBar_n, 1.0 / n_vg);
  pc = pcBar / alpha;

  // First derivative -- C-infinity smooth in Se now.
  //   d(pcBar_n)/d(Se)   = d(pcBar_n_raw)/d(Se)  (since eps_pcBar is constant)
  //                      = -(1/m) * Se^(-1/m - 1)
  //                      = -(1/m) * SemInvM / SeStar
  //   d(pcBar)/d(Se)     = (1/n) * pcBar_n^(1/n - 1) * d(pcBar_n)/d(Se)
  // For Se > 1 (numerical excursion) we hold dpcBar_n/dSe at its Se = 1 value
  // (which is small) -- no discontinuity introduced because pcBar_n_raw was
  // clamped above.
  const double dpcBar_n_dSe = -(1.0 / m_vg) * (SemInvM / SeStar);
  const double dpcBar_dSe   = (1.0 / n_vg) * (pcBar / pcBar_n) * dpcBar_n_dSe;
  // pc is built from SeStar = max(Se, 1e-3); below the floor pc is FLAT, so the
  // chain rule gives d(pc)/d(Se) = d(pc)/d(SeStar) * d(SeStar)/d(Se) with
  // d(SeStar)/d(Se) = 0 for Se < 1e-3.  Without this factor pc(Se) is flat but
  // Dpc_DSe is nonzero -> residual/Jacobian inconsistency at gas-saturated nodes
  // (Se_a -> 0), which stalls Newton near the saturation ceiling.
  const double dSeStar_dSe = (Se < 1.0e-3) ? 0.0 : 1.0;
  Dpc_DSe = (dpcBar_dSe / alpha) * dSeStar_dSe;

  // Second derivative (gas-eq (1,1) capillary sensitivity), also C-infinity.
  //   d2(pcBar_n)/d(Se)2 = (1/m)*(1/m + 1) * Se^(-1/m - 2)
  //                      = (1/m)*(1/m + 1) * SemInvM / Se^2
  //   d2(pcBar)/d(Se)2   = (1/n) * [(1/n - 1) * pcBar_n^(1/n - 2) * (d pcBar_n / dSe)^2
  //                                  + pcBar_n^(1/n - 1) * d2(pcBar_n)/d(Se)2]
  const double inv_m = 1.0 / m_vg;
  const double inv_n = 1.0 / n_vg;
  const double d2pcBar_n_dSe2 = inv_m * (inv_m + 1.0) * (SemInvM / (SeStar * SeStar));
  const double pcBar_n_pow_a  = pow(pcBar_n, inv_n - 2.0);                  // pcBar_n^(1/n - 2)
  const double pcBar_n_pow_b  = pow(pcBar_n, inv_n - 1.0);                  // pcBar_n^(1/n - 1)
  const double d2pcBar_dSe2 = inv_n * ((inv_n - 1.0) * pcBar_n_pow_a * dpcBar_n_dSe * dpcBar_n_dSe
                                       + pcBar_n_pow_b * d2pcBar_n_dSe2);
  D2pc_DSe2 = (d2pcBar_dSe2 / alpha) * dSeStar_dSe;   // 0 below the Se floor (flat pc)
}

// =============================================================================
// Brooks-Corey (Burdine relative-permeability) closures.
// Parallel API to the vgm_* functions: same parameter slots, same output args.
// In the BC parameterisation the second numeric parameter is the pore-size
// index lambda (taking the slot of vgm's n_vg). alpha = 1/p_d is the inverse
// entry-pressure head.
//
//   S_e = (alpha * p_c)^(-lambda)              for alpha*p_c >= 1
//   S_e = 1                                    for alpha*p_c <  1   (saturated)
//   theta_w = theta_R + theta_SR * S_e
//   k_rw    = S_e^((2+3*lambda)/lambda)        (Burdine)
//   k_rn    = (1-S_e)^2 * (1 - S_e^((2+lambda)/lambda))
//   p_c(S_e) = S_e^(-1/lambda) / alpha         (analytic inverse)
//
// BC has a discontinuity in (k_rw, k_rn, theta_w) derivatives at p_c = p_d
// (alpha*p_c = 1), i.e. at S_e = 1.  Unlike the raw model, the routines below
// regularise it: bc_pc_from_Se carries the C1 Hermite ramp across S_e = 1 and
// the linear cap below Se_min_pc, and bc_kr_nonwetting_from_Se bridges k_rn's
// triple zero at S_e = 1.  Each is documented at its own definition.
// =============================================================================

inline void bc_pc_from_Se(const double Se,
                          const double alpha,
                          const double lam,
                          double &pc,
                          double &Dpc_DSe,
                          double &D2pc_DSe2)
{
  // ----------------------------------------------------------------------
  // Brooks-Corey p_c(S_e) with smoothed transition out of the physical range
  // (so Newton can move through Se = 1 without seeing a value-jump).
  //
  //   Se in [0, 1):       pc = (1/alpha) * Se^(-1/lambda)  (standard BC)
  //                       At Se -> 1: pc -> 1/alpha (the "entry pressure"
  //                       p_d). At Se -> 0: pc -> +infinity (asymptotic).
  //
  //   Se in [1, 1+delta]: pc smoothly ramps from 1/alpha down to 0 using a
  //                       cubic Hermite smoothstep phi(t) = 1 - 3t^2 + 2t^3
  //                       in t = (Se - 1)/delta. phi(0) = 1, phi(1) = 0,
  //                       phi'(0) = phi'(1) = 0.
  //
  //   Se > 1 + delta:     pc = 0.
  //
  // Value continuity at Se = 1: pc(1-) = pc(1+) = 1/alpha. ENTRY PRESSURE
  // PRESERVED. The previous "if (Se >= 1.0) { pc = 0 }" cutoff produced a
  // finite jump of size 1/alpha at Se = 1 and stalled Newton whenever an
  // iterate crossed this threshold.
  //
  // C1 at Se = 1: the Se >= 1 ramp's left slope is matched to the BC branch
  // slope -(1/lambda)/alpha (see that branch), so dpc/dSe is now continuous
  // across gas appearance.  (Previously the ramp left with slope 0, leaving a
  // slope kink that mispredicted Newton steps overshooting across Se = 1.)
  // ----------------------------------------------------------------------
  const double delta_smooth = 5.0e-2;

  if (Se >= 1.0 + delta_smooth) {
    pc        = 0.0;
    Dpc_DSe   = 0.0;
    D2pc_DSe2 = 0.0;
    return;
  }

  if (Se >= 1.0) {
    // Non-physical overshoot region (S_g < 0, only reached when a Newton iterate
    // crosses gas appearance from above).  Cubic-Hermite ramp from pc(1) = 1/alpha
    // (the Brooks-Corey entry pressure p_d) down to pc(1+delta) = 0.
    //
    // C1 ACROSS GAS APPEARANCE: the LEFT-end slope is matched to the BC branch
    // slope at Se = 1, dpc/dSe = -1/(alpha*lam) (see the Se in [Se_min_pc,1)
    // branch below).  The earlier ramp used phi'(0) = 0, leaving a slope kink at
    // Se = 1 (BC side -p_d/lam vs ramp side 0); a Newton iterate overshooting
    // across Se = 1 then saw an inconsistent linearization and mispredicted the
    // step.  Right end keeps value 0 and slope 0 so pc joins the pc = 0 region C1.
    //
    // Hermite basis on t = (Se - 1)/delta in [0,1] with endpoints
    //   left  (t=0): value p_d = 1/alpha, slope_t m0_t = (-1/(alpha*lam))*delta
    //   right (t=1): value 0,             slope_t 0
    // so only H00 (value) and H10 (left slope) contribute.
    const double inv_alpha = 1.0 / alpha;
    const double m0_t = (-inv_alpha / lam) * delta_smooth;   // BC slope*delta at Se=1
    const double t  = (Se - 1.0) / delta_smooth;
    const double t2 = t * t, t3 = t2 * t;
    const double H00  = 2.0 * t3 - 3.0 * t2 + 1.0;
    const double H10  = t3 - 2.0 * t2 + t;
    const double dH00 = 6.0 * t2 - 6.0 * t;
    const double dH10 = 3.0 * t2 - 4.0 * t + 1.0;
    const double d2H00 = 12.0 * t - 6.0;
    const double d2H10 = 6.0 * t - 4.0;
    pc        = H00 * inv_alpha + H10 * m0_t;
    Dpc_DSe   = (dH00 * inv_alpha + dH10 * m0_t) / delta_smooth;
    D2pc_DSe2 = (d2H00 * inv_alpha + d2H10 * m0_t) / (delta_smooth * delta_smooth);
    return;
  }

  // ----------------------------------------------------------------------
  // Gas-saturated cap.  BC's pc and dpc/dSe diverge as Se -> 0 (i.e.
  // S_n -> 1 - S_wr).  In the FluidFlower disk-injection case this kills
  // Newton once the disk fills toward the saturation ceiling -- the
  // Jacobian sees |dpc/dSe| ~ 1e2 .. 1e8 m/unit and the S_n step blows up.
  //
  // Fix: below Se_min_pc, freeze the BC tangent so pc and dpc/dSe stay
  // finite and the Jacobian stays bounded.  C^1 continuous at Se_min_pc
  // (value + slope match by construction).  d2pc/dSe2 has a finite jump
  // at the seam, which Newton tolerates (only pc and dpc/dSe enter the
  // residual and Jacobian).  Standard "Pc cap" trick from reservoir
  // simulation; doesn't change pc above Se_min_pc, so undisturbed in
  // normal operation.
  //
  // Tuning Se_min_pc: smaller = sharper cap (closer to true BC), larger
  // = more robust Newton but more physics distortion near the gas-sat
  // endpoint.  1e-2 is a typical default; reduce to 1e-3 if the cap is
  // active across an unacceptable fraction of the domain.
  // ----------------------------------------------------------------------
  const double Se_min_pc = 1.0e-2;
  if (Se < Se_min_pc) {
    const double SemInvLam_min = pow(Se_min_pc, -1.0 / lam);
    const double pc_min        = SemInvLam_min / alpha;
    const double slope_min     = -(SemInvLam_min / Se_min_pc) / (alpha * lam);
    pc        = pc_min + slope_min * (Se - Se_min_pc);
    Dpc_DSe   = slope_min;
    D2pc_DSe2 = 0.0;
    return;
  }

  // Standard BC in the physical range Se in [Se_min_pc, 1).
  const double SemInvLam = pow(Se, -1.0 / lam);
  pc = SemInvLam / alpha;
  // dpc/dS_e = -(1/(alpha*lambda)) * S_e^(-1/lambda - 1)
  //         = -(1/(alpha*lambda)) * SemInvLam / S_e
  Dpc_DSe = -(SemInvLam / Se) / (alpha * lam);
  // d2pc/dS_e2 = (1/(alpha*lambda)) * (1/lambda + 1) * S_e^(-1/lambda - 2)
  //            = (1/(alpha*lambda)) * (1/lambda + 1) * SemInvLam / S_e^2
  const double inv_lam = 1.0 / lam;
  D2pc_DSe2 = inv_lam * (inv_lam + 1.0) * (SemInvLam / (Se * Se)) / alpha;
}

// =============================================================================
// theta_w(Se) and k_r(Se) (Phase B Step 3a).  Same physics as the psiC-based
// closures in proteus/pskRelations.h, but parameterised directly by effective
// saturation Se.  Used by the formulation-(psi_w, S_w) wetting equation, where
// saturation is the primary variable rather than something inverted from psiC.
//
// Conventions:
//   Se in [0, 1] is the effective saturation = (theta_w - theta_R)/theta_SR.
//   In formulation B, u_v IS S_w; if a residual saturation theta_R/theta_S is
//   in play the caller must convert before calling these.
//   Outputs are derivatives with respect to Se (NOT psiC).
// =============================================================================

inline void vgm_wetting_from_Se(const double Se,
                                const double /*alpha*/,    // unused, kept for API parity
                                const double n_vg,
                                const double thetaR,
                                const double thetaSR,
                                double &thetaW,
                                double &DthetaW_DSe,
                                double &KWr,
                                double &DKWr_DSe)
{
  const double m_vg = 1.0 - 1.0 / n_vg;
  thetaW       = thetaR + thetaSR * Se;
  DthetaW_DSe  = thetaSR;
  if (Se >= 1.0) {
    KWr = 1.0;
    DKWr_DSe = 0.0;
    return;
  }
  if (Se <= 0.0) {
    KWr = 0.0;
    DKWr_DSe = 0.0;
    return;
  }
  // Floor for the powers near the endpoints.
  const double SeStar = (Se < 1.0e-12) ? 1.0e-12 : Se;
  const double SeInvM = pow(SeStar, 1.0 / m_vg);          // Se^(1/m)
  const double x_     = 1.0 - SeInvM;                     // 1 - Se^(1/m)
  const double xStar  = (x_ < 1.0e-12) ? 1.0e-12 : x_;
  const double y_     = pow(xStar, m_vg);                 // (1 - Se^(1/m))^m
  const double term   = 1.0 - y_;                         // 1 - (1 - Se^(1/m))^m
  const double sqrtSe = sqrt(SeStar);
  const double sqrtSe_safe = (sqrtSe < 1.0e-12) ? 1.0e-12 : sqrtSe;
  // Mualem k_rw = sqrt(Se) * [1 - (1 - Se^(1/m))^m]^2
  KWr = sqrtSe * term * term;
  // dterm/dSe = x^(m-1) * Se^(1/m - 1)  (chain rule through y = x^m, x = 1 - Se^(1/m))
  const double Dterm_DSe = pow(xStar, m_vg - 1.0) * pow(SeStar, 1.0 / m_vg - 1.0);
  DKWr_DSe = (0.5 / sqrtSe_safe) * term * term + 2.0 * sqrtSe * term * Dterm_DSe;
}

inline void vgm_kr_nonwetting_from_Se(const double Se_w,
                                      const double /*alpha*/,
                                      const double n_vg,
                                      double &KNr,
                                      double &DKNr_DSe,
                                      const double Se_trap = 1.0)
{
  // Gas-only residual trapping: see bc_kr_nonwetting_from_Se.  Remap the
  // wetting Se_w over the mobile range [0,Se_trap]; k_rn=0 for S_g<=S_gr.
  // Se_trap=1 -> original.  DKNr_DSe is d k_rn/d Se_w (1/Se_trap folded in).
  const double inv_trap = 1.0 / Se_trap;
  const double Se       = Se_w * inv_trap;
  const double m_vg = 1.0 - 1.0 / n_vg;
  if (Se >= 1.0) {
    KNr = 0.0;
    DKNr_DSe = 0.0;
    return;
  }
  if (Se <= 0.0) {
    // S_n = 1 -> k_rn = 1 in the limit; derivative degenerate.
    KNr = 1.0;
    DKNr_DSe = 0.0;
    return;
  }
  const double SeStar = (Se < 1.0e-12) ? 1.0e-12 : Se;
  const double SeInvM = pow(SeStar, 1.0 / m_vg);
  const double x_     = 1.0 - SeInvM;
  const double xStar  = (x_ < 1.0e-12) ? 1.0e-12 : x_;
  // Mualem k_rn = sqrt(1 - Se) * (1 - Se^(1/m))^(2m)
  const double y2m    = pow(xStar, 2.0 * m_vg);
  const double oneMinusSe     = 1.0 - Se;
  const double oneMinusSeStar = (oneMinusSe < 1.0e-12) ? 1.0e-12 : oneMinusSe;
  const double sqrt_oms       = sqrt(oneMinusSeStar);
  KNr = sqrt_oms * y2m;
  // d(y^(2m))/dSe = 2m * x^(2m - 1) * dx/dSe = -2 * x^(2m - 1) * Se^(1/m - 1)
  const double Dy2m_DSe = -2.0 * pow(xStar, 2.0 * m_vg - 1.0) * pow(SeStar, 1.0 / m_vg - 1.0);
  DKNr_DSe = ((-0.5 / sqrt_oms) * y2m + sqrt_oms * Dy2m_DSe) * inv_trap;
}

inline void bc_wetting_from_Se(const double Se,
                               const double /*alpha*/,
                               const double lam,
                               const double thetaR,
                               const double thetaSR,
                               double &thetaW,
                               double &DthetaW_DSe,
                               double &KWr,
                               double &DKWr_DSe)
{
  thetaW       = thetaR + thetaSR * Se;
  DthetaW_DSe  = thetaSR;
  if (Se >= 1.0) {
    // Saturated branch; BC has a one-sided derivative jump here.
    KWr = 1.0;
    DKWr_DSe = 0.0;
    return;
  }
  if (Se <= 0.0) {
    KWr = 0.0;
    DKWr_DSe = 0.0;
    return;
  }
  // Burdine k_rw = Se^((2 + 3*lambda)/lambda)
  const double exp_w = (2.0 + 3.0 * lam) / lam;
  KWr      = pow(Se, exp_w);
  DKWr_DSe = exp_w * pow(Se, exp_w - 1.0);
}

inline void bc_kr_nonwetting_from_Se(const double Se_w,
                                     const double /*alpha*/,
                                     const double lam,
                                     double &KNr,
                                     double &DKNr_DSe,
                                     const double Se_trap = 1.0)
{
  // ---------------------------------------------------------------------
  // Gas-only residual trapping (Jun 2026): remap the wetting effective
  // saturation Se_w onto [0,1] over the MOBILE gas range Se_w in [0,Se_trap],
  //   Se_trap = (1 - S_wr - S_gr)/(1 - S_wr) = 1 - S_gr/(1 - S_wr).
  // Gas is immobile (k_rn = 0) for Se_w >= Se_trap, i.e. S_g <= S_gr.
  // Se_trap = 1.0 recovers the original no-trapping closure.  Only k_rn is
  // remapped here (gas-only) -- k_rw and p_c keep the drainage Se_w.  The
  // returned DKNr_DSe is d k_rn / d Se_w (the 1/Se_trap chain factor is folded
  // in), so every caller's existing dkrn*dSe_dp chain stays correct unchanged.
  const double inv_trap = 1.0 / Se_trap;
  const double Se       = Se_w * inv_trap;
  // ---------------------------------------------------------------------
  // Option A regularisation (May 2026): cubic-Hermite endpoint bridge for
  // k_rn near the wet endpoint Se = 1.
  //
  // Burdine k_rn = (1 - Se)^2 * (1 - Se^((2+lambda)/lambda)) has a TRIPLE
  // zero at Se = 1 (value, 1st, and 2nd derivative all vanish).  Newton
  // cannot predict gas-mobile breakthrough from local linearisation: when
  // its trial step lands at Se = 1 - epsilon, the residual jumps and the
  // step direction was wrong.  This is the FluidFlower disk-injection
  // breakthrough stall (k_rn turning on at t ~ 47 s under benchmark
  // conditions).
  //
  // Fix: on a narrow window Se in [Se_a, 1] with Se_a = 1 - delta_brn,
  // replace BC by a cubic Hermite that matches BC value+slope at Se = Se_a
  // and ends at (k_rn = 0, dk_rn/dSe = -s_min_brn) at Se = 1.  Bulk physics
  // (Se in [0, Se_a]) is pure Brooks-Corey.
  //
  // Tuning:
  //   delta_brn = 5e-2: bridge spans 5% of the S_e range
  //   s_min_brn = 1e-3: |slope| at Se = 1; controls how much "advance
  //               warning" Newton gets about k_rn turning on.  Larger ->
  //               earlier breakthrough prediction, more distortion in the
  //               bridge.  At Se = 1-delta the bridge value is ~21% above
  //               pure BC k_rn (negligible compared to absolute magnitudes
  //               here ~1e-4).
  //
  // For Se > 1 (Newton overshoot) the value clamps to 0 but the derivative
  // continues at -s_min_brn so the chain rule has C^1 continuity across
  // the seam.
  // ---------------------------------------------------------------------
  const double delta_brn = 5.0e-2;
  const double s_min_brn = 1.0e-3;
  const double Se_a      = 1.0 - delta_brn;

  if (Se <= 0.0) {
    // S_n = 1 - S_wr (gas-saturated): non-wetting endpoint.
    KNr      = 1.0;
    DKNr_DSe = 0.0;
    return;
  }
  if (Se >= 1.0) {
    // Beyond the wet endpoint: value clamps to 0, derivative continues at
    // -s_min_brn for C^1 consistency with the bridge at Se = 1.
    KNr      = 0.0;
    DKNr_DSe = -s_min_brn * inv_trap;
    return;
  }

  const double exp_n = (2.0 + lam) / lam;

  if (Se < Se_a) {
    // Bulk Brooks-Corey (Burdine) k_rn = (1 - Se)^2 * (1 - Se^((2+lambda)/lambda))
    const double SeExp      = pow(Se, exp_n);
    const double oneMinusSe = 1.0 - Se;
    const double Y          = 1.0 - SeExp;
    KNr      = oneMinusSe * oneMinusSe * Y;
    DKNr_DSe = (-2.0 * oneMinusSe * Y
             - oneMinusSe * oneMinusSe * exp_n * pow(Se, exp_n - 1.0)) * inv_trap;
    return;
  }

  // Cubic-Hermite bridge on [Se_a, 1].
  // Anchors:
  //   left  (t=0): k_rn = fa = k_rn_BC(Se_a),  dk/dSe = da = dk_rn_BC/dSe(Se_a)
  //   right (t=1): k_rn = 0,                    dk/dSe = -s_min_brn
  // Parameterised by t = (Se - Se_a)/delta_brn in [0, 1].
  const double SeExp_a    = pow(Se_a, exp_n);
  const double oneMinusSe_a = 1.0 - Se_a;
  const double Y_a        = 1.0 - SeExp_a;
  const double fa = oneMinusSe_a * oneMinusSe_a * Y_a;
  const double da = -2.0 * oneMinusSe_a * Y_a
                  - oneMinusSe_a * oneMinusSe_a * exp_n * pow(Se_a, exp_n - 1.0);

  const double t  = (Se - Se_a) / delta_brn;
  const double t2 = t * t;
  const double t3 = t2 * t;

  // Hermite basis (only H00, H10, H11 contribute since fb = 0).
  const double H00 = 2.0 * t3 - 3.0 * t2 + 1.0;
  const double H10 = t3 - 2.0 * t2 + t;
  const double H11 = t3 - t2;
  const double dH00 = 6.0 * t2 - 6.0 * t;
  const double dH10 = 3.0 * t2 - 4.0 * t + 1.0;
  const double dH11 = 3.0 * t2 - 2.0 * t;

  KNr      = H00 * fa + H10 * delta_brn * da + H11 * delta_brn * (-s_min_brn);
  DKNr_DSe = (dH00 * fa + dH10 * delta_brn * da
              + dH11 * delta_brn * (-s_min_brn)) / delta_brn * inv_trap;
}

} // namespace psk
} // namespace m_comp_co2
} // namespace proteus

/** @} */
#endif
