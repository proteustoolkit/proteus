#ifndef ELASTOPLASTIC_H
#define ELASTOPLASTIC_H
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
//extern "C"
//{
#include PROTEUS_LAPACK_H
#include PROTEUS_BLAS_H
//}
#include "proteus_blas.h"
#include "CompKernel.h"
#include "ModelFactory.h"
#include "mohrCoulomb.h"
#include "mohrCoulomb2.h"
#include "vonMises.h"
#include "tresca.h"
#include "druckerPrager.h"
#include "mcdp.h"
#include "../mprans/ArgumentsDict.h"
#include "pyarray.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace proteus
{
    class ElastoPlastic_base
    {
        public:
            virtual ~ElastoPlastic_base(){}
            virtual void calculateResidual(arguments_dict& args)=0;
            virtual void calculateJacobian(arguments_dict& args)=0;
    };

    template<class CompKernelType,
        int nSpace,
        int nQuadraturePoints_element,
        int nDOF_mesh_trial_element,
        int nDOF_trial_element,
        int nDOF_test_element,
        int nQuadraturePoints_elementBoundary>
            class ElastoPlastic : public ElastoPlastic_base
        {
            public:
                CompKernelType ck;

                const int nDOF_test_X_trial_element;

                EIndex<nSpace> ei;

                const int X,Y,Z,
                      XX,XY,XZ,
                      YX,YY,YZ,
                      ZX,ZY,ZZ,
                      sXX,sXY,sXZ,
                      sYX,sYY,sYZ,
                      sZX,sZY,sZZ,
                      nSymTen,
                      XHX,XHY,
                      YHX,YHY,
                      ZHX,ZHY,
                      HXHX,HXHY,
                      HYHX,HYHY;

                ElastoPlastic():
                    ck(),
                    nDOF_test_X_trial_element(nDOF_test_element*nDOF_trial_element),
                    X(ei.X),
                    Y(ei.Y),
                    Z(ei.Z),
                    XX(ei.XX),XY(ei.XY),XZ(ei.XZ),
                    YX(ei.YX),YY(ei.YY),YZ(ei.YZ),
                    ZX(ei.ZX),ZY(ei.ZY),ZZ(ei.ZZ),
                    sXX(ei.sXX),sXY(ei.sXY),sXZ(ei.sXZ),
                    sYX(ei.sYX),sYY(ei.sYY),sYZ(ei.sYZ),
                    sZX(ei.sZX),sZY(ei.sZY),sZZ(ei.sZZ),
                    nSymTen(ei.nSymTen),
                    XHX(ei.XHX),XHY(ei.XHY),
                    YHX(ei.YHX),YHY(ei.YHY),
                    ZHX(ei.ZHX),ZHY(ei.ZHY),
                    HXHX(ei.HXHX),HXHY(ei.HXHY),
                    HYHX(ei.HYHX),HYHY(ei.HYHY)
            {}

                inline void calculateStrain(double* D, double* strain)
                {
                    //Voigt notation from Belytschko, Liu, Moran
                    strain[sXX] = D[XX];//du/dx
                    strain[sYY] = D[YY];//dv/dy
                    strain[sZZ] = D[ZZ];//dw/dz
                    strain[sYZ] = D[YZ]+D[ZY];//(dv/dz + dz/dy)
                    strain[sXZ] = D[XZ]+D[ZX];//(du/dz + dw/dx)
                    strain[sXY] = D[XY]+D[YX];//(du/dy + dv/dx)
                }

                inline void elasticStress(const double* C, const double* strain, double* stress)
                {
                    stress[sXX] = C[0]*strain[sXX] + C[1]*(strain[sYY] +strain[sZZ]);
                    stress[sYY] = C[0]*strain[sYY] + C[1]*(strain[sXX] +strain[sZZ]);
                    stress[sZZ] = C[0]*strain[sZZ] + C[1]*(strain[sXX] +strain[sYY]);
                    stress[sYZ] = C[sYZ+sYZ*nSymTen]*strain[sYZ];
                    stress[sXZ] = C[sYZ+sYZ*nSymTen]*strain[sXZ];
                    stress[sXY] = C[sYZ+sYZ*nSymTen]*strain[sXY];
                }

                inline void elasticStrain(const double* Cinv, const double* stress, double* strain)
                {
                    strain[sXX] = Cinv[0]*stress[sXX] + Cinv[1]*(stress[sYY] +stress[sZZ]);
                    strain[sYY] = Cinv[0]*stress[sYY] + Cinv[1]*(stress[sXX] +stress[sZZ]);
                    strain[sZZ] = Cinv[0]*stress[sZZ] + Cinv[1]*(stress[sXX] +stress[sYY]);
                    strain[sYZ] = Cinv[sYZ+sYZ*nSymTen]*stress[sYZ];
                    strain[sXZ] = Cinv[sYZ+sYZ*nSymTen]*stress[sXZ];
                    strain[sXY] = Cinv[sYZ+sYZ*nSymTen]*stress[sXY];
                }

                inline void elasticModuli(const double& E,const double& nu,double* C,double* Cinv)
                {
                    for (int i=0;i<nSymTen;i++)
                        for (int j=0;j<nSymTen;j++)
                        {
                            C[i+nSymTen*j] = 0.0;
                            Cinv[i+nSymTen*j] = 0.0;
                        }
                    C[sXX+sXX*nSymTen] = (E/(1.0+nu))*(1.0 + (nu/(1.0-2.0*nu)));
                    C[sXX+sYY*nSymTen] = (E/(1.0+nu))*(nu/(1.0-2.0*nu));
                    C[sXX+sZZ*nSymTen] = (E/(1.0+nu))*(nu/(1.0-2.0*nu));
                    C[sYY+sXX*nSymTen] = (E/(1.0+nu))*(nu/(1.0-2.0*nu));
                    C[sYY+sYY*nSymTen] = (E/(1.0+nu))*(1.0 + (nu/(1.0-2.0*nu)));
                    C[sYY+sZZ*nSymTen] = (E/(1.0+nu))*(nu/(1.0-2.0*nu));
                    C[sZZ+sXX*nSymTen] = (E/(1.0+nu))*(nu/(1.0-2.0*nu));
                    C[sZZ+sYY*nSymTen] = (E/(1.0+nu))*(nu/(1.0-2.0*nu));
                    C[sZZ+sZZ*nSymTen] = (E/(1.0+nu))*(1.0 + (nu/(1.0-2.0*nu)));
                    C[sYZ+sYZ*nSymTen] = (E/(1.0+nu))*0.5;
                    C[sXZ+sXZ*nSymTen] = (E/(1.0+nu))*0.5;
                    C[sXY+sXY*nSymTen] = (E/(1.0+nu))*0.5;

                    Cinv[sXX+sXX*nSymTen] = 1.0/E;
                    Cinv[sXX+sYY*nSymTen] = -nu/E;
                    Cinv[sXX+sZZ*nSymTen] = -nu/E;
                    Cinv[sYY+sXX*nSymTen] = -nu/E;
                    Cinv[sYY+sYY*nSymTen] = 1.0/E;
                    Cinv[sYY+sZZ*nSymTen] = -nu/E;
                    Cinv[sZZ+sXX*nSymTen] = -nu/E;
                    Cinv[sZZ+sYY*nSymTen] = -nu/E;
                    Cinv[sZZ+sZZ*nSymTen] = 1.0/E;
                    Cinv[sYZ+sYZ*nSymTen] = ((1.0+nu)/E)*2.0;
                    Cinv[sXZ+sXZ*nSymTen] = ((1.0+nu)/E)*2.0;
                    Cinv[sXY+sXY*nSymTen] = ((1.0+nu)/E)*2.0;
                    //cek debug
                    double I[nSymTen*nSymTen];
                    for (int i=0;i<nSymTen;i++)
                    {
                        for (int j=0;j<nSymTen;j++)
                        {
                            I[i+j*nSymTen] = 0.0;
                            for(int k=0;k<nSymTen;k++)
                                I[i+j*nSymTen] += C[i+k*nSymTen]*Cinv[k+j*nSymTen];
                            if(i==j)
                                if(fabs(1.0-I[i+j*nSymTen]) > 1.0e-12)
                                    std::cout<<"I["<<i<<"]["<<j<<"]="<<I[i+j*nSymTen]<<std::endl;
                            if(i!=j)
                                if(fabs(I[i+j*nSymTen]) > 1.0e-12)
                                    std::cout<<"I["<<i<<"]["<<j<<"]="<<I[i+j*nSymTen]<<std::endl;
                        }
                    }
                }

                inline void evaluateConstitutiveModel(const double* materialProperties, const double* stress,
                        double& f, double* df, double* r, double* dr, double& stress_3)
                {
                    double g;
                    /* std::cout<<materialProperties[5]<<"phi \n" */
                    /*            <<materialProperties[7]<<"psi \n" */
                    /*            <<materialProperties[6]<<"c \n"<<std::endl; */
                    /* mohrCoulomb(materialProperties[5],//phi */
                    /*           materialProperties[7],//psi */
                    /*           materialProperties[6],//c */
                    /*           stress, */
                    /*           f, */
                    /*           df, */
                    /*           g, */
                    /*           r, */
                    /*           dr); */
                    /* mohrCoulomb2(materialProperties[5],//phi */
                    /*           materialProperties[7],//psi */
                    /*           materialProperties[6],//c */
                    /*           stress, */
                    /*           f, */
                    /*           df, */
                    /*           g, */
                    /*           r, */
                    /*           dr); */
                    /* if (! (std::isfinite(f) &&   */
                    /*          std::isfinite(df[sXX]) && std::isfinite(df[sYY]) && std::isfinite(df[sZZ]) && std::isfinite(df[sYZ]) && std::isfinite(df[sXZ]) && std::isfinite(df[sXY]) &&  */
                    /*          std::isfinite(r[sXX]) && std::isfinite(r[sYY]) && std::isfinite(r[sZZ]) && std::isfinite(r[sYZ]) && std::isfinite(r[sXZ]) && std::isfinite(r[sXY]))) */
                    /*     { */
                    /*       std::cout<<"f "<<f<<std::endl */
                    /*            <<"stress "<<stress[sXX]<<'\t'<<stress[sYY]<<'\t'<<stress[sZZ]<<'\t'<<stress[sYZ]<<'\t'<<stress[sXZ]<<'\t'<<stress[sXY]<<std::endl */
                    /*            <<"df "<<df[sXX]<<'\t'<<df[sYY]<<'\t'<<df[sZZ]<<'\t'<<df[sYZ]<<'\t'<<df[sXZ]<<'\t'<<df[sXY]<<std::endl */
                    /*            <<"r "<<r[sXX]<<'\t'<<r[sYY]<<'\t'<<r[sZZ]<<'\t'<<r[sYZ]<<'\t'<<r[sXZ]<<'\t'<<r[sXY]<<std::endl; */
                    /*     } */
                    /* vonMises(materialProperties[5],//phi */
                    /*           materialProperties[7],//psi */
                    /*           materialProperties[6],//c */
                    /*           stress, */
                    /*           f, */
                    /*           df, */
                    /*           g, */
                    /*           r, */
                    /*           dr); */
                    /* tresca(materialProperties[5],//phi */
                    /*           materialProperties[7],//psi */
                    /*           materialProperties[6],//c */
                    /*           stress, */
                    /*           f, */
                    /*           df, */
                    /*           g, */
                    /*           r, */
                    /*           dr); */
                    /* druckerPrager(materialProperties[5],//phi */
                    /*             materialProperties[7],//psi */
                    /*             materialProperties[6],//c */
                    /*             stress, */
                    /*             f, */
                    /*             df, */
                    /*             g, */
                    /*             r, */
                    /*             dr); */
                    mcdp(materialProperties[5],//phi
                            materialProperties[7],//psi
                            materialProperties[6],//c
                            stress,
                            f,
                            df,
                            g,
                            r,
                            dr,
                            stress_3);
                }

                inline void differenceJacobian(const double* materialProperties,const double* stress,const double& f,double *df, const double* r, double* dr)
                {
                    double f_delta,
                           df_delta[nSymTen],
                           stress_plus_delta[nSymTen],
                           r_delta[nSymTen],
                           dr_delta[nSymTen*nSymTen],stress_3;
                           for (int i=0;i<nSymTen;i++)
                           {
                               stress_plus_delta[i] = stress[i];
                           }
                           for (int j=0;j<nSymTen;j++)
                           {
                               double Delta_stress  = 1.0e-8*stress[j] + 1.0e-8;
                               stress_plus_delta[j] += Delta_stress;
                               evaluateConstitutiveModel(materialProperties,stress_plus_delta,f_delta,df_delta,r_delta,dr_delta,stress_3);
                               df[j] = (f_delta - f)/Delta_stress;
                               for(int k=0;k<nSymTen;k++)
                               {
                                   dr[k+nSymTen*j] = (r_delta[k] - r[k])/Delta_stress;
                               }
                               stress_plus_delta[j] = stress[j];
                           }
                }
                inline void evaluateCoefficients(int usePicard,
                        const double pore_pressure,
                        const double* materialProperties,
                        const double* strain0,
                        const double* strain_last,
                        const double* Delta_strain,
                        const double* plasticStrain_last,
                        double* plasticStrain,
                        double* stress,
                        double* dstress)
                {
                    int its=0, maxIts=25;
                    double E=materialProperties[0];
                    const double nu=materialProperties[1],
                    phi=materialProperties[5];
                    double f,
                           df[nSymTen],
                           r[nSymTen],
                           dr[nSymTen*nSymTen],
                           r0[nSymTen],
                           C[nSymTen*nSymTen],
                           Cinv[nSymTen*nSymTen],
                           A[nSymTen*nSymTen],
                           B[(nSymTen+1)*(nSymTen+1)],
                           a[nSymTen],
                           Delta_lambda,
                           Delta_lambda_old,
                           Delta_lambda_incr,
                           Delta_stress[nSymTen],
                           Delta_stress_old[nSymTen],
                           stress_old[nSymTen],
                           Delta_stress_incr[nSymTen],
                           minusDelta_plasticStrain[nSymTen],
                           effectiveStrain[nSymTen],
                           aNorm,
                           stressNorm,
                           f_atol=1.0e-6,
                           a_atol=1.0e-6,
                           dfA[nSymTen],
                           Aa[nSymTen],
                           Ar[nSymTen],
                           dfAa,
                           dfAr,
                           WORK[nSymTen],
                           R[nSymTen+1],
                           WORKR[nSymTen+1],
                           stress_3_init,stress_3;
                           PROTEUS_LAPACK_INTEGER N=nSymTen,NR=nSymTen+1,
                                                  INFO=0,
                                                  NRHS=1,
                                                  IPIV[nSymTen],
                                                  JPIV[nSymTen],
                                                  LWORK=nSymTen,
                                                  INFOR=0,
                                                  IPIVR[nSymTen+1],
                                                  JPIVR[nSymTen+1],
                                                  LWORKR=nSymTen+1;
                                                  double SCALE,SCALER;
                                                  char TRANS='N';
                                                  bool useSemiImplicit=false;
                                                  bool useDifferenceJacobian=false;
                                                  bool useFullPivoting=false;
                                                  bool useDumbNewton=false;
                                                  bool useContinuumTangentModulus=false;
                                                  bool useLineSearch=false;
                                                  bool useInnerPicard=false;
                                                  bool useOuterPicard=bool(usePicard);
                                                  bool predictorPhase = false;
                                                  int predictorPhaseIts=5;
                                                  std::vector<double> fhist,ahist;
                                                  //get modulus from initial strain
                                                  elasticModuli(E,nu,C,Cinv);
                                                  elasticStress(C,strain0,stress);
                                                  evaluateConstitutiveModel(materialProperties,stress,f,df,r,dr,stress_3_init);//cek hack, this is just to get stress_3_init
                                                  const double n_e = 0.6,
                                                        P_a = materialProperties[12],
                                                        K_i = 500.0;
                                                        if (P_a > 0.0)//cek hack, if not a soil then set P_a=0.0
                                                        {
                                                            E =K_i*P_a*pow(fmax(stress_3_init,P_a)/P_a,n_e);
                                                            //std::cout<<"E "<<E<<'\t'<<K_i<<'\t'<<P_a<<'\t'<<n_e<<'\t'<<stress_3_init<<std::endl;
                                                            elasticModuli(E,nu,C,Cinv);
                                                        }
                                                        //
                                                        //fully implicit Backward Euler integration
                                                        //
                                                        //input is strain_last, Delta_strain, and plasticStrain_last
                                                        //output is stress,dstress= dstress/dDelta_strain, and plasticStrain
                                                        //
                                                        //get stress at last step and evaluate r0 there for semi-implicit scheme because it lies on the yield surface
                                                        //
                                                        for (int i=0;i<nSymTen;i++)
                                                        {
                                                            effectiveStrain[i] = strain_last[i] - plasticStrain_last[i];
                                                        }
                                                        elasticStress(C,effectiveStrain,stress);
                                                        evaluateConstitutiveModel(materialProperties,stress,f,df,r,dr,stress_3);
                                                        for (int i=0;i<nSymTen;i++)
                                                        {
                                                            r0[i]=r[i];
                                                        }
                                                        //
                                                        //get elastic predictor stress
                                                        //
                                                        for (int i=0;i<nSymTen;i++)
                                                        {
                                                            Delta_lambda = 0.0;
                                                            Delta_stress[i] = 0.0;//tricky, this is not (stress - stress_last) but rather (stress - stress_elasticPredictor) which is 0 initially
                                                            minusDelta_plasticStrain[i] = 0.0;
                                                            plasticStrain[i] = plasticStrain_last[i];
                                                            effectiveStrain[i] = strain_last[i] + Delta_strain[i] - plasticStrain_last[i];
                                                            a[i] = 0.0;
                                                        }
                                                        elasticStress(C,effectiveStrain,stress);
                                                        evaluateConstitutiveModel(materialProperties,stress,f,df,r,dr,stress_3);
                                                        if (useDifferenceJacobian)
                                                        {
                                                            differenceJacobian(materialProperties,stress,f,df,r,dr);//recalculate df and dr
                                                        }
                                                        if (useSemiImplicit)//overwrite r and dr
                                                        {
                                                            for (int i=0;i<nSymTen;i++)
                                                            {
                                                                r[i]=r0[i];
                                                                for(int j=0;j<nSymTen;j++)
                                                                {
                                                                    dr[i+j*nSymTen] = 0.0;
                                                                }
                                                            }
                                                        }
                                                        aNorm=0.0;//a = minusDelta_plasticStrain + Delta_lambda*r (==0 initially)
                                                        double f_last=f;
                                                        //main Newton iteration loop
                                                        while ((f >= f_atol || aNorm >= a_atol) && its < maxIts) //since aNorm=0 this will be skipped if f < f_atol i.e. within the yield surface up to the tolerance)
                                                        {
                                                            //fhist.push_back(f);
                                                            //ahist.push_back(aNorm);
                                                            //std::cout<<"plastic it "<<its<<" aNorm "<<aNorm<<" f "<<f<<std::endl;
                                                            //
                                                            //calculate Newton increment lambda_incr using block gaussian elimination
                                                            //
                                                            for(int i=0;i<nSymTen;i++)
                                                            {
                                                                for(int j=0;j<nSymTen;j++)
                                                                {
                                                                    //
                                                                    //set to A^{-1} and factor below
                                                                    //
                                                                    A[i+j*nSymTen] = Cinv[i+j*nSymTen];
                                                                    if (!useInnerPicard)
                                                                        A[i+j*nSymTen] +=  Delta_lambda*dr[i+j*nSymTen];
                                                                }
                                                                //dfA[i] = df[i];
                                                                Aa[i] = a[i];
                                                                Ar[i] = r[i];
                                                                //load the whole Jacobian if true to do brute force factorization instead of block factor
                                                                if (useDumbNewton)
                                                                {
                                                                    for(int j=0;j<nSymTen;j++)
                                                                    {
                                                                        B[i+j*(nSymTen+1)] = A[i+j*nSymTen];
                                                                    }
                                                                    B[i+nSymTen*(nSymTen+1)] = r[i];
                                                                    B[nSymTen + i*(nSymTen+1)] = df[i];
                                                                    R[i] = -a[i];
                                                                }
                                                            }
                                                            if (useDumbNewton)
                                                            {
                                                                B[nSymTen + nSymTen*(nSymTen+1)] = 0.0;
                                                                R[nSymTen] = -f;
                                                            }
                                                            TRANS='N';
                                                            if(useFullPivoting)
                                                            {
                                                                dgetc2_(&N,A,&N,IPIV,JPIV,&INFO);
                                                                dgesc2_(&N,A,&N,Aa,IPIV,JPIV,&SCALE);
                                                                dgesc2_(&N,A,&N,Ar,IPIV,JPIV,&SCALE);
                                                            }
                                                            else
                                                            {
                                                                dgetrf_(&N,&N,A,&N,IPIV,&INFO);
                                                                dgetrs_(&TRANS,&N,&NRHS,A,&N,IPIV,Aa,&N,&INFO);
                                                                dgetrs_(&TRANS,&N,&NRHS,A,&N,IPIV,Ar,&N,&INFO);
                                                            }
                                                            //
                                                            //calculate increment in Delta_lamba
                                                            //
                                                            dfAa = 0.0;
                                                            dfAr = 0.0;
                                                            for(int i=0;i<nSymTen;i++)
                                                            {
                                                                dfAa  += df[i]*Aa[i];
                                                                dfAr  += df[i]*Ar[i];
                                                            }
                                                            if (fabs(dfAr) < 1.0e-8)
                                                                dfAr = copysign(1.0e-8,dfAr);
                                                            Delta_lambda_incr = (f-dfAa)/dfAr;
                                                            //calculate increment in Delta_stress
                                                            for(int i=0;i<nSymTen;i++)
                                                            {
                                                                Delta_stress_incr[i] = - Aa[i] -  Delta_lambda_incr*Ar[i];
                                                            }
                                                            //overwrite increments if using brute force Newton
                                                            if(useDumbNewton)
                                                            {
                                                                dgetc2_(&NR,B,&NR,IPIVR,JPIVR,&INFOR);
                                                                dgesc2_(&NR,B,&NR,R,IPIVR,JPIVR,&SCALER);
                                                                for(int i=0;i<nSymTen;i++)
                                                                {
                                                                    Delta_stress_incr[i] = R[i];
                                                                }
                                                                Delta_lambda_incr = R[nSymTen];
                                                            }
                                                            //increment unknowns
                                                            Delta_lambda_old = Delta_lambda;
                                                            Delta_lambda += Delta_lambda_incr;
                                                            for(int i=0;i<nSymTen;i++)
                                                            {
                                                                Delta_stress_old[i] = Delta_stress[i];
                                                                Delta_stress[i] += Delta_stress_incr[i];
                                                                stress_old[i] = stress[i];
                                                                stress[i] += Delta_stress_incr[i];
                                                            }
                                                            evaluateConstitutiveModel(materialProperties,stress,f,df,r,dr,stress_3);
                                                            if (useDifferenceJacobian)
                                                            {
                                                                differenceJacobian(materialProperties,stress,f,df,r,dr);
                                                            }
                                                            //reset r and r0 to do sem-implicit BackwardEuler integration if true
                                                            if (useSemiImplicit)
                                                            {
                                                                for(int i=0;i<nSymTen;i++)
                                                                {
                                                                    r[i]=r0[i];
                                                                    for(int j=0;j<nSymTen;j++)
                                                                    {
                                                                        dr[i + j*nSymTen]=0.0;
                                                                    }
                                                                }
                                                            }
                                                            //try linesearch if true and f has increased
                                                            int linesearches=0,max_linesearches=100;
                                                            while(useLineSearch && (f > 0.99*f_last) && (linesearches < max_linesearches))
                                                            {
                                                                std::cout<<"+";
                                                                double ds = pow(0.5,linesearches+1);
                                                                Delta_lambda = Delta_lambda_old + ds*Delta_lambda_incr;
                                                                for(int i=0;i<nSymTen;i++)
                                                                {
                                                                    Delta_stress[i] =  Delta_stress_old[i] + ds*Delta_stress_incr[i];
                                                                    stress[i] = stress_old[i] + ds*Delta_stress_incr[i];
                                                                }
                                                                evaluateConstitutiveModel(materialProperties,stress,f,df,r,dr,stress_3);
                                                                if (useDifferenceJacobian)
                                                                {
                                                                    differenceJacobian(materialProperties,stress,f,df,r,dr);
                                                                }
                                                                //reset r and r0 to do sem-implicit BackwardEuler integration if true
                                                                if (useSemiImplicit || predictorPhase)
                                                                {
                                                                    for(int i=0;i<nSymTen;i++)
                                                                    {
                                                                        r[i]=r0[i];
                                                                        for(int j=0;j<nSymTen;j++)
                                                                        {
                                                                            dr[i + j*nSymTen]=0.0;
                                                                        }
                                                                    }
                                                                }
                                                                linesearches +=1;
                                                            }
                                                            if (linesearches > 0)
                                                            {
                                                                if (f > 0.99*f_last)
                                                                {
                                                                    std::cout<<"f"<<std::endl;
                                                                    its = maxIts-1;//force terminate
                                                                }
                                                                else
                                                                    std::cout<<"s"<<std::endl;

                                                            }
                                                            //calculate residuals and norm(f is already calculated)
                                                            f_last = f;
                                                            aNorm=0.0;
                                                            stressNorm=0.0;
                                                            elasticStrain(Cinv,Delta_stress,minusDelta_plasticStrain);
                                                            for (int i=0;i<nSymTen;i++)
                                                            {
                                                                a[i] = minusDelta_plasticStrain[i] + Delta_lambda*r[i];
                                                                aNorm += a[i]*a[i];
                                                                stressNorm+=stress[i]*stress[i];
                                                            }
                                                            aNorm = sqrt(aNorm);
                                                            stressNorm = sqrt(stressNorm);
                                                            its++;
                                                            /* if (its > predictorPhaseIts) */
                                                            /*   predictorPhase = false; */
                                                            /* // */
                                                            /* //if we haven't converged then try switchign to semi-implicit integration */
                                                            /* // */
                                                            /* bool resetSolution = false; */
                                                            /* if (its == maxIts && useSemiImplicit==true && useDifferenceJacobian==true  */
                                                            /*     && useFullPivoting==true && useDumbNewton == true && useLineSearch == false) */
                                                            /*   { */
                                                            /*     its = 0; */
                                                            /*     useLineSearch=true; */
                                                            /*     resetSolution = true; */
                                                            /*     std::cout<<"linesearch failed"<<std::endl; */
                                                            /*   } */
                                                            /* if (its == maxIts && useSemiImplicit==true && useDifferenceJacobian==true && useFullPivoting==true && useDumbNewton == false) */
                                                            /*   { */
                                                            /*     its = 0; */
                                                            /*     useDumbNewton=true; */
                                                            /*     resetSolution = true; */
                                                            /*     std::cout<<"dumb Newton failed"<<std::endl; */
                                                            /*   } */
                                                            /* if (its == maxIts && useSemiImplicit==true && useDifferenceJacobian==true && useFullPivoting==false) */
                                                            /*   { */
                                                            /*     its = 0; */
                                                            /*     useFullPivoting=true; */
                                                            /*     resetSolution = true; */
                                                            /*     std::cout<<"full pivoting failed"<<std::endl; */
                                                            /*   } */
                                                            /* if (its == maxIts && useSemiImplicit==true && useDifferenceJacobian==false) */
                                                            /*   { */
                                                            /*     its = 0; */
                                                            /*     useDifferenceJacobian=true; */
                                                            /*     resetSolution = true; */
                                                            /*     std::cout<<"semi implicit BE failed"<<std::endl; */
                                                            /*   } */
                                                            /* if (its == maxIts && useSemiImplicit==false) */
                                                            /*   { */
                                                            /*     std::cout<<"+"<<"        f               a       "<<std::endl; */
                                                            /*     for(int i=0;i<fhist.size();i++) */
                                                            /*     std::cout<<std::setprecision(16)<<fhist[i]<<" "<<ahist[i]<<std::endl; */
                                                            /*     its = 0; */
                                                            /*     useSemiImplicit = true; */
                                                            /*     //useContinuumTangentModulus=false; */
                                                            /*     resetSolution = true; */
                                                            /*     //std::cout<<"fully implicit BE failed, trying semi-implicit"<<std::endl; */
                                                            /*   } */
                                                            /* if (resetSolution) */
                                                            /*   { */
                                                            /*     //reset to elastic predictor */
                                                            /*     for(int i=0;i<nSymTen;i++) */
                                                            /*     { */
                                                            /*       r[i]=r0[i]; */
                                                            /*       dr[i]=0.0; */
                                                            /*       Delta_lambda = 0.0; */
                                                            /*       Delta_stress[i] = 0.0; */
                                                            /*       minusDelta_plasticStrain[i] = 0.0; */
                                                            /*       plasticStrain[i] = plasticStrain_last[i]; */
                                                            /*       effectiveStrain[i] = strain_last[i] + Delta_strain[i] - plasticStrain_last[i]; */
                                                            /*       a[i] = 0.0; */
                                                            /*     } */
                                                            /*     elasticStress(C,effectiveStrain,stress); */
                                                            /*     evaluateConstitutiveModel(materialProperties,stress,f,df,r,dr,stress_3); */
                                                            /*     if (useDifferenceJacobian) */
                                                            /*     { */
                                                            /*       differenceJacobian(materialProperties,stress,f,df,r,dr); */
                                                            /*     } */
                                                            /*     if (useSemiImplicit) */
                                                            /*     { */
                                                            /*       for(int i=0;i<nSymTen;i++) */
                                                            /*         { */
                                                            /*           r[i]=r0[i]; */
                                                            /*           for(int j=0;j<nSymTen;j++) */
                                                            /*         dr[i + j*nSymTen]=0.0; */
                                                            /*         } */
                                                            /*     } */
                                                            /*   } */
                                                        }
                                                        if (its > 0)//non-zero plastic strain increment
                                                        {
                                                            //std::cout<<"plastic it "<<its<<" aNorm "<<aNorm<<" f "<<f<<std::endl;
                                                            //
                                                            //stress and Delta_lambda have been set, need to set plasticStrain
                                                            //
                                                            for(int i=0;i<nSymTen;i++)
                                                            {
                                                                plasticStrain[i] = plasticStrain_last[i] - minusDelta_plasticStrain[i];
                                                            }
                                                            //now calculate the algorithm tangent modulus (dstress/dstrain)
                                                            //
                                                            //if true or the Newton iteration didn't converge try using the continuum modulus
                                                            if (its == maxIts)
                                                            {
                                                                //failed, reset stress to elastic predictor
                                                                /* std::cout<<"constitutive relation interation failed"<<std::endl; */
                                                                /* std::cout<<"        f               a       "<<std::endl; */
                                                                /* for(int i=0;i<fhist.size();i++) */
                                                                /*     std::cout<<std::setprecision(16)<<fhist[i]<<" "<<ahist[i]<<std::endl; */
                                                                /* std::cout<<"-------------------------------------"<<std::endl; */
                                                                std::cout<<"Constiutive relation iteration failed "<<f<<'\t'<<stressNorm<<std::endl;
                                                                //useOuterPicard=true;
                                                                useContinuumTangentModulus=true;
                                                            }

                                                            if(useOuterPicard)
                                                            {
                                                                for (int i=0; i<nSymTen; i++)
                                                                    for (int j=0; j<nSymTen; j++)
                                                                        dstress[i*nSymTen+j] = C[i+j*nSymTen];
                                                            }
                                                            else if(useContinuumTangentModulus)// || its == maxIts)
                                                            {
                                                                double Cr[nSymTen],dfC[nSymTen],dfCr=0.0;
                                                                for (int i=0; i<nSymTen; i++)
                                                                {
                                                                    Cr[i] = 0.0;
                                                                    dfC[i] = 0.0;
                                                                }
                                                                for (int i=0; i<nSymTen; i++)
                                                                    for (int j=0; j<nSymTen; j++)
                                                                    {
                                                                        Cr[i]  += C[i+j*nSymTen]*r[j];
                                                                        dfC[j] += df[i]*C[i+j*nSymTen];
                                                                        dfCr   += df[i]*C[i+j*nSymTen]*r[j];
                                                                    }
                                                                if (fabs(dfCr) < 1.0e-8)
                                                                    dfCr = copysign(1.0e-8,dfCr);
                                                                for (int i=0; i<nSymTen; i++)
                                                                    for (int j=0; j<nSymTen; j++)
                                                                    {
                                                                        dstress[i*nSymTen+j] = C[i+j*nSymTen] - (Cr[i]*dfC[j])/dfCr;//cek hack, storing dstress in C ordering
                                                                    }
                                                            }
                                                            else //should be correct for both Backward Euler and Semi-implicit scheme
                                                            {
                                                                double Ainv[nSymTen*nSymTen];
                                                                for(int i=0;i<nSymTen;i++)
                                                                {
                                                                    for(int j=0;j<nSymTen;j++)
                                                                    {
                                                                        A[i+j*nSymTen] = Cinv[i+j*nSymTen] + Delta_lambda*dr[i+j*nSymTen];//set to A^{-1}
                                                                    Ainv[i+j*nSymTen] = A[i+j*nSymTen];
                                                                    }
                                                                    Aa[i] = a[i];
                                                                    Ar[i] = r[i];
                                                                    dfA[i] = df[i];
                                                                }
                                                                dgetrf_(&N,&N,A,&N,IPIV,&INFO);//factor
                                                                TRANS='T';
                                                                dgetrs_(&TRANS,&N,&NRHS,A,&N,IPIV,dfA,&N,&INFO);
                                                                TRANS='N';
                                                                dgetrs_(&TRANS,&N,&NRHS,A,&N,IPIV,Aa,&N,&INFO);//back substitute to get products with A
                                                                dgetrs_(&TRANS,&N,&NRHS,A,&N,IPIV,Ar,&N,&INFO);
                                                                dfAr = 0.0;
                                                                for(int i=0;i<nSymTen;i++)
                                                                {
                                                                    dfAr  += df[i]*Ar[i];
                                                                }
                                                                dgetri_(&N,A,&N,IPIV,WORK,&LWORK,&INFO);
                                                                if (fabs(dfAr) < 1.0e-8)
                                                                    dfAr = copysign(1.0e-8,dfAr);
                                                                for (int i=0; i<nSymTen; i++)
                                                                    for (int j=0; j<nSymTen; j++)
                                                                    {
                                                                        dstress[i*nSymTen+j] = A[i+j*nSymTen] - (Ar[i]*dfA[j])/dfAr;//cek hack, storing dstress in C ordering
                                                                    }
                                                            }
                                                        }
                                                        else//0 plastic strain i.e. elastic
                                                        {
                                                            for (int i=0; i<nSymTen; i++)
                                                                for (int j=0; j<nSymTen; j++)
                                                                    dstress[i*nSymTen+j] = C[i+j*nSymTen];
                                                        }
                                                        //apply pore pressure
                                                        for (int i=0; i<nSpace; i++)
                                                            stress[i] -= pore_pressure;
                }

                inline void exteriorNumericalStressFlux(const double& pore_pressure_ext,
                        const int& isDOFBoundary_u,
                        const int& isDOFBoundary_v,
                        const int& isDOFBoundary_w,
                        const int& isStressFluxBoundary_u,
                        const int& isStressFluxBoundary_v,
                        const int& isStressFluxBoundary_w,
                        const double& penalty,
                        const double& u,
                        const double& v,
                        const double& w,
                        const double& bc_u,
                        const double& bc_v,
                        const double& bc_w,
                        const double& bc_stressFlux_u,
                        const double& bc_stressFlux_v,
                        const double& bc_stressFlux_w,
                        const double* stress,
                        const double* normal,
                        double& stressFlux_u,
                        double& stressFlux_v,
                        double& stressFlux_w)
                        {
                            //note: pore pressure is already in stress but assumed not in stress flux BC
                            if (isDOFBoundary_u == 1)
                            {
                                stressFlux_u = -(stress[sXX]*normal[X] + stress[sXY]*normal[Y] + stress[sXZ]*normal[Z] - penalty*(u - bc_u));
                            }
                            else if(isStressFluxBoundary_u == 1)
                            {
                                stressFlux_u = -bc_stressFlux_u + normal[X]*pore_pressure_ext;
                            }
                            else
                            {
                                stressFlux_u = 0.0;
                            }

                            if (isDOFBoundary_v == 1)
                            {
                                stressFlux_v = -(stress[sYX]*normal[X] + stress[sYY]*normal[Y] + stress[sYZ]*normal[Z] - penalty*(v - bc_v));
                            }
                            else if(isStressFluxBoundary_v == 1)
                            {
                                stressFlux_v = -bc_stressFlux_v + normal[Y]*pore_pressure_ext;
                            }
                            else
                            {
                                stressFlux_v = 0.0;
                            }

                            if (isDOFBoundary_w  == 1)
                            {
                                stressFlux_w = -(stress[sZX]*normal[X] + stress[sZY]*normal[Y] + stress[sZZ]*normal[Z] - penalty*(w - bc_w));
                            }
                            else if(isStressFluxBoundary_w == 1)
                            {
                                stressFlux_w = -bc_stressFlux_w + normal[Z]*pore_pressure_ext;
                            }
                            else
                            {
                                stressFlux_w = 0.0;
                            }
                        }

                inline void exteriorNumericalStressFluxJacobian(const int& isDOFBoundary_u,
                        const int& isDOFBoundary_v,
                        const int& isDOFBoundary_w,
                        const double* normal,
                        const double* dstress,
                        const double& penalty,
                        const double& disp_trial,
                        const double* disp_grad_trial,
                        double& dstressFlux_u_u,
                        double& dstressFlux_u_v,
                        double& dstressFlux_u_w,
                        double& dstressFlux_v_u,
                        double& dstressFlux_v_v,
                        double& dstressFlux_v_w,
                        double& dstressFlux_w_u,
                        double& dstressFlux_w_v,
                        double& dstressFlux_w_w)
                {
                    //here we use both the symmetry of the stress tensor and the fact that dstress is w.r.t. the strain in Voigt notation to go directly to derivatives w.r.t. displacement DOF
                    if (isDOFBoundary_u == 1)
                    {
                        //stressFlux_u = -(stress[sXX]*normal[X] + stress[sXY]*normal[Y] + stress[sXZ]*normal[Z] - h_penalty*(u - bc_u));
                        dstressFlux_u_u = -(
                                (dstress[sXX*nSymTen+sXX]*disp_grad_trial[X] + dstress[sXX*nSymTen+sXY]*disp_grad_trial[Y] + dstress[sXX*nSymTen+sXZ]*disp_grad_trial[Z])*normal[X]+
                                (dstress[sXY*nSymTen+sXX]*disp_grad_trial[X] + dstress[sXY*nSymTen+sXY]*disp_grad_trial[Y] + dstress[sXY*nSymTen+sXZ]*disp_grad_trial[Z])*normal[Y]+
                                (dstress[sXZ*nSymTen+sXX]*disp_grad_trial[X] + dstress[sXZ*nSymTen+sXY]*disp_grad_trial[Y] + dstress[sXZ*nSymTen+sXZ]*disp_grad_trial[Z])*normal[Z]
                                -
                                penalty*disp_trial);
                        dstressFlux_u_v = -(
                                (dstress[sXX*nSymTen+sYX]*disp_grad_trial[X] + dstress[sXX*nSymTen+sYY]*disp_grad_trial[Y] + dstress[sXX*nSymTen+sYZ]*disp_grad_trial[Z])*normal[X]+
                                (dstress[sXY*nSymTen+sYX]*disp_grad_trial[X] + dstress[sXY*nSymTen+sYY]*disp_grad_trial[Y] + dstress[sXY*nSymTen+sYZ]*disp_grad_trial[Z])*normal[Y]+
                                (dstress[sXZ*nSymTen+sYX]*disp_grad_trial[X] + dstress[sXZ*nSymTen+sYY]*disp_grad_trial[Y] + dstress[sXZ*nSymTen+sYZ]*disp_grad_trial[Z])*normal[Z]);
                        dstressFlux_u_w = -(
                                (dstress[sXX*nSymTen+sZX]*disp_grad_trial[X] + dstress[sXX*nSymTen+sZY]*disp_grad_trial[Y] + dstress[sXX*nSymTen+sZZ]*disp_grad_trial[Z])*normal[X]+
                                (dstress[sXY*nSymTen+sZX]*disp_grad_trial[X] + dstress[sXY*nSymTen+sZY]*disp_grad_trial[Y] + dstress[sXY*nSymTen+sZZ]*disp_grad_trial[Z])*normal[Y]+
                                (dstress[sXZ*nSymTen+sZX]*disp_grad_trial[X] + dstress[sXZ*nSymTen+sZY]*disp_grad_trial[Y] + dstress[sXZ*nSymTen+sZZ]*disp_grad_trial[Z])*normal[Z]);
                    }
                    else
                    {
                        dstressFlux_u_u = 0.0;
                        dstressFlux_u_v = 0.0;
                        dstressFlux_u_w = 0.0;
                    }

                    if (isDOFBoundary_v == 1)
                    {
                        //stressFlux_v = -(stress[sYX]*normal[X] + stress[sYY]*normal[Y] + stress[sYZ]*normal[Z] - h_penalty*(v - bc_v));
                        dstressFlux_v_u = -(
                                (dstress[sYX*nSymTen+sXX]*disp_grad_trial[X] + dstress[sYX*nSymTen+sXY]*disp_grad_trial[Y] + dstress[sYX*nSymTen+sXZ]*disp_grad_trial[Z])*normal[X]+
                                (dstress[sYY*nSymTen+sXX]*disp_grad_trial[X] + dstress[sYY*nSymTen+sXY]*disp_grad_trial[Y] + dstress[sYY*nSymTen+sXZ]*disp_grad_trial[Z])*normal[Y]+
                                (dstress[sYZ*nSymTen+sXX]*disp_grad_trial[X] + dstress[sYZ*nSymTen+sXY]*disp_grad_trial[Y] + dstress[sYZ*nSymTen+sXZ]*disp_grad_trial[Z])*normal[Z]);
                        dstressFlux_v_v = -(
                                (dstress[sYX*nSymTen+sYX]*disp_grad_trial[X] + dstress[sYX*nSymTen+sYY]*disp_grad_trial[Y] + dstress[sYX*nSymTen+sYZ]*disp_grad_trial[Z])*normal[X]+
                                (dstress[sYY*nSymTen+sYX]*disp_grad_trial[X] + dstress[sYY*nSymTen+sYY]*disp_grad_trial[Y] + dstress[sYY*nSymTen+sYZ]*disp_grad_trial[Z])*normal[Y]+
                                (dstress[sYZ*nSymTen+sYX]*disp_grad_trial[X] + dstress[sYZ*nSymTen+sYY]*disp_grad_trial[Y] + dstress[sYZ*nSymTen+sYZ]*disp_grad_trial[Z])*normal[Z]
                                -
                                penalty*disp_trial);
                        dstressFlux_v_w = -(
                                (dstress[sYX*nSymTen+sZX]*disp_grad_trial[X] + dstress[sYX*nSymTen+sZY]*disp_grad_trial[Y] + dstress[sYX*nSymTen+sZZ]*disp_grad_trial[Z])*normal[X]+
                                (dstress[sYY*nSymTen+sZX]*disp_grad_trial[X] + dstress[sYY*nSymTen+sZY]*disp_grad_trial[Y] + dstress[sYY*nSymTen+sZZ]*disp_grad_trial[Z])*normal[Y]+
                                (dstress[sYZ*nSymTen+sZX]*disp_grad_trial[X] + dstress[sYZ*nSymTen+sZY]*disp_grad_trial[Y] + dstress[sYZ*nSymTen+sZZ]*disp_grad_trial[Z])*normal[Z]);
                    }
                    else
                    {
                        dstressFlux_v_u = 0.0;
                        dstressFlux_v_v = 0.0;
                        dstressFlux_v_w = 0.0;
                    }

                    if (isDOFBoundary_w  == 1)
                    {
                        //stressFlux_w = -(stress[sZX]*normal[X] + stress[sZY]*normal[Y] + stress[sZZ]*normal[Z] - h_penalty*(w - bc_w));
                        dstressFlux_w_u = -(
                                (dstress[sZX*nSymTen+sXX]*disp_grad_trial[X] + dstress[sZX*nSymTen+sXY]*disp_grad_trial[Y] + dstress[sZX*nSymTen+sXZ]*disp_grad_trial[Z])*normal[X]+
                                (dstress[sZY*nSymTen+sXX]*disp_grad_trial[X] + dstress[sZY*nSymTen+sXY]*disp_grad_trial[Y] + dstress[sZY*nSymTen+sXZ]*disp_grad_trial[Z])*normal[Y]+
                                (dstress[sZZ*nSymTen+sXX]*disp_grad_trial[X] + dstress[sZZ*nSymTen+sXY]*disp_grad_trial[Y] + dstress[sZZ*nSymTen+sXZ]*disp_grad_trial[Z])*normal[Z]);
                        dstressFlux_w_v = -(
                                (dstress[sZX*nSymTen+sYX]*disp_grad_trial[X] + dstress[sZX*nSymTen+sYY]*disp_grad_trial[Y] + dstress[sZX*nSymTen+sYZ]*disp_grad_trial[Z])*normal[X]+
                                (dstress[sZY*nSymTen+sYX]*disp_grad_trial[X] + dstress[sZY*nSymTen+sYY]*disp_grad_trial[Y] + dstress[sZY*nSymTen+sYZ]*disp_grad_trial[Z])*normal[Y]+
                                (dstress[sZZ*nSymTen+sYX]*disp_grad_trial[X] + dstress[sZZ*nSymTen+sYY]*disp_grad_trial[Y] + dstress[sZZ*nSymTen+sYZ]*disp_grad_trial[Z])*normal[Z]);
                        dstressFlux_w_w = -(
                                (dstress[sZX*nSymTen+sZX]*disp_grad_trial[X] + dstress[sZX*nSymTen+sZY]*disp_grad_trial[Y] + dstress[sZX*nSymTen+sZZ]*disp_grad_trial[Z])*normal[X]+
                                (dstress[sZY*nSymTen+sZX]*disp_grad_trial[X] + dstress[sZY*nSymTen+sZY]*disp_grad_trial[Y] + dstress[sZY*nSymTen+sZZ]*disp_grad_trial[Z])*normal[Y]+
                                (dstress[sZZ*nSymTen+sZX]*disp_grad_trial[X] + dstress[sZZ*nSymTen+sZY]*disp_grad_trial[Y] + dstress[sZZ*nSymTen+sZZ]*disp_grad_trial[Z])*normal[Z]
                                -
                                penalty*disp_trial);
                    }
                    else
                    {
                        dstressFlux_w_u = 0.0;
                        dstressFlux_w_v = 0.0;
                        dstressFlux_w_w = 0.0;
                    }
                }

                virtual void calculateResidual(arguments_dict& args)
                        {
        proteus::pyarray<double>& mesh_trial_ref = args.array<double>("mesh_trial_ref");
        proteus::pyarray<double>& mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
        proteus::pyarray<double>& mesh_dof = args.array<double>("mesh_dof");
        proteus::pyarray<proteus_index_t>& mesh_l2g = args.array<proteus_index_t>("mesh_l2g");
        proteus::pyarray<double>& dV_ref = args.array<double>("dV_ref");
        proteus::pyarray<double>& disp_trial_ref = args.array<double>("disp_trial_ref");
        proteus::pyarray<double>& disp_grad_trial_ref = args.array<double>("disp_grad_trial_ref");
        proteus::pyarray<double>& disp_test_ref = args.array<double>("disp_test_ref");
        proteus::pyarray<double>& disp_grad_test_ref = args.array<double>("disp_grad_test_ref");
        proteus::pyarray<double>& mesh_trial_trace_ref = args.array<double>("mesh_trial_trace_ref");
        proteus::pyarray<double>& mesh_grad_trial_trace_ref = args.array<double>("mesh_grad_trial_trace_ref");
        proteus::pyarray<double>& dS_ref = args.array<double>("dS_ref");
        proteus::pyarray<double>& disp_trial_trace_ref = args.array<double>("disp_trial_trace_ref");
        proteus::pyarray<double>& disp_grad_trial_trace_ref = args.array<double>("disp_grad_trial_trace_ref");
        proteus::pyarray<double>& disp_test_trace_ref = args.array<double>("disp_test_trace_ref");
        proteus::pyarray<double>& disp_grad_test_trace_ref = args.array<double>("disp_grad_test_trace_ref");
        proteus::pyarray<double>& normal_ref = args.array<double>("normal_ref");
        proteus::pyarray<double>& boundaryJac_ref = args.array<double>("boundaryJac_ref");
        proteus::pyarray<double>& ebqe_penalty = args.array<double>("ebqe_penalty");
        int gravityStep = args.scalar<int>("gravityStep");
        proteus_index_t nElements_global = args.scalar<proteus_index_t>("nElements_global");
        proteus::pyarray<int>& materialTypes = args.array<int>("materialTypes");
        int nMaterialProperties = args.scalar<int>("nMaterialProperties");
        proteus::pyarray<double>& materialProperties = args.array<double>("materialProperties");
        double pore_fluid_unit_weight = args.scalar<double>("pore_fluid_unit_weight");
        proteus::pyarray<double>& pore_pressure_head_dof = args.array<double>("pore_pressure_head_dof");
        proteus::pyarray<double>& q_strain = args.array<double>("q_strain");
        proteus::pyarray<double>& q_strain0 = args.array<double>("q_strain0");
        proteus::pyarray<double>& q_strain_last = args.array<double>("q_strain_last");
        proteus::pyarray<double>& q_plasticStrain = args.array<double>("q_plasticStrain");
        proteus::pyarray<double>& q_plasticStrain_last = args.array<double>("q_plasticStrain_last");
        proteus::pyarray<double>& ebqe_strain = args.array<double>("ebqe_strain");
        proteus::pyarray<double>& ebqe_strain0 = args.array<double>("ebqe_strain0");
        proteus::pyarray<double>& ebqe_strain_last = args.array<double>("ebqe_strain_last");
        proteus::pyarray<double>& ebqe_plasticStrain = args.array<double>("ebqe_plasticStrain");
        proteus::pyarray<double>& ebqe_plasticStrain_last = args.array<double>("ebqe_plasticStrain_last");
        proteus::pyarray<proteus_index_t>& disp_l2g = args.array<proteus_index_t>("disp_l2g");
        proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
        proteus::pyarray<double>& v_dof = args.array<double>("v_dof");
        proteus::pyarray<double>& w_dof = args.array<double>("w_dof");
        proteus::pyarray<double>& bodyForce = args.array<double>("bodyForce");
        proteus_index_t offset_u = args.scalar<proteus_index_t>("offset_u");
        proteus_index_t offset_v = args.scalar<proteus_index_t>("offset_v");
        proteus_index_t offset_w = args.scalar<proteus_index_t>("offset_w");
        proteus_index_t stride_u = args.scalar<proteus_index_t>("stride_u");
        proteus_index_t stride_v = args.scalar<proteus_index_t>("stride_v");
        proteus_index_t stride_w = args.scalar<proteus_index_t>("stride_w");
        proteus::pyarray<double>& globalResidual = args.array<double>("globalResidual");
        proteus_index_t nExteriorElementBoundaries_global = args.scalar<proteus_index_t>("nExteriorElementBoundaries_global");
        proteus::pyarray<proteus_index_t>& exteriorElementBoundariesArray = args.array<proteus_index_t>("exteriorElementBoundariesArray");
        proteus::pyarray<proteus_index_t>& elementBoundaryElementsArray = args.array<proteus_index_t>("elementBoundaryElementsArray");
        proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
        proteus::pyarray<int>& isDOFBoundary_u = args.array<int>("isDOFBoundary_u");
        proteus::pyarray<int>& isDOFBoundary_v = args.array<int>("isDOFBoundary_v");
        proteus::pyarray<int>& isDOFBoundary_w = args.array<int>("isDOFBoundary_w");
        proteus::pyarray<int>& isStressFluxBoundary_u = args.array<int>("isStressFluxBoundary_u");
        proteus::pyarray<int>& isStressFluxBoundary_v = args.array<int>("isStressFluxBoundary_v");
        proteus::pyarray<int>& isStressFluxBoundary_w = args.array<int>("isStressFluxBoundary_w");
        proteus::pyarray<double>& ebqe_bc_u_ext = args.array<double>("ebqe_bc_u_ext");
        proteus::pyarray<double>& ebqe_bc_v_ext = args.array<double>("ebqe_bc_v_ext");
        proteus::pyarray<double>& ebqe_bc_w_ext = args.array<double>("ebqe_bc_w_ext");
        proteus::pyarray<double>& ebqe_bc_stressFlux_u_ext = args.array<double>("ebqe_bc_stressFlux_u_ext");
        proteus::pyarray<double>& ebqe_bc_stressFlux_v_ext = args.array<double>("ebqe_bc_stressFlux_v_ext");
        proteus::pyarray<double>& ebqe_bc_stressFlux_w_ext = args.array<double>("ebqe_bc_stressFlux_w_ext");
                            //
                            //loop over elements to compute volume integrals and load them into element and global residual
                            //
                            //std::cout<<"nElements_global"<<nElements_global<<std::endl;
                            //std::cout<<"nQuadraturePoints_element"<<nQuadraturePoints_element<<std::endl;
                            const int usePicard = 0;
                            for(proteus_index_t eN=0;eN<nElements_global;eN++)
                            {
                                //declare local storage for element residual and initialize
                                double
                                    elementResidual_u[nDOF_test_element],
                                    elementResidual_v[nDOF_test_element],
                                    elementResidual_w[nDOF_test_element];
                                    for (int i=0;i<nDOF_test_element;i++)
                                    {
                                        elementResidual_u[i]=0.0;
                                        elementResidual_v[i]=0.0;
                                        elementResidual_w[i]=0.0;
                                    }//i
                                    //
                                    //loop over quadrature points and compute integrands
                                    //
                                    for(int k=0;k<nQuadraturePoints_element;k++)
                                    {
                                        //compute indices and declare local storage
                                        proteus_index_t eN_k = eN*nQuadraturePoints_element+k,
                                                 eN_k_nSpace=eN_k*nSpace,
                                                 eN_nDOF_trial_element = eN*nDOF_trial_element,
                                                 eN_nDOF_mesh_trial_element = eN*nDOF_mesh_trial_element; //index to a vector at a quadrature point
                                        double u=0.0,v=0.0,w=0.0,
                                                 D[nSpace*nSpace],
                                                 *grad_u(&D[0]),
                                                 *grad_v(&D[nSpace]),
                                                 *grad_w(&D[2*nSpace]),
                                                 jac[nSpace*nSpace],
                                                 jacDet,
                                                 jacInv[nSpace*nSpace],
                                                 disp_grad_trial[nDOF_trial_element*nSpace],
                                                 disp_test_dV[nDOF_trial_element],
                                                 disp_grad_test_dV[nDOF_test_element*nSpace],
                                                 dV,x,y,z,
                                                 G[nSpace*nSpace],G_dd_G,tr_G,
                                                 stress[ck.nSymTen],strain[ck.nSymTen],
                                                 Delta_strain[ck.nSymTen],dstress[ck.nSymTen*ck.nSymTen],pore_pressure=0.0;
                                                 //get jacobian, etc for mapping reference element
                                                 ck.calculateMapping_element(eN,
                                                         k,
                                                         mesh_dof.data(),
                                                         mesh_l2g.data(),
                                                         mesh_trial_ref.data(),
                                                         mesh_grad_trial_ref.data(),
                                                         jac,
                                                         jacDet,
                                                         jacInv,
                                                         x,y,z);
                                                 //get the pressure head
                                                 ck.mapping.valFromDOF(pore_pressure_head_dof.data(),&mesh_l2g.data()[eN_nDOF_mesh_trial_element],&mesh_trial_ref.data()[k*nDOF_mesh_trial_element],pore_pressure);
                                                 //std::cout<<"pore_pressure \t"<<(5.0-pore_pressure)<<"\t z \t"<<z<<std::endl;
                                                 pore_pressure *= pore_fluid_unit_weight;
                                                 pore_pressure = fmax(pore_pressure,0.0);//cek hack
                                                 //std::cout<<"pore_pressure "<<pore_pressure<<std::endl;
                                                 //get the physical integration weight
                                                 dV = fabs(jacDet)*dV_ref.data()[k];
                                                 ck.calculateG(jacInv,G,G_dd_G,tr_G);
                                                 //get the trial function gradients
                                                 ck.gradTrialFromRef(&disp_grad_trial_ref.data()[k*nDOF_trial_element*nSpace],jacInv,disp_grad_trial);
                                                 //get the solution
                                                 ck.valFromDOF(u_dof.data(),&disp_l2g.data()[eN_nDOF_trial_element],&disp_trial_ref.data()[k*nDOF_trial_element],u);
                                                 ck.valFromDOF(v_dof.data(),&disp_l2g.data()[eN_nDOF_trial_element],&disp_trial_ref.data()[k*nDOF_trial_element],v);
                                                 ck.valFromDOF(w_dof.data(),&disp_l2g.data()[eN_nDOF_trial_element],&disp_trial_ref.data()[k*nDOF_trial_element],w);
                                                 //get the solution gradients
                                                 ck.gradFromDOF(u_dof.data(),&disp_l2g.data()[eN_nDOF_trial_element],disp_grad_trial,grad_u);
                                                 ck.gradFromDOF(v_dof.data(),&disp_l2g.data()[eN_nDOF_trial_element],disp_grad_trial,grad_v);
                                                 ck.gradFromDOF(w_dof.data(),&disp_l2g.data()[eN_nDOF_trial_element],disp_grad_trial,grad_w);
                                                 //precalculate test function products with integration weights
                                                 for (int j=0;j<nDOF_trial_element;j++)
                                                 {
                                                     disp_test_dV[j] = disp_test_ref.data()[k*nDOF_trial_element+j]*dV;
                                                     for (int I=0;I<nSpace;I++)
                                                     {
                                                         disp_grad_test_dV[j*nSpace+I] = disp_grad_trial[j*nSpace+I]*dV;//cek warning won't work for Petrov-Galerkin
                                                     }
                                                 }
                                                 //save displacement at quadrature points for other models to use
                                                 //q_displacement[eN_k_nSpace+0]=u;
                                                 //q_displacement[eN_k_nSpace+1]=v;
                                                 //q_displacement[eN_k_nSpace+2]=w;

                                                 calculateStrain(D,Delta_strain);
                                                 //std::cout<<"here "<<materialProperties[0]<<'\t'<<materialProperties[1]<<std::endl<<std::flush;

                                                 double C[nSymTen*nSymTen],Cinv[nSymTen*nSymTen];
                                                 if (gravityStep)
                                                 {
                                                     elasticModuli(materialProperties.data()[materialTypes.data()[eN]*nMaterialProperties],//E,
                                                             materialProperties.data()[materialTypes.data()[eN]*nMaterialProperties+1],//nu,
                                                             C,Cinv);
                                                     if (gravityStep == 2)
                                                     {
                                                         double f,df[nSymTen],r[nSymTen],dr[nSymTen*nSymTen],stress_3_init,E=materialProperties.data()[materialTypes.data()[eN]*nMaterialProperties],nu=materialProperties.data()[materialTypes.data()[eN]*nMaterialProperties+1];
                                                         elasticStress(C,&q_strain0.data()[eN_k*nSymTen],stress);
                                                         evaluateConstitutiveModel(&materialProperties.data()[materialTypes.data()[eN]*nMaterialProperties],
                                                                 stress,f,df,r,dr,stress_3_init);//cek hack, this is just to get stress_3_init
                                                         const double n_e = 0.6,
                                                               P_a = materialProperties.data()[materialTypes.data()[eN]*nMaterialProperties+12],
                                                               K_i = 500.0;
                                                               if (P_a > 0.0)//cek hack, if not a soil then set P_a=0.0
                                                               {
                                                                   E =K_i*P_a*pow(fmax(stress_3_init,P_a)/P_a,n_e);
                                                                   elasticModuli(E,nu,C,Cinv);
                                                               }
                                                     }
                                                     for (int i=0; i<nSymTen; i++)
                                                     {
                                                         strain[i] = Delta_strain[i] + q_strain_last.data()[eN_k*nSymTen+i];
                                                         for (int j=0; j<nSymTen; j++)
                                                             dstress[i*nSymTen+j] = C[i+j*nSymTen];
                                                     }
                                                     elasticStress(C,strain,stress);
                                                     for (int i=0; i<nSpace; i++)
                                                         stress[i] -= pore_pressure;
                                                 }
                                                 else
                                                     evaluateCoefficients(usePicard,
                                                             pore_pressure,
                                                             &materialProperties.data()[materialTypes.data()[eN]*nMaterialProperties],
                                                             &q_strain0.data()[eN_k*nSymTen],
                                                             &q_strain_last.data()[eN_k*nSymTen],
                                                             Delta_strain,
                                                             &q_plasticStrain_last.data()[eN_k*nSymTen],
                                                             &q_plasticStrain.data()[eN_k*nSymTen],
                                                             stress,
                                                             dstress);
                                                 for (int i=0;i<nSymTen;i++)
                                                 {
                                                     q_strain.data()[eN_k*nSymTen+i] = q_strain_last.data()[eN_k*nSymTen + i] + Delta_strain[i];
                                                 }
                                                 //
                                                 //moving mesh
                                                 //
                                                 //omit for now
                                                 //
                                                 //update element residual
                                                 //
                                                 for(int i=0;i<nDOF_test_element;i++)
                                                 {
                                                     int i_nSpace=i*nSpace;
                                                     elementResidual_u[i] += ck.Stress_u_weak(stress,&disp_grad_test_dV[i_nSpace]) +
                                                         ck.Reaction_weak(-bodyForce.data()[eN_k_nSpace+0],disp_test_dV[i]);
                                                     elementResidual_v[i] += ck.Stress_v_weak(stress,&disp_grad_test_dV[i_nSpace]) +
                                                         ck.Reaction_weak(-bodyForce.data()[eN_k_nSpace+1],disp_test_dV[i]);
                                                     elementResidual_w[i] += ck.Stress_w_weak(stress,&disp_grad_test_dV[i_nSpace]) +
                                                         ck.Reaction_weak(-bodyForce.data()[eN_k_nSpace+2],disp_test_dV[i]);
                                                     // if (k == nQuadraturePoints_element-1)
                                                     //     {
                                                     //       std::cout<<"element residual "<<eN<<'\t'<<i<<std::endl;
                                                     //       std::cout<<elementResidual_u[i]<<std::endl
                                                     //            <<elementResidual_v[i]<<std::endl
                                                     //            <<elementResidual_w[i]<<std::endl;
                                                     //     }
                                                 }//i
                                    }
                                    //
                                    //load element into global residual and save element residual
                                    //
                                    for(int i=0;i<nDOF_test_element;i++)
                                    {
                                        proteus_index_t eN_i=eN*nDOF_test_element+i;

                                        globalResidual.data()[offset_u+stride_u*disp_l2g.data()[eN_i]] += elementResidual_u[i];
                                        globalResidual.data()[offset_v+stride_v*disp_l2g.data()[eN_i]] += elementResidual_v[i];
                                        globalResidual.data()[offset_w+stride_w*disp_l2g.data()[eN_i]] += elementResidual_w[i];
                                    }//i
                            }//elements
                            //
                            //loop over exterior element boundaries to calculate surface integrals and load into element and global residuals
                            //
                            //ebNE is the Exterior element boundary INdex
                            //ebN is the element boundary INdex
                            //eN is the element index
                            for (proteus_index_t ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
                            {
                                proteus_index_t ebN = exteriorElementBoundariesArray.data()[ebNE],
                                eN  = elementBoundaryElementsArray.data()[ebN*2+0],
                                eN_nDOF_trial_element = eN*nDOF_trial_element,
                                eN_nDOF_mesh_trial_element = eN*nDOF_mesh_trial_element;
        int ebN_local = elementBoundaryLocalElementBoundariesArray.data()[ebN*2+0];
                                double elementResidual_u[nDOF_test_element],
                                elementResidual_v[nDOF_test_element],
                                elementResidual_w[nDOF_test_element];
                                for (int i=0;i<nDOF_test_element;i++)
                                {
                                    elementResidual_u[i]=0.0;
                                    elementResidual_v[i]=0.0;
                                    elementResidual_w[i]=0.0;
                                }
                                for  (int kb=0;kb<nQuadraturePoints_elementBoundary;kb++)
                                {
                                    proteus_index_t ebNE_kb = ebNE*nQuadraturePoints_elementBoundary+kb;
                                    int ebN_local_kb = ebN_local*nQuadraturePoints_elementBoundary+kb,
                                             ebN_local_kb_nSpace = ebN_local_kb*nSpace;
                                    double u_ext=0.0,
                                             v_ext=0.0,
                                             w_ext=0.0,
                                             D[nSpace*nSpace],
                                             *grad_u_ext(&D[0]),
                                             *grad_v_ext(&D[nSpace]),
                                             *grad_w_ext(&D[2*nSpace]),
                                             bc_u_ext=0.0,
                                             bc_v_ext=0.0,
                                             bc_w_ext=0.0,
                                             jac_ext[nSpace*nSpace],
                                             jacDet_ext,
                                             jacInv_ext[nSpace*nSpace],
                                             boundaryJac[nSpace*(nSpace-1)],
                                             metricTensor[(nSpace-1)*(nSpace-1)],
                                             metricTensorDetSqrt,
                                             dS,disp_test_dS[nDOF_test_element],
                                             disp_grad_trial_trace[nDOF_trial_element*nSpace],
                                             normal[3],x_ext,y_ext,z_ext,
                                             G[nSpace*nSpace],G_dd_G,tr_G,h_penalty,
                                             Delta_strain[ck.nSymTen],
                                             strain[ck.nSymTen],stress[ck.nSymTen],dstress[ck.nSymTen*ck.nSymTen],
                                             stressFlux_u,stressFlux_v,stressFlux_w,pore_pressure_ext;
                                             //compute information about mapping from reference element to physical element
                                             ck.calculateMapping_elementBoundary(eN,
                                                     ebN_local,
                                                     kb,
                                                     ebN_local_kb,
                                                     mesh_dof.data(),
                                                     mesh_l2g.data(),
                                                     mesh_trial_trace_ref.data(),
                                                     mesh_grad_trial_trace_ref.data(),
                                                     boundaryJac_ref.data(),
                                                     jac_ext,
                                                     jacDet_ext,
                                                     jacInv_ext,
                                                     boundaryJac,
                                                     metricTensor,
                                                     metricTensorDetSqrt,
                                                     normal_ref.data(),
                                                     normal,
                                                     x_ext,y_ext,z_ext);
                                             //get the pressure head
                                             ck.mapping.valFromDOF(pore_pressure_head_dof.data(),&mesh_l2g.data()[eN_nDOF_mesh_trial_element],&mesh_trial_trace_ref.data()[ebN_local_kb*nDOF_mesh_trial_element],pore_pressure_ext);
                                             pore_pressure_ext *= pore_fluid_unit_weight;
                                             pore_pressure_ext = fmax(pore_pressure_ext,0.0);
                                             dS = metricTensorDetSqrt*dS_ref.data()[kb];
                                             //get the metric tensor
                                             //cek todo use symmetry
                                             ck.calculateG(jacInv_ext,G,G_dd_G,tr_G);
                                             //compute shape and solution information
                                             //shape
                                             ck.gradTrialFromRef(&disp_grad_trial_trace_ref.data()[ebN_local_kb_nSpace*nDOF_trial_element],jacInv_ext,disp_grad_trial_trace);
                                             //solution and gradients
                                             ck.valFromDOF(u_dof.data(),&disp_l2g.data()[eN_nDOF_trial_element],&disp_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],u_ext);
                                             ck.valFromDOF(v_dof.data(),&disp_l2g.data()[eN_nDOF_trial_element],&disp_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],v_ext);
                                             ck.valFromDOF(w_dof.data(),&disp_l2g.data()[eN_nDOF_trial_element],&disp_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],w_ext);
                                             ck.gradFromDOF(u_dof.data(),&disp_l2g.data()[eN_nDOF_trial_element],disp_grad_trial_trace,grad_u_ext);
                                             ck.gradFromDOF(v_dof.data(),&disp_l2g.data()[eN_nDOF_trial_element],disp_grad_trial_trace,grad_v_ext);
                                             ck.gradFromDOF(w_dof.data(),&disp_l2g.data()[eN_nDOF_trial_element],disp_grad_trial_trace,grad_w_ext);
                                             //precalculate test function products with integration weights
                                             for (int j=0;j<nDOF_trial_element;j++)
                                             {
                                                 disp_test_dS[j] = disp_test_trace_ref.data()[ebN_local_kb*nDOF_test_element+j]*dS;
                                             }
                                             //
                                             //load the boundary values
                                             //
                                             bc_u_ext = isDOFBoundary_u.data()[ebNE_kb]*ebqe_bc_u_ext.data()[ebNE_kb]+(1-isDOFBoundary_u.data()[ebNE_kb])*u_ext;
                                             bc_v_ext = isDOFBoundary_v.data()[ebNE_kb]*ebqe_bc_v_ext.data()[ebNE_kb]+(1-isDOFBoundary_v.data()[ebNE_kb])*v_ext;
                                             bc_w_ext = isDOFBoundary_w.data()[ebNE_kb]*ebqe_bc_w_ext.data()[ebNE_kb]+(1-isDOFBoundary_w.data()[ebNE_kb])*w_ext;
                                             //
                                             //calculate the pde coefficients using the solution and the boundary values for the solution
                                             //
                                             calculateStrain(D,Delta_strain);
                                             double C[nSymTen*nSymTen],Cinv[nSymTen*nSymTen];
                                             if (gravityStep)
                                             {
                                                 elasticModuli(materialProperties.data()[materialTypes.data()[eN]*nMaterialProperties],//E,
                                                         materialProperties.data()[materialTypes.data()[eN]*nMaterialProperties+1],//nu,
                                                         C,Cinv);
                                                 if (gravityStep == 2)
                                                 {
                                                     double f,df[nSymTen],r[nSymTen],dr[nSymTen*nSymTen],stress_3_init,E=materialProperties.data()[materialTypes.data()[eN]*nMaterialProperties],nu=materialProperties.data()[materialTypes.data()[eN]*nMaterialProperties+1];
                                                     elasticStress(C,&ebqe_strain0.data()[ebNE_kb*nSymTen],stress);
                                                     evaluateConstitutiveModel(&materialProperties.data()[materialTypes.data()[eN]*nMaterialProperties],
                                                             stress,f,df,r,dr,stress_3_init);//cek hack, this is just to get stress_3_init
                                                     const double n_e = 0.6,
                                                           P_a = materialProperties.data()[materialTypes.data()[eN]*nMaterialProperties+12],
                                                           K_i = 500.0;
                                                           if (P_a > 0.0)//cek hack, if not a soil then set P_a=0.0
                                                           {
                                                               E =K_i*P_a*pow(fmax(stress_3_init,P_a)/P_a,n_e);
                                                               elasticModuli(E,nu,C,Cinv);
                                                           }
                                                 }
                                                 for (int i=0; i<nSymTen; i++)
                                                 {
                                                     strain[i] = Delta_strain[i] + ebqe_strain_last.data()[ebNE_kb*nSymTen+i];
                                                     for (int j=0; j<nSymTen; j++)
                                                         dstress[i*nSymTen+j] = C[i+j*nSymTen];
                                                 }
                                                 elasticStress(C,strain,stress);
                                                 for (int i=0;i<nSpace;i++)
                                                     stress[i] -= pore_pressure_ext;
                                             }
                                             else
                                                 evaluateCoefficients(usePicard,
                                                         pore_pressure_ext,
                                                         &materialProperties.data()[materialTypes.data()[eN]*nMaterialProperties],
                                                         &ebqe_strain0.data()[ebNE_kb*nSymTen],
                                                         &ebqe_strain_last.data()[ebNE_kb*nSymTen],
                                                         Delta_strain,
                                                         &ebqe_plasticStrain_last.data()[ebNE_kb*nSymTen],
                                                         &ebqe_plasticStrain.data()[ebNE_kb*nSymTen],
                                                         stress,
                                                         dstress);
                                             for (int i=0;i<nSymTen;i++)
                                             {
                                                 ebqe_strain.data()[ebNE_kb*nSymTen+i] = ebqe_strain_last.data()[ebNE_kb*nSymTen+i] + Delta_strain[i];
                                             }
                                             //
                                             //calculate the numerical fluxes
                                             //
                                             double E=materialProperties.data()[materialTypes.data()[eN]*nMaterialProperties],
                                             nu=materialProperties.data()[materialTypes.data()[eN]*nMaterialProperties+1];
                                             h_penalty=(E/(1.0+nu))*(1.0 + (nu/(1.0-2.0*nu)))*ebqe_penalty.data()[ebNE_kb];
                                             exteriorNumericalStressFlux(pore_pressure_ext,
                                                     isDOFBoundary_u.data()[ebNE_kb],
                                                     isDOFBoundary_v.data()[ebNE_kb],
                                                     isDOFBoundary_w.data()[ebNE_kb],
                                                     isStressFluxBoundary_u.data()[ebNE_kb],
                                                     isStressFluxBoundary_v.data()[ebNE_kb],
                                                     isStressFluxBoundary_w.data()[ebNE_kb],
                                                     h_penalty,
                                                     u_ext,
                                                     v_ext,
                                                     w_ext,
                                                     bc_u_ext,
                                                     bc_v_ext,
                                                     bc_w_ext,
                                                     ebqe_bc_stressFlux_u_ext.data()[ebNE_kb],
                                                     ebqe_bc_stressFlux_v_ext.data()[ebNE_kb],
                                                     ebqe_bc_stressFlux_w_ext.data()[ebNE_kb],
                                                     stress,
                                                     normal,
                                                     stressFlux_u,
                                                     stressFlux_v,
                                                     stressFlux_w);
                                                     //
                                                     //update residuals
                                                     //
                                                     for (int i=0;i<nDOF_test_element;i++)
                                                     {
                                                         elementResidual_u[i] += ck.ExteriorElementBoundaryStressFlux(stressFlux_u,disp_test_dS[i]);
                                                         elementResidual_v[i] += ck.ExteriorElementBoundaryStressFlux(stressFlux_v,disp_test_dS[i]);
                                                         elementResidual_w[i] += ck.ExteriorElementBoundaryStressFlux(stressFlux_w,disp_test_dS[i]);
                                                     }//i
                                }//kb
                                //
                                //update the element and global residual storage
                                //
                                for (int i=0;i<nDOF_test_element;i++)
                                {
                                    proteus_index_t eN_i = eN*nDOF_test_element+i;

                                    globalResidual.data()[offset_u+stride_u*disp_l2g.data()[eN_i]]+=elementResidual_u[i];
                                    globalResidual.data()[offset_v+stride_v*disp_l2g.data()[eN_i]]+=elementResidual_v[i];
                                    globalResidual.data()[offset_w+stride_w*disp_l2g.data()[eN_i]]+=elementResidual_w[i];
                                }//i
                            }//ebNE
                        }

                virtual void calculateJacobian(arguments_dict& args)
                        {
        int usePicard = args.scalar<int>("usePicard");
        proteus::pyarray<double>& mesh_trial_ref = args.array<double>("mesh_trial_ref");
        proteus::pyarray<double>& mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
        proteus::pyarray<double>& mesh_dof = args.array<double>("mesh_dof");
        proteus::pyarray<proteus_index_t>& mesh_l2g = args.array<proteus_index_t>("mesh_l2g");
        proteus::pyarray<double>& dV_ref = args.array<double>("dV_ref");
        proteus::pyarray<double>& disp_trial_ref = args.array<double>("disp_trial_ref");
        proteus::pyarray<double>& disp_grad_trial_ref = args.array<double>("disp_grad_trial_ref");
        proteus::pyarray<double>& disp_test_ref = args.array<double>("disp_test_ref");
        proteus::pyarray<double>& disp_grad_test_ref = args.array<double>("disp_grad_test_ref");
        proteus::pyarray<double>& mesh_trial_trace_ref = args.array<double>("mesh_trial_trace_ref");
        proteus::pyarray<double>& mesh_grad_trial_trace_ref = args.array<double>("mesh_grad_trial_trace_ref");
        proteus::pyarray<double>& dS_ref = args.array<double>("dS_ref");
        proteus::pyarray<double>& disp_trial_trace_ref = args.array<double>("disp_trial_trace_ref");
        proteus::pyarray<double>& disp_grad_trial_trace_ref = args.array<double>("disp_grad_trial_trace_ref");
        proteus::pyarray<double>& disp_test_trace_ref = args.array<double>("disp_test_trace_ref");
        proteus::pyarray<double>& disp_grad_test_trace_ref = args.array<double>("disp_grad_test_trace_ref");
        proteus::pyarray<double>& normal_ref = args.array<double>("normal_ref");
        proteus::pyarray<double>& boundaryJac_ref = args.array<double>("boundaryJac_ref");
        proteus::pyarray<double>& ebqe_penalty = args.array<double>("ebqe_penalty");
        int gravityStep = args.scalar<int>("gravityStep");
        proteus_index_t nElements_global = args.scalar<proteus_index_t>("nElements_global");
        proteus::pyarray<int>& materialTypes = args.array<int>("materialTypes");
        int nMaterialProperties = args.scalar<int>("nMaterialProperties");
        proteus::pyarray<double>& materialProperties = args.array<double>("materialProperties");
        double pore_fluid_unit_weight = args.scalar<double>("pore_fluid_unit_weight");
        proteus::pyarray<double>& pore_pressure_head_dof = args.array<double>("pore_pressure_head_dof");
        proteus::pyarray<double>& q_strain = args.array<double>("q_strain");
        proteus::pyarray<double>& q_strain0 = args.array<double>("q_strain0");
        proteus::pyarray<double>& q_strain_last = args.array<double>("q_strain_last");
        proteus::pyarray<double>& q_plasticStrain = args.array<double>("q_plasticStrain");
        proteus::pyarray<double>& q_plasticStrain_last = args.array<double>("q_plasticStrain_last");
        proteus::pyarray<double>& ebqe_strain = args.array<double>("ebqe_strain");
        proteus::pyarray<double>& ebqe_strain0 = args.array<double>("ebqe_strain0");
        proteus::pyarray<double>& ebqe_strain_last = args.array<double>("ebqe_strain_last");
        proteus::pyarray<double>& ebqe_plasticStrain = args.array<double>("ebqe_plasticStrain");
        proteus::pyarray<double>& ebqe_plasticStrain_last = args.array<double>("ebqe_plasticStrain_last");
        proteus::pyarray<proteus_index_t>& disp_l2g = args.array<proteus_index_t>("disp_l2g");
        proteus::pyarray<double>& u_dof = args.array<double>("u_dof");
        proteus::pyarray<double>& v_dof = args.array<double>("v_dof");
        proteus::pyarray<double>& w_dof = args.array<double>("w_dof");
        proteus::pyarray<double>& bodyForce = args.array<double>("bodyForce");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_u_u = args.array<proteus_csr_index_t>("csrRowIndeces_u_u");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_u_u = args.array<proteus_csr_index_t>("csrColumnOffsets_u_u");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_u_v = args.array<proteus_csr_index_t>("csrRowIndeces_u_v");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_u_v = args.array<proteus_csr_index_t>("csrColumnOffsets_u_v");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_u_w = args.array<proteus_csr_index_t>("csrRowIndeces_u_w");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_u_w = args.array<proteus_csr_index_t>("csrColumnOffsets_u_w");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_v_u = args.array<proteus_csr_index_t>("csrRowIndeces_v_u");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_v_u = args.array<proteus_csr_index_t>("csrColumnOffsets_v_u");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_v_v = args.array<proteus_csr_index_t>("csrRowIndeces_v_v");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_v_v = args.array<proteus_csr_index_t>("csrColumnOffsets_v_v");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_v_w = args.array<proteus_csr_index_t>("csrRowIndeces_v_w");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_v_w = args.array<proteus_csr_index_t>("csrColumnOffsets_v_w");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_w_u = args.array<proteus_csr_index_t>("csrRowIndeces_w_u");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_w_u = args.array<proteus_csr_index_t>("csrColumnOffsets_w_u");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_w_v = args.array<proteus_csr_index_t>("csrRowIndeces_w_v");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_w_v = args.array<proteus_csr_index_t>("csrColumnOffsets_w_v");
        proteus::pyarray<proteus_csr_index_t>& csrRowIndeces_w_w = args.array<proteus_csr_index_t>("csrRowIndeces_w_w");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_w_w = args.array<proteus_csr_index_t>("csrColumnOffsets_w_w");
        proteus::pyarray<double>& globalJacobian = args.array<double>("globalJacobian");
        proteus_index_t nExteriorElementBoundaries_global = args.scalar<proteus_index_t>("nExteriorElementBoundaries_global");
        proteus::pyarray<proteus_index_t>& exteriorElementBoundariesArray = args.array<proteus_index_t>("exteriorElementBoundariesArray");
        proteus::pyarray<proteus_index_t>& elementBoundaryElementsArray = args.array<proteus_index_t>("elementBoundaryElementsArray");
        proteus::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
        proteus::pyarray<int>& isDOFBoundary_u = args.array<int>("isDOFBoundary_u");
        proteus::pyarray<int>& isDOFBoundary_v = args.array<int>("isDOFBoundary_v");
        proteus::pyarray<int>& isDOFBoundary_w = args.array<int>("isDOFBoundary_w");
        proteus::pyarray<int>& isStressFluxBoundary_u = args.array<int>("isStressFluxBoundary_u");
        proteus::pyarray<int>& isStressFluxBoundary_v = args.array<int>("isStressFluxBoundary_v");
        proteus::pyarray<int>& isStressFluxBoundary_w = args.array<int>("isStressFluxBoundary_w");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_u_u = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_u_u");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_u_v = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_u_v");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_u_w = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_u_w");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_v_u = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_v_u");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_v_v = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_v_v");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_v_w = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_v_w");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_w_u = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_w_u");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_w_v = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_w_v");
        proteus::pyarray<proteus_csr_index_t>& csrColumnOffsets_eb_w_w = args.array<proteus_csr_index_t>("csrColumnOffsets_eb_w_w");
                            CompKernel<nSpace,nDOF_mesh_trial_element,nDOF_trial_element,nDOF_test_element> ck;
                            const int nSymTen(ck.nSymTen);
                            //
                            //loop over elements to compute volume integrals and load them into the element Jacobians and global Jacobian
                            //
                            for(proteus_index_t eN=0;eN<nElements_global;eN++)
                            {
                                double
                                    elementJacobian_u_u[nDOF_test_element][nDOF_trial_element],
                                    elementJacobian_u_v[nDOF_test_element][nDOF_trial_element],
                                    elementJacobian_u_w[nDOF_test_element][nDOF_trial_element],
                                    elementJacobian_v_u[nDOF_test_element][nDOF_trial_element],
                                    elementJacobian_v_v[nDOF_test_element][nDOF_trial_element],
                                    elementJacobian_v_w[nDOF_test_element][nDOF_trial_element],
                                    elementJacobian_w_u[nDOF_test_element][nDOF_trial_element],
                                    elementJacobian_w_v[nDOF_test_element][nDOF_trial_element],
                                    elementJacobian_w_w[nDOF_test_element][nDOF_trial_element];
                                    for (int i=0;i<nDOF_test_element;i++)
                                        for (int j=0;j<nDOF_trial_element;j++)
                                        {
                                            elementJacobian_u_u[i][j]=0.0;
                                            elementJacobian_u_v[i][j]=0.0;
                                            elementJacobian_u_w[i][j]=0.0;
                                            elementJacobian_v_u[i][j]=0.0;
                                            elementJacobian_v_v[i][j]=0.0;
                                            elementJacobian_v_w[i][j]=0.0;
                                            elementJacobian_w_u[i][j]=0.0;
                                            elementJacobian_w_v[i][j]=0.0;
                                            elementJacobian_w_w[i][j]=0.0;
                                        }
                                    for  (int k=0;k<nQuadraturePoints_element;k++)
                                    {
                                        const proteus_index_t eN_k=eN*nQuadraturePoints_element+k,
                                              eN_nDOF_trial_element = eN*nDOF_trial_element, //index to a vector at a quadrature point
                                              eN_nDOF_mesh_trial_element = eN*nDOF_mesh_trial_element; //index to a vector at a quadrature point

                                        //declare local storage
                                        double u=0.0,v=0.0,w=0.0,
                                                 D[nSpace*nSpace],
                                                 *grad_u(&D[0]),
                                                 *grad_v(&D[nSpace]),
                                                 *grad_w(&D[2*nSpace]),
                                                 jac[nSpace*nSpace],
                                                 jacDet,
                                                 jacInv[nSpace*nSpace],
                                                 disp_grad_trial[nDOF_trial_element*nSpace],
                                                 dV,
                                                 disp_test_dV[nDOF_test_element],
                                                 disp_grad_test_dV[nDOF_test_element*nSpace],
                                                 x,y,z,
                                                 G[nSpace*nSpace],G_dd_G,tr_G,
                                                 Delta_strain[nSymTen],strain[nSymTen],
                                                 stress[nSymTen],dstress[nSpace*nSpace*nSpace*nSpace],pore_pressure=0.0;
                                                 //get jacobian, etc for mapping reference element
                                                 ck.calculateMapping_element(eN,
                                                         k,
                                                         mesh_dof.data(),
                                                         mesh_l2g.data(),
                                                         mesh_trial_ref.data(),
                                                         mesh_grad_trial_ref.data(),
                                                         jac,
                                                         jacDet,
                                                         jacInv,
                                                         x,y,z);
                                                 //get the pressure head
                                                 ck.mapping.valFromDOF(pore_pressure_head_dof.data(),&mesh_l2g.data()[eN_nDOF_mesh_trial_element],&mesh_trial_ref.data()[k*nDOF_mesh_trial_element],pore_pressure);
                                                 pore_pressure *= pore_fluid_unit_weight;
                                                 pore_pressure = fmax(pore_pressure,0.0);//cek hack
                                                 //get the physical integration weight
                                                 dV = fabs(jacDet)*dV_ref.data()[k];
                                                 ck.calculateG(jacInv,G,G_dd_G,tr_G);
                                                 //get the trial function gradients
                                                 ck.gradTrialFromRef(&disp_grad_trial_ref.data()[k*nDOF_trial_element*nSpace],jacInv,disp_grad_trial);
                                                 //get the solution
                                                 ck.valFromDOF(u_dof.data(),&disp_l2g.data()[eN_nDOF_trial_element],&disp_trial_ref.data()[k*nDOF_trial_element],u);
                                                 ck.valFromDOF(v_dof.data(),&disp_l2g.data()[eN_nDOF_trial_element],&disp_trial_ref.data()[k*nDOF_trial_element],v);
                                                 ck.valFromDOF(w_dof.data(),&disp_l2g.data()[eN_nDOF_trial_element],&disp_trial_ref.data()[k*nDOF_trial_element],w);
                                                 //get the solution gradients
                                                 ck.gradFromDOF(u_dof.data(),&disp_l2g.data()[eN_nDOF_trial_element],disp_grad_trial,grad_u);
                                                 ck.gradFromDOF(v_dof.data(),&disp_l2g.data()[eN_nDOF_trial_element],disp_grad_trial,grad_v);
                                                 ck.gradFromDOF(w_dof.data(),&disp_l2g.data()[eN_nDOF_trial_element],disp_grad_trial,grad_w);
                                                 //precalculate test function products with integration weights
                                                 for (int j=0;j<nDOF_trial_element;j++)
                                                 {
                                                     disp_test_dV[j] = disp_test_ref.data()[k*nDOF_trial_element+j]*dV;
                                                     for (int I=0;I<nSpace;I++)
                                                     {
                                                         disp_grad_test_dV[j*nSpace+I] = disp_grad_trial[j*nSpace+I]*dV;//cek warning won't work for Petrov-Galerkin}
                                                 }
                                    }
                                    calculateStrain(D,Delta_strain);
                                    if (gravityStep)
                                    {
                                        double C[nSymTen*nSymTen],Cinv[nSymTen*nSymTen];
                                        elasticModuli(materialProperties.data()[materialTypes.data()[eN]*nMaterialProperties],//E,
                                                materialProperties.data()[materialTypes.data()[eN]*nMaterialProperties+1],//nu,
                                                C,Cinv);
                                        if (gravityStep == 2)
                                        {
                                            double f,df[nSymTen],r[nSymTen],dr[nSymTen*nSymTen],stress_3_init,E=materialProperties.data()[materialTypes.data()[eN]*nMaterialProperties],nu=materialProperties.data()[materialTypes.data()[eN]*nMaterialProperties+1];
                                            elasticStress(C,&q_strain0.data()[eN_k*nSymTen],stress);
                                            evaluateConstitutiveModel(&materialProperties.data()[materialTypes.data()[eN]*nMaterialProperties],
                                                    stress,f,df,r,dr,stress_3_init);//cek hack, this is just to get stress_3_init
                                            const double n_e = 0.6,
                                                  P_a = materialProperties.data()[materialTypes.data()[eN]*nMaterialProperties+12],
                                                  K_i = 500.0;
                                                  if (P_a > 0.0)//cek hack, if not a soil then set P_a=0.0
                                                  {
                                                      E =K_i*P_a*pow(fmax(stress_3_init,P_a)/P_a,n_e);
                                                      elasticModuli(E,nu,C,Cinv);
                                                  }
                                        }
                                        for (int i=0; i<nSymTen; i++)
                                        {
                                            strain[i] = Delta_strain[i] + q_strain_last.data()[eN_k*nSymTen+i];
                                            for (int j=0; j<nSymTen; j++)
                                                dstress[i*nSymTen+j] = C[i+j*nSymTen];
                                        }
                                        elasticStress(C,strain,stress);
                                        for (int i=0; i<nSpace; i++)
                                            stress[i] -= pore_pressure;
                                    }
                                    else
                                        evaluateCoefficients(usePicard,
                                                pore_pressure,
                                                &materialProperties.data()[materialTypes.data()[eN]*nMaterialProperties],
                                                &q_strain0.data()[eN_k*nSymTen],
                                                &q_strain_last.data()[eN_k*nSymTen],
                                                Delta_strain,
                                                &q_plasticStrain_last.data()[eN_k*nSymTen],
                                                &q_plasticStrain.data()[eN_k*nSymTen],
                                                stress,
                                                dstress);
                                    //
                                    //moving mesh
                                    //
                                    //omit for now
                                    //
                                    for(int i=0;i<nDOF_test_element;i++)
                                    {
                                        int i_nSpace = i*nSpace;
                                        for(int j=0;j<nDOF_trial_element;j++)
                                        {
                                            int j_nSpace = j*nSpace;

                                            elementJacobian_u_u[i][j] += ck.StressJacobian_u_u_weak(dstress,&disp_grad_trial[j_nSpace],&disp_grad_test_dV[i_nSpace]);
                                            elementJacobian_u_v[i][j] += ck.StressJacobian_u_v_weak(dstress,&disp_grad_trial[j_nSpace],&disp_grad_test_dV[i_nSpace]);
                                            elementJacobian_u_w[i][j] += ck.StressJacobian_u_w_weak(dstress,&disp_grad_trial[j_nSpace],&disp_grad_test_dV[i_nSpace]);

                                            elementJacobian_v_u[i][j] += ck.StressJacobian_v_u_weak(dstress,&disp_grad_trial[j_nSpace],&disp_grad_test_dV[i_nSpace]);
                                            elementJacobian_v_v[i][j] += ck.StressJacobian_v_v_weak(dstress,&disp_grad_trial[j_nSpace],&disp_grad_test_dV[i_nSpace]);
                                            elementJacobian_v_w[i][j] += ck.StressJacobian_v_w_weak(dstress,&disp_grad_trial[j_nSpace],&disp_grad_test_dV[i_nSpace]);

                                            elementJacobian_w_u[i][j] += ck.StressJacobian_w_u_weak(dstress,&disp_grad_trial[j_nSpace],&disp_grad_test_dV[i_nSpace]);
                                            elementJacobian_w_v[i][j] += ck.StressJacobian_w_v_weak(dstress,&disp_grad_trial[j_nSpace],&disp_grad_test_dV[i_nSpace]);
                                            elementJacobian_w_w[i][j] += ck.StressJacobian_w_w_weak(dstress,&disp_grad_trial[j_nSpace],&disp_grad_test_dV[i_nSpace]);
                                        }//j
                                    }//i
                            }//k
                            //
                            //load into element Jacobian into global Jacobian
                            //
                            for (int i=0;i<nDOF_test_element;i++)
                            {
                                proteus_index_t eN_i = eN*nDOF_test_element+i;
                                for (int j=0;j<nDOF_trial_element;j++)
                                {
                                    proteus_index_t eN_i_j = eN_i*nDOF_trial_element+j;
                                    // std::cout<<"i "<<i<<"j "<<j<<std::endl
                                    //            <<elementJacobian_u_u[i][j]<<'\t'
                                    //            <<elementJacobian_u_v[i][j]<<'\t'
                                    //            <<elementJacobian_u_w[i][j]<<std::endl
                                    //            <<elementJacobian_v_u[i][j]<<'\t'
                                    //            <<elementJacobian_v_v[i][j]<<'\t'
                                    //            <<elementJacobian_v_w[i][j]<<std::endl
                                    //            <<elementJacobian_w_u[i][j]<<'\t'
                                    //            <<elementJacobian_w_v[i][j]<<'\t'
                                    //            <<elementJacobian_w_w[i][j]<<std::endl;

                                    globalJacobian.data()[csrRowIndeces_u_u[eN_i] + csrColumnOffsets_u_u[eN_i_j]] += elementJacobian_u_u[i][j];
                                    globalJacobian.data()[csrRowIndeces_u_v[eN_i] + csrColumnOffsets_u_v[eN_i_j]] += elementJacobian_u_v[i][j];
                                    globalJacobian.data()[csrRowIndeces_u_w[eN_i] + csrColumnOffsets_u_w[eN_i_j]] += elementJacobian_u_w[i][j];

                                    globalJacobian.data()[csrRowIndeces_v_u[eN_i] + csrColumnOffsets_v_u[eN_i_j]] += elementJacobian_v_u[i][j];
                                    globalJacobian.data()[csrRowIndeces_v_v[eN_i] + csrColumnOffsets_v_v[eN_i_j]] += elementJacobian_v_v[i][j];
                                    globalJacobian.data()[csrRowIndeces_v_w[eN_i] + csrColumnOffsets_v_w[eN_i_j]] += elementJacobian_v_w[i][j];

                                    globalJacobian.data()[csrRowIndeces_w_u[eN_i] + csrColumnOffsets_w_u[eN_i_j]] += elementJacobian_w_u[i][j];
                                    globalJacobian.data()[csrRowIndeces_w_v[eN_i] + csrColumnOffsets_w_v[eN_i_j]] += elementJacobian_w_v[i][j];
                                    globalJacobian.data()[csrRowIndeces_w_w[eN_i] + csrColumnOffsets_w_w[eN_i_j]] += elementJacobian_w_w[i][j];
                                }//j
                            }//i
                        }//elements
                //
                //loop over exterior element boundaries to compute the surface integrals and load them into the global Jacobian
                //
                for (proteus_index_t ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++)
                {
                    proteus_index_t ebN = exteriorElementBoundariesArray.data()[ebNE],
                    eN  = elementBoundaryElementsArray.data()[ebN*2+0],
                    eN_nDOF_trial_element = eN*nDOF_trial_element,
                    eN_nDOF_mesh_trial_element = eN*nDOF_mesh_trial_element;
        int ebN_local = elementBoundaryLocalElementBoundariesArray.data()[ebN*2+0];
                    for  (int kb=0;kb<nQuadraturePoints_elementBoundary;kb++)
                    {
                        proteus_index_t ebNE_kb = ebNE*nQuadraturePoints_elementBoundary+kb;
                        int ebN_local_kb = ebN_local*nQuadraturePoints_elementBoundary+kb,
                                 ebN_local_kb_nSpace = ebN_local_kb*nSpace;

                        double
                            u_ext=0.0,
                            v_ext=0.0,
                            w_ext=0.0,
                            D_ext[nSpace*nSpace],
                            *grad_u_ext(&D_ext[0]),
                            *grad_v_ext(&D_ext[nSpace]),
                            *grad_w_ext(&D_ext[2*nSpace]),
                            fluxJacobian_u_u[nDOF_trial_element],
                            fluxJacobian_u_v[nDOF_trial_element],
                            fluxJacobian_u_w[nDOF_trial_element],
                            fluxJacobian_v_u[nDOF_trial_element],
                            fluxJacobian_v_v[nDOF_trial_element],
                            fluxJacobian_v_w[nDOF_trial_element],
                            fluxJacobian_w_u[nDOF_trial_element],
                            fluxJacobian_w_v[nDOF_trial_element],
                            fluxJacobian_w_w[nDOF_trial_element],
                            jac_ext[nSpace*nSpace],
                            jacDet_ext,
                            jacInv_ext[nSpace*nSpace],
                            boundaryJac[nSpace*(nSpace-1)],
                            metricTensor[(nSpace-1)*(nSpace-1)],
                            metricTensorDetSqrt,
                            disp_grad_trial_trace[nDOF_trial_element*nSpace],
                            dS,
                            disp_test_dS[nDOF_test_element],
                            normal[3],
                            x_ext,y_ext,z_ext,
                            G[nSpace*nSpace],G_dd_G,tr_G,h_penalty,
                            Delta_strain[nSymTen],
                            strain[nSymTen],
                            stress[nSymTen],
                            dstress[nSymTen*nSymTen],pore_pressure_ext;
                            ck.calculateMapping_elementBoundary(eN,
                                    ebN_local,
                                    kb,
                                    ebN_local_kb,
                                    mesh_dof.data(),
                                    mesh_l2g.data(),
                                    mesh_trial_trace_ref.data(),
                                    mesh_grad_trial_trace_ref.data(),
                                    boundaryJac_ref.data(),
                                    jac_ext,
                                    jacDet_ext,
                                    jacInv_ext,
                                    boundaryJac,
                                    metricTensor,
                                    metricTensorDetSqrt,
                                    normal_ref.data(),
                                    normal,
                                    x_ext,y_ext,z_ext);
                            //get the pressure head
                            ck.mapping.valFromDOF(pore_pressure_head_dof.data(),&mesh_l2g.data()[eN_nDOF_mesh_trial_element],&mesh_trial_trace_ref.data()[ebN_local_kb*nDOF_mesh_trial_element],pore_pressure_ext);
                            pore_pressure_ext *= pore_fluid_unit_weight;
                            pore_pressure_ext = fmax(pore_pressure_ext,0.0);
                            dS = metricTensorDetSqrt*dS_ref.data()[kb];
                            ck.calculateG(jacInv_ext,G,G_dd_G,tr_G);
                            //compute shape and solution information
                            //shape
                            ck.gradTrialFromRef(&disp_grad_trial_trace_ref.data()[ebN_local_kb_nSpace*nDOF_trial_element],jacInv_ext,disp_grad_trial_trace);
                            //solution and gradients
                            ck.valFromDOF(u_dof.data(),&disp_l2g.data()[eN_nDOF_trial_element],&disp_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],u_ext);
                            ck.valFromDOF(v_dof.data(),&disp_l2g.data()[eN_nDOF_trial_element],&disp_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],v_ext);
                            ck.valFromDOF(w_dof.data(),&disp_l2g.data()[eN_nDOF_trial_element],&disp_trial_trace_ref.data()[ebN_local_kb*nDOF_test_element],w_ext);
                            ck.gradFromDOF(u_dof.data(),&disp_l2g.data()[eN_nDOF_trial_element],disp_grad_trial_trace,grad_u_ext);
                            ck.gradFromDOF(v_dof.data(),&disp_l2g.data()[eN_nDOF_trial_element],disp_grad_trial_trace,grad_v_ext);
                            ck.gradFromDOF(w_dof.data(),&disp_l2g.data()[eN_nDOF_trial_element],disp_grad_trial_trace,grad_w_ext);
                            //precalculate test function products with integration weights
                            for (int j=0;j<nDOF_trial_element;j++)
                            {
                                disp_test_dS[j] = disp_test_trace_ref.data()[ebN_local_kb*nDOF_test_element+j]*dS;
                            }
                            //
                            //calculate the internal and external trace of the pde coefficients
                            //
                            calculateStrain(D_ext,Delta_strain);
                            if (gravityStep)
                            {
                                double C[nSymTen*nSymTen],Cinv[nSymTen*nSymTen];
                                elasticModuli(materialProperties.data()[materialTypes.data()[eN]*nMaterialProperties],//E,
                                        materialProperties.data()[materialTypes.data()[eN]*nMaterialProperties+1],//nu,
                                        C,Cinv);
                                if (gravityStep == 2)
                                {
                                    double f,df[nSymTen],r[nSymTen],dr[nSymTen*nSymTen],stress_3_init,E=materialProperties.data()[materialTypes.data()[eN]*nMaterialProperties],nu=materialProperties.data()[materialTypes.data()[eN]*nMaterialProperties+1];
                                    elasticStress(C,&ebqe_strain0.data()[ebNE_kb*nSymTen],stress);
                                    evaluateConstitutiveModel(&materialProperties.data()[materialTypes.data()[eN]*nMaterialProperties],
                                            stress,f,df,r,dr,stress_3_init);//cek hack, this is just to get stress_3_init
                                    const double n_e = 0.6,
                                          P_a = materialProperties.data()[materialTypes.data()[eN]*nMaterialProperties+12],
                                          K_i = 500.0;
                                          if (P_a > 0.0)//cek hack, if not a soil then set P_a=0.0
                                          {
                                              E =K_i*P_a*pow(fmax(stress_3_init,P_a)/P_a,n_e);
                                              elasticModuli(E,nu,C,Cinv);
                                          }
                                }
                                for (int i=0; i<nSymTen; i++)
                                {
                                    strain[i] = Delta_strain[i] + ebqe_strain_last.data()[ebNE_kb*nSymTen+i];
                                    for (int j=0; j<nSymTen; j++)
                                        dstress[i*nSymTen+j] = C[i+j*nSymTen];
                                }
                                elasticStress(C,strain,stress);
                                for (int i=0;i<nSpace;i++)
                                    stress[i] -= pore_pressure_ext;
                            }
                            else
                                evaluateCoefficients(usePicard,
                                        pore_pressure_ext,
                                        &materialProperties.data()[materialTypes.data()[eN]*nMaterialProperties],
                                        &ebqe_strain0.data()[ebNE_kb*nSymTen],
                                        &ebqe_strain_last.data()[ebNE_kb*nSymTen],
                                        Delta_strain,
                                        &ebqe_plasticStrain_last.data()[ebNE_kb*nSymTen],
                                        &ebqe_plasticStrain.data()[ebNE_kb*nSymTen],
                                        stress,
                                        dstress);
                            //
                            //calculate the flux jacobian
                            //
                            double E=materialProperties.data()[materialTypes.data()[eN]*nMaterialProperties],
                            nu=materialProperties.data()[materialTypes.data()[eN]*nMaterialProperties+1];
                            h_penalty=(E/(1.0+nu))*(1.0 + (nu/(1.0-2.0*nu)))*ebqe_penalty.data()[ebNE_kb];
                            for (int j=0;j<nDOF_trial_element;j++)
                            {
                                int j_nSpace = j*nSpace;

                                exteriorNumericalStressFluxJacobian(isDOFBoundary_u.data()[ebNE_kb],
                                        isDOFBoundary_v.data()[ebNE_kb],
                                        isDOFBoundary_w.data()[ebNE_kb],
                                        normal,
                                        dstress,
                                        h_penalty,
                                        disp_trial_trace_ref.data()[ebN_local_kb*nDOF_trial_element+j],
                                        &disp_grad_trial_trace[j_nSpace],
                                        fluxJacobian_u_u[j],
                                        fluxJacobian_u_v[j],
                                        fluxJacobian_u_w[j],
                                        fluxJacobian_v_u[j],
                                        fluxJacobian_v_v[j],
                                        fluxJacobian_v_w[j],
                                        fluxJacobian_w_u[j],
                                        fluxJacobian_w_v[j],
                                        fluxJacobian_w_w[j]);
                            }//j
                            //
                            //update the global Jacobian from the flux Jacobian
                            //
                            for (int i=0;i<nDOF_test_element;i++)
                            {
                                proteus_index_t eN_i = eN*nDOF_test_element+i;
                                for (int j=0;j<nDOF_trial_element;j++)
                                {
                                    proteus_index_t ebN_i_j = ebN*4*nDOF_test_X_trial_element + i*nDOF_trial_element + j;

                                    globalJacobian.data()[csrRowIndeces_u_u[eN_i] + csrColumnOffsets_eb_u_u[ebN_i_j]] += fluxJacobian_u_u[j]*disp_test_dS[i];
                                    globalJacobian.data()[csrRowIndeces_u_v[eN_i] + csrColumnOffsets_eb_u_v[ebN_i_j]] += fluxJacobian_u_v[j]*disp_test_dS[i];
                                    globalJacobian.data()[csrRowIndeces_u_w[eN_i] + csrColumnOffsets_eb_u_w[ebN_i_j]] += fluxJacobian_u_w[j]*disp_test_dS[i];

                                    globalJacobian.data()[csrRowIndeces_v_u[eN_i] + csrColumnOffsets_eb_v_u[ebN_i_j]] += fluxJacobian_v_u[j]*disp_test_dS[i];
                                    globalJacobian.data()[csrRowIndeces_v_v[eN_i] + csrColumnOffsets_eb_v_v[ebN_i_j]] += fluxJacobian_v_v[j]*disp_test_dS[i];
                                    globalJacobian.data()[csrRowIndeces_v_w[eN_i] + csrColumnOffsets_eb_v_w[ebN_i_j]] += fluxJacobian_v_w[j]*disp_test_dS[i];

                                    globalJacobian.data()[csrRowIndeces_w_u[eN_i] + csrColumnOffsets_eb_w_u[ebN_i_j]] += fluxJacobian_w_u[j]*disp_test_dS[i];
                                    globalJacobian.data()[csrRowIndeces_w_v[eN_i] + csrColumnOffsets_eb_w_v[ebN_i_j]] += fluxJacobian_w_v[j]*disp_test_dS[i];
                                    globalJacobian.data()[csrRowIndeces_w_w[eN_i] + csrColumnOffsets_eb_w_w[ebN_i_j]] += fluxJacobian_w_w[j]*disp_test_dS[i];
                                }//j
                            }//i
                    }//kb
                }//ebNE
        }//computeJacobian
    };//ElastoPlastic

    inline ElastoPlastic_base* newElastoPlastic(int nSpaceIn,
            int nQuadraturePoints_elementIn,
            int nDOF_mesh_trial_elementIn,
            int nDOF_trial_elementIn,
            int nDOF_test_elementIn,
            int nQuadraturePoints_elementBoundaryIn,
            int CompKernelFlag)
    {
        return proteus::chooseAndAllocateDiscretization<ElastoPlastic_base,ElastoPlastic,CompKernel>(nSpaceIn,
                nQuadraturePoints_elementIn,
                nDOF_mesh_trial_elementIn,
                nDOF_trial_elementIn,
                nDOF_test_elementIn,
                nQuadraturePoints_elementBoundaryIn,
                CompKernelFlag);
    }
}//proteus

#endif
