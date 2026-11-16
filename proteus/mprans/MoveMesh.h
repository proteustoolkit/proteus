#ifndef MOVEMESH_H
#define MOVEMESH_H
#include <cmath>
#include <iostream>
#include "CompKernel.h"
#include "ModelFactory.h"
#include "ArgumentsDict.h"
#include "pyarray.h"

namespace py = pybind11;

namespace proteus
{
  class MoveMesh_base
  {
  public:
    virtual ~MoveMesh_base(){}
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
  class MoveMesh : public MoveMesh_base
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

    MoveMesh():
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

    inline void evaluateCoefficients(double det_J,
                                     const double* materialProperties,
                                     double* strain,
                                     double* stress,
                                     double* dstress)
    {
      //cek hack/todo need to set E based on reference configuration
      const double strainTrace=(strain[sXX]+strain[sYY]+strain[sZZ]),
        E=materialProperties[0]/det_J,//for mesh motion penalize small elements
        nu=materialProperties[1];

      const double shear = E/(1.0+nu);
      const double bulk  = shear*(nu/(1.0-2.0*nu));

      for (int i=0;i<nSymTen;i++)
        for (int j=0;j<nSymTen;j++)
          dstress[i*nSymTen+j] = 0.0;
      stress[sXX] = shear*strain[sXX] + bulk*strainTrace;
      dstress[sXX*nSymTen+sXX] = bulk + shear;
      dstress[sXX*nSymTen+sYY] = bulk;
      dstress[sXX*nSymTen+sZZ] = bulk;

      stress[sYY] = shear*strain[sYY] + bulk*strainTrace;
      dstress[sYY*nSymTen+sXX] = bulk;
      dstress[sYY*nSymTen+sYY] = bulk + shear;
      dstress[sYY*nSymTen+sZZ] = bulk;

      stress[sZZ] = shear*strain[sZZ] + bulk*strainTrace;
      dstress[sZZ*nSymTen+sXX] = bulk;
      dstress[sZZ*nSymTen+sYY] = bulk;
      dstress[sZZ*nSymTen+sZZ] = bulk + shear;

      stress[sYZ] = shear*0.5*strain[sYZ];//the 1/2 comes from the Voigt notation
      dstress[sYZ*nSymTen+sYZ] = shear*0.5;

      stress[sXZ] = shear*0.5*strain[sXZ];//the 1/2 comes from the Voigt notation
      dstress[sXZ*nSymTen+sXZ] = shear*0.5;

      stress[sXY] = shear*0.5*strain[sXY];//the 1/2 comes from the Voigt notation
      dstress[sXY*nSymTen+sXY] = shear*0.5;
    }

    inline void exteriorNumericalStressFlux(const int& isDOFBoundary_u,
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
      if (isDOFBoundary_u == 1)
        {
          stressFlux_u = -(stress[sXX]*normal[X] + stress[sXY]*normal[Y] + stress[sXZ]*normal[Z] - penalty*(u - bc_u));
        }
      else if(isStressFluxBoundary_u == 1)
        {
          stressFlux_u = bc_stressFlux_u;
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
          stressFlux_v = bc_stressFlux_v;
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
          stressFlux_w = bc_stressFlux_w;
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
        proteus_index_t nElements_global = args.scalar<proteus_index_t>("nElements_global");
        proteus::pyarray<int>& materialTypes = args.array<int>("materialTypes");
        int nMaterialProperties = args.scalar<int>("nMaterialProperties");
        proteus::pyarray<double>& materialProperties = args.array<double>("materialProperties");
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
              proteus_index_t //eN_k = eN*nQuadraturePoints_element+k,
                //eN_k_nSpace=eN_k*nSpace,
                eN_nDOF_trial_element = eN*nDOF_trial_element;
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
                strain[ck.nSymTen],stress[ck.nSymTen],dstress[ck.nSymTen*ck.nSymTen];
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

              calculateStrain(D,strain);
              evaluateCoefficients(fabs(jacDet),
                                   &materialProperties.data()[materialTypes.data()[eN]*nMaterialProperties],
                                   strain,
                                   stress,
                                   dstress);
              //
              //update element residual
              //
              for(int i=0;i<nDOF_test_element;i++)
                {
                  int i_nSpace=i*nSpace;

                  elementResidual_u[i] += ck.Stress_u_weak(stress,&disp_grad_test_dV[i_nSpace]);
                  elementResidual_v[i] += ck.Stress_v_weak(stress,&disp_grad_test_dV[i_nSpace]);
                  elementResidual_w[i] += ck.Stress_w_weak(stress,&disp_grad_test_dV[i_nSpace]);
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
            eN  = elementBoundaryElementsArray.data()[ebN*2+0];
          int ebN_local = elementBoundaryLocalElementBoundariesArray.data()[ebN*2+0];
          proteus_index_t eN_nDOF_trial_element = eN*nDOF_trial_element;
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
                strain[ck.nSymTen],stress[ck.nSymTen],dstress[ck.nSymTen*ck.nSymTen],
                stressFlux_u,stressFlux_v,stressFlux_w;
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
              calculateStrain(D,strain);
              evaluateCoefficients(fabs(jacDet_ext),
                                   &materialProperties.data()[materialTypes.data()[eN]*nMaterialProperties],
                                   strain,
                                   stress,
                                   dstress);
              //
              //calculate the numerical fluxes
              //
              //cek debug
              //ebqe_penalty_ext[ebNE_kb] = 10.0;
              //
              ck.calculateGScale(G,normal,h_penalty);
              //cek hack
              h_penalty = 100.0/h_penalty;
              exteriorNumericalStressFlux(isDOFBoundary_u.data()[ebNE_kb],
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
        proteus_index_t nElements_global = args.scalar<proteus_index_t>("nElements_global");
        proteus::pyarray<int>& materialTypes = args.array<int>("materialTypes");
        int nMaterialProperties = args.scalar<int>("nMaterialProperties");
        proteus::pyarray<double>& materialProperties = args.array<double>("materialProperties");
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
              const proteus_index_t eN_nDOF_trial_element = eN*nDOF_trial_element; //index to a vector at a quadrature point

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
                strain[nSymTen],stress[nSymTen],dstress[nSpace*nSpace*nSpace*nSpace];
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
              calculateStrain(D,strain);
              evaluateCoefficients(fabs(jacDet),
                                   &materialProperties.data()[materialTypes.data()[eN]*nMaterialProperties],
                                   strain,
                                   stress,
                                   dstress);
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

                  globalJacobian.data()[csrRowIndeces_u_u.data()[eN_i] + csrColumnOffsets_u_u.data()[eN_i_j]] += elementJacobian_u_u[i][j];
                  globalJacobian.data()[csrRowIndeces_u_v.data()[eN_i] + csrColumnOffsets_u_v.data()[eN_i_j]] += elementJacobian_u_v[i][j];
                  globalJacobian.data()[csrRowIndeces_u_w.data()[eN_i] + csrColumnOffsets_u_w.data()[eN_i_j]] += elementJacobian_u_w[i][j];

                  globalJacobian.data()[csrRowIndeces_v_u.data()[eN_i] + csrColumnOffsets_v_u.data()[eN_i_j]] += elementJacobian_v_u[i][j];
                  globalJacobian.data()[csrRowIndeces_v_v.data()[eN_i] + csrColumnOffsets_v_v.data()[eN_i_j]] += elementJacobian_v_v[i][j];
                  globalJacobian.data()[csrRowIndeces_v_w.data()[eN_i] + csrColumnOffsets_v_w.data()[eN_i_j]] += elementJacobian_v_w[i][j];

                  globalJacobian.data()[csrRowIndeces_w_u.data()[eN_i] + csrColumnOffsets_w_u.data()[eN_i_j]] += elementJacobian_w_u[i][j];
                  globalJacobian.data()[csrRowIndeces_w_v.data()[eN_i] + csrColumnOffsets_w_v.data()[eN_i_j]] += elementJacobian_w_v[i][j];
                  globalJacobian.data()[csrRowIndeces_w_w.data()[eN_i] + csrColumnOffsets_w_w.data()[eN_i_j]] += elementJacobian_w_w[i][j];
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
            eN_nDOF_trial_element = eN*nDOF_trial_element;
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
                strain[nSymTen],
                stress[nSymTen],
                dstress[nSymTen*nSymTen];
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
              calculateStrain(D_ext,strain);
              evaluateCoefficients(fabs(jacDet_ext),
                                   &materialProperties.data()[materialTypes.data()[eN]*nMaterialProperties],
                                   strain,
                                   stress,
                                   dstress);
              //
              //calculate the flux jacobian
              //
              ck.calculateGScale(G,normal,h_penalty);
              //cek hack
              h_penalty = 100.0/h_penalty;
              // for (int II=0;II<nSymTen;II++)
              //   for (int JJ=0;<nSymTen;JJ++)
              //     dstress[II*nSymTen+JJ] = 0.0;
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

                      globalJacobian.data()[csrRowIndeces_u_u.data()[eN_i] + csrColumnOffsets_eb_u_u.data()[ebN_i_j]] += fluxJacobian_u_u[j]*disp_test_dS[i];
                      globalJacobian.data()[csrRowIndeces_u_v.data()[eN_i] + csrColumnOffsets_eb_u_v.data()[ebN_i_j]] += fluxJacobian_u_v[j]*disp_test_dS[i];
                      globalJacobian.data()[csrRowIndeces_u_w.data()[eN_i] + csrColumnOffsets_eb_u_w.data()[ebN_i_j]] += fluxJacobian_u_w[j]*disp_test_dS[i];

                      globalJacobian.data()[csrRowIndeces_v_u.data()[eN_i] + csrColumnOffsets_eb_v_u.data()[ebN_i_j]] += fluxJacobian_v_u[j]*disp_test_dS[i];
                      globalJacobian.data()[csrRowIndeces_v_v.data()[eN_i] + csrColumnOffsets_eb_v_v.data()[ebN_i_j]] += fluxJacobian_v_v[j]*disp_test_dS[i];
                      globalJacobian.data()[csrRowIndeces_v_w.data()[eN_i] + csrColumnOffsets_eb_v_w.data()[ebN_i_j]] += fluxJacobian_v_w[j]*disp_test_dS[i];

                      globalJacobian.data()[csrRowIndeces_w_u.data()[eN_i] + csrColumnOffsets_eb_w_u.data()[ebN_i_j]] += fluxJacobian_w_u[j]*disp_test_dS[i];
                      globalJacobian.data()[csrRowIndeces_w_v.data()[eN_i] + csrColumnOffsets_eb_w_v.data()[ebN_i_j]] += fluxJacobian_w_v[j]*disp_test_dS[i];
                      globalJacobian.data()[csrRowIndeces_w_w.data()[eN_i] + csrColumnOffsets_eb_w_w.data()[ebN_i_j]] += fluxJacobian_w_w[j]*disp_test_dS[i];
                    }//j
                }//i
            }//kb
        }//ebNE
    }//computeJacobian


    inline void Invert3by3(double* Amat,double* Ainv)
    {

      Ainv[0*3+0] =  Amat[1*3+1] * Amat[2*3+2] - Amat[2*3+1] * Amat[1*3+2];
      Ainv[0*3+1] =  Amat[2*3+1] * Amat[0*3+2] - Amat[0*3+1] * Amat[2*3+2];
      Ainv[0*3+2] =  Amat[0*3+1] * Amat[1*3+2] - Amat[0*3+2] * Amat[1*3+1];

      double tmp  = 1.0 / ( Ainv[0*3+0] * Amat[0*3+0]
                          + Ainv[0*3+1] * Amat[1*3+0]
                          + Ainv[0*3+2] * Amat[2*3+0] );
      Ainv[0*3+0] = Ainv[0*3+0] * tmp;
      Ainv[0*3+1] = Ainv[0*3+1] * tmp;
      Ainv[0*3+2] = Ainv[0*3+2] * tmp;

      Ainv[1*3+0] = (Amat[1*3+2] * Amat[2*3+0] - Amat[1*3+0] * Amat[2*3+2]) * tmp;
      Ainv[1*3+1] = (Amat[0*3+0] * Amat[2*3+2] - Amat[2*3+0] * Amat[0*3+2]) * tmp;
      Ainv[1*3+2] = (Amat[1*3+0] * Amat[0*3+2] - Amat[0*3+0] * Amat[1*3+2]) * tmp;

      Ainv[2*3+0] = (Amat[1*3+0] * Amat[2*3+1] - Amat[1*3+1] * Amat[2*3+0]) * tmp;
      Ainv[2*3+1] = (Amat[2*3+0] * Amat[0*3+1] - Amat[0*3+0] * Amat[2*3+1]) * tmp;
      Ainv[2*3+2] = (Amat[0*3+0] * Amat[1*3+1] - Amat[0*3+1] * Amat[1*3+0]) * tmp;

    }

    /* //---------------------------------------------------------------------------- */
    /* virtual void  moveRigidBody(double  mass,           double* inertiaRef, */
    /*                          double* force,          double* moment, */
    /*                          double* disp0,          double* disp1, */
    /*                          double* vel0,           double* vel1, */
    /*                          double* rot0,           double* rot1, */
    /*                          double* angVel0,        double* angVel1, */
    /*                          double  deltaT, */
    /*                          int*    linConstraints, int*    angConstraints, */
    /*                          double  linRelaxFac,    double  angRelaxFac, */
    /*                          double  linNorm,        double  angNorm, */
    /*                          int     iterMax) */
    /* {                          */
    /*   int i,j,k,l, iter; */
    /*   double inertia0[3*3], inertia1[3*3], res[3], res2[3*3], Ainv[3*3], angVelTen[3*3],RotLHS[3*3]; */

    /*   //std::cout<<" lin const = "<<linConstraints[0]<<"  "<<linConstraints[1]<<"  "<<linConstraints[2]<<std::endl;  */
    /*   //std::cout<<" ang const = "<<angConstraints[0]<<"  "<<angConstraints[1]<<"  "<<angConstraints[2]<<std::endl;  */

    /*   //----------------------------------------------- */
    /*   //   Linear balance */
    /*   //   Integrated using Crank-Nicolson (midpoint) */
    /*   //----------------------------------------------- */
    /*   for (i = 0; i < 3; i++) */
    /*   { */
    /*      if(linConstraints[i] >= 1)  */
    /*      { */
    /*        vel1[i]  = 0.0; */
    /*        disp1[i] = disp0[i]; */
    /*        res[i] = 0.0; */
    /*      }   */
    /*      else                 */
    /*      { */
    /*         res[i] = force[i] - (mass/deltaT)*(vel1[i]-vel0[i]); */
    /*         vel1[i]  += linRelaxFac*(deltaT/mass)*res[i]; */

    /*         disp1[i] = disp0[i] + 0.5*deltaT*(vel1[i] + vel0[i]); */
    /*      } */
    /*   } */

    /*   linNorm = 0.0; */
    /*   for (i = 0; i < 3; i++) */
    /*   { */
    /*      linNorm += res[i]*res[i];  */
    /*   } */
    /*   linNorm = sqrt(linNorm); */

    /*   //--------------------------------------- */
    /*   // Get inertia tensor - at time level 0 */
    /*   //---------------------------------------       */
    /*   for (i = 0; i < 3; i++) */
    /*   { */
    /*     for (j = 0; j < 3; j++) */
    /*     { */
    /*       inertia0[i*3+j] = 0.0; */
    /*       for (k = 0; k < 3; k++) */
    /*       { */
    /*         for (l = 0; l < 3; l++) */
    /*         { */
    /*             inertia0[i*3+j] += rot0[i*3+k]*inertiaRef[k*3+l]*rot0[j*3+l]; */
    /*         } */
    /*       } */
    /*     } */
    /*   } */

    /*   //--------------------------------------- */
    /*   for (iter = 0; iter<iterMax; iter++) */
    /*   {  */
    /*     //--------------------------------------- */
    /*     // Get inertia tensor - at time level 1 */
    /*     //---------------------------------------            */
    /*     for (i = 0; i < 3; i++) */
    /*     { */
    /*       for (j = 0; j < 3; j++) */
    /*       { */
    /*         inertia1[i*3+j] = 0.0; */
    /*         for (k = 0; k < 3; k++) */
    /*         { */
    /*           for (l = 0; l < 3; l++) */
    /*           { */
    /*              inertia1[i*3+j] += rot1[i*3+k]*inertiaRef[k*3+l]*rot1[j*3+l]; */
    /*           } */
    /*         } */
    /*       } */
    /*     } */
    /*     //--------------------------------------- */
    /*     // Get angular residuals */
    /*     //---------------------------------------   */
    /*     for (i = 0; i < 3; i++) */
    /*     { */
    /*       if(angConstraints[i] >= 1)  */
    /*       { */
    /*         res[i] =0.0; */

    /*         for (j = 0; j < 3; j++) */
    /*         {            */
    /*           inertia1[i*3+j] = 0.0; */
    /*           inertia1[j*3+i] = 0.0; */
    /*         } */
    /*         inertia1[i*3+i] = 1.0; */
    /*       } */
    /*       else */
    /*       {       */
    /*         res[i] = moment[i]; */
    /*         for (j = 0; j < 3; j++) */
    /*         { */
    /*           res[i] -= (inertia1[i*3+j]*angVel1[j] - inertia0[i*3+j]*angVel0[j])/deltaT; */
    /*         }   */
    /*       }   */
    /*     } */

    /*     angNorm = 0.0; */
    /*     for (i = 0; i < 3; i++) */
    /*     { */
    /*       angNorm += res[i]*res[i];  */
    /*     } */
    /*     angNorm = sqrt(angNorm); */

    /*     //--------------------------------------- */
    /*     //  Compute angular velocity */
    /*     //---------------------------------------       */
    /*     Invert3by3(inertia1,Ainv); */

    /*     for (i = 0; i < 3; i++) */
    /*     { */
    /*         for (j = 0; j < 3; j++) */
    /*         { */
    /*            angVel1[i] += (angRelaxFac*deltaT)*Ainv[i*3+j]*res[j]; */
    /*         } */
    /*     } */

    /*     //------------------------------------------------- */
    /*     //  Compute rotation matrix */
    /*     // */
    /*     //  Integrating using Crank-Nicolson (midpoint) */
    /*     //  According to Hughes-Winget the rotation matrix  */
    /*     //  remains orthonormal (a proper rotation matrix)  */
    /*     //   */
    /*     //------------------------------------------------- */
    /*     angVelTen[0*3+0] =  0.0; */
    /*     angVelTen[0*3+1] = -0.5*(angVel0[2]+angVel1[2]); */
    /*     angVelTen[0*3+2] =  0.5*(angVel0[1]+angVel1[1]); */

    /*     angVelTen[1*3+0] =  0.5*(angVel0[2]+angVel1[2]); */
    /*     angVelTen[1*3+1] =  0.0;       */
    /*     angVelTen[1*3+2] = -0.5*(angVel0[0]+angVel1[0]); */

    /*     angVelTen[2*3+0] = -0.5*(angVel0[1]+angVel1[1]); */
    /*     angVelTen[2*3+1] =  0.5*(angVel0[0]+angVel1[0]); */
    /*     angVelTen[2*3+2] =  0.0; */

    /*     for (i = 0; i < 3; i++) */
    /*     { */
    /*       for (j = 0; j < 3; j++) */
    /*       { */
    /*         res2[i*3+j] = - (rot1[i*3+j]-rot0[i*3+j])/deltaT; */

    /*         for (k = 0; k < 3; k++) */
    /*         { */
    /*            res2[i*3+j] += angVelTen[i*3+k]*0.5*(rot1[k*3+j]+rot0[k*3+j]); */
    /*         } */
    /*       } */
    /*     } */

    /*     for (i = 0; i < 3; i++) */
    /*     { */
    /*       for (j = 0; j < 3; j++) */
    /*       { */
    /*         RotLHS[i*3+j] = -0.5*angVelTen[i*3+j]; */
    /*       } */
    /*       RotLHS[i*3+i] += 1.0/deltaT; */
    /*     } */

    /*     Invert3by3(RotLHS,Ainv); */

    /*     for (i = 0; i < 3; i++) */
    /*     { */
    /*       for (j = 0; j < 3; j++) */
    /*       { */
    /*         for (k = 0; k < 3; k++) */
    /*         { */
    /*           rot1[i*3+j] += Ainv[i*3+k]*res2[k*3+j]; */
    /*         }   */
    /*       } */
    /*     } */

    /*   } // End of newton loop     */

    /*   //std::cout<<" lin const = "<<linConstraints[0]<<"  "<<linConstraints[1]<<"  "<<linConstraints[2]<<std::endl;  */
    /*   //std::cout<<" ang const = "<<angConstraints[0]<<"  "<<angConstraints[1]<<"  "<<angConstraints[2]<<std::endl;  */
    /* } */

  };                            //MoveMesh

  inline MoveMesh_base* newMoveMesh(int nSpaceIn,
                                int nQuadraturePoints_elementIn,
                                int nDOF_mesh_trial_elementIn,
                                int nDOF_trial_elementIn,
                                int nDOF_test_elementIn,
                                int nQuadraturePoints_elementBoundaryIn,
                                int CompKernelFlag)
  {
    return proteus::chooseAndAllocateDiscretization<MoveMesh_base,MoveMesh,CompKernel>(nSpaceIn,
                                                                                   nQuadraturePoints_elementIn,
                                                                                   nDOF_mesh_trial_elementIn,
                                                                                   nDOF_trial_elementIn,
                                                                                   nDOF_test_elementIn,
                                                                                   nQuadraturePoints_elementBoundaryIn,
                                                                                   CompKernelFlag);
  }
}//proteus

#endif
