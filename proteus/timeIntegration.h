#ifndef VTIMEINTEGRATION_H
#define VTIMEINTEGRATION_H
#include "index_type.h"
/*!
 \file timeIntegration.h
 \brief c interface for implementation of time discretization algorithms
*/

/*Pseudo-Transient Continuation TTE formula choice for dt*/
extern
void psiTCtteDT(int nPoints,
		double tau,
		double dtn,
		double dtnm1,
		double * yn,
		double * ypn,
		double * ypnm1,
		double * dtnp1);


/*************************************************************************
   DG limiting schemes
 *************************************************************************/
extern
void applyDGlimitingP1Lagrange1d(int limiterFlag,
				 proteus_index_t nElements_global,
				 int nNodes_element,
				 int nElementBoundaries_element,
				 int nDOF_element,
				 proteus_index_t * elementNodesArray,
				 proteus_index_t * elementNeighborsArray,
				 double * nodeArray,
				 double * elementBarycentersArray,
				 proteus_index_t * l2g,
				 int * tag,
				 double * Uin,
				 double * Uout);

extern
void applyDGlimitingP1Lagrange1d_withVacuumTol(int enforcePositivity,
	       double vacuumTol,
	       proteus_index_t nElements_global,
	       int nNodes_element,
	       int nElementBoundaries_element,
	       int nDOF_element,
	       proteus_index_t * elementNodesArray,
	       proteus_index_t * elementNeighborsArray,
	       double * nodeArray,
	       double * elementBarycentersArray,
	       proteus_index_t * l2g,
	       int * tag,
	       double * Uin,
	       double * Uout);
extern
void computeElementNeighborShapeGradients(proteus_index_t nElements_global,
					  int nElementBoundaries_element,
					  int nSpace,
					  const proteus_index_t * elementBoundariesArray,
					  const proteus_index_t * elementNeighborsArray,
					  double * elementBarycentersArray,
					  double * elementBoundaryBarycentersArray,
					  double * elementNeighborShapeGradients);
extern
void computeCockburnDGlimiterArrays2d(proteus_index_t nElements_global,
				      int nElementBoundaries_element,
				      int nSpace,
				      const proteus_index_t * elementBoundariesArray,
				      const proteus_index_t * elementNeighborsArray,
				      const double * elementBarycentersArray,
				      const double * elementBoundaryBarycentersArray,
				      const double * elementNeighborShapeGradients,
				      double * alphas,
				      proteus_index_t * alphaNeighbors);
extern
void applyCockburnDGlimiterP1Lagrange2d(double nu,
					double Mh2,
					proteus_index_t nElements_global,
					int nElementBoundaries_element,
					int nSpace,
					int nDOF_element,
					proteus_index_t * elementNeighborsArray,
					proteus_index_t * l2g,
					int * tag,
					double * alphas,
					proteus_index_t * alphaNeighbors,
					double * Uin,
					double * Uout);
extern
void applyDurlofskyDGlimiterP1Lagrange2d(int killExtrema,
					 int allowMinWithUndershoot,
					 proteus_index_t nElements_global,
					 int nElementBoundaries_element,
					 int nNodes_element,
					 int nSpace,
					 int nDOF_element,
					 const proteus_index_t * elementNeighborsArray,
					 const proteus_index_t * elementBoundariesArray,
					 const proteus_index_t * elementNodesArray,
					 const double * nodeArray,
					 const double * elementBarycentersArray,
					 const double * elementBoundaryBarycentersArray,
					 const double * elementNeighborShapeGradients,
					 const proteus_index_t * l2g,
					 const double * grad_v0,
					 double * elementAverages,
					 int * tag,
					 double * Uin,
					 double * Uout);

extern
void applyDurlofskyDGlimiterP1Lagrange3d(int killExtrema,
					 int allowMinWithUndershoot,
					 proteus_index_t nElements_global,
					 int nElementBoundaries_element,
					 int nNodes_element,
					 int nSpace,
					 int nDOF_element,
					 const proteus_index_t * elementNeighborsArray,
					 const proteus_index_t * elementBoundariesArray,
					 const proteus_index_t * elementNodesArray,
					 const double * nodeArray,
					 const double * elementBarycentersArray,
					 const double * elementBoundaryBarycentersArray,
					 const double * elementNeighborShapeGradients,
					 const proteus_index_t * l2g,
					 const double * grad_v0,
					 double * elementAverages,
					 int * tag,
					 double * Uin,
					 double * Uout);
extern
void applyDurlofskyDGlimiterP1Lagrange2d_withVacuumTol(int killExtrema,
						       int allowMinWithUndershoot,
						       int enforcePositivity,
						       double vacuumTol,
						       proteus_index_t nElements_global,
						       int nElementBoundaries_element,
						       int nNodes_element,
						       int nSpace,
						       int nDOF_element,
						       const proteus_index_t * elementNeighborsArray,
						       const proteus_index_t * elementBoundariesArray,
						       const proteus_index_t * elementNodesArray,
						       const double * nodeArray,
						       const double * elementBarycentersArray,
						       const double * elementBoundaryBarycentersArray,
						       const double * elementNeighborShapeGradients,
						       const proteus_index_t * l2g,
						       const double * grad_v0,
						       double * elementAverages,
						       int * tag,
						       double * Uin,
						       double * Uout);

#endif
