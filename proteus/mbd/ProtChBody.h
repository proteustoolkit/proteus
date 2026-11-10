//#pragma once

#include "chrono/physics/ChSystemSMC.h"
#include "chrono/physics/ChSystem.h"
#include "chrono/timestepper/ChTimestepper.h"
#include "chrono/timestepper/ChTimestepperHHT.h"
#include "chrono/core/ChFrame.h"
#include "chrono/physics/ChLinkTSDA.h"
#include "chrono/geometry/ChTriangleMeshConnected.h"
#include <iostream>
#include <vector>
#include <stdexcept>
#include <fstream>

using namespace chrono;
using namespace std;

class cppSystem {
public:
  std::shared_ptr<ChSystemSMC> systemSMC;
  std::shared_ptr<ChSystem> system;
  double chrono_dt;
  std::string directory;
  cppSystem();
  void step(double proteus_dt, int n_substeps);
  void setDirectory(std::string dir);
  void setTimestepperType(std::string tstype, bool verbose);
  void setCollisionEnvelopeMargin(double envelope, double margin);
  void addMesh(std::shared_ptr<ChMesh> mesh);
  // the integrable state of the Chrono system, [T, nx, nv, x(nx), v(nv), a(nv)]: ChSystem::StateGather and
  // StateGatherAcceleration, and StateScatter back (a step retry puts the system back where it was)
  std::vector<double> stateGather();
  void stateScatter(const std::vector<double>& y);
};


class cppRigidBody {
public:
  ChVector3d free_x;
  ChVector3d free_r;
  ChVector3d pos;
  ChVector3d pos_last;
  ChVector3d pos0;
  std::vector<ChVector3d> trimesh_pos;
  std::vector<ChVector3d> trimesh_pos_last;
  std::vector<ChVector3d> trimesh_pos0;
  ChVector3d pos0_trimesh;
  ChQuaternion<> rotq0_trimesh;
  ChVector3d vel;
  ChVector3d vel_last;
  ChVector3d acc;
  ChVector3d acc_last;
  ChVector3d angvel;
  ChVector3d angvel_last;
  ChVector3d angacc;
  ChVector3d angacc_last;
  ChMatrix33<double> rotm;
  ChMatrix33<double> rotm_last;
  ChQuaternion<double> rotq;
  ChQuaternion<double> rotq_last;
  ChQuaternion<double> rotq0;
  ChVector3d F;
  ChVector3d F_last;
  ChVector3d M;
  ChVector3d M_last;
  std::shared_ptr<ChLinkLockLock> lock_motion;  // lock for prescribed motion
  double lock_motion_t_max;  // max time up to which lock motion is used
  double mass;
  double mooring_restlength;
  std::shared_ptr<ChLinkTSDA> spring;
  /* ChVector <> inertia; */
  double* inertia;
  unsigned int accumulator_force_idx, accumulator_torque_idx;
  shared_ptr<ChTriangleMeshConnected> trimesh;
  bool has_trimesh;
  std::shared_ptr<ChBody> body;
  cppSystem* system;
  cppRigidBody(cppSystem* system);
  ChVector3d hxyz(double* x, double t);
  double hx(double* x, double t);
  double hy(double* x, double t);
  double hz(double* x, double t);
  void calculate_init();
  void addAccumulators();
  void prestep(double* force, double* torque);
  void poststep();
  void setConstraints(double* free_x, double* free_y);
  void addSpring(double stiffness,
                 double damping,
                 double* fairlead,
                 double* anchor,
                 double rest_length);
  void addPrismaticLinksWithSpring(double* pris1,
                                   double* pris2,
                                   double stiffness,
                                   double damping,
                                   double rest_length);
  void addPrismaticLinkX(double* pris1);
  void setName(std::string name);
  void setPrescribedMotionPoly(double coeff1);
  void setPrescribedMotionSine(double a, double f);
  void setPrescribedMotionCustom(std::vector<double> t, std::vector<double> x,
                                 std::vector<double> y, std::vector<double> z,
                                 std::vector<double> ang, std::vector<double> ang2,
                                 std::vector<double> ang3, double t_max);
  void getTriangleMeshSDF(ChVector3d pos_node,
                          double* dist_n);
  void getTriangleMeshVel(double *x,
                          double dt,
                          double *vel);
  void updateTriangleMeshVisualisationPos();
};

cppSystem::cppSystem()
{
  /* systemSMC_sharedptr = chrono_types::make_shared<ChSystemSMC>(); */
  /* systemSMC = systemSMC_sharedptr.get(); */
  /* system = systemSMC; */
  chrono_dt = 0.000001;
  directory = "./";
  // SOLVER OPTIONS
  /* system->SetSolverType(ChSolver::Type::MINRES);  // SOLVER_MINRES: good convergence, supports FEA, does not support DVI yet */
  /* auto msolver = std::static_pointer_cast<ChSolverMINRES>(system->GetSolver()); */
  /* msolver->SetDiagonalPreconditioning(true); */
  /* system->SetSolverWarmStarting(true);  // this helps a lot to speedup convergence in this class of problems */
  /* system->SetMaxItersSolverSpeed(100); // max iteration for iterative solvers */
  /* system->SetMaxItersSolverStab(100); // max iteration for stabilization (iterative solvers) */
  /* system->SetTolForce(1e-10); */
  //system->SetMaxItersSolverSpeed(100);  
  //system->SetMaxItersSolverStab(100);  
  //system->SetTolForce(1e-14); // default: 0.001
  //system->SetMaxiter(200); // default: 6. Max constraints to reach tolerance on constraints.
  //system->SetTol(1e-10); // default: 0.0002. Tolerance for keeping constraints together.
  /* system->SetTimestepperType(ChTimestepper::Type::EULER_IMPLICIT_LINEARIZED); // used before: ChSystem::INT_EULER_IMPLICIT_LINEARIZED */
  /* if (auto mystepper = std::dynamic_pointer_cast<ChTimestepperHHT>(system->GetTimestepper())) { */
  /*   mystepper->SetAlpha(-0.2); */
  /* } */
}

void cppSystem::setTimestepperType(std::string tstype, bool verbose=false) {
  if (tstype == "HHT") {
    system->SetTimestepperType(ChTimestepper::Type::HHT);
    auto mystepper = std::dynamic_pointer_cast<ChTimestepperHHT>(system->GetTimestepper());
    mystepper->SetAlpha(-0.2);
    mystepper->SetMaxItersSuccess(10);
    mystepper->SetAbsTolerances(1e-6);
    //mystepper->SetMode(ChTimestepperHHT::POSITION);
    //mystepper->SetScaling(false);
    mystepper->SetVerbose(verbose);
  }
  else if (tstype == "Euler") {
    system->SetTimestepperType(ChTimestepper::Type::EULER_IMPLICIT_LINEARIZED);
  }
  else if (tstype == "Trapezoidal") {
    system->SetTimestepperType(ChTimestepper::Type::TRAPEZOIDAL);
  }
}

void cppSystem::step(double proteus_dt, int n_substeps=1)
{
  double dt2 = proteus_dt/(double)n_substeps;
  for (int i = 0; i < n_substeps; ++i) {
    system->DoStepDynamics(dt2);
  }
}

void cppSystem::addMesh(std::shared_ptr<ChMesh> mesh) {
  system->Add(mesh);
}

std::vector<double> cppSystem::stateGather()
{
  system->Setup();
  unsigned int nx = system->GetNumCoordsPosLevel();
  unsigned int nv = system->GetNumCoordsVelLevel();
  ChState x(nx, system.get());
  ChStateDelta v(nv, system.get());
  ChStateDelta a(nv, system.get());
  double T = 0.;
  system->StateGather(x, v, T);
  system->StateGatherAcceleration(a);
  std::vector<double> y;
  y.reserve(3 + nx + 2*nv);
  y.push_back(T);
  y.push_back((double)nx);
  y.push_back((double)nv);
  for (unsigned int i = 0; i < nx; ++i) y.push_back(x(i));
  for (unsigned int i = 0; i < nv; ++i) y.push_back(v(i));
  for (unsigned int i = 0; i < nv; ++i) y.push_back(a(i));
  return y;
}

void cppSystem::stateScatter(const std::vector<double>& y)
{
  system->Setup();
  unsigned int nx = system->GetNumCoordsPosLevel();
  unsigned int nv = system->GetNumCoordsVelLevel();
  if (y.size() != 3 + nx + 2*nv || (unsigned int)y[1] != nx || (unsigned int)y[2] != nv)
    throw std::runtime_error("cppSystem::stateScatter: the state does not match the Chrono system (a body or link was added or removed since it was gathered)");
  ChState x(nx, system.get());
  ChStateDelta v(nv, system.get());
  ChStateDelta a(nv, system.get());
  for (unsigned int i = 0; i < nx; ++i) x(i) = y[3 + i];
  for (unsigned int i = 0; i < nv; ++i) v(i) = y[3 + nx + i];
  for (unsigned int i = 0; i < nv; ++i) a(i) = y[3 + nx + nv + i];
  system->StateScatter(x, v, y[0], UpdateFlags::UPDATE_ALL);
  system->StateScatterAcceleration(a);
}

cppRigidBody::cppRigidBody(cppSystem* system):
  system(system)
{

  body = chrono_types::make_shared<ChBody>();
  accumulator_force_idx = body->AddAccumulator();
  accumulator_torque_idx = body->AddAccumulator();
  // add body to system
  /* system->system->AddBody(body); */ // now added externally in cython
  // basic attributes of body
  rotm = body->GetRotMat();
  rotm_last = body->GetRotMat();
  pos = body->GetPos();
  pos_last = body->GetPos();
  body->SetMass(mass);
  free_x = ChVector3d(1., 1., 1.);
  free_r = ChVector3d(1., 1., 1.);
  lock_motion_t_max = 0.;
  has_trimesh = false;
}

// The constructor's own body (above) is a throwaway placeholder -- the
// Cython layer (CouplingFSI.pyx's ProtChBody.__cinit__) immediately
// overwrites `body` with the ChBody actually shared with a ChBodyAddedMass
// instance ("self.thisptr.body = self.ChBodyAddedMass.sharedptr_chbody"),
// discarding the constructor's own ChBody along with the accumulator
// indices obtained from it. Without this, accumulator_force_idx/
// accumulator_torque_idx are stale indices into the *new* body's own
// (empty) accumulators vector -- ChBody::GetAccumulatedForce()/
// EmptyAccumulator()/AccumulateForce() index into it with no bounds
// checking, so prestep() segfaults on the very first call. Call this again
// on whatever body ends up actually being used, right after reassigning it.
void cppRigidBody::addAccumulators() {
  accumulator_force_idx = body->AddAccumulator();
  accumulator_torque_idx = body->AddAccumulator();
}

void cppSystem::setDirectory(std::string dir) {
  directory = dir;
}

void cppRigidBody::updateTriangleMeshVisualisationPos() {
  /* rotm = body->GetRotMat(); */
  for (int i = 0; i < trimesh_pos.size(); i++) {
    ChVector3d local = ChFrame<double>(pos0_trimesh,rotq0_trimesh).TransformPointParentToLocal(trimesh_pos0[i]);
    ChVector3d xNew  = ChFrame<double>(pos,rotq).TransformPointLocalToParent(local);
    trimesh_pos[i].Set(xNew.x(), xNew.y(), xNew.z());
  }
}

ChVector3d cppRigidBody::hxyz(double* x, double t)
{
  /* rotm = body->GetRotMat(); */
  ChVector3d xx = ChVector3d(x[0], x[1], x[2]);
  ChVector3d local = ChFrame<double>(pos_last, rotq_last).TransformPointParentToLocal(xx);
  ChVector3d xNew  = ChFrame<double>(pos, rotq).TransformPointLocalToParent(local);
  return xNew - xx;
}


void cppSystem::setCollisionEnvelopeMargin(double envelope, double margin) {
  ChCollisionModel::SetDefaultSuggestedEnvelope(envelope);
  ChCollisionModel::SetDefaultSuggestedMargin(margin);
}

double cppRigidBody::hx(double* x, double t)
{
  /* rotm = body->GetRotMat(); */
  ChVector3d local = ChFrame<double>(pos_last, rotq_last).TransformPointParentToLocal(ChVector3d(x[0],x[1],x[2]));
  ChVector3d xNew  = ChFrame<double>(pos, rotq).TransformPointLocalToParent(local);
  return xNew.x() - x[0];
}

double cppRigidBody::hy(double* x, double t)
{
  /* rotm = body->GetRotMat(); */
  ChVector3d local = ChFrame<double>(pos_last, rotq_last).TransformPointParentToLocal(ChVector3d(x[0],x[1],x[2]));
  ChVector3d xNew  = ChFrame<double>(pos, rotq).TransformPointLocalToParent(local);
  return xNew.y() - x[1];
}

double cppRigidBody::hz(double* x, double t)
{
  /* rotm = body->GetRotMat(); */
  ChVector3d local = ChFrame<double>(pos_last, rotq_last).TransformPointParentToLocal(ChVector3d(x[0],x[1],x[2]));
  ChVector3d xNew = ChFrame<double>(pos, rotq).TransformPointLocalToParent(local);
  return xNew.z() - x[2];
}

void cppRigidBody::calculate_init() {
  pos0 = body->GetPos();
  rotq0 = body->GetRot();
  if (has_trimesh == true) {
    trimesh_pos.clear();
    trimesh_pos0.clear();
    auto trimesh_coords = trimesh->GetCoordsVertices();
    for (int i = 0; i < trimesh_coords.size(); i++) {
      trimesh_pos0.push_back(ChVector3d(trimesh_coords[i].x(),
					trimesh_coords[i].y(),
					trimesh_coords[i].z()));
      trimesh_pos.push_back(ChVector3d(trimesh_coords[i].x(),
				       trimesh_coords[i].y(),
				       trimesh_coords[i].z()));
    }
  }
}

void cppRigidBody::prestep(double* force, double* torque)
{
  /* step to call before running chrono system step */
  pos_last = body->GetPos();
  vel_last = body->GetPosDt();
  if (has_trimesh == true) {
    trimesh_pos_last = trimesh->GetCoordsVertices();
  }
  acc_last = body->GetPosDt2();
  rotm_last = body->GetRotMat();
  rotq_last = body->GetRot();
  angacc_last = body->GetAngAccLocal();
  angvel_last = body->GetAngVelLocal();
  F_last = body->GetAccumulatedForce(accumulator_force_idx);
  M_last = body->GetAccumulatedTorque(accumulator_torque_idx);
  // apply external forces
  body->EmptyAccumulator(accumulator_force_idx);
  body->EmptyAccumulator(accumulator_torque_idx);
  // calculate opposite force of gravity if free_x is 0
  double forceG[3]={0.,0.,0.};
  if (free_x.x() == 0) {forceG[0] = -system->system->GetGravitationalAcceleration().x()*body->GetMass();}
  if (free_x.y() == 0) {forceG[1] = -system->system->GetGravitationalAcceleration().y()*body->GetMass();}
  if (free_x.z() == 0) {forceG[2] = -system->system->GetGravitationalAcceleration().z()*body->GetMass();}
  body->AccumulateForce(accumulator_force_idx, ChVector3d(forceG[0]+force[0]*free_x.x(),
				   forceG[1]+force[1]*free_x.y(),
				   forceG[2]+force[2]*free_x.z()),
			pos_last,
			false);
  body->AccumulateTorque(accumulator_torque_idx,ChVector3d(torque[0]*free_r.x(),
				    torque[1]*free_r.y(),
				    torque[2]*free_r.z()),
			 false);
  if (spring!=0) {
    double spring_length = spring->GetLength();
    if (spring_length < mooring_restlength) {
      spring->SetDisabled(true);//SetRestLength(spring_length);
    }
    else {
      spring->SetDisabled(false);//SetRestLength(mooring_restlength);
    }
  }
}



void cppRigidBody::poststep()
{
  pos = body->GetPos();
  vel = body->GetPosDt();
  acc = body->GetPosDt2();
  rotm = body->GetRotMat();
  rotq = body->GetRot();
  angacc = body->GetAngAccLocal();
  angvel = body->GetAngVelLocal();
  F = body->GetAccumulatedForce(accumulator_force_idx);
  M = body->GetAccumulatedTorque(accumulator_torque_idx);
  if (lock_motion_t_max > 0) {
    double t = system->system->GetChTime();
    if (lock_motion_t_max < t && lock_motion->IsDisabled() == false) {
      lock_motion->SetDisabled(true);
    }
  }
}

void cppRigidBody::setPrescribedMotionCustom(std::vector<double> t,
                                             std::vector<double> x,
                                             std::vector<double> y,
                                             std::vector<double> z,
                                             std::vector<double> ang,
                                             std::vector<double> ang2,
                                             std::vector<double> ang3,
                                             double t_max) {
  auto fixed_body = chrono_types::make_shared<ChBody>();
  fixed_body->SetPos(body->GetPos());
  fixed_body->SetFixed(true);
  system->system->Add(fixed_body);
  lock_motion = chrono_types::make_shared<ChLinkLockLock>();
  lock_motion_t_max = t_max;
  lock_motion->Initialize(body, fixed_body, fixed_body->GetFrameCOMToAbs());
  system->system->Add(lock_motion);
  if (x.size() > 0) {
    auto forced_motion = chrono_types::make_shared<ChFunctionInterp>();
    for (int i = 0; i < x.size(); i++) {
      forced_motion->AddPoint(t[i], x[i]);
    }
    std::shared_ptr<ChFunction> forced_ptr = forced_motion;
    lock_motion->SetMotionX(forced_ptr);
  }
  if (y.size() > 0) {
    auto forced_motion = chrono_types::make_shared<ChFunctionInterp>();
    for (int i = 0; i < y.size(); i++) {
      forced_motion->AddPoint(t[i], y[i]);
    }
    std::shared_ptr<ChFunction> forced_ptr = forced_motion;
    lock_motion->SetMotionY(forced_ptr);
  }
  if (z.size() > 0) {
    auto forced_motion = chrono_types::make_shared<ChFunctionInterp>();
    for (int i = 0; i < z.size(); i++) {
      forced_motion->AddPoint(t[i], z[i]);
    }
    std::shared_ptr<ChFunction> forced_ptr = forced_motion;
    lock_motion->SetMotionZ(forced_ptr);
  }
  if (ang.size() > 0) {
    auto forced_motion = chrono_types::make_shared<ChFunctionInterp>();
    for (int i = 0; i < ang.size(); i++) {
      forced_motion->AddPoint(t[i], ang[i]);
    }
    std::shared_ptr<ChFunction> forced_ptr = forced_motion;
    lock_motion->SetMotionAng1(forced_ptr);
  }
  if (ang2.size() > 0) {
    auto forced_motion = chrono_types::make_shared<ChFunctionInterp>();
    for (int i = 0; i < ang2.size(); i++) {
      forced_motion->AddPoint(t[i], ang2[i]);
    }
    std::shared_ptr<ChFunction> forced_ptr = forced_motion;
    lock_motion->SetMotionAng2(forced_ptr);
  }
  if (ang3.size() > 0) {
    auto forced_motion = chrono_types::make_shared<ChFunctionInterp>();
    for (int i = 0; i < ang3.size(); i++) {
      forced_motion->AddPoint(t[i], ang3[i]);
    }
    std::shared_ptr<ChFunction> forced_ptr = forced_motion;
    lock_motion->SetMotionAng3(forced_ptr);
  }
}

void cppRigidBody::setPrescribedMotionPoly(double coeff1) {
  auto fixed_body = chrono_types::make_shared<ChBody>();
  fixed_body->SetPos(body->GetPos());
  fixed_body->SetFixed(true);
  system->system->Add(fixed_body);
  auto lock = chrono_types::make_shared<ChLinkLockLock>();
  lock->Initialize(body, fixed_body, fixed_body->GetFrameCOMToAbs());
  system->system->Add(lock);
  auto forced_motion = chrono_types::make_shared<ChFunctionPoly>();
  forced_motion->SetCoefficients(std::vector<double>{coeff1});
  std::shared_ptr<ChFunction> forced_ptr = forced_motion;
  lock->SetMotionX(forced_ptr);
}


void cppRigidBody::setPrescribedMotionSine(double a, double f) {
  auto fixed_body = chrono_types::make_shared<ChBody>();
  fixed_body->SetPos(body->GetPos());
  fixed_body->SetFixed(true);
  system->system->Add(fixed_body);
  auto lock = chrono_types::make_shared<ChLinkLockLock>();
  lock->Initialize(body, fixed_body, fixed_body->GetFrameCOMToAbs());
  system->system->Add(lock);
  auto forced_motion = chrono_types::make_shared<ChFunctionSine>();
  forced_motion->SetAmplitude(a);
  forced_motion->SetFrequency(f);
  std::shared_ptr<ChFunction> forced_ptr = forced_motion;
  lock->SetMotionX(forced_ptr);
}

void cppRigidBody::setConstraints(double* free_x_in, double* free_r_in){
  free_x = ChVector3d(free_x_in[0], free_x_in[1], free_x_in[2]);
  free_r = ChVector3d(free_r_in[0], free_r_in[1], free_r_in[2]);
}

void cppRigidBody::addSpring(double stiffness,
                             double damping,
                             double* fairlead,
                             double* anchor,
                             double rest_length)
{
  mooring_restlength = rest_length;
  spring = chrono_types::make_shared<ChLinkTSDA>();
  std::shared_ptr<ChBody> anchor_body = chrono_types::make_shared<ChBody>();
  anchor_body->SetPos(ChVector3d(anchor[0], anchor[1], anchor[2]));
  anchor_body->SetFixed(true);
  system->system->AddBody(anchor_body);
  spring->Initialize(body,
                     anchor_body,
                     true, // true for pos relative to bodies
                     ChVector3d(fairlead[0], fairlead[1], fairlead[2]),
                     ChVector3d(0.,0.,0.));
  spring->SetSpringCoefficient(stiffness);
  spring->SetDampingCoefficient(damping);
  system->system->AddLink(spring);
}

void cppRigidBody::addPrismaticLinkX(double* pris1)
{
  auto mybod2 = chrono_types::make_shared<ChBody>();
  mybod2->SetName("PRIS1");
  mybod2->SetPos(ChVector3d(pris1[0], pris1[1], pris1[2]));
  mybod2->SetMass(0.00001);
  mybod2->SetFixed(true);
  system->system->AddBody(mybod2);
  auto mylink1 = chrono_types::make_shared<ChLinkLockPrismatic>();
  ChQuaternion<double> Q;
  Q.SetFromAngleAxis(CH_PI/2., VECT_Y);
  auto mycoordsys1 = ChFrame<double>(mybod2->GetPos(), Q);//Q_from_AngAxis(CH_C_PI / 2, VECT_X));
  mylink1->Initialize(mybod2, body, mycoordsys1);
  system->system->AddLink(mylink1);
}

void cppRigidBody::addPrismaticLinksWithSpring(double* pris1,
                                               double* pris2,
                                               double stiffness,
                                               double damping,
                                               double rest_length)
{
  mooring_restlength = rest_length;
  auto fairlead = chrono_types::make_shared<ChBody>();
  fairlead->SetName("PRIS3");
  fairlead->SetPos(body->GetPos());
  fairlead->SetMass(0.00001);
  system->system->AddBody(fairlead);
  auto mybod2 = chrono_types::make_shared<ChBody>();
  mybod2->SetName("PRIS1");
  mybod2->SetPos(ChVector3d(pris1[0], pris1[1], pris1[2]));
  mybod2->SetMass(0.00001);
  //mybod2->AddForce(-system->system->GetGravitationalAcceleration());
  //mybod2->SetFixed(true);
  system->system->AddBody(mybod2);
  auto mybod3 = chrono_types::make_shared<ChBody>();
  mybod3->SetName("PRIS2");
  mybod3->SetPos(ChVector3d(pris2[0], pris2[1], pris2[2]));
  mybod3->SetFixed(true);
  system->system->AddBody(mybod3);

  auto mylink1 = chrono_types::make_shared<ChLinkLockPrismatic>();
  system->system->AddLink(mylink1);
  ChQuaternion<double> QX;
  QX.SetFromAngleAxis(CH_PI/2., VECT_Y);
  auto mycoordsys1 = ChFrame<double>(mybod2->GetPos(), QX);//Q_from_AngAxis(CH_C_PI / 2, VECT_X));
  mylink1->Initialize(fairlead, mybod2, mycoordsys1);



  auto mylink2 = chrono_types::make_shared<ChLinkLockPrismatic>();
  system->system->AddLink(mylink2);
  ChQuaternion<double> QY;
  QY.SetFromAngleAxis(CH_PI/2., VECT_X);
  auto mycoordsys2 = ChFrame<double>(mybod3->GetPos(), QY);//Q_from_AngAxis(CH_C_PI / 2, VECT_X));
  mylink2->Initialize(mybod2, mybod3,mycoordsys2);

  auto mylink3 = chrono_types::make_shared<ChLinkLockSpherical>();
  //auto mylink3 = chrono_types::make_shared<ChLinkLockRevolute>();
  //mylink3->SetMotion_axis(ChVector3d(0.,1.,0.));
  system->system->AddLink(mylink3);
  mylink3->Initialize(fairlead, body, false, fairlead->GetFrameCOMToAbs(), body->GetFrameCOMToAbs());



  spring = chrono_types::make_shared<ChLinkTSDA>();
  spring->Initialize(fairlead,
                     mybod2,
                     true, // true for pos relative to bodies
                     ChVector3d(0.,0.,0.),
                     ChVector3d(0.,0.,0.));
  spring->SetSpringCoefficient(stiffness);
  spring->SetDampingCoefficient(damping);
  spring->SetName("SPRING1");
  system->system->AddLink(spring);
}

void cppRigidBody::setName(std::string name) {
  body->SetName(name);
}

void cppRigidBody::getTriangleMeshSDF(ChVector3d pos,
                                      double* dist_n) {
  auto xxs = trimesh->GetCoordsVertices();
  auto nns = trimesh->GetCoordsNormals();
  ChVector3d dist_vec;
  double min_dist=1e10;
  double dist;
  for (int i = 0; i < xxs.size(); i++) {
    dist_vec = pos-xxs[i];
    dist = dist_vec.Length();
    if (dist < min_dist) {
      min_dist = dist;
    }
    if (dist_vec.Dot(nns[i]) > 0) { // outside
      min_dist = min_dist;
    }
    else {  // inside
      min_dist = -min_dist;
    }
  }
  dist_n[0] = min_dist;
  // normal to shape
  // actually just vector to closest node here
  dist_n[1] = dist_vec[0];
  dist_n[1] = dist_vec[1];
  dist_n[2] = dist_vec[2];
};

void cppRigidBody::getTriangleMeshVel(double *x,
                                      double dt,
                                      double *vel) {
  auto xxs = trimesh->GetCoordsVertices();
  auto nns = trimesh->GetCoordsNormals();
  double min_dist = 1e10;
  ChVector3d p(x[0], x[1], x[2]);
  ChVector3d d_vector(0.0);
  ChVector3d ddlast;
  // find closest node
  int node_closest = -1;
  for (int i = 0; i < xxs.size(); i++) {
    double dist = (p - xxs[i]).Length();
    if (dist < min_dist) {
      min_dist = dist;
      node_closest = i;
    }
  }
  if (node_closest != -1) {
    ddlast = xxs[node_closest]-trimesh_pos_last[node_closest];
    vel[0] = ddlast.x()/dt;
    vel[1] = ddlast.y()/dt;
    vel[2] = ddlast.z()/dt;
  }
}


cppSystem * newSystem()
{
  return new cppSystem();
}



cppRigidBody * newRigidBody(cppSystem* system)
{
  return new cppRigidBody(system);
}



void ChLinkLockBodies(std::shared_ptr<ChBody> body1,
                      std::shared_ptr<ChBody> body2,
                      std::shared_ptr<ChSystem> system,
                      ChFrame<double> frame,
                      double limit_X=0.,
                      double limit_Y=0.,
                      double limit_Z=0.,
                      double limit_Rx=0.,
                      double limit_Ry=0.,
                      double limit_Rz=0.) {
  auto mylink = chrono_types::make_shared<ChLinkLock>();
  system->AddLink(mylink);
  auto chlimit_X = mylink->LimitX();
  chlimit_X.SetActive(true);
  chlimit_X.SetMax(limit_X);
  auto chlimit_Y = mylink->LimitY();
  chlimit_Y.SetActive(true);
  chlimit_Y.SetMax(limit_Y);
  auto chlimit_Z = mylink->LimitZ();
  chlimit_Z.SetActive(true);
  chlimit_Z.SetMax(limit_Z);
  auto chlimit_Rx = mylink->LimitRx();
  chlimit_Rx.SetMax(limit_Rx);
  chlimit_Rx.SetActive(true);
  auto chlimit_Ry = mylink->LimitRy();
  chlimit_Ry.SetActive(true);
  chlimit_Ry.SetMax(limit_Ry);
  auto chlimit_Rz = mylink->LimitRz();
  chlimit_Rz.SetActive(true);
  chlimit_Rz.SetMax(limit_Rz);
  mylink->Initialize(body1, body2, frame);
}

struct no_op_delete
{
  void operator()(void*) { }
};

std::shared_ptr<ChPhysicsItem> getPhysicsItemSharedPtr(ChPhysicsItem* item) {
  std::shared_ptr<ChPhysicsItem> sp(item, no_op_delete());
  return sp;
}
static std::map<ChPhysicsItem*, std::shared_ptr<ChPhysicsItem>> spans;

std::shared_ptr<ChPhysicsItem> getPhysicsItemSharedPtr2(ChPhysicsItem* item) {
  std::shared_ptr<ChPhysicsItem> sp = spans.at(item);
  /* std::shared_ptr<ChPhysicsItem> sp(item, no_op_delete()); */
  return sp;
}

