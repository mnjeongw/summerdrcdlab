//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <stdlib.h>
#include <set>
#include "../../RaisimGymEnv.hpp"

namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {

 public:

  explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable), normDist_(0, 1) {

    /// create world
    world_ = std::make_unique<raisim::World>();

    /// add objects
    //The robot is loaded from a URDF file (a standard robot description format defining links, joints, masses, limits)
    hound_ = world_->addArticulatedSystem(resourceDir_+"/hound/urdf/HoundOne.urdf");
    hound_->setName("hound");
    //The control mode connects directly to the actor's outputs
    hound_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    world_->addGround();

    /// get robot data
    //gcDim includes position + orientation (7 numbers, x, y, z + quaternion) plus joint angles (12 for Hound's 12 actuated joints) = 19
    gcDim_ = hound_->getGeneralizedCoordinateDim(); //generalized coordinates
    //gvDim is 6 (linear + angular velocity of the base) + 12 (joint velocities) = 18
    gvDim_ = hound_->getDOF(); //generalized velocities
    nJoints_ = gvDim_ - 6; //n-joint action space

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nJoints_);
    command_.setZero();

    /// this is nominal configuration of hound
    //This is the reset pose: the exact joint configuration the robot snaps back to at the start of every episode.
    //(0,0,0.55885) = starting position, (1.0, 0.0, 0.0, 0.0) = quaternion meaning 'no rotation'
    //remaining 12 numbers = initial angles for each of the 12 joints
    //if you want the robot to learn from a different starting condition, this is where you'd change it
    gc_init_ << 0, 0, 0.55885, 1.0, 0.0, 0.0, 0.0, 0.00, 0.8, -1.5, 0.00, 0.8, -1.5, 0.00, 0.8, -1.5, -0.00, 0.8, -1.5;

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    //P and D gains for the PD controller. Addresses how stiff/responsive the joints are to position commands.
    //Higher P gain = more aggressive correction toward the target angle
    //This is the physical tuning parameter, not something PPO learns.
    jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(300.0);
    jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(3);
    hound_->setPdGains(jointPgain, jointDgain);
    hound_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 37;
    actionDim_ = nJoints_; actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
    obDouble_.setZero(obDim_);

    /// action scaling
    //Prepare the action mean and action stdev vectors that will be used later in step() function to convert whatever the neural network outputs
    //into an actual physical joint command.
    //gc_init_ is the full 19 number initial pose vector (x, y, z + quaternion, 12 joint angles)
    //.tail(nJoints_) grabs just the last 12 numbers, so actionMean = 12 number vector holding the robot's default standing joint angles
    actionMean_ = gc_init_.tail(nJoints_); 
    double roll_action_std, hip_action_std, knee_action_std;
    READ_YAML(double, roll_action_std, cfg_["roll_action_std"]) //reading params from the config
    READ_YAML(double, hip_action_std, cfg_["hip_action_std"])
    READ_YAML(double, knee_action_std, cfg_["knee_action_std"])
    //actionStd_ = 12 number vector (one entry per joint).
    //setting each roll hip knee action std for four legs
    for (int leg = 0; leg < 4; leg++) {
      actionStd_[leg * 3 + 0] = roll_action_std;
      actionStd_[leg * 3 + 1] = hip_action_std;
      actionStd_[leg * 3 + 2] = knee_action_std;
    }

    /// Reward coefficients
    rewards_.initializeFromConfigurationFile (cfg["reward"]);

    /// indices of links that should not make contact with ground
    footIndices_.insert(hound_->getBodyIdx("FR_calf"));
    footIndices_.insert(hound_->getBodyIdx("FL_calf"));
    footIndices_.insert(hound_->getBodyIdx("RR_calf"));
    footIndices_.insert(hound_->getBodyIdx("RL_calf"));

    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer();
      server_->focusOn(hound_);
    }
  }

  void init() final { }

  //After every episode, snap back to the fixed initial pose, refresh the observation buffer.
  void reset() final {
    hound_->setState(gc_init_, gv_init_);
    command_ << normDist_(gen_),  normDist_(gen_), normDist_(gen_);
    updateObservation();
  }

  //The core per-timestep logic, called once per control_dt (Recall control_dt = how often the policy gets to act)
  //look at the parameter action. Action comes in from the actor network
  float step(const Eigen::Ref<EigenVec>& action) final {
    /// action scaling
    pTarget12_ = action.cast<double>();
    //action gets multiplied element wise by actionStd_ and shifted by actionMean_
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    //final joint target = standing pose angle + (actionStd_) * action network output
    //They did this such that if the action network outputs small numbers, the joint shifts only slightly away from standing.
    //Even with completely random initial weights, the multiplier actionStd_ squashes down the network's random output, so that very
    //first actions the robot ever takes are small wobbles around a known-good standing pose, rather than wild swings to an arbitrary angle.
    pTarget_.tail(nJoints_) = pTarget12_;

    //So the design trick of the robot is hardcoding the standing upright position to the formula, leaving the network to only solve 
    //the small useful angle adjustments to move forward. This shrinks the search space the network has to explore and thus speeds up the learning.

    hound_->setPdTarget(pTarget_, vTarget_);

    //The physics substep loop. control_dt_ / simulation_dt = substeps per control step
    //The PD target is set once, then physics integrates n substep times before the policy gets to act again
    for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-10); i++){
      if(server_) server_->lockVisualizationServerMutex();
      world_->integrate();
      if(server_) server_->unlockVisualizationServerMutex();
    }

    updateObservation();

    //This is where the reward gets computed. 
    //This is the exact place where you'd add a new term if you wanted a new task.
    rewards_.record("torque", hound_->getGeneralizedForce().squaredNorm()); //the sum of squared torques across all joints
    
    //reward for tracking the velocity command
    double linvelerr = (command_.head(2) - bodyLinearVel_.head(2)).squaredNorm();
    double angvelerr = (command_[2] - bodyAngularVel_[2]);
    rewards_.record("linearvelerr", -linvelerr);
    rewards_.record("angularvelerr", -(angvelerr * angvelerr));
    
    Eigen::Vector3d currentVel(bodyLinearVel_[0], bodyLinearVel_[1], bodyAngularVel_[2]); //(Vx, Vy, yaw rate)
    Eigen::Vector3d velError = command_ - currentVel;
    rewards_.record("velerror", -velError.squaredNorm());
    
    //penalizing joint velocity so the robot takes fewer, bigger steps
    rewards_.record("jointvel", gv_.tail(12).squaredNorm());

    //This applies the coeff multipliers from yaml automatically
    return rewards_.sum();
  }

  //Building the observation vector
  void updateObservation() {
    hound_->getState(gc_, gv_);
    raisim::Vec<4> quat;
    raisim::Mat<3,3> rot;
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);
    bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);

    //This is the entire sensory input the actor has access to. No terrain info, no camera, no foot contact sensors listed
    //If your new task needs information the policy doesn't currently have access to, (e.g. distance to a goal, a target heading, contact state per foot),
    //This is exactly where you'd add it. You'd need to also bump obDim_ in the constructor to match the new total length
    obDouble_ << gc_[2], /// body height
        rot.e().row(2).transpose(), /// body orientation 
        //rot.e().row(2) is the third row of the rotation matrix, which gives the orientation of the robot's 'up' vector relative to gravity
        //This is useful to tell the network 'how tilted the robot is' without needing the full quaternion.
        gc_.tail(12), /// joint angles
        bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity
        gv_.tail(12), /// joint velocity
        command_; //command velocity vector
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to float
    ob = obDouble_.cast<float>();
  }

  //Defining failure
  bool isTerminalState(float& terminalReward) final {
    terminalReward = float(terminalRewardCoeff_);

    ///If any contact involves a body part other than the four feet, the episode terminates immediately and the robot receives a one time big penalty and the episode is over.
    for(auto& contact: hound_->getContacts())
      //If only the feet are touching ground, no termination, no penalty, episode continues.
      if(footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end())
        return true;

    terminalReward = 0.f;
    return false;
  }

  void curriculumUpdate() { };

 private:
  int gcDim_, gvDim_, nJoints_;
  bool visualizable_ = false;
  raisim::ArticulatedSystem* hound_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
  double terminalRewardCoeff_ = -10.;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  Eigen::Vector3d command_;
  std::set<size_t> footIndices_;

  /// these variables are not in use. They are placed to show you how to create a random number sampler.
  std::normal_distribution<double> normDist_;
  thread_local static std::mt19937 gen_;
};
thread_local std::mt19937 raisim::ENVIRONMENT::gen_;

}

