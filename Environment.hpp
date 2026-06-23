//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <stdlib.h>
#include <set>
#include "../../RaisimGymEnv.hpp"
#include <cmath>

namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {

 public:

  explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable), uniDist_ (-1.0, 1.0) {

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
    gaitclock_.setZero(8);

    rewardpos_ = 0.0;
    rewardneg_ = 0.0;

    ///for creating a bound in cmd velocity generation
    maxlinvel_ = 1.0;
    maxangvel_ = 0.7;
    READ_YAML(int, commandresamplestep_, cfg_["commandresamplestep"])

    //foot clearance
    footframeindex_[0] = hound_ -> getFrameIdxByName("FR_foot");
    footframeindex_[1] = hound_ -> getFrameIdxByName("FL_foot");
    footframeindex_[2] = hound_ -> getFrameIdxByName("RR_foot");
    footframeindex_[3] = hound_ -> getFrameIdxByName("RL_foot");
    READ_YAML(double, gaitperiod_, cfg_["gaitperiod"])

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
    obDim_ = 45;
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
  
    //indices of links that should not make contact with ground
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
    command_ << maxlinvel_ * uniDist_(gen_), maxlinvel_ * uniDist_(gen_), maxangvel_ * uniDist_(gen_);
    stepcounter_ = 0;
    rewardpos_ = 0.0;
    rewardneg_ = 0.0;
    /*
    for (size_t i = 0; i < 4; i++){
      stancetime_[i] = 0.0;
      airtime_[i] = 0.0;
    }
    */
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

    //for airtime calculation, detect current contact state per foot
    //bool currentcontact_[4] = {false, false, false, false};
    //getting all the current contact information of hound
    //for (auto& contact : hound_ -> getContacts()){
      //size_t currentcontactinfo_ = contact.getlocalBodyIndex();
      //for (size_t i = 0; i < 4; i++){
        //if (currentcontactinfo_ == footbodyindex_[i]) {
          //currentcontact_[i] = true;
        //}
      //}
    //}

    rewardpos_ = 0.0;
    rewardneg_ = 0.0;

    rewards_.record("torque", hound_->getGeneralizedForce().squaredNorm()); //the sum of squared torques across all joints
    
    //reward for tracking the velocity command
    double linvelerr = (command_.head(2) - bodyLinearVel_.head(2)).squaredNorm();
    double angvelerr = (command_[2] - bodyAngularVel_[2]);
    rewards_.record("linearvelerr", exp(-linvelerr));
    rewards_.record("angularvelerr", exp(-0.5 * (angvelerr * angvelerr)));

    //penalize vertical motion to fix bounciness
    rewards_.record("verticalvelocity", bodyLinearVel_[2] * bodyLinearVel_[2]);

    //penalize dragging/jumping behavior
    /*
    //double airtimereward_ = 0.0;
    //bool stancecommand_ = (command_.head(2).norm() < 0.1 && abs(command_[2]) < 0.1); //first two computes vx, vy and second option computes yaw rate

    for (size_t i = 0; i < 4; i++){
      if (currentcontact_[i]) {
        stancetime_[i] += control_dt_;
        airtime_[i] = 0.0; //reset airtime while the leg is touching the ground
      } else {
        airtime_[i] += control_dt_;
        stancetime_[i] = 0.0; //reset stance timer while the leg is not touching the ground
      }

      double Ts = stancetime_[i];
      double Ta = airtime_[i];
      
      if (stancecommand_) {
        if (-0.3 > (Ts - Ta)) { airtimereward_ += -0.3; }
        else if (0.3 < (Ts - Ta)) { airtimereward_ += 0.3; }
        else { airtimereward_ += (Ts-Ta); }
      } else {
        if (std::max(Ts, Ta) < 0.25) { airtimereward_ += std::min(std::max(Ts, Ta), 0.2); } 
      }
    }

    rewards_.record("airtime", airtimereward_);
    */

    //foot clearance
    double gaittime_ = stepcounter_ * control_dt_;
    double footclearancereward_ = 0.0;
    double maxheight_ = 0.08;

    for (size_t i = 0; i < 4; i++) {
      double gaitcycle = sin(2 * M_PI * ((gaittime_ / gaitperiod_) + footphase_[i]));
      double desiredheight = maxheight_ * std::max(0.0, gaitcycle);
      
      hound_ -> getFramePosition(footframeindex_[i], footposition_[i]);
      double currentheight = footposition_[i][2];
      double heighterrorsquared_ = (currentheight - desiredheight) * (currentheight - desiredheight);

      hound_ -> getFrameVelocity(footframeindex_[i], footvelocity_[i]);
      double horizontalspeed = sqrt(footvelocity_[i][0] * footvelocity_[i][0] + footvelocity_[i][1] * footvelocity_[i][1]);

      footclearancereward_ += heighterrorsquared_ * horizontalspeed;
    }

    rewards_.record("footclearance", footclearancereward_);

    //periodically resamples the command velocity
    stepcounter_++;
    if (stepcounter_ % commandresamplestep_ == 0) {
      command_ << maxlinvel_ * uniDist_(gen_), maxlinvel_ * uniDist_(gen_), maxangvel_ * uniDist_(gen_);
      updateObservation();
    }

    static const std::set<std::string> positiveRewards_ = {"linearvelerr", "angularvelerr"};
    static const std::set<std::string> negativeRewards_ = {"torque", "verticalvelocity", "footclearance"};
    
    for (const auto& iterator : rewards_.getStdMap()) {
      const std::string& name = iterator.first;
      double rewardweight = static_cast<double>(iterator.second);

      if (positiveRewards_.count(name)) {
        rewardpos_ += rewardweight;
      } else if (negativeRewards_.count(name)) {
        rewardneg_ += rewardweight;
      }
    }

    return rewardpos_ * exp(0.2 * rewardneg_);
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

    double gaittime_ = stepcounter_ * control_dt_;
    for (size_t i = 0; i < 4; i++) {
      double gaitphase_ = 2 * M_PI * ((gaittime_ / gaitperiod_) + footphase_[i]);
      gaitclock_[i * 2] = sin(gaitphase_);
      gaitclock_[i * 2 + 1] = cos(gaitphase_);
    }

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
        command_, //command velocity vector
        gaitclock_;
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

  void curriculumUpdate() { 
    episodecounter_++;
    maxlinvel_ = 1.0 + (2.5 / (1.0 + exp(-0.002 * (episodecounter_ - 1000))));
  };

 private:
  int gcDim_, gvDim_, nJoints_;
  bool visualizable_ = false;
  raisim::ArticulatedSystem* hound_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
  double terminalRewardCoeff_ = -10.;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  std::set<size_t> footIndices_;

  ///command velocity
  Eigen::Vector3d command_;
  std::uniform_real_distribution<double> uniDist_;
  double maxlinvel_, maxangvel_;
  int commandresamplestep_;
  int stepcounter_ = 0;
  int episodecounter_ = 0;

  //cmd velocity (airtime)
  Eigen::VectorXd gaitclock_;
  double stancetime_[4] = {0, 0, 0, 0}; //T_s,i
  double airtime_[4] = {0, 0, 0, 0}; //T_a,i
  size_t footbodyindex_[4];

  //foot clearance
  size_t footframeindex_[4];
  double gaitperiod_;
  double footphase_[4] = {0, M_PI, M_PI, 0}; //FR, FL, RR, RL
  raisim::Vec<3> footposition_[4];
  raisim::Vec<3> footvelocity_[4];

  double rewardpos_, rewardneg_;

  /// these variables are not in use. They are placed to show you how to create a random number sampler.
  std::normal_distribution<double> normDist_;
  thread_local static std::mt19937 gen_;
};
thread_local std::mt19937 raisim::ENVIRONMENT::gen_;

}

