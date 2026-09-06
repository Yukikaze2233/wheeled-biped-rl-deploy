// Implementation of WheeledMujocoSystem — see mujoco_system.hpp.
#include "wheeled_mujoco_system/mujoco_system.hpp"

#include <algorithm>
#include <cmath>

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/rclcpp.hpp"

namespace wheeled_mujoco_system
{

hardware_interface::CallbackReturn WheeledMujocoSystem::on_init(
  const hardware_interface::HardwareInfo & info)
{
  if (hardware_interface::SystemInterface::on_init(info) !=
      hardware_interface::CallbackReturn::SUCCESS) {
    return hardware_interface::CallbackReturn::ERROR;
  }

  mjcf_path_ = info_.hardware_parameters.count("mjcf_path")
                 ? info_.hardware_parameters.at("mjcf_path")
                 : "";

  joint_names_.clear();
  for (const auto & joint : info_.joints) {
    joint_names_.push_back(joint.name);
  }
  const size_t n = joint_names_.size();
  joint_position_.assign(n, 0.0);
  joint_velocity_.assign(n, 0.0);
  position_commands_.assign(n, 0.0);
  velocity_commands_.assign(n, 0.0);
  gyro_.fill(0.0);
  gravity_.fill(0.0);
  return hardware_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface>
WheeledMujocoSystem::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> states;
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    states.emplace_back(joint_names_[i], hardware_interface::HW_IF_POSITION, &joint_position_[i]);
    states.emplace_back(joint_names_[i], hardware_interface::HW_IF_VELOCITY, &joint_velocity_[i]);
  }
  // IMU exposed on a synthetic "imu" joint so the controller's
  // state_interface_configuration() names resolve ("imu/gyro.x", ...)
  states.emplace_back("imu", "gyro.x", &gyro_[0]);
  states.emplace_back("imu", "gyro.y", &gyro_[1]);
  states.emplace_back("imu", "gyro.z", &gyro_[2]);
  states.emplace_back("imu", "gravity.x", &gravity_[0]);
  states.emplace_back("imu", "gravity.y", &gravity_[1]);
  states.emplace_back("imu", "gravity.z", &gravity_[2]);
  return states;
}

std::vector<hardware_interface::CommandInterface>
WheeledMujocoSystem::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> commands;
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    // contract: legs get position targets, wheels velocity targets; the
    // ros2_control yaml declares which is which per joint.
    const auto & type = info_.joints[i].command_interfaces[0].name;
    if (type == hardware_interface::HW_IF_POSITION) {
      commands.emplace_back(joint_names_[i], hardware_interface::HW_IF_POSITION, &position_commands_[i]);
    } else {
      commands.emplace_back(joint_names_[i], hardware_interface::HW_IF_VELOCITY, &velocity_commands_[i]);
    }
  }
  return commands;
}

hardware_interface::CallbackReturn WheeledMujocoSystem::on_configure(
  const rclcpp_lifecycle::State &)
{
#ifdef HAVE_MUJOCO
  if (!mjcf_path_.empty()) {
    model_ = mj_loadXML(mjcf_path_.c_str(), nullptr, nullptr, 0);
    if (model_ == nullptr) {
      RCLCPP_ERROR(rclcpp::get_logger("WheeledMujocoSystem"), "mj_loadXML failed: %s", mjcf_path_.c_str());
      return hardware_interface::CallbackReturn::ERROR;
    }
    data_ = mj_makeData(model_);
    // root body for IMU reads: the free-joint body (scene has floating_base),
    // falling back to "base_link" for non-floating models.
    root_body_id_ = -1;
    for (int j = 0; j < model_->njnt; ++j) {
      if (model_->jnt_type[j] == mjJNT_FREE) { root_body_id_ = model_->jnt_bodyid[j]; break; }
    }
    if (root_body_id_ < 0) {
      root_body_id_ = mj_name2id(model_, mjOBJ_BODY, "base_link");
    }
    // absolute-height standing pose: base z = 0.22 m, joints at ref
    if (root_body_id_ >= 0 && model_->nq > 2) data_->qpos[2] = kInitBaseHeight;
    for (const auto & name : joint_names_) {
      const int id = mj_name2id(model_, mjOBJ_JOINT, name.c_str());
      if (id < 0) {
        RCLCPP_ERROR(rclcpp::get_logger("WheeledMujocoSystem"), "joint not in model: %s", name.c_str());
        return hardware_interface::CallbackReturn::ERROR;
      }
      // actuators are addressed BY NAME ({joint}_ctrl): MJCF actuator order
      // is not the contract order (CONTRACT.md, verified in sim2sim).
      const std::string act_name = name + "_ctrl";
      const int act = mj_name2id(model_, mjOBJ_ACTUATOR, act_name.c_str());
      if (name.find("wheel") != std::string::npos) {
        wheel_joint_ids_.push_back(id);
        wheel_act_ids_.push_back(act);
      } else {
        leg_joint_ids_.push_back(id);
        leg_act_ids_.push_back(act);
      }
    }
    // gas-spring actuators driven by this plugin (not ros2_control interfaces)
    for (int a = 0; a < model_->nu; ++a) {
      const char * aname = mj_id2name(model_, mjOBJ_ACTUATOR, a);
      if (aname && std::string(aname).find("spring2") != std::string::npos) {
        spring_act_ids_.push_back(a);
      }
    }
    mj_forward(model_, data_);
  }
#endif
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn WheeledMujocoSystem::on_activate(
  const rclcpp_lifecycle::State &)
{
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::return_type WheeledMujocoSystem::read(
  const rclcpp::Time &, const rclcpp::Duration &)
{
#ifdef HAVE_MUJOCO
  if (model_ == nullptr) {
    return hardware_interface::return_type::OK;
  }
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    const int id = mj_name2id(model_, mjOBJ_JOINT, joint_names_[i].c_str());
    joint_position_[i] = data_->qpos[model_->jnt_qposadr[id]];
    joint_velocity_[i] = data_->qvel[model_->jnt_dofadr[id]];
  }
  // root body-frame angular velocity + projected gravity (free-joint body)
  mjtNum vel[6];
  mj_objectVelocity(model_, data_, mjOBJ_BODY, root_body_id_, vel, 1);
  gyro_ = {vel[0], vel[1], vel[2]};
  const double * R = data_->xmat + 9 * root_body_id_;
  // projected gravity = R^T @ [0, 0, -1] -> negate the third column of R
  gravity_ = {-R[2], -R[5], -R[8]};
#endif
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type WheeledMujocoSystem::write(
  const rclcpp::Time &, const rclcpp::Duration &)
{
#ifdef HAVE_MUJOCO
  if (model_ == nullptr) {
    return hardware_interface::return_type::OK;
  }
  // close the low-level loops ("hardware_pd_vel"), 2 physics substeps per
  // 500 Hz tick (scene timestep 0.001), PD recomputed every substep (official
  // semantics), torque-clamped, springs applied.
  for (int s = 0; s < kPhysicsStepsPerTick; ++s) {
    for (size_t i = 0; i < leg_joint_ids_.size(); ++i) {
      const int id = leg_joint_ids_[i];
      const double q = data_->qpos[model_->jnt_qposadr[id]];
      const double dq = data_->qvel[model_->jnt_dofadr[id]];
      double tau = kLegKp * (position_commands_[i] - q) - kLegKd * dq;
      tau = std::clamp(tau, -kLegTorqueLimit, kLegTorqueLimit);
      const int act = leg_act_ids_[i];
      if (act >= 0) data_->ctrl[act] = tau;
    }
    for (size_t i = 0; i < wheel_joint_ids_.size(); ++i) {
      const int id = wheel_joint_ids_[i];
      const double dq = data_->qvel[model_->jnt_dofadr[id]];
      double tau = kWheelKv * (velocity_commands_[i] - dq);
      tau = std::clamp(tau, -kWheelTorqueLimit, kWheelTorqueLimit);
      const int act = wheel_act_ids_[i];
      if (act >= 0) data_->ctrl[act] = tau;
    }
    // gas springs: F = 400 + 200/0.07 * clamp(0.06076 - q, min=0) N
    for (const int act : spring_act_ids_) {
      const int jnt = model_->actuator_trnid[act * 2];
      const double q = data_->qpos[model_->jnt_qposadr[jnt]];
      const double compression = std::max(kSpringOffset - q, 0.0);
      data_->ctrl[act] = kSpringForceFree +
        (kSpringForceCompressed - kSpringForceFree) / kSpringTravel * compression;
    }
    mj_step(model_, data_);
  }
#endif
  return hardware_interface::return_type::OK;
}

}  // namespace wheeled_mujoco_system

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(wheeled_mujoco_system::WheeledMujocoSystem,
                       hardware_interface::SystemInterface)
