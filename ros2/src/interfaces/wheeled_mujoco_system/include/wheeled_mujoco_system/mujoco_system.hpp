// MuJoCo-backed ros2_control hardware plugin (the "mujoco_bridge" pattern).
//
// Presents the wheeled biped as a SystemInterface so ros2_control treats
// MuJoCo like real hardware — switching sim <-> real is a launch parameter,
// not a controller change (sim and real differ only by hardware plugin).
//
// Semantics per CONTRACT.md (authoritative: official training cfg):
//   update_rate = 500 Hz, scene timestep = 0.001 s -> TWO mj_step per write
//   cycle. "hardware_pd_vel": this plugin closes the leg position PD
//   (Kp 60 / Kd 2, |tau| <= 40 Nm) and wheel velocity servos (Kd 0.2,
//   |tau| <= 5 Nm) around the targets the controller writes, applies the gas
//   spring force (400->600 N linear curve), mirroring the training env.
//
// MuJoCo is optional at build time (HAVE_MUJOCO); without it the plugin
// compiles to an inert skeleton so the rest of the workspace still builds.
#ifndef WHEELED_MUJOCO_SYSTEM__MUJOCO_SYSTEM_HPP_
#define WHEELED_MUJOCO_SYSTEM__MUJOCO_SYSTEM_HPP_

#include <memory>
#include <string>
#include <vector>

#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "rclcpp/macros.hpp"
#include "rclcpp_lifecycle/state.hpp"

#ifdef HAVE_MUJOCO
#include <mujoco/mujoco.h>
#endif

namespace wheeled_mujoco_system
{

class WheeledMujocoSystem : public hardware_interface::SystemInterface
{
public:
  RCLCPP_SHARED_PTR_DEFINITIONS(WheeledMujocoSystem)

  hardware_interface::CallbackReturn on_init(
    const hardware_interface::HardwareInfo & info) override;
  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;
  hardware_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State &) override;
  hardware_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State &) override;
  hardware_interface::return_type read(const rclcpp::Time &, const rclcpp::Duration &) override;
  hardware_interface::return_type write(const rclcpp::Time &, const rclcpp::Duration &) override;

private:
  // model path from ros2_control yaml: <hardware><param key="mjcf_path">
  std::string mjcf_path_;

  // per-joint mirrors of the ros2_control interfaces (contract joint order)
  std::vector<std::string> joint_names_;
  std::vector<double> joint_position_{};
  std::vector<double> joint_velocity_{};
  std::vector<double> position_commands_{};
  std::vector<double> velocity_commands_{};
  // imu state interfaces exposed on a synthetic "imu" joint
  std::array<double, 3> gyro_{};
  std::array<double, 3> gravity_{};

  // low-level loop gains ("hardware_pd_vel") + torque clamps (official)
  static constexpr double kLegKp = 60.0;
  static constexpr double kLegKd = 2.0;
  static constexpr double kLegTorqueLimit = 40.0;
  static constexpr double kWheelKv = 0.2;
  static constexpr double kWheelTorqueLimit = 5.0;
  // gas spring (official spring_settings, linear mode, no damping)
  static constexpr double kSpringOffset = 0.06076;
  static constexpr double kSpringTravel = 0.07;
  static constexpr double kSpringForceFree = 400.0;
  static constexpr double kSpringForceCompressed = 600.0;
  static constexpr int kPhysicsStepsPerTick = 2;   // timestep 0.001 -> 500 Hz control
  static constexpr double kInitBaseHeight = 0.22;  // absolute-height standing pose

#ifdef HAVE_MUJOCO
  mjModel * model_ = nullptr;
  mjData * data_ = nullptr;
  int root_body_id_ = -1;
  std::vector<int> leg_joint_ids_;
  std::vector<int> leg_act_ids_;     // actuator ids (by name {joint}_ctrl)
  std::vector<int> wheel_joint_ids_;
  std::vector<int> wheel_act_ids_;
  std::vector<int> spring_act_ids_;  // spring2 actuator ids (driven by this plugin)
#endif
};

}  // namespace wheeled_mujoco_system

#endif  // WHEELED_MUJOCO_SYSTEM__MUJOCO_SYSTEM_HPP_
