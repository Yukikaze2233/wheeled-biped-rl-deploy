// MuJoCo-backed ros2_control hardware plugin (the "mujoco_bridge" pattern).
//
// Presents the wheeled biped as a SystemInterface so ros2_control treats
// MuJoCo like real hardware — switching sim <-> real is a launch parameter,
// not a controller change (sim and real differ only by hardware plugin).
//
// Semantics per CONTRACT.md:
//   update_rate = 500 Hz, MuJoCo timestep = 0.002 s -> one mj_step per read/
//   write cycle. "hardware_pd_vel": this plugin closes the leg position PD
//   (Kp 60 / Kd 2) and wheel velocity servos (0.2) around the targets the
//   controller writes, mirroring the firmware's low-level loops.
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

  // low-level loop gains ("hardware_pd_vel")
  static constexpr double kLegKp = 60.0;
  static constexpr double kLegKd = 2.0;
  static constexpr double kWheelKv = 0.2;

#ifdef HAVE_MUJOCO
  mjModel * model_ = nullptr;
  mjData * data_ = nullptr;
  std::vector<int> leg_joint_ids_;
  std::vector<int> wheel_joint_ids_;
#endif
};

}  // namespace wheeled_mujoco_system

#endif  // WHEELED_MUJOCO_SYSTEM__MUJOCO_SYSTEM_HPP_
