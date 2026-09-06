// Wheeled-biped RL controller — interface wiring + 500 Hz update loop.
// Logic lives in the modules (module layout): fsm/, robot_state/, onnxruntime/.
#ifndef WHEELED_RL_CONTROLLER__RL_CONTROLLER_HPP_
#define WHEELED_RL_CONTROLLER__RL_CONTROLLER_HPP_

#include <array>
#include <memory>
#include <string>

#include "controller_interface/controller_interface.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64.hpp"
#include "wheeled_rl_controller/policy_runtime.hpp"
#include "wheeled_rl_controller/robot_state.hpp"
#include "wheeled_rl_controller/state_machine.hpp"

namespace wheeled_rl
{

constexpr size_t kActionDim = 6;  // CONTRACT.md section 3
constexpr int kControlHz = 500;
constexpr int kPolicyHz = 50;
constexpr int kTicksPerPolicy = kControlHz / kPolicyHz;
constexpr double kLegActionScale = 0.5;
constexpr double kWheelActionScale = 10.0;   // wheel_vel_action_scale (velocity mode)
constexpr double kMaxWheelVel = 150.0;       // max_wheel_vel = 100 * 1.5 (official)
// pipeline delays in POLICY steps (20 ms each); training always uses
// obs 20-80 ms / act 20-60 ms. Measured best on the official 13k policy:
// obs 4 steps (80 ms), act 3 steps (60 ms).
constexpr int kMaxObsDelaySteps = 4;
constexpr int kMaxActDelaySteps = 3;

class WheeledRLController : public controller_interface::ControllerInterface
{
public:
  controller_interface::CallbackReturn on_init() override;
  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;
  controller_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State & previous) override;
  controller_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State & previous) override;
  controller_interface::return_type update(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  // [0..3] leg position targets (rad), [4..5] wheel velocity targets (rad/s)
  std::array<double, kActionDim> joint_commands_{};
  std::array<double, 4> position_targets_{};      // PREPARE interpolation state
  std::array<float, robot_state::kObsDim> obs_{};
  std::array<float, robot_state::kActionDim> action_{};  // raw (undelayed) last action
  robot_state::RobotStateBuffer state_buffer_;
  fsm::StateMachine fsm_;
  onnxruntime::PolicyRuntime policy_;
  int tick_ = 0;
  int policy_step_ = 0;
  double height_cmd_ = 0.22;
  int obs_delay_steps_ = 4;  // 80 ms (ros2_control param obs_delay_steps overrides)
  int act_delay_steps_ = 3;  // 60 ms
  std::array<std::array<float, robot_state::kObsDim>, kMaxObsDelaySteps + 1> obs_ring_{};
  std::array<std::array<float, robot_state::kActionDim>, kMaxActDelaySteps + 1> act_ring_{};

  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr height_sub_;

  void run_policy_if_due();
  void write_commands();
};

}  // namespace wheeled_rl

#endif  // WHEELED_RL_CONTROLLER__RL_CONTROLLER_HPP_
