// Wheeled-biped RL controller — interface wiring + 500 Hz update loop.
// Logic lives in the modules (module layout): fsm/, robot_state/, onnxruntime/.
#include "wheeled_rl_controller/rl_controller.hpp"

#include <algorithm>
#include <string>
#include <vector>

#include "controller_interface/controller_interface.hpp"

namespace wheeled_rl
{

controller_interface::CallbackReturn WheeledRLController::on_init()
{
  joint_commands_.fill(0.0);
  obs_.fill(0.0f);
  obs_[28] = 1.0f;  // normal-mode flag stays 1 (flat contract)
  auto node = get_node();
  if (!node->has_parameter("model_path")) {
    node->declare_parameter<std::string>("model_path", "");
  }
  if (!node->has_parameter("obs_delay_steps")) {
    node->declare_parameter<int>("obs_delay_steps", 4);
  }
  if (!node->has_parameter("act_delay_steps")) {
    node->declare_parameter<int>("act_delay_steps", 3);
  }
  obs_delay_steps_ = std::clamp(node->get_parameter("obs_delay_steps").as_int(), 0, kMaxObsDelaySteps);
  act_delay_steps_ = std::clamp(node->get_parameter("act_delay_steps").as_int(), 0, kMaxActDelaySteps);
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration
WheeledRLController::command_interface_configuration() const
{
  // leg position targets x4 + wheel velocity targets x2 ("hardware_pd_vel");
  // the hardware plugin closes the PD/servo loops at its own rate.
  controller_interface::InterfaceConfiguration conf;
  conf.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  // CONTRACT.md section 3: action dims are rear-first
  // [left_rear, right_rear, left_front, right_front, left_wheel, right_wheel]
  // (authoritative: official training legs_act joint order).
  const std::array<std::string, kActionDim> names = {
    "left_rear1_joint/position",  "right_rear1_joint/position",
    "left_front1_joint/position", "right_front1_joint/position",
    "left_wheel_joint/velocity",  "right_wheel_joint/velocity",
  };
  for (const auto & n : names) conf.names.push_back(n);
  return conf;
}

controller_interface::InterfaceConfiguration
WheeledRLController::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration conf;
  conf.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  // same rear-first contract order as the command interfaces
  const std::array<std::string, 16> names = {
    "left_rear1_joint/position",  "right_rear1_joint/position",
    "left_front1_joint/position", "right_front1_joint/position",
    "left_rear1_joint/velocity",  "right_rear1_joint/velocity",
    "left_front1_joint/velocity", "right_front1_joint/velocity",
    "left_wheel_joint/velocity",  "right_wheel_joint/velocity",
    "imu/gyro.x", "imu/gyro.y", "imu/gyro.z",
    "imu/gravity.x", "imu/gravity.y", "imu/gravity.z",
  };
  for (const auto & n : names) conf.names.push_back(n);
  return conf;
}

controller_interface::CallbackReturn WheeledRLController::on_configure(
  const rclcpp_lifecycle::State &)
{
  const std::string model_path = get_node()->get_parameter("model_path").as_string();
  policy_.load(model_path);
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn WheeledRLController::on_activate(
  const rclcpp_lifecycle::State &)
{
  auto node = get_node();
  cmd_sub_ = node->create_subscription<geometry_msgs::msg::Twist>(
    "~/motion_command", 10,
    [this](const geometry_msgs::msg::Twist::SharedPtr msg) {
      state_buffer_.set_command(msg->linear.x, msg->angular.z);
      fsm_.request_rl();
    });
  height_sub_ = node->create_subscription<std_msgs::msg::Float64>(
    "~/height_command", 10,
    [this](const std_msgs::msg::Float64::SharedPtr msg) { height_cmd_ = msg->data; });
  return controller_interface::CallbackReturn::SUCCESS;
}

void WheeledRLController::run_policy_if_due()
{
  if (tick_ % kTicksPerPolicy != 0) return;
  // observation ring: policy sees the obs from obs_delay_steps_ policy
  // steps ago (one policy step = 20 ms). Training always runs with
  // obs 20-80 ms / act 20-60 ms delays; 80/60 ms measured best for the
  // official 13k flat_and_rotation policy (see docs/timing.md).
  state_buffer_.assemble_observation(obs_ring_[policy_step_ % (obs_delay_steps_ + 1)],
                                     state_buffer_.vx(), state_buffer_.wz(),
                                     height_cmd_, action_.data());
  const int obs_delay_slots = obs_delay_steps_ + 1;
  const auto & delayed_obs =
    obs_ring_[((policy_step_ - obs_delay_steps_) % obs_delay_slots + obs_delay_slots) %
              obs_delay_slots];
  std::array<float, robot_state::kActionDim> raw_action{};
  policy_.infer(delayed_obs, raw_action);
  action_ = raw_action;
  act_ring_[policy_step_ % (act_delay_steps_ + 1)] = raw_action;
  ++policy_step_;
}

void WheeledRLController::write_commands()
{
  // applied action comes from the action-delay ring (act_delay_steps_ * 20 ms old)
  const int act_delay_slots = act_delay_steps_ + 1;
  const auto & applied = act_ring_[((policy_step_ - act_delay_steps_) % act_delay_slots +
                                    act_delay_slots) % act_delay_slots];
  for (int k = 0; k < 4; ++k) {
    joint_commands_[k] = kLegActionScale * applied[k];
  }
  joint_commands_[4] = std::clamp(kWheelActionScale * applied[4], -kMaxWheelVel, kMaxWheelVel);
  joint_commands_[5] = std::clamp(kWheelActionScale * applied[5], -kMaxWheelVel, kMaxWheelVel);
}

controller_interface::return_type WheeledRLController::update(
  const rclcpp::Time &, const rclcpp::Duration & period)
{
  // state interfaces -> module cache (leg pos/vel, wheel vel, imu)
  std::vector<double> si_values(state_interfaces_.size());
  for (size_t k = 0; k < state_interfaces_.size(); ++k) {
    si_values[k] = state_interfaces_[k].get_value();
  }
  state_buffer_.update(si_values.data(), si_values.size());

  ++tick_;
  const std::array<double, 4> current_leg_pos{
    si_values[0], si_values[1], si_values[2], si_values[3]};
  switch (fsm_.state()) {
    case fsm::Fsm::INIT:
    case fsm::Fsm::IDLE:
      joint_commands_.fill(0.0);  // hardware holds damping
      fsm_.hold(position_targets_, current_leg_pos);
      break;
    case fsm::Fsm::PREPARE:
      fsm_.step_prepare(position_targets_, period.seconds());
      joint_commands_[0] = position_targets_[0];
      joint_commands_[1] = position_targets_[1];
      joint_commands_[2] = position_targets_[2];
      joint_commands_[3] = position_targets_[3];
      joint_commands_[4] = joint_commands_[5] = 0.0;
      break;
    case fsm::Fsm::RL:
      run_policy_if_due();
      write_commands();
      break;
  }
  for (size_t k = 0; k < kActionDim; ++k) {
    command_interfaces_[k].set_value(joint_commands_[k]);
  }
  return controller_interface::return_type::OK;
}

}  // namespace wheeled_rl

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(wheeled_rl::WheeledRLController,
                       controller_interface::ControllerInterface)
