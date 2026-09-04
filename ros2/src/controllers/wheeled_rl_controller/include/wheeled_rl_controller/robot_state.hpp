// Robot-state module — caches state-interface values and assembles the 35D
// policy observation per CONTRACT.md section 2 (the robot_state module role).
#ifndef WHEELED_RL_CONTROLLER__ROBOT_STATE_HPP_
#define WHEELED_RL_CONTROLLER__ROBOT_STATE_HPP_

#include <array>
#include <cstddef>

namespace wheeled_rl::robot_state
{

constexpr size_t kStateCount = 16;
// layout: [0..3] leg pos, [4..7] leg vel, [8..9] wheel vel, [10..12] gyro, [13..15] proj gravity
constexpr double kAngVelScale = 0.5;
constexpr double kHeightCmdScale = 5.0;
constexpr double kJointVelScale = 0.1;
constexpr size_t kObsDim = 35;
constexpr size_t kActionDim = 6;

class RobotStateBuffer
{
public:
  void update(const double * state_values, size_t count)
  {
    for (size_t k = 0; k < kStateCount && k < count; ++k) values_[k] = state_values[k];
  }

  /// Fill the 35D observation; cmd/height and previous action come from the caller.
  template <typename ObsArray>
  void assemble_observation(ObsArray & obs, double vx_cmd, double wz_cmd, double height_cmd,
                            const float * previous_action) const
  {
    for (int k = 0; k < 4; ++k) obs[10 + k] = static_cast<float>(values_[k]);
    for (int k = 0; k < 4; ++k) obs[16 + k] = static_cast<float>(values_[4 + k] * kJointVelScale);
    obs[20] = static_cast<float>(values_[8] * kJointVelScale);
    obs[21] = static_cast<float>(values_[9] * kJointVelScale);
    for (int k = 0; k < 3; ++k) obs[4 + k] = static_cast<float>(values_[10 + k] * kAngVelScale);
    for (int k = 0; k < 3; ++k) obs[7 + k] = static_cast<float>(values_[13 + k]);
    obs[0] = static_cast<float>(vx_cmd);
    obs[1] = 0.0f;  // vy fixed 0
    obs[2] = static_cast<float>(wz_cmd_);
    obs[3] = static_cast<float>(height_cmd * kHeightCmdScale);
    for (int k = 0; k < 6; ++k) obs[22 + k] = previous_action[k];
    for (int k = 29; k < 35; ++k) obs[k] = 0.0f;
    obs[28] = 1.0f;  // normal mode
  }

  void set_command(double vx, double wz) { vx_cmd_ = vx; wz_cmd_ = wz; }
  double vx() const { return vx_cmd_; }
  double wz() const { return wz_cmd_; }

private:
  std::array<double, kStateCount> values_{};
  double vx_cmd_ = 0.0, wz_cmd_ = 0.0;
};

}  // namespace wheeled_rl::robot_state

#endif  // WHEELED_RL_CONTROLLER__ROBOT_STATE_HPP_
