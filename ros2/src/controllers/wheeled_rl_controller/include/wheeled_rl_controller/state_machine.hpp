// FSM module — state transitions incl. the 1 rad/s PREPARE interpolation.
// Extracted from the controller body following the the fsm/ module layout.
#ifndef WHEELED_RL_CONTROLLER__STATE_MACHINE_HPP_
#define WHEELED_RL_CONTROLLER__STATE_MACHINE_HPP_

#include <algorithm>
#include <array>
#include <cmath>

namespace wheeled_rl::fsm
{

enum class Fsm : int8_t { INIT = 0, IDLE = 1, PREPARE = 2, RL = 3 };

constexpr size_t kNumLegs = 4;
constexpr double kPrepareMaxVel = 1.0;  // rad/s (CONTRACT.md section 5)
constexpr double kPrepareTol = 0.02;    // rad
constexpr double kDefaultLegPose = 0.0;

class StateMachine
{
public:
  Fsm state() const { return state_; }
  void request_rl() { if (state_ == Fsm::IDLE) state_ = Fsm::PREPARE; }

  /// INIT/IDLE hold damping and latch the current pose as PREPARE start.
  template <typename Array>
  void hold(Array & position_targets, const Array & current_leg_pos)
  {
    for (size_t k = 0; k < kNumLegs; ++k) position_targets[k] = current_leg_pos[k];
  }

  /// PREPARE: interpolate toward the default pose at <=1 rad/s, enter RL on arrival.
  template <typename Array>
  bool step_prepare(Array & position_targets, double dt)
  {
    bool reached = true;
    for (size_t k = 0; k < kNumLegs; ++k) {
      const double step = std::clamp(kDefaultLegPose - position_targets[k],
                                     -kPrepareMaxVel * dt, kPrepareMaxVel * dt);
      position_targets[k] += step;
      if (std::abs(kDefaultLegPose - position_targets[k]) > kPrepareTol) reached = false;
    }
    if (reached) state_ = Fsm::RL;
    return state_ == Fsm::RL;
  }

private:
  Fsm state_ = Fsm::INIT;
};

}  // namespace wheeled_rl::fsm

#endif  // WHEELED_RL_CONTROLLER__STATE_MACHINE_HPP_
