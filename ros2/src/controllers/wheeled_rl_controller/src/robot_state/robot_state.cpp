// Robot-state module translation unit: observation assembly instantiation
// (the robot_state module role).
#include "wheeled_rl_controller/robot_state.hpp"

namespace wheeled_rl::robot_state
{
template void RobotStateBuffer::assemble_observation<std::array<float, 35>>(
  std::array<float, 35> &, double, double, double, const float *) const;
}  // namespace wheeled_rl::robot_state
