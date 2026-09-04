// FSM module translation unit: explicit template instantiations for the
// array types the controller uses (logic is header-inline, fsm/ module layout).
#include "wheeled_rl_controller/state_machine.hpp"

namespace wheeled_rl::fsm
{
template void StateMachine::hold<std::array<double, 4>>(
  std::array<double, 4> &, const std::array<double, 4> &);
template bool StateMachine::step_prepare<std::array<double, 4>>(
  std::array<double, 4> &, double);
}  // namespace wheeled_rl::fsm
