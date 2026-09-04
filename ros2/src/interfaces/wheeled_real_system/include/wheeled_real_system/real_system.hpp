// RealBridge SystemInterface (skeleton): single-serial-port link to the
// robot's low-level controller, single-serial-port link to the low-level controller.
//
// Protocol (frame layout mirrors a typical real_msg.hpp; align bytes with the
// firmware before first use):
//   TX 500 Hz: header(2) | fsm(1) | leg pos targets 4xf32 | wheel vel 2xf32 |
//              height cmd f32 | crc16(2)
//   RX 500 Hz: header(2) | leg pos 4xf32 | leg vel 4xf32 | wheel vel 2xf32 |
//              gyro 3xf32 | proj-gravity 3xf32 | fsm feedback(1) | crc16(2)
//
// Serial I/O (termios) is complete in outline; CRC and byte-order alignment
// with the firmware are the remaining integration steps.
#ifndef WHEELED_REAL_SYSTEM__REAL_SYSTEM_HPP_
#define WHEELED_REAL_SYSTEM__REAL_SYSTEM_HPP_

#include <array>
#include <string>
#include <vector>

#include "hardware_interface/system_interface.hpp"
#include "rclcpp/macros.hpp"
#include "rclcpp_lifecycle/state.hpp"

namespace wheeled_real_system
{

constexpr int kFrameHeader0 = 0x42;
constexpr int kFrameHeader1 = 0x24;

class WheeledRealSystem : public hardware_interface::SystemInterface
{
public:
  RCLCPP_SHARED_PTR_DEFINITIONS(WheeledRealSystem)

  hardware_interface::CallbackReturn on_init(const hardware_interface::HardwareInfo & info) override;
  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;
  hardware_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State &) override;
  hardware_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State &) override;
  hardware_interface::return_type read(const rclcpp::Time &, const rclcpp::Duration &) override;
  hardware_interface::return_type write(const rclcpp::Time &, const rclcpp::Duration &) override;

private:
  std::string port_{"/dev/ttyUSB0"};
  int baudrate_{2000000};
  int fd_ = -1;

  std::vector<std::string> joint_names_;
  std::vector<double> joint_position_{}, joint_velocity_{};
  std::vector<double> position_commands_{}, velocity_commands_{};
  std::array<double, 3> gyro_{}, gravity_{};

  bool open_serial();
  void close_serial();
  bool exchange_frames();  // TX command frame + RX state frame
};

}  // namespace wheeled_real_system

#endif  // WHEELED_REAL_SYSTEM__REAL_SYSTEM_HPP_
