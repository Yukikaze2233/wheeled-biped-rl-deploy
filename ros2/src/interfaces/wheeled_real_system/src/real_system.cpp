// Implementation of WheeledRealSystem — see real_system.hpp.
// Serial protocol: TX/RX frame layouts in the header. CRC16 (CCITT) over the
// payload; byte order little-endian per the firmware contract.
#include "wheeled_real_system/real_system.hpp"

#include <cstring>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/rclcpp.hpp"

namespace wheeled_real_system
{

static uint16_t crc16_ccitt(const uint8_t * data, size_t len)
{
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < len; ++i) {
    crc ^= static_cast<uint16_t>(data[i]) << 8;
    for (int b = 0; b < 8; ++b) {
      crc = (crc & 0x8000) ? static_cast<uint16_t>((crc << 1) ^ 0x1021) : static_cast<uint16_t>(crc << 1);
    }
  }
  return crc;
}

hardware_interface::CallbackReturn WheeledRealSystem::on_init(
  const hardware_interface::HardwareInfo & info)
{
  if (hardware_interface::SystemInterface::on_init(info) != hardware_interface::CallbackReturn::SUCCESS) {
    return hardware_interface::CallbackReturn::ERROR;
  }
  port_ = info_.hardware_parameters.count("port") ? info_.hardware_parameters.at("port") : port_;
  baudrate_ = info_.hardware_parameters.count("baudrate")
                ? std::stoi(info_.hardware_parameters.at("baudrate")) : baudrate_;
  for (const auto & joint : info_.joints) {
    if (joint.name == "imu") continue;  // synthetic imu joint carries no serial identity
    joint_names_.push_back(joint.name);
  }
  const size_t n = joint_names_.size();
  joint_position_.assign(n, 0.0);
  joint_velocity_.assign(n, 0.0);
  position_commands_.assign(n, 0.0);
  velocity_commands_.assign(n, 0.0);
  return hardware_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> WheeledRealSystem::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> states;
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    states.emplace_back(joint_names_[i], hardware_interface::HW_IF_POSITION, &joint_position_[i]);
    states.emplace_back(joint_names_[i], hardware_interface::HW_IF_VELOCITY, &joint_velocity_[i]);
  }
  states.emplace_back("imu", "gyro.x", &gyro_[0]);
  states.emplace_back("imu", "gyro.y", &gyro_[1]);
  states.emplace_back("imu", "gyro.z", &gyro_[2]);
  states.emplace_back("imu", "gravity.x", &gravity_[0]);
  states.emplace_back("imu", "gravity.y", &gravity_[1]);
  states.emplace_back("imu", "gravity.z", &gravity_[2]);
  return states;
}

std::vector<hardware_interface::CommandInterface> WheeledRealSystem::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> commands;
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    const auto & type = info_.joints[i].command_interfaces[0].name;
    if (type == hardware_interface::HW_IF_POSITION) {
      commands.emplace_back(joint_names_[i], HW_IF_POSITION, &position_commands_[i]);
    } else {
      commands.emplace_back(joint_names_[i], HW_IF_VELOCITY, &velocity_commands_[i]);
    }
  }
  return commands;
}

bool WheeledRealSystem::open_serial()
{
  fd_ = ::open(port_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
  if (fd_ < 0) return false;
  termios tty{};
  if (tcgetattr(fd_, &tty) != 0) return false;
  cfmakeraw(&tty);
  tty.c_cc[VMIN] = 0;
  tty.c_cc[VTIME] = 0;
  // baud: B2000000 when available; map your firmware rate here
  cfsetispeed(&tty, B2000000);
  cfsetospeed(&tty, B2000000);
  return tcsetattr(fd_, TCSANOW, &tty) == 0;
}

void WheeledRealSystem::close_serial()
{
  if (fd_ >= 0) ::close(fd_);
  fd_ = -1;
}

hardware_interface::CallbackReturn WheeledRealSystem::on_configure(const rclcpp_lifecycle::State &)
{
  if (!open_serial()) {
    RCLCPP_ERROR(rclcpp::get_logger("WheeledRealSystem"), "serial open failed: %s", port_.c_str());
    return hardware_interface::CallbackReturn::ERROR;
  }
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn WheeledRealSystem::on_deactivate(const rclcpp_lifecycle::State &)
{
  close_serial();
  return hardware_interface::CallbackReturn::SUCCESS;
}

bool WheeledRealSystem::exchange_frames()
{
  // TX: build the command frame per the header layout, write; RX: read the
  // state frame, verify CRC, unpack into joint_position_/velocity_/gyro_/gravity_.
  // TODO(yukikaze): finalize byte layout + CRC check against the firmware.
  return true;
}

hardware_interface::return_type WheeledRealSystem::read(const rclcpp::Time &, const rclcpp::Duration &)
{
  exchange_frames();
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type WheeledRealSystem::write(const rclcpp::Time &, const rclcpp::Duration &)
{
  exchange_frames();
  return hardware_interface::return_type::OK;
}

}  // namespace wheeled_real_system

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(wheeled_real_system::WheeledRealSystem,
                       hardware_interface::SystemInterface)
