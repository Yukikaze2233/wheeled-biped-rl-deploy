cmake_minimum_required(VERSION 3.16)
project(wheeled_rl_controller LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(controller_interface REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# Optional ONNX Runtime: provide via -DONNXRUNTIME_ROOT=/path/to/libonnxruntime
find_package(ONNXRuntime QUIET)
if(ONNXRuntime_FOUND)
  add_compile_definitions(HAVE_ONNXRUNTIME)
  message(STATUS "onnxruntime found — inference enabled")
else()
  message(STATUS "onnxruntime NOT found — building control-path-only skeleton")
endif()

add_library(wheeled_rl_controller SHARED
  src/rl_controller.cpp
  src/fsm/state_machine.cpp
  src/robot_state/robot_state.cpp
  src/onnxruntime/policy_runtime.cpp)
target_include_directories(wheeled_rl_controller PRIVATE include)
ament_target_dependencies(wheeled_rl_controller
  controller_interface hardware_interface pluginlib rclcpp rclcpp_lifecycle
  geometry_msgs std_msgs)
if(ONNXRuntime_FOUND)
  target_link_libraries(wheeled_rl_controller ${ONNXRuntime_LIBRARIES})
  target_include_directories(wheeled_rl_controller PRIVATE ${ONNXRuntime_INCLUDE_DIRS})
endif()

pluginlib_export_plugin_description_file(controller_interface wheeled_rl_controller.xml)

install(TARGETS wheeled_rl_controller DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

ament_export_dependencies(controller_interface pluginlib)
ament_package()
