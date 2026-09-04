cmake_minimum_required(VERSION 3.16)
project(wheeled_mujoco_system LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)

# Optional MuJoCo: -DMUJOCO_ROOT=/path/to/mujoco (headers + libmujoco.so)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(MUJOCO QUIET mujoco)
endif()
if(MUJOCO_FOUND)
  add_compile_definitions(HAVE_MUJOCO)
  message(STATUS "MuJoCo found — hardware plugin enabled")
else()
  message(STATUS "MuJoCo NOT found — building inert skeleton")
endif()

add_library(wheeled_mujoco_system SHARED src/mujoco_system.cpp)
target_include_directories(wheeled_mujoco_system PRIVATE include)
ament_target_dependencies(wheeled_mujoco_system
  hardware_interface pluginlib rclcpp rclcpp_lifecycle)
if(MUJOCO_FOUND)
  target_include_directories(wheeled_mujoco_system SYSTEM PRIVATE ${MUJOCO_INCLUDE_DIRS})
  target_link_libraries(wheeled_mujoco_system ${MUJOCO_LIBRARIES})
endif()

pluginlib_export_plugin_description_file(hardware_interface wheeled_mujoco_system.xml)

install(TARGETS wheeled_mujoco_system DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

ament_export_dependencies(hardware_interface pluginlib)
ament_package()
