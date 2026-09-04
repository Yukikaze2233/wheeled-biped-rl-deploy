cmake_minimum_required(VERSION 3.16)
project(wheeled_bringup)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch config DESTINATION share/wheeled_bringup)

ament_package()
