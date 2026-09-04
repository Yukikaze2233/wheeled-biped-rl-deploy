"""Bringup: ros2_control + RL controller, sim (MuJoCo) or real backend.

    ros2 launch wheeled_bringup bringup.launch.py backend:=sim
    ros2 launch wheeled_bringup bringup.launch.py backend:=real

Same controller/contract either way — only the hardware plugin changes
(S CUT-style). "real" expects a RealBridge SystemInterface package (serial
protocol) to be provided at integration time.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    backend = LaunchConfiguration("backend")
    pkg = get_package_share_directory("wheeled_bringup")
    deploy_pkg = get_package_share_directory("wheeled_rl_controller")

    declare_backend = DeclareLaunchArgument("backend", default_value="sim",
                                            description="sim (MuJoCo) or real (serial)")
    declare_policy = DeclareLaunchArgument("policy_path",
                                           default_value=os.path.join(deploy_pkg, "policy", "policy.onnx"))

    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            os.path.join(pkg, "config", "wheeled_biped.ros2_control.yaml"),
            {"update_rate": 500},
        ],
        output="both",
    )
    # the yaml selects the hardware plugin via the `backend` xacro arg
    robot_description = {
        "robot_description": Command([
            "xacro ", os.path.join(pkg, "config", "wheeled_biped.xacro"), " backend:=", backend,
            " policy_path:=", LaunchConfiguration("policy_path"),
        ])
    }
    joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
    )
    rl_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["wheeled_rl_controller", "--controller-manager", "/controller_manager"],
    )

    return LaunchDescription([
        declare_backend,
        declare_policy,
        control_node,
        joint_state_broadcaster,
        rl_controller,
    ])
