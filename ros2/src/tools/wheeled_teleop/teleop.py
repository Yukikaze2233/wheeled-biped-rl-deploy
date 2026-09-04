#!/usr/bin/env python3
"""Keyboard teleop: publish motion/height commands for the RL controller.

    ros2 run wheeled_teleop teleop.py

Keys: w/s = vx +/-, a/d = yaw +/-, r/f = height +/-, space = zero command,
q = quit. Publishes Twist on /wheelbipe/motion_command and Float64 on
/wheelbipe/height_command at 50 Hz.
"""
import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64


class Teleop(Node):
    def __init__(self):
        super().__init__("wheeled_teleop")
        self.declare_parameter("namespace", "wheelbipe")
        ns = self.get_parameter("namespace").value
        self.cmd_pub = self.create_publisher(Twist, f"/{ns}/motion_command", 10)
        self.height_pub = self.create_publisher(Float64, f"/{ns}/height_command", 10)
        self.vx = self.wz = 0.0
        self.height = 0.22
        self.timer = self.create_timer(0.02, self.publish)  # 50 Hz

    def publish(self):
        msg = Twist()
        msg.linear.x, msg.angular.z = self.vx, self.wz
        self.cmd_pub.publish(msg)
        h = Float64()
        h.data = self.height
        self.height_pub.publish(h)

    def handle(self, key: str) -> bool:
        step_v, step_w, step_h = 0.25, 0.5, 0.02
        if key == "w":
            self.vx = min(self.vx + step_v, 2.5)
        elif key == "s":
            self.vx = max(self.vx - step_v, -2.5)
        elif key == "a":
            self.wz = min(self.wz + step_w, 3.0)
        elif key == "d":
            self.wz = max(self.wz - step_w, -3.0)
        elif key == "r":
            self.height = min(self.height + step_h, 0.40)
        elif key == "f":
            self.height = max(self.height - step_h, 0.20)
        elif key == " ":
            self.vx = self.wz = 0.0
        elif key == "q":
            return False
        self.get_logger().info(f"vx={self.vx:+.2f} wz={self.wz:+.2f} h={self.height:.2f}")
        return True


def main():
    rclpy.init()
    node = Teleop()
    print("w/s vx, a/d yaw, r/f height, space zero, q quit")
    old_attrs = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    try:
        while rclpy.ok():
            key = sys.stdin.read(1)
            if key and not node.handle(key):
                break
            rclpy.spin_once(node, timeout_sec=0.01)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_attrs)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
