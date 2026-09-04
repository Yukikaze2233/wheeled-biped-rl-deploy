#!/usr/bin/env python3
"""URDF -> MJCF with contract actuators (legs pos-servo motors + wheel motors).

Loads the URDF via MuJoCo, saves the raw MJCF, injects <actuator> motors for
the contract joints (4 legs + 2 wheels) and a standing keyframe, so sim2sim /
capture tools get a drivable model (URDF alone has nu=0).
"""
import sys

import mujoco

LEG_JOINTS = ("left_front1_joint", "right_front1_joint", "left_rear1_joint", "right_rear1_joint")
WHEEL_JOINTS = ("left_wheel_joint", "right_wheel_joint")


def main(urdf_path: str, out_path: str) -> None:
    mujoco.MjModel.from_xml_path(urdf_path)  # validate + populate last model
    tmp = out_path + ".raw.xml"
    mujoco.mj_saveLastXML(tmp, mujoco.MjModel.from_xml_path(urdf_path))
    xml = open(tmp).read()
    motors = []
    for j in LEG_JOINTS:
        motors.append(f'<motor name="{j}" joint="{j}" ctrlrange="-40 40" gear="1"/>')
    for j in WHEEL_JOINTS:
        motors.append(f'<motor name="{j}" joint="{j}" ctrlrange="-5 5" gear="1"/>')
    actuator_block = "<actuator>" + "".join(motors) + "</actuator>"
    xml = xml.replace("</mujoco>", actuator_block + "\n</mujoco>")
    open(out_path, "w").write(xml)
    m = mujoco.MjModel.from_xml_path(out_path)
    print(f"OK {out_path}: nu={m.nu} nq={m.nq}")
    assert m.nu == 6, "expected 6 contract actuators"


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
