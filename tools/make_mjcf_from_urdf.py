#!/usr/bin/env python3
"""URDF -> MJCF with contract actuators (legs pos-servo motors + wheel motors).

Loads the URDF via MuJoCo, saves the raw MJCF, injects <actuator> motors for
the contract joints (4 legs + 2 wheels), so sim2sim / capture tools get a
drivable model (URDF alone has nu=0).

Joint lists are CLI-driven (defaults = official V14 contract, REAR-FIRST leg
order per the training legs_act joint order):
  python3 tools/make_mjcf_from_urdf.py robot.urdf robot.xml \\
      --legs left_rear1_joint right_rear1_joint left_front1_joint right_front1_joint \\
      --wheels left_wheel_joint right_wheel_joint
For our serial-leg robot (V3.x): --legs L_joint1 L_joint2 R_joint1 R_joint2
--wheels L_joint3 R_joint3 (R_joint2 maps to the firmware name R_jonit2).
"""
import argparse
import sys
import xml.etree.ElementTree as ET

import mujoco

# official V14 contract order (rear-first), per the pretrained env.yaml
LEG_JOINTS = ("left_rear1_joint", "right_rear1_joint", "left_front1_joint", "right_front1_joint")
WHEEL_JOINTS = ("left_wheel_joint", "right_wheel_joint")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("urdf_path")
    p.add_argument("out_path")
    p.add_argument("--legs", nargs=4, default=LEG_JOINTS)
    p.add_argument("--wheels", nargs=2, default=WHEEL_JOINTS)
    args = p.parse_args()

    m0 = mujoco.MjModel.from_xml_path(args.urdf_path)  # validate
    tmp = args.out_path + ".raw.xml"
    mujoco.mj_saveLastXML(tmp, m0)
    xml = open(tmp).read()
    # MuJoCo's URDF importer welds a jointless ROOT link to the world (its
    # mass is dropped and its geoms become static world geoms). Wrap the
    # worldbody into a free-jointed root body and restore the root inertial
    # from the URDF so the robot is a proper free-flying articulation.
    urdf_root = ET.parse(args.urdf_path).getroot()
    root_link = urdf_root.find("link")  # first link = URDF root
    root_name = root_link.attrib["name"]
    if f'<body name="{root_name}"' not in xml:
        inertial = root_link.find("inertial")
        mass = inertial.find("mass").attrib["value"]
        inn = inertial.find("inertia").attrib
        origin = inertial.find("origin")
        pos = origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0"
        # MJCF fullinertia order: ixx iyy izz ixy ixz iyz (diagonals first)
        fullinertia = (f'{inn["ixx"]} {inn["iyy"]} {inn["izz"]} '
                       f'{inn["ixy"]} {inn["ixz"]} {inn["iyz"]}')
        wrap_open = (f'<body name="{root_name}">\n    <freejoint/>\n'
                     f'    <inertial pos="{pos}" mass="{mass}" '
                     f'fullinertia="{fullinertia}"/>\n')
        xml = xml.replace("<worldbody>", "<worldbody>\n" + wrap_open, 1)
        xml = xml.replace("</worldbody>", "</body>\n</worldbody>", 1)
    motors = []
    for j in args.legs:
        motors.append(f'<motor name="{j}" joint="{j}" ctrlrange="-40 40" gear="1"/>')
    for j in args.wheels:
        motors.append(f'<motor name="{j}" joint="{j}" ctrlrange="-5 5" gear="1"/>')
    actuator_block = "<actuator>" + "".join(motors) + "</actuator>"
    xml = xml.replace("</mujoco>", actuator_block + "\n</mujoco>")
    open(args.out_path, "w").write(xml)
    m = mujoco.MjModel.from_xml_path(args.out_path)
    print(f"OK {args.out_path}: nu={m.nu} nq={m.nq}")
    assert m.nu == 6, "expected 6 contract actuators"


if __name__ == "__main__":
    main()
