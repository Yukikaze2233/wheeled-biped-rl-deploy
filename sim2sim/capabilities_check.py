#!/usr/bin/env python3
"""Probe what a policy can actually do: stand / drive / spin / self-right.

Each scenario runs the 500Hz/50Hz loop and reports objective metrics
(displacement, yaw rate, recovery height) + end screenshot.
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from mujoco_sim2sim import actuator_ids_of  # noqa: E402
from mujoco_sim2sim import (  # noqa: E402
    CTRL_DT, LEG_JOINTS, LEG_KD, LEG_KP, STEPS_PER_POLICY, WHEEL_JOINTS,
    WHEEL_KV, _root_body_id, build_obs,
)


def run(mj, data, sess, i_name, vx, wz, seconds, spring_ff=0.0):
    leg_act, wheel_act = actuator_ids_of(mj)
    spring_act = [mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{j}_ctrl")
                  for j in ("left_spring2_joint", "right_spring2_joint")]
    spring_act = [a for a in spring_act if a >= 0]
    last_action = np.zeros(6, np.float32)
    action = np.zeros(6, np.float32)
    cmd = np.array([vx, 0.0, wz], np.float32)
    yaw0 = None
    n = int(seconds / CTRL_DT)
    for tick in range(n):
        if tick % STEPS_PER_POLICY == 0:
            obs = build_obs(mj, data, cmd, 0.22, last_action)
            action = sess.run(None, {i_name: obs[None]})[0][0]
            last_action = action.copy()
        leg_t = action[:4] * 0.5
        wheel_v = np.clip(action[4:] * 10.0, -30, 30)
        for k, jname in enumerate(LEG_JOINTS):
            j = mj.joint(jname).id
            q, dq = data.qpos[mj.jnt_qposadr[j]], data.qvel[mj.jnt_dofadr[j]]
            data.ctrl[leg_act[k]] = LEG_KP * (leg_t[k] - q) - LEG_KD * dq
        for k, jname in enumerate(WHEEL_JOINTS):
            dq = data.qvel[mj.jnt_dofadr[mj.joint(jname).id]]
            data.ctrl[wheel_act[k]] = WHEEL_KV * (wheel_v[k] - dq)
        if spring_act:
            for a in spring_act:
                data.ctrl[a] = spring_ff
        mujoco.mj_step(mj, data)
    return data


def quat_yaw(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def scenario(mj_path, policy, name, vx, wz, tip_over=False, spring_ff=0.0):
    import onnxruntime as ort
    mj = mujoco.MjModel.from_xml_path(mj_path)
    data = mujoco.MjData(mj)
    data.qpos[2] = 0.30
    if tip_over:  # roll 90deg -> lying on its side
        data.qpos[3:7] = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0]
        data.qpos[2] = 0.15
    mujoco.mj_forward(mj, data)
    x0, y0, z0 = data.qpos[:3].copy()
    yaw0 = quat_yaw(data.qpos[3:7])

    sess = ort.InferenceSession(policy, providers=["CPUExecutionProvider"])
    run(mj, data, sess, sess.get_inputs()[0].name, vx, wz, 4.0, spring_ff=spring_ff)

    x, y, z = data.qpos[:3]
    yaw = quat_yaw(data.qpos[3:7])
    dyaw = (yaw - yaw0 + math.pi) % (2 * math.pi) - math.pi
    dist = math.hypot(x - x0, y - y0)
    print(f"[{name:8s}] dx={x - x0:+6.2f}m dy={y - y0:+6.2f}m dist={dist:5.2f}m "
          f"dyaw={math.degrees(dyaw):+7.1f}deg ({dyaw / 4:+.2f}rad/s) "
          f"final_z={z:.3f} {'UPRIGHT' if z > 0.15 else 'FALLEN'}")
    return data


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--policy", required=True)
    args = p.parse_args()
    print(f"policy: {os.path.basename(args.policy)}")
    for ff in (0.0, 500.0, -500.0):
        print(f"-- spring_ff={ff:+.0f}N")
        scenario(args.model, args.policy, "stand", 0.0, 0.0, spring_ff=ff)
        scenario(args.model, args.policy, "drive", 1.5, 0.0, spring_ff=ff)
        scenario(args.model, args.policy, "spin", 0.0, 2 * math.pi, spring_ff=ff)
