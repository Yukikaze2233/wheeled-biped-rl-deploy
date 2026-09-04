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
    CTRL_DT, STEPS_PER_POLICY, actuator_ids_of, build_obs, spring_binding,
    step_control,
)


def run(mj, data, sess, i_name, vx, wz, seconds, spring_ff=0.0):
    leg_act, wheel_act = actuator_ids_of(mj)
    spring_act = spring_binding(mj)
    last_action = np.zeros(6, np.float32)
    action = np.zeros(6, np.float32)
    cmd = np.array([vx, 0.0, wz], np.float32)
    xs = []  # x-position history: oscillation metric
    n = int(seconds / CTRL_DT)
    for tick in range(n):
        if tick % STEPS_PER_POLICY == 0:
            obs = build_obs(mj, data, cmd, 0.22, last_action)
            action = sess.run(None, {i_name: obs[None]})[0][0]
            last_action = action.copy()
        step_control(mj, data, action, leg_act, wheel_act,
                     spring_act=spring_act, spring_bias=spring_ff)
        xs.append(data.qpos[0])
    run.last_xs = xs  # exposed for oscillation metrics
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
    xs = getattr(run, "last_xs", [])
    osc = float(np.std(xs)) if xs else float("nan")
    print(f"[{name:8s}] dx={x - x0:+6.2f}m dy={y - y0:+6.2f}m dist={dist:5.2f}m osc_x={osc:.3f}m "
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
