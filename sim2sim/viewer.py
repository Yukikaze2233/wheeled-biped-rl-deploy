#!/usr/bin/env python3
"""Interactive MuJoCo viewer: watch the policy balance the robot live.

    python3 sim2sim/viewer.py --model <scene.xml> --policy <policy.onnx>

Keys (sim running):  W/S = forward/backward command   A/D = spin left/right
                     R = reset    SPACE = zero command   ESC/Q = quit
The camera follows the robot (drag with mouse to orbit freely).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from mujoco import viewer as mj_viewer  # noqa: E402

from mujoco_sim2sim import (  # noqa: E402
    LEG_JOINTS, LEG_KD, LEG_KP, STEPS_PER_POLICY, WHEEL_JOINTS, WHEEL_KV,
    actuator_ids_of, build_obs,
)


class Cmd:
    vx = 0.0
    wz = 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--spring-ff", type=float, default=0.0)
    args = p.parse_args()

    import onnxruntime as ort

    mj = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(mj)
    data.qpos[2] = 0.30
    mujoco.mj_forward(mj, data)

    sess = ort.InferenceSession(args.policy, providers=["CPUExecutionProvider"])
    i_name = sess.get_inputs()[0].name
    leg_act, wheel_act = actuator_ids_of(mj)
    spring_act = [a for a in (mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{j}_ctrl")
                              for j in ("left_spring2_joint", "right_spring2_joint")) if a >= 0]

    last_action = np.zeros(6, np.float32)
    action = np.zeros(6, np.float32)
    cmd = Cmd()

    def key_cb(keycode):
        if keycode == ord('w') or keycode == ord('W'):
            cmd.vx = min(cmd.vx + 0.25, 2.5)
        elif keycode == ord('s') or keycode == ord('S'):
            cmd.vx = max(cmd.vx - 0.25, -2.5)
        elif keycode == ord('a') or keycode == ord('A'):
            cmd.wz = min(cmd.wz + 0.5, 6.28)
        elif keycode == ord('d') or keycode == ord('D'):
            cmd.wz = max(cmd.wz - 0.5, -6.28)
        elif keycode == ord(' '):
            cmd.vx = cmd.wz = 0.0
        elif keycode == ord('r') or keycode == ord('R'):
            data.qpos[:] = mj.key_qpos[0].copy() if mj.nkey else data.qpos
            data.qpos[2] = 0.30
            data.qvel[:] = 0
            mujoco.mj_forward(mj, data)
        print(f"cmd: vx={cmd.vx:+.2f} m/s  wz={cmd.wz:+.2f} rad/s")

    def step_cb(m, d):
        nonlocal action, last_action
        obs_cmd = np.array([cmd.vx, 0.0, cmd.wz], np.float32)
        obs = build_obs(m, d, obs_cmd, 0.22, last_action)
        action = sess.run(None, {i_name: obs[None]})[0][0]
        last_action = action.copy()

    with mj_viewer.launch_passive(mj, data, key_callback=key_cb) as v:
        tick = 0
        while v.is_running():
            if tick % STEPS_PER_POLICY == 0:
                step_cb(mj, data)
            leg_t = action[:4] * 0.5
            wheel_v = np.clip(action[4:] * 10.0, -30, 30)
            for k, jname in enumerate(LEG_JOINTS):
                j = mj.joint(jname).id
                q, dq = data.qpos[mj.jnt_qposadr[j]], data.qvel[mj.jnt_dofadr[j]]
                data.ctrl[leg_act[k]] = LEG_KP * (leg_t[k] - q) - LEG_KD * dq
            for k, jname in enumerate(WHEEL_JOINTS):
                dq = data.qvel[mj.jnt_dofadr[mj.joint(jname).id]]
                data.ctrl[wheel_act[k]] = WHEEL_KV * (wheel_v[k] - dq)
            for a in spring_act:
                data.ctrl[a] = args.spring_ff
            mujoco.mj_step(mj, data)
            if tick % 10 == 0:
                v.sync()
            tick += 1
            if data.qpos[2] < 0.05:
                data.qpos[2] = 0.30
                data.qvel[:] = 0
                mujoco.mj_forward(mj, data)


if __name__ == "__main__":
    main()
