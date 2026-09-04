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
    STEPS_PER_POLICY, actuator_ids_of, build_obs, spring_binding, step_control,
)


class Cmd:
    vx = 0.0
    wz = 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--spring-ff", type=float, default=0.0)
    p.add_argument("--no-follow", action="store_true",
                   help="disable camera look-at tracking (keep free camera)")
    p.add_argument("--height", type=float, default=0.38,
                   help="spawn height (training init pose)")
    p.add_argument("--prepare-s", type=float, default=1.0,
                   help="PREPARE phase: legs PD-hold at 0 rad before RL engages")
    args = p.parse_args()

    import onnxruntime as ort

    mj = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(mj)
    data.qpos[2] = args.height
    mujoco.mj_forward(mj, data)

    sess = ort.InferenceSession(args.policy, providers=["CPUExecutionProvider"])
    i_name = sess.get_inputs()[0].name
    leg_act, wheel_act = actuator_ids_of(mj)
    spring_act = spring_binding(mj)

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

    prepare_ticks = int(args.prepare_s * 500)

    def prepare_control():
        """official PREPARE: legs PD at default pose (Kp80/Kd2), wheels free."""
        for k, jname in enumerate(LEG_JOINTS):
            j = mj.joint(jname).id
            q, dq = data.qpos[mj.jnt_qposadr[j]], data.qvel[mj.jnt_dofadr[j]]
            data.ctrl[leg_act[k]] = 80.0 * (0.0 - q) - 2.0 * dq
        for a in wheel_act:
            data.ctrl[a] = 0.0
        for _, a in spring_act:
            from mujoco_sim2sim import gas_spring_force
            pass  # spring handled below

    with mj_viewer.launch_passive(mj, data, key_callback=key_cb) as v:
        tick = 0
        while v.is_running():
            if tick < prepare_ticks:
                prepare_control()
                # keep the gas spring modeled during PREPARE too
                for (qadr, vadr), a in spring_act:
                    from mujoco_sim2sim import gas_spring_force
                    data.ctrl[a] = gas_spring_force(mj, data, qadr, vadr)
                mujoco.mj_step(mj, data)
                tick += 1
                if tick % 10 == 0:
                    if not args.no_follow:
                        v.cam.lookat[:] = [data.qpos[0], data.qpos[1],
                                           max(data.qpos[2], 0.0) + 0.15]
                    v.sync()
                continue
            if tick % STEPS_PER_POLICY == 0:
                step_cb(mj, data)
            step_control(mj, data, action, leg_act, wheel_act, spring_act=spring_act)
            if tick % 10 == 0:
                v.sync()
            tick += 1
            if data.qpos[2] < 0.05:
                data.qpos[2] = 0.30
                data.qvel[:] = 0
                mujoco.mj_forward(mj, data)


if __name__ == "__main__":
    main()
