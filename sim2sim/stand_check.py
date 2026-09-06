#!/usr/bin/env python3
"""Standing-balance acceptance test: PREPARE -> N-second zero-command hold.

    python3 sim2sim/stand_check.py --model <scene.xml> --policy <onnx> \
        [--seconds 10] [--height 0.38] [--prepare-s 1.0]

Pass criteria: never falls (z > 0.10), steady-state height std < 0.03 m,
total drift < 0.5 m over the hold. Exit code 0 on pass, 1 on fail —
usable as the regression gate for any future self-trained policy.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from mujoco_sim2sim import (  # noqa: E402
    CTRL_DT, LEG_JOINTS, STEPS_PER_POLICY, actuator_ids_of, build_obs,
    gas_spring_force, spring_binding, step_control,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--height", type=float, default=0.38)
    p.add_argument("--prepare-s", type=float, default=1.0)
    p.add_argument("--height-cmd", type=float, default=0.22)
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

    def spring_apply():
        for (qadr, vadr), a in spring_act:
            data.ctrl[a] = gas_spring_force(mj, data, qadr, vadr)

    # PREPARE: legs PD-hold at default pose (Kp80/Kd2), wheels free
    for _ in range(int(args.prepare_s * 500)):
        for k, jn in enumerate(LEG_JOINTS):
            j = mj.joint(jn).id
            q, dq = data.qpos[mj.jnt_qposadr[j]], data.qvel[mj.jnt_dofadr[j]]
            data.ctrl[leg_act[k]] = 80.0 * (0.0 - q) - 2.0 * dq
        for k, jn in enumerate(("left_wheel_joint", "right_wheel_joint")):
            dq = data.qvel[mj.jnt_dofadr[mj.joint(jn).id]]
            data.ctrl[wheel_act[k]] = float(np.clip(2.0 * (0.0 - dq), -9.99, 9.99))
        spring_apply()
        mujoco.mj_step(mj, data)

    # zero-command RL hold
    cmd = np.zeros(3, np.float32)
    last = np.zeros(6, np.float32)
    action = np.zeros(6, np.float32)
    zs, xs = [], []
    for tick in range(int(args.seconds / CTRL_DT)):
        if tick % STEPS_PER_POLICY == 0:
            obs = build_obs(mj, data, cmd, args.height_cmd, last)
            action = sess.run(None, {i_name: obs[None]})[0][0]
            last = action.copy()
        step_control(mj, data, action, leg_act, wheel_act, spring_act=spring_act)
        zs.append(data.qpos[2])
        xs.append(data.qpos[0])
    zs, xs = np.array(zs), np.array(xs)
    steady = slice(len(zs) // 2, None)

    z_mean, z_std = float(zs[steady].mean()), float(zs[steady].std())
    drift = float(abs(xs[-1]))
    fell = bool(zs.min() < 0.10)
    ok = (not fell) and z_std < 0.03 and drift < 0.5
    print(f"PREPARE end z={data.qpos[2]:.3f}")
    print(f"hold {args.seconds:.0f}s: z={z_mean:.3f}±{z_std:.4f}  drift={drift:.2f}m  "
          f"osc={xs[steady].std():.3f}m  min_z={zs.min():.3f}  "
          f"{'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
