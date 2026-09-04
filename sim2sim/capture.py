#!/usr/bin/env python3
"""Run the policy on a real MJCF and save periodic screenshots (no display).

    MUJOCO_GL=disable python3 sim2sim/capture.py --model models/wheelbipe_v14.xml \
        --policy <scut.onnx> --duration 4 --shot-every 0.5 --out shots/
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from mujoco_sim2sim import (  # noqa: E402
    CTRL_DT, LEG_JOINTS, LEG_KD, LEG_KP, STEPS_PER_POLICY, WHEEL_JOINTS,
    WHEEL_KV, build_obs,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--duration", type=float, default=4.0)
    p.add_argument("--vx", type=float, default=0.5)
    p.add_argument("--shot-every", type=float, default=0.5, help="seconds between shots")
    p.add_argument("--out", default="shots")
    args = p.parse_args()

    from PIL import Image

    import onnxruntime as ort

    os.makedirs(args.out, exist_ok=True)
    mj = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(mj)
    # start from a plausible standing height
    for jn, val in (("left_front1_joint", 0.0), ("left_rear1_joint", 0.0)):
        j = mj.joint(jn).id
        data.qpos[mj.jnt_qposadr[j]] = val
    data.qpos[2] = 0.30  # contract standing height
    renderer = mujoco.Renderer(mj, height=480, width=640)

    sess = ort.InferenceSession(args.policy, providers=["CPUExecutionProvider"])
    i_name = sess.get_inputs()[0].name

    # actuator ids by contract-joint name (MJCFs name them "{joint}_ctrl")
    def act_id(joint):
        for cand in (f"{joint}_ctrl", joint):
            a = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_ACTUATOR, cand)
            if a >= 0:
                return a
        raise KeyError(f"no actuator for {joint}")

    leg_act = [act_id(j) for j in LEG_JOINTS]
    wheel_act = [act_id(j) for j in WHEEL_JOINTS]

    cmd = np.array([args.vx, 0.0, 0.0], np.float32)
    last_action = np.zeros(6, np.float32)
    action = np.zeros(6, np.float32)

    n_ticks = int(args.duration / CTRL_DT)
    shot_every = max(int(args.shot_every / CTRL_DT), 1)
    for tick in range(n_ticks):
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
        mujoco.mj_step(mj, data)

        if tick % shot_every == 0:
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.35]
            cam.distance = 2.5
            cam.azimuth = 130
            cam.elevation = 15
            renderer.update_scene(data, camera=cam)
            img = np.flipud(renderer.render())
            path = os.path.join(args.out, f"t{tick * CTRL_DT:04.1f}s.png")
            Image.fromarray(img).save(path)
            if tick % (STEPS_PER_POLICY * 20) == 0:
                print(f"t={tick * CTRL_DT:4.1f}s base_z={data.qpos[2]:.3f} "
                      f"shot={path}")
    print(f"done: {n_ticks} ticks, shots in {args.out}/ (final z={data.qpos[2]:.3f})")


if __name__ == "__main__":
    main()
