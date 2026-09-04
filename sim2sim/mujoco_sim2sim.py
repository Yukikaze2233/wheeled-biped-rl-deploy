#!/usr/bin/env python3
"""MuJoCo sim2sim: replay a trained ONNX policy with deploy-grade timing.

Pipeline mirrors the real robot loop (CONTRACT.md):
  500 Hz physics/PD loop; every 10 control ticks (50 Hz) assemble the 35D
  observation, run ONNX inference, hold the 6D action; legs get position-PD
  torques, wheels get velocity-PD torques; optional obs/action delay and noise.

Usage:
  MUJOCO_GL=disable python3 sim2sim/mujoco_sim2sim.py \
      --model models/toy_wheeled_biped.xml --policy policy.onnx --duration 5

The toy MJCF validates the whole loop without a real robot model; swap in our
robot MJCF (converted from URDF, joint names per contract) for real sim2sim.
"""
import argparse
import time
from collections import deque

import mujoco
import numpy as np

CTRL_HZ = 500.0
POLICY_HZ = 50.0
CTRL_DT = 1.0 / CTRL_HZ
STEPS_PER_POLICY = int(CTRL_HZ / POLICY_HZ)

# CONTRACT.md sections 2/3
SCALE_ANG_VEL = 0.5
SCALE_HEIGHT_CMD = 5.0
SCALE_JOINT_VEL = 0.1
LEG_ACTION_SCALE = 0.5
WHEEL_ACTION_SCALE = 10.0
MAX_WHEEL_VEL = 30.0
CLIP_OBS = 100.0
LEG_KP, LEG_KD = 60.0, 2.0
WHEEL_KV = 0.2  # velocity servo torque per rad/s error
LEG_JOINTS = ("left_front1_joint", "left_rear1_joint", "right_front1_joint", "right_rear1_joint")
WHEEL_JOINTS = ("left_wheel_joint", "right_wheel_joint")
DEFAULT_LEG_POS = np.zeros(4)


def actuator_ids_of(mj: mujoco.MjModel) -> tuple[list[int], list[int]]:
    """(leg_act_ids, wheel_act_ids) by contract joint name; MJCFs name
    actuators "{joint}_ctrl" and order them per-side, NOT per contract."""
    def act_id(joint):
        for cand in (f"{joint}_ctrl", joint):
            a = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_ACTUATOR, cand)
            if a >= 0:
                return a
        raise KeyError(f"no actuator for {joint}")
    return [act_id(j) for j in LEG_JOINTS], [act_id(j) for j in WHEEL_JOINTS]


def _root_body_id(mj: mujoco.MjModel) -> int:
    """Body attached to the free joint (root link name varies per model)."""
    for j in range(mj.njnt):
        if mj.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return mj.jnt_bodyid[j]
    raise ValueError("model has no free joint")


def build_obs(mj: mujoco.MjModel, data: mujoco.MjData, cmd: np.ndarray,
              height_cmd: float, last_action: np.ndarray) -> np.ndarray:
    """Assemble the 35D observation per CONTRACT.md section 2."""
    torso = _root_body_id(mj)

    # body-frame angular velocity + linear velocity (flg_local=True)
    vel = np.zeros(6)
    mujoco.mj_objectVelocity(mj, data, mujoco.mjtObj.mjOBJ_BODY, torso, vel, True)
    ang_vel = vel[0:3]

    # projected gravity: R^T @ g_hat
    R = data.xmat[torso].reshape(3, 3)
    grav = R.T @ np.array([0.0, 0.0, -1.0])

    leg_pos = np.array([
        data.qpos[mj.jnt_qposadr[mj.joint(n).id]] for n in LEG_JOINTS
    ]) - DEFAULT_LEG_POS

    leg_vel = np.array([data.qvel[mj.jnt_dofadr[mj.joint(n).id]] for n in LEG_JOINTS])
    wheel_vel = np.array([data.qvel[mj.jnt_dofadr[mj.joint(n).id]] for n in WHEEL_JOINTS])

    obs = np.concatenate([
        cmd,                                   # 0-2   cmd vx, vy, wz
        [height_cmd * SCALE_HEIGHT_CMD],       # 3
        ang_vel * SCALE_ANG_VEL,               # 4-6
        grav,                                  # 7-9
        leg_pos,                               # 10-13
        np.zeros(2),                           # 14-15 wheel pos slots (muted)
        leg_vel * SCALE_JOINT_VEL,             # 16-19
        wheel_vel * SCALE_JOINT_VEL,           # 20-21
        last_action,                           # 22-27
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # 28-34 normal mode
    ]).astype(np.float32)
    return np.clip(np.nan_to_num(obs), -CLIP_OBS, CLIP_OBS)


class ObsDelay:
    """Fixed observation delay in control ticks (0 = none)."""

    def __init__(self, steps: int):
        self.steps = steps
        self.buf: deque[np.ndarray] = deque(maxlen=steps) if steps > 0 else None

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        if self.buf is None:
            return obs
        self.buf.append(obs)
        return self.buf[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="MuJoCo model (MJCF)")
    p.add_argument("--policy", required=True, help="ONNX policy obs[1,35] -> actions[1,6]")
    p.add_argument("--duration", type=float, default=10.0, help="simulated seconds")
    p.add_argument("--vx", type=float, default=0.0)
    p.add_argument("--wz", type=float, default=0.0)
    p.add_argument("--height-cmd", type=float, default=0.22)
    p.add_argument("--obs-delay-ticks", type=int, default=0, help="delay in 500 Hz ticks")
    p.add_argument("--action-delay-ticks", type=int, default=0)
    p.add_argument("--noise-std", type=float, default=0.0, help="obs gaussian noise std")
    args = p.parse_args()

    import onnxruntime as ort

    mj = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(mj)
    if any(n not in [mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(mj.njnt)]
           for n in LEG_JOINTS + WHEEL_JOINTS):
        print("note: model lacks contract joint names — obs/action mapping will fail")

    sess = ort.InferenceSession(args.policy, providers=["CPUExecutionProvider"])
    i_name = sess.get_inputs()[0].name

    act_delay_q: deque[np.ndarray] = deque(
        [np.zeros(6, np.float32)] * (args.action_delay_ticks + 1), maxlen=args.action_delay_ticks + 1)
    obs_delay = ObsDelay(args.obs_delay_ticks)
    rng = np.random.default_rng(0)

    cmd = np.array([args.vx, 0.0, args.wz], np.float32)
    last_action = np.zeros(6, np.float32)
    action = np.zeros(6, np.float32)

    n_ticks = int(args.duration * CTRL_HZ)
    t0 = time.time()
    for tick in range(n_ticks):
        if tick % STEPS_PER_POLICY == 0:
            obs = build_obs(mj, data, cmd, args.height_cmd, last_action)
            if args.noise_std > 0:
                obs = obs + rng.normal(0, args.noise_std, obs.shape).astype(np.float32)
            obs = obs_delay(obs)
            action = sess.run(None, {i_name: obs[None]})[0][0]
            act_delay_q.append(action)
            last_action = action.copy()

        applied = act_delay_q[0]
        leg_t = DEFAULT_LEG_POS + LEG_ACTION_SCALE * applied[:4]
        wheel_v = np.clip(WHEEL_ACTION_SCALE * applied[4:], -MAX_WHEEL_VEL, MAX_WHEEL_VEL)

        # 500 Hz software PD ("hardware_pd_vel" semantics)
        for k, jname in enumerate(LEG_JOINTS):
            j = mj.joint(jname).id
            q = data.qpos[mj.jnt_qposadr[j]]
            dq = data.qvel[mj.jnt_dofadr[j]]
            data.ctrl[k] = LEG_KP * (leg_t[k] - q) - LEG_KD * dq
        for k, jname in enumerate(WHEEL_JOINTS):
            dq = data.qvel[mj.jnt_dofadr[mj.joint(jname).id]]
            data.ctrl[4 + k] = WHEEL_KV * (wheel_v[k] - dq)

        mujoco.mj_step(mj, data)

        if data.qpos[2] < 0.05:
            print(f"[t={tick * CTRL_DT:.2f}s] base below 0.05 m — episode ended")
            break

    wall = time.time() - t0
    print(f"done: {n_ticks} ticks in {wall:.1f}s wall "
          f"(RTF {n_ticks * CTRL_DT / max(wall, 1e-6):.1f}x), final base z = {data.qpos[2]:.3f} m")


if __name__ == "__main__":
    main()
