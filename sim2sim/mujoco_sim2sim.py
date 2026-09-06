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
# physics decimation: official MJCF runs timestep=0.001 -> 2 physics steps
# per 2ms control tick; PD is recomputed EVERY physics step (their write())
PHYSICS_STEPS_PER_TICK = 2

# torque saturation (CONTRACT.md section 3; unbounded PD oscillates)
LEG_TORQUE_LIMIT = 40.0
WHEEL_TORQUE_LIMIT = 5.0
# gas spring: authoritative training cfg (pretrained env.yaml spring_settings):
# mode linear, spring_offset 0.06076, linear_up 600 @ full compression,
# linear_down 400 @ free length, linear_length 0.07, damping False.
SPRING_OFFSET = 0.06076
SPRING_TRAVEL = 0.07
SPRING_FORCE_FREE, SPRING_FORCE_COMPRESSED = 400.0, 600.0
SPRING_VISCOUS_DAMPING = 0.0  # training used damping=False; enable for real2sim tuning


def gas_spring_force(mj, data, spring_qadr, spring_vel_adr) -> float:
    q = data.qpos[spring_qadr]
    compression = min(max(SPRING_OFFSET - q, 0.0), SPRING_TRAVEL)
    spring_force = SPRING_FORCE_FREE + (SPRING_FORCE_COMPRESSED - SPRING_FORCE_FREE) / SPRING_TRAVEL * compression
    damping = -SPRING_VISCOUS_DAMPING * data.qvel[spring_vel_adr]
    return spring_force + damping


class ActuatorModel:
    """Official leg-motor second-order response (semi-implicit Euler with
    torque slew limit): wn=50Hz zeta=0.10 slew=1500Nm/s, speed droop after
    4 rad/s, output limit 54 Nm."""

    def __init__(self):
        self.y = 0.0    # applied effort
        self.yr = 0.0   # its rate

    def step(self, request: float, vel: float, dt: float) -> float:
        droop = max(0.0, abs(vel) - 4.0)
        gain = max(1.0 - 0.055 * droop, 0.55)
        target = gain * request
        wn = 2 * 3.14159265 * 50.0
        zeta = 0.10
        acc = wn * wn * (target - self.y) - 2 * zeta * wn * self.yr
        self.yr = min(max(self.yr + dt * acc, -1500.0), 1500.0)
        self.y = min(max(self.y + dt * self.yr, -54.0), 54.0)
        return self.y


def spring_binding(mj):
    """[((qpos_adr, vel_adr), actuator_id)] for both gas-spring joints."""
    out = []
    for side in ("left", "right"):
        jn = f"{side}_spring2_joint"
        a = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{jn}_ctrl")
        if a >= 0:
            j = mj.joint(jn).id
            out.append(((mj.jnt_qposadr[j], mj.jnt_dofadr[j]), a))
    return out


def step_control(mj, data, action, leg_act, wheel_act, spring_act=None,
                 spring_bias=0.0, leg_models=None, dt=0.001):
    """One 2ms control tick with official semantics: PD per physics substep,
    torque saturation, gas-spring model force, leg-motor second-order model."""
    leg_t = action[:4] * 0.5
    wheel_v = np.clip(action[4:] * WHEEL_ACTION_SCALE, -MAX_WHEEL_VEL, MAX_WHEEL_VEL)
    if leg_models is None:
        # per-model state keyed by the MjModel id (no cross-sim leakage)
        store = getattr(step_control, "_stores", {})
        key = id(mj)
        if key not in store:
            store[key] = [ActuatorModel() for _ in LEG_JOINTS]
        leg_models = store[key]
    for _ in range(PHYSICS_STEPS_PER_TICK):
        for k, jname in enumerate(LEG_JOINTS):
            j = mj.joint(jname).id
            q, dq = data.qpos[mj.jnt_qposadr[j]], data.qvel[mj.jnt_dofadr[j]]
            tau = LEG_KP * (leg_t[k] - q) - LEG_KD * dq
            tau = float(np.clip(tau, -LEG_TORQUE_LIMIT, LEG_TORQUE_LIMIT))
            data.ctrl[leg_act[k]] = leg_models[k].step(tau, dq, dt)
        for k, jname in enumerate(WHEEL_JOINTS):
            dq = data.qvel[mj.jnt_dofadr[mj.joint(jname).id]]
            tau = WHEEL_KV * (wheel_v[k] - dq)
            data.ctrl[wheel_act[k]] = np.clip(tau, -WHEEL_TORQUE_LIMIT, WHEEL_TORQUE_LIMIT)
        if spring_act:
            for (qadr, vadr), act in spring_act:
                data.ctrl[act] = gas_spring_force(mj, data, qadr, vadr) + spring_bias
        mujoco.mj_step(mj, data)

# CONTRACT.md sections 2/3
SCALE_ANG_VEL = 0.5
SCALE_HEIGHT_CMD = 5.0
SCALE_JOINT_VEL = 0.1
LEG_ACTION_SCALE = 0.5
# wheel velocity mode (use_wheel_vel_control=True): env.py overrides
# wheel_action_scale = wheel_vel_action_scale = 10.0 and
# max_wheel_vel = 100.0 * 1.5 = 150.0. Authoritative from the training code.
WHEEL_ACTION_SCALE = 10.0
MAX_WHEEL_VEL = 150.0
CLIP_OBS = 100.0
LEG_KP, LEG_KD = 60.0, 2.0
WHEEL_KV = 0.2  # velocity servo torque per rad/s error
# Authoritative order from the official training cfg (legs_act joint_names_expr =
# [".*_rear1_joint", ".*_front1_joint"] -> rear leg first). Verified against
# pretrained env.yaml: action dims = [left_rear, right_rear, left_front, right_front].
LEG_JOINTS = ("left_rear1_joint", "right_rear1_joint", "left_front1_joint", "right_front1_joint")
WHEEL_JOINTS = ("left_wheel_joint", "right_wheel_joint")
DEFAULT_LEG_POS = np.zeros(4)


def actuator_ids_of(mj: mujoco.MjModel, leg_names=LEG_JOINTS,
                     wheel_names=WHEEL_JOINTS) -> tuple[list[int], list[int]]:
    """(leg_act_ids, wheel_act_ids) by contract joint name; MJCFs name
    actuators "{joint}_ctrl" and order them per-side, NOT per contract."""
    def act_id(joint):
        for cand in (f"{joint}_ctrl", joint):
            a = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_ACTUATOR, cand)
            if a >= 0:
                return a
        raise KeyError(f"no actuator for {joint}")
    return [act_id(j) for j in leg_names], [act_id(j) for j in wheel_names]


def _root_body_id(mj: mujoco.MjModel) -> int:
    """Body attached to the free joint (root link name varies per model)."""
    for j in range(mj.njnt):
        if mj.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return mj.jnt_bodyid[j]
    raise ValueError("model has no free joint")


def build_obs(mj: mujoco.MjModel, data: mujoco.MjData, cmd: np.ndarray,
              height_cmd: float, last_action: np.ndarray,
              leg_names=LEG_JOINTS, wheel_names=WHEEL_JOINTS,
              default_leg_pos=None) -> np.ndarray:
    """Assemble the 35D observation per CONTRACT.md section 2."""
    if default_leg_pos is None:
        default_leg_pos = DEFAULT_LEG_POS
    torso = _root_body_id(mj)

    # body-frame angular velocity + linear velocity (flg_local=True)
    vel = np.zeros(6)
    mujoco.mj_objectVelocity(mj, data, mujoco.mjtObj.mjOBJ_BODY, torso, vel, True)
    ang_vel = vel[0:3]

    # projected gravity: R^T @ g_hat
    R = data.xmat[torso].reshape(3, 3)
    grav = R.T @ np.array([0.0, 0.0, -1.0])

    leg_pos = np.array([
        data.qpos[mj.jnt_qposadr[mj.joint(n).id]] for n in leg_names
    ]) - np.asarray(default_leg_pos)

    leg_vel = np.array([data.qvel[mj.jnt_dofadr[mj.joint(n).id]] for n in leg_names])
    wheel_vel = np.array([data.qvel[mj.jnt_dofadr[mj.joint(n).id]] for n in wheel_names])

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
    p.add_argument("--init-z", type=float, default=0.22, help="initial base height (training uses absolute height; default standing 0.22)")
    p.add_argument("--obs-delay-ticks", type=int, default=4,
                    help="obs delay in policy steps of 20 ms (default 4 = 80 ms; "
                         "training always uses 20-80 ms)")
    p.add_argument("--action-delay-ticks", type=int, default=3,
                    help="action delay in policy steps of 20 ms (default 3 = 60 ms; "
                         "training always uses 20-60 ms)")
    p.add_argument("--noise-std", type=float, default=0.0, help="obs gaussian noise std")
    p.add_argument("--legs", nargs=4, default=list(LEG_JOINTS),
                   help="leg joint names, contract order (default: official V14 rear-first)")
    p.add_argument("--wheels", nargs=2, default=list(WHEEL_JOINTS),
                   help="wheel joint names [left, right]")
    p.add_argument("--leg-defaults", nargs=4, type=float, default=[0.0, 0.0, 0.0, 0.0],
                   help="default leg pose subtracted in obs / added in action decode")
    p.add_argument("--no-springs", action="store_true",
                   help="model has no gas springs (serial-leg robot)")
    p.add_argument("--wheel-kv", type=float, default=0.2,
                   help="wheel velocity servo gain Nm/(rad/s); V14=0.2 (official), "
                        "own serial-leg V3.3=1.0 (trained value)")
    args = p.parse_args()

    import onnxruntime as ort

    mj = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(mj)
    # absolute-height semantics (use_absolute_height=True in training):
    # the height command is the base origin z in the world frame.
    if mj.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        data.qpos[2] = args.init_z
    mujoco.mj_forward(mj, data)
    leg_names = tuple(args.legs)
    wheel_names = tuple(args.wheels)
    default_leg_pos = np.asarray(args.leg_defaults)
    if any(n not in [mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(mj.njnt)]
           for n in leg_names + wheel_names):
        print("note: model lacks contract joint names — obs/action mapping will fail")
    leg_act, wheel_act = actuator_ids_of(mj, leg_names, wheel_names)
    spring_act = [] if args.no_springs else spring_binding(mj)
    print(f"actuators: legs={leg_act} wheels={wheel_act} springs={[a for (_, a) in spring_act]}")

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
            obs = build_obs(mj, data, cmd, args.height_cmd, last_action,
                            leg_names, wheel_names, default_leg_pos)
            if args.noise_std > 0:
                obs = obs + rng.normal(0, args.noise_std, obs.shape).astype(np.float32)
            obs = obs_delay(obs)
            action = sess.run(None, {i_name: obs[None]})[0][0]
            act_delay_q.append(action)
            last_action = action.copy()

        applied = act_delay_q[0]
        leg_t = default_leg_pos + LEG_ACTION_SCALE * applied[:4]
        wheel_v = np.clip(WHEEL_ACTION_SCALE * applied[4:], -MAX_WHEEL_VEL, MAX_WHEEL_VEL)

        # 500 Hz software PD ("hardware_pd_vel" semantics), actuator ids resolved
        # BY NAME — official MJCF actuator order is not contract order.
        # 2 physics substeps per 2 ms tick (timestep 0.001); PD recomputed every
        # substep to mirror the official write() semantics.
        for _ in range(PHYSICS_STEPS_PER_TICK):
            for k, jname in enumerate(leg_names):
                j = mj.joint(jname).id
                q = data.qpos[mj.jnt_qposadr[j]]
                dq = data.qvel[mj.jnt_dofadr[j]]
                data.ctrl[leg_act[k]] = np.clip(LEG_KP * (leg_t[k] - q) - LEG_KD * dq,
                                                -LEG_TORQUE_LIMIT, LEG_TORQUE_LIMIT)
            for k, jname in enumerate(wheel_names):
                dq = data.qvel[mj.jnt_dofadr[mj.joint(jname).id]]
                data.ctrl[wheel_act[k]] = np.clip(args.wheel_kv * (wheel_v[k] - dq),
                                                  -WHEEL_TORQUE_LIMIT, WHEEL_TORQUE_LIMIT)
            # gas springs: per-step force (authoritative linear curve, no damping)
            for (qadr, vadr), act in spring_act:
                data.ctrl[act] = gas_spring_force(mj, data, qadr, vadr)

            mujoco.mj_step(mj, data)

        if data.qpos[2] < 0.05:
            print(f"[t={tick * CTRL_DT:.2f}s] base below 0.05 m — episode ended")
            break

    wall = time.time() - t0
    print(f"done: {n_ticks} ticks in {wall:.1f}s wall "
          f"(RTF {n_ticks * CTRL_DT / max(wall, 1e-6):.1f}x), final base z = {data.qpos[2]:.3f} m")


if __name__ == "__main__":
    main()
