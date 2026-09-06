"""Interactive live window (grid floor, tracking camera, keyboard control).

Usage:  .venv_mj314/bin/python live_control.py --robot v33|v14
Keys:   W/S  forward speed +/-,  A/D  yaw rate +/-,  R  reset pose,  ESC  quit
        (hold keys: they are sampled every frame)
The robot auto-resets when it falls (base below threshold or tilted over).
"""
import argparse
import json
import sys

import glfw as _glfw
import OpenGL.GL as gl
import numpy as np
import onnxruntime as ort
import mujoco
from mujoco import Renderer

ROOT = "/home/yukikaze/Documents/workspace/robot_rl"
W, H = 1280, 720


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--robot", default="v33", choices=["v33", "v14"])
    args = p.parse_args()

    if args.robot == "v33":
        scene = f"{ROOT}/isaac_wheeled_rl_deploy/models/urdf_v3.3_scene.xml"
        policy = f"{ROOT}/runs_v33/v33_policy.onnx"
        legs = ("L_joint1", "L_joint2", "R_joint1", "R_joint2")
        wheels = ("L_joint3", "R_joint3")
        springs = False
        wheel_kv = 1.0
        init_z = 0.48
        height_cmd = 0.48
        defaults = json.load(open(f"{ROOT}/isaac_wheeled_rl_train/assets/urdf_v33/defaults.json"))
        default_pose = np.array([defaults["default_joint_pos"][n] for n in legs])
        delays = (0, 0)
        vx_step, wz_step = 0.1, 0.4
        vx_max = 1.0
        fall_z = 0.15
    else:
        sys.path.insert(0, f"{ROOT}/isaac_wheeled_rl_deploy/sim2sim")
        import mujoco_sim2sim as S
        scene = f"{ROOT}/isaac_wheeled_rl_deploy/models/wheelbipe_v14_gimbal_scene.xml"
        policy = (f"{ROOT}/scut-wheeled-legged-rl/pretrained/26_infantry/"
                  "flat_and_rotation/2026-07-19_09-14-50/exported/2026-07-19_09-14-50_13k.onnx")
        legs = S.LEG_JOINTS
        wheels = S.WHEEL_JOINTS
        springs = True
        wheel_kv = S.WHEEL_KV
        init_z = 0.22
        height_cmd = 0.22
        default_pose = np.zeros(4)
        delays = (4, 3)  # official training-typical 80/60 ms
        vx_step, wz_step = 0.2, 0.5
        vx_max = 1.5
        fall_z = 0.08

    mj = mujoco.MjModel.from_xml_path(scene)
    d = mujoco.MjData(mj)
    leg_ids = [mj.joint(n).id for n in legs]
    wheel_ids = [mj.joint(n).id for n in wheels]
    leg_act = [mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in legs]
    wheel_act = [mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in wheels]
    leg_lim_lo = np.array([mj.jnt_range[j][0] for j in leg_ids])
    leg_lim_hi = np.array([mj.jnt_range[j][1] for j in leg_ids])
    spring_act = []
    if springs:
        sys.path.insert(0, f"{ROOT}/isaac_wheeled_rl_deploy/sim2sim")
        import mujoco_sim2sim as S
        spring_act = S.spring_binding(mj)

    def reset_pose():
        d.qpos[0:2] = 0
        d.qpos[2] = init_z
        d.qpos[3:7] = [1, 0, 0, 0]
        d.qvel[:] = 0
        for k, j in enumerate(leg_ids):
            d.qpos[mj.jnt_qposadr[j]] = default_pose[k]
        mujoco.mj_forward(mj, d)

    reset_pose()
    sess = ort.InferenceSession(policy, providers=["CPUExecutionProvider"])
    i_name = sess.get_inputs()[0].name
    vx_cmd, wz_cmd = 0.3 if args.robot == "v33" else 0.5, 0.0
    cmd = np.array([vx_cmd, 0.0, wz_cmd], np.float32)
    last_action = np.zeros(6, np.float32)
    action = np.zeros(6, np.float32)
    obs_ring = [np.zeros(35, np.float32)] * (delays[0] + 1)
    act_ring = [np.zeros(6, np.float32)] * (delays[1] + 1)
    policy_step = 0

    # window + offscreen renderer + tracking camera
    _glfw.init()
    window = _glfw.create_window(W, H, "wheeled-biped RL live control", None, None)
    _glfw.make_context_current(window)
    renderer = Renderer(mj, H, W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = 1
    cam.lookat[:] = [0.0, 0.0, 0.3]
    cam.distance = 1.6
    cam.azimuth = 160.0
    cam.elevation = -18.0
    # mouse interaction: left-drag orbit, right-drag pan, scroll zoom
    drag = {"button": -1, "x": 0.0, "y": 0.0}

    def _on_mouse_button(w, btn, act, mod):
        if act == _glfw.PRESS:
            drag["button"] = btn
            drag["x"], drag["y"] = _glfw.get_cursor_pos(w)
        else:
            drag["button"] = -1

    _glfw.set_mouse_button_callback(window, _on_mouse_button)
    _glfw.set_scroll_callback(window, lambda w, dx, dy: setattr(
        cam, "distance", max(0.3, cam.distance * (1.0 + dy * 0.08))))

    tick = 0
    title_t = 0.0
    while not _glfw.window_should_close(window):
        if _glfw.get_key(window, _glfw.KEY_ESCAPE) == _glfw.PRESS:
            break
        if _glfw.get_key(window, _glfw.KEY_W) == _glfw.PRESS:
            vx_cmd = min(vx_cmd + vx_step * 0.02, vx_max)
        if _glfw.get_key(window, _glfw.KEY_S) == _glfw.PRESS:
            vx_cmd = max(vx_cmd - vx_step * 0.02, -vx_max)
        if _glfw.get_key(window, _glfw.KEY_A) == _glfw.PRESS:
            wz_cmd = wz_cmd + wz_step * 0.02
        if _glfw.get_key(window, _glfw.KEY_D) == _glfw.PRESS:
            wz_cmd = wz_cmd - wz_step * 0.02
        if _glfw.get_key(window, _glfw.KEY_R) == _glfw.PRESS:
            reset_pose()
            last_action[:] = 0
            action[:] = 0
        cmd[0], cmd[2] = vx_cmd, wz_cmd

        if tick % 10 == 0:
            vel = np.zeros(6)
            mujoco.mj_objectVelocity(mj, d, mujoco.mjtObj.mjOBJ_BODY, 1, vel, 1)
            ang = vel[0:3]
            R = d.xmat[1].reshape(3, 3)
            grav = R.T @ np.array([0.0, 0.0, -1.0])
            leg_pos = np.array([d.qpos[mj.jnt_qposadr[j]] for j in leg_ids]) - default_pose
            leg_vel = np.array([d.qvel[mj.jnt_dofadr[j]] for j in leg_ids])
            wheel_vel = np.array([d.qvel[mj.jnt_dofadr[j]] for j in wheel_ids])
            obs = np.concatenate([
                cmd, [height_cmd * 5.0], ang * 0.5, grav,
                leg_pos, np.zeros(2), leg_vel * 0.1, wheel_vel * 0.1,
                last_action, [1.0, 0, 0, 0, 0, 0, 0],
            ]).astype(np.float32)
            obs = np.clip(np.nan_to_num(obs), -100.0, 100.0)
            obs_ring[policy_step % (delays[0] + 1)] = obs
            delayed = obs_ring[(policy_step - delays[0]) % (delays[0] + 1)]
            action = sess.run(None, {i_name: delayed[None]})[0][0]
            last_action = action
            act_ring[policy_step % (delays[1] + 1)] = action
            policy_step += 1
        applied = act_ring[(policy_step - delays[1]) % (delays[1] + 1)]
        leg_t = np.clip(default_pose + 0.5 * applied[:4], leg_lim_lo, leg_lim_hi)
        wheel_t = np.clip(10.0 * applied[4:], -150.0, 150.0)
        for _ in range(2):
            for k, j in enumerate(leg_ids):
                q = d.qpos[mj.jnt_qposadr[j]]
                dq = d.qvel[mj.jnt_dofadr[j]]
                d.ctrl[leg_act[k]] = float(np.clip(60.0 * (leg_t[k] - q) - 2.0 * dq, -40, 40))
            for k, j in enumerate(wheel_ids):
                dq = d.qvel[mj.jnt_dofadr[j]]
                d.ctrl[wheel_act[k]] = float(np.clip(wheel_kv * (wheel_t[k] - dq), -5, 5))
            for (qadr, vadr), act in spring_act:
                d.ctrl[act] = S.gas_spring_force(mj, d, qadr, vadr) if springs else 0.0
            mujoco.mj_step(mj, d)
        tick += 1

        # auto-reset on fall
        if d.qpos[2] < fall_z or abs(grav[2]) < 0.5:
            reset_pose()
            last_action[:] = 0
            action[:] = 0

        if tick % 5 == 0:
            # apply mouse motion (accumulated via callbacks in drag dict)
            if drag["button"] >= 0:
                cx, cy = _glfw.get_cursor_pos(window)
                dx, dy = cx - drag["x"], cy - drag["y"]
                if drag["button"] == _glfw.MOUSE_BUTTON_LEFT:
                    cam.azimuth -= 0.4 * dx
                    cam.elevation = float(np.clip(cam.elevation - 0.3 * dy, -80.0, 80.0))
                elif drag["button"] == _glfw.MOUSE_BUTTON_RIGHT:
                    scale = cam.distance / 500.0
                    right = np.array([np.cos(np.radians(cam.azimuth + 90.0)),
                                      np.sin(np.radians(cam.azimuth + 90.0)), 0.0])
                    up = np.array([0.0, 0.0, 1.0])
                    cam.lookat[:] += scale * (-dx * right + dy * up)
                drag["x"], drag["y"] = cx, cy
            renderer.update_scene(d, camera=cam)
            # command-direction arrow above the robot (green), like the
            # official Isaac Lab goal_vel_visualizer
            if abs(vx_cmd) > 0.05:
                base_pos = d.xpos[1].copy()
                Rb = d.xmat[1].reshape(3, 3)
                dir_w = Rb @ np.array([0.0, -np.sign(vx_cmd), 0.0])  # forward = -y_base
                dir_w /= np.linalg.norm(dir_w)
                z = dir_w
                x = np.cross([0.0, 0.0, 1.0], z)
                if np.linalg.norm(x) < 1e-6:
                    x = np.array([1.0, 0.0, 0.0])
                x /= np.linalg.norm(x)
                y = np.cross(z, x)
                idx = renderer.scene.ngeom
                renderer.scene.ngeom = idx + 1
                g = renderer.scene.geoms[idx]
                g.type = mujoco.mjtGeom.mjGEOM_ARROW
                g.size[:] = [0.02, 0.02, 0.15 + 0.25 * abs(vx_cmd)]
                g.pos[:] = base_pos + [0, 0, 0.30]
                g.mat[:] = np.column_stack([x, y, z])
                g.rgba[:] = [0.2, 0.95, 0.2, 1.0]
            rgb = renderer.render()
            _glfw.make_context_current(window)  # renderer owns a private GL context
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            # glDrawPixels origin is bottom-left; MuJoCo frames are top-down:
            # flip vertically so the window is NOT upside-down.
            gl.glDrawPixels(W, H, gl.GL_RGB, gl.GL_UNSIGNED_BYTE,
                            np.ascontiguousarray(rgb[::-1]))
            _glfw.swap_buffers(window)
            _glfw.poll_events()
            if tick % 50 == 0:
                _glfw.set_window_title(
                    window,
                    f"[{args.robot}] vx={vx_cmd:+.2f} m/s  wz={wz_cmd:+.2f} rad/s  "
                    f"base_z={d.qpos[2]:.3f} m  (W/S 速度, A/D 转向, R 复位, ESC 退出)")
    _glfw.destroy_window(window)
    _glfw.terminate()


if __name__ == "__main__":
    main()
