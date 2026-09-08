#!/usr/bin/env python3
"""Drop-test + render the RMUC2026 field scene (works headless via capture)."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import mujoco, numpy as np
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--scene", default="models/rmuc2026/scene.xml")
p.add_argument("--out", default="shots/rmuc2026")
p.add_argument("--drop", action="store_true", help="drop a probe ball from 1.5m")
p.add_argument("--seconds", type=float, default=2.0)
args = p.parse_args()

mj = mujoco.MjModel.from_xml_path(args.scene)
data = mujoco.MjData(mj)
if args.drop:
    xml_add = ""
ball_body = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "probe") if mj.nbody > 1 else -1
mj.vis.global_.offwidth = 2048
mj.vis.global_.offheight = 2048
renderer = mujoco.Renderer(mj, 540, 960)
os.makedirs(args.out, exist_ok=True)
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat[:] = [-8, 0, 0.3]; cam.distance = 45; cam.azimuth = 90; cam.elevation = -18
for tick in range(int(args.seconds / 0.001)):
    mujoco.mj_step(mj, data)
    if tick % 500 == 0:
        cam.lookat[:] = [0, 0, 0.5]
        renderer.update_scene(data, camera=cam)
        Image.fromarray(np.flipud(renderer.render())).save(f"{args.out}/t{tick*0.001:.1f}s.png")
if ball_body >= 0:
    print("probe ball final z =", round(float(data.xpos[ball_body][2]), 3))
print(f"scene rendered -> {args.out}/")
