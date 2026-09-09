# RMUC2026 场地模型(官方 STEP 转换)

来源:官方论坛开源帖(bbs.robomaster.com/article/814728)RMUC2026_V2.0.0.stp(1.25GB)

## 已完成
- STEP → OCP 拆解:1025 个 solid,取 top-60 大件(阈值 5e7 mm³)
- 完整网格:74 块 STL / 2,892,422 面 / 零降采样(chunks/ + visual/)
- 尺寸:29.75 × 16.0 × 1.68 m(与规则手册一致)
- 转换工具:tools/step_to_field.py(--chunk-dir 分块模式)

## ✅ 可用方案(当前生效):高度场

STEP/FBX 网格因自交/非流形无法直接作 MuJoCo 碰撞体(两种来源均实证),
改用**高度场**方案已验证可用:

- `rmuc2026_hfield.png`:官方 FBX(Blender 导出)射线采样 4cm 网格 → 灰度高度图
- `scene_hfield.xml`:MuJoCo hfield 场地(28×15m, 高 0.2–3.4m, 含环形高地/坡道/资源岛)
- 验证:4 探针全场地稳定落座(pB 落在 1.49m 高地上)
- 注意:本机 mujoco 3.12 的 hfield 用户数据走 PNG 灰度(bin 格式不可用)
- 局限:纯高度场无法表达悬空结构(如横杆);视觉细节看 FBX 原文件

## 未解决(诚实挂账)
MuJoCo 编译 mesh 时自动做惯性主轴重对齐(mesh_quat ≠ identity),
74 块各自被旋转到不同朝向,直接当碰撞体不可用。两条出路:
1. geom 加 mesh_quat 逆补偿(已试,需精调符号;补偿脚本在 git 历史)
2. 每 mesh 单独放一个 body 并显式写 inertial+geom pose(工作量大但确定可行)

## 查看地图(截图见 docs/assets/rmuc2026_field_views.png)

![field](../docs/assets/rmuc2026_field_views.png)
(左:俯视轮廓——双 C 形环形高地、资源岛、启动区清晰可辨;右:等距 3D)

交互查看:
```bash
cd models/rmuc2026 && python3 -m mujoco.viewer --mjcf=scene.xml
```

## ⚠️ 渲染问题(新发现, 2026-09-10)

MuJoCo 3.13 对本场地 STL(195k 面)的**渲染失败**:碰撞数据编译正常(195k 面在),
但视觉顶点 buffer 未建立 → viewer/Renderer 只画包围球(那颗"球"的真身)。
3.12 渲染 3k 面小场正常, 推断与面数/顶点 buffer 上限或 3.13 回归有关。

规避: 分块加载(每个 chunk <20 万面单独 asset, 已有 74 块分块文件)或降级 mujoco 3.12。

## 现状可用的替代
- rough 任务:程序化高度场(台阶/坡面)已可用
- 档位 A/B 部署验证:toy 场地即可
