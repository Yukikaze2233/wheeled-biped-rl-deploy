# RMUC2026 场地模型(官方 STEP 转换)

来源:官方论坛开源帖(bbs.robomaster.com/article/814728)RMUC2026_V2.0.0.stp(1.25GB)

## 已完成
- STEP → OCP 拆解:1025 个 solid,取 top-60 大件(阈值 5e7 mm³)
- 完整网格:74 块 STL / 2,892,422 面 / 零降采样(chunks/ + visual/)
- 尺寸:29.75 × 16.0 × 1.68 m(与规则手册一致)
- 转换工具:tools/step_to_field.py(--chunk-dir 分块模式)

## 未解决(诚实挂账)
MuJoCo 编译 mesh 时自动做惯性主轴重对齐(mesh_quat ≠ identity),
74 块各自被旋转到不同朝向,直接当碰撞体不可用。两条出路:
1. geom 加 mesh_quat 逆补偿(已试,需精调符号;补偿脚本在 git 历史)
2. 每 mesh 单独放一个 body 并显式写 inertial+geom pose(工作量大但确定可行)

## 现状可用的替代
- rough 任务:程序化高度场(台阶/坡面)已可用
- 档位 A/B 部署验证:toy 场地即可
