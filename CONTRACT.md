# 策略部署合同(Frozen Policy Contract)

训练端(`isaac_wheeled_rl_train`)、sim2sim(`sim2sim/`)、ROS2 部署端(`ros2/`)
三方共同遵守的唯一接口定义。**修改本文件必须同步训练 cfg 与两处部署代码。**

参照 华南虎(SCUT)赛季方案 V14 flat 合同设计,索引顺序与缩放与官方训练 cfg 一致。
本文件内容以 scut-wheeled-legged-rl/pretrained/26_infantry/flat_and_rotation/.../params/env.yaml
(官方训练 dump)与 agent_tasks/.../wheelbipe25_v3/env.py 的代码语义为唯一权威,
已在 MuJoCo sim2sim 中按此合同跑通官方预训练 ONNX(sim2sim/mujoco_sim2sim.py)。

## 1. 模型 I/O

| 项目 | 值 |
| --- | --- |
| 输入 | `obs` `float32[1,35]` |
| 输出 | `actions` `float32[1,6]` |
| 推理频率 | 50 Hz |
| 控制频率 | 500 Hz(策略输出在 10 个控制周期内保持) |
| 控制器默认输出模式 | `hardware_pd_vel`(腿位置 PD + 轮速度) |

ONNX Runtime 启动时严格校验:单输入单输出、名称、dtype、静态 shape。
校验工具:`tools/check_onnx_contract.py`。

## 2. 35D 观测索引

| 索引 | 含义 | 缩放 |
| ---: | --- | ---: |
| 0 | 前进速度指令 vx | ×1.0 |
| 1 | 横向速度指令 vy(本合同恒 0) | ×1.0 |
| 2 | 偏航角速度指令 wz | ×1.0 |
| 3 | 目标高度 | ×5.0 |
| 4–6 | 机体角速度 x/y/z(陀螺仪) | ×0.5 |
| 7–9 | 机体系投影重力 x/y/z | ×1.0 |
| 10–13 | 4 个主动腿关节位置 − 默认位置,顺序 = [左后, 右后, 左前, 右前] | ×1.0 |
| 14–15 | 轮位置槽(恒 0) | — |
| 16–19 | 腿关节速度(顺序同上:左后, 右后, 左前, 右前) | ×0.1 |
| 20–21 | 轮关节速度(左, 右) | ×0.1 |
| 22–27 | 上一推理周期 6 维策略动作 | ×1.0 |
| 28 | `normal` 模式标志 | 固定 1(flat 任务) |
| 29–34 | stair/slope/recover/jump/height_target/state_time | 固定 0 |

全观测默认 clamp `[-100, 100]`,nan/inf 置 0。

## 3. 6D 动作与执行器

| 动作 | 含义 | 解码 | 输出限幅 | Kp | Kd |
| ---: | --- | --- | --- | ---: | ---: |
| 0–3 | 腿关节位置目标,顺序 = [左后, 右后, 左前, 右前] | `default_pos + 0.5 × action` | ±40 Nm(力矩侧) | 60 | 2 |
| 4–5 | 轮速度目标(rad/s),顺序 = [左, 右] | `clamp(10 × action, ±150)` | ±5 Nm | 0 | 0.2 |

- 轮速模式缩放为 `wheel_vel_action_scale = 10.0`,速度钳位 `max_wheel_vel = 100 × 1.5 = 150 rad/s`
  (官方 env.py 在 `use_wheel_vel_control=True` 时覆盖;`wheel_action_scale=1.0` 仅用于力矩模式)。

- `default_dof_pos` 取训练资产默认站姿(官方 V14 全部关节 ref=0)。
- 弹簧关节不由策略驱动;部署端按训练曲线做前馈:
  `F = 400 + (600−400)/0.07 × clamp(0.06076 − q, min=0)` N(无上钳位,训练无阻尼;
  spring_settings: linear_up=600, linear_down=400, linear_length=0.07, spring_offset=0.06076)。
- 云台:训练中 pitch 由位置伺服固定在 0(Kp=20, Kd=0.5),yaw 仅阻尼(Kd=0.5)自由摆动;
  部署模型若缺 pitch 关节等价于刚性固定,可接受。
- 高度指令语义:训练 `use_absolute_height=True`,obs[3] = 底座绝对高度 z × 5.0,
  flat 默认 0.22 m(训练高度指令范围 0.20–0.42)。
- 训练中 obs/act 延迟常开(obs 20–80 ms,act 20–60 ms,逐 env 随机);部署端必须复现:
  sim2sim 默认 `--obs-delay-ticks 4` / `--action-delay-ticks 3`(80/60 ms),
  ROS2 控制器参数 `obs_delay_steps=4` / `act_delay_steps=3`。
  实测(官方 13k flat_and_rotation 策略,vx=1.0):0 延迟约 0.19 m/s 且极限环,
  80/60 ms 延迟 0.935 m/s —— 延迟不是可选项,是合同的一部分。
- PREPARE 阶段(上电):四腿目标插值到 0 rad,最大 1 rad/s。

## 4. 观测扰动(部署端可配)

| 参数 | 默认 |
| --- | --- |
| obs/action 延迟 | obs 80 ms(4 步)/ act 60 ms(3 步),实测最优 |
| 噪声注入 | 关闭 |

## 5. 状态机

`0=INIT → 1=IDLE → 2=PREPARE → 3=RL`。flat 合同不新增状态 ID;
7D 模式标志是策略观测合同的一部分,不是状态机的输入。
