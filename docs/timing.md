# 控制时序详解

500 Hz 控制 / 50 Hz 推理的双频循环,以及它与训练端时序的对齐关系。

## 全链路时序

```text
训练端(Isaac Lab)                    部署端(ros2_control / sim2sim)
─────────────────────                ─────────────────────────────
物理步 200 Hz (dt=5ms)                控制 tick 500 Hz (dt=2ms)
  └─ decimation ×4                      └─ 每 10 tick 一次推理
策略推理 50 Hz (20ms)        ◄──对齐──►   策略推理 50 Hz (20ms)
动作保持 10 个控制周期                     动作保持 10 个控制周期
```

两侧策略频率完全一致(50 Hz);物理频率差异(200 vs 500 Hz)只影响底层 PD
闭环带宽——部署端更高的控制频率让 PD 更接近连续,这个方向的不对称是安全的。

## 推理 tick 内发生什么(`rl_controller.cpp::update`)

```text
tick++
  ├─ 读 16 路状态接口(关节位置/速度、IMU 陀螺、投影重力)
  ├─ FSM 分支:
  │    INIT/IDLE → 零指令(硬件阻尼保持),锁存当前姿态为 PREPARE 起点
  │    PREPARE   → 腿目标以 ≤1 rad/s 向默认位姿插值,到位(±0.02rad)自动转 RL
  │    RL        → 每 10 tick:拼 35D obs → ONNX 推理 → 6D action
  │                 其余 tick:沿用上次 action
  └─ 写 6 路命令接口(4×腿位置目标 + 2×轮速度目标)
```

## 动作语义("hardware_pd_vel")

控制器只输出**目标**,不输出力矩——PD 闭环在硬件层闭合:

| 通道 | 目标 | 硬件层闭环 |
|---|---|---|
| 腿 ×4 | 位置:default + 0.5 × action | Kp=60, Kd=2(固件或 MuJoCo 插件) |
| 轮 ×2 | 速度:clamp(10 × action, ±150 rad/s) | 速度环 0.2,|τ|≤5 Nm |

(轮速模式官方缩放 `wheel_vel_action_scale=10`、钳位 `max_wheel_vel=100×1.5=150`,
腿 PD 力矩钳位 ±40 Nm——见 CONTRACT.md 第 3 节。

理由:固件闭环频率远高于上位机,且串口链路抖动被 PD 吸收;力矩直通模式
(`torque`)保留为可选项但默认不用。

## 延迟预算

```text
观测路径:  IMU/编码器 → 状态接口 → 拼装      (固定)
推理:      ONNX Runtime CPU,单线程            (~1ms 量级)
动作路径:  保持 10 tick(20ms,设计使然)      (训练时已随机化 20–60ms)
```

延迟**不是可选项**:训练时 obs 20–80 ms / act 20–60 ms 常开,0 延迟会
显著改变策略闭环(实测 13k 策略 vx=1.0:0 延迟 ≈0.19 m/s 且极限环,
80/60 ms 延迟 0.935 m/s)。sim2sim 默认 `--obs-delay-ticks 4`(80 ms)、
`--action-delay-ticks 3`(60 ms);ROS2 控制器参数 `obs_delay_steps=4`、
`act_delay_steps=3`(每步 20 ms)。实机链路延迟实测后在此范围附近微调;
超出范围的链路改动(如加感知前级)需要重训。

## PREPARE 状态的意义

上电时关节可能在任意位置,直接进 RL 会让策略吃到合同外的观测分布。
PREPARE 以受限速度(1 rad/s)把腿带到默认位姿,保证 RL 首帧观测落在
训练分布内。这也是真机安全链的一部分:任何异常回落 IDLE 即零力矩阻尼。
