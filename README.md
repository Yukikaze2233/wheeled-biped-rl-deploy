# isaac_wheeled_rl_deploy

轮足 RL 策略的部署与 sim2sim 验证仓库,**架构对齐赛季方案**:ROS2 + ros2_control,
控制器/hardware 插件/bringup 三层分离,sim(MuJoCo)与 real(串口)只换 hardware
插件,上层合同不动。训练端(`isaac_wheeled_rl_train`)导出 ONNX 后经本仓库完成
合同校验 → sim2sim → 真机部署。

## 创新点(部署侧)

- **sim/real 同构**:仿真与真机只是 hardware 插件不同(`backend:=sim|real`),
  控制器/合同/FSM 零改动——上层永远不知道下面是谁;
- **合同第一道闸**:任何 ONNX 上真机前必须过 `check_onnx_contract.py`
  (名称/shape/dtype/零输入前向四道检查);
- **延迟可复现**:sim2sim 暴露 `--obs-delay-ticks/--action-delay-ticks/--noise-std`,
  实机实测延迟可先在仿真里复现再排查;
- **排查方法论文档化**:sim2real 排查表按"症状→首查/次查"组织(见 docs/sim2real.md)。

## 结构

```text
CONTRACT.md                       # 冻结的 35D→6D 接口合同(唯一权威)
tools/check_onnx_contract.py      # ONNX 合同校验器(名称/shape/dtype/零输入前向)
sim2sim/mujoco_sim2sim.py         # 轻量 sim2sim(不经 ROS):500Hz PD + 50Hz ONNX
models/toy_wheeled_biped.xml      # 玩具模型(仅用于验证部署链路,非真实机器人)
ros2/src/
├── controllers/wheeled_rl_controller/   # 策略控制器(ControllerInterface,C++)
│   ├── include/.../rl_controller.hpp    #   接口声明 + update 循环
│   ├── include/.../state_machine.hpp    #   fsm 模块:INIT→IDLE→PREPARE(1rad/s插值)→RL
│   ├── include/.../robot_state.hpp      #   robot_state 模块:状态缓存 + 35D 拼装
│   ├── include/.../policy_runtime.hpp   #   onnxruntime 模块:ORT 会话封装
│   └── src/{rl_controller.cpp, fsm/, robot_state/, onnxruntime/}
├── interfaces/wheeled_mujoco_system/    # MuJoCo 硬件插件(SystemInterface)
├── interfaces/wheeled_real_system/      # 串口 RealBridge(termios + CRC16 骨架)
├── middlewares/wheeled_bringup/         # launch + ros2_control yaml + xacro(sim/real)
└── tools/wheeled_teleop/                # keyboard teleop(50Hz Twist + 高度)
```

## 架构(赛季方案同款)

```text
                     ros2_control (500 Hz)
  ┌────────────────────────────────────────────────────┐
  │  wheeled_rl_controller (ControllerInterface)       │
  │    50 Hz: 35D obs → ONNX Runtime → 6D action       │
  │    输出: 4×腿位置目标 + 2×轮速度目标("hardware_pd_vel")│
  └──────────────┬─────────────────────┬───────────────┘
                 │ command/state 接口   │
       backend:=sim                    backend:=real
  ┌──────────────▼──────────┐   ┌──────▼────────────────┐
  │ wheeled_mujoco_system   │   │ RealBridge SystemInterface│
  │ (MuJoCo SystemInterface,│   │ (串口协议,集成时提供)      │
  │  插件内闭环 PD + mj_step)│   └───────────────────────┘
  └─────────────────────────┘
```

## 文档

| 文档 | 内容 |
|---|---|
| [docs/contract_changes.md](docs/contract_changes.md) | 合同为什么冻结、变更五处同步流程 |
| [docs/timing.md](docs/timing.md) | 500Hz/50Hz 双频时序与训练端对齐 |
| [docs/sim2real.md](docs/sim2real.md) | 上真机核对清单 + real2sim 辨识 + 逆向排查表 |
| [docs/ros2_architecture.md](docs/ros2_architecture.md) | 三层结构、sim/real 切换、扩展路径 |
| [docs/testing.md](docs/testing.md) | 两仓验证体系与"改动→必跑"对照表 |
| [docs/environment.md](docs/environment.md) | 部署环境三档搭建(A 轻量 sim2sim / B ROS2 / C 真机) |
| 训练仓 [docs/project_tree.md](../isaac_wheeled_rl_train/docs/project_tree.md) | 两仓库完整架构层级树(文件级) |
| 训练仓 [docs/server_setup.md](../isaac_wheeled_rl_train/docs/server_setup.md) | 训练服务器环境搭建 |

## 快速开始

```bash
pip install onnxruntime mujoco numpy

# 1. 校验训练导出的策略(也可用赛季方案发布的 ONNX 验证工具本身)
python3 tools/check_onnx_contract.py policy.onnx

# 2a. 轻量 sim2sim(不经 ROS,迭代最快;玩具模型验证 loop)
MUJOCO_GL=disable python3 sim2sim/mujoco_sim2sim.py \
    --model models/toy_wheeled_biped.xml --policy policy.onnx --duration 5 \
    --vx 0.5 --obs-delay-ticks 10 --action-delay-ticks 5 --noise-std 0.05

# 2b. ros2_control 全链路 sim(与真机同构)
cd ros2 && colcon build
ros2 launch wheeled_bringup bringup.launch.py backend:=sim

# 3. 真机:backend:=real,需补 RealBridge SystemInterface(串口协议)
```

## sim2sim 与真机一致性清单(赛季方案排查方法论)

- [ ] 观测 35D 索引/缩放/符号(投影重力方向)/clamp 一致
- [ ] 动作解码:腿 = 默认角 + 0.5×action;轮 = 10×action,限幅一致
- [ ] PD 参数与训练一致(腿 Kp=60/Kd=2;轮 0.2)
- [ ] 延迟:实机实测回环 → sim2sim 先复现(训练时已随机化 20–80 ms)
- [ ] 推理时序:50 Hz 推理 + 500 Hz 控制,动作保持 10 个控制周期
- [ ] 逐帧对比真机视频与 sim2sim 曲线,定位 gap 环节

## 完成度说明

| 组件 | 状态 |
|---|---|
| ControllerInterface + FSM + 35D 拼装 + ORT 推理 | 代码完整,需链接你的 ORT(HAVE_ONNXRUNTIME) |
| PREPARE 1 rad/s 插值到默认位姿,到位自动进 RL | 完整(step_prepare) |
| MuJoCo SystemInterface(插件内 PD 闭环 + mj_step) | 代码完整,需链接 MuJoCo(HAVE_MUJOCO) |
| wheeled_real_system(串口 RealBridge,termios + CRC16 骨架) | 骨架;帧布局与固件字节对齐待集成 |
| bringup(launch + yaml + xacro,sim/real 切换) | sim 完整;real 由 wheeled_real_system 提供 |
| keyboard teleop(w/s/a/d/r/f/空格/q,50 Hz Twist+高度) | 完整(纯 rclpy) |
| Xbox/DT7 遥控器、观测滤波选项 | 未实现(扩展点) |

## 技术栈

- **框架**:ROS2 Humble + ros2_control(ControllerInterface / SystemInterface / pluginlib)
- **推理**:ONNX Runtime C++(可选编译开关);轻量 sim2sim 用 onnxruntime Python
- **仿真**:MuJoCo C API(硬件插件)+ MuJoCo Python(轻量 sim2sim)
- **时序**:500 Hz 控制 / 50 Hz 推理,与训练端 200 Hz×4 对齐(CONTRACT.md)
