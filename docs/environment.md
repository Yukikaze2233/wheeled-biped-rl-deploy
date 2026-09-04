# 部署环境搭建

部署侧两档环境:轻量 sim2sim(任意机器)+ ROS2 全链路(Ubuntu)。
策略 ONNX 由训练服务器产出后,先过轻量档验证行为,再上 ROS2 档,最后真机。

## 档位 A:轻量 sim2sim(任意 Linux/本机,5 分钟)

```bash
pip install mujoco onnxruntime numpy

# 校验策略合同(训练服务器拉回的 ONNX)
python3 tools/check_onnx_contract.py policy.onnx

# 全链路回放(玩具模型验证 loop;换自有 MJCF 做真实 sim2sim)
MUJOCO_GL=disable python3 sim2sim/mujoco_sim2sim.py \
    --model models/toy_wheeled_biped.xml --policy policy.onnx --duration 5 \
    --vx 0.5 --obs-delay-ticks 10 --action-delay-ticks 5 --noise-std 0.05
```

无 ROS 依赖;改代码后最快速的行为验证通道(RTF ~60×)。

## 档位 B:ROS2 全链路(与真机同构)

### 依赖

| 项 | 版本 |
|---|---|
| 系统 | Ubuntu 22.04 |
| ROS2 | Humble(Desktop) |
| 构建 | colcon、CMake ≥3.16、C++17 |
| 物理 | MuJoCo 3.5 |
| 推理 | ONNX Runtime 1.20.0 CPU |
| 其他 | libglfw3-dev、rosdep |

### 安装与构建

```bash
# ROS2 Humble(官方源)
sudo apt install software-properties-common
sudo add-apt-repository universe && sudo apt update
sudo apt install ros-humble-desktop ros-dev-tools
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# 本仓库
git clone https://github.com/Yukikaze2233/wheeled-biped-rl-deploy.git
cd wheeled-biped-rl-deploy
./scripts/bootstrap.sh            # 若存在;否则手动装 MuJoCo 3.5 + ORT 1.20:
#   MuJoCo:  解压到 /opt/mujoco 并 export MUJOCO_ROOT=/opt/mujoco
#   ORT:     解压 libonnxruntime 到 /opt/onnxruntime 并 export ONNXRUNTIME_ROOT=...

# rosdep + 构建
sudo rosdep init && rosdep update
cd ros2 && rosdep install --from-paths src --ignore-src -y
colcon build
source install/setup.bash

# 启动(sim 后端)
ros2 launch wheeled_bringup bringup.launch.py backend:=sim
# 另一终端发指令
ros2 run wheeled_teleop teleop.py
```

### 编译开关

| 缺库时 | 行为 |
|---|---|
| 无 ONNX Runtime | 控制器编译通过但推理关闭(零动作保持)——可先验证 FSM/PD 链 |
| 无 MuJoCo | mujoco_system 编译为惰性骨架——只验证 controller 层 |

## 档位 C:真机

前置:完成 `docs/sim2real.md` 全部核对清单 + real2sim 辨识。

1. **帧对齐**:与固件约定 `real_system.hpp` 头注释的帧布局(字节序/CRC)
2. **环回验证**:逻辑分析仪/固件回环确认帧收发,再上电机
3. **五阶段上电**:PREPARE 静态 → RL 架空 → 降系数落地 → 低速 → 放开
   (详见 docs/sim2real.md §D)
4. 推理宿主二选一:
   - 上位机路线:本 ROS2 栈 + RealBridge 串口(延迟 ~20ms 级)
   - 下位机路线:STM32H7 + CubeAI(把 ONNX 转 C 编进固件,砍掉串口链路延迟)

## 环境对照速查

| 你要做什么 | 用哪档 | 在哪台机器 |
|---|---|---|
| 训练策略 | 训练环境(见训练仓 docs/server_setup.md) | GPU 服务器 |
| 看策略行为对不对 | 档位 A | 任意机器 |
| 验证部署链路/调 PD | 档位 B | Ubuntu + ROS2 |
| 上真机 | 档位 C | 实车 + 固件 |
