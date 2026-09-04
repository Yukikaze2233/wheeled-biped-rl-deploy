# ROS2 架构与扩展指南

ros2_control 三层结构、sim/real 切换机制、以及常见扩展的开发路径。

## 三层结构

```text
ros2/src/
├── controllers/wheeled_rl_controller/    # 策略层(ControllerInterface)
│   ├── src/rl_controller.cpp             #   接口声明 + 500Hz update 循环
│   ├── include/.../state_machine.hpp     #   fsm 模块:INIT→IDLE→PREPARE→RL
│   ├── include/.../robot_state.hpp       #   robot_state 模块:状态缓存 + 35D 拼装
│   ├── include/.../policy_runtime.hpp    #   onnxruntime 模块:ORT 封装
│   └── src/{fsm,robot_state,onnxruntime} #   模块的翻译单元(模板实例化)
├── interfaces/                            # 硬件层(SystemInterface 插件)
│   ├── wheeled_mujoco_system/            #   sim:MuJoCo 物理 + 插件内 PD 闭环
│   └── wheeled_real_system/              #   real:串口 RealBridge(termios+CRC16)
├── middlewares/wheeled_bringup/          # 组装层:launch + ros2_control yaml + xacro
└── tools/wheeled_teleop/                 # 键盘遥控(50Hz Twist + 高度)
```

## 数据流

```text
                    ros2_control (500 Hz)
  wheeled_rl_controller ──6 命令接口──► hardware 插件 ──► MuJoCo / 串口
        ▲                                     │
        └──────────16 状态接口◄───────────────┘
  (关节位置/速度 ×10、IMU 陀螺 ×3、投影重力 ×3)
```

关键接口命名(`command_interface_configuration` / `state_interface_configuration`):

```text
命令:{left,right}_{front1,rear1}_joint/position,{left,right}_wheel_joint/velocity
状态:同上 position+velocity,外加合成关节 "imu" 的 gyro.{x,y,z} / gravity.{x,y,z}
```

## sim / real 切换

切换只发生在 xacro 的 hardware 插件选择(`backend:=sim|real`),
控制器、合同、FSM、yaml 全部不动:

```bash
ros2 launch wheeled_bringup bringup.launch.py backend:=sim    # MuJoCo
ros2 launch wheeled_bringup bringup.launch.py backend:=real   # 串口
```

这是"仿真与真机同构"的核心:上层永远不知道下面是谁。

## 编译开关

| 开关 | 缺省行为 | 打开后 |
|---|---|---|
| `HAVE_ONNXRUNTIME`(controller)| 零动作保持(控制路径可独立验证) | 策略推理生效 |
| `HAVE_MUJOCO`(mujoco_system)| 惰性骨架(read/write 空转) | 物理仿真生效 |

均通过 CMake 的 find_package/pkg-config 自动探测,或显式
`-DONNXRUNTIME_ROOT=...` / `-DMUJOCO_ROOT=...`。

## 扩展路径

### 换机器人
1. MJCF/URDF 按合同关节命名
2. `mujoco_system.cpp` 的 LEG_JOINTS / WHEEL_JOINTS 常量
3. xacro 的 ros2_control 段逐关节声明
4. CONTRACT.md 关节表同步

### 接真机(完成 wheeled_real_system)
1. 与固件对齐帧布局(real_system.hpp 头注释有建议布局)与字节序
2. 实现 `exchange_frames()`:组帧 + CRC 校验 + 解包到 state 缓存
3. yaml/xacro 的 real 分支替换 NOT_IMPLEMENTED 占位
4. 先在环回/逻辑分析仪验证帧,再上电机

### 加观测(如新增 IMU 加速度计)
1. hardware 插件 export 新状态接口
2. 控制器 state_interface_configuration + RobotStateBuffer 扩维
3. **CONTRACT.md 升版**(五处同步,见 docs/contract_changes.md)

### 加控制器模式(如力矩直通)
- 命令接口已有 position/velocity 二分,新增 `effort` 接口声明即可;
  合同的输出模式字段同步
