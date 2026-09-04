# 测试与验证指南

两个仓库各自的验证体系、本机跑法、以及"改代码后要跑什么"的对照表。

## 训练仓库(7 套件,全本机可跑)

```bash
cd isaac_wheeled_rl_train
export PYTHONPATH=src:tests        # 依赖:torch(CPU)+ rsl-rl-lib==2.3.3 + numpy

python tests/test_mdp.py              # 延迟 ring 语义 + 指令采样器(桶互斥/迭代门控)
python tests/test_env_features.py     # 腾空 FSM 转换 + 课程推进 + 地形 flag 映射
python tests/test_season_features.py  # 电机曲线 + 跳跃轨迹族 + 指令覆盖 + GRU 分支
python tests/test_algorithms.py       # HIM/DreamWaQ/NP3O toy 收敛断言(~3 min)
python tests/test_rsl_rl_smoke.py     # 官方 rsl_rl 栈冒烟 + checkpoint 往返
python tests/test_experiments.py      # 实验注册表/续训/对比 CLI(~2 min)
```

### 断言 philosophy

- **行为断言,非实现断言**:delay 测试验证"lag=k 返回 k 步前的帧"这一语义,
  ring/猫拼接是实现细节,换实现测试不变
- **收敛断言**:算法测试要求 rollout 奖励前后改进量超阈值——算法"能跑"不算过,
  "能学"才算过
- **性质断言**:轨迹 h(0)=h0 / h(T)=hf / h'(T)=0、扭矩限幅随速度 droop、
  桶互斥——数学性质逐条验证

### toy 环境

`toy_env.py`(rsl_rl 协议)与 `toy_env_ext.py`(扩展历史/特权流)模拟 IsaacLab
VecEnv 语义(内部自动 reset),是算法层与仿真解耦的关键——算法分支对
toy 与 Isaac Lab 环境不可区分。

## 部署仓库(3 个验证入口)

```bash
cd isaac_wheeled_rl_deploy
pip install onnxruntime mujoco numpy

# 1) 合同校验(任何 ONNX 上真机前必过)
python3 tools/check_onnx_contract.py policy.onnx
#   校验:单输入输出、名称 obs/actions、float32、静态 [1,35]→[1,6]、
#         零观测前向有限性(发散策略红旗)

# 2) sim2sim 全链路(不需要 ROS,玩具模型验证 loop;RTF ~60x)
MUJOCO_GL=disable python3 sim2sim/mujoco_sim2sim.py \
    --model models/toy_wheeled_biped.xml --policy policy.onnx --duration 5 \
    --vx 0.5 --obs-delay-ticks 10 --action-delay-ticks 5 --noise-std 0.05

# 3) 语法/编译(C++ 需 ROS2 环境)
python3 -m py_compile tools/*.py sim2sim/*.py
cd ros2 && colcon build --packages-select ...
```

## 改动 → 必跑测试对照表

| 改了什么 | 必跑 | 理由 |
|---|---|---|
| mdp/ 下任何组件 | test_mdp + test_env_features | 组件语义 |
| 算法分支 / ppo_base | test_algorithms + test_season_features + test_rsl_rl_smoke | 收敛性 |
| experiments/registry | test_experiments | 注册/续训/对比链路 |
| env.py 观测拼装 | 全部 + 部署 CONTRACT 对照 | 合同冻结 |
| 延迟/时序 | test_mdp + sim2sim 延迟参数 | 双端时序一致 |
| 部署 controller | check_onnx_contract + sim2sim | 合同与循环 |

## CI 建议(未建)

最小集:训练仓 7 套件(约 8 分钟 CPU)+ 部署仓 py_compile + 合同校验
(用仓库外任一 35D→6D ONNX)。sim2sim 冒烟需要 mujoco pip 包,可缓存。
