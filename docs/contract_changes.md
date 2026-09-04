# 合同变更指南

CONTRACT.md 是 35D→6D 接口的唯一权威。本文说明为什么合同必须冻结、
以及确实要改时的完整流程。

## 为什么合同要冻结

训练端(torch 图)与部署端(ONNX Runtime / C++)是两个独立实现,中间只靠
obs 逐维的索引、缩放、符号约定对齐。任何一维不一致都是实机行为异常:
仿真里看不出来,上真机直接表现为抖动、方向颠倒或姿态漂移。
历史经验:大部分 sim2real bug 都在合同层,不在算法层。

## 改合同的触发条件

| 想改的东西 | 是否需要动合同 |
|---|---|
| reward 权重、课程、DR 范围、算法分支 | ❌ 纯训练侧 |
| critic 维度(特权观测) | ❌ critic 不参与部署 |
| 新增策略输入(如新增传感器流) | ✅ |
| 新增执行器/动作维 | ✅ |
| 改缩放/符号/clamp | ✅ |
| 改控制/推理频率 | ✅ |

## 变更流程(五处同步,缺一不可)

1. **`CONTRACT.md`** 改索引表,版本号 +1,变更行加注
2. **训练端** `env_cfg.py`(scale 常量 + `env.py` cat 顺序)同步修改
3. **sim2sim** `mujoco_sim2sim.py::build_obs` 索引与缩放同步
4. **ROS2 控制器** `robot_state.hpp::assemble_observation` 同步
5. **校验器** `check_onnx_contract.py` 的 obs-dim / action-dim 默认值(如有变化)

然后:

```bash
# 训练侧自检
PYTHONPATH=src python tests/...            # 相关套件
# 导出新模型并校验
./isaaclab.sh -p scripts/export_onnx.py --checkpoint ... --output policy.onnx
python3 tools/check_onnx_contract.py policy.onnx --obs-dim <新维度>
# sim2sim 曲线核对后再上真机
```

## 历史变更记录(模板)

| 版本 | 变更 | 原因 |
|---|---|---|
| 1.0 | 初版:35D→6D,50/500 Hz | — |
