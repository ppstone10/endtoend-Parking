# MineParkingNet Spec

## 元数据

- Spec ID 前缀：`MODEL`
- 强度：轻量
- 状态：已采纳
- 最后更新：2026-08-21

## 目标

- 端到端生成未来 N 个局部轨迹点：输入 BEV 环境表示 + 目标泊车位姿 + 车辆运动状态，输出轨迹。
- 提供数据落盘/加载与训练流程，使阶段五 MPC 闭环可加载训练好的网络。

## 非目标

- 不做控制量输出（油门/转向），底层控制由 MPC 完成。
- 不实现多模态轨迹、障碍物避让约束的显式建模。
- 本阶段不要求真实泊车成功率，只要求训练可收敛。

## 边界与约束

- 网络输出为车辆中心**局部坐标**轨迹点；训练数据（专家轨迹、目标）由全局坐标转换到起始局部系后喂入。
- BEV 通道数固定为融合后 5 通道（occupancy/height/density/target/vehicle）。
- 依赖 PyTorch（CPU 版即可运行），Python 3.12 conda 环境。
- 轨迹长度以最长样本为准补零，用 mask 屏蔽无效点。

## 行为与验收

### `MODEL-NET-001`：轨迹生成前向

- 前置：输入 BEV `(B,C,H,W)`、目标位姿 `(B,3)`、运动状态 `(B,2)`（v,omega）。
- 行为：CNN 编码 BEV 展平，与条件拼接经 MLP 回归。
- 结果：输出 `(B, N, 3)` 局部轨迹点。
- 验收：`tests/test_model.py::TestMineParkingNet` 形状断言通过。

### `MODEL-LOSS-001`：掩码 MSE 损失

- 前置：预测、目标 `(B,N,3)` 与掩码 `(B,N)`。
- 行为：对有效点计算 MSE 并按有效点数平均。
- 结果：返回标量损失。
- 异常与恢复：掩码全零时按 clamp(min=1) 避免除零。
- 验收：`tests/test_model.py::TestLossFn` 通过。

### `MODEL-TRAIN-001`：训练收敛

- 前置：小批量随机数据。
- 行为：Adam 优化 30 步后损失应下降。
- 结果：`loss_fn` 在训练后低于初始值。
- 验收：`tests/test_model.py::TestTrainingConvergence` 通过；`scripts/train.py --samples 20 --epochs 20` loss 从 3.38 降至 0.021。

### `MODEL-DATA-001`：数据落盘与加载

- 前置：样本列表。
- 行为：`DatasetGenerator.save` 写入 npz（bevs/goals/states/trajs/masks/dt），`load` 读回字典。
- 结果：字段完整且轨迹按最长补零。
- 验收：`scripts/train.py` 数据集生成后可加载训练。

## 追溯

| Spec ID | 验收 | 测试或人工入口 | 实现符号 | 实际验证 | 状态 |
|---|---|---|---|---|---|
| `MODEL-NET-001` | 前向形状 | `tests/test_model.py::TestMineParkingNet` | `model/network.py::MineParkingNet` | unittest 通过 | ✅ |
| `MODEL-LOSS-001` | 掩码 MSE | `tests/test_model.py::TestLossFn` | `model/network.py::loss_fn` | unittest 通过 | ✅ |
| `MODEL-TRAIN-001` | 损失下降 | `tests/test_model.py::TestTrainingConvergence` + train.py | `scripts/train.py` | loss 3.38→0.021 | ✅ |
| `MODEL-DATA-001` | 落盘往返 | `scripts/train.py` 数据链路 | `dataset/generator.py::save/load` | train.py 生成并加载成功 | ✅ |

## 待人工确认

- 无。