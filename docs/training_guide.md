# MineParkingNet 训练与人工调优指南

## 冻结的 v7 net-v1 流程

正式入口是 `configs/training/net-v1.yaml`。当前流程固定为：

1. seed 派生的逐 epoch 样本级 shuffle；
2. Adam，学习率 `3e-4`；
3. 轨迹 mask MSE + 权重 `0.2` 的累计停止 BCE；
4. teacher forcing 从 `1.0` 经 15 epoch 线性降至 `0.2`；
5. 前 15 轮只保存和监控、不累计 early-stopping patience；teacher forcing 在第 16 轮达到 `0.2`，从该轮起以 val loss、`patience=5` 早停；
6. 恢复 best 后仅在 val 上扫描停止阈值，保留 `best.pt`，另存不可恢复训练的 `deployment.pt`；
7. test 只用于最终验收，不参与学习率、阈值或 checkpoint 选择。

选择这套流程的 v7 同 seed 消融证据：

| 方案 | 最佳 Val ADE/FDE | 停止长度结论 | 决策 |
|---|---:|---:|---|
| 单终点全平衡、`1e-3` | 0.585/1.109m | 阈值扫描最佳 MAE 27.04 点；系统性提前 | 历史对照 |
| 累计停止、`1e-3`、4 epoch | 0.762/1.316m | 停止覆盖约 45%，验证振荡 | 淘汰 |
| 累计停止、`3e-4`、12 epoch | 0.312/0.552m | val 校准阈值 0.20，MAE 23.84 点、偏差 -8.61 点 | 采用 |

采用方案的 S9 test 为 ADE/FDE 0.293/0.527m、长度 MAE 26.83 点；test 未用于调参。

## 产物和恢复

- `last.pt`：中断续训入口，必须与训练语义字段完全一致。
- `best.pt`：val loss 最佳的原始训练 checkpoint，可复盘，不带部署阈值校准。
- `deployment.pt`：从 best 复制并写入 val 校准阈值，只用于推理、分组分析和闭环，不可续训。
- `history.json`：逐 epoch loss、teacher-forcing 比例、自由滚动 ADE/FDE、停止率、长度 MAE和早停阶段。
- `stop_threshold_calibration.json`：完整阈值网格、选择目标和同分决胜证据。
- `training_curve.png/.pdf`：loss、自由滚动、课程/早停阶段和停止质量。

## 正式验收

先看 val，再只执行一次 test。推荐准入线：

| 指标 | 推荐准入 |
|---|---:|
| Val ADE | ≤ 0.40m |
| Val FDE | ≤ 0.80m |
| Val 长度 MAE | ≤ 30 点 |
| Val 长度绝对偏差 | ≤ 20 点 |
| Val 停止命中率 | ≥ 95% |
| Test ADE/FDE | 不显著差于 val，且分别 ≤ 0.50/0.90m |

同时检查分组；T3 与 S6 可以高于总体，但若 T3 FDE 超过 1.5m 或 S6 FDE 超过 1.8m，应先查看预测—专家叠加图，不以总体均值掩盖长轨迹失败。

## 人工调优顺序

一次只改变一个轴，固定数据、seed、split 和其余配置，并写入新的输出目录。

1. **先判断学习率。** val ADE/FDE 连续大幅上下跳动时，将 `3e-4` 降到 `1.5e-4`；稳定下降但 15 epoch 后仍明显欠拟合时才试 `5e-4`。不要同时改停止损失。
2. **再判断课程。** early stopping 必须满足 `early_stopping_start_epoch >= teacher_forcing_decay_epochs`。自由滚动在 teacher forcing 降低时持续恶化，可把衰减期从 15 延长到 20，不先提高模型容量。
3. **最后判断 patience。** 课程完成后曲线仍周期性刷新 best 时，从 5 增至 7；连续 5 轮轨迹与长度指标同时恶化时，不用盲目增加 epoch。
4. **停止阈值不手填。** 使用 val 自动校准结果；若最优阈值落在扫描边界 `0.05/0.95`，说明停止头未学好，应修训练而非扩大阈值范围。
5. **损失权重保持 `0.2`。** 只有轨迹指标合格而长度 MAE 连续两次独立运行均不合格，才单独比较 `0.1/0.3`；禁止重新启用单终点全平衡并与其他参数同时变化。

每次实验至少记录：配置路径、输出目录、best epoch、Val ADE/FDE、长度 MAE/偏差、校准阈值、T3/S6 分组和是否使用 test 作选择。test 一旦用于选参数，就必须换新的最终保留集，不能继续声称原 test 是无偏验收。
