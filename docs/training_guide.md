# MineParkingNet 训练与人工调优指南

## 冻结的 v7 net-v1 基线流程

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

## v9 连续净空安全与闭环状态微调

`configs/training/net-v1-safe.yaml` 从已准入的 v7 `best.pt` 只加载模型权重，不继承 optimizer、epoch 或 patience，再使用两个有明确职责的输入进行安全微调：

1. `scripts/build_recovery_dataset.py` 在原 train 任务中运行当前 deployment。普通分布偏移样本按固定步长及位置偏离 ≥0.25m 或航向偏离 ≥5° 选取；发生碰撞时不使用紧邻碰撞帧，而是沿全部已执行状态回溯到距离碰撞最近、且满足专家规划安全余量的状态，再由专家规划到原 selected goal。回溯范围和停止条件由轨迹历史与统一碰撞检查契约决定，不增加可调窗口。
2. `collision_loss_weight=0.5` 对预测轨迹的完整 6×3m 车体和相邻点连续扫掠计算连续净空损失：由 occupancy 和地图边界预计算截断有符号距离场，在数据 `collision_margin` 加额外 0.1m 训练余量内惩罚二次净空缺口。车辆尺寸和 BEV 几何必须从 schema v2 元数据解析，不能在场景代码中硬编码；旧 `occupancy_max` 只用于消融复现。

恢复采集目录的 `.checkpoints/identity.json` 绑定数据文件、计划指纹、deployment SHA-256、车辆模型、选择参数、优先任务证据和基础恢复归档；每个源任务有独立完成标记，原命令可续建，身份变化必须换输出目录。首轮之后使用 `--priority-from` 自动选择上一轮“碰撞或超时且恢复为零”的源任务，使用 `--base-recovery` 保留既有样本；合并时以“来源索引+闭环步”拒绝重复。`report.json` 同时记录固定步长、紧邻碰撞帧和完整安全余量回溯的同回放覆盖率。

当前冻结数据证据：首轮 240 回合生成 243 条恢复样本，但 18 个碰撞回合有 15 个零恢复。定向补采这 15 个任务时，固定步长安全候选与紧邻碰撞帧均覆盖 0/15，完整安全余量回溯覆盖 15/15，且 15/15 专家重规划成功；回溯距离为 2–17 个控制步。最终恢复集 258 条、合并训练集 2658 条，位于 `data/task_dataset/tracked_pivot_v8_recovery_final/`。该选择策略已经定型，不再通过修改偏离阈值或回溯参数重复试验。

安全训练固定使用推理阶段 teacher forcing 比例 `0.2`，最多 15 轮并从第 4 轮启用 `patience=5`。`initialize_from` 与 `resume_from` 互斥：前者是新运行的权重初始化，后者才是同一训练语义的断点续训。`history.json` 和曲线额外记录未乘权重的 train/val collision loss；checkpoint 绑定初始化身份、安全模式、权重、采样参数和从数据解析出的车辆/BEV 几何。

三轮同 seed 消融使用同一 v7 初始化、最终恢复集和 val：旧损失＋三池均衡的统一净空损失为 1.306765；净空＋真实占比 shuffle 为 0.115092、ADE/FDE 0.228/0.572m；净空＋三池均衡为 0.128024、ADE/FDE 0.263/0.590m；冻结 v7 基线为 1.102705。正式配置因此采用净空＋真实占比 shuffle（安全损失下降 89.6%），不强制过采样恢复小类；三池确定性采样能力只保留给后续独立消融。报告位于 `runs/training/v9-safety-ablation/ablation_report.json`。

正式训练后先用 `scripts/evaluate_safety_ablation.py` 对 v7 与 v9 `best.pt` 统一复算净空损失，再按相同样本索引分别执行 `--safety-mode none` 和 `expert_fallback`。运行时门禁先审查整段轨迹，再在每个控制真正执行前检查下一状态扫掠；不安全时仍从当前安全状态回退，回退控制仍不安全则以 `safety_stop` 结束单回合，不中断批量评测。模型自身运行异常不会伪装成安全停止。

后续调优保持单变量：如果正式 v9 的统一净空损失未比 v7 至少下降 10%，或 ADE/FDE 超过 0.40/0.80m，停止闭环准入并检查数据/梯度，不反复改回溯阈值。只有开环与净空均通过但纯网络碰撞率仍高，才单独比较 `collision_loss_weight=0.25/1.0`；不得同时改模型容量、恢复采样和控制参数。带专家回退的成功率不能替代纯网络能力。

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

进入安全训练后增加闭环准入：纯网络分层集目标为成功率 ≥70%、碰撞率 ≤10%；安全门禁回退失败必须为 0，碰撞率目标 ≤5%。两种报告必须使用同一数据、索引、控制 seed 与重规划周期。干预率只作为网络安全缺口指标，不把被专家接管的回合计为网络独立成功。

同时检查分组；T3 与 S6 可以高于总体，但若 T3 FDE 超过 1.5m 或 S6 FDE 超过 1.8m，应先查看预测—专家叠加图，不以总体均值掩盖长轨迹失败。

## 人工调优顺序

一次只改变一个轴，固定数据、seed、split 和其余配置，并写入新的输出目录。

1. **先判断学习率。** val ADE/FDE 连续大幅上下跳动时，将 `3e-4` 降到 `1.5e-4`；稳定下降但 15 epoch 后仍明显欠拟合时才试 `5e-4`。不要同时改停止损失。
2. **再判断课程。** early stopping 必须满足 `early_stopping_start_epoch >= teacher_forcing_decay_epochs`。自由滚动在 teacher forcing 降低时持续恶化，可把衰减期从 15 延长到 20，不先提高模型容量。
3. **最后判断 patience。** 课程完成后曲线仍周期性刷新 best 时，从 5 增至 7；连续 5 轮轨迹与长度指标同时恶化时，不用盲目增加 epoch。
4. **停止阈值不手填。** 使用 val 自动校准结果；若最优阈值落在扫描边界 `0.05/0.95`，说明停止头未学好，应修训练而非扩大阈值范围。
5. **损失权重保持 `0.2`。** 只有轨迹指标合格而长度 MAE 连续两次独立运行均不合格，才单独比较 `0.1/0.3`；禁止重新启用单终点全平衡并与其他参数同时变化。

每次实验至少记录：配置路径、输出目录、best epoch、Val ADE/FDE、长度 MAE/偏差、校准阈值、T3/S6 分组和是否使用 test 作选择。test 一旦用于选参数，就必须换新的最终保留集，不能继续声称原 test 是无偏验收。
