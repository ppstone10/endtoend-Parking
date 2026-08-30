# MineParkingNet Spec

## 元数据

- Spec ID 前缀：`MODEL`
- 强度：完整
- 状态：已采纳
- 最后更新：2026-08-27

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
- `net-v0` 保持既有固定 horizon Tensor 输出兼容；`net-v1`/`net-v2` 额外输出逐步终止 logits，但训练和评估层统一读取轨迹点与可选终止证据。
- 模型构造只经注册表名称与可序列化配置切换，不允许训练脚本硬编码具体变体。
- Trainer 拥有设备选择、训练/验证循环、early stopping 和 checkpoint；数据坐标转换与 batch 准备不属于模型实现。

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

### `MODEL-REG-001`：模型注册表

- 前置：名称为已注册变体，配置字段可序列化。
- 行为：`build_model(name, cfg)` 构造 `net-v0`、`net-v1` 或 `net-v2`，调用方无需修改代码。
- 异常：未知名称或非法配置明确失败并列出可选名称。
- 兼容：`MineParkingNet` 继续表示 `net-v0`，已有导入和固定 horizon 前向不变。

### `MODEL-VAR-001`：v1 变长 GRU 解码

- 前置：BEV、goal、state 输入契约与 v0 一致，最大 horizon 为正。
- 行为：条件化 GRU 自回归生成轨迹点和逐步终止 logits；训练可提供 teacher forcing 轨迹，推理按终止阈值裁剪且至少返回一个点。
- 结果：批量前向返回 `(B,N,3)` 点与 `(B,N)` 终止 logits，`predict` 返回有效 `Trajectory`。

### `MODEL-VAR-002`：v2 U-Net 跳连与交叉注意力

- 行为：BEV 经过带跳连的多尺度编码/融合后形成空间 token，goal/state 条件作为 query 进行交叉注意力，再经与 v1 同口径的变长解码器输出。
- 结果：保持网络输入、轨迹点和终止 logits 契约，可由注册表配置切换。

### `MODEL-TRAIN-002`：可恢复训练器核心

- 前置：训练/验证 batch 统一为 BEV、局部目标、运动状态、局部轨迹和 mask；配置给出 epochs、学习率、patience、设备和 checkpoint 目录。
- 行为：Trainer 执行训练与验证，按 val loss 保存原子 best/last checkpoint，patience 到达后 early stopping，并返回逐 epoch 历史。
- 恢复与迁移：checkpoint 保存模型、优化器、epoch、best val 和配置；不同模型变体或不兼容配置不得静默恢复。

### `MODEL-LOSS-002`：平衡变长终止监督

- 前置：每条有效轨迹前缀只有最后一点为终止正类，其余有效点为负类，padding 不参与损失。
- 行为：启用平衡模式时，按当前 batch 有效前缀中的负/正数量动态提高终止正类贡献；轨迹 MSE 与终止损失权重仍由训练配置显式组合。
- 结果：全负预测不再通过类别数量优势获得低终止损失；长度为 1 的轨迹和不存在负类的 batch 仍产生有限损失。
- 兼容：损失函数保留可关闭平衡模式的参数；正式 v7 v1/v2 配置必须启用平衡模式。

### `MODEL-TRAIN-003`：可复现样本级 shuffle 与 scheduled sampling

- 行为：训练集每个 epoch 使用由训练 seed 和 epoch 派生的确定性样本级排列重新组 batch，验证集保持稳定顺序；变长解码按配置的起止 teacher-forcing 比例和衰减 epoch 线性调度，每个样本/时间步在真实前点与已预测前点之间可复现选择。
- 边界：模型反馈点在再次输入解码器前停止梯度，不改变轨迹点与停止 logits 的公开输出形状；固定长度 v0 不应用 scheduled sampling。
- 恢复：同一 seed、epoch、checkpoint 和配置恢复时必须重现相同 shuffle 与采样决策；新增训练语义字段属于 checkpoint 兼容字段，旧 checkpoint 必须明确拒绝而非静默续训。

### `MODEL-REPORT-002`：逐 epoch 自由滚动监控

- 行为：每个 epoch 在无 teacher forcing 下评估 train/val，记录 ADE、FDE；变长模型额外记录停止命中率和预测长度 MAE，并记录实际 teacher-forcing 比例。
- 结果：`history.json`、checkpoint 和训练曲线持久化上述指标；控制台进度显示关键自由滚动与停止指标，不能再以 teacher-forcing train loss 代替推理质量。
- 性能：监控允许增加有限的每 epoch 推理耗时，但不执行反向传播，也不改变 best/last 的原子保存语义。

### `MODEL-CONFIG-001`：安全 YAML 训练配置

- 前置：配置包含 model、data、training 和 output 映射，数据路径与输出路径允许相对 YAML 文件定位。
- 行为：入口使用安全 YAML loader，只接受已定义字段和可序列化标量/映射；构造注册表模型与 Trainer，不把 YAML 标签解析为任意 Python 对象。
- 异常：缺失字段、未知字段、非法类型、数据不存在或模型 horizon 小于数据有效长度时明确失败。
- 结果：同一 YAML 可确定模型、数据、训练参数、输出目录和可选恢复 checkpoint。

### `MODEL-REPORT-001`：训练证据落盘

- 行为：配置化训练在输出目录原子写入历史和最终报告，保存 train/val loss 曲线 PNG+PDF；训练完成后以 best checkpoint 在 val 数据上计算开环指标。
- 结果：报告记录模型/配置身份、epoch 历史、best 状态、ADE/FDE/航向误差和产物路径；正式训练未执行时不得宣称模型收敛或优于基线。

### `MODEL-EVAL-001`：统一开环轨迹指标

- 前置：预测与目标是 `(B,N,3)`，mask 是 `(B,N)` 且每条样本至少一个有效点。
- 行为：ADE 聚合全部有效点 XY 欧氏误差；FDE 聚合每条样本最后有效点 XY 误差；航向误差对角差环绕后聚合有效点绝对误差。
- 异常：形状不一致、非前缀 mask、非有限值或预测 horizon 短于目标有效长度时明确失败，不静默截断。

### `MODEL-EVAL-002`：checkpoint 开环比较入口

- 行为：独立 CLI 从 checkpoint 中恢复模型注册名与模型配置，在同一 NPZ 验证集上评估一个或多个模型，原子写入 JSON，并生成 PNG+PDF 指标对比图。
- 兼容：仅接受 Trainer schema v1 checkpoint；不修改 checkpoint 或数据集。完整 V3 的实际闭环轨迹与误差-时间图不在本轮开环入口中伪造。

### `MODEL-EVAL-003`：预测误差诊断与轨迹叠加

- 前置：单个 Trainer schema v1 checkpoint 与包含逐样本 `task_meta`、BEV 元数据的 schema v2 数据集。
- 行为：在目标有效前缀上计算逐样本与按场景、任务类型、机动方向、噪声和相邻占用分组的 ADE/FDE/环绕航向误差；变长模型额外报告预测终止长度误差。按 FDE 选择全局最差样本及每类任务最差样本，将网络预测、专家轨迹、目标和 BEV 叠加到同一车体局部坐标图。
- 结果：原子写入 JSON 报告，并保存分组指标、全局最差和按任务最差叠加图的 PNG/PDF；报告保留样本索引和任务 ID，使错误可回查到原始数据。
- 异常：元数据缺失或未与样本逐项对齐、checkpoint 与数据不兼容、预测或停止输出形状不一致时明确失败，不静默降级为无分组结论。

## 追溯

| Spec ID | 验收 | 测试或人工入口 | 实现符号 | 实际验证 | 状态 |
|---|---|---|---|---|---|
| `MODEL-NET-001` | 前向形状 | `tests/test_model.py::TestMineParkingNet` | `model/network.py::MineParkingNet` | unittest 通过 | ✅ |
| `MODEL-LOSS-001` | 掩码 MSE | `tests/test_model.py::TestLossFn` | `model/network.py::loss_fn` | unittest 通过 | ✅ |
| `MODEL-TRAIN-001` | 损失下降 | `tests/test_model.py::TestTrainingConvergence` + train.py | `scripts/train.py` | loss 3.38→0.021 | ✅ |
| `MODEL-DATA-001` | 落盘往返 | `scripts/train.py` 数据链路 | `dataset/generator.py::save/load` | train.py 生成并加载成功 | ✅ |
| `MODEL-REG-001` | 名称构造 v0/v1/v2，未知名称拒绝 | `tests/test_model_registry.py` | `model/registry.py::available_models/build_model` | 三变体构造/前向与未知名称拒绝通过 | ✅ |
| `MODEL-VAR-001` | v1 点/终止形状与 teacher forcing | `tests/test_model_variants.py` | `model/variants.py::MineParkingNetV1/_ConditionedGRUDecoder` | 点 `(B,N,3)`、终止 `(B,N)` 与变长损失通过；正式收敛待数据 | ✅ |
| `MODEL-VAR-002` | v2 多尺度条件编码与输出契约 | `tests/test_model_variants.py` | `model/variants.py::MineParkingNetV2` | U-Net 跳连、交叉注意力及同口径输出测试通过；正式收敛待数据 | ✅ |
| `MODEL-TRAIN-002` | val、early stopping、兼容恢复、best/last checkpoint | `tests/test_trainer.py` | `training/trainer.py::Trainer` | early stopping、原子 checkpoint、模型/超参数不兼容拒绝通过 | ✅ |
| `MODEL-LOSS-002` | 平衡终止正负类且边界损失有限 | `tests/test_model_variants.py` | `model/network.py::variable_loss_fn` | 全负捷径与单点边界测试通过；v7 smoke 停止命中率 99.67% | ✅ |
| `MODEL-TRAIN-003` | 每 epoch 确定性样本 shuffle、teacher-forcing 调度与恢复一致 | `tests/test_trainer.py`、`tests/test_model_variants.py` | `training/data.py::epoch_batches`、`training/trainer.py::TrainerConfig/Trainer`、`model/variants.py::_ConditionedGRUDecoder` | shuffle 跨 epoch、线性边界、采样复现和 checkpoint 语义拒绝测试通过 | ✅ |
| `MODEL-CONFIG-001` | 安全 YAML、严格 schema、相对路径 | `tests/test_training_config.py` | `training/config.py::load_training_run_config` | SafeLoader、未知/非序列化字段拒绝和相对路径通过 | ✅ |
| `MODEL-REPORT-001` | 历史/报告原子落盘与 PNG+PDF 曲线 | `tests/test_training_reporting.py`、`tests/test_training_runner.py` | `training/reporting.py`、`training/runner.py`、`scripts/train_model.py` | 1 epoch 合成训练贯通并生成全套产物；正式数据训练后置 | ✅ |
| `MODEL-REPORT-002` | history/checkpoint/曲线包含自由滚动和停止监控 | `tests/test_trainer.py`、`tests/test_training_reporting.py`、`tests/test_training_runner.py` | `training/trainer.py::TrainingHistory/Trainer._rollout_metrics`、`training/reporting.py`、`training/runner.py` | 合成 runner 与 v7 全量 1 epoch smoke 均生成完整指标、checkpoint 和 PNG/PDF | ✅ |
| `MODEL-EVAL-001` | mask ADE/FDE/环绕航向、短 horizon 拒绝 | `tests/test_open_loop_metrics.py` | `metrics/open_loop.py::compute_open_loop_metrics` | 精确数值、±π 环绕、mask 与短 horizon 拒绝通过 | ✅ |
| `MODEL-EVAL-002` | checkpoint 恢复、多模型 JSON/图 | `tests/test_eval_openloop.py`、CLI smoke | `training/checkpoint.py`、`scripts/eval_openloop.py` | checkpoint 恢复、JSON 与 PNG/PDF 图 smoke 通过；完整 V3 闭环后置 | ✅ |
| `MODEL-EVAL-003` | 分组指标、终止误差与预测/专家叠加图 | `tests/test_prediction_analysis.py`、v7 train/val/test CLI | `metrics/prediction_analysis.py`、`scripts/analyze_predictions.py`、`viz/prediction_analysis.py` | 9 项定向测试通过；v7 `net-v1` 三 split 生成 JSON 与三组 PNG/PDF，逐样本索引可回查 | ✅ |

## 待人工确认

- 无。
