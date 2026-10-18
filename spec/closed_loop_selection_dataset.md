# 独立闭环选型与运动学配对 Spec

## 元数据

- 前缀：`LOOP-SELECT`；强度：轻量；状态：已采纳；日期：2026-10-18。

## 目标

提供独立于已知训练/恢复任务身份的选型数据，并将原验证集的近端运动学指标与真实闭环结果对齐。

## 非目标

更改训练集、模型、控制参数或主验收阈值；把选型数据称为最终测试集；生成或评测 S9。

## 边界与约束

沿用 TaskSampler、已有专家/可行性门禁、schema v2 和批次恢复；manifest 标明 checkpoint_selection_only。去重保证以实际扫描且可读取的归档为范围，报告所有来源和未扫描/无身份来源，不宣称覆盖未知外部历史数据。

## 行为与验收

### `LOOP-SELECT-001`：新任务和训练身份排除

由已有七场景验证集的场景×任务×机动×噪声×相邻占用配额，使用不同根 seed 采样新任务。排除已知归档的 task_id 和恢复样本 source_task_id，拒绝重复，生成后再次检查。不得通过复用旧训练样本构成新集。拒绝含 S9 模板。

### `LOOP-SELECT-002`：可恢复构建与用途冻结

生成真实专家标签，保留同单元重采失败证据，批次原子检查点绑定 seed、模板、排除来源和计划指纹。参数不同拒绝误续建。最终文件与 manifest、配额和去重报告一致。

### `LOOP-SELECT-003`：近端指标按任务身份对齐

模型自由反馈推理，按基线 selected/evaluated dataset_index 对齐成功、碰撞和振荡；分别报告起点误差、近端横向运动残差、速度/角速度及专家对照，按整体与场景分组。该分析仅为统计关联，不宣称因果。

### `LOOP-SELECT-004`：联合选型门槛

对相同数据、索引、控制参数的无干预 E5 报告，联合检查成功率≥70%、碰撞率≤10%。未达标时区分双指标非退化改善和指标权衡，不把低碰撞但大量停滞当作已达标。选型集通过仍须原完整主协议确认，不自动写部署权重。

## 追溯

| ID | 验收入口 | 实现 | 实际验证 | 状态 |
|---|---|---|---|---|
| LOOP-SELECT-001 | tests.test_independent_selection.TestIndependentSelection | scripts/build_selection_dataset.py::collect_metadata_ids、sample_selection_tasks | 300 条新任务唯一，与 47 份归档及模板排除身份零重叠；新测试通过 | ✅ |
| LOOP-SELECT-002 | test_changed_selection_identity_cannot_resume；tests.test_dataset_build | build_selection_dataset；复用 _prepare_checkpoints、_build_split_in_batches | 30 批真实生成完成，失败/重采 0；300 条闭环复原失败 0；相关测试通过 | ✅ |
| LOOP-SELECT-003 | tests.test_independent_selection.TestPrefixKinematics | scripts/analyze_prefix_kinematics.py::prefix_metrics、align_outcomes、run_analysis | 原 300 条自由反馈指标已按身份对齐闭环结果；数值和错误路径测试通过 | ✅ |
| LOOP-SELECT-004 | tests.test_independent_selection.TestJointSelection | scripts/compare_selection_candidates.py::assess_candidate_metrics、compare_reports | 双门槛与协议拒绝测试通过；实际选型基线判定未准入，不改部署 | ✅ |
