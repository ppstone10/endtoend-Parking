# 人工Commit文稿

建议标题：`feat(verification): 补齐候选五层输入采集与分级质量验收`

建议正文：

- 为LT26候选新增真实BEV、局部预测、模型输入/目标/类型及语义归档；从初态独立重放感知、网络、候选MPC与两个接口，核对原始/使用参考和证书对象保持。
- 分开证据、接口、质量与覆盖资格，完整参考实体/余量和近终端目标支撑不再被重放一致覆盖，实际安全与完整任务到达分别判定；缺证据/profile明确blocked。
- 增加只读训练源感知经验包络及独立留出流程，排除val交集/恢复/重复seed，少样本/缺组/留出失败阻断。包络不代表绝对感知质量，L1/L1_L2未建立绝对门槛时不授予模块资格。
- 保持旧174源/模型/数据/控制策略及既有产物不变，新181源/41输入冻结，LT27交用户完成校准与固定14补证，不启动全300。

验证：44项相关聚合unittest通过；8新增Python文件编译/导入通过；#0从起点188步/19预测采集与LT26逐值一致，独立五层重放通过，正确报告L1 blocked、L2/L1_L2 failed、L3/L2_L3仅本回合passed；真实缺BEV被拒绝。两个训练任务校准预检正确保持ready_groups=0，完整校准/14补证尚未运行。

范围：新增verification/candidate_layer_{capture,analysis,quality,study}.py、verification/perception_calibration.py、scripts/verify_candidate_layers.py、scripts/calibrate_perception_acceptance.py、tests/test_candidate_layer_acceptance.py、对应Spec/LT27与文档/TRACE/worklog。保留用户已有改动，勿整仓一并暂存；runs证据按既有归档策略保留。

风险：固定训练源经验参考不保证绝对质量或泛化；S8部分组已知样本不足；当前策略仍未改善LT26真实闭环到达。默认入口未替换，无公共旧schema破坏/数据迁移；未执行git add/commit/push。
