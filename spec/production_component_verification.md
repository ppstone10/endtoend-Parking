# 生产闭环分层验证 Spec

## 元数据

- Spec ID 前缀：`VERIFY-E5`
- 强度：完整
- 状态：已采纳
- 最后更新：2026-10-23

## 目标

将 L1 感知、L2 预测、L3 控制及两个耦合验证绑定到同一个原生 E5 运行与任务终止标准；将碰撞和未到达作为回归样本，给出有证据边界的分层定位。

## 非目标

此变更不训练模型、不修改生产感知/网络/MPC，不把几何反事实或局部通过当成修复。

## 边界与约束

生产定义为 sensor BEV + network + mpc_vehicle + safety_mode=none。直接复用 validation_matrix 的组件、控制时序、终止判定、完整车体连续扫掠。规划余量与物理碰撞分别报告。GT、理想执行、单周期和旧工具仅是诊断，不提供生产验收。现有冻结历史代码与产物保持不变。超过五分钟或无法可靠估时的正式批量验证由用户执行。

## 行为与验收

### `VERIFY-E5-001`：统一生产身份和运行

要求已完成纯 E5 基线、非空唯一任务索引、相同数据/模型 SHA、车辆及协议。采集从原任务起点到原生终止的所有预测和控制，不重置 MPC 进度/热启动/随机流。逐回合成功、碰撞、分类、步数、末态误差必须复现基线；漂移、缺失、重复或代码变化时停止归因，不得通过。

### `VERIFY-E5-002`：模块与耦合使用真实全程状态

L1 检查每个实际预测时刻的 BEV 有限性、通道/空间/输入契约，以及同位姿高束数无噪声 LiDAR 和目标几何保真度；IoU 只作诊断，不虚构未校准门槛。L2 检查实际发出参考的端点达标、连续扫掠及各消费窗口。L3 检查每个实际控制周期的完整车体扫掠、xy/yaw 跟踪和选中控制序列预测，不以起点十步或中心 RMS 代替全程。耦合检查局部到全局变换、预测/控制 dt、同一状态/参考/窗口与连续状态。局部安全不意味着可到达。

### `VERIFY-E5-003`：统一终止、结果与归因

终止直接用每任务 TerminalChecker 双阈值（严格小于）；碰撞用 E5 零余量矩形连续扫掠，规划余量单列。区分运行 completed、检查状态、回归任务通过、正式总体通过。碰撞按参考内部/桥接/执行危险给出事件证据，不能当唯一根因；未到达报告原失败类别、距离变化、端点误差及计划切换。仅连接检查存在具体不一致才标记 boundary_mismatch；所有已检查局部通过但未到达必须 unresolved，禁止自动推断链接故障。

### `VERIFY-E5-004`：准入与历史迁移

旧三个 CLI 默认转入统一生产入口；旧参数需显式 --legacy-diagnostic，旧函数返回报告加诊断身份，历史文件不重写。现有矩阵套件仍可作对照，但汇总必须显示没有匹配分层证据时生产资格为 false。正式资格要求同一完整 300 条、原生 E5 成功率≥70%且碰撞率≤10%、完整身份/复现/结构检查，以及对应分层证据；已知失败子集/冒烟不可晋升。分层报告不允许用 diagnostic_only 或未测状态声明模块质量通过。

### `VERIFY-E5-005`：产物与长任务恢复

新输出目录必须不存在；先冻结配置（含基线/数据/manifest/模型/相关代码哈希），precheck 不运行回合，run 只消费冻结配置。记录逐回合 JSON/NPZ、统一报告与可读定位摘要、产物 SHA。中断保留部分产物，不自动续跑或覆盖；使用新目录重跑。短冒烟预算有上限，不自动转全量。正式计划必须注明输入、命令、资源、分母、异常和验收。

## 数据、接口与迁移

- 新 schema `production_component_verification_v1` 包含 identity、coverage、layers、couplings、episodes、acceptance；原矩阵 schema 不改。
- CLI：三个旧入口转发共享参数 --baseline/--config/--output，prepare/precheck/run/inspect 分开。原诊断函数和 --legacy-diagnostic 保留。v1固定当前val同300数据；其他模型可提供其完整原生E5基线，冻结当次输入后必须逐回合复现，不将新冻结SHA冒认为历史来源证明。
- 数据只读，输出仅写新目录；失败报告不是模型故障恢复器；本工具不改训练/部署。
- 回退：移除新入口转发与新模块即可恢复旧 CLI；旧产物无迁移，禁止把旧报告追认为新验收。
- 安全与隐私：本地离线仿真与模型/数据只读，不发送外部消息，无新增依赖或部署。

## 追溯

| Spec ID | 验收 | 测试或人工入口 | 实现符号 | 实际验证 | 状态 |
|---|---|---|---|---|---|
| VERIFY-E5-001 | 原生无干预、身份与复现 | test_diagnostic_substitutions_and_incomplete_denominator_rejected；test_capture_factories_restore_after_native_error | component_verification.validate_baseline；component_capture.capture_native_e5 | 四回合原结果/步数/末态误差精确一致；当前身份/协议检查 | ✅ |
| VERIFY-E5-002 | 全程三模块与耦合覆盖 | test_transform_uses_float32_rounding_tolerance_but_rejects_axis_error；test_reference_demands_wrap_yaw_and_expose_untrackable_speed；EXEC-UPDATE-001 | component_analysis.analyze_episode；verification.typed_replay/report_update纠错迁移 | 全量300/76,282控制/7,758预测覆盖；L2/L3契约一致。4条L1为原生Python float与回放NumPy float64的类型误报，已在v2按精确状态/类型重建纠正，非随机降采样；旧源码/产物保留，历史声明错误与当前校正副本分别记录 | ✅ |
| VERIFY-E5-003 | 碰撞/未到达进入局部且谨慎归因 | test_capture_risks_include_full_yaw_footprint_not_center_only；test_local_checks_do_not_prove_link_fault | component_verification.locate_failure；component_analysis.analyze_episode | S4#163危险参考、S8#292安全窗口实际碰撞、S6#247振荡进入验证；不宣称充分因果 | ✅ |
| VERIFY-E5-004 | 诊断不背书，正式联合门槛 | test_production_entrypoints_and_explicit_legacy_mode；test_joint_thresholds_and_exact_denominator | 三evaluate入口；run_validation_suite._summary_rows；component_verification.acceptance | 59项聚合定向测试；三个旧入口默认生产、历史显式诊断；冒烟不晋升 | ✅ |
| VERIFY-E5-005 | 不覆盖、冻结、用户长任务 | test_artifact_tampering_is_rejected_before_inspection；test_existing_output_is_not_overwritten；test_budget_stops_without_autocontinuation；LT21 | verify_closed_loop_components.prepare/precheck/run/inspect_completed | 正式v2配置全量300条执行完成（1405.3秒、退出码0）：precheck零回合、新目录、300唯一分母、609产物SHA与inspect通过；正式资格false为模型负结果 | ✅ |

## 待人工确认

无。长任务由用户自行运行，结果复核后才更新正式证据状态。

## 实现完成与能力准入的区别（2026-10-26）

上述实现追溯通过不代表五项模块质量通过。历史300回合五层module_qualified均为false；L1无校准质量门槛，L2/L3及耦合存在参考风险、碰撞和未到达。历史L1_L2声明错误由独立image模型副本对齐实际输入，旧报告保留，不追认为历史全绿。

LT26是独立GT约束候选的小集审计，未归档BEV、局部预测及完整模型输入，不能直接套用本Spec的原生E5全五层覆盖。本轮只读短重放九失败回合540次原始预测全部一致，其中381条完整参考不满足实体连接扫掠、454条不满足0.2m。输入重建/接口一致不等于感知或参考质量合格；详见runs/verification/five-module-status-20261026/review.md。此处记录实际覆盖缺口，不修改原生E5契约或把诊断追认为正式准入。


当前候选的后续补齐采用独立spec/candidate_layer_acceptance.md：新增全输入归档、完整参考/终端支撑质量规则及训练源经验参考；LT27已完成校准（1780任务/24组：16 ready、3 failed、5 blocked）与固定14补证（采集`observer_exact_to_LT26=true`、L2/L1_L2全failed、L1 1 failed+4 blocked、L3全passed），`all_five_modules_qualified=false`。旧LT26文件不追认为完整归档，原生E5规范及历史资格不改，绝对感知质量门槛仍未建立。

## 环视新版重放补证（2026-10-27）

VERIFY-E5-001/002既有输入身份要求同样适用于归档BEV重建：必须保留真实camera_projection，缺失仍为历史None，不推测或补造per_view_v2。component_analysis.analyze_episode已修正字段遗漏；test_archived_projection_reaches_network_replay_without_relabeling覆盖新版、显式旧版、缺省字段，相关18项测试通过。固定四例两模型共八回合精确采集与感知/网络/MPC独立重放均无接口错误，质量仍存在危险参考和执行碰撞；不构成正式300条或五模块质量通过。证据runs/verification/native-continuation-precheck-v2-20261027/review.md，未修改资格门槛。
