# 当前闭环候选的五单元验收

## 元数据

前缀：LAYER-ACCEPT；强度：完整；状态：工具/短预检完成，LT27完整训练源校准与固定14补证已执行——证据链完整，但五层质量未全部通过（L2/L1_L2全部failed、L1有1 failed+4 blocked、L3全passed、L2_L3随任务成败）；任务T-190818；更新2026-10-26。

## 目标

为当前候选补全实际输入采集、独立重放与质量判定，明确区分采集完整、接口一致、质量通过及生产资格。危险参考、终端无支撑、安全但未到达必须显示失败，缺校准或缺输入必须显示blocked。

## 非目标

不修改模型/训练/控制策略/停止阈值/原车位标准，不将val用于校准或训练，不回写旧报告；不以经验感知包络宣称绝对检测质量、泛化或实车安全，不根据本次已知失败14例准备生产或全300晋升。

## 边界与约束

旧174执行源与结果冻结，新目录和新入口复用FeasibleReferenceCandidateMPC及原证书保持。源LT26固定14例；短预检仅0号，不截断原600步。全部观察不得改变原随机流、引用对象、预测dt、计划更新和控制进度。两种矩形余量分别核验；静态GT辅助、可立即停车假设不变，production_qualified=false。

## 行为与验收

### LAYER-ACCEPT-001：全输入采集与不干预

每个实际查询归档BEV张量/元数据、实际模型状态/局部目标、采集位姿及Python/NumPy标量类型、局部预测/世界原始参考/dt、实际管道target与height语义。只调用原capture/predict各一次，不额外消耗传感器或模型随机流。保持原始/使用参考及证书账本；逐回合结果、动作、前后态、原始与使用参考须与同身份LT26逐值相等，否则中断。所有工厂/方法finally恢复，异常产物保留incomplete。

### LAYER-ACCEPT-002：五层独立重放与证据完整

从归档初态与命令独立推进，重建真实状态与类型。重新感知并逐元素比较实际归档BEV及元数据；按归档输入预测局部参考，验证目标及世界变换、dt/语义。原控制器候选从起点连续重放，证书保持时复用同一参考对象；实际窗口/进度/所选完整序列/动作逐值一致，实际状态和首次原联合终端一致。文件/分母/身份漂移不能算通过。

### LAYER-ACCEPT-003：质量独立于重放

L2完整原始参考及其当前状态连接须满足实体与0.2m完整车体扫掠，单列近端窗口及完整参考。近终端定义沿用3×原tol_pos；这些查询须有原位置与航向联合达标点，非近端查询不强制含终点。未覆盖近端则终端支撑not_observed，不因空集合通过。L3所有实际转移和完整H10均须满足实体及0.2m；L2_L3单列原车位600步完整任务成功，安全停车不算到达。可观察输入/输出一致的L1_L2仅获接口一致，不自动表示组合预测质量通过。

### LAYER-ACCEPT-004：感知经验包络校准与适用边界

只从已存在训练源重新感知并比较同位姿3600束clean LiDAR及目标几何真值，禁止用val指标拟合。按scene/noise/目标是否在BEV内分组，按原任务seed去重，同seed只取一个原任务起点；任务ID/seed与val相交者排除。固定每组三分之二拟合、三分之一留出；拟合至少20、留出至少10才可形成组参考。指标为occupancy IoU/recall、target IoU/hit_rate，空真值指标记not_observed，不把空图当高质量样本。

组下界为拟合样本各指标最小值，留出中任何指标低于界的任务比例须≤0.05，失败不重新拟合/选组。该界是固定源经验参考包络，不是置信区间或绝对检测安全门槛；不假设连续帧独立。缺组、样本不足、留出失败、来源/代码/车辆/空间/通道/语义不同须blocked或failed。包络一致只允许声明reference_conformance，绝对感知质量未建立时module_qualified不得为true。

### LAYER-ACCEPT-005：分层判定与资格

每层分别输出evidence_status、contract_status、quality_status、module_qualified与原因。明确安全/目标不达标为failed；缺证据/校准为blocked；接口一致不能覆盖质量失败。L1及L1_L2的质量声明限定经验参考一致性，absolute_quality_status=not_established时不发绝对质量资格。其他层只在本次覆盖内、有足够质量观察且全通过时获covered_scope_qualified，不将已知14例当全域生产证明。总体保留5模块未通过/blocked原因、原任务指标及旧成功保持，production_qualified固定false。

### LAYER-ACCEPT-006：冻结、阶段与恢复

CLI分prepare/precheck/capture/analyze/inspect，校准另有prepare/precheck/run/inspect。冻结代码/运行时/模型/数据/manifest/源报告与产物摘要，校准产物在分析配置中固定SHA；新目录禁止覆盖。预检不运行回合。短预检不产生可用完整校准；完整训练源校准与固定14采集/重放由用户LT27执行。采集与分析分开完成，正常执行可得到质量负结果；退出0不等于质量通过。不自动重跑/续跑/全300，异常新目录诊断后恢复。

## 数据、接口与迁移

新增candidate_layer_capture、candidate_layer_analysis、candidate_layer_quality、perception_calibration、candidate_layer_study及两个CLI，各自拥有采集、重放、规则、训练源校准和阶段编排。旧五层与LT26 schema不改；新层报告拥有本次采集身份及profile身份，不继承历史通过标记。旧配置及模型只读，无数据迁移；停用新入口回退。

## 异常、安全与隐私

仅本地读写，不外发、不改变训练或默认运行入口。捕获/分析非有限、重复/缺失查询、语义不一致、原始/使用参考漂移或预算耗尽均留incomplete并报错；校准不足属于可报告blocked，不伪装异常成功或放宽门槛。

## 追溯

| Spec ID | 验收 | 测试或人工入口 | 实现 | 实际验证 | 状态 |
|---|---|---|---|---|---|
| LAYER-ACCEPT-001 | 全输入及原动作/参考精确保留 | 定向测试/短预检/LT27 capture | candidate_layer_capture.capture_inputs_scope；candidate_layer_study.assert_observer_invariance | 固定14全部采集完成、111.60秒，`observer_exact_to_LT26=true`（结果/动作/状态/原始与使用参考/证书逐值一致）；87登记产物SHA一致 | ✅ |
| LAYER-ACCEPT-002 | 五层逐值重放/缺失拒绝 | 短预检/LT27 analyze | candidate_layer_analysis.analyze_episode | 14例分析132.95秒完成，五层`contract_status`全部passed；真实BEV/局部预测/完整控制独立重放一致；16登记产物SHA一致 | ✅ |
| LAYER-ACCEPT-003 | 安全和终端风险不能被重放覆盖 | 定向测试/真实短预检 | candidate_layer_quality.episode_verdict | 14例中L2对全部14例报failed（完整参考每例1–60条实体/余量危险）、L3全部passed（实际转移与H10无危险）、L2_L3对9条未到达failed，说明质量与接口/执行分离 | ✅ |
| LAYER-ACCEPT-004 | 训练源/去重/分组/留出/空真值/身份 | 定向测试/校准预检/LT27 | perception_calibration | 1780唯一训练任务、24组、94.27秒：16 ready/3 failed（留出失败率0.062–0.125）/5 blocked（S2 target=0组2–5个样本、S8 clean 19与high 17低于20）；`absolute_quality_status=not_established` | ✅ |
| LAYER-ACCEPT-005 | passed/failed/blocked及范围 | 定向测试/inspect | candidate_layer_quality；candidate_layer_study.inspect | 五层接口通过但质量分离：L1 9 passed/1 failed(#131 outside_perception_reference)/4 blocked(139/183/292/296 缺可用参考)；`all_five_modules_qualified=false`、`production_qualified=false`，L1绝对质量仍not_established | ✅ |
| LAYER-ACCEPT-006 | 冻结/阶段/不覆盖/人工执行 | prepare/precheck/LT27 | candidate_layer_study及两个CLI | 181源/41输入零漂移；校准→inspect→prepare→precheck→capture→analyze→inspect全阶段退出0，profile非preflight、capture/analyze均completed、14唯一任务、inspect通过 | ✅ |


## LT27独立复核补充

五层接口在694预测/6919步全部通过；L1按回合为9经验包络内/1越界/4受阻，绝对模块资格0；L2和L1_L2质量全部failed，L3仅本14覆盖passed，L2_L3为5到达/9未到达。16个ready感知组target两个指标下界都为0，不可据此证明目标检测健康。

LT27之后的方向对照另确认旧环视采集叠图后使用前视IPM丢失方向，左右目标输入image和输出BEV均相同但真值不相交；既有训练第2条target也前后错位。该次冻结运行证据保持不变；现行几何修复遵守独立spec/surround_camera_projection.md，以per_view_v2和训练输入重建隔离新旧版本，不追认LT27相对包络为绝对质量通过。独立重放须保留归档BEV的投影版本。证据runs/verification/lt27-independent-review-20261026/review.md。
