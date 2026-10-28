# 分层验证修正与完整车体约束执行候选 Spec

## 元数据

- Spec ID 前缀：`EXEC-UPDATE`
- 强度：完整
- 状态：已采纳
- 最后更新：2026-10-23

## 目标

修正LT21感知类型重放和共同声明错误覆盖质量事件的问题；提供显式完整车体约束执行候选，分别检验参考安全与实际控制安全，保持原到达/碰撞/旧能力联合验收。

## 非目标

不覆盖LT21源码/配置/产物、不自动部署、不训练、不把过滤/安全停车计为原生网络学习收益，不通过停止偏置或放宽车体/余量/终止容差取得指标。

## 边界与约束

新增`verification/`拥有修正版验证与离线执行候选，旧137个冻结源不变；新入口与新输出。候选把仿真完整场景几何检查器提供给控制输出选择，是几何辅助条件实验，尚未接入仅由实际BEV建立的自由空间，不能作为原生纯NN/传感器闭环准入。MPC时序、CEM采样/跟踪成本、车辆模型和原停止阈值保持原值。参考过滤复用已有SweptFootprintProjector，明确标记干预，无专家回退。实际控制物理安全按零余量完整矩形连续扫掠；0.2m规划余量另行审计，不偷换物理碰撞口径。

## 行为与验收

### `EXEC-UPDATE-001`：精确类型重放与旧证据迁移

从原始起点与保存控制命令按同车辆模型恢复每步真实状态及标量类型，先逐元素核对全部保存前后态。感知回放必须使用原采集调用的标量类型；不使用宽容误差吞掉差异，不把所有时刻统一强转float。LT21四个plan0误报须在新报告明确更正，原609产物及137源SHA保持。缺失/漂移/无法重建时不得修正通过。

### `EXEC-UPDATE-002`：正交结果轴与输入语义

声明契约、结构重放、碰撞当步、全程历史风险、非到达分别展示；共同target声明错误不覆盖实际质量。原35/42/18当步分组和6个早期危险但末窗安全的案例必须保持。target训练实际来源尚未证实前不改变原checkpoint或生产输入；记录声明/实际/数据三方证据与未关闭项。

### `EXEC-UPDATE-003`：完整车体约束控制输出

候选控制器沿用原CEM随机采样、成本与均值拟合。原最终均值整时域接触安全时原样返回，不额外推进随机流；否则按原成本稳定排序，检查最多16条原采样序列以及原移位热启动与零动作，只有完整车体连续扫掠安全的序列可输出。采样安全不保证均值安全，必须单独验证最终输出。无安全前进候选时使用已验证安全的零动作并计数，不当作到达；初始已接触时明确失败。保存选择、代价、完整序列、检查数量与干预原因，可独立重算首个实际转移。有限采样不证明无解。

### `EXEC-UPDATE-004`：参考与控制分项消融

四臂固定为原网络/原MPC、原网络/约束MPC、既有几何过滤/原MPC、几何过滤/约束MPC。过滤保持原target/网络/stop阈值，余量固定车辆0.2m、最大侧移1m/10轮；干预/保持/截断单列。新版本显式标记NN加几何约束系统，不能混报为原生纯NN学习收益。分别记录每步实际物理接触和规划余量，不用低RMS替代安全。

### `EXEC-UPDATE-005`：短检与长任务准入

短检固定成功及S3/S4/S8案例，不按结果换源、扫参数或自动训练，整体预算最多180秒。完成后按实际结果决定是否值得正式全300；无进展或明显旧成功回归则停止当前候选并保留负结果，不伪造可晋升。正式长任务仅交付待用户运行计划，不自动启动。正式验收沿用同300成功≥210/碰撞≤30并报告配对旧成功回归、余量、安全停车和干预；不因去掉碰撞改成超时就宣布成功。

### `EXEC-UPDATE-006`：几何来源与准入边界

报告/检查器必须标明控制约束使用simulator_ground_truth；candidate_qualified仅表示该条件下系统实验的指标门槛，production_qualified必须为false。没有实现/验证BEV占用构造自由空间、遮挡/漏检/不确定性时，不作为完整传感器闭环已修复或实车可用。即使LT22指标通过也不自动替换默认方案，下一阶段另立感知来源下的执行约束验收。

## 数据、接口与迁移

- 新报告schema为`component_verification_v2`、`footprint_execution_candidate_v1`，引用旧源报告SHA；旧schema/旧入口保留可复现。
- 只读旧报告/NPZ/模型/数据；所有输出必须新目录，冻结所有实际源与输入、登记SHA；prepare/precheck不运行回合，异常保留incomplete，不自动续跑。
- 新候选进程内显式工厂注入，finally恢复；不改变默认控制器、runtime、训练和部署。
- 回退仅停用新入口；旧137源和609产物不迁移/覆写。无外部依赖/外发或私有信息变化。

## 追溯

| Spec ID | 验收 | 测试或人工入口 | 实现符号 | 实际验证 | 状态 |
|---|---|---|---|---|---|
| EXEC-UPDATE-001 | 四误报修正且状态/类型可复现 | test_typed_replay_keeps_native_scalar_evolution_and_rejects_drift；test_perception_restore_keeps_python_and_numpy_types_distinct | verification.typed_replay；report_update.migrate_report | 76,282步逐位重建；四初态精确纠正，609旧产物保持 | ✅ |
| EXEC-UPDATE-002 | 声明与质量并列 | LT21迁移；训练谱系/数据直接输入审计 | report_update；input_semantics.audit_and_correct | 保留35/42/18质量轴；R0=v40 epoch7权重；2918原数据/5目标探针/4预测探针，生成image声明副本，不计停车收益 | ✅ |
| EXEC-UPDATE-003 | 输出序列精确几何安全 | test_unsafe_cem_mean_cannot_escape_safe_candidate_check；test_low_cost_safe_sample_is_used_instead_of_unsafe_mean；test_safe_legacy_has_same_sampling_stream_and_numeric_output | safety_candidate.FootprintConstrainedMPC；candidate_audit.audit_control | 四回合1181步普通MPC独立选择/成本/窗口/状态精确一致；所有选中时域/实际转移接触安全 | ✅ |
| EXEC-UPDATE-004 | 原与组合消融/指标分列 | test_candidate_scope_restores_factory_on_exception；固定四臂短检 | candidate_execution.candidate_scope/compare_rows | 最终短检40.25秒：native1/3、control4/0、reference1/2、combined1/0且3振荡；只control有新增到达；377余量侵入步如实报告 | ✅ |
| EXEC-UPDATE-005 | 不覆盖、预算、联合门槛 | test_tampered_pilot_cannot_open_formal_gate；test_safe_stop_after_losing_old_success_cannot_claim_pilot_benefit；LT22 | candidate_study.prepare/precheck/run；inspect_footprint_execution | 69聚合测试通过；LT22正式300条568.7秒、退出码0、inspect通过：243成功/0碰撞，旧成功丢失#168，新增39，净+38；干预15,411/95,225步、余量侵入17,337步；candidate_qualified=false，唯一未过项是旧成功丢失（243≥210、0≤30已满足） | ✅ 已执行，未通过联合门槛 |
| EXEC-UPDATE-006 | 明确几何来源且禁止生产准入 | inspector实际短检核验；LT22条件说明 | inspect_footprint_execution.inspect | 约束使用仿真场景完整几何，BEV占用接入未实施；正式结果标记geometry_source=simulator_ground_truth、production_qualified=false，243/0不替换默认控制器，真实感知/实车仍不在范围 | ✅ |

## 待人工确认

无。正式长任务仍由用户自行执行；本轮候选不自动部署。
