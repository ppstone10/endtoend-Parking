# 带到达证书的网络参考保持

## 元数据

- 前缀：REF-KEEP；强度：完整；状态：LT24正式同300已完成并通过全部预声明门禁（263成功/0碰撞，双基线零丢失、+8新成功），`production_qualified`仍为false
- 最后更新：2026-10-26；前置：LT23保持255成功/0碰撞（正式结果已并入本Spec）

## 目标

对近端反复重规划打断可执行参考的情况，仅保持已由同一控制上下文验证能在原预算内严格到达的网络参考；保持原感知/推理节奏和控制求解。

## 非目标

不生成新参考、不加专家/平移接管/目标偏置，不改网络/停止阈值/默认控制器、原600步或车位容差；不解决所有危险参考、安全停车、BEV来源或泛化，不训练或回灌val。

## 边界与约束

新增verification的私有副本证书、参考源组合、独立审计与实验入口；LT21/LT22/LT23源/身份/产物不改。约束仍依赖仿真完整几何，production_qualified固定false。静态场景、精确确定性动力学是证书的条件，不作实车/动态障碍保证。原车位tol_pos/tol_yaw从同一引擎取得，不统一套默认值。

## 行为与验收

### REF-KEEP-001：限定资格与原预算

在原重规划点，当前距离不超过3倍本车位原位置容差、当前满足0.2m余量，且当前原始网络参考存在原联合合格点，才允许短前瞻。私有控制副本最多推演min(80,600-已执行步)步；没有在该预算内到达，或任何真实扫掠接触/0.2m余量侵入，则不保持参考。不用场景/任务索引选择策略，不扫参数。

### REF-KEEP-002：副本隔离与可执行证书

完整复制LT23控制器的热启动、进度/轨迹对象关系和RNG；副本不能调用实例上录制/观察闭包，不能改变真实控制器RNG、缓存、动作或感知。仅使用原轨迹与原LT23控制/终端动作。证书保留全部预测真实前后状态、动作、目标/原容差、参考及原控制上下文，独立核验到达与扫掠；不能仅以参考有合格点当作可执行证明。

### REF-KEEP-003：保持时仍感知与逐步核对

每个原重规划点仍正常感知/推理并保存原始新参考；证书执行期继续返回同一个已验证参考对象，维持原MPC进度。每个实际控制步必须与证书预测前态/动作/后态精确一致。漂移、证书耗尽却未到达等违反静态确定性假设时明确中断实验，保存incomplete，不默默作安全保证。上下文异常退出恢复工厂/观察钩子。

### REF-KEEP-004：固定短检与旧能力

固定0/167/168/247/267/292六案例、180秒单次短检；含LT23成功、S4负探针、S6负探针和#267保持正探针。不按结果换源/参数。相对LT23成功不丢失、零碰撞、至少新增成功且余量侵入不增加，独立证书审计通过才可准备正式同300。

### REF-KEEP-005：版本化联合验收

冻结代码/输入/运行环境与全部新账本，输出新目录；正式300任务只交付用户运行，要求保留LT23全部255成功（覆盖旧两基线）、成功≥210、零碰撞、余量侵入≤14,944，并有新增成功。区分保持参考、终端动作、无证书退回原行为；不把条件实验晋升为生产或原生网络收益。异常不覆盖/不自动恢复，停用新CLI即可回退，无旧数据迁移。

## 追溯

| Spec ID | 验收 | 测试或人工入口 | 实现 | 实际验证 | 状态 |
|---|---|---|---|---|---|
| REF-KEEP-001 | 资格/严格原预算/余量 | test_budget_cannot_be_extended_to_reach_goal；test_margin_intrusion_rejects_physical_safe_certificate；test_vehicle_specific_tolerances_are_preserved | reference_certificate.certify_reference；CertifiedReferenceSource.next_trajectory | 9新增/87聚合测试通过，独立证书原预算/余量/车位标准一致 | ✅ |
| REF-KEEP-002 | RNG/闭包隔离与证书 | test_certificate_isolated_from_live_rng_and_observer；test_real_controller_rng_and_cached_state_unchanged；独立审计 | reference_certificate.certify_reference；retention_audit.verify_certificate | 短检三个证书41步通过；正式192证书2486步由干净LT23控制器恢复热启动/RNG/进度重新求解，动作/状态/原车位首次联合到达/实体及0.2m余量逐位一致，见runs/verification/lt24-independent-review-20261026/report.json | ✅ |
| REF-KEEP-003 | 正常查询/逐步精确/恢复 | test_retention_still_queries_network_and_reuses_same_object；test_state_drift_is_rejected_before_real_control；test_retention_factory_is_restored_on_exception | CertifiedReferenceSource；retention_scope | 正式86,334实际转移、8758原查询/原始与使用参考独立核验；未跨更新保持的196回合全部NPZ数组精确等于LT23，含全部37个剩余失败；非证书CEM未独立重算 | ✅ |
| REF-KEEP-004 | 固定六例门禁 | compare_reference_retention.py --phase smoke | retention_study.run/compare | 13.326秒，LT23 3成功→4成功/0碰撞，新增267在420步到达，0缩短285→188步，旧成功无丢失，余量607不变；167/247仍失败 | ✅ |
| REF-KEEP-005 | 冻结/人工长任务/不晋升 | CLI prepare/precheck/inspect/audit；test_no_new_arrival_cannot_open_formal_gate | retention_study；retention_audit.audit | 153源/26输入冻结、precheck零回合；LT24正式300条497.95秒、退出码0、inspect返回completed_and_verified：263成功/0碰撞，LT23的255成功零丢失、新增8（全在S6）、余量14,944不增、`candidate_qualified=true`、`production_qualified=false`；291次尝试/192证书/158次保持，192/192证书与实际账本逐步一致 | ✅ 条件实验通过，非生产准入 |
