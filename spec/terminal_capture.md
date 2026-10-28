# 约束控制的终端朝向接管

## 元数据

- Spec ID前缀：TERM-CAPTURE
- 强度：完整；状态：正式LT23同300已完成，条件实验资格通过（255成功/0碰撞、双基线零丢失），`production_qualified`仍为false
- 最后更新：2026-10-26

## 目标

在网络参考与严格终端目标不一致时，允许已进入原位置容差的差分驱动车辆完成可验证的原地朝向收敛。

## 非目标

只提供显式系统候选，不修改网络、停止阈值、默认控制器或LT21/LT22冻结代码；不修复远离目标的S6参考，不引入平移接管、专家规划、训练或BEV占用接入。

## 边界与约束

新`verification/terminal_capture.py`拥有终端模式，新版本实验编排复用LT22只读录制器。目标和终端容差取自同一原生引擎，不从案例编号推断。物理约束仍来自simulator_ground_truth，production_qualified始终false。仅适用于原有可原地转向、无额外加速度状态约束的仿真动力学，不作实车保证。

## 行为与验收

### TERM-CAPTURE-001：窄范围终端资格

每周期重新检查位置误差严格小于该车位原tol_pos、航向尚未满足原tol_yaw。必须从同一原生引擎/复原车位取得容差，不统一套用0.3m/10°默认值。不能锁存已进入位置范围的历史状态来绕过联合验收；不改变终端判据。无目标、目标非有限值应拒绝绑定/执行。

### TERM-CAPTURE-002：完整转向证书

仅在从当前姿态到同XY目标航向的最短角转向完整车体扫掠无物理接触时接管。必须检查完整角程，而非仅首动作或一个预测窗口；预测序列线速度全部为0，角速度不超过原max_omega，最多旋转到目标角，剩余时域保持。物理接触安全与0.2m余量分别报告；已有余量侵入不能被称为严格安全。

### TERM-CAPTURE-003：退回原约束控制

不满足资格或转向证书失败时，精确保留LT22求解结果与随机流。终端接管仍先消费同次原约束求解，明确记录terminal_rotation、目标/容差/转向证书和干预，不把该动作说成原代价最优或纯网络改善。离开实验上下文或异常时恢复工厂。

### TERM-CAPTURE-004：固定短检与配对门禁

固定案例0/123/166/168/247/292：覆盖旧成功、LT22新增成功、危险参考近端失败、唯一旧成功回退和S6远端未到达。至多180秒，不换案例、不扫参数。与缓存LT22及原生两套结果比较；#168恢复、LT22短检成功无丢失、无碰撞、合计余量侵入不增加，才允许准备正式同300。#247若仍失败如实保留，不能声称S6已修复。

### TERM-CAPTURE-005：身份、独立审计及正式边界

输入/代码/运行环境逐文件冻结，输出新目录；未完成或证据漂移不得晋升。短检独立按归档动作推进实际动力学与扫掠，并在每个终端接管步从原目标和姿态独立重建零线速度/限幅角速度及完整转向证书。正式同300仅用户执行，至少保持原205与LT22的243个成功、成功≥210/碰撞≤30、余量侵入不高于LT22；正式通过仍仅为GT几何辅助条件实验，不自动替换默认方案。异常保留incomplete，不自动续跑或覆盖。

## 异常、兼容与回退

旧checkpoint/数据/执行入口与归档只读；新CLI失败不降级到专家或伪造成功。回退为不启用该显式入口，无schema迁移或外部依赖。模型/控制/评测身份与每条回合证据完整保存，已知回归集不作为泛化证据。

## 追溯

| Spec ID | 验收 | 测试或人工入口 | 实现符号 | 实际验证 | 状态 |
|---|---|---|---|---|---|
| TERM-CAPTURE-001 | 原联合容差/非法目标 | test_position_boundary_does_not_relax_tolerance；test_missing_or_nonfinite_goal_rejected | TerminalCaptureMPC.bind_terminal/_solve | 9项新增测试与69项既有测试聚合通过 | ✅ |
| TERM-CAPTURE-002 | 全角程扫掠/限幅/保持 | test_full_rotation_must_be_safe_beyond_first_horizon；test_rotation_clips_and_holds_without_translation；test_wrap_uses_shortest_rotation | TerminalCaptureMPC._solve；terminal_audit.audit | 六回合短审计通过；正式300的91,103实际步、109终端步已按复原车位原容差独立重建状态/实际扫掠/完整转向证书及终端序列，全部精确一致，见runs/verification/lt23-independent-review-20261026/report.json；非终端CEM与候选全时域未独立重算 | ✅ |
| TERM-CAPTURE-003 | 未接管数值与随机流不变/恢复 | test_outside_position_keeps_old_output_and_rng；test_terminal_factory_is_restored_after_exception | terminal_capture.terminal_scope | 聚合测试通过，#0/#123/#292短检结果保持 | ✅ |
| TERM-CAPTURE-004 | 六固定案例双基线配对 | compare_terminal_capture.py --phase smoke | terminal_study.run/comparisons | 12.135秒，LT22原3/6成功→5/6成功，#166/#168恢复，零碰撞，余量1191→375；#247仍失败 | ✅ |
| TERM-CAPTURE-005 | 身份/独立审计/正式边界 | CLI prepare/precheck/inspect/audit；test_safe_stopping_without_arrival_cannot_open_formal_gate | terminal_study.prepare/precheck/inspect；terminal_audit.audit | 148源/20输入冻结、precheck零回合通过；LT23正式300条485.4秒、退出码0、inspect返回completed_and_verified：255成功/0碰撞，原205与LT22的243均零丢失，+12新成功（S4+8/S6+3/S5+1），#168恢复（183步），余量14,944≤17,337且按步归一0.164<0.182，`candidate_qualified=true`、`production_qualified=false` | ✅ 条件实验通过，非生产准入 |
