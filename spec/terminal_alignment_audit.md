# 部署停点与真值末点对齐审计

## 元数据

L3轻量规格；独立离线诊断，不改变训练或部署契约。

## 目标

量化原训练域自由滚动预测的停止/终点不一致，按场景保留完整分母。

## 非目标

只读量化原训练域自由滚动预测的停止/终点不一致；不训练、不校准阈值、不修补输出、不把开环联合达标当闭环成功、几何安全或执行可行。

## 边界与约束

使用原坐标转换、原父模型阈值与max_horizon，输入仅train。0.3m/10°用于目标相对位姿检查，yaw按周期差；固定原始样本顺序，保留task_id/scene。无新依赖，默认模型/训练逻辑不变。

## 行为与验收

### TERM-AUDIT-001：三种索引分开

冻结阈值下取sigmoid(logit)首次≥阈值的点，无交叉则末点；记录真值有效末点索引与停止长度、停点/真值索引的预测误差、标签末点误差、全自由滚动域首次/最后联合达标点及数量。不得把真值索引预测误差冒充部署停点表现。输入形状/前缀mask/有限值/阈值/容差须校验。

### TERM-AUDIT-002：互斥分类与分母

每样本恰分一类：deployment_joint_valid、stop_misaligned_with_joint_point、no_joint_point。在完整预测有联合达标点而停点不合格时另记相对首末联合点的early/late/inside_gap；这些点可在停止之后，仅离线诊断，不是部署可用oracle。全体与按scene分组计数分母必须一致。

### TERM-AUDIT-003：冻结与人工批量

冻结模型/原train/manifest/既有LT10证据、执行源与运行时；不覆盖输出或自动重试，保存完整自由滚动points/logits/目标/mask/局部goal及逐样本分类。六源预检不代表正式2400训练域，通过身份校验才允许用户运行正式审计。formal报告仅diagnostic，无自动训练/扩大/部署，不使用val/selection作该推理。

## 追溯

| ID | 实现/验收 | 状态 |
|---|---|---|
| TERM-AUDIT-001 | experiments/terminal_alignment_audit.py；tests/test_terminal_alignment_audit.py | ✅ 指标/索引/有限值/周期角与阈值边界测试通过；六初态NPZ重算一致 |
| TERM-AUDIT-002 | analyze_terminal_alignment；分类/分组测试 | ✅ 三类/early-late-gap/场景与索引测试通过；六初态1/2/3不外推总体 |
| TERM-AUDIT-003 | scripts/audit_terminal_alignment.py；docs/long_tasks/LT-20261020-11-terminal-alignment.md；runs/diagnostics/terminal-alignment-train-20261020-v1/results/agent-audit.md | ✅ 5输入/99源/环境与2400索引覆盖、NPZ/逐样本及独立公式重算通过；仅诊断，不训练/改阈值 |
