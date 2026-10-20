# 冻结轨迹的停止头双臂训练

## 元数据

L3轻量规格，独立离线实验；现有模型与Trainer默认行为不变。

## 目标

在完整轨迹生成固定的条件下隔离专家点引导与自由滚动特征对停止头训练的影响。

## 非目标

不修复336条无联合点轨迹、不保证闭环改善、不改阈值/MPC/标签，不新增教师、不使用val/selection标签优化、不挑最好epoch或部署。

## 边界与约束

R0同起点，两臂同2918既有平衡池，原2400输入/有效标签不变，来源不与val/selection重叠，禁止S9。H=320、dt0.2、阈值继承父模型。tf1引导臂与tf0自由臂是机制对照，不是父模型训练过程的精确复现。

## 行为与验收

### STOP-HEAD-001：特征与冻结

只读捕获每步stop_head输入，缓存必须与直接前向logits在1e-6内一致；模型eval/no_grad，hook无论成功失败均移除。仅训练独立复制的stop_head，原模型保持不变。导出checkpoint除decoder.stop_head.weight/bias外所有state_dict张量逐元素相等，同输入完整320轨迹点必须torch.equal。

### STOP-HEAD-002：可比优化

两臂使用同初始head、固定20 epoch、Adam lr2e-4、梯度裁剪1、累计停止BCE系数0.2，batch8/seed20261020每epoch确定性打乱相同。目标仍为真值有效长度前继续、末点及以后停止，保留尾部累计监督、不做类别平衡加权，不把当前预测首联合点作标签。所有epoch保留，固定最后一轮交付，不按训练/评测表现选epoch。缓存一次仅在冻结生成器及tf0/1确定性条件下允许复用。

### STOP-HEAD-003：产物与隔离

单独输出两臂兼容只读加载的checkpoint，resumable=false，标明源模型epoch与阈值继承而未重校准；不伪造Trainer断点。缓存/模型/报告记录摘要与冻结主干身份。原2400自由滚动轨迹须与LT11归档逐元素相同，分别重算父模型/两臂终点审计；无联合点数必须相同，停止指标不可替代几何/闭环验收。

### STOP-HEAD-004：长任务与失败

冻结输入/代码/运行时，拒绝旧输出/漂移，无自动重试/恢复/扩大/评测/部署。smoke最多8样本×1 epoch每臂，仅验证可运行与冻结，不证明训练有效。正式任务由用户执行，身份或冻结不成立立即停止，记录失败而不跳过样本。缓存为实验中间物，不是训练标签集。

## 追溯

| ID | 实现/测试/验收 | 状态 |
|---|---|---|
| STOP-HEAD-001 | experiments/frozen_stop_head.py；tests/test_frozen_stop_head.py；results/agent-audit.md | ✅ 两缓存直接特征核验、40个checkpoint仅两head键变化、两最终导出重载完整点相同 |
| STOP-HEAD-002 | fit_stop_head；累计BCE/确定性/冻结测试；LT12正式报告 | ✅ 两臂同预算20轮/固定末轮已完成，元数据与history核验；不代表效果通过 |
| STOP-HEAD-003 | scripts/train_frozen_stop_head.py；verify_export；runs/training/frozen-stop-head-ab-20261020-v1/results/agent-audit.md | ✅ 原2400与LT11逐元素复现、三方统计重算一致、阈值未改；性能下降已报告 |
| STOP-HEAD-004 | 冻结CLI/覆盖拒绝测试；docs/long_tasks/LT-20261020-12-frozen-stop-head.md | ✅ 11输入/101源身份及正式完整性已验，无自动扩大/评测/部署；候选失败停止扩大 |
