# 场景覆盖双臂训练交付 Spec

## 元数据

- 前缀：`RECOVERY-AB`；强度：轻量；状态：已实现，两臂各 4 轮已运行完成（待闭环对照）；日期：2026-10-18。

## 目标

提供可由用户手动执行、固定阈值、可恢复且不自动晋升的恢复样本配比双臂训练。

## 非目标

运行正式训练、改变默认训练器/runner、网络、损失或MPC；自动校准停止阈值、自动选部署模型、使用验证恢复标签训练或评测S9。

## 边界与约束

仅新增实验入口。指定conda Python；原产物只读，新产物进入独立任务目录；长任务由用户执行，Agent只作短预检。训练结束后等待Agent核验，不能仅凭loss宣布达标。

## 行为与验收

### `RECOVERY-AB-001`：分层样本选择

固定原2400条、倒车恢复259条；前进恢复259条优先保留全部28条碰撞回溯，剩余名额按场景×触发类别最大余数配额与固定seed选择。总2918条、恢复来源状态唯一，原标签逐点不变；主val和独立选型身份零交集。拒绝配额不足、缺失来源身份及不满足固定计数的输入。候选清单与前轮试算一致。

### `RECOVERY-AB-002`：可运行配置与身份冻结

准备入口拒绝覆盖已有目录，原子写候选NPZ和配置，manifest最后标明ready。记录输入文件、执行代码与两臂配置摘要、模型配置和阈值；正式入口核对摘要，输入/代码/配置改变拒绝续跑。两臂只在训练数据与输出身份上不同。

### `RECOVERY-AB-003`：固定阈值短预算训练

沿用Trainer，4轮同预算、batch8、lr5e-5、相同seed与TF .44/.36/.28/.2，原损失和恢复batch策略不变。直接用校准父模型权重、两臂重置优化器，阈值保持父值约.26。每轮保存可推理及可恢复快照、日志与history，不加载best替代末轮，不写deployment，不校准、不闭环选型。

### `RECOVERY-AB-004`：中断与人工交接

初次运行拒绝已有正式产物；恢复必须显式指定resume，校验身份并从已完成epoch的last恢复模型/优化器/history，修复中断时尚未保存的同轮快照。互斥锁防并发，正常/可捕获异常释放；强制终止的残留锁须人工核实后处理。只在4轮与快照完整时写completed，失败/中断记录状态。check-only和smoke不调用fit、不更新权重或写正式检查点；长任务计划交付到待用户运行后停止。

## 追溯

| ID | 验收入口 | 实现 | 实际验证 | 状态 |
|---|---|---|---|---|
| RECOVERY-AB-001 | TestRecoverySelection；真实prepare | scripts/prepare_recovery_balance_ab.py::select_recovery_indices、assert_original_equivalence、prepare_experiment | 总2918/原2400/前倒恢复各259/回溯98，原有效标签一致及评测身份零重叠通过；候选清单同试算 | ✅ |
| RECOVERY-AB-002 | TestRunGuards；R1 check-only | execution_code_identity；scripts/train_fixed_threshold.py::load_frozen_config、check_run_state | 覆盖/配置/输入/代码拒绝测试通过，v2冻结manifest与R1真实预检通过 | ✅ |
| RECOVERY-AB-003 | TestSyntheticTrainingResume；R2 smoke；正式 R1/R2 | make_training、ProgressTrainer、run_training | 玩具模型4轮保存与恢复一致，真实R2单batch有限loss/梯度且权重不变；2026-10-18 正式模型两臂各完成4轮（R1 val_loss 1.530351、R2 1.672187），逐轮快照与report齐全，无deployment、无自动校准 | ✅ |
| RECOVERY-AB-004 | TestRunGuards、TestSyntheticTrainingResume；LT-20261018-01 | exclusive_run、run_preflight、run_training；docs/long_tasks/LT-20261018-01-recovery-balance-ab.md | 锁/跨臂拒绝/缺失快照修复/冒烟无fit测试通过；真实两臂长跑完成并核实锁释放与completed状态，运行结果记入任务目录summary.md；真实模型中断恢复仍未实际触发 | ⚠️ |
