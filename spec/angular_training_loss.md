# 航向周期损失

## 元数据

等级：L3；状态：实现与短验证完成，正式训练/闭环效果待验证。

## 目标

目标：修正归一化 yaw 标签在 ±π 两侧的虚假大误差；不改变网络输入/输出、控制器、停止阈值和验收。

## 非目标

不承诺消除全部闭环失败，不修改角度输入编码、网络结构、MPC或部署模型。

## 边界与约束

历史raw默认不变，新模式显式选择；长任务由用户按冻结计划运行。

## 行为与验收

- YAW-LOSS-001：`yaw_error_mode=wrapped` 时，逐点、末尾对齐、终点位姿三处 yaw 残差均为 `atan2(sin(pred-target),cos(pred-target))`；位置、有效点掩码和停止 BCE 不变。±2π 等价，不区分车头车尾的 π 周期错误不得引入。恰好 π 处允许最短方向分支不唯一。
- YAW-LOSS-002：默认 `raw` 保留历史数值；旧 checkpoint 缺少该配置时视为 raw，仅允许 raw 语义恢复。raw/wrapped 不得跨语义恢复优化器；只加载父模型权重建立新实验不受限制。
- YAW-LOSS-003：改进实验从冻结原父模型起步，同一候选数据、随机种子、课程、学习率与四轮预算，两臂只改变 yaw 模式；独立目录与身份审计，保留每轮快照，不自动部署、评测或启动训练。旧计划代码身份失配必须拒绝，不能改写旧 manifest。

## 追溯

| 规范 | 实现 | 验证 |
|---|---|---|
| YAW-LOSS-001 | model/network.py；Trainer._batch_loss | ✅ tests.test_angular_training_loss：周期、梯度、位置/mask/停止项与Trainer贯通 |
| YAW-LOSS-002 | TrainerConfig；Trainer.load_checkpoint | ✅ 同测试：缺字段raw恢复、跨模式拒绝；历史raw数值逐位一致 |
| YAW-LOSS-003 | scripts/prepare_yaw_loss_ab.py；train_fixed_threshold.py | ✅ tests.test_continuation_drift_audit配置单变量；既有训练护栏测试；v2两臂真实smoke未更新权重。正式训练不属于本条已完成证据 |

这是待闭环验证的修正候选，不表示已证明它解释所有 S4 退化或能达到70%/10%。
