# 本轮增量 Commit 文稿

```text
feat(experiments): 审计S6优化传输并验证局部恢复与重规划缺口

- 新增梯度统计、有限Adam副本及递归张量状态相等工具
- 精确重放六条C240至264历史路径，区分停止梯度、全局裁剪与真实优化位移
- 新增同120步边界/联合目标隔离，保留旧累计loss缓存控制及非候选边界
- 增加9项测试覆盖原对象不变、有限更新等价、固定步数、身份/预算/输出保护
- 实证联合模型前19点原MPC28步严格可执行，但自主滚动仍被后续重规划破坏
- 同步S6-RECOVERY-009至011和下一连续恢复链验证关口，不扩大旧配方或部署
```

范围：新`experiments/s6_update_transport.py`、`scripts/audit_s6_update_transport.py`、`scripts/preflight_s6_event_learnability.py`、两个对应测试文件及本轮文档增量。运行目录中的模型/NPZ、oracle和单例bias脚本默认忽略，不将大模型混入源码提交。仓库另有此前大量未提交改动，本文不代表其统一提交说明；不建议直接`git add -A`。

验证：指定conda的六组聚合测试33项通过、两真实precheck零梯度/训练/回合、py_compile及差异/空白检查；六历史模型/Adam/损失精确重放，两个新Adam360/完整S6重放、原教师31步与学习前缀28步精确复现。全局bias改变显式仅机制诊断，无可晋升收益，主68.3%/25.7%未重跑。未git add/commit/push。
