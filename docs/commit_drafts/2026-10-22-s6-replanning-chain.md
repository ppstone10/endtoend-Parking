# 本轮增量 Commit 文稿

```text
feat(experiments): 验证连续重规划恢复与成功策略标签

- 新增完整消费mask、teacher目标及逐控制步参考索引回归保护
- 分离参考内容、MPC对象进度保留和同点重发布，验证410真实专家标签及float32导出执行
- 新增同120步400/410交替机制短检与固定检查点只读补评测，保留故障产物、不重复拟合
- 增加11项测试覆盖索引/消费/损失梯度等价、配额/身份/预算/调用路径
- 实证连续覆盖与停止选择在本S6条件下共同恢复，但原生停止/旧能力尚未过线
- 同步Spec及离线恢复教师迁移关口，不回灌probe、不修改默认模型或部署
```

本轮范围：新`experiments/s6_replanning_chain.py`、`scripts/preflight_s6_replanning_chain.py`、`scripts/preflight_s6_chain_learning.py`、两个对应测试文件、`spec/s6_replanning_chain.md`及相关文档增量。运行目录中的独立重放/交叉/教师银行脚本、模型与NPZ默认忽略，不将大产物混入源代码提交。此前未提交改动另归并，不建议直接`git add -A`。

验证：指定conda聚合29项测试、真实两precheck、py_compile、Spec与diff/空白检查通过；136/137源冻结、Adam360/120实际批次及故障前模型精确复用、关键539/340/140步回合逐元素重放，固定交叉及训练源53帧标签门禁完成。原停止政策失败、7挑战保旧失败，全局bias仅机制诊断，主68.3%/25.7%未重跑，未git add/commit/push。
