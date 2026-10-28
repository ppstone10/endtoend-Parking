# 本轮增量 Commit 文稿

```text
feat(experiments): 增加S6停止边界与终点监督闭环对照

- 精确复现完整19点控制排程，仅在原step400恢复槽替换terminal均衡stop并配对终点位姿监督
- 保留原角色归一及其他95槽损失，冻结旧129源、输入标签、C240/Adam240和原MPC
- 新增180秒四模型/44回合入口、梯度前身份检查和双顺序保旧增量资格
- 增加8项测试覆盖损失/梯度、触发边界、标签漂移、预算、输出保护及资格反例
- 实跑114.725秒及只读重放：两方向均未过线，自主首停仍35/教师19，不扩大该配方
- 同步Spec、结果入口、架构与经验，保留主68.3%/25.7%未变及未部署边界
```

范围：新`experiments/s6_terminal_objective.py`、`scripts/preflight_s6_terminal_objective.py`、`tests/test_s6_terminal_objective.py`及本轮相关文档增量。仓库还有此前大量未提交改动，本稿不替它们统一归类，不建议直接`git add -A`。运行目录默认忽略；四模型、NPZ和本次只读脚本/结果留在原路径，不将大模型混入源代码提交。

验证：指定conda执行`-m unittest tests.test_s6_terminal_objective tests.test_s6_terminal_consumption tests.test_s6_recovery_learning tests.test_angular_training_loss`，24项通过；真实`--precheck`零训练/回合；四模型44回合完成；131源/53产物、四Adam264/96批次及四S6完整数组精确重放和同状态迁移复核通过。未执行git add/commit/push。
