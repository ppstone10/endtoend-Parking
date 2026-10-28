建议标题：

```text
feat(experiments): 验证S8固定教师净空对照
```

必要正文：

```text
- 独立复制教师规划器，将规划余量固定为0.3m，保持原MPC和0.2m执行审计
- 冻结六实际查询、原上下文与最多两次补偿，拒绝外廓扩大后不可行起点/目标
- 完成32秒零训练对照：2338起点不可行，2351参考/执行未变，0新增监督
- 核验148源/770输入/37产物，12项定向测试及4关键回合精确重放通过
- 补充S8-ENTRY-006及知识证据，保持训练/部署放行关闭
```

本轮代码范围为`experiments/s8_teacher_clearance.py`、`scripts/preflight_s8_teacher_clearance.py`、`tests/test_s8_teacher_clearance.py`；配套`spec/s8_entry_recovery.md`、README/ARCHITECTURE/LEARNING/主验收计划和T-201852轨迹/日志。新入口依赖此前未提交实验模块，人工提交时核对依赖；不要用`git add .`混入无关ACL恢复和其他工作。原冻结源及产物不改，runs下报告/银行/独立复核保留为本地证据。未自动add/commit/push。
