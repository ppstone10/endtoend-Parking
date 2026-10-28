建议标题：

```text
feat(experiments): 验证S8起点恢复教师与固定时序反事实
```

必要正文：

```text
- 新增独立起点/危险前访问选择与frame0查询审计，不改旧非零查询契约
- 冻结四学生来源、六查询，保留原MPC/余时/余量并核验原回合及后缀
- 有限航向补偿仅取得1个严格新增状态，执行与float32导出均可复现
- 固定双倍时长保留原节点并重验几何/执行，六查询无新增严格教师
- 完成20项定向测试和5个关键回合精确重放；无学生训练或扩训放行
```

本轮直接范围为`s8_entry_recovery`、`s8_reference_timing`模块，两个对应preflight CLI及测试，`spec/s8_entry_recovery.md`，相关README/ARCHITECTURE/LEARNING/主计划与Spec链接、T-173859轨迹/日志。保留全部旧冻结代码和用户已有改动；人工提交时检查上游未跟踪依赖，不使用`git add .`混入无关ACL恢复或其他工作。运行银行/报告保留在本地runs目录，未自动add/commit/push。
