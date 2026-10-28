# 本轮手工 Commit 文稿

```text
feat(experiments): 增加有效恢复监督与广覆盖回放四臂探索

- 按完整输入/标签审计新增，拒绝重复、异标签和局部目标舍入别名
- 从实际非零访问状态采严格教师首帧，分层回放并在训练前校验覆盖
- 增加四臂双顺序seed、预声明时点选择、旧新双60确认和冻结恢复入口
- 补充定向测试、轻量规范、LT16用户运行计划及使用边界
```

本轮范围：新增`experiments/effective_recovery_data.py`、`scripts/explore_effective_replay.py`、`tests/test_effective_recovery.py`、对应Spec/LT16计划；定向同步README/ARCHITECTURE/LEARNING/主计划及本日TRACE/worklog。保留之前未提交的LT15及其他改动，不把它们误写成本轮新增。没有执行暂存、commit或push。

验证：39项定向测试；最终真实smoke-v3完整22单元/8检查点；正式236来源初态BEV与归档全同及只读precheck。正式采集/训练/双60确认由用户后续运行，尚无闭环收益或主目标达标结论。
