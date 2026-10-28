# 本轮手工 Commit 文稿

```text
feat(experiments): 增加保旧回放分层采样与有界诊断入口

- 补关键任务正标签与源内时间帧，审计实际采样并固定恢复角色序列
- 从同LT16联合240点和Adam进行四臂双顺序seed的24步匹配续训
- 加入数据与基线可测性门禁、闭环保旧出口及准备/预检/互斥保护
- 补充测试、轻量Spec和LT17用户诊断完整化计划，保留短检失败证据
```

主要源文件为新增`experiments/retention_sampling.py`、`scripts/preflight_retention_sampling.py`、`tests/test_retention_sampling.py`。LT15/16冻结执行源与原模型/报告未回写；保留工作树之前已有改动，勿将其误写成本轮新增。未暂存/commit/push。

验证：50项定向测试；真实短检完成数据采集、6份Adam264点及5/8组合完整评测，第2次180秒预算停止；最终入口只读precheck通过，完整诊断尚待用户执行。没有已验证的保旧收益或主目标达标结论。
