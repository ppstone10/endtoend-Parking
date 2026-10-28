建议标题：

```text
feat(experiments): 准备S8起点参考搜索与固定见证核验
```

必要正文：

```text
- 新增七参数航向参考、NumPy有界DE及矩形SAT次级排名，保留原MPC和安全判据
- 固定2338最多168尝试，2351只复核预检得到的244步严格起点参考
- 提供冻结身份、失败关闭、首窗/导出/独立控制流编排与LT20人工计划
- 修正新工具的全局/局部目标及舍入输入核验，保留失败版本和代码快照
- 完成11项定向测试、四回合smoke、971转移及两个分支独立精确重放
- 正式搜索/导出和独立控制流待用户运行，未训练或建立教师银行
```

本轮直接代码为`experiments/s8_reference_search.py`、`experiments/s8_clearance_surrogate.py`、`scripts/search_s8_start_references.py`、`tests/test_s8_start_reference_search.py`；协同新增`spec/s8_start_reference_search.md`、LT20计划与README/ARCHITECTURE/LEARNING/主验收计划、T-094833追踪/日志。旧149源及其他用户改动全部保留；新入口依赖此前未提交实验模块，人工提交须核对上游依赖，不使用`git add .`混入ACL恢复或其他工作。runs内配置/快照/参考/报告为本地证据，不自动add/commit/push。
