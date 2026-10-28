建议标题：

```text
feat(experiments): 增加S6成功策略迁移与旧能力保持配对实验
```

必要正文：

```text
- 冻结训练源53个新增安全策略帧，隔离评测来源，保留旧保持/恢复回放
- 新增双顺序同输入自目标/成功策略目标对照，固定28更新和原生停止策略
- 核验全53输入实际曝光计划、梯度批次身份、Adam240→268及失败关闭
- 完成31项定向测试和真实零更新梯度冒烟，交付LT18人工计划
- 正式四模型44回合尚未运行，不声明闭环收益、晋升或部署
```

本轮范围：`experiments/s6_policy_transfer.py`、`scripts/preflight_s6_policy_transfer.py`、`tests/test_s6_policy_transfer.py`、`spec/s6_policy_transfer.md`、LT18计划，以及README/ARCHITECTURE/主验收计划的对应新增段落、T-145305轨迹与日志。仓库已有大量用户改动和上游未跟踪文件，人工提交时审查依赖闭包及文件选择，不使用`git add .`混入其他工作；冻结配置和运行证据位于本地runs目录，按项目产物策略保留。不自动commit/push。
