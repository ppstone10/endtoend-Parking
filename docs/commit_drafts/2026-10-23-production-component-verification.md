# 人工 Commit 文稿

建议标题：

```text
feat(validation)!: 统一生产闭环分层验证并纳入碰撞与未到达回归
```

建议正文：

```text
让L1感知、L2预测、L3控制及两个耦合验证共用原生E5的全程实际记录、车体扫掠与终止标准。
增加同输入的感知/网络独立重放、从回合起点保持进度/热启动/RNG的MPC连续重放；
分别报告声明契约、结构一致性、参考/实际执行风险和整任务质量，局部通过不自动归因为链接故障。

三个现有验证CLI默认进入统一生产验证，旧行为保留为显式legacy诊断；矩阵汇总不提供生产晋升资格。
新增配置/代码/产物哈希、完整分母、精确基线复现、预算及不覆盖保护。
更新现行验证设计、历史误判说明、Spec和LT21同300人工运行计划，不改变生产模型、感知、MPC或训练。

验证：59项聚合定向测试通过；Windows路径登记修正后17项新模块测试通过；
最终四案例精确复现1成功/2碰撞/1振荡，1288控制步、130预测全程重放与检查；
最终产物inspect、正式300配置precheck、Python静态编译及Spec检查通过。
正式300分层运行尚未执行，用户按LT21运行；未修复或晋升停车策略。

BREAKING CHANGE: 三个evaluate验证CLI的旧参数需要增加--legacy-diagnostic；
默认使用--phase prepare/precheck/run/inspect及统一配置/输出参数。旧函数和历史产物保留。
```

本轮新增：`experiments/component_{verification,capture,analysis}.py`、`scripts/verify_closed_loop_components.py`、`tests/test_component_verification.py`、对应Spec及LT21计划。

本轮修改：三个evaluate入口、run_validation_suite汇总、验证/训练说明和README/DESIGN/ARCHITECTURE/LEARNING相关段落、T-170000日志。

当前工作树还有大量此前任务和用户改动；人工提交时仅选择本轮文件/补丁，不要整仓暂存。本轮未git add/commit/push。运行产物在runs目录保留，不默认纳入Git。
