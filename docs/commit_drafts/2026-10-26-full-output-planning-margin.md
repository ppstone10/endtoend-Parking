# 人工提交文稿

```text
feat(verification): 统一闭环候选全部输出分支的规划余量

新增显式PlanningMarginMPC，以同一0.2m完整矩形检查器约束均值、样本、热启动、停车及终端转向。
复用原CEM成本、采样/RNG和LT24参考保持，保留原车位标准；物理0余量由原引擎独立核验。
保留仅约束新增补查的负结果，并增加干净旧控制器的样本/选择/全H10独立复核。
固定八例4到5成功、零碰撞、余量607到0；60项相关测试及10新文件编译通过。
交付165源/32输入冻结的LT25人工同300计划，正式运行待用户执行。
默认入口未替换，GT静态仿真边界保留；已侵入状态退出恢复尚未实现。
```

本轮文件：verification/margin_recovery.py、margin_study.py、margin_audit.py、planning_margin.py、planning_study.py、planning_audit.py；scripts/compare_margin_recovery.py、compare_planning_margin.py；tests/test_margin_recovery.py、test_planning_margin.py；spec/margin_candidate_recovery.md和zero_candidate_recovery.md本轮关联说明；README/ARCHITECTURE/LEARNING本轮新增段；LT25计划；当天TRACE/worklog的T-144155。其他工作树改动属于前轮任务，人工审查时分开处理。runs产物按项目方式保留，不自动加入Git。

未执行git add、git commit或git push。
