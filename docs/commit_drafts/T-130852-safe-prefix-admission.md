# 建议 Commit 文稿

仅人工提交本轮范围；工作树含大量既有改动，请勿直接git add全部。未执行commit/push。

```text
feat(runtime): 引入安全前缀与近端联合目标准入

在显式safe_prefix完整闭环模式中保留连续安全的原样网络前缀，
近端要求原位置/航向联合目标支撑，不足时复用严格回退。
保持权重、停止阈值、MPC、终端接管、0.2m余量和600步不变。

增加原始/发出参考留证、独立前缀和安全审计、冻结交接与LT33计划。
94项相关测试通过；固定24例11→19成功，11个抽查成功全部保持，
LT32丢失的7个历史成功全部恢复，碰撞/余量/危险参考与时域均0。
仍有5例失败；全量300待用户执行，GT混合候选不授予生产或五模块资格。
```

本轮范围：runtime/safe_prefix.py；experiments/validation_matrix.py及scripts/run_validation_experiment.py的safe_prefix分支；verification/safe_prefix_preflight.py、safe_prefix_handoff.py；scripts/preflight_safe_prefix.py、validate_safe_prefix.py；tests/test_safe_prefix_admission.py；spec/safe_prefix_admission.md；README/ARCHITECTURE/LEARNING相关增量；LT33计划和T-130852 trace/worklog。controller和其他既有修改不属于本轮。
