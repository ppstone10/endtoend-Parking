# 人工提交文稿

```text
feat(verification): 增加零动作候选补查与独立裁决

仅在LT24安全约束选择全零序列时补查同次已有CEM样本，保持原成本、随机流和非零选择。
新增固定八例、版本身份和负结果门禁；用干净旧控制器重建原400样本与实际补查选择。
短检4到6成功、零碰撞，但余量607到933，故不准备全300、不替换LT24默认行为。
49项相关测试、5文件编译及独立3212实际步/464补查核对通过。
记录前16覆盖和余量可行集的两个缺口，以及已侵入状态需单独退出的边界。
```

本轮范围：verification/zero_recovery.py、zero_study.py、zero_audit.py；scripts/compare_zero_recovery.py；tests/test_zero_recovery.py；spec/zero_candidate_recovery.md；README/ARCHITECTURE/LEARNING仅本轮新增段；当天TRACE/worklog的T-101211。其余已有变更属于前轮工作，人工提交时分别审查。runs下证据按项目产物管理方式保留，不自动加入Git。

未执行git add、git commit或git push。
