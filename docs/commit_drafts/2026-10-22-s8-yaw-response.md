建议标题：

```text
feat(experiments): 验证S8航向窗口响应与严格恢复分支
```

必要正文：

```text
- 新增固定事件MPC共同深复制探针，保留参考身份、进度、热启动和随机流
- 固定五事件、0及±0.01rad窗口响应，分开参考几何和实际单步扫掠
- 对唯一局部安全见证验证完整分支：2351原step40为205步严格到达
- 独立重建15响应、精确重放完整分支并验证float32导出和实际首窗
- 完成15项定向测试及S8-ENTRY-007至009追溯，未建银行或训练学生
```

本轮代码为`experiments/s8_yaw_response.py`、`scripts/diagnose_s8_yaw_response.py`、`scripts/validate_s8_yaw_pulse.py`、`tests/test_s8_yaw_response.py`；配套Spec、README/ARCHITECTURE/LEARNING/主验收计划及T-214636轨迹/日志。保留全部原冻结源及其他用户改动；新入口依赖此前尚未提交实验模块，人工提交须核对依赖，不用`git add .`混入ACL恢复目录或无关内容。runs内诊断、快照、独立复核及导出留作本地证据。未自动add/commit/push。
