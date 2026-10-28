feat(runtime): 接入共享余量约束与终端转向证书

- 新增显式strict_footprint完整闭环：实际发出参考、选中H10和实际转移共享VehicleConfig规划余量，当前0.2m；保留none原生对照。
- 复用原CEM搜索/候选，危险输出改选安全样本、移位热启动或零动作；原位置容差内的纯朝向接管需完整转角及H10余量证书。
- 执行器把专用控制安全异常转为安全停止，引擎覆盖首次/替换提议并保存已执行前缀；普通错误继续传播。CLI明确应用用户选择的安全模式。
- 新增冻结配对/落盘独立审计入口、严格模式Spec及LT32人工计划。最终九例原生2到达/5碰撞→新模式8到达/0碰撞/0余量，16终端证书独立一致；83相关测试通过。
- GT几何与专家回退属于混合系统收益，未改训练/模型/停止阈值，未运行完整300或授予生产/五模块资格。

直接代码：controller/constrained_mpc.py、terminal_capture_mpc.py、__init__.py；runtime/strict_footprint.py、execution.py、engine.py；experiments/validation_matrix.py；scripts/run_validation_experiment.py、validate_strict_footprint.py；verification/strict_footprint_study.py、strict_footprint_audit.py；tests/test_strict_footprint_runtime.py。
文档：spec/strict_footprint_runtime.md、README.md、ARCHITECTURE.md、LEARNING.md、LT32计划及T-090318 trace/worklog。

工作树含大量原有修改，人工只暂存本轮相关差异；无需把runs生成产物批量加入Git。源码快照与冻结产物原地保留。不自动commit/push。
