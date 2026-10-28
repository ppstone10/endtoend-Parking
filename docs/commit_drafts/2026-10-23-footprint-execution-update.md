# 人工 Commit 文稿

建议标题：

```text
feat(parking): 增加车体约束控制候选并修正验证契约
```

建议正文：

```text
新增verification所有者，按归档控制命令精确恢复状态和采集标量类型，修正LT21四个L1初态误报。
将声明契约、当步碰撞、历史参考风险及未到达独立展示，保留旧137源/609产物和原入口复现。

核对R0=v40 epoch7权重及训练实际BEV，生成仅校正target声明为image的独立模型副本；
原权重、停止阈值、实际输入和原模型文件保持，抽查预测逐位一致，不计为策略收益。

增加显式FootprintConstrainedMPC：原CEM采样/拟合/成本保持，最终均值接触安全则原样输出；
危险时按原成本从最多16条原采样、移位热启动及零动作中选择完整矩形扫掠安全序列。
原网络/参考过滤×原MPC/约束MPC四臂消融、普通MPC独立重算、冻结身份/产物与联合门禁分开。
短检旧成功丢失或停车替代到达不得算收益；正式prepare要求完整产物及匹配独立审计。
约束读取仿真完整场景几何，BEV占用接入未实现；candidate_qualified仅为该条件下的实验指标，
检查器明确返回geometry_source=simulator_ground_truth、production_qualified=false。

验证：69项定向聚合测试通过，15个新增Python文件静态编译和Spec检查通过。
固定四回合原版1成功/3碰撞，约束控制4成功/0碰撞，1181控制步的选择/成本/窗口/状态独立精确复现。
参考过滤及组合未增加到达，正式不推进；约束控制仍377步侵入0.2m规划余量，不作严格教师或实车保证。
正式control同300已准备LT22（143源/14输入）并precheck通过，尚未运行。
默认控制器/模型/训练未替换，未自动部署或声称完整验收通过。
```

本轮新增：`verification/`九个Python文件、五个新脚本、`tests/test_footprint_execution_update.py`、`spec/footprint_execution_update.md`、LT22计划。

本轮文档更新：README、ARCHITECTURE、LEARNING、验证设计/训练指南及T-191307记录。工作树包含大量此前任务和用户差异，人工提交时只选本轮文件/补丁；`runs/`运行产物和模型副本不默认纳入Git。本轮未执行git add/commit/push。
