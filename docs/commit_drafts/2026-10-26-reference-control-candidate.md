# 人工Commit文稿

建议标题：`feat(verification): 增加合法参考窗口的可行控制候选与分阶段验收`

建议正文：

- 在显式闭环实验中完整保留LT25求解；仅安全介入后的全零解、消费窗口满足0.2m完整车体扫掠时，补充当前参考按原模型推导的控制候选。完整H10安全且原成本严格更低才接受，不增加随机采样。
- 分离六安全教师、14条真实NN闭环和独立审计，保存原400样本上下文、候选与非法窗口跳过，保持旧源/模型/数据/结果冻结；默认入口不变。
- 保留v1负结果：教师6/6但NN5→4丢292；v2定点重放保留292原选择，同时296合法初态仍接受安全候选。v2整组验证由LT26交用户运行，未宣称闭环收益或生产资格。
- 验证：v1相关聚合67项通过；v2相关子集54项通过、补工厂恢复后v2五项通过；11个新增Python文件编译通过。v1全部20回合8785实际转移/H10及候选独立重建一致；v2两个冻结状态精确预检通过，174源/38输入prepare/precheck通过。

范围：新增verification/reference_{candidate,teacher,study,audit}.py、verification/feasible_reference_{candidate,study,audit}.py、两个compare脚本、两个test文件、spec/reference_control_candidate.md、LT26及对应README/ARCHITECTURE/LEARNING/TRACE/worklog更新。已有未提交内容不应整仓一并暂存；runs证据按项目既有归档策略保留。

未执行git add/commit/push；无不兼容公共接口或数据迁移。剩余风险：v2六教师与真实NN保持未验证；GT几何/可立即停车假设尚未接入BEV占用、动态障碍或实车。
