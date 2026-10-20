# 原代价CEM局部精修

## 元数据

L3轻量规格；仅离线实验，默认controller不变。

## 目标

在相同原MPC代价下改善有限CEM求解误差，验证两个首危险事件及原六源闭环是否改善。

## 非目标

不改变参考时序/控制限幅/原cost权重/原CEM种群迭代，不加碰撞约束、环境真值、几何筛选、全零候选，不调阈值/降低余量/训练/部署。

## 边界与约束

现有PyTorch CPU库，无新依赖。只读状态/参考/原CEM序列；几何只在离线记录中检查，不影响返回。局部优化/低代价不保证全局最优或安全。

## 行为与验收

### CEM-POLISH-001：原代价与边界一致

双精度可微原cost必须与MPCController._batch_cost一致，含原欧拉动态、环绕yaw、stage/terminal及控制惩罚；控制通过limits*tanh(z)保持原限幅。从原CEM均值起跑，不从事后安全解起跑。每步原CEM完整执行且RNG消耗不变，平局保留原解，最终用原NumPy cost复核严格更优且有限才返回。不得引用checker/环境/安全标签。

### CEM-POLISH-002：预算、失败和快照

固定L-BFGS最大60迭代、80次closure评估、history10、strong_wolfe、tolerance_grad1e-9/change1e-12。独立closure硬上限不能因线搜索超过80；到限可用已评估最优，数值非有限回退原解，非预期异常不吞掉。另有原cost前后NumPy复核，不混报为原搜索预算。max_evaluations=0精确复现原解。复制MPC保持热启动/参考对象与缓存别名关系/RNG，不改原实例。保存每步原CEM序列/cost、返回cost、次数、接受与停止原因。

### CEM-POLISH-003：固定窗口与完整闭环分开

1298相对46/1455相对48固定窗口确认原cost更优及事后首步/rollout几何，不能当闭环改善。正式固定LT09六源/上下文，先复现完整原源与原网络/持续教师两臂，再运行精修两臂；每一来源首次调用的状态/窗口/未精修CEM cost应与原臂相同。后续精修改变状态及参考属真实反馈，不强求原轨迹一致。正常网络与持续教师指标不混报，专家轨迹不是训练集。

### CEM-POLISH-004：冻结、门槛和人工长任务

冻结全部使用输入/代码/运行时，拒绝旧输出/漂移，失败保留，无自动重试/扩大/训练/部署。短预检强制不通过。正式六源仅可进入独立回归方案评审：精修教师四恢复源及两成功对照全程严格通过，精修纯网络保留原两成功且碰撞不增；否则停止扩大。教师通过不等于纯网络70%/10%达标，也不等于原MPC标签已可执行。

## 追溯

| ID | 实现/验证 | 状态 |
|---|---|---|
| CEM-POLISH-001 | experiments/cem_polishing.py；tests/test_cem_polishing.py | ✅ 原cost/梯度与返回规则通过 |
| CEM-POLISH-002 | CEMPolishedMPC.from_controller/polish_controls；tests/test_cem_polishing.py | ✅ 硬预算/禁用/失败/复制隔离通过 |
| CEM-POLISH-003 | experiments/polishing_execution.py；scripts/compare_cem_polishing.py；runs/diagnostics/cem-polishing-ab-20261020-v1/results/agent-audit.md | ✅ 六源四分支完成，30回合2770转移/1475控制记录及109教师参考独立复核；原臂完整复现LT09 |
| CEM-POLISH-004 | 冻结CLI/门槛测试；docs/long_tasks/LT-20261020-10-cem-polishing.md；results/agent-audit.md | ⚠ 79输入/97源身份及停止规则已验；教师6/6严格，但网络保留1/2、碰撞4→5，联合门槛失败，停止扩大 |
