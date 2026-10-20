# 无人矿卡端到端自动泊车系统

从纯 Python 仿真、CARLA 高保真仿真到履带式钻机实物平台逐步迁移的低速自动泊车系统。当前理论车型是以两履带几何中心为旋转/轨迹中心的 6×3 m 居中矩形，允许停车原地转向；外廓、速度、角速度、安全余量和 Hybrid A* 搜索参数均由 JSON 配置。三个阶段共用同一套输入输出接口：原始传感器数据（LiDAR 点云 / Camera 图像）经 Sensor2BEV 转为统一 BEV 表示，端到端网络输出未来轨迹点，MPC 控制器完成轨迹跟踪。

## 环境要求

- 本项目的开发、测试、运行、装包一律使用 conda 虚拟环境 `endtoend-parking`（Python 3.12，位于 `D:\conda\envs\endtoend-parking`），禁止使用本地环境 `C:\Python314`。
- numpy、PyTorch（CPU 版即可）；配置化训练额外使用 PyYAML 与 matplotlib
- 测试使用标准库 `unittest`，无需额外安装

## 安装

```powershell
conda activate endtoend-parking
pip install numpy
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-training.txt
```

## 快速开始

```bash
# 激活环境（或直接用完整路径 D:\conda\envs\endtoend-parking\python.exe）
conda activate endtoend-parking

# 运行阶段一闭环演示：仿真环境 → LiDAR → BEV → 简单跟踪控制
python scripts/run_sim.py
```

```bash
# 运行全部单元测试
python -m unittest discover -s tests -v
```

```powershell
# 生成小批量训练样本（默认 5 条，可传数量参数）
python scripts/generate_dataset.py 5

# 预览 3000 条任务分层计划（不生成 BEV/轨迹）
python scripts/build_dataset.py --count 3000 --dry-run

# 旧 v4/v5 校准和未完成数据保留作旧几何证据；v5 全能力校准已完成 117/117 case，
# 完整证据保留在下列目录（v5 之后进入 v6 几何与对齐采样阶段）
& 'D:\conda\envs\endtoend-parking\python.exe' scripts/calibrate_dataset.py --samples-per-cell 3 --max-retries 2 --task-budget-s 30 --probe-full-capability --vehicle-config configs/vehicles/tracked_drill_rig.json --output runs/dataset-calibration/tracked_pivot_v5_full

# 校准中断后使用完全相同的命令和输出目录恢复；已完成 case 会按 identity 校验后跳过
# report.json、cells.csv 和 run_state.json 可用于一次性判断成功率、超时与剩余量

# tracked_pivot_v7：继承 v6 场景复原，并让普通垂直/斜列车位的目标航向随前进/倒车入位翻转；
# 结构化入口检查整段车辆外廓走廊，相邻占用只选择入口未被堵住的目标，T4 先规划起点参考目标。
# 408 例全准入复核（34 单元×12、关闭重采）为 408/408；v6 检查点不得续入 v7。
& 'D:\conda\envs\endtoend-parking\python.exe' scripts/build_dataset.py --count 3000 --seed 20260824 --vehicle-config configs/vehicles/tracked_drill_rig.json --output data/task_dataset/tracked_pivot_v7_3000 --batch-size 5 --max-retries 10

# 构建中断后使用完全相同的参数重启；已完成检查点会复核后跳过
# 新验收图以“前方 x、车体左方 y”为局部坐标，正 Left 显示在画面左侧；原地旋转标注 body LEFT/RIGHT 与真实有符号旋转弧
foreach ($split in 'train', 'val', 'test') {
  & 'D:\conda\envs\endtoend-parking\python.exe' scripts/inspect_dataset.py "data/task_dataset/tracked_pivot_v7_3000/$split.npz" --output "data/task_dataset/tracked_pivot_v7_3000/inspection/$split" --require-maneuver-consistency --require-trajectory-feasibility
}

# 旧全量数据已清理；仅保留 7 条 tracked_pivot_v3 真实样本用于历史可视化参考，不得用于训练或当作全量备份
& 'D:\conda\envs\endtoend-parking\python.exe' scripts/inspect_dataset.py data/task_dataset/visual_reference_archive_v3/reference_samples.npz --output runs/visual-reference-v3-review
```

```powershell
# 生成数据并训练 MineParkingNet（输出 data_training.npz 与 mineparkingnet.pt）
& 'D:\conda\envs\endtoend-parking\python.exe' scripts/train.py --samples 40 --epochs 30

# v7 定型训练流程 smoke（全量数据、1 epoch，输出到 runs/training/v7-flow-v3/net-v1-smoke）
& 'D:\conda\envs\endtoend-parking\python.exe' scripts/train_model.py --config configs/training/net-v1-smoke.yaml

# smoke 通过后正式训练 net-v1；每 epoch 更新 loss、自由滚动 ADE/FDE、停止率和长度误差
# 中断恢复时只可使用同一训练语义生成的 checkpoint：
# resume_from: ../../runs/training/v7-flow-v3/net-v1/last.pt
& 'D:\conda\envs\endtoend-parking\python.exe' scripts/train_model.py --config configs/training/net-v1.yaml

# 首轮从当前策略闭环访问状态采集专家恢复标签；按源任务原子检查点，可直接重复原命令续建
& 'D:\conda\envs\endtoend-parking\python.exe' scripts/build_recovery_dataset.py --data data/task_dataset/tracked_pivot_v7_3000/train.npz --model runs/training/v7-flow-v3/net-v1/deployment.pt --output data/task_dataset/tracked_pivot_v8_recovery --samples 240

# 从首轮检查点自动补采“碰撞/超时且零恢复”任务；完整回溯最近的规划安全余量状态，保留并去重合并首轮恢复集
& 'D:\conda\envs\endtoend-parking\python.exe' scripts/build_recovery_dataset.py --data data/task_dataset/tracked_pivot_v7_3000/train.npz --model runs/training/v7-flow-v3/net-v1/deployment.pt --output data/task_dataset/tracked_pivot_v8_recovery_final --priority-from data/task_dataset/tracked_pivot_v8_recovery --base-recovery data/task_dataset/tracked_pivot_v8_recovery/recovery.npz

# 从已准入 v7 权重初始化，使用“原专家 + 完整恢复集”和连续净空扫掠损失微调；输出写入 v9-safety-v1
& 'D:\conda\envs\endtoend-parking\python.exe' scripts/train_model.py --config configs/training/net-v1-safe.yaml

# 中断恢复优先使用 last.pt；部署和开环分析使用自动校准后的 deployment.pt
# 完整参数含义、验收门槛和人工单变量调优方法见 docs/training_guide.md

# 需要复核安全损失时，以同一验证集/净空定义统一比较 checkpoint（首项为下降率基线）
& 'D:\conda\envs\endtoend-parking\python.exe' scripts/evaluate_safety_ablation.py --data data/task_dataset/tracked_pivot_v7_3000/val.npz --checkpoint v7=runs/training/v7-flow-v3/net-v1/best.pt --checkpoint v9=runs/training/v9-safety-v1/net-v1/best.pt --output runs/training/v9-safety-v1/safety_comparison.json

# 用同一验证集比较一个或多个 Trainer checkpoint，输出 report.json 与 PNG/PDF 对比图
& 'D:\conda\envs\endtoend-parking\python.exe' scripts/eval_openloop.py --data data/task_dataset/tracked_pivot_v7_3000/val.npz --checkpoint v0=runs/training/v7/net-v0/best.pt --checkpoint v1=runs/training/v7/net-v1/best.pt --checkpoint v2=runs/training/v7/net-v2/best.pt --output runs/openloop-eval/v7

# 对单个 checkpoint 按场景/任务/方向/噪声/相邻占用分组，并输出最差预测—专家叠加图
& 'D:\conda\envs\endtoend-parking\python.exe' scripts/analyze_predictions.py --data data/task_dataset/tracked_pivot_v7_3000/val.npz --checkpoint runs/training/v7/net-v1/best.pt --output runs/openloop-analysis/v7/net-v1/val
```

```bash
# 闭环泊车演示（滚动闭环引擎）：专家轨迹+MPC 基线（M1 地基，成功率验收）
python scripts/run_closed_loop.py --source expert --samples 10

# 端到端主线：按 schema v2 元数据恢复原场景/占用/噪声/车位阈值，deployment→MPC 滚动闭环
# 小规模分层检查；320 点模型默认每 10 个控制周期重规划，报告含整体/分组/逐回合指标
python scripts/run_closed_loop.py --source network --data data/task_dataset/tracked_pivot_v7_3000/val.npz --model runs/training/v7-flow-v3/net-v1/deployment.pt --samples 34 --output runs/closed-loop/v7-flow-v3/net-v1/val-stratified-34-k10/report.json

# 全量评测使用 --samples 0；T5 当前只按静态场景闭环，不代表动态障碍验证
python scripts/run_closed_loop.py --source network --data data/task_dataset/tracked_pivot_v7_3000/test.npz --model runs/training/v7-flow-v3/net-v1/deployment.pt --samples 0 --output runs/closed-loop/v7-flow-v3/net-v1/test-full-k10/report.json

# 安全训练完成后分别保留纯网络和安全门禁两种口径；后者审查完整矩形扫掠并在不安全时专家重规划
python scripts/run_closed_loop.py --source network --data data/task_dataset/tracked_pivot_v7_3000/val.npz --model runs/training/v9-safety-v1/net-v1/deployment.pt --samples 34 --safety-mode none --output runs/closed-loop/v9-safety-v1/net-v1/val-pure/report.json
python scripts/run_closed_loop.py --source network --data data/task_dataset/tracked_pivot_v7_3000/val.npz --model runs/training/v9-safety-v1/net-v1/deployment.pt --samples 34 --safety-mode expert_fallback --output runs/closed-loop/v9-safety-v1/net-v1/val-shield/report.json
```

对已完成的无干预 E5 报告，可回放失败并保存真实初态、MPC 实际参考窗口、控制命令、网络参考、起始/末次 BEV 和碰撞前后状态。诊断不改变原算法，会检查回放结果是否匹配基线；输出目录须尚无最终诊断报告。

```powershell
& 'D:\conda\envs\endtoend-parking\python.exe' -B scripts/diagnose_closed_loop_failures.py --baseline-report runs/validation/live-20261018-170806/full/E5/report.json --scenes S8_weigh_station --render --output runs/diagnostics/failure-replay-s8
```

省略 `--scenes` 可回放全部失败；机制分类仅作候选解释，参考窗口安全不代表运动学可跟踪。后续自主闭环攻坚按 [执行计划](docs/closed_loop_acceptance_plan.md) 推进。

独立选型数据沿用原场景分层、使用新 seed，并对已知训练/恢复任务身份去重；只用于 checkpoint 选型，不替代 S9 最终测试。构建可用相同参数续建，参数或排除来源变化需新目录。

```powershell
& 'D:\conda\envs\endtoend-parking\python.exe' -B scripts/build_selection_dataset.py --template data/task_dataset/tracked_pivot_v8_3000/val.npz --exclude-root data/task_dataset --exclude-root runs/validation --output data/task_dataset/independent_selection_v1 --seed 2026101801
& 'D:\conda\envs\endtoend-parking\python.exe' -B scripts/analyze_prefix_kinematics.py --baseline-report runs/validation/live-20261018-170806/full/E5/report.json --output runs/diagnostics/prefix-paired.json
```

使用 `scripts/compare_selection_candidates.py --baseline <基线报告> --candidate <候选报告> --output <比较.json>` 对同协议无干预 E5 报告联合检查成功与碰撞门槛；该工具不修改权重，选型过线仍需原完整主协议确认。实验性控制器报告被拒绝，不能混入固定控制协议的模型选型。

当前闭环收益出口[LT14固定R0/R2e2完整独立对照](docs/long_tasks/LT-20261020-14-closed-loop-benefit.md)已完成并[独立复核](runs/validation/closed-loop-benefit-20261020-v1/results/independent-review.md)：选型成功200→213、碰撞85→79，有真实收益但不足且S4/S5退化，不晋升/部署/扩大主300或追加旧配方。入口`scripts/validate_closed_loop_benefit.py`保留作可复现记录，不重跑或扫参；71.0%/26.3%不是原主300成绩，原部署仍68.3%/25.7%。

求解器机制实验可使用以下独立入口；`--candidate` 可选 `baseline`、`incumbent`（保留已找到的低代价解）、`correlated`（仅使用时间相关采样）。输出目录必须不存在。省略索引文件表示完整基线索引，须先通过小规模可信改善判断；控制器候选不是原主协议验收，也不证明网络学习能力提升。

```powershell
& 'D:\conda\envs\endtoend-parking\python.exe' -B scripts/probe_mpc_solver.py --baseline-report runs/validation/live-20261018-170806/full/E5/report.json --indices-file runs/validation/independent-acceptance/p2-regression/protocol.json --candidate incumbent --output runs/diagnostics/mpc-incumbent-regression
```

长任务由用户手动执行。恢复数据配比双臂训练及 [8组69条闭环配对](docs/long_tasks/LT-20261019-01-recovery-balance-regression.md) 均已完成并核验，没有候选通过扩大门槛；本对照停止，未替换部署权重。原 [训练计划](docs/long_tasks/LT-20261018-01-recovery-balance-ab.md) 保留追溯。`scripts/prepare_recovery_balance_ab.py` 只写新任务的数据/配置；`scripts/train_fixed_threshold.py` 用固定父阈值保存逐轮快照，支持显式resume，不自动校准、部署或启动闭环评测。默认训练入口行为不变。

续训退化调查确认了归一化yaw标签与普通实数航向损失不一致。训练配置新增显式 `yaw_error_mode: wrapped`（最短角差），默认 `raw` 保留历史行为，禁止跨模式恢复优化器。[W0/W1训练](docs/long_tasks/LT-20261019-02-yaw-loss-ab.md) 与 [8组69条闭环](docs/long_tasks/LT-20261019-03-yaw-loss-regression.md) 已完成并独立核验，W0逐轮/逐回合复现旧R2；W1四轮未降低碰撞、均未通过联合筛选，本单变量停止，不追加训练、不扩大300条、不部署。下一步先诊断真实失效状态与实际MPC窗口。由于执行代码已变，旧冻结计划会拒绝重跑/续跑；不要改旧manifest绕过。

分层诊断入口 `scripts/diagnose_reference_execution.py` 必须显式指定任务索引，保存原预测、MPC实际窗口/控制序列和真实推进，分别检查窗口内部、状态连接和执行风险，不改变控制器。已完成S4/S8的7条任务×3模型共21回放；[诊断与训练方向](runs/diagnostics/reference-execution-20261019/analysis.md)表明参考危险与执行偏离可先后叠加，下一步先验证训练域恢复标签在固定MPC下可执行，不直接重训。需要批量长诊断或训练时仍先交人工运行计划。

训练域跨重规划[12源任务验证](docs/long_tasks/LT-20261019-04-recovery-execution-gate.md)已完成，网络后缀精确复现；S4专家安全恢复6/6、S8严格余量通过0/6且缺成功对照，不进入训练。后续[参考兼容性诊断](runs/diagnostics/s8-reference-compatibility-20261019-v2/analysis.md)确认原求解近似误差。用户同意离线求解器试验，首版[69条双臂](docs/long_tasks/LT-20261019-05-mpc-candidate-regression.md)已完成并联合失败，停止该候选；原MPC/默认入口不变，不训练、不部署。新代码全集身份变化后不要改旧hash续跑，使用新冻结配置。

上一候选保留版和当前搜索样本方案的69条双臂均已完成且未通过联合门槛，全部停止；[LT-20261019-06](docs/long_tasks/LT-20261019-06-mpc-search-best-regression.md)保留结果。原MPC、上一版代码和模型均不变。

训练域参考更新验证入口`scripts/validate_reference_updates.py`观察同状态对齐窗口，并从复制的热启动/RNG先精确复现正常后缀，再仅延迟一次参考更新，随后恢复网络重规划。[LT-20261019-07](docs/long_tasks/LT-20261019-07-reference-update-stability.md)正式24源已完成：19对完整后缀复现，605次更新有13次规划/2次接触安全回退，延迟没有改变7成功/12碰撞分类。机制覆盖满足但不代表稳定或可执行；[独立复核](runs/diagnostics/reference-update-stability-20261019-v1/results/agent-audit.md)区分首次回退与分叉错位、S8首次重规划前执行偏离。停止重跑原计划，不自动训练或部署。

事件对齐入口`scripts/probe_reference_events.py`的[LT-20261019-08](docs/long_tasks/LT-20261019-08-reference-event-probe.md)11源正式验证已完成并[独立复核](runs/diagnostics/reference-event-probe-20261019-v1/results/agent-audit.md)：7全程事件组正常/延迟均2成功5碰撞，专家一次恢复#1455却丢失2成功对照；S4专家前10步均不失余量，退化在恢复网络后。S8初态专家仅1/4保持前10步余量，其余安全窗口仍执行越界。短窗不作全程标签，不重跑旧计划、不直接训练、不改默认MPC。

连续恢复前置入口`scripts/validate_recovery_compatibility.py`的[LT-20261020-09](docs/long_tasks/LT-20261020-09-s4-continuous-recovery.md)六源已完成并[独立复核](runs/diagnostics/recovery-compatibility-20261020-v3/results/agent-audit.md)：四恢复源仅1559/1441严格通过，1298到达但损失3步余量，1455执行越界后于相对step50重规划失败；两教师成功对照严格到达。53条已应用专家参考均几何安全，但两首事件安全实际窗口仍执行越界，说明原MPC兼容性也限制S4。未过4/4数据门槛、不生成训练集或开训；学生保留未评估，S8失败cost候选继续停止，原MPC/模型不改。

原代价求解精度对照入口`scripts/compare_cem_polishing.py`的[LT-20261020-10](docs/long_tasks/LT-20261020-10-cem-polishing.md)六源四分支已完成并[独立复核](runs/diagnostics/cem-polishing-ab-20261020-v1/results/agent-audit.md)：精修教师6/6严格成功，但纯网络从2成功/4碰撞退为1成功/5碰撞，#1334丢失，联合门槛失败。其危险窗口先于实际越界；同状态下精修也出现cost更低而首步不安全，低cost不是安全保证。停止扩大、不直接训练或部署，默认控制器不改；教师可执行不等于原MPC监督可执行。

训练前停点对齐入口`scripts/audit_terminal_alignment.py`的[LT-20261020-11](docs/long_tasks/LT-20261020-11-terminal-alignment.md)原train2400条已完成并[独立复核](runs/diagnostics/terminal-alignment-train-20261020-v1/results/agent-audit.md)：部署点达标1144、错停但完整预测有联合点920、无联合点336。错停中147条只在真值长度之外有合格点；无联合点中179条位置/航向合格时刻错开。结果仅诊断，不能当闭环成绩或直接改成oracle停点标签；不训练或改阈值、不绕过原MPC标签兼容性。

独立停止头入口`scripts/train_frozen_stop_head.py`的[LT-20261020-12](docs/long_tasks/LT-20261020-12-frozen-stop-head.md)两臂20轮已完成并[独立复核](runs/training/frozen-stop-head-ab-20261020-v1/results/agent-audit.md)：40个checkpoint仅改变129参数停止头，原2400自由轨迹与LT11一致，但部署停点开环合格数由父1144降为guided774/free184。free平均BCE下降同时前缀误触发增加，目标与首次停止/位姿验收不一致；候选失败，停止扩大、不扫epoch/阈值、不进入闭环或部署，默认模型/MPC保持原状。

后续[停止目标短审计](runs/diagnostics/stop-objective-audit-20261020-v2/summary.md)已完成：`scripts/audit_stop_objectives.py --output <新的独立目录>`在指定conda环境只读计算固定head的损失与梯度，不训练。事件前缀/最坏前缀两个候选仍存在总体或分组取舍，尚未进入新训练；拒绝覆盖已有输出，公式及边界见[规格](spec/stop_objective_audit.md)。

进一步的[完整池跨组审计](runs/diagnostics/stop-cross-effects-20261020-v1/summary.md)由`scripts/audit_stop_cross_effects.py --output <新的独立目录>`执行：固定完整池方向检查分组损失和原1144首停索引解析保护范围，无参数更新。最坏前缀方向仍有原合格组前缀风险与S8事件风险，尚不能直接重训；28相关测试通过。

[原合格保护可行性审计](runs/diagnostics/stop-direction-feasibility-20261020-v2/summary.md)入口为`scripts/audit_stop_direction_feasibility.py --output <新的独立目录>`，只检查三个旧方向非负组合，不训练。总体条件有见证，但分组条件及两种原合格保护均有该方向族内的不相容证据；34相关测试通过。停止旧方向混权路线，该有限族结论不代表完整129维停止头无解。

[完整129维检查](runs/diagnostics/stop-fullspace-extended-20261020-v1/results/agent-audit.md)及[LT13延长审计](docs/long_tasks/LT-20261020-13-stop-fullspace-feasibility.md)已完成独立复核：两无保护方向成立，两原合格保护仍4096迭代耗尽，无受保护方向见证，不直接训练。合法对偶上界与约6.60e−5的原首停索引解析保持区间已确认；其后有限损失/float32单步检查见下项，不再盲目加迭代，模型与标签未改。

[单次有限步长验证](runs/diagnostics/stop-finite-step-20261020-v1/summary.md)已完成，入口`scripts/probe_stop_finite_step.py`只取解析上界的1/4：float32下原1144缓存开环合格全保留，八项损失约10⁻⁷下降，但全部2918缓存停点及8条原生predict都不变、无新增合格，故不训练/部署或扩大闭环。完整2400原生B1未复评，默认模型不变。

[终点合格区间审计](runs/diagnostics/stop-endpoint-intervals-20261020-v1/summary.md)入口`scripts/audit_stop_endpoint_intervals.py --output <新目录>`只读固定方向的float64缓存区间，不更新参数：允许合格集合内换停点后初始无损范围扩大15.2倍，但首次损失原合格早于首次潜在新增，初始无损区间新增0。当前方向不继续放大或训练；不代表其他方向无解，也不是实际float32或闭环成绩。

## 目录结构

```
interfaces/    统一接口定义（传感器帧、BEV、车辆状态、轨迹、控制指令）
sensor2bev/    Sensor2BEV 环境表示模块（LiDAR/Camera → BEV）
sim/           Python 仿真环境（二维矿区、车辆运动模型、模拟传感器）
model/         MineParkingNet v0/v1/v2、变长轨迹输出与模型注册表（PyTorch）
training/      安全 YAML 配置、数据准备、完整车体扫掠损失、训练/验证、early stopping 与原子 checkpoint
controller/    MPC 轨迹跟踪控制器（CEM 滚动时域优化）
planner/       履带 Hybrid A* 专家轨迹（前后弧线、原地转向、RS/履带解析终连、完整扫掠碰撞）
dataset/       全能力校准、Task 分层/划分/重采、闭环恢复重标注、专家轨迹与融合 BEV、schema v2 统计/验收图
configs/       可编辑理论钻机与实验 JSON 配置
runtime/       滚动闭环引擎（轨迹源→安全门禁/专家回退→MPC→车辆，终止判定与失败分类）
metrics/       回合指标定义与聚合（成功率/碰撞率/误差等）
scripts/       运行脚本
tests/         单元测试
docs/          任务日志与修改轨迹
spec/          行为规范
```

## 系统流程

```
LiDAR点云 / Camera图像
      ↓ 传感器适配器
      ↓ Sensor2BEV 环境表示模块
BEV + 车辆状态 + 目标泊车位姿
      ↓ MineParkingNet 端到端轨迹生成网络
未来N个局部轨迹点
      ↓ MPC 轨迹跟踪控制器
[v_cmd, omega_cmd]
      ↓ 平台执行器
Python车辆 / CARLA车辆 / 履带车底盘
```

详细设计见 `DESIGN.md`，模块与数据流见 `ARCHITECTURE.md`。

默认钻机配置位于 `configs/vehicles/tracked_drill_rig.json`。它是理论等比模型，不代表某台实车标定；后续可直接修改外廓、底盘执行上限、规划速度/角速度、碰撞余量、搜索分辨率、原地旋转开关与代价。新正式数据会记录完整配置和模型版本，参数变化后必须重新生成或重新审计，旧数据不会自动视为兼容。

## 开发方法

本项目使用自适应 Agent 开发工作流（见 `AGENTS.md` 与 `.agents/`）。
