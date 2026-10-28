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

L1/L2/L3 与两个耦合验证共用原生 E5 的全程时序、车辆扫掠和终止判据。[LT21 同300结果](runs/verification/component-v1-formal-20261023-v2/result/summary.md)为205成功/77碰撞/18振荡，仍未通过≥70%/≤10%门槛。原三个`evaluate_*`入口和v1源码保持冻结复现；旧参数需显式`--legacy-diagnostic`。

当前修正版用`scripts/review_component_verification.py`只读迁移旧证据，[v2报告](runs/verification/component-v2-migrated-20261023-v2/summary.md)精确重建76,282步状态/标量类型，修正4条L1初态误报；错误不是随机降采样，360束未触发10000点门槛。契约、当步碰撞、历史风险和未到达分别展示。[训练输入审计](runs/verification/input-contract-corrected-20261023/report.json)生成仅将target声明对齐image的模型副本，权重/停止阈值/实际输入不变，原模型保留。

五个验证单元已统一主要运行口径，质量尚未全部通过。[新版候选五层验收](spec/candidate_layer_acceptance.md)已补实际BEV/局部预测/模型输入归档、独立重放与passed/failed/blocked分级；[LT27](docs/long_tasks/LT-20261026-27-candidate-five-layer-acceptance.md)已完成训练源经验参考校准（1780任务/24组：16 ready、3 failed、5 blocked）与固定14全输入补证：采集`observer_exact_to_LT26=true`、五层接口全passed，但**L2/L1_L2在14/14例failed（完整参考危险）、L1有1 failed+4 blocked、L3全部passed、L2_L3随任务成败**，`all_five_modules_qualified=false`。入口`scripts/verify_candidate_layers.py`和`scripts/calibrate_perception_acceptance.py`；经验包络不等于绝对感知质量，L1/L1_L2绝对门槛未建立仍不授予资格。原生E5旧报告和LT26结果不改，见[短预检](runs/verification/candidate-layers-pilot-20261026/review.md)。

环视几何已修复为保留各视角图像/标定后分别投影到统一车体系，并修正正深度裁剪。新image输入声明`camera_projection: per_view_v2`；旧checkpoint缺省仍为`legacy_overlay_v1`，遇到新输入会拒绝推理，不能只改声明冒充适配。新训练配置须显式使用`target_channel: image`与`camera_projection: per_view_v2`，train/val均须完成同版本重建。`scripts/repair_camera_targets.py`只在新目录修复target，保留标签、恢复帧顺序和其他通道，`--inspect`独立核验；恢复源可通过`--original-data`提供原任务归档，严格认证索引/ID并区分原任务机动与恢复监督标签。[LT29重跑](docs/long_tasks/LT-20261027-29-recovery-lineage-target-rebuild.md)已完成全量重建：train 2918条74.71秒、val 300条7.77秒，两次inspect `passed`，独立复算标签/任务顺序/其他四通道逐值保持、仅target更新（2318/248条），`recovery_labels_different`=83/0。数据位于`runs/verification/surround-target-rebuild-lineage-20261027/{train,val}`。[完成反馈后的独立复核](runs/verification/lt29-independent-review-20261027/review.md)确认全量保持与指标复算，目标在BEV内非空分母2884/295，平均IoU由0.280/0.275升至0.858/0.875；残余漏检、视野及边缘采样限制仍须保留。[LT30配对训练](docs/long_tasks/LT-20261027-30-surround-input-paired-training.md)已完成：同seed冷启动两臂各20轮/7300更新（2918训练+300验证），原生`last.pt`（epoch19、阈值0.5）开环300条比较——**ADE 0.3685→0.2593 m（−29.6%）、FDE 0.4976→0.3187 m（−36.0%）**，但航向MAE 0.2056→0.2082 rad（+1.3%）、预测长度MAE 16.83→22.33点（+32.7%）变差，**forward FDE 0.2538→0.3141 与 S8 FDE 0.1199→0.1685 出现退化**；收益主要来自reverse（ADE 0.4231→0.2475）。[独立复核](runs/verification/lt30-independent-review-20261027/review.md)进一步发现：上述FDE按专家长度取点，模型实际停止点均距反由0.973增至1.695m、静态联合匹配28/300降至20/300；终点损失与停止长度的L2内部对齐缺口已确认，LT30权重中尚未解决。**闭环收益与旧205成功保持仍未测**：碰撞/规划余量/联合到达须另立计划验收，单seed开环改善只作下一阶段证据，不晋升、不替换默认入口。见[配对训练规范](spec/surround_input_training.md)、[几何契约](spec/surround_camera_projection.md)；LT28因恢复标签误作任务采样参数失败，83条并非镜像增强，样本分母保持2918。[LT27旧证据](runs/verification/lt27-independent-review-20261026/review.md)保持不变，冻结实验须使用原代码快照。

实际输出末点诊断现已接入`scripts/analyze_predictions.py`：保留专家长度ADE/FDE，新增原生发出末点的位置/航向、联合目标支撑、后缀点数及全量`sample_metrics.json`；缺容差不算通过。可选`terminal_hold_weight`默认0，启用时以专家末位姿监督后缀并同步补齐teacher输入；旧checkpoint缺字段按0兼容，跨策略恢复拒绝。[LT31配对](docs/long_tasks/LT-20261027-31-terminal-hold-paired-validation.md)已执行完成并给出**负结果**：两臂同LT30权重各4轮/1460更新，hold（`terminal_hold_weight=0.25`+专家末点teacher补齐）相对同剂量control几乎全面退化——实际末点联合通过84→**14**、停止前联合支撑253→**15**、ADE 0.1736→**0.3403**、FDE 0.1449→**0.3150**、原205成功输入静态联合61→**13**；12项预声明筛选仅2项为真，`stage_screen_passed=false`。[独立复核](runs/verification/lt31-independent-review-20261027/review.md)进一步确认hold有285条早于专家长度结束输出，260条在首次联合达标点前停止；但未发出的完整320点中有275条曾联合达标，第320点均距由control的7.887m降至1.736m，说明末端坐标保持部分学到、停止决策却失配。固定原生输出交叉仅作归因，不能当候选成绩。该hold配方停止扩大；control静态指标改善可保留为待验证候选，真实闭环/保旧仍未测。见[规范](spec/emitted_terminal_alignment.md)。

显式完整车体约束MPC候选已完成同300正式条件实验：[LT22](docs/long_tasks/LT-20261023-22-footprint-constrained-control-validation.md)中243成功/0碰撞（原生205/77），但旧成功#168丢失，`candidate_qualified=false`；16.2%控制步需干预，0.2m规划余量侵入17,337步（原生3,164步），网络危险参考仍未修。控制约束使用仿真完整几何、BEV占用接入尚未实施，`production_qualified`固定false，默认控制器未替换。入口为`scripts/compare_footprint_execution.py`，读后核验用`scripts/inspect_footprint_execution.py`。参见[验证设计](docs/closed_loop_verification_design.md)与[更新Spec](spec/footprint_execution_update.md)。

终端朝向接管候选仅在原位置容差内且全角程车体扫掠安全时原地对齐目标，严格联合验收不变。[LT23](docs/long_tasks/LT-20261023-23-terminal-capture-validation.md)同300已完成：**255成功/0碰撞**，原205与LT22的243均零丢失、新增12（S4 8/S6 3/S5 1）、#168恢复；终端接管共43个回合109步，余量侵入14,944步低于LT22的17,337（按步归一0.164<0.182），`candidate_qualified=true`但`production_qualified=false`。仍有45条振荡/超时失败（S4 25、S6 19、S8 1），网络危险参考11,034步未修。入口`scripts/compare_terminal_capture.py`提供prepare/precheck/smoke/run/inspect/audit/diagnose；仍依赖仿真完整几何，不是纯网络改进或BEV约束已接入。参见[终端契约](spec/terminal_capture.md)。

带到达证书的参考保持候选用私有LT23控制上下文验证原网络参考能在剩余预算内、以0.2m余量严格到达，才保持同一参考对象；感知/推理仍按原节奏运行。[LT24](docs/long_tasks/LT-20261026-24-certified-reference-retention.md)同300已完成：**263成功/0碰撞**，LT23的255成功零丢失、新增8（全部S6，#267由振荡变420步到达），余量侵入14,944步不增但按步归一0.1640→0.1731（分母因提前到达变短，须单列）；`candidate_qualified=true`、`production_qualified=false`。291次尝试/192证书/158次保持，192份证书与实际账本逐步一致。仍有37条失败（S4 25、S6 11、S8 1）与11,035步危险参考窗口。入口`scripts/compare_reference_retention.py`；仍为GT几何辅助条件实验，无BEV或生产结论。见[参考保持Spec](spec/certified_reference_retention.md)。

零动作后的同次候选补查仅完成固定八例短检：4→6成功、0碰撞，但0.2m余量侵入607→933，联合门槛失败，**不进入全300、未替换LT24**。原前16样本截断与物理可行集未约束规划余量均有独立重建证据；入口`scripts/compare_zero_recovery.py`及[短检裁决](runs/verification/zero-recovery-review-20261026/review.md)用于复核，正式prepare会拒绝该短检。参见[补查契约](spec/zero_candidate_recovery.md)。

完整输出分支统一规划余量候选将均值、样本、热启动、停车及终端转向全部纳入同一0.2m矩形约束；只约束新增补查的中间版已被完整闭环短测否定。[LT25](docs/long_tasks/LT-20261026-25-full-output-planning-margin-validation.md)同300已完成：**268成功/0碰撞、规划余量侵入0步**（LT24为14,944步），但丢失LT24的#131/#132/#139/#161/#279/#296六条旧成功、新增11条（S4×8、S6×3），净+5，`candidate_qualified=false`——当前候选未同时满足严格余量与旧成功保持（#131亦为原生成功），不证明安全路径不存在。全部83,710实际步及完整H10已独立重算，六条丢失旧路径均曾侵入余量。入口`scripts/compare_planning_margin.py`，默认控制未替换，已侵入状态恢复未实现、仍非生产资格；见[契约](spec/margin_candidate_recovery.md)与[正式独立复核](runs/verification/lt25-independent-review-20261026/review.md)。

合法参考下的MPC候选覆盖修复入口为`scripts/compare_feasible_reference_candidate.py`，原LT25全零且消费窗口满足0.2m扫掠时才补安全、更低原成本候选。v1真实14例5→4、丢292；[LT26](docs/long_tasks/LT-20261026-26-feasible-reference-candidate-validation.md)的v2六教师6/6合格，但真实14例仍5成功、无新增，未恢复LT24子集六条。14条中12条动作/状态完全等于LT25；九失败末轮完整NN参考全部不满足连接余量，八条无原车位联合达标点。两版均未获条件资格、不进入全300；下一步重点是实际交付参考的可执行性和末段目标支撑。默认入口/模型不变，见[契约](spec/reference_control_candidate.md)与[LT26独立复核](runs/verification/lt26-independent-review-20261026/review.md)。

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

新方案[成功保持与可执行教师短预检](runs/diagnostics/behavior-repair-preflight-20261021-v2/summary.md)已执行，入口`scripts/preflight_behavior_repair.py --output <不存在的新目录>`；实验组合教师显式添加`--refine-teacher --rolling-s4`，不修改默认训练器/MPC。组合教师在固定S4/S8源上已严格到达，但12步拟合后5条未拟合纯网络探针仍2成功/2碰撞/1振荡，没有闭环收益，不生成长任务计划。不要把该入口当已验证有效的训练配方或反复换目录扫结果；旧固定教师失败结果继续保留。

按后续用户要求运行的[LT15连续闭环学习探索](docs/long_tasks/LT-20261021-15-closed-loop-learning-exploration.md)已完整完成并[独立复核](runs/diagnostics/learning-exploration-20261021-v1/independent-review.md)：24.7分钟、16单元/28检查点，两个最终方向在60条确认上均退化，不晋升。聚合各追加27帧实际与旧库完全重复，有效新输入/标签为0；恢复专训某60步点虽救回S8但丢旧S4成功，不能断言模型不可学习。入口及冻结产物保持原样，不重新跑本版本找好结果。

[LT16有效恢复监督×广覆盖回放](docs/long_tasks/LT-20261021-16-effective-recovery-replay.md)已完成并[独立复核](runs/diagnostics/effective-replay-20261021-v2/independent-review.md)：48.3分钟、158单元/24检查点，数据门禁通过（62有效新查询＋240回放帧/60成功源）。联合臂在旧/新两套确认各四组合均成功增加、碰撞减少，但丢3–4条旧成功、发现及部分场景门槛失败，不晋升；主成绩未变。已定位S4-T2、S6-T5前进监督空缺及保持采样/时序覆盖不足，但修补的因果收益未验证。冻结入口`scripts/explore_effective_replay.py`与产物保留，不重跑旧配方找好结果、不自动部署；下一步先做保旧短预检，不直接加长训练。原运行摘要的88/27帧、60/12步笔误以独立复核勘误为准。

[LT17诊断完整化](docs/long_tasks/LT-20261021-17-retention-sampling-diagnostic.md)及[独立核验](runs/diagnostics/retention-sampling-diagnostic-20261021-v2/independent-review.md)未过双seed门槛。[消费保障](runs/diagnostics/retention-consumption-preflight-20261021-v1/summary.md)取得4成功/2碰撞、6/1，[窗口质量对照](runs/diagnostics/retention-quality-preflight-20261021-v1/summary.md)仍退化；这些单顺序成绩不能作主验收。后续[严格保持教师](runs/diagnostics/retention-teacher-preflight-20261022-v1/summary.md)发现C240在原四源仅4/8成功，按门禁停训练；[来源分工同输入对照](runs/diagnostics/retention-teacher-roles-preflight-20261022-v1/summary.md)全R0目标5/1、4/2，分工目标4/2、3/3，均不放行，停止扩大本配方。

[S3/S6实际失败状态教师](runs/diagnostics/learner-failure-teacher-preflight-20261022-v1/summary.md)26.353秒完成9查询，S6三个查询均严格恢复并得到3个有效新增首帧，S3六查询均不合格。[S3固定参考及两次既有航向校正](runs/diagnostics/failure-teacher-refinement-preflight-20261022-v1/summary.md)另用10.120秒完成：6次均零接触到达、但侵入余量11–19步，不收正标签。独立入口`scripts/preflight_learner_failure_teacher.py`（180秒）及`scripts/preflight_failure_teacher_refinement.py`（120秒）的`--precheck`只核验身份/路径，不运行回合/规划/训练；实际执行须不存在的新`--output`，不重跑旧目录找好结果。S6后续学习对照见下段；S3仍须解决几何教师与原MPC执行兼容性，不直接加步数或降低余量。旧122源及结果不改，主成绩不变，不自动长训或部署。

[S6纠偏配对学习](runs/diagnostics/s6-recovery-learning-preflight-20261022-v1/summary.md)已完整运行111.299秒、4模型/44回合：同输入self控制4成功/2碰撞、5/1，strict目标5/1、4/2；目标振荡源顺序20仍未恢复，顺序21丢S8成功，不放行。入口`scripts/preflight_s6_recovery_learning.py --precheck`只核验缓存/自由预测/计划，实际180秒短检须新`--output`。同一失败上下文也未学会恢复；step400首停仍35而教师19，原轨迹mask仅0–10，原教师19点固定执行31步严格到达、全段消费0–18。后续终段监督对照见下段，旧125源不改。

[完整终段监督对照](runs/diagnostics/s6-terminal-consumption-preflight-20261022-v1/summary.md)70.821秒完成两新模型/22回合，仅扩step400单槽mask，首窗5/1、4/2变为3/2、4/3，未救回原S6振荡，不放行。入口`scripts/preflight_s6_terminal_consumption.py --precheck`核验原配置/实际排程精确重建与同19点执行证据，无训练/回合；120秒实际短检须新`--output`。另48.806秒单查询120更新诊断虽降误差但仍不能局部恢复，且原7挑战0成功/5碰撞；已知教师长度oracle可无接触到达但58步侵入余量，均非候选/自主首停收益。停止扩大该mask/专训路线，后续联合目标对照见下段。旧127源、主成绩与安全门槛不变，无有效长训或自动部署。

[停止边界与终点位姿对照](runs/diagnostics/s6-terminal-objective-preflight-20261022-v1/summary.md)114.725秒完成四模型/44回合；仅改单个完整19点查询的实验损失，boundary_only双顺序5成功/1碰撞、3/4，joint5/1、4/2，均未放行。四模型同原查询自主长度仍35（教师19），目标S6顺序20未救回、joint顺序21还丢其原完整回合成功。入口`scripts/preflight_s6_terminal_objective.py --precheck`零训练/回合核验原配置/梯度身份和唯一目标槽；新`--output`固定180秒。24测试、131源/53产物及四S6逐元素重放/同状态迁移已复核；不继续该24步加项配方或生成扩训计划，不据此断言容量不足，默认Trainer/网络/MPC不变。

[优化传输与局部学习性](runs/diagnostics/s6-event-learnability-preflight-20261022-v1/summary.md)进一步复现六训练路径，确认原纠正步并无断梯度但实际停止logit推进远小于阈值缺口，清动量反而更少；新同120步隔离能把长度35→25/32，却仍自主失败。联合模型前19点已可在原MPC28步严格到达；单例离线bias选中同参考后，原生滚动仍在首次更新后分岔并振荡，7挑战0成功，非候选。下一方向转为连续重规划恢复/一致性预检，不推荐全局bias或继续单帧加项。入口`audit_s6_update_transport.py --precheck`、`preflight_s6_event_learnability.py --precheck`均在`scripts/`下、零梯度/回合，实际新output预算180/150秒；33测试与完整复核通过，无长训/部署，主成绩不变。

[首次更新与连续恢复链](runs/diagnostics/s6-chain-learning-preflight-20261022-v2/summary.md)已核实410/420参考均车体安全但终点持续漂离；同点重发布仍严格成功，进度重置不是本例充分原因。410真实状态的专家17点导出已严格执行。交替400/410各60次的原生模型仍失败；配已有单例bias后S6#2149从起点539步严格成功，固定交叉对照支持覆盖与选择共同作用，但7挑战仅1成功/5碰撞，不晋升。取得训练源#2149的53个clean新非零成功策略帧，评测源不回灌。`preflight_s6_replanning_chain.py --precheck`与`preflight_s6_chain_learning.py --precheck`均在`scripts/`、无训练/回合；29测试及137源/31产物/关键回合精确复核通过。下一关口为离线恢复教师迁移、原生停止与旧成功保持，不把全局bias作正式改法，主成绩不变。

成功策略迁移 [LT-20261022-18](docs/long_tasks/LT-20261022-18-s6-policy-transfer.md) 已完成四模型各28更新、44回合并独立复核。两个policy学生不加bias，在训练源#2149以424/472步严格到达；self两顺序仍600步振荡。但7挑战同输入配对成功/碰撞变化为0/+1、−2/+2，两顺序均丢两条S8成功，联合门槛失败，不扩大/晋升。139源/606文件/53产物、全53新增输入曝光、Adam268及8关键回合精确重放已验证；详见[独立复核](runs/diagnostics/s6-policy-transfer-preflight-20261022-v1/independent-review-20261022-154339/summary.md)。

S8保持审计与短检已完整完成：[摘要](runs/diagnostics/s8-preservation-learning-preflight-20261022-v1/summary.md)。实际13个训练策略来源中9个成功，可取得27个安全保持窗口；保留原LT18目标另加固定0.25的C240保持项后，两个学生#2149仍423/478步严格恢复，保持查询漂移下降约24%/36%，但S8挑战仍失败，不晋升。`scripts/audit_s8_preservation.py`与`preflight_s8_preservation_learning.py`分别拥有零更新审计和有界短检；[LT19](docs/long_tasks/LT-20261022-19-s8-preservation-source-validation.md)完整来源核验已完成18回合：缓存C240的9个成功策略来源，原policy和保持学生均0成功/9碰撞，局部保持尚未保住训练源完整任务；[复核摘要](runs/diagnostics/s8-preservation-source-validation-20261022-v1/summary.md)。下一关口转为训练来源起点/实际访问上的安全恢复监督，不直接扫权重。

S8起点/实际访问[教师短检](runs/diagnostics/s8-entry-recovery-preflight-20261022-v1/summary.md)已完成：固定四训练策略来源/六查询，仅#2351顺序20/step20取得224步严格恢复与float32导出、1个真实新增状态，起点覆盖仍缺。原节点不变的固定双倍时长反事实0/6严格通过，#2338反而碰撞；不扫倍率、不扩训。入口`scripts/preflight_s8_entry_recovery.py`（180秒）、`preflight_s8_entry_timing.py`（90秒）均有零运行precheck；20项测试和5关键回合精确重放通过，默认Trainer/网络/MPC不变。

补充[固定教师净空对照](runs/diagnostics/s8-teacher-clearance-preflight-20261022-v1/summary.md)：`scripts/preflight_s8_teacher_clearance.py --precheck`零规划/回合/训练；180秒上限，只在独立教师副本将余量0.2→0.3m，原MPC与0.2m执行审计不变。六查询完整，2338两起点不可行，2351十二次参考/执行与旧逐元素相同；1严格标签为旧重复、0新增。12项定向测试及4关键重放通过，不据此开长训。运行输出必须使用不存在的新目录。

[航向响应与完整分支诊断](runs/diagnostics/s8-yaw-response-diagnostic-20261022-v1/summary.md)固定五事件/15首步响应，唯一局部安全脉冲使2351顺序21原step40分支从205步/2侵入变为205步严格到达；独立重放、首窗及float32导出均通过。入口`scripts/diagnose_s8_yaw_response.py`（90秒）、`scripts/validate_s8_yaw_pulse.py`（60秒）均支持零运行`--precheck`，正式`--output`须指定新目录；15项测试通过。新增一处非零状态恢复证据，两个任务起点仍缺，不建银行/训练或推广。

[LT20起点参考搜索](docs/long_tasks/LT-20261023-20-s8-start-reference-search.md)已完整执行（71.760秒、170候选尝试/130真实回合、退出码0），两源均未通过资格，`qualification_passed=false`。固定四回合[冒烟](runs/diagnostics/s8-start-reference-search-smoke-20261023-v4/summary.md)曾取得2351从任务起点244步/零侵入参考、首窗clean；正式搜索中2338的168次有界尝试（128真实回合）全部是"到达但仍有0.2m余量侵入"，最佳243步/24侵入步，未找到严格见证；2351固定见证在control0/control1与float32导出严格，但control2出现2侵入步（同流原基线0侵入），故不判资格。入口`scripts/search_s8_start_references.py`只使用冻结v4配置，`--prepare/--precheck`零回合、`--smoke`固定短检、`--run`为正式搜索；不接训练或自动部署，主成绩/门槛不变。

LT20只读复核入口为`scripts/audit_s8_start_reference_search.py --source <正式结果目录> --output <独立新目录>`：核验冻结身份、全部参考/执行余量/排名及DE代际，重放三个关键回合，180秒上限；具体命令和结果见[独立审计摘要](runs/diagnostics/s8-start-reference-search-review-20261023-v2/summary.md)。不续跑搜索或训练。

[起步消费与控制诊断](runs/diagnostics/s8-start-tracking-diagnostic-20261023-v1/summary.md)精确复现六前缀144步；三个事件的参考窗口安全但原输出首步危险。唯一安全首步脉冲从完整起点执行，将2351 control2侵入由11/12移至218/219，仍不严格，已停止。入口`diagnose_s8_start_tracking.py`/`validate_s8_start_pulse.py`分别90/60秒，均位于`scripts/`，支持零执行`--precheck`及独立新`--output`；没有新教师资格或学生收益，不改生产MPC。

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

[LT31 control原生短检](runs/verification/native-continuation-precheck-v2-20261027/review.md)固定4例：LT30父模型2成功/2碰撞，control 1成功/3碰撞；S6#247从202步到达退化为239步碰撞，静态改善未兑现为已验证闭环收益。八回合五层契约独立重放零错误，但S4存在危险消费参考，S6存在实体安全窗口下选中控制导致碰撞。只换已有GT车体约束MPC可使#247恢复、四例零碰撞，仍两例振荡并侵入0.2m余量772步，不能作生产收益。已修复分层BEV重建遗漏camera_projection的工具缺陷；默认生产模型和MPC不改，该四例短检未提供正式300条与旧205保持验收。

新增可显式启用的完整闭环模式`strict_footprint`（`scripts/run_validation_experiment.py --experiment E5 --safety-mode strict_footprint`，仍须传同版本数据与模型）：完整参考/已有专家回退、MPC选中H10及纯终端朝向接管共享VehicleConfig规划余量，当前0.2m。新执行器/引擎在无安全动作时明确safety_stop，不把停车算到达；none默认与原生对照保留。它依赖仿真完整几何，不能算纯网络或感知安全收益。[最终九例复核](runs/verification/strict-footprint-review-20261028/review.md)：同模型原生2到达/5碰撞，新模式8到达/0碰撞/0余量；原六个成功任务抽查全保持，S8#292仍振荡。2325实际步/237发出参考/完整H10/16终端证书独立审计通过，83相关测试通过；短检本身不提供完整300、原205保持或五模块绝对质量资格。[LT32正式配对](docs/long_tasks/LT-20261028-32-strict-footprint-runtime-validation.md)已完成并[独立复核](runs/verification/lt32-independent-review-20261028/review.md)：原生93/300到达、157碰撞；严格模式287/300到达（95.67%）、0碰撞/0余量，但历史205成功丢7条、同模型原生成功丢3条，联合候选门槛失败。600回合/1204产物及严格86431实际步/8760参考/724终端证书独立一致；收益依赖GT/专家回退，不晋升默认或五模块资格。后续优先修复参考替换与跨周期进度回归，近端支撑/S8推进分别处理。

参考准入现有显式`safe_prefix`模式（E5使用`--safety-mode safe_prefix`，同样必须传原模型/数据），保留从当前状态连续可行、时长足够的原样网络前缀，并在近端检查原位置/航向联合目标支撑；控制器、0.2m余量、原车位容差、权重/停止阈值不变。[固定24例复核](runs/verification/safe-prefix-review-20261028/review.md)由LT32的11成功提升至19，恢复全部7条历史回退，抽查11个LT32成功全保持；8241实际步/833参考/全部H10/26终端证书零安全违规，94项相关测试通过。仍有183/186/207/292/296失败，不能外推全300或升级。[LT33全量保持](docs/long_tasks/LT-20261028-33-safe-prefix-full-preservation.md)已完成并[独立复核](runs/verification/lt33-independent-review-20261028/review.md)：273/300到达、0碰撞/0安全违规，相对LT32恢复8却丢22，净少14；native93全保持，历史205仅保持194，联合候选失败。89187实际步/9031参考/全部H10/629证书独立一致；22个新回退首次参考分歧均是同状态在远端采用安全前缀，证明几何及时长不足以保证推进与恢复，不升级。默认none与strict_footprint原定义保留；此策略仍为GT/专家辅助，生产/五模块资格未授予。

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
