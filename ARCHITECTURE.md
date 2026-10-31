# 架构

## 组件与所有权

| 模块 | 目录 | 职责 |
|---|---|---|
| 实际输出终点与可选保持训练 | `metrics/emitted_terminal.py`、`prediction_analysis.py`、`scripts/analyze_predictions.py`；`model/terminal_hold.py`与`training/trainer.py` | 指标层按原生截断检查真实输出末点/联合目标点并保留真实容差分母，CLI导出全部逐样本记录；模型包拥有专家末位姿补齐与样本等权后缀损失，Trainer仅在terminal_hold_weight>0时应用一致teacher输入，默认0保持旧行为，跨策略优化器恢复拒绝。静态诊断不授予闭环资格；生产参考扫掠/控制器保持原所有权 |
| 相机target版本迁移 | `verification/surround_rebuild.py`、`scripts/repair_camera_targets.py` | 在新目录仅替换target并独立核验其他数组/通道；显式原任务归档与manifest拥有恢复身份权威，source_dataset_index/source_task_id关联原任务机动，恢复difficulty.maneuver保持监督语义。解析器仅在内存中解释身份，原公共复原器继续严格校验；来源/代码/产物及谱系登记SHA，不删帧、不去重、不放宽坐标检查 |
| 生产分层验证v1（冻结） | `experiments/component_verification.py`、`component_capture.py`、`component_analysis.py`与`scripts/verify_closed_loop_components.py` | 分别拥有统一结果/准入、原生E5全程只读观察、生产组件独立重放、身份/预算编排。三个旧evaluate入口保持v1复现；历史诊断显式legacy。旧137源不因修正版迁移而改写；4个初态类型误报通过v2纠正，正式300/609旧产物只读 |
| 验证修正版及约束执行候选 | `verification/`、`review_component_verification.py`、`audit_parking_input_contract.py`、`compare_footprint_execution.py`、`audit_footprint_execution.py`与`inspect_footprint_execution.py` | typed_replay按真实命令重建状态与标量类型；report_update正交展示契约/当步风险/历史风险/非到达；input_semantics核训练谱系并生成不改权重/输入的声明副本。safety_candidate在原CEM最终均值危险时，在原样本/热启动/零动作中按精确车体扫掠可行性与原成本选序列；当前约束几何直接来自仿真环境，BEV占用自由空间接入未实施。candidate_execution/study拥有显式工厂注入、四臂短检、独立审计资格与同300条件实验。新所有者独立于旧冻结源；不接默认训练/部署，不把停车、参考过滤、小样本或GT几何辅助改善算作原生学习收益 |
| 终端朝向接管候选 | `verification/terminal_capture.py`、`terminal_study.py`、`terminal_audit.py`、`terminal_diagnosis.py`、`scripts/compare_terminal_capture.py` | 同一引擎绑定目标/终端容差；只在位置严格达标而航向未达标且全角程扫掠安全时输出原地转向，否则精确保留LT22求解与随机流。旧录制器仅上下文工厂注入，不改冻结源；独立重建短检实际动作/终端证书，最后参考支撑只作归档分诊。正式两套基线配对保持原205与LT22的243个成功，余量单列，仍为GT几何辅助实验且不自动部署 |
| 带到达证书的参考保持 | `verification/reference_certificate.py`、`reference_retention.py`、`retention_study.py`、`retention_audit.py`、`scripts/compare_reference_retention.py` | 私有复制LT23控制器/轨迹/RNG并移除实例观察闭包，仅当原网络参考在80步及原剩余预算内以0.2m余量达到原车位联合标准时保持同一参考对象；正常感知/推理继续，每个真实前态/动作/后态须精确等于证书。原始与使用参考分别归档，干净控制器独立重建证书。GT辅助、静态确定性仿真假设不变，默认运行与此前148个冻结执行源均不改 |
| 零动作候选覆盖诊断 | `verification/zero_recovery.py`、`zero_study.py`、`zero_audit.py`、`scripts/compare_zero_recovery.py` | 独立上下文复用LT24，原约束选全零才补查同次原400样本中前16之外、更低原成本且完整时域物理安全的首个候选；不消费额外RNG。保存求解前上下文，独立旧控制器重建原零解/400样本/排名与实际序列。固定短检余量回退、正式门禁拒绝，不接默认路径；物理安全未等同规划余量或全程收敛 |
| 全输出规划余量及分支诊断 | `verification/margin_recovery.py`、`margin_study.py`、`margin_audit.py`及`compare_margin_recovery.py`；`planning_margin.py`、`planning_study.py`、`planning_audit.py`及`compare_planning_margin.py` | 首阶段仅给新增补查加0.2m，普通输出仍侵入，保留为负证据；统一候选PlanningMarginMPC以原0.2m矩形检查器复用LT24控制及原样本补查，将均值/样本/热启动/零解/终端完整转向全部约束。引擎实体0余量与原车位终端独立保留；干净旧控制器重建补查与所有实际H10。初态已侵入则中断、不声称恢复；独立作用域，旧157源/默认入口冻结，LT25正式同300已完成（余量0但丢6条旧成功，条件资格false） |
| 合法参考窗口的可行控制候选 | `verification/reference_candidate.py`、`reference_teacher.py`、`reference_study.py`、`reference_audit.py`；`feasible_reference_candidate.py`、`feasible_reference_study.py`、`feasible_reference_audit.py`及对应compare脚本 | v1仅原全零时从真实消费窗口按原模型递推一条候选，完整H10余量安全且低于原成本才选；教师6/6但NN回退，冻结为负结果。v2先检查当前状态连接整个消费窗口的0.2m扫掠，非法则保留LT25，合法才补候选；独立重建原400样本/窗口前置条件/候选/实际转移。教师验收、真实NN研究、审计分开，工厂仅显式作用域注入；LT26已完成：教师6/6、真实NN无新增且未恢复LT24子集，v1/v2均判负结果，不改默认入口，无训练/数据迁移 |
| 候选五单元完整验收 | `verification/candidate_layer_capture.py`、`candidate_layer_analysis.py`、`candidate_layer_quality.py`、`candidate_layer_study.py`、`perception_calibration.py`及两个新CLI | 只读观察当前候选真实BEV/局部预测/模型输入和语义，连续独立重放感知/网络/MPC及接口；完整参考安全/近终端支撑/实际H10/任务到达分层判定，缺证据blocked。训练源唯一任务拟合与留出形成经验参考，缺组/不足/漂移阻断，不提供绝对感知质量资格。新schema独立于旧174源；181源冻结；LT27已完成校准与固定14补证（采集与重放精确、L3全passed，L2/L1_L2因完整参考危险全failed，L1 1 failed+4 blocked），`all_five_modules_qualified=false`，默认入口/策略不变 |
| 严格车体约束完整闭环 | `controller/constrained_mpc.py`、`terminal_capture_mpc.py`；`runtime/strict_footprint.py`、`execution.py`、`engine.py`；`validation_matrix.py` | controller复用原CEM采样，选中H10先满足注入车体检查器，原位置容差内的朝向接管另需完整转角证书；runtime严格审查最终发出参考/实际转移并把专用控制安全异常转为safety_stop，普通异常传播。矩阵strict_footprint注入同一VehicleConfig余量检查器和同一动力学/车位终止器，显式GT几何/专家回退，none及旧模式保留。`strict_footprint_study.py`拥有冻结/配对编排，`strict_footprint_audit.py`从落盘独立审计；LT32完整配对独立复核287到达/0碰撞/0余量，历史成功丢7/同模型成功丢3而未获联合候选资格；原生/五模块资格仍不授予 |
| 安全前缀参考准入 | `runtime/safe_prefix.py`；`validation_matrix.py`；`verification/safe_prefix_preflight.py`、`safe_prefix_handoff.py` | 显式safe_prefix复用严格检查器/控制器/回退；source拥有当前桥接开始的连续安全原样前缀和近端原联合目标支撑准入，矩阵注入同一终止器及实际H10时长；不改网络点值或控制算法。preflight保存原始/发出参考并独立审计准入，handoff冻结代码/输入与全量保持门槛；LT33完整300为273到达/0安全违规，但丢LT32成功22、历史成功11，候选失败；几何及时长准入不证明跨周期推进/恢复，GT混合身份与生产/五模块限制不变 |
| 统一接口 | `interfaces/` | 定义三阶段共用的数据契约：传感器帧、BEV 空间配置/张量、车辆状态、目标位姿、轨迹、控制指令；坐标统一为车辆中心局部坐标系 |
| Sensor2BEV | `sensor2bev/` | 将 LiDAR 点云/Camera 图像按共享 `BEVConfig` 转换为统一 BEV 表示：`lidar_bev.py`（ROI→降采样→地面滤除→栅格投影）、`camera_bev.py`（各CameraView按yaw/height/pitch与内参投到统一车体网格再取最大值）、`fusion.py`（通道级后融合）。image携带per_view_v2，几何目标/明确GT源携带geometry_v1；模型推理及训练验证拒绝image投影版本不匹配；旧无版本输入保持legacy_overlay_v1，不提供错误环视的在线兼容实现 |
| Python 仿真与任务 | `sim/` | 二维矿区泊车环境、携带场景级 `BEVConfig` 的 S1–S9 场景与 T1–T5 任务层；`noise.py` 提供传感器噪声；`vehicle_config.py` 严格加载 `configs/vehicles/`，是履带钻机外廓、执行上限和规划搜索参数的统一来源 |
| 端到端网络 | `model/` | 注册表按 `net-v0/v1/v2` 构造模型：v0 为固定 horizon CNN+MLP，v1 为 CNN+GRU 变长解码和终止 logits，v2 增加 U-Net 跳连空间编码与 goal/state 交叉注意力；损失支持掩码轨迹 MSE 与终止 BCE |
| 训练体系 | `training/` | 安全 YAML 配置解析并相对配置文件定位数据/输出；支持只初始化模型权重或严格断点恢复；`Trainer` 拥有确定性 shuffle、可选恢复三池采样、累计停止监督、连续净空完整车体扫掠损失、课程感知 early stopping 和逐 epoch 自由滚动/安全诊断；runner 从 schema v2 解析车辆/BEV 安全几何，并在 best 上用 val 校准停止阈值 |
| 轨迹控制器 | `controller/` | MPC 轨迹跟踪：CEM 交叉熵求解 + 差分驱动模型预测，输出 `[v_cmd, omega_cmd]` |
| 终点对齐离线审计 | `experiments/terminal_alignment_audit.py` | 基于既有自由滚动采集，按冻结部署阈值区分停点/真值末点索引/全预测联合达标点；scripts/audit_terminal_alignment.py只推理原train并冻结身份，不校准阈值、不判断几何/控制可执行或训练成功 |
| 原生发出轨迹联合目标 | `experiments/emitted_output_joint.py`、`emitted_output_data.py`、`emitted_output_study.py`与`scripts/train_emitted_output_joint.py` | 独立实验分别拥有可微首停位姿目标、训练八源/沿程教师门禁、同权重双seed配对有限更新及硬首停静态验收；默认关闭，不修改Trainer/模型/MPC。拒绝未过教师门禁的拟合及覆盖/续跑；LT34教师严格6/8，因两源执行余量侵入暂停拟合，原MPC契约拟合未执行。静态通过才另准备固定/滚动计划，125唯一任务仅冻结为后续评估身份，无生产资格 |
| 原生参考三分支执行诊断 | `scripts/emitted_execution_capture.py`、`diagnose_emitted_execution.py` | 前者只读观察同一次公开predict、缓存固定全局参考及原MPC窗口/H10/转移，复用既有独立风险导出；后者冻结同八任务、复用LT34教师、配对父网络固定/滚动并分别统计到达/碰撞/余量。新增scripts不改变旧LT34源码清单/训练门禁；LT35正式完成：教师8成功0碰撞→网络固定7/1→滚动4/4，全八配对前缀/122预测/2185控制步独立一致。定位3重规划相关回退和1固定参考已失败，不训练、不自动扩样或作唯一因果/生产资格判断 |
| 搜索中可行CEM实验 | `scripts/feasible_cem_mpc.py`、`feasible_cem_capture.py`、`compare_feasible_cem.py` | 搜索子类复用既有完整扫掠检查与安全异常，每代仅以可行候选拟合精英并复核最终均值/备选；capture拥有实际转移/H10/原生输出独立审计，CLI拥有149源冻结及同八任务固定/滚动/教师对照。GT几何辅助、0.2m不放宽，无专家替换/终端接管，不注册生产工厂；旧146源不改。LT36正式滚动6/8且旧4保持，固定丢860换1290、教师7/8、48.8%调用超100ms，整体拒绝扩大；5834实际步/全部H10、240预测及19保存状态CEM已核验，不授予实时/训练资格 |
| 执行局限独立修正 | `scripts/batched_footprint_check.py`、`runtime_refinement_mpc.py`、`runtime_refinement_capture.py`、`runtime_refinement_preflight.py`、`compare_runtime_refinement.py` | 批量内核读取当前几何并在数值边界/未知模型回退原判据；私有控制器拥有分布/选择/进度/终端开关，所有实际输出仍完整标量验证；capture审计实际转移、RNG/统计/代价和终端证书，preflight拥有19状态等价见证，CLI冻结154源及30回合对照。无工厂注册/新依赖，旧149源保持；LT37正式滚动7/8、教师8/8、观测控制时延通过，但固定降5/8丢1290/2315，组合拒绝扩大；全部7466实际步/H10、299预测及两条fast完整等价核验，不解锁LT34或部署 |
| 撤销硬进度门控对照 | `scripts/ungated_refinement_mpc.py`、`ungated_refinement_capture.py`、`compare_ungated_refinement.py` | 私有子类仅关闭progress并继承冻结求解器；capture逐状态重建原MPC时间进度、以冻结父求解器重放全部实际调用，另复用标量H10/原生预测审计；CLI冻结157源/24回合/66文件及LT36/37双基线保持与两级退出。旧154源/默认工厂不改，无新依赖；LT38正式固定7/8、滚动7/8，但教师6/8丢860/2315、1次控制超100ms，候选拒绝。5855实际步/H10/父求解器重放与256预测核验，不赋予训练或生产资格 |
| 全零运动候选补全 | `scripts/zero_hold_motion_rescue.py`、`zero_hold_motion_capture.py`、`compare_zero_hold_motion.py` | 子类保留冻结LT38求解器，仅全零H10后检查固定8条常速候选，完整安全且严格降原cost才替换；capture拥有实际热启动下父重放、补全候选标量/代价/计数与原输出审计，CLI冻结160源/24回合/66文件及三基线成功并集。原157源/工厂不改，无依赖；LT39八任务三分支各8/8、三基线旧成功全保留、零风险，4135次实际控制/父重放、203预测核验，最大74.639ms。几何仍来自仿真真值，仅允许准备下一验证，无训练或生产资格 |
| 全零补全的条件回归验证 | `scripts/zero_motion_retention_capture.py`、`validate_zero_motion_retention.py` | capture仅构造冻结LT38/LT39两臂并采集/审计原生正常重预测、完整实际H10、父重放和全网络前向；CLI拥有原验证索引/task_id/重叠组绑定、162源冻结、125任务/250回合/752文件、历史120与配对成功保持及风险/时延裁决。原160源和默认工厂不改，无依赖；LT40正式102→104/125、父成功全保留且零风险，但历史丢18及1次117.407ms，候选拒绝；完整inspect核869130个H10转移/8783前向，训练/生产仍禁用 |
| 固定参考/滚动收敛诊断 | `scripts/fixed_reference_analysis.py`、`diagnose_fixed_reference.py` | analysis拥有同起点原生输出/首10步配对、实际收敛和同状态旧续用/新初始化窗口观察；CLI复用冻结录制/审计、拥有164源与LT40完整身份、21新固定/21旧滚动分母、86新文件和分流。窗口度量复用reference_update_stability；原162源、控制器和默认工厂不改。LT41正式10固定恢复/11固定仍失败，完整inspect核新88320/旧126000个H10转移和1281前向；零风险但1次128.531ms，不提供固定参考策略或训练/生产资格 |
| 再认证监督的原生发出拟合 | `scripts/emitted_teacher_recertification.py`、`recertified_emitted_fit.py`、`train_recertified_emitted_joint.py` | 三者分别拥有LT39实验教师连续回合32训练输入/原剩余标签认证、既有联合目标配对拟合及公开predict硬验收、167源身份和阶段编排。LT42四臂800更新/16检查点/512输入审计完成，joint双seed均5/8起点、16/32全部输入，未过关口2；停止此具体配方/预算，转目标/梯度/表示及路径几何诊断。原164源/原MPC失败/旧fit阻断/LT40拒绝均保留；验证数据不拟合，无默认路径或生产资格变更 |
| 自由输出停止与几何训练 | `scripts/native_output_geometry.py`、`native_output_objectives.py`、`native_output_fit.py`、`train_native_output_contract.py` | 分别拥有GT形状的可微穿透/原硬前缀、自由坐标与首次阈值窗口、A/B/AB分解及检查点审计、171源/旧LT42全证据冻结和阶段编排。A无教师混入，空窗口保留坐标训练并延后停止项；B覆盖真实发出桥接/全段，硬检查仍归原标量几何。LT43正式1200更新/24检查点/768输入审计完成；唯一候选AB双seed起点5/8退至1/8，拒绝且停止配方。B局部降风险但无新增起点成功、有沿程回退，不替代AB。旧167源/默认路径/拒绝结论不改，无闭环或生产资格 |
| 原生验收一致的停止监督 | `scripts/accepted_native_objectives.py`、`accepted_native_fit.py`、`accepted_native_preflight.py`、`train_accepted_native_stopping.py` | 分别拥有原硬完整合格集合/合格零停止压力和集合正例、gate/accept配对拟合与16检查点审计、旧64输出及非停止分量短检、175源及LT43完整证据绑定。LT44正式800更新/16检查点/512输入审计完成；合格输出零压力生效，但唯一候选accept双seed仅2/8、1/8起点，拒绝并停止配方。原171源/自由坐标/几何/默认路径保持；空集合显式deferred，坐标/几何训练保留但学习覆盖未过，无闭环或生产资格 |
| 冻结停止头训练试验 | `experiments/frozen_stop_head.py` | 只读hook缓存stop_head输入，独立复制129参数head按原累计BCE训练；主干/point_head不更新。scripts/train_frozen_stop_head.py冻结2918旧池及guided/free同预算，保留独立非resumable模型与缓存/摘要，完整轨迹/阈值不变，不修改现有Trainer或部署 |
| 停止目标独立审计 | `experiments/stop_objective_audit.py`、`experiments/stop_cross_effects.py`、`experiments/stop_direction_feasibility.py`、`experiments/stop_fullspace_feasibility.py`、`experiments/stop_finite_step.py`、`experiments/stop_endpoint_intervals.py` | 固定梯度/跨组/有限族/全空间审计，内存head副本单次有限步长验证，以及只读仿射首次停点区间；对应audit_stop_*.py与probe_stop_finite_step.py。复用LT12/LT13身份，float64解析、float32缓存统计与原生predict抽查分开；延长任务人工执行，不更新默认模型或自动训练 |
| 闭环收益裁决 | `experiments/closed_loop_benefit.py`、`scripts/validate_closed_loop_benefit.py` | 前者校验完整纯E5逐回合/协议并统计联合收益、配对转移与场景回归；后者冻结固定R0/R2e2及独立300条，复用validation_matrix，不修改模型/MPC/感知。每臂报告SHA登记后才可显式恢复；短smoke与正式证据隔离，不自动训练/部署 |
| 行为保持短预检 | `experiments/behavior_preservation.py`、`experiments/executable_behavior.py`、`experiments/reference_compensation.py`、`scripts/preflight_behavior_repair.py` | 分别拥有角色均衡损失/模型副本短拟合/门禁、实际MPC消费记录与固定/持续教师离线分支、有限航向误差补偿、固定训练域5+5来源隔离及180秒编排。教师组合仅显式选项启用，原MPC super计算不改，严格教师通过后才拟合，纯网络探针收益通过后才可准备长计划；不接入默认训练/部署 |
| 连续学习探索 | `experiments/behavior_exploration.py`、`experiments/behavior_exploration_data.py`、`experiments/exploration_store.py`、`scripts/explore_closed_loop_learning.py` | 分别拥有源/角色均衡的学习对照与裁决、拟合源离线可执行监督/学习器状态聚合、90分钟活动预算/互斥/哈希单元恢复、配置冻结与多臂编排。发现集排序和60条训练域确认隔离，不接默认Trainer/MPC，不自动晋升；正式多轮探索由用户显式运行 |
| 有效恢复与分层回放 | `experiments/effective_recovery_data.py`、`scripts/explore_effective_replay.py` | 新版本拥有完整输入/标签身份及舍入别名拒绝、先固定确认/恢复/回放来源、实际非零访问起点的严格教师首帧与数据覆盖门禁；入口复用既有均衡拟合/原MPC离线分支和哈希单元存储，完整2×2对照、两时点发现排序与旧/新双60确认。旧LT15按存档源逐项只读核验，新增文件不重写旧identity；数据不过线不训练，正式长run由用户执行，不接生产或自动部署 |
| 保旧采样匹配诊断 | `experiments/retention_sampling.py`、`scripts/preflight_retention_sampling.py` | 分别拥有预固定训练来源/基线候选、保持角色层级时间计划、同Adam有限续训与闭环门槛，以及旧存档逐项核验/成功源补帧/数据与基线可测性门禁/配对编排。所有臂固定原池恢复角色序列，避免新增保持池暗改恢复RNG；默认180秒，900秒完整诊断只支持显式prepare/precheck/run-prepared，互斥后才写运行状态，不自动恢复/扩大更新量或部署 |
| 保持新增帧消费短检 | `experiments/retention_consumption.py`、`scripts/preflight_retention_consumption.py` | 前者在原48保持槽/每场景8槽内保障四关键源各3个唯一新增时间帧，保持恢复序列并叠加同池对照裁决；后者只读核验并复用LT17缓存/基线/控制，逐批核对真实消费，同C240/Adam各续24步，单次180秒。新入口不修改旧114源、生产训练或MPC，无自动恢复/扫配额/部署 |
| 保持窗口质量对照 | `experiments/retention_quality.py`、`scripts/preflight_retention_quality.py` | 前者分项审计冻结R0实际执行窗口/原始消费参考链，并在同源两槽构造干净替换与风险换样及联合裁决；后者验证前一消费身份、冻结118源，编排双臂双顺序各24更新与同探针评测，180秒。旧116源不改，窗口选择不读候选结果，缓存与新运行明确分开，无自动恢复/重选或部署 |
| 保持教师及来源分工 | `experiments/retention_teacher.py`、`retention_teacher_roles.py`与对应`preflight_retention_teacher*.py`入口 | 共同实际访问输入/mask上的R0与当前成功教师软目标配对，以及C240成功来源保持/R0成功但C240失败来源恢复的可审计分工。严格180秒版在固定8来源门禁失败时停训练；独立120秒分工版只读复用该结果，不追认门禁、重跑来源或改旧118源。各臂同C240/Adam240及24更新，不改默认Trainer/MPC，不自动扩源/加时/部署 |
| 实际失败状态教师可执行性 | `experiments/learner_failure_teacher.py`、`scripts/preflight_learner_failure_teacher.py`、`scripts/preflight_failure_teacher_refinement.py` | 首次危险前/至少200步余时的非零安全上下文、原网络后缀复现、滚动教师首帧真实输入与参考/执行窗口分审计；独立S3校正复用固定参考及最多两次既有航向补偿。分别180/120秒、只读冻结前置身份、不改旧122源/旧教师，不训练、不放宽原MPC/600总步数/0.2余量，不自动续跑或部署 |
| S6恢复目标配对学习 | `experiments/s6_recovery_learning.py`、`scripts/preflight_s6_recovery_learning.py` | 复用三个真实失败查询，在原24批次的固定三恢复槽构造同输入/mask的C240自目标与严格教师目标；其余48保持/45恢复不变，实际梯度前核对输入和标签SHA。每顺序继承C240/Adam240到264，固定7挑战＋4源监测，180秒；旧125源/默认Trainer/MPC不改，无自动恢复/动态配额/部署。只读迁移与完整消费证据归档运行目录，不接入生产路径 |
| S6已执行终段监督对照 | `experiments/s6_terminal_consumption.py`、`scripts/preflight_s6_terminal_consumption.py` | 精确重建前轮配置/实际排程，只扩固定step400单槽mask至原19点全段实际消费，其余95槽/points/stop不变；复用旧strict首窗控制，只新增双顺序各24更新/同7＋4评测，120秒。冻结原127源、单槽/原资格/增量门槛均审计，无默认模型/控制器修改；单查询隔离与真值长度oracle仅为运行目录中的非晋升诊断 |
| S6停止边界与终点目标对照 | `experiments/s6_terminal_objective.py`、`scripts/preflight_s6_terminal_objective.py` | 原完整19点排程/标签不变，仅唯一真实终段槽复用terminal类别均衡stop，再配对既有terminal_pose_loss，保留原角色归一/他样本梯度；冻结129旧源，新入口131源、双臂双顺序各24更新/同7＋4评测、180秒，原资格和相对mask控制增量合取。只读独立审计拥有重放/同状态定位，不修改默认Trainer/predict/原MPC，无自动扩训或部署 |
| S6优化传输与目标学习性 | `experiments/s6_update_transport.py`、`scripts/audit_s6_update_transport.py`、`scripts/preflight_s6_event_learnability.py` | 分别拥有梯度/有限Adam副本与递归状态相等核验、六历史模型/优化状态精确重放及180秒更新统计、同120步目标隔离/150秒原MPC局部与7＋4副作用观察。冻结131旧源，新增后133/134源；运行目录另存长度oracle和单例bias机制检查，显式非候选/等价全局截断变更，不接默认训练/部署，不自动扩训 |
| S6连续重规划恢复链 | `experiments/s6_replanning_chain.py`、`scripts/preflight_s6_replanning_chain.py`、`scripts/preflight_s6_chain_learning.py` | 完整消费/通用teacher目标、逐控制步参考索引及窗口审计；原MPC参考内容/对象进度分叉、410标签原始与float32导出严格执行/新增门禁；同120步单帧缓存/相邻状态机制对照及非候选bias隔离。原134源不改，新数据/学习136/137源，预算90/120秒。运行目录拥有关键恢复重放、固定交叉和单训练源成功策略银行，不回灌probe、不接默认Trainer/部署 |
| S6成功策略迁移 | `experiments/s6_policy_transfer.py`、`scripts/preflight_s6_policy_transfer.py` | 配对计划/实际批次核验/Adam240→268与独立文件编排分工；保留旧监督加53个成功策略输入，仅改变新增目标，学生原生停止；precheck零更新，冻结139源/606输入文件。正式四模型44回合按LT18人工执行，独立新目录/失败关闭，不接默认Trainer/部署 |
| S8保持审计与约束 | `experiments/s8_preservation_audit.py`、`s8_preservation_learning.py`，对应audit/preflight CLI与`validate_s8_preservation_sources.py` | 同输入指标/局部梯度/实际窗口筛选与独立文件编排分工；从LT18实际S8源核验C240成功对象，固定0.25项不改变原六样本loss，eval辅助前向保留梯度且不推进训练随机流；来源完整验证冻结144代码/704文件，LT19由用户执行，零训练、不接默认Trainer/部署 |
| S8起点/实际访问教师 | `experiments/s8_entry_recovery.py`、`s8_reference_timing.py`、`s8_teacher_clearance.py`及对应preflight CLI | 起点/危险前访问选择、frame0窗口、固定双倍时长与独立教师0.3m规划副本分工；净空对照只在副本变更规划外廓/元数据，执行仍用原case/0.2m审计。复用旧规划/有限补偿/原MPC，只把全程严格且导出严格的真实首窗作为证据，新增/冲突单独审计。146/148代码冻结，不改旧144源；局部可恢复不授权学习或部署 |
| S8航向响应诊断 | `experiments/s8_yaw_response.py`、`scripts/diagnose_s8_yaw_response.py`、`validate_s8_yaw_pulse.py` | 五固定首次侵入事件共同深复制MPC/参考/状态，保留对象关系/缓存/进度/热启动/RNG；原值及固定±0.01rad窗口只算首步响应，参考内部/当前窗口/实际扫掠分开。唯一局部见证由独立完整分支入口验证；148/149代码冻结、不改旧146源，不接训练或部署，不输出训练银行 |
| S8起点离线参考搜索 | `experiments/s8_reference_search.py`、`s8_clearance_surrogate.py`、`scripts/search_s8_start_references.py` | 参数/NumPy有界DE、矩形SAT代理次级排名和原MPC/文件编排分工；原检查器唯一安全判据。152源/821输入冻结，v3新代码用摘要匹配快照保留；2338最多168尝试，2351只复核冻结见证，首个严格后独立核验、不再优化。prepare/precheck零回合，smoke固定短检，formal按LT20用户执行；不接Trainer/模型/控制器或部署，无自动银行 |
| S8起点结果独立审计 | `scripts/audit_s8_start_reference_search.py` | 只读冻结LT20结果，重算身份/参考/原检查器余量/排名，使用归档排名重建DE代际，不调用新搜索目标；重放三个关键执行并核对首窗。180秒上限、独立新输出，不改变正式文件/优化预算/教师资格或训练授权 |
| S8起步消费与控制响应 | `experiments/s8_start_diagnostic.py`、`scripts/diagnose_s8_start_tracking.py`、`validate_s8_start_pulse.py` | 纯预测/原扫掠风险统计与固定前缀/单一完整候选编排分离。原MPC采样/成本/精英均值不改，快照保存同流状态；90秒六前缀三事件、60秒唯一见证验证，失败不换候选。只读冻结LT20/诊断，新目录输出，不接生产控制/银行/Trainer |
| 原代价局部精修试验 | `experiments/cem_polishing.py`、`experiments/polishing_execution.py` | 独立CEMPolishedMPC保留原CEM后作有界L-BFGS精修，原NumPy cost严格更优才选，不读几何，额外最多80次梯度评估。compare_cem_polishing.py冻结六源原/精修×网络/持续教师，私有快照与逐控审计，默认controller及训练schema不变 |
| 离线求解器试验 | `experiments/mpc_candidate_solver.py` | 显式实验子类仅比较原CEM均值/移位热启动/全零候选，沿用原代价；专用scripts/compare_mpc_solvers.py进程内注入并finally恢复。无障碍输入/安全干预，controller默认实现不变，不可与原控制器成绩混报 |
| 当前搜索样本试验 | `experiments/mpc_search_best_solver.py` | 独立SearchBestMPC保留原CEM采样/拟合，比较本次最佳样本与最终均值；热启动仅作搜索初值。专用scripts/compare_search_best_mpc.py显式注入、输出逐步选择账本并finally恢复，与上一候选版/生产控制器隔离 |
| 参考更新诊断 | `experiments/reference_update_stability.py` | 在原MPC私有副本对齐旧/新窗口，实际执行不受观察影响；复制既有上下文，正常分支精确复现后缀，再单次丢弃新参考但仍消费感知/推理，之后恢复更新。专用scripts/validate_reference_updates.py冻结原训练源/身份并记录覆盖；不修改默认运行时、不作训练标签或生产保持策略 |
| 参考事件探针 | `experiments/reference_event_probe.py` | 既有安全事件精确时点分叉，正常/旧参考/专家仅首次替换，随后恢复网络。实际solve窗口审计只包装不再采集快照的私有MPC，避免闭包随上下文复制；原采样/RNG不变。scripts/probe_reference_events.py冻结来源/输入与审计，S8初态10步显式区分全程；不是生产策略或训练标签 |
| 连续恢复前置 | `experiments/recovery_compatibility.py` | 连续教师每次正常感知/网络旁路推理后执行同状态专家，原MPC私有快照续行并记录每次全局参考/状态；scripts/validate_recovery_compatibility.py冻结来源/身份，正式只运行S4。无训练schema导出或自动回灌，原成功记录保留不代表学生已保留成功 |
| 几何航向代价试验 | `experiments/footprint_yaw_mpc.py` | FootprintYawMPC仅增加按车身尺寸缩放的周期航向残差，不读取环境/碰撞、不改原CEM预算/返回或默认MPC；其S8短预检未改善，已停止扩大，仅保留证据与测试，不进入正式S4或部署 |
| 专家轨迹 | `planner/` | Hybrid A* 生成履带低速运动学可行轨迹（前后差速弧线 + 左右原地旋转 + 48 词族 Reeds–Shepp/履带解析候选）；`collision.py` 拥有完整矩形与连续扫掠碰撞，`smoothing.py`/`profile.py` 提供可选平滑以及含原地旋转耗时的速度剖面 |
| 数据管线 | `dataset/` | `calibration.py` 直接枚举全部专家能力单元；Task 驱动生成经机动与可行性双门禁保存 schema v2；`recovery.py` 从学习器闭环偏离状态生成重新审计的专家恢复样本，碰撞时完整回溯最近的规划安全余量状态；构建脚本按源任务原子续建、从失败检查点派生困难补采并去重合并既有恢复集和原训练集 |
| 闭环运行时 | `runtime/` | `engine.py` 执行轨迹源→MPC→车辆并在状态更新前调用转移门禁，再以完整矩形连续扫掠判碰撞；`sources.py` 提供 Expert/Network、当前安全状态专家重规划、专用安全停止、门禁组合与轨迹级几何过滤组合；`safety.py` 定义场景无关的轨迹审查接口与干预统计；`trajectory_repair.py` 把预测轨迹中不可行的部分按"可达锥"侧向投影回可行域（只做侧移，不改航向与训练目标），失败时由包装源退化为计划保持或安全截断 |
| 实验指标 | `metrics/` | `EpisodeResult` 与闭环聚合；开环层在目标有效前缀上统计 ADE/FDE/环绕航向 MAE，并拒绝预测 horizon 不足的比较；预测诊断层保留逐样本误差与终止长度，按场景、任务、方向、噪声和相邻占用聚合 |
| 可视化 | `viz/`、`dataset/inspection.py` | 统一风格（`style.py` 色表/PNG+PDF 双格式）、世界俯视渲染、轨迹三线叠加、单回合总图与分组开环图；专家验收图和预测叠加图使用“前方 x、车体左方 y”右手局部系，将正 Left 显示在画面左侧，并把连续零位移航向变化汇总为从旋转前航向出发的有符号旋转弧 |
| 批量实验 | `experiments/` | 配置驱动专家 runner；`closed_loop_evaluation.py` 从 schema v2 NPZ/manifest 确定性复原任务场景并加载 Trainer deployment，输出网络闭环整体、分组与逐回合 JSON；`recovery_execution.py` 仅供离线诊断，复制真实车辆/MPC热启动与RNG/感知管道，在私有MPC副本上保留续执行状态，原engine/MPC默认reset及生产执行不变 |
| 运行脚本 | `scripts/` | 阶段演示与数据流串联 |

## 数据流

```
ParkingEnvironment（真值）→ SimulatedLiDAR/Camera + NoiseProfile → LiDARFrame/CameraFrame
SceneBundle.bev_config → LiDAR2BEV/Camera2BEV → SensorBEVPipeline 配置一致性门
LiDARFrame/CameraFrame → sensor2bev(BEVConfig) → BEVTensor
  LiDARFrame → LiDAR2BEV → [occupancy, height, density]
  CameraFrame → Camera2BEV → [target]
  BEVFusion 拼接两路并追加 [vehicle] → 统一 BEVTensor
VehicleConfig 尺寸/margin → TaskSampler → 车辆相对 S3/S4/S7/S9 SceneBundle
SceneBundle + (TaskType, difficulty axes, seed, sample index) → TaskSampler → Task
  Task = scene + start + single/candidate goals + stable metadata + optional T5 event
  结构化车位按请求机动确定目标航向与入口侧，并只从连续 footprint 自由的轴线走廊采样起点
VehicleConfig JSON → HybridAStarPlanner/MPC/车辆模型/碰撞/inspection
VehicleState + GoalPose → HybridAStarPlanner（前后弧线 + 原地旋转 + RS/履带解析候选）→ Trajectory（中心位姿专家轨迹）
  Trajectory → 碰撞安全三次捷径（可选）→ Trajectory
  Trajectory → 梯形速度剖面（可选）→ VelocityProfile
Task[] → DatasetSplits（S9→test；其余按场景×类型分层 train/val）
专家能力矩阵 → 等额 CalibrationCase[]（不经过 split 配额）
  → 独立 worker 硬预算 → case 终态原子 JSON → 可恢复 report.json/cells.csv
Task → task 组件工厂 → HybridAStarPlanner + SensorBEVPipeline → TrainingSample
  轨迹方向距离审计：请求方向占比默认 ≥50%；不一致候选拒绝，失败在原场景×类型×难度单元重采
  轨迹可行性审计：起终位姿、线/角速度、横向残差、完整矩形扫掠、模型版本全部通过
  T4 按稳定候选顺序解析首个可规划且机动一致目标；显式关闭门禁时保留旧策略
  TrainingSample[] → NPZ schema v2（数组 + bev_meta + 逐样本车辆模型/机动/可行性证据）；无版本 NPZ 按 v1 读取
  固定批次 → 失败前原子 retry 游标 → 独立机动与运动学复算 + 碰撞证据/模型版本核对 → 原子成功检查点
  同身份检查点（seed/计划/车辆模型/重试参数/批大小）+ 未完成批次游标 → 不重放已排除 task ID 地补齐合并 → 正式 NPZ/manifest
训练 YAML → SafeLoader/严格 schema → 注册表模型 + train/val NPZ → `Trainer`
  → 可选基线 checkpoint 仅初始化模型权重 + 样本级确定性 shuffle + 累计停止 BCE
  → schema v2 车辆/BEV 几何 → occupancy 预计算截断有符号净空场 → 预测轨迹完整矩形连续扫掠净空/越界损失
  → 每 epoch train/val 自由滚动 ADE/FDE/停止诊断 + 课程预热后 early stopping + 原子 history/best/last
  → best checkpoint 的 val 停止阈值扫描 → 只读 deployment checkpoint + 校准 JSON
  → report.json + training_curve PNG/PDF
同一 val NPZ + 一个或多个 Trainer checkpoint → `eval_openloop.py`
  → ADE/FDE/航向 MAE report.json + openloop_comparison PNG/PDF
  NPZ → 长度/方向/可行性/分层统计 + occupancy/target/中心点/中间车身验收图
BEVTensor + VehicleState + GoalPose → 模型注册表（net-v0/v1/v2）→ Trajectory
  （训练：全局专家轨迹/目标 → 起始局部系 → Trainer → 掩码轨迹/终止损失 → 原子 checkpoint）
Trajectory + VehicleState → MPCController → ControlCmd[v, omega] → 平台执行器
ClosedLoopEngine：TrajectorySource(Expert/Network) → MPC → 车辆模型滚动循环
  → 可选 SafetyShieldSource（完整扫掠审查 → 当前安全状态专家回退 → 每控制转移执行前复核）
  → 可选 GeometricFilterSource（完整扫掠审查 → 不可行段按可达锥侧向投影 → 计划保持 → 安全截断）
  → 终止（到达双阈值/完整矩形连续扫掠碰撞/安全停止/超时/振荡）→ EpisodeResult（含轨迹/转移干预统计与过滤统计）
schema v2 NPZ + manifest + deployment checkpoint → 闭环评测编排
  → 复原 scene/occupancy/noise/BEV/selected goal → NetworkSource（目标通道随回合更新）→ 分组 JSON
  → 学习器闭环状态（固定步长偏离；碰撞时完整回溯最近的规划安全余量状态）→ 专家重规划与重新审计
  → 上轮碰撞/超时且零恢复检查点 → 通用困难任务补采 → recovery 去重合并 + 原 train 合并
```

- 障碍物碰撞与点云语义分离：`is_free`/`has_collision` 只检查 forbidden 障碍与地图边界；`raycast` 只与 emits_points 障碍求交（悬崖禁止进入但不挡射线，地面标线可通行）。
- Camera→BEV 与模拟相机共用同一套单应几何（`sim/camera_model.py` 与 `sensor2bev/camera_bev.py` 推导一致），保证渲染与反投影互逆。
- MPC 进度锚定轨迹时间轴（单调推进），参考轨迹 dt 与控制周期不一致时按时间插值对齐；求解器为 CEM（纯 numpy）。

## 统一接口（interfaces/）

- 传感器帧：`LiDARFrame`（N×4 `[x,y,z,intensity]` 点云）、`CameraFrame`（同步`CameraView`集合，各自图像/内参/yaw/height/pitch；旧image字段仅为预览；无views的旧单帧按单前视处理）。当前模拟相机共用车体中心上方安装点，不宣称已支持任意六自由度外参。
- 环境表示：`BEVConfig`（分辨率、前后左右范围与确定栅格形状）和 `BEVTensor`（`[C,H,W]`，通道语义：障碍物占据、高度、点云密度、目标区域、车辆轮廓）。
- 运动状态：`VehicleState`（`[x,y,yaw,v,omega]`）。
- 目标位姿：`GoalPose`（`[x_goal,y_goal,yaw_goal]`）。
- 轨迹：`Trajectory`（未来 N 个局部轨迹点 `[x_i,y_i,yaw_i]`）。
- 控制指令：`ControlCmd`（`[v_cmd, omega_cmd]`）。

## 不变量

- 网络输入输出契约在三个阶段保持不变；迁移只替换传感器来源和底盘执行接口。
- 坐标统一使用车辆中心局部坐标系。
- 专家验收图的数值局部系为前方 `x`、车体左方 `y`、正 `yaw` 为车体左转；画面水平轴反向显示数值 `y`，使数值正 Left 在画面上仍为左，不得直接按屏幕右正方向绘制。
- `interfaces/` 不依赖任何仿真、网络或硬件实现，反向依赖。
- `sim/`、`sensor2bev/`、`model/`、`controller/` 依赖 `interfaces/` 与各自必要的 `numpy`/`torch` 数值运行时，模块间不互相耦合；`training/` 依赖 `model/` 与 PyTorch，不反向进入模型或数据层。
- `runtime/` 依赖 `interfaces/`、`metrics/` 与注入的轨迹源/MPC/车辆模型（依赖注入，不直接 import sim/model）；`metrics/` 无内部依赖。
- 闭环执行统一经 `runtime/ClosedLoopEngine`，轨迹源策略可替换（Expert/Network/后续基线），指标口径唯一。
- 数据集网络闭环由 `experiments/closed_loop_evaluation.py` 编排，必须在回合开始前核对 manifest 车辆模型、任务确定性身份、selected goal、BEV 配置与 checkpoint 通道/dt；任何漂移都拒绝运行，不得回退到通用环境。T5 元数据当前不触发动态障碍注入，报告显式标为静态闭环。
- 默认理论车型由 `configs/vehicles/tracked_drill_rig.json` 定义为以两履带几何中心居中的 6×3 m 矩形；`VehicleConfig` 将外廓、执行上限、规划速度/角速度、安全余量、搜索分辨率与解析接管范围统一注入规划器/MPC/车辆模型/碰撞/inspection。`tracked_pivot_v5` 的解析接管范围为 T3 距离上限 30m。
- S4/S9 卸载区由 `TaskSampler` 将同一 `VehicleConfig` 的长、宽和 `collision_margin` 注入场景：双向主路至少 3.5 倍车宽、卸载位中心距至少 3 倍车宽，车尾到挡墙的物理净空为 `collision_margin + 0.3m`；`geometry_profile` 进入任务元数据和计划指纹，旧几何检查点不得续入。
- 批量实验统一经 `experiments/run_experiment.py`（JSON 配置驱动，结果落盘 `experiments/results/`），可视化统一经 `viz/`（PNG+PDF 双格式输出）。
- 低速泊车采用理想履带差速运动学，控制量为线速度 v 与角速度 omega；允许 `v=0, omega≠0`，原地旋转中心固定为两履带几何中心。履带滑移、沉陷、质量和惯量不属于当前理论模型。
- 平移弧线的 Reeds–Shepp 半径由 `|plan_v / plan_max_omega|` 推导，但它不代表履带全部可达集合；Hybrid A* 另有原地旋转基元和履带解析候选。默认/前进请求下倒车或反方向平移、以及原地旋转相对常规前进/小转向按 2:1 时间代价计；显式倒车监督任务仍以倒车为请求方向。原地旋转对矩形最远角点扫掠按配置分辨率加密。
- `_pose_free` 检查完整定向矩形与圆/矩形/多边形障碍相交，不再只验四角；任一运动基元、解析候选和数据可行性审计都复用相同外廓与 `collision_margin`。
- 平滑与速度剖面位于 `planner/` 内且为可选后处理，不改变三阶段共用的 `interfaces/Trajectory(points, dt)` 契约。
- `sim/tasks.py` 不依赖规划器：9×5 能力矩阵显式保留不支持单元，支持单元只保证任务几何契约；规划失败的重采样由后续数据/实验编排层负责。
- `TaskSampler` 将 `perpendicular_bay`/`diagonal_bay` 的基准航向解释为前进入位航向，倒车任务翻转 180° 使车头朝入口；`_sample_axial_start` 对普通结构化车位和紧 bay（maintenance_bay/crusher_slot/fuel_bay）从与请求机动一致的入口侧采样，并用 0.1m 步长确认目标到起点的完整车辆外廓走廊连续无碰撞。相邻占用会先过滤堵住入口走廊的目标；T4 仍携带 3–6 个候选，但把用于构造起点的参考目标放在首位，避免先在不匹配候选上耗尽规划预算。
- `dataset/build.py::expert_maneuvers` 拥有专家可生成能力：不改变任务几何矩阵，根据与车辆模型身份绑定的完整校准证据排除持续不可达/不稳定监督单元并限制单向可达单元；正式构建与后续校准共享该入口，偶发规划失败仍保持原单元难度重采。对齐采样后 S3 全单元与 S5/T1/T2/T5、S9/T2/T5 限制为倒车，S8/T2/T3/T5 限制为前进。
- `dataset/calibration.py` 的 case 计划不复用 8:1:1 数据集配额：默认枚举当前专家准入单元，新版本重校准可显式探测全部几何支持单元及双向机动。身份绑定 seed、车辆模型、探测模式、case 计划、重试和预算；只有原子终态检查点算完成，中断恢复跳过完成项并重做当前未落盘 case。
- `model/registry.py` 是模型名称到实现的唯一构造入口；`MineParkingNet` 保持 `net-v0` 兼容，v1/v2 的终止 logits 不改变闭环最终消费的 `Trajectory` 契约。
- `training/Trainer` 的 checkpoint 必须核对模型名称、模型配置和影响恢复语义的训练超参数，包括 shuffle、停止平衡和 teacher-forcing 调度；允许改变总 epochs、设备和输出目录，但不得静默加载不兼容状态。
- 启用安全损失时，train/val 必须为车辆模型与 BEV 几何一致的 schema v2；完整车体尺寸、数据碰撞余量和 occupancy 通道只从元数据解析。正式模式从 occupancy 与地图边界预计算截断有符号净空场，以规划余量加训练余量内的连续二次缺口提供梯度；旧二值最大占用仅保留作消融。安全几何、模式、额外余量、采样间距、扫掠上限和损失权重属于 checkpoint 恢复语义。
- 闭环恢复样本只从仍无碰撞的学习器访问状态生成，位置与航向偏离都参与选择，并优先保留碰撞前最后安全状态；新轨迹必须重新计算机动方向与可行性证据，不得复制原起点审计。恢复检查点绑定输入计划、权重摘要、车辆和选择参数。
- 运行时安全门禁与纯网络源正交组合：主轨迹不通过完整矩形/连续扫掠检查时从当前安全状态调用专家回退；每次控制落地前再检查下一状态转移，回退后仍不安全则以专用 `safety_stop` 结束而非执行或中断批次。报告必须区分 `none` 与 `expert_fallback`，并分别记录轨迹检查、转移检查、干预、回退失败和安全停止。闭环实际碰撞使用无安全余量的完整外廓，门禁使用车辆配置安全余量。
- `best.pt`/`last.pt` 保留训练恢复语义；`deployment.pt` 写入只在 val 上选择的停止阈值并显式标记不可恢复训练，下游推理不得退回未经校准的默认阈值，也不得使用 test 选择阈值。
- 相邻占用能力按场景×任务类型枚举实际可表达的 0/1/2 等级；T4 在占用后仍必须保留至少 3 个空闲候选位。
- 任务随机流由根 seed、场景稳定序号、任务类型稳定序号和样本索引派生；T4 不预选目标，T5 只描述触发与载荷且不在采样时修改环境。
- 传感器噪声只改变观测帧，不改变环境真值；默认 `clean` 与原输出兼容，非干净 profile 使用各传感器私有 seed/RNG，不读写 NumPy 全局随机状态。
- BEV 以车辆为中心生成；默认 40×40m@0.25m，场景可覆盖范围/分辨率但同一融合管道两路必须共用配置；S9 用 80×80m@0.5m，并与默认配置保持 160×160 栅格。
- 数据集 schema v2 将 `BEVTensor.to_metadata()` 与逐样本任务元数据编码为 Unicode JSON 数组，加载始终禁用 pickle；同一归档不允许混合 BEV 元数据或轨迹 `dt`，无版本旧归档识别为 v1。
- 默认数据构建按 8:1:1 输出 train/val/test，S9 全部且仅进入 test；构建失败只在同一场景×任务类型×难度重采，替代任务避开全计划、已用与 retry 状态排除的 task ID；未完成批次每次失败前原子持久化下一重采游标，成功 NPZ 与批次报告都落盘后才清除；未完成 split 保持 `.partial.npz`，三个 split 完成后才写 manifest。
- `dataset/maneuver.py` 是轨迹实际方向距离的唯一判定所有者；Task 请求方向默认必须占总行驶距离至少 50%，允许短距离反向调整。Task 生成先门禁候选，构建脚本再独立审计 partial；任一不一致、无效或缺失声明样本都不得晋升为正式数据集。
- `dataset/feasibility.py` 是归档运动学与模型版本审计所有者；生成期用规划器环境复核完整扫掠，partial 晋升前从数组独立复算并核对碰撞证据。旧 NPZ 或配置版本不匹配样本不得通过严格门禁。

## 阶段迁移边界

- 阶段一→二：替换传感器模拟为 CARLA 传感器，其余模块不动。
- 阶段二→三：替换 CARLA 车辆为履带底盘（新增 CAN 执行适配），其余模块不动。
