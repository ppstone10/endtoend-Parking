# 环视相机几何与输入版本 Spec

## 元数据

- Spec ID 前缀：`CAM-PROJ`
- 强度：完整
- 状态：已采纳
- 最后更新：2026-10-27

## 目标

相机目标在统一车辆局部BEV保持方向，裁剪不制造近零深度伪影；修复输入与旧训练输入可区分，模型不能静默跨版本使用。

## 非目标

不改变MPC、网络结构、已有权重或监督轨迹；不以几何回归通过声明完整闭环晋升；不解决真实相机畸变、遮挡或完整六自由度安装标定。

## 边界与约束

当前模拟相机安装在车体中心上方，保留每视角yaw、height、pitch与内参。逐栅格在该相机局部系采样，再在同一BEV网格取最大值；禁止先把不同视角像素合并后按单一单应投影。噪声按确定性随机种子生成。

## 行为与验收

### `CAM-PROJ-001`：视角身份与车体几何

- CameraFrame携带各视角图像和标定；image仅作兼容预览，投影消费各视角。
- 前后左右8m干净目标应与独立几何栅格一致；左右输出不能相同。覆盖非零全局车姿、非正交yaw和非对称BEV范围；不使用数组旋转代替一般坐标变换。
- 无视角集合的旧单帧仍按单前视处理；非法图像、非有限标定或无效高度必须拒绝。
- 原L1诊断采样不再强制车辆朝向目标，报告记录独立均匀航向采样；仍只作诊断，不替代实际任务分布五层验收。

### `CAM-PROJ-002`：正深度裁剪

- 内外判断和交点计算使用同一正深度eps裁剪面；透视除法不得对零深度交点求值。
- 相机平面交叉多边形应保持有限坐标；后向8m对照不得出现已知34伪格。无目标、视野外和视角重叠不产生重复强度。

### `CAM-PROJ-003`：数据/模型版本关卡

- 新image BEV元数据携带`camera_projection=per_view_v2`；缺省旧输入解释为`legacy_overlay_v1`，绝不补写新版本冒充重建。
- 模型配置接受显式投影版本并持久化；未声明的旧checkpoint保持旧语义。训练数据校验与numpy推理入口均拒绝image版本不匹配。
- 明确的新image输入不能被声明geometry的模型消费；train/val投影版本必须一致，未知版本拒绝。
- geometry目标与明确GT诊断源独立于相机投影版本，不把GT对照称为真实相机修复。
- 数据保存保留版本，同一数据集混入版本必须拒绝。

### `CAM-PROJ-004`：保留身份与监督的重建

- 只在新目录写入目标通道修复副本；保存原样本顺序、任务元数据、goals/states/trajs/masks/dt和其他BEV通道，支持同任务多条恢复帧，不按task_id去重或重新规划目标。
- 使用原任务噪声种子、存储目标和当前存储状态生成新相机target；保存来源SHA与代码身份、旧新版本、样本分母及按方向目标命中指标，错误停止、不跳过样本。
- 输出与源目录重合或目标目录已存在、无完整任务身份、目标与所选车位不一致、非有限状态/目标或版本未知时拒绝；试运行限制不得被标为完整重建。
- 合并训练源无manifest时，必须显式提供原任务manifest路径；来源路径/SHA登记并逐任务验证seed/车辆/身份，不猜测或伪造manifest。
- 长时间运行由用户执行；短预检验证产物身份和逐元素保持，不训练模型。

### `CAM-PROJ-005`：原任务机动与恢复标签分离

- 恢复生产者会把difficulty.maneuver改写为恢复轨迹标签，该标签不代表原始场景采样机动；不能据错误复原出来的yaw取反判断存储样本是镜像增强。
- 重建可显式接收原任务归档，通过recovery.source_dataset_index/source_task_id定位原任务，登记归档路径/SHA及原manifest身份。恢复协议必须为既有closed_loop_recovery/expert_replan_from_learner_state，机动标签须与其maneuver_audit一致且审计通过。
- 只在内存中用原任务maneuver解释任务身份；核心任务字段、其他difficulty字段、噪声、车辆和selected_goal仍须严格一致，再调用原严格复原器验证。无法核实谱系或真实坐标漂移必须失败，不尝试相反方向碰运气。
- 存储元数据、恢复标签及所有监督数组原样保留，不删除83条、不改变2918分母、不镜像坐标或标签。报告记录原任务机动、恢复机动及标签变化计数；inspect独立核验谱系和保留项。
- 不修改公共reconstruct_dataset_task的既有严格规则；旧无原任务归档调用仍保持原有严格拒绝行为。

## 数据、接口与迁移

- CameraView描述当前安装约束下的单视角；CameraFrame维持旧image/intrinsics/timestamp参数位置，新增views。
- BEVTensor增加可选投影来源；未声明元数据仍可读取，不能与per_view_v2混用。新模型需在配置显式设置版本；旧权重保持可加载，遇到新image输入报告可操作错误。
- 回退通过保留的旧代码/数据/权重进行；修复不提供可上线的错误环视兼容路径。旧LT27证据与SHA不改写，新结果须用新协议。
- LT29已完成2918训练/300验证全量重建并独立验收；重训与闭环收益仍为后续人工阶段。不修改已有源文件或执行不可逆迁移。数据为本地模拟产物，不新增外部传输。

## 追溯

| Spec ID | 验收 | 测试或人工入口 | 实现符号 | 实际验证 | 状态 |
|---|---|---|---|---|---|
| CAM-PROJ-001 | 四方向/车姿/一般yaw/标定 | tests.test_surround_projection.TestSurroundGeometry | CameraView；SimulatedCamera.capture；Camera2BEV.to_bev | 四方向8m逐格真值一致；非零车姿/一般yaw/非对称网格与安装参数、重叠/噪声/空目标回归通过；138项聚合通过 | ✅ |
| CAM-PROJ-002 | 裁剪正深度及伪格归零 | TestSurroundGeometry.test_clipped_intersections_stay_at_positive_near_plane；四方向测试 | SimulatedCamera._clip_to_camera_front | 修复前5处失败；修复后四方向含后向无伪格，与独立真值逐格一致 | ✅ |
| CAM-PROJ-003 | 训练与推理拒绝跨版本 | TestProjectionContract；TestSurroundRebuild.test_checkpoint_roundtrip_preserves_version_without_changing_weights | BEVTensor；assert_camera_projection_matches；validate_model_dataset；runner._validate_dataset_pair；模型predict | v0/v1/v2新旧版本交叉拒绝；旧权重逐值恢复；真实旧部署模型拒绝per_view_v2；未知输入/geometry声明及混版本关卡通过 | ✅ |
| CAM-PROJ-004 | 重建仅改target/新目录/原身份 | tests.test_surround_rebuild；scripts/repair_camera_targets.py --inspect；LT29 | repair_camera_targets；inspect_camera_target_repair | LT28在索引2413停止（恢复机动标签误作采样参数）；LT29重跑完成：train 2918条74.71秒、val 300条7.77秒均`completed`/`partial=False`，两次inspect `passed`且`labels_task_order_and_other_channels_exact=True`；独立复算task_meta/goals/states/trajs/masks/dt逐值相同、非target四通道摘要与源一致、target更新2318/248条、`recovery_labels_different`=83/0 | ✅ |
| CAM-PROJ-005 | 认证原任务谱系，不混淆机动语义 | TestSurroundRebuild.test_recovery_relabel_uses_original_identity_without_changing_stored_labels；test_recovery_lineage_rejects_real_drift_and_forged_parent；真实身份预检 | load_original_tasks；resolve_rebuild_task；inspect_camera_target_repair | 518恢复帧中83仅标签改写，2918条严格身份/目标核验通过；7条真实边界副本与独立谱系核验通过；45相关测试通过；真实坐标/ID/索引/噪声/selected_goal篡改拒绝 | ✅ |

LT29补充独立证据：目标真值在BEV内非空的训练2884/验证295条，平均IoU由0.280/0.275提升至0.858/0.875；零交集由1887/201降至171/15。剩余项包含配置漏检、视野外及1条边缘采样；12条高噪声恢复帧新增零交集，噪声参数/seed保持不代表旧新随机流相同。详见[runs/verification/lt29-independent-review-20261027/review.md](../runs/verification/lt29-independent-review-20261027/review.md)。该数据验收不替代五层质量及完整闭环验收。

## 待人工确认

无。用户已授权落实修复；长任务按项目固定分工交用户运行。
