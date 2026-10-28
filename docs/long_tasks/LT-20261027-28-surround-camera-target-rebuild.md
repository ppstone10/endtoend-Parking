# LT-20261027-28 · 环视相机target全量版本化重建

状态：**执行失败，旧身份与失败产物保留**。训练阶段在索引2413因`任务身份复原漂移`停止，未运行val阶段。后续独立复核已否定「83条镜像增强」判断：83条恢复帧只把机动标签改为恢复轨迹主导方向，物理任务未镜像。重建器已增加认证原任务归档的解析修复，保留2918分母；重跑使用新冻结[LT29计划](LT-20261027-29-recovery-lineage-target-rebuild.md)，不要修改或绕过本计划旧handoff。

## 目的、唯一主要变量与边界

将已核实为旧image语义的训练2918条、验证300条，仅target生成版本替换为`per_view_v2`。该版本包含逐视角图像/标定、正深度裁剪和逐视角像素/伪目标噪声；保持各档噪声参数和原任务seed，不把本实验描述为仅修改一个矩阵系数。LiDAR输入、其他四通道、任务顺序/元数据、存储状态与目标、trajs/masks/dt逐值保留，不重新规划，不按重复task_id去重。训练与验证分目录处理，不把验证样本混入训练。

本计划只重建和独立核验数据，不启动训练、停止偏置搜索或闭环评测。几何修复不证明网络收敛；旧image权重不能直接消费新输入。下一轮训练必须显式配置`target_channel: image`与`camera_projection: per_view_v2`，并重新审定训练/保持验收，不能只改旧checkpoint声明。

## 固定输入、代码与参数

- 工作目录：`D:\program\endtoend Parking`。
- 环境：`D:\conda\envs\endtoend-parking\python.exe`，Python3.12，CPU，不需要GPU。
- 训练源：`data/task_dataset/v40_dagger_collision/train-balanced.npz`，2918条；SHA `b9531bbfc3577b9a8e7db3e622cb2578fc0d3cb926fc792dfa0eadb5cf2cb8cd`。
- 验证源：`data/task_dataset/tracked_pivot_v8_3000/val.npz`，300条；SHA `ed0dcb12832354dd0155c8d2daa7c2faef03c21882b2287597393a5e5c029963`。
- 原任务manifest：`data/task_dataset/tracked_pivot_v8_3000/manifest.json`；合并训练源无自己的manifest，必须显式提供这份已核谱系/seed/车辆配置，逐任务复原仍会校验身份。
- 代码和输入冻结：`runs/verification/surround-projection-fix-20261027/handoff.json`，145份代码、3份输入；handoff本身SHA：`a9834cb1f32f119747e23c32eaca50532cd8340d4dffb5590938049d0c612285`。命令先检查全部身份，变化时停止，不绕过。
- 固定参数：`--limit 0`（完整），每个重建阶段预算3600秒；原视角0/90/-90/180度，各视角安装height1.5m/pitch30度，BEV0.25m、前后左右20m；具体原任务噪声/车辆/目标取其存储身份。
- 短预检：train前8/val前4重建与独立inspect通过，非target及监督/身份顺序精确保持；训练3条后向零命中改为IoU1；验证一条high噪声保留配置漏检并非几何复发。另抽查真实恢复2400/2917通过；138项聚合测试及最终15项定向检查通过。

## 耗时与资源估算

预计重建加核验共约3–10分钟，CPU为主，可用内存建议至少8GB，额外磁盘建议至少1GB。依据：本机train前8重建0.325秒、val前4为0.134秒；全量还包括1.5GB训练BEV解压/压缩、任务复原与独立保留项比较，不能线性保证耗时。新NPZ压缩后远小于内存展开尺寸，独立核验会同时持有新旧数组，峰值约4–6GB，具体依NumPy/文件缓存而变。按正式大规模数据重建交用户执行。

## PowerShell执行步骤（有先后依赖）

### 1. 环境、身份与新输出目录

```powershell
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$pythonExecutable = 'D:\conda\envs\endtoend-parking\python.exe'
$handoffPath = 'runs/verification/surround-projection-fix-20261027/handoff.json'
$handoffSha = (Get-FileHash -LiteralPath $handoffPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($handoffSha -ne 'a9834cb1f32f119747e23c32eaca50532cd8340d4dffb5590938049d0c612285') { throw 'LT28 handoff身份变化，停止并反馈Agent' }
$frozen = Get-Content -LiteralPath $handoffPath -Raw | ConvertFrom-Json
foreach ($group in @($frozen.inputs, $frozen.code)) {
    foreach ($entry in $group.PSObject.Properties) {
        $actualSha = (Get-FileHash -LiteralPath $entry.Name -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualSha -ne $entry.Value) { throw "LT28文件漂移：$($entry.Name)" }
    }
}
$runRoot = 'runs/verification/surround-target-rebuild-20261027'
if (Test-Path -LiteralPath $runRoot) { throw '输出目录已存在；禁止覆盖，按恢复说明换新目录' }
New-Item -ItemType Directory -Path $runRoot | Out-Null
Start-Transcript -Path (Join-Path $runRoot 'console.log') -NoClobber
```

### 2. 重建完整训练源（2918）

```powershell
& $pythonExecutable -X utf8 scripts/repair_camera_targets.py --data $frozen.train --manifest $frozen.manifest --output (Join-Path $runRoot 'train') --limit 0 --budget-seconds 3600
if ($LASTEXITCODE -ne 0) { throw 'LT28 train失败，停止，不运行后续阶段' }
```

### 3. 独立核验训练副本

```powershell
& $pythonExecutable -X utf8 scripts/repair_camera_targets.py --inspect --output (Join-Path $runRoot 'train')
if ($LASTEXITCODE -ne 0) { throw 'LT28 train inspect失败，停止并反馈' }
```

### 4. 重建完整验证源（300）并独立核验

```powershell
& $pythonExecutable -X utf8 scripts/repair_camera_targets.py --data $frozen.val --manifest $frozen.manifest --output (Join-Path $runRoot 'val') --limit 0 --budget-seconds 3600
if ($LASTEXITCODE -ne 0) { throw 'LT28 val失败，停止，不宣布完成' }
& $pythonExecutable -X utf8 scripts/repair_camera_targets.py --inspect --output (Join-Path $runRoot 'val')
if ($LASTEXITCODE -ne 0) { throw 'LT28 val inspect失败，停止并反馈' }
Stop-Transcript
```

## 输出、日志、检查点与成功标志

- `runs/verification/surround-target-rebuild-20261027/console.log`：全部原生命令输出。
- `train/train-balanced.npz`、`val/val.npz`：新数据。
- 每个子目录的`report.json`、`manifest.json`、`artifacts.json`：完成状态、来源/代码身份、样本计数与分方向/场景/噪声命中指标、文件SHA。
- 正常进度每100条输出`[camera-target n/N]`；本计划不设中间NPZ检查点，不支持从未完成目录续写，失败后换新目录重跑。
- 重建两份均输出`status=completed`、`partial=False`，条数严格2918和300；两次inspect均`status=passed`且`labels_task_order_and_other_channels_exact=True`；各原生命令退出码0。
- 不要求high/low噪声目标命中率全部为1，也不能把`completed`或四方向测试通过当成全五层/生产闭环合格。

## 异常停止、恢复和禁止覆盖

- 任一身份检查、异常、预算耗尽、内存不足、任务复原漂移或inspect失败立即停止后续步骤。若Transcript仍开启，执行`Stop-Transcript`。
- 保留失败目录和`report.json`，反馈首个异常与日志尾部。`status=incomplete`或缺少artifacts的目录不可用于训练。
- 重跑先由Agent诊断；仅输出路径冲突时可将步骤1的`$runRoot`改为新的`...-retry1`等未存在目录，重新执行全部身份检查；不得覆盖或删除旧产物以绕过保护。
- 若训练阶段已完成并inspect通过、验证阶段异常，训练副本可保留；重跑验证须使用新子目录并再次指定相同冻结输入。重新打开PowerShell须先恢复步骤1中的工作目录、Python/frozen变量并重做身份检查，日志使用新文件。
- 禁止覆盖任一`data/task_dataset`源、现有权重、LT21–LT27产物、短预检目录或已有重建目录；不启动任何训练或后续闭环作业。

## 完成后反馈与Agent验收

反馈输出根目录和`console.log`、两份`report.json/manifest.json/artifacts.json`。NPZ保留在原位置即可，无需上传。

Agent将核验任务身份、日志退出/完成状态、2918/300分母、来源/代码/产物SHA及独立逐字段/逐通道保持，再分析每场景×噪声×方向的目标命中变化，区分配置漏检、视野覆盖与几何残余。只在数据修复验收后设计训练与旧能力保持实验，不自动晋升旧模型或串行启动下一长任务。

## 执行结果（受阻）

以下保留当时执行与诊断记录；其中「镜像副本」解释及删样本/放宽身份选项已被文末独立复核取代，不作为当前处理依据。

- 身份预检：handoff SHA `a9834cb1…2285`与计划一致；3输入/145代码逐文件SHA全部一致，零漂移；输出根目录`runs/verification/surround-target-rebuild-20261027`此前不存在。
- 训练阶段：`repair_camera_targets.py --data …train-balanced.npz --limit 0` 正常推进到`[camera-target 2400/2918]`后**异常退出**：`ValueError: 任务身份复原漂移：S1_parking_lot-T1-0009-6cb9bd09`（`experiments/closed_loop_evaluation.reconstruct_dataset_task`第103行）。按计划"任务复原漂移立即停止"，**未运行训练inspect，也未运行val阶段/val inspect**。目录中保留`train/report.json`（`status=incomplete`、记录了错误）与`console.log`，未产生任何NPZ。
- 诊断（Agent只读复算，未改任何代码/数据）：用同一manifest（seed 20260824）对全部2918条训练任务重做身份复原，发现**83条漂移**，索引连续为2413–2682（S1_parking_lot 50条、S2_diagonal_lot 33条）；差异字段为`start`+`goal`（71条）、`start`+`candidate_goals`（11条，goal为None）、`start`+`goal`+`dynamic_event`（1条）。其中50条的start yaw、42条的goal yaw恰为复原值的**取反**（镜像增强），且沿用原`task_id`（例如`S1_parking_lot-T1-0009-6cb9bd09`在索引7/2412/2413各出现一次，前两条可复原、第三条为镜像副本）。
- 验证源不受影响：同法扫描`val.npz`300条为**0漂移、0复原错误**，说明身份关卡本身工作正常，问题只在合并训练源的增强尾巴。
- 待决定（需用户/新计划授权，本轮未执行任何绕过）：
  1. **排除**这83条镜像副本后重建（训练集变为2835条），需接受样本数变化；
  2. **放宽身份复原**：允许"同task_id + start/goal yaw镜像"的已登记增强谱系，重建时以存储start位姿而非seed复原场景（需改`surround_rebuild`与验收契约并重新冻结身份）；
  3. 其它口径（例如上游把镜像增强改为独立task_id重新生成）。
- 边界与保护：源`data/task_dataset/**`、旧权重、LT21–LT27产物、短检目录均未改动；未训练、未评测、未部署；失败目录与`report.json`按要求保留，可换新目录重跑但需先决定上述口径。

## 独立复核更正与修复

- 原任务归档`tracked_pivot_v8_3000/train.npz`与合并训练源逐项核对：518恢复帧中83条只改变difficulty.maneuver，其他任务核心字段、噪声、车辆与selected_goal均一致。生产者`dataset/recovery.py::build_recovery_sample`明确执行此标签改写。
- 用恢复标签重新采样原始任务会得到另一朝向/起点，旧诊断把这两个采样结果的差异误当成存储数据镜像。数据自身未发生该坐标镜像，不能据此删除83条。
- 新重建器显式接收原任务归档，按source_dataset_index/source_task_id认证谱系，仅在内存中以原任务机动解释身份，再调用原严格复原器。恢复标签及全部存储数组不改，未知坐标/事件漂移仍拒绝。
- 2918条训练元数据严格复原和目标核验全通过；7条真实边界样本（含2413/2414/2682）短重建与独立inspect通过；45项相关回归通过。正式2918+300重建尚未重跑。
- 证据：`runs/verification/lt28-recovery-lineage-review-20261027/{audit.json,preflight.json,tests.json}`；新入口/身份与命令见LT29。
