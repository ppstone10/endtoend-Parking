# LT-20261027-29 · 保留恢复谱系的环视target全量重建

状态：**已完成**（用户要求执行；train 2918条74.71秒、val 300条7.77秒，两次inspect `passed`，全部退出0）。重建与独立核验结果见文末「执行结果」。LT28失败已复核并修复解析原因；本计划新冻结145代码/4输入并使用新输出目录。

LT28的83条不是镜像增强，而是恢复生成器改写了主导机动标签。现通过原归档source_dataset_index/source_task_id认证谱系，用原任务机动严格复原场景；恢复标签与全部存储坐标保持不变。**不删样本、不放宽坐标检查，仍重建2918训练+300验证。** 旧LT28目录及handoff保留，不再用旧命令绕过身份检查。

## 目的、唯一主要变量与边界

将已核实为旧image语义的训练2918条、验证300条，仅target生成版本替换为`per_view_v2`。该版本包含逐视角图像/标定、正深度裁剪和逐视角像素/伪目标噪声；保持各档噪声参数和原任务seed，不把本实验描述为仅修改一个矩阵系数。LiDAR输入、其他四通道、任务顺序/元数据、存储状态与目标、trajs/masks/dt逐值保留，不重新规划，不按重复task_id去重。训练与验证分目录处理，不把验证样本混入训练。

本计划只重建和独立核验数据，不启动训练、停止偏置搜索或闭环评测。几何修复不证明网络收敛；旧image权重不能直接消费新输入。下一轮训练必须显式配置`target_channel: image`与`camera_projection: per_view_v2`，并重新审定训练/保持验收，不能只改旧checkpoint声明。

## 固定输入、代码与参数

- 工作目录：`D:\program\endtoend Parking`。
- 环境：`D:\conda\envs\endtoend-parking\python.exe`，Python3.12，CPU，不需要GPU。
- 训练源：`data/task_dataset/v40_dagger_collision/train-balanced.npz`，2918条；SHA `b9531bbfc3577b9a8e7db3e622cb2578fc0d3cb926fc792dfa0eadb5cf2cb8cd`。
- 验证源：`data/task_dataset/tracked_pivot_v8_3000/val.npz`，300条；SHA `ed0dcb12832354dd0155c8d2daa7c2faef03c21882b2287597393a5e5c029963`。
- 原任务manifest：`data/task_dataset/tracked_pivot_v8_3000/manifest.json`；合并训练源无自己的manifest，必须显式提供这份已核谱系/seed/车辆配置，逐任务复原仍会校验身份。
- 代码和输入冻结：`runs/verification/lt28-recovery-lineage-review-20261027/handoff.json`，145份代码、4份输入；handoff本身SHA：`8a874bd1d19e2e4d2895df6dcf6811894956efe97c35bde37cf00cd0b088a3e0`。命令先检查全部身份，变化时停止，不绕过。
- 额外原任务归档：`data/task_dataset/tracked_pivot_v8_3000/train.npz`，2400原任务；SHA `8313e72bca5539f2ecade47ecf159181c5da27db716ea2fbd075408599a6ba1f`。训练重建必须传`--original-data`；据此认证恢复来源而不把恢复机动标签用来重新采样物理任务。此文件只读，不混入新样本。
- 新短预检：518恢复帧中83条仅标签改写，2918训练身份/目标全检通过（约8.02秒）；真实源索引0/7/2400/2413/2414/2682/2917的7条边界子集重建与独立inspect通过，其中3条覆盖已改标签；45项相关单测通过。边界子集7/7只表示该子集完成，绝不是正式2918全量产物。
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
$handoffPath = 'runs/verification/lt28-recovery-lineage-review-20261027/handoff.json'
$handoffSha = (Get-FileHash -LiteralPath $handoffPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($handoffSha -ne '8a874bd1d19e2e4d2895df6dcf6811894956efe97c35bde37cf00cd0b088a3e0') { throw 'LT29 handoff身份变化，停止并反馈Agent' }
$frozen = Get-Content -LiteralPath $handoffPath -Raw | ConvertFrom-Json
foreach ($group in @($frozen.inputs, $frozen.code)) {
    foreach ($entry in $group.PSObject.Properties) {
        $actualSha = (Get-FileHash -LiteralPath $entry.Name -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualSha -ne $entry.Value) { throw "LT29文件漂移：$($entry.Name)" }
    }
}
$runRoot = 'runs/verification/surround-target-rebuild-lineage-20261027'
if (Test-Path -LiteralPath $runRoot) { throw '输出目录已存在；禁止覆盖，按恢复说明换新目录' }
New-Item -ItemType Directory -Path $runRoot | Out-Null
Start-Transcript -Path (Join-Path $runRoot 'console.log') -NoClobber
```

### 2. 重建完整训练源（2918）

```powershell
& $pythonExecutable -X utf8 scripts/repair_camera_targets.py --data $frozen.train --manifest $frozen.manifest --original-data $frozen.original_data --output (Join-Path $runRoot 'train') --limit 0 --budget-seconds 3600
if ($LASTEXITCODE -ne 0) { throw 'LT29 train失败，停止，不运行后续阶段' }
```

### 3. 独立核验训练副本

```powershell
& $pythonExecutable -X utf8 scripts/repair_camera_targets.py --inspect --output (Join-Path $runRoot 'train')
if ($LASTEXITCODE -ne 0) { throw 'LT29 train inspect失败，停止并反馈' }
```

### 4. 重建完整验证源（300）并独立核验

```powershell
& $pythonExecutable -X utf8 scripts/repair_camera_targets.py --data $frozen.val --manifest $frozen.manifest --output (Join-Path $runRoot 'val') --limit 0 --budget-seconds 3600
if ($LASTEXITCODE -ne 0) { throw 'LT29 val失败，停止，不宣布完成' }
& $pythonExecutable -X utf8 scripts/repair_camera_targets.py --inspect --output (Join-Path $runRoot 'val')
if ($LASTEXITCODE -ne 0) { throw 'LT29 val inspect失败，停止并反馈' }
Stop-Transcript
```

## 输出、日志、检查点与成功标志

- `runs/verification/surround-target-rebuild-lineage-20261027/console.log`：全部原生命令输出。
- `train/train-balanced.npz`、`val/val.npz`：新数据。
- 每个子目录的`report.json`、`manifest.json`、`artifacts.json`：完成状态、来源/代码身份、样本计数与分方向/场景/噪声命中指标、文件SHA。
- train报告的`recovery_labels_different`应为83，指恢复标签与原任务机动不同，**不是改写83条存储标签**；输出task_meta/监督应逐值保持。val该计数应为0。
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

## 执行结果

- 身份预检：handoff SHA `8a874bd1…83e0`与计划一致；4输入/145代码逐文件SHA零漂移；输出根目录`runs/verification/surround-target-rebuild-lineage-20261027`此前不存在。源数据SHA重建后复检未变：train `b9531bbf…`（8,613,994字节）、val `ed0dcb12…`、original `8313e72b…`。
- 训练重建：`rebuilt=2918`、`partial=False`、74.71秒、`status=completed`；`camera_projection`由`legacy_overlay_v1`改为`per_view_v2`；`recovery_labels_different=83`（恢复标签与原任务机动不同，**不是改写存储标签**）；输出`train/train-balanced.npz` 8,610,777字节。
- 训练inspect：`status=passed`、2918条、`labels_task_order_and_other_channels_exact=True`。
- 验证重建+inspect：300条、7.77秒、`completed`/`partial=False`，inspect `passed`；`recovery_labels_different=0`；`val/val.npz` 882,164字节。
- 两目录均产出`report.json`/`manifest.json`/`artifacts.json`（各3个登记产物，SHA全对）。所有原生命令退出码0，`console.log`为3225字节转录。
- Agent独立复核（只读）：`schema_version`/`task_meta`/`goals`/`states`/`trajs`/`masks`/`dt`在train与val均与源**逐值相同**；非target四通道摘要重算后与源及报告`other_channels_sha256`一致（train `39f2032a…`、val `a9bf5b66…`）；`bev_meta`通道列表与shape不变，仅新增`camera_projection=per_view_v2`；target通道实际改变2318/2918与248/300条，与报告`changed`计数一致；分组`groups`样本合计分别为2918/300。
- 判读：数据修复验收通过——同一批任务、同一顺序、同一监督与其他四通道，只替换target生成版本，2918+300分母完整，恢复谱系经原任务归档认证而未删除样本或放宽坐标检查。**本计划不训练**：旧image权重不能消费新输入，下一轮训练必须显式配置`target_channel: image`与`camera_projection: per_view_v2`并重新审定训练/保持验收；几何修复不证明网络收敛或闭环收益。

## 完成反馈后的独立复核

全量3218条再次流式比较与目标指标独立复算通过，耗时10.736秒；145代码/4输入及6项登记产物SHA均一致。目标在BEV内非空的分母为train 2884、val 295，平均IoU由0.280/0.275升至0.858/0.875，零交集由1887/201降至171/15。剩余分别为167/13条配置漏检、3/2条视野外、1/0条边缘栅格采样。12条high恢复帧新增零交集符合新版漏检事件，旧新随机流不保证相同；验证无新增零交集。日志到达带退出码保护的最终完成标志，但未逐项打印四个退出码与全部原生输出。数据修复验收通过，尚未训练或证明完整闭环收益。详见[独立复核](../../runs/verification/lt29-independent-review-20261027/review.md)。

