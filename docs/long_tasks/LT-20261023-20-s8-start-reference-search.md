# LT20：S8起点参考搜索与固定见证核验

状态：**已执行完成**（用户要求执行本计划命令块；71.760秒跑完，退出码0）。两源均未通过资格，`qualification_passed=false`，按计划结束本固定配方。结果与独立重放见文末「执行结果」。工具/冻结配置及短时冒烟此前已完成。只使用下列v4配置，v1/v2为失败预检、v3为收敛前配置，均保留而不继续使用。

## 目的与唯一主要变量

保持原MPC、参考xy/端点/dt、任务起点、600步上限和0.2m连续扫掠验收，只改变离线参考的七个有界航向参数，用真实完整执行反馈寻找严格教师。不是学生训练、控制器修改、主协议评测或部署。

短检已发现2351的固定见证：七参数全−.001rad，从任务起点244步无碰撞/无余量侵入，首窗clean；尚未做float32导出和独立控制流资格核验。因此正式任务只对2338搜索，2351直接核验已有候选，不重复搜索。

## 固定输入与版本

- 工作目录：`D:\program\endtoend Parking`。
- 环境：`D:\conda\envs\endtoend-parking\python.exe`，Python3.12；不使用系统Python，不安装新依赖。NumPy2.5.2；SciPy未安装，本工具不依赖它。
- 入口：`scripts/search_s8_start_references.py`，模块`experiments/s8_reference_search.py`、`s8_clearance_surrogate.py`；以配置中的152份代码SHA256为实际版本，当前工作树有既有未提交/未跟踪内容，不以HEAD代替版本身份。运行期间不要修改冻结代码/数据。
- 准备配置：`runs/prepared/s8-start-reference-search-20261023-v4.json`。
- 配置SHA256：`8077359c541ae34f3f9f258e6052b2998e13db189e8a42efb0736b783b49dcb8`。
- 冻结821输入文件，包含原训练数据、旧模型/运行身份与参考证据链；正式搜索只执行原MPC固定参考，不更新或生成模型检查点。
- 训练数据：`data/task_dataset/tracked_pivot_v8_3000/train.npz`及原manifest/车辆配置，源2338/2351，无val/selection/挑战查询。
- 原参考：`runs/diagnostics/s8-entry-recovery-preflight-20261022-v1/source-20261020-2338/query-0/teacher-1.npz`、同目录体系2351的`teacher-2.npz`；未补偿`teacher-0`仅约束总航向偏移。fresh基线必须精确重现240步37侵入、245步2侵入。
- 固定见证：`runs/diagnostics/s8-start-reference-search-smoke-20261023-v3/source-2351/eval-0001/execution.npz`及v3报告/配置/17产物、三个新源快照；新工具核验归档代码SHA，不要求旧CLI与当前CLI字节相同。

## 固定参数与预算

- 航向结点相位`[0,.025,.10,.25,.50,.75,.90,.975,1]`，首尾0，七内部参数各±.03rad，线性插值；相对未补偿参考总偏移≤.1rad（float32容差1e-7），xy/端点/H/dt不变，不裁剪非法航向参考。
- 2338：NumPy DE/rand/1/bin，七维、28种群、5代、F=.5、CR=.7、立即严格改善更新，初始首个为零参数基线，种子`20261023+2338`，最多168尝试。几何/航向拒绝也计尝试，实际回合数另记；首次control0严格即停，不做第二配方。
- 2351：仅零参数基线和七参数全−.001rad两个尝试；候选参考、状态、控制、余量必须精确重现已存见证，否则停止，不搜索替代参数。
- 每源出现严格见证后：检查真实首窗/输入，float32局部→全局导出control0完整执行；再在control_seed1、2分别配对执行原参考/新参考。这两个随机流不参与优化，核验失败不重新搜索。
- 排名先看严格/接触/到达/实际侵入，再用与原矩形SAT对应的连续代理穿入程度及终点误差作字典序次级比较。代理不是安全判据；禁止用“更少侵入”冒充严格成功。
- 最多170候选尝试，加上至多10次导出/独立控制流核验，实际完整回合最多180；单回合最多600步。总墙钟硬上限1800秒，预算耗尽保留未完整状态，不自动扩大。

## 耗时与资源估计

实测原240/245步单回合约0.35–0.73秒；候选可能到600步，另有几何筛选、输入校验及落盘。预计约3–10分钟，慢盘/较多600步回合可能更久，最迟由30分钟预算停止；这不是可靠短任务，按规则由用户正式运行。若很早得到严格见证也可能提前结束。

CPU单线程、无需GPU；训练归档展开约1.18GiB，预计进程内存2–4GiB（未测峰值），建议留至少4GiB可用内存。当前四回合smoke约285KiB，正式结果通常几十MiB，建议预留0.5GiB磁盘余量。上述是基于本机短检的估算，不保证耗时或一定找到可用教师。

## PowerShell命令：必须按顺序执行

### 1. 身份与输出占用预检

在PowerShell中执行整段；所有路径都已实际存在，只有正式输出/日志必须不存在。

```powershell
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$lt20Python = 'D:\conda\envs\endtoend-parking\python.exe'
$lt20Config = 'D:\program\endtoend Parking\runs\prepared\s8-start-reference-search-20261023-v4.json'
$lt20Output = 'D:\program\endtoend Parking\runs\diagnostics\s8-start-reference-search-20261023-v1'
$lt20Log = 'D:\program\endtoend Parking\runs\diagnostics\s8-start-reference-search-20261023-v1.console.log'
$lt20ExitFile = 'D:\program\endtoend Parking\runs\diagnostics\s8-start-reference-search-20261023-v1.exit-code.txt'
if (-not (Test-Path -LiteralPath $lt20Python -PathType Leaf)) { throw '指定conda Python不存在' }
if ((Get-FileHash -LiteralPath $lt20Config -Algorithm SHA256).Hash.ToLowerInvariant() -ne '8077359c541ae34f3f9f258e6052b2998e13db189e8a42efb0736b783b49dcb8') { throw 'LT20配置SHA256不符' }
if ((Test-Path -LiteralPath $lt20Output) -or (Test-Path -LiteralPath $lt20Log) -or (Test-Path -LiteralPath $lt20ExitFile)) { throw '输出/日志/退出文件已有内容，禁止覆盖；先反馈现有产物' }
& $lt20Python -u scripts/search_s8_start_references.py --precheck --config $lt20Config
if ($LASTEXITCODE -ne 0) { throw 'LT20预检失败，不启动正式任务' }
```

预检末行应为`LT20 INPUTS/PROTOCOL VERIFIED; ZERO ROLLOUT/SEARCH/TRAINING`。失败时先反馈错误，不删除文件或重建配置绕过身份检查。

### 2. 正式运行

必须在步骤1成功的同一PowerShell窗口执行。该命令只运行本LT20，不串行启动学生训练或其他阶段。

```powershell
& $lt20Python -u scripts/search_s8_start_references.py --run --config $lt20Config --output $lt20Output 2>&1 | Tee-Object -FilePath $lt20Log
$lt20Exit = $LASTEXITCODE
Set-Content -LiteralPath $lt20ExitFile -Value $lt20Exit -Encoding utf8
if ($lt20Exit -ne 0) { throw "LT20退出码 $lt20Exit；保留产物并反馈，不自动重跑" }
```

### 3. 查看完成摘要

```powershell
$lt20Report = Get-Content -LiteralPath (Join-Path $lt20Output 'report.json') -Raw | ConvertFrom-Json
$lt20Report | Select-Object status, stage, mode, elapsed_seconds, decision, complete_sources, qualification_passed, training_updates
$lt20Report.sources | Select-Object index, baseline_exact_verified, fixed_witness_exact_verified, attempt_count, executed_episodes, decision, qualified
Get-Content -LiteralPath $lt20Log -Tail 12
```

## 正常进度、产物与成功标志

控制台持续输出`EVAL <source> <number> <decision> <rank>`，可能因几何拒绝很快，也可能完整执行数秒；不是固定每条都成功。源2338首次严格后提前核验，未找到则最多168尝试；2351只2尝试及固定核验。

输出根目录保存`config.json`、持续更新的`report.json`；各`source-2338/2351/`下有`eval-NNNN/attempt.json`、`reference.npz`，实际运行另有`execution.npz`。`best.json`和`best-reference.npz`为当前最好检查点；2338完整代际还存`generation-NN.npz/json`（种群、排名、RNG）。严格候选另有`first-frame.pt`（仅诊断）、`float32-export-control0.npz`以及baseline/candidate的control1/2执行记录。没有新网络模型检查点或训练银行。

正常退出码0、末行`FINISHED teacher_search_only_not_learning_or_deployment ...`，根`status=completed`、`stage=decision`、`complete_sources=2`。**运行完整不等于教师达标**：`qualification_passed=true`必须同时两起点通过control0/1/2严格执行、导出严格和首窗clean；false是完整负结果，不是要求继续搜索的信号。`training_updates=0`、`teacher_bank_created=false`、训练/主验收/部署放行始终false。

## 异常停止、恢复与重跑

- 身份不符、基线/固定见证漂移、非有限、未知障碍类型或异常时立即停止并保留报告/日志。不得修改冻结文件、降低余量或换参数继续。
- Ctrl+C可主动停止；若Python收到中断，会尽量写`status=interrupted`。强制结束可能只留下running报告，以日志/文件为准，不能宣布完成。
- 超30分钟会报`budget_exhausted`，缺项按未完成处理。不要后台续跑、拆阶段补跑或扩大种群/代数。
- 本版本不支持原位续算；代际文件用于复核，不保证优化器恢复。确需重跑时先反馈，由Agent诊断后沿用同v4配置，仅把输出、日志、退出码路径都改为新的`...-retry1`等目录，从同seed重启。旧产物不删、不覆盖。
- 若终端提示很久无新日志，先反馈当前日志尾部/报告stage，不另起重复进程。

## 禁止覆盖与完成反馈

所有已有`runs/diagnostics/s8-*20261022*`、`s8-start-reference-search-smoke-20261023-v1/v2/v3/v4`、已准备v1/v2/v3/v4配置及其他用户产物都不得覆盖。正式输出目录若存在也不可复用。不要执行`git add .`、commit或push。

完成后回复“LT20执行完毕”；若沿用默认路径无需另列文件，Agent会核验：正式根`config.json/report.json`、控制台日志与退出码文件、两源全部attempt/NPZ、best/代际、存在时的首帧/导出/独立控制流记录。使用其他输出路径时请给出绝对路径；异常时附日志末尾和`report.json`。

## Agent后续验收与决策

先核对v4配置SHA、152代码/821输入身份、日志/退出状态/非截断、候选与实际回合分母、全部登记文件SHA及代际/最好引用；再重算原基线和2351固定见证的精确一致性、候选排名、首窗/导出/配对随机流。独立重放关键严格/失败回合后再裁决。

两起点均合格时，才另行设计监督银行/学习对照；单源合格只保存局部证据。无严格候选或独立核验失败时，结束本固定配方，不盲扫参数、不从有限族失败推断架构无解，不以教师结果代替学生闭环或主≥70%/≤10%联合验收。

规范：[S8-START](../../spec/s8_start_reference_search.md)。准备与短检证据见`runs/diagnostics/s8-start-reference-search-smoke-20261023-v4/summary.md`。

## 执行结果

- 执行：`status=completed`、`stage=decision`、`complete_sources=2`、`elapsed_seconds=71.7604`、`decision=teacher_search_only_not_learning_or_deployment`、`qualification_passed=false`、`training_updates=0`、`teacher_bank_created=false`、`main_protocol_passed/automatic_deployment/allowed_to_prepare_training_plan`均false；退出码0（`...v1.exit-code.txt`=0），控制台171行=170条`EVAL`+1条`FINISHED`，无Traceback。
- 身份：v4配置SHA256与`8077359c…b49dcb8`一致，报告登记的`config_sha256`重算一致；precheck核验152代码/821输入通过；报告登记493个产物SHA在本机重算全部一致（仅`report.json`自身不在清单内）。
- 分母：候选尝试170（2338=168、2351=2，等于上限）；真实回合130（128+2）；另加2351的首窗/float32导出/4条配对核验共5回合，合计135≤180，单回合≤600步；耗时71.76秒远低于1800秒预算。
- 2338：DE/rand/1/bin 28种群×6代=168尝试，40次参考几何拒绝、128次真实执行；128次全部为stage1（无碰撞到达但仍有余量侵入），0次stage0，0次碰撞；最佳#129为243步/24侵入步（基线为240步/37侵入），最大代理穿入0.142、终点误差0.210/0.077。
- 2351：2尝试。零基线245步/2侵入，与归档逐元素一致；固定见证（七参数全−0.001rad）control0为244步/0侵入严格。核验：首窗clean（执行10步0侵入、参考11段0侵入、输入SHA`42674bb4…`）；float32导出244步/0侵入严格；配对control1候选240步/0侵入、基线240步/0；**配对control2候选239步/2侵入（首侵入第11步）、同流基线235步/0侵入**。故决策为`strict_search_witness_failed_independent_checks`。
- Agent独立重放（内存内、不写产物）：2338零基线精确复现归档`teacher-1.npz`与本轮`eval-0000`；2351 witness control0与v3 smoke记录（reference/states/commands/margin_violations）逐元素一致；witness control1/2、baseline control1/2、float32导出与登记NPZ逐元素一致。导出参考与witness参考不同属既有设计（float32监督点重新转换），其执行仍严格。
- 判读：固定七参数±0.03rad/总偏移≤0.1rad族在2338上不足（最好仍24侵入步，仅比基线37步略好）；2351的单点见证不跨独立控制流稳健，且同流原基线反而更干净，说明该−0.001rad补偿不是稳定教师，不能作为监督银行依据。两源均不合格：按计划结束本固定配方，不盲扫参数、不从有限族失败推断架构无解、不建立训练银行或学生对照，主成绩68.3%/25.7%与全部门禁不变。

### 独立审计与更正 · T-132839

新增只读入口`scripts/audit_s8_start_reference_search.py`及[审计摘要](../../runs/diagnostics/s8-start-reference-search-review-20261023-v2/summary.md)。11.894秒复核152代码/821输入/493产物、170候选、全部参考几何及135回合32,685次转移，6份DE代际的种群/排名/RNG精确重建；新鲜重放2338最好#129及2351 control2原参考/候选共3回合，状态/控制/余量/汇总精确一致。未重跑搜索或训练。

更正上方执行结果的“终点误差0.210/0.077”：#129实际位置误差为**0.0771758701m**、航向误差为**0.0460746312rad**；0.210415927是代理穿入平方和。相较基线侵入步数减少，但平方和从0.198968780上升，不能声称全面安全改善。旧执行记录保留，此处更正。

早段分析：2338全部128真实分支首次侵入在索引5/7/8（53/1/74条）；2351 control2候选第1步控制分岔，第11/12步侵入，原参考同流全程零侵入。后续优先调查起步参考消费与实际跟踪，暂不扩大同族搜索或设计学生长训。独立审计首版仅因Windows UTF16控制台解码失败，修正审计器后v2通过，未修改原152冻结源或正式结果。
