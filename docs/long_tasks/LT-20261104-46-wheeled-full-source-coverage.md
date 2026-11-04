# LT-20261104-46 · 全训练/验证源轮式教师覆盖调查

**状态：待用户运行。入口、2700任务冻结配置和短预检齐备；全量run/inspect尚未运行。**

## 目的与唯一主要变量

LT45原22任务发现3例跟踪超时，停车段几何前馈修复后22/22通过并独立重放。本任务将调查范围从固定分层面板扩大为原train2400和val300全部源任务；唯一主要变量为源任务覆盖范围。车型、原几何/目标/容差、规划与执行预算保持，不扫参、不筛掉失败、不修改场景、不训练。

本轮回答有限RS教师与同一轮式MPC能覆盖哪些原任务，哪些任务需要进一步定位或重新定义数据建设方案。完整负结果同样有研究价值，不能因22例成功宣布2700例监督就绪。保留test不读取，旧标签和检查点均不覆盖。

## 固定输入与版本

- 工作目录：`D:\program\endtoend Parking`；唯一Python：`D:\conda\envs\endtoend-parking\python.exe`，Python3.12.13、NumPy2.5.2。
- prepared：`runs/prepared/wheeled-coverage-20261104-v1/config.json`，SHA256 `e319cf5bdfc830227b11ef5791f5f0024da4447b18536ef416ee43f76ac741b9`。
- 使用当前任务T-145635修改后的源码；精确版本由prepared内333项源码/输入SHA确定。Git提交不改变内容时不影响前检；改源码或源文件则停止，不手工重算SHA放行。
- 输入：`data/task_dataset/tracked_pivot_v8_3000/{train,val}.npz`及原manifest。选择`all_train_val`，先train索引0～2399，再val索引0～299，共2700个唯一task ID。不含恢复样本、不触碰test，不读取BEV大数组。
- 轮式参数：`configs/vehicles/wheeled_research_v1.json`，身份`122399edc4d42fcf1c616c4e25636891827a34127e27e0a6be01db36642a1003`。参数为研究假设，非真实矿卡标定。外廓6×3m、余量0.2m、dt0.1；初始v0、转角0是明确假设。
- 有限RS最多48词族、巡航上限0.5m/s、规划3秒/最多600步；同模型停车转轮/采样加减速。CEM horizon8、population8、iterations1、seed7，执行最多600步、单例资源预算120秒，各原车位位置/航向容差，教师请求方向里程占比至少50%。
- 所有参考及真实执行按GT车体转移审核；T5事件仍不动态注入，输出显式标记静态。只报告位姿到达，不是停稳、感知安全或真实制动保证。
- 无GPU、无网络推理、无训练NPZ导出。新全源模式保留旧默认22面板语义。

## 已完成预检

- 指定conda下92项聚合测试通过，覆盖轮式模型/执行/规划/数据与旧MPC/执行/RS/车辆回归，包含停车预转轮、全2700源顺序/唯一性、缩分母/改模式/test/恢复数据拒绝。
- 全源prepare与precheck确认2700源任务及333冻结文件，无全量回合；运行脚本以`-Phase Precheck`进行实际短检查，不创建正式输出目录。
- 最终源码的22面板独立运行与逐控制审核见`runs/verification/wheeled-calibration-formal-20261104-v3/`及同级日志。原LT45 v1的19/22负结果和修复后v2现场保留。

## 执行方法

在PowerShell执行下面这一条命令。脚本依次进行环境/身份前检、2700任务run、完整inspect，并保存两阶段日志和退出码；**不要提前创建输出目录**。

```powershell
& 'D:\program\endtoend Parking\docs\long_tasks\LT-20261104-46-wheeled-full-source-coverage.ps1'
```

脚本相对自身位置解析项目工作目录；使用指定conda解释器、单线程OMP/MKL/OPENBLAS，并检查Python/NumPy版本及prepared SHA。环境/身份失败、程序异常或非零退出会停止，不能带错进入后段。按`START i/2700`顺序生成证据，质量失败仍保留并继续完成固定分母。run完成才进行inspect；`gate_passed=false`不阻止独立核验，不启动训练或其他作业。

需要只作前检（短任务）时：

```powershell
& 'D:\program\endtoend Parking\docs\long_tasks\LT-20261104-46-wheeled-full-source-coverage.ps1' -Phase Precheck
```

## 耗时与资源

- 粗估运行及核验合计**1.5～3小时**：修复后22例运行合计42.71秒，线性外推2700例约87分钟；22例独立审核约8秒，线性外推约16分钟。2700例难度与面板不同，复杂候选、长轨迹、机器负载会使估算明显偏离。
- 单例规划3秒、执行120秒是资源预算，2700例最坏预算量级超过90小时，不是进程硬截止保证。若明显超出估算、长时间无START/DONE或出现异常，保留日志反馈，不擅自加预算或开第二个作业。
- CPU单线程，无GPU；建议至少2GB可用内存和3GB磁盘空间。每例保存教师与执行状态/控制，最终目录可能达到数百MB至GB量级。
- 预计小时级，按用户当前分工由用户执行；Agent只完成预检和短面板验证。

## 输出、进度与成功标志

- 正式目录：`runs/verification/wheeled-coverage-formal-20261104-v1/`。应包含config.json、2700份逐例JSON、report.json及artifacts.json。
- 同级：`wheeled-coverage-formal-20261104-v1-run.log`、`-run-exit.json`、`-inspect.log`、`-inspect-exit.json`。退出记录task必须为LT-20261104-46。
- run日志逐例打印`START i/2700 split:index`及`DONE planned=... passed=...`，规划失败也必须占原分母；原索引不替换。
- 作业完整：run退出0、report.status=completed、expected_cases=2700及2700条rows；inspect退出0、verified=true、cases=2700。
- 科研全通过：还需passed=2700、gate_passed=true。若gate_passed=false但文件审核完整，则是有效覆盖负结果；先分析失败分布及原因，不导出训练集或开始训练。

## 异常、恢复与重跑

- Python traceback、身份漂移、磁盘不足或持续无进展时停止并反馈。Ctrl+C中断后保留全部目录/日志；不完整现场可能没有report/artifacts/退出码，不能视为完成。
- **本版没有断点续跑。** 不再次运行默认All，也不删除目录。由Agent诊断后准备新目录/新计划；源码修复必须重新冻结prepared，不能修改原SHA绕过。
- 如果run已完整退出0，尚未开始inspect（例如关机前主动使用了`-Phase Run`），可执行下列命令；它仍需冻结前检和本任务run退出0记录，且审核日志尚不存在。已有inspect日志即使中断也禁止覆盖，交Agent诊断。

```powershell
& 'D:\program\endtoend Parking\docs\long_tasks\LT-20261104-46-wheeled-full-source-coverage.ps1' -Phase Inspect
```

- 不可覆盖：原NPZ/manifest/test、旧训练权重/评测、静态审核、所有prepared、LT45 v1/v2/v3现场及任何已有LT46输出/日志。不并行启动重复作业。

## 完成后反馈与Agent验收

反馈正式输出目录路径和四个日志/退出记录；默认路径未改时说明LT46已完成即可。异常时说明最后START/DONE和报错，保留现场。

Agent核对任务身份、冻结文件/源分母、退出状态、产物SHA和逐控制重放，再按split×scene×maneuver统计教师规划失败、请求方向不足、执行超时/安全停止及精度缺口。全量未通过时留在环节4定位；即使全通过，也仅说明当前假设配置/GT静态条件下的专家覆盖，之后另行建立轮式训练数据schema与checkpoint身份，不以本报告替代主300、泛化、动态障碍或实物验证。
