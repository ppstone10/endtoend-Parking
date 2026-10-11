# 验收台账（Python 仿真阶段）

> 本台账是 Paper 阶段验收的**唯一当前事实来源**：逐条列出 `docs/training_guide.md`「正式验收」
> 与 `docs/REQUIREMENTS.md` C1–C6 / M1–M4 的判据、实测值、以及**测量所依据的协议**。
> 历史收尾报告 [closed_loop_final_report.md](closed_loop_final_report.md) 的 §0 结论已被本台账取代。
>
> 最后更新：2026-10-10。变更请追加，不要覆盖旧值（旧值带协议标注后保留作对照）。

---

## 0. 协议口径（一切数字的前提）

| 协议 | 数据 | 可复原 | 场景覆盖 | 用途 |
|---|---|---|---|---|
| **v34（当前验收口径）** | `tracked_pivot_v8_3000/val.npz` | **300 / 300** | **7 场景**：S1 55 / S2 55 / S3 52 / S4 50 / S5 33 / S6 44 / S8 11 | 验收与论文主表 |
| v34-test | `tracked_pivot_v8_3000/test.npz` | 300 / 300 | S9 综合矿场 300（训练未见布局） | 最终验收，**只跑一次** |
| v13（历史，仅对照） | `tracked_pivot_v7_3000/val.npz` + v13 索引 | 136 / 204 | 4 场景：S1 48 / S2 45 / S4 28 / S6 15 | 历史横向对照，不再作验收依据 |

统一参数：`seed=0`、`replan_every=10`、`max_steps=600`、`control_seed=0`、MPC `dt=0.1/horizon=10`、
车辆 `tracked_drill_rig` 6×3m、评测碰撞用真实接触（`collision_margin=0`）。

**泄漏核验（本轮）**：v34 val、v34 test 与 `v25_dagger_s46/train-balanced.npz`、
`tracked_pivot_v8_3000/train.npz` 的**任务身份重叠均为 0**；`slice-4scenes/val-4scenes.npz`
正好是 v34 val 的 4 场景子集（204 条），即模型此前的 val 是本协议的严格子集。

**任务可解性对照（本轮新增，防"任务本身不可解"）**：

| 对照 | 成功 | 碰撞 | 说明 |
|---|---|---|---|
| E1 专家规划一次 + MPC | **99.7%** | 0.3% | S1–S6 全 100%，S8 90.9% |
| E1r 专家滚动重规划 + MPC | **83.7%** | 0.3% | S3 仅 **30.8%**、S8 45.5%（超时为主） |

→ 任务全部可解；S3/S8 的低分**不是任务不可解**，而是滚动重规划本身在紧场景吃力。

复现：

```powershell
& 'D:\conda\envs\endtoend-parking\python.exe' scripts/run_validation_suite.py `
    --freeze-protocol --data data/task_dataset/tracked_pivot_v8_3000/val.npz `
    --model runs/training/v26-targeted-dagger/net-v1/deployment.pt `
    --output runs/validation/v34-protocol
& 'D:\conda\envs\endtoend-parking\python.exe' scripts/run_validation_suite.py `
    --experiment E1 --experiment E1r --experiment E3 --experiment E5 `
    --experiment E3f --experiment E5f --experiment E5s `
    --indices-file runs/validation/v34-protocol/protocol.json `
    --model runs/training/v26-targeted-dagger/net-v1/deployment.pt `
    --output runs/validation/v34-protocol/<子目录>
```

---

## 1. 开环验收：**全部达标**

模型 `v26-targeted-dagger/net-v1/deployment.pt`（`scripts/analyze_predictions.py`，
口径为推理时自由反馈 + 部署停止阈值）。

| 判据 | 准入线 | val（300 条 / 7 场景） | test（S9 300 条） | 判定 |
|---|---|---|---|---|
| ADE | val ≤0.40m / test ≤0.50m | **0.301m** | **0.216m** | ✅ |
| FDE | val ≤0.80m / test ≤0.90m | **0.392m** | **0.189m** | ✅ |
| 长度 MAE | ≤30 点 | **8.12** | **4.98** | ✅ |
| 长度绝对偏差 | ≤20 点 | **−2.43** | **+2.69** | ✅ |
| 停止命中率 | ≥95% | **100%** | **100%** | ✅ |
| test 不显著差于 val | — | — | **test 三项均优于 val** | ✅ |

产物：`runs/validation/v34-openloop/val-v26/report.json`、`runs/validation/v34-openloop/test-v26/report.json`。

**按场景（val）**：S5 0.161 / S8 0.200 / S3 0.211 / S1 0.220 / S2 0.241 / S6 0.419 / **S4 0.506**。
→ 开环层面**未见场景（S3/S5/S8）反而最好**，误差集中在 S4/S6。

---

## 2. 闭环验收：**未达标**

### 2.1 当前口径（v34，300 条 / 7 场景）

| 口径 | 模型 | 成功率（≥70%） | 碰撞率（≤10%） | 失败分布 | 判定 |
|---|---|---|---|---|---|
| E3 裸网络 | v26 | **45.0%** | **52.3%** | collision 157 / osc 8 | ❌ |
| E5 裸网络 | v26 | **41.3%** | **54.7%** | collision 164 / osc 12 | ❌ |
| E3f +几何过滤 | v26 | 52.7% | 31.7% | collision 95 / osc 47 | ❌ |
| E5f +几何过滤 | v26 | 45.7% | 30.3% | collision 91 / osc 72 | ❌ |
| E5s 安全门禁 | v26 | **78.3%** | **0.0%** | safety_stop 49 / osc 16 | ⚠️ 成功率✅ 碰撞✅，但**回退失败 2 ❌** |
| E3 裸网络 | v22（全 7 场景训练） | 27.3% | 39.0% | osc 101 / collision 117 | ❌ |
| E5 裸网络 | v22（全 7 场景训练） | 26.7% | 40.0% | osc 100 / collision 120 | ❌ |

门禁统计：干预率 33.9%，`prevented_transitions` 394，**`fallback_failures = 2`**（准入要求 =0），
`safety_stops = 49`。

产物：`runs/validation/v34-protocol/{net,net-v22,filter,gate,expert-e1,expert-e1r}/`。

### 2.2 历史口径（v13，136 条 / 4 场景）——仅作对照，**不再作验收依据**

| 口径 | 成功率 | 碰撞率 | 说明 |
|---|---|---|---|
| E3 裸网络 v26 | 69.9% | 26.5% | 缺 S3/S5/S8，样本为 v7 数据子集 |
| E5 裸网络 v26 | 72.1% | 23.5% | 同上 |
| E3f/E5f 几何过滤 | 76.5% / 73.5% | 5.9% / 4.4% | 看起来"达标"，实为受限口径产物 |
| E5s 门禁 | 92.6% | 0.0% | 回退失败 0（在 4 场景上） |

> **口径修正是本轮最重要的结论**：旧数字里的"纯网络已达标"只在 4 个场景、且是 v7 数据子集上成立。
> 换成身份合法、覆盖完整的 7 场景协议后，裸网络成功率 41–45%、碰撞率 52–55%。

### 2.3 test（S9）闭环：**未执行**

按 `training_guide.md`「先看 val，再只执行一次 test」，val 未达标前**不跑**——跑了就烧掉了唯一的
一次性验收机会。开环 test 已跑（§1，达标），闭环 test 待 val 达标后执行。

---

## 3. 分场景明细（成功率，v34）

| 场景 | n | E1 专家 | E1r 专家滚动 | E3 v26 裸网络 | E3f 几何过滤 | E5s 门禁 | E3 v22 全场景 |
|---|---:|---:|---:|---:|---:|---:|---:|
| S1 驻地停车 | 55 | 100% | 100% | 81.8% | 87.3% | 92.7% | 47.3% |
| S2 斜列停车 | 55 | 100% | 92.7% | 94.5% | 94.5% | 92.7% | 43.6% |
| **S3 维修紧 bay** | 52 | 100% | **30.8%** | **0.0%** | **0.0%** | 36.5% | 3.8% |
| S4 排土场卸载 | 50 | 100% | 94.0% | 42.0% | 36.0% | **98.0%** | 22.0% |
| **S5 破碎站窄槽** | 33 | 100% | 100% | **0.0%** | **0.0%** | 100% | 54.5% |
| S6 装载对位 | 44 | 100% | 100% | 38.6% | **90.9%** | 61.4% | 2.3% |
| **S8 称重站** | 11 | 90.9% | **45.5%** | **0.0%** | **0.0%** | 45.5% | 0.0% |

碰撞数（v34）：裸网络 v26 在 S3/S5/S8 上分别是 **52/52、33/33、11/11 全部碰撞**（96 条占 32%）；
几何过滤把它们降到 44 / 28 / 11，但没有一条转为成功——**过滤器只能防撞，不能创造能力**。

---

## 4. 未达标清单（按影响排序）

| # | 未达标项 | 判据 | 实测 | 性质 |
|---|---|---|---|---|
| 1 | **闭环成功率（裸网络）** | ≥70% | E3 45.0% / E5 41.3% | 能力缺口：S3/S5/S8 零样本 |
| 2 | **闭环碰撞率（裸网络）** | ≤10% | 52.3% / 54.7% | 同上 |
| 3 | **闭环成功率/碰撞率（+几何过滤）** | ≥70% / ≤10% | 52.7%/31.7%、45.7%/30.3% | 过滤只能防撞 |
| 4 | **门禁回退失败** | =0 | **2** | 专家回退在紧场景无解 |
| 5 | 门禁成功率 | ≥70% | 78.3%（✅）但 S3 36.5% / S8 45.5% | 受专家滚动重规划上限约束（E1r S3 30.8%） |
| 6 | test 闭环验收 | 一次 | 未执行 | 刻意推迟（val 未达标） |
| 7 | C4 可视化 V2/V4/V5 | 全套 | 不存在 | 工程量 |
| 8 | C6 对比基线组 / ≥3 seed | — | 仅 E1/E4 上界；1 个模型 3 seed | 工程量（M4 P4.2） |
| 9 | M4 P4.1 多车位选优 | FR-SELECT-01..04 | 完全空白 | 方法贡献点 |
| 10 | 主实验网格 FR-METRIC-03 | 8×50×5×3 | 未跑 | 工程量 |
| 11 | 工程债 | — | 4 项既有失败 + 1 项错误；`PLAN.md` 勾选落后于实际 | 收尾 |

---

## 5. 根因（本轮固证，三条互不替代）

1. **训练场景覆盖不足**：v26 的训练集 `v25_dagger_s46/train-balanced.npz` 只含
   S1 467 / S2 438 / S4 1105 / S6 686，**S3/S5/S8 一个样本都没有** → 这三场景闭环成功率 0%、碰撞率 100%。
2. **只补场景覆盖不够**：v22 用全 7 场景 `tracked_pivot_v8_3000/train.npz` 训练，
   学会 S5（0%→54.5%），但丢掉 S1/S2/S4/S6（S6 38.6%→2.3%、S2 94.5%→43.6%），
   S3/S8 仍近 0。→ 需要"全场景数据 + 已积累的配方（终点位姿损失 + 配比平衡 DAgger）"两者叠加。
3. **专家滚动重规划本身在紧场景吃力**：E1r 在 S3 只有 30.8%、S8 45.5%（超时为主），
   与门禁在这两处的上限（36.5% / 45.5%）一致 → **门禁天花板受专家回退质量限制**，
   不是网络安全缺口一个变量能解释的。

---

## 6. 下一步（重排后）

**P0-A 全场景重训（唯一能同时修 S3/S5/S8 的路）**
- 数据：`tracked_pivot_v8_3000/train.npz`（7 场景 2400 条），val 用 `.../val.npz`（300 条）。
- 配方：`v19` 终点位姿损失（`terminal_pose_weight=1.0` / `terminal_yaw_weight=10.0`）
  + `v22` 的 `endpoint_alignment_weight=0.5`
  + 配比平衡 DAgger（对 **S3/S5/S8 定向补采**，按 `--scenes` 与
  `--max-recoveries-per-maneuver` 控制配比，合并端用 `balance_recovery_dataset.py` 裁剪）。
- 闭环选型：`closed_loop_selection.data` 指向新协议，`safety_mode: none`。

**P0-B 若重训后 S3/S8 仍无解 → 场景/规划专项**
- 用本台账 §0 的 E1/E1r 对照区分"任务不可解"与"滚动重规划不可行"：
  目前证据指向后者（E1 100%，E1r 30.8%）。
- 方向：S3 紧 bay 的起点轴线对齐、S8 称重台净空审查（见 `LEARNING.md` 两条既有经验）。

**P1 门禁回退失败归零**：定位那 2 次 `fallback_failures` 的具体场景与状态。
**P2 val 达标后跑一次 test（S9）**，开环已达标，只需补闭环。
**P3 M4 实验矩阵**：基线组（FR-BASE-01/02）、主网格（FR-METRIC-02/03/06）、
鲁棒性与泛化（FR-METRIC-04/05）、可视化（V2/V4/V5）、多车位选优（FR-SELECT）。

---

## 7. 已排除，不要再投入

- **训练侧损失侧改动压碰撞**：净空损失三种形式（v18/v20/v27/v28）全部无效或有害；
  根因是预测误差 0.32m > S4/S6 可操作间隙 0.23–0.30m，与损失形状无关。
- **把几何过滤当作达标的替代品**：过滤能把碰撞压到 30%（4 场景下达标），
  但在零样本场景上只能把"撞"换成"停"，成功率不动。
- **在受限口径上调参**：旧 136 条口径给出的"达标"结论经不起协议修正，不能作为论文依据。

---

## 附：本轮新增的工具与修复

| 项 | 说明 |
|---|---|
| `scripts/diagnose_collision_mechanism.py` | 碰撞机制诊断（参考穿障 vs 执行偏离），可加装几何过滤并落盘不可行参考 |
| `runtime/trajectory_repair.py` + `GeometricFilterSource` | 推理侧轨迹级几何过滤（`LOOP-FILTER-001`） |
| `runs/diagnostics/*.py` | 本轮分析脚本（数据集覆盖、泄漏核验、协议对照、开环指标汇总），随 `runs/` 忽略，不入库 |
| `runtime/sources.py::_plan_or_infeasible`（修复） | 规划器"无可行轨迹"的 `RuntimeError` 归一为 `ValueError`，把"某任务专家无解"从**整批中止**降级为**回合内归因**；`ExpertSource` 改为首次取轨迹时才规划（`begin` 不在引擎归因保护区内）。回归测试见 `tests/test_engine_robustness.py::TestExpertPlannerInfeasibility` |
