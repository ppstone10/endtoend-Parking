# 架构

## 组件与所有权

| 模块 | 目录 | 职责 |
|---|---|---|
| 统一接口 | `interfaces/` | 定义三阶段共用的数据契约：传感器帧、BEV、车辆状态、目标位姿、轨迹、控制指令；坐标统一为车辆中心局部坐标系 |
| Sensor2BEV | `sensor2bev/` | 将 LiDAR 点云/Camera 图像转换为统一 BEV 表示：`lidar_bev.py`（ROI→降采样→地面滤除→栅格投影）、`camera_bev.py`（IPM 单应反投影目标区域）、`fusion.py`（通道级后融合） |
| Python 仿真 | `sim/` | 二维矿区泊车环境（`obstacles.py` 障碍体系：矩形/多边形/圆 + kind/emits_points/forbidden 语义，解析射线投射）、差分驱动车辆模型、模拟 LiDAR 与相机（`camera_model.py` 提供地面↔像素单应）、`vehicle_config.py` 车辆参数统一来源（矿卡 6×3m 预设，规划器/MPC/碰撞统一注入） |
| 端到端网络 | `model/` | MineParkingNet：BEV CNN 编码 + 目标/状态融合，输出未来 N 个局部轨迹点；`loss_fn` 为掩码 MSE |
| 轨迹控制器 | `controller/` | MPC 轨迹跟踪：CEM 交叉熵求解 + 差分驱动模型预测，输出 `[v_cmd, omega_cmd]` |
| 专家轨迹 | `planner/` | Hybrid A* 从起始状态到目标泊车位姿生成差分驱动可行轨迹（运动基元 + 栅格/yaw 离散 + C-space 膨胀裕度 + 拼接段加密校验） |
| 数据管线 | `dataset/` | 随机采样泊车位姿对，规划专家轨迹并复用传感器→BEV 链路生成训练样本（`SensorBEVPipeline`） |
| 闭环运行时 | `runtime/` | 滚动闭环引擎：`engine.py`（轨迹源→MPC→车辆循环、终止与失败分类）、`sources.py`（ExpertSource/NetworkSource 轨迹源策略）、`termination.py`（双阈值到达判定）、`recorder.py`（逐步记录供指标与回放） |
| 实验指标 | `metrics/` | `EpisodeResult`（单回合 8 项指标）与 `summarize`（多回合聚合：成功率/碰撞率/均值±标准差） |
| 可视化 | `viz/` | 统一风格（`style.py` 色表/PNG+PDF 双格式）、世界俯视渲染、轨迹三线叠加、单回合总图（动画与实验图后续里程碑） |
| 批量实验 | `experiments/` | 配置驱动 runner（JSON 配置 → 引擎批量回合 → 指标汇总 → 结果落盘），配置与结果归档 `configs/`、`results/` |
| 运行脚本 | `scripts/` | 阶段演示与数据流串联 |

## 数据流

```
LiDARFrame/CameraFrame → sensor2bev → BEVTensor
  LiDARFrame → LiDAR2BEV → [occupancy, height, density]
  CameraFrame → Camera2BEV → [target]
  BEVFusion 拼接两路并追加 [vehicle] → 统一 BEVTensor
VehicleState + GoalPose → HybridAStarPlanner → Trajectory（专家轨迹）
DatasetGenerator：采样位姿对 + SensorBEVPipeline(融合BEV) + 专家轨迹 → TrainingSample
BEVTensor + VehicleState + GoalPose → MineParkingNet → Trajectory
  （训练：全局专家轨迹/目标 → 起始局部系 → 掩码 MSE 训练 MineParkingNet）
Trajectory + VehicleState → MPCController → ControlCmd[v, omega] → 平台执行器
ClosedLoopEngine：TrajectorySource(Expert/Network) → MPC → 车辆模型滚动循环
  → 终止（到达双阈值/碰撞/超时/振荡）→ EpisodeResult（metrics/ 聚合）
```

- 障碍物碰撞与点云语义分离：`is_free`/`has_collision` 只检查 forbidden 障碍与地图边界；`raycast` 只与 emits_points 障碍求交（悬崖禁止进入但不挡射线，地面标线可通行）。
- Camera→BEV 与模拟相机共用同一套单应几何（`sim/camera_model.py` 与 `sensor2bev/camera_bev.py` 推导一致），保证渲染与反投影互逆。
- MPC 进度锚定轨迹时间轴（单调推进），参考轨迹 dt 与控制周期不一致时按时间插值对齐；求解器为 CEM（纯 numpy）。

## 统一接口（interfaces/）

- 传感器帧：`LiDARFrame`（N×4 `[x,y,z,intensity]` 点云）、`CameraFrame`（图像与标定参数）。
- 环境表示：`BEVTensor`（`[C,H,W]`，通道语义：障碍物占据、高度、点云密度、目标区域、车辆轮廓）。
- 运动状态：`VehicleState`（`[x,y,yaw,v,omega]`）。
- 目标位姿：`GoalPose`（`[x_goal,y_goal,yaw_goal]`）。
- 轨迹：`Trajectory`（未来 N 个局部轨迹点 `[x_i,y_i,yaw_i]`）。
- 控制指令：`ControlCmd`（`[v_cmd, omega_cmd]`）。

## 不变量

- 网络输入输出契约在三个阶段保持不变；迁移只替换传感器来源和底盘执行接口。
- 坐标统一使用车辆中心局部坐标系。
- `interfaces/` 不依赖任何仿真、网络或硬件实现，反向依赖。
- `sim/`、`sensor2bev/`、`model/`、`controller/` 只依赖 `interfaces/` 与 `numpy`，模块间不互相耦合。
- `runtime/` 依赖 `interfaces/`、`metrics/` 与注入的轨迹源/MPC/车辆模型（依赖注入，不直接 import sim/model）；`metrics/` 无内部依赖。
- 闭环执行统一经 `runtime/ClosedLoopEngine`，轨迹源策略可替换（Expert/Network/后续基线），指标口径唯一。
- 车辆尺寸与控制上限统一来自 `sim/vehicle_config.py::VehicleConfig`（预设矿卡 6×3m），规划器/MPC/车辆模型/碰撞检测由同一 config 构造注入。
- 批量实验统一经 `experiments/run_experiment.py`（JSON 配置驱动，结果落盘 `experiments/results/`），可视化统一经 `viz/`（PNG+PDF 双格式输出）。
- 低速泊车采用差分驱动运动模型，控制量为线速度 v 与角速度 omega。
- BEV 以车辆为中心生成，语义通道由 `BEVTensor` 的 `channels` 说明。

## 阶段迁移边界

- 阶段一→二：替换传感器模拟为 CARLA 传感器，其余模块不动。
- 阶段二→三：替换 CARLA 车辆为履带底盘（新增 CAN 执行适配），其余模块不动。