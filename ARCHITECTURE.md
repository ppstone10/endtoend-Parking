# 架构

## 组件与所有权

| 模块 | 目录 | 职责 |
|---|---|---|
| 统一接口 | `interfaces/` | 定义三阶段共用的数据契约：传感器帧、BEV、车辆状态、目标位姿、轨迹、控制指令；坐标统一为车辆中心局部坐标系 |
| Sensor2BEV | `sensor2bev/` | 将 LiDAR 点云/Camera 图像转换为统一 BEV 表示 |
| Python 仿真 | `sim/` | 二维矿区泊车环境、车辆运动模型、模拟传感器 |
| 端到端网络 | `model/` | MineParkingNet：输入 BEV+目标位姿+运动状态，输出未来 N 个局部轨迹点 |
| 轨迹控制器 | `controller/` | MPC 轨迹跟踪，输出 `[v_cmd, omega_cmd]` |
| 专家轨迹 | `planner/` | Hybrid A* 等生成专家轨迹，用于训练数据 |
| 数据管线 | `dataset/` | 训练数据生成与加载 |
| 运行脚本 | `scripts/` | 阶段演示与数据流串联 |

## 数据流

```
LiDARFrame/CameraFrame → sensor2bev → BEVTensor
BEVTensor + VehicleState + GoalPose → MineParkingNet → Trajectory
Trajectory + VehicleState → MPCController → ControlCmd[v, omega] → 平台执行器
```

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
- 低速泊车采用差分驱动运动模型，控制量为线速度 v 与角速度 omega。
- BEV 以车辆为中心生成，语义通道由 `BEVTensor` 的 `channels` 说明。

## 阶段迁移边界

- 阶段一→二：替换传感器模拟为 CARLA 传感器，其余模块不动。
- 阶段二→三：替换 CARLA 车辆为履带底盘（新增 CAN 执行适配），其余模块不动。