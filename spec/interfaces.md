# 统一接口 Spec

## 元数据

- Spec ID 前缀：`IFACE`
- 强度：完整
- 状态：已采纳
- 最后更新：2026-08-21

## 目标

- 定义三阶段（Python 仿真 / CARLA / 履带车实物）共用的数据契约，使迁移只替换传感器来源和底盘执行接口。
- 保证 BEV 通道语义、坐标约定、控制指令形态在三个阶段保持一致。

## 非目标

- 不规定端到端网络内部结构、MPC 算法细节或具体传感器型号。
- 不定义 CARLA 或 CAN 协议实现，只定义契约层。

## 边界与约束

- 坐标统一使用车辆中心局部坐标系；点云与 BEV 均以车辆为中心。
- 控制指令采用差分驱动形态 `[v, omega]`，线速度 v 米/秒、角速度 omega 弧度/秒。
- 接口类型只依赖 `numpy`，不依赖仿真、网络或硬件实现（反向依赖）。
- 对非法输入（形状错误、通道不匹配、尺寸不匹配）应抛出 `ValueError`。

## 行为与验收

### `IFACE-SENSOR-001`：LiDAR 点云帧

- 前置：点云为形状 `(N, 4)` 的数组，列为 `[x, y, z, intensity]`，单位米。
- 行为：构造 `LiDARFrame` 并校验形状。
- 结果：提供 `count` 属性返回点数。
- 异常与恢复：形状非 `(N, 4)` 时抛出 `ValueError`。

### `IFACE-BEV-001`：BEV 张量

- 前置：数据为 `(C, H, W)` 数组，提供 resolution 与 extent（米）及通道名列表。
- 行为：构造 `BEVTensor`，校验通道数、H/W 与 extent/resolution 推导值一致。
- 结果：提供 `shape`、`height`、`width` 属性。
- 异常与恢复：通道数不匹配或尺寸不一致时抛出 `ValueError`。
- 验收：默认通道顺序为 `occupancy`、`height`、`density`，由 `LiDAR2BEV` 保证。

### `IFACE-STATE-001`：车辆状态与目标位姿

- 前置：状态由 `[x, y, yaw, v, omega]` 描述。
- 行为：`VehicleState` 支持 `to_array`/`from_array` 往返转换。
- 结果：数组形状为 `(5,)`。
- 异常与恢复：数组元素不足 5 个时抛出 `ValueError`。

### `IFACE-TRAJ-001`：未来轨迹

- 前置：轨迹点形状为 `(N, 3)`，列为 `[x, y, yaw]`，含时间间隔 `dt`。
- 行为：构造 `Trajectory` 并提供 `horizon` 属性。
- 异常与恢复：形状非 `(N, 3)` 时抛出 `ValueError`。

### `IFACE-CTRL-001`：控制指令

- 前置：控制指令为 `[v, omega]`。
- 行为：`ControlCmd` 提供 `to_array` 输出 `(2,)` 数组。
- 结果：`v` 为线速度，`omega` 为角速度。

### `IFACE-BEV-002`：Camera→BEV 目标区域

- 前置：相机图像（单通道或三通道）、内参、相机位姿（高度、俯仰角）、BEV 分辨率与覆盖范围；目标区域须位于相机视野内。
- 行为：将地面目标反投影到车辆中心局部 BEV，生成 `target` 通道，高于灰度阈值判定为目标区域。
- 结果：输出单通道 `BEVTensor`，通道名为 `target`。
- 边界：仅覆盖相机可见区域；ROI 外或反投影出图像范围的位置置 0；视野外目标不得产生横跨图像的误填充。
- 验收：正前方 5m 处泊车位目标在 BEV 中落回 X≈5m、Y≈0 附近，Y 范围 ≤ ±1.5m。

### `IFACE-FUSION-001`：LiDAR/Camera BEV 融合

- 前置：两路 BEVTensor 分辨率与 extent 一致。
- 行为：按通道拼接 LiDAR（occupancy/height/density）与 Camera（target）BEV，并追加 `vehicle` 车辆轮廓通道。
- 结果：输出多通道 `BEVTensor`，通道顺序为 `occupancy, height, density, target, vehicle`。
- 异常与恢复：两路 BEV 分辨率或覆盖范围不一致时抛出 `ValueError`。
- 验收：融合后张量通道数 = LiDAR 通道数 + Camera 通道数 + 1。

## 数据、接口与迁移（完整档）

- 数据与状态：接口类型为不可变数据容器（dataclass），所有权归创建方；`BEVTensor` 通道语义由 `channels` 自描述。
- 接口与协议：`interfaces/` 暴露 `LiDARFrame`、`CameraFrame`、`CameraIntrinsics`、`BEVTensor`、`VehicleState`、`GoalPose`、`Trajectory`、`ControlCmd`。
- 迁移与回退：新增字段应保持向后兼容（dataclass 字段追加默认值）；废弃字段需先废弃后移除并记录。
- 安全与隐私：本契约不涉及授权或敏感数据；点云/图像为本地仿真数据。

## 追溯

| Spec ID | 验收 | 测试或人工入口 | 实现符号 | 实际验证 | 状态 |
|---|---|---|---|---|---|
| `IFACE-SENSOR-001` | 形状校验与 count | `tests/test_interfaces.py::TestLiDARFrame` | `interfaces/sensor.py::LiDARFrame` | unittest 通过 | ✅ |
| `IFACE-BEV-001` | 通道/尺寸校验 | `tests/test_interfaces.py::TestBEVTensor` | `interfaces/bev.py::BEVTensor` | unittest 通过 | ✅ |
| `IFACE-STATE-001` | 数组往返转换 | `tests/test_interfaces.py::TestVehicleState` | `interfaces/state.py::VehicleState` | unittest 通过 | ✅ |
| `IFACE-TRAJ-001` | horizon 与形状校验 | `tests/test_interfaces.py::TestTrajectory` | `interfaces/trajectory.py::Trajectory` | unittest 通过 | ✅ |
| `IFACE-CTRL-001` | to_array 输出 | `tests/test_interfaces.py::TestControlCmd` | `interfaces/control.py::ControlCmd` | unittest 通过 | ✅ |
| `IFACE-BEV-002` | 目标区域前向半区占据 | `tests/test_fusion.py::TestCamera2BEV` | `sensor2bev/camera_bev.py::Camera2BEV` | unittest 通过 | ✅ |
| `IFACE-FUSION-001` | 通道顺序与数量、异常校验 | `tests/test_fusion.py::TestBEVFusion` | `sensor2bev/fusion.py::BEVFusion` | unittest 通过 | ✅ |

## 待人工确认

- 无。