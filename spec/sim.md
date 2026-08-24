# 仿真障碍物体系 Spec

## 元数据

- Spec ID 前缀：`SIM`
- 强度：轻量
- 状态：已采纳
- 最后更新：2026-08-24

## 目标

- 为场景库提供形状（矩形/多边形/圆）与语义（kind/emits_points/forbidden）统一的障碍物体系。
- 碰撞检测与点云射线投射语义分离，支撑"看得见的挡墙 + 看不见的崖"等矿区语义。

## 非目标

- 不做 3D 形状与高度场（2D 平面语义）。
- 不做动态障碍物（T5 注入由闭环引擎/任务层负责）。

## 边界与约束

- 坐标为全局世界坐标（米）；障碍物为 frozen dataclass。
- `RectangleObstacle` 保留原四字段签名（kind/emits_points/forbidden 带默认值），既有构造代码零改动。
- `PolygonObstacle` 支持简单多边形（凸或凹）；顶点归一化为 tuple 存储。
- 地图边界：出界即碰撞，且射线在出界处截断（与旧步进行为一致）。

## 行为与验收

### `SIM-OBS-001`：解析射线投射

- 前置：环境含矩形/圆/多边形障碍，起点在自由空间。
- 行为：`ParkingEnvironment.raycast` 以解析求交返回射线首次进入 emits_points 障碍或出界的距离。
- 结果：与旧步进实现（step=0.05）偏差 ≤ 0.05m（量化误差上界）；无命中返回 max_range。
- 验收：`tests/test_obstacles.py::TestRaycastRegression` 通过；LiDAR 一帧（360 束）耗时 56ms→1.2ms。

### `SIM-SEM-001`：碰撞/点云语义分离

- 前置：障碍物语义属性组合。
- 行为：`is_free`/`has_collision` 只检查 `forbidden=True` 障碍与地图边界；`raycast` 只与 `emits_points=True` 障碍求交。
- 结果：悬崖（kind=cliff, emits_points=False, forbidden=True）内部 is_free=False 但射线穿过；地面标线（kind=line, forbidden=False）可通行且不挡射线。
- 验收：`tests/test_obstacles.py::TestCliffSemantics` 通过。

### `SIM-SHAPE-001`：形状几何契约

- 前置：三种形状障碍。
- 行为：`contains_point` 边界含入；`ray_entry_distance` 单位方向射线首次进入距离（无交/后方返回 None，起点在内部返回 0）。
- 结果：矩形 slab 法、多边形边求交（含凹形）、圆二次求交几何正确。
- 验收：`tests/test_obstacles.py` 形状测试（矩形/圆/多边形各一组）通过。

## 追溯

| Spec ID | 验收 | 测试或人工入口 | 实现符号 | 实际验证 | 状态 |
|---|---|---|---|---|---|
| `SIM-OBS-001` | 新旧一致性 ≤0.05m | `tests/test_obstacles.py::TestRaycastRegression` | `sim/environment.py::ParkingEnvironment.raycast` | unittest 通过；45x 加速 | ✅ |
| `SIM-SEM-001` | 悬崖/标线语义 | `tests/test_obstacles.py::TestCliffSemantics` | `sim/obstacles.py` 语义字段 + `environment.py` | unittest 通过 | ✅ |
| `SIM-SHAPE-001` | 形状几何 | `tests/test_obstacles.py` 形状组 | `sim/obstacles.py` 各类 | unittest 通过 | ✅ |

## 待人工确认

- 无。
