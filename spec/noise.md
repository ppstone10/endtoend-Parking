# 传感器噪声 Spec

## 元数据

- Spec ID 前缀：`NOISE`
- 强度：完整
- 状态：已采纳
- 最后更新：2026-08-24

## 目标

- 为模拟 LiDAR 与相机提供干净/低/高三档可配置、seed 可复现的观测噪声。
- 让任务元数据中的噪声等级能够映射为真实传感器行为，并为后续鲁棒性实验提供可自定义参数入口。

## 非目标

- 不拟合或声称代表某一真实矿卡传感器的标定分布。
- 不改变环境、障碍物或车位真值，不在本层生成 BEV/数据集元数据。
- 不实现时延、滚动快门、镜头畸变、运动模糊或跨帧相关噪声。

## 边界与约束

- `clean` 为默认档，必须保持修改前传感器输出和调用签名兼容。
- `low`/`high` 参数单调增强；概率位于 `[0,1]`，标准差非负，非法 profile 构造时拒绝。
- 每个传感器实例拥有独立 RNG；同一类型、profile、seed 与调用序列产生相同输出，不读写 NumPy 全局随机状态。
- 噪声只作用于 `capture()` 返回值；`ParkingEnvironment` 及其 `raycast()` 结果不被修改。

## 行为与验收

### `NOISE-PROFILE-001`：三档与自定义配置

- 前置：请求 `clean`、`low`、`high` 或合法自定义 `NoiseProfile`。
- 行为：解析出 LiDAR 和 Camera 的完整噪声参数，并可通过 `to_metadata()` 导出 JSON 基础类型。
- 结果：三档参数逐档不减；未知等级或非法概率/标准差抛 `ValueError`。
- 异常与恢复：构造失败不创建传感器；改回 `clean` 即恢复原始行为。
- 验收：`tests/test_noise.py::TestNoiseProfiles` 通过。

### `NOISE-LIDAR-001`：LiDAR 距离、丢点与量程噪声

- 前置：环境、波束数、量程及非干净 profile。
- 行为：每帧先抖动有效量程，再对保留波束沿射线方向施加高斯距离扰动，并按独立概率删除波束。
- 结果：输出仍为合法 `(N,4)` `LiDARFrame`，`0≤N≤beams`；clean 输出保持每束一点，非干净档产生受控扰动/丢点。
- 异常与恢复：抖动后的有效量程下限保持为正；单帧全丢点时返回 `(0,4)` 空点云而非失败。
- 验收：`tests/test_noise.py::TestLiDARNoise` 通过。

### `NOISE-CAMERA-001`：Camera 像素与目标检测噪声

- 前置：灰度模拟图像和非干净 profile。
- 行为：目标按漏检概率移除、按误检概率增加一个图像内伪目标块，最后施加零均值像素高斯噪声并裁剪至 uint8。
- 结果：输出尺寸/内参不变；漏检率 1 时真实目标消失，误检率 1 时无目标环境仍出现伪目标。
- 异常与恢复：噪声不改变输入环境；clean 输出与修改前逐像素一致。
- 验收：`tests/test_noise.py::TestCameraNoise` 通过。

### `NOISE-SEED-001`：随机复现与隔离

- 前置：两个参数相同且 seed 相同的传感器实例。
- 行为：按相同 capture 调用序列生成观测。
- 结果：对应帧逐元素相同；不同 seed 的非干净观测至少一项不同；创建/调用传感器不改变 NumPy 全局随机序列。
- 异常与恢复：重新以原 seed 构造实例即可从序列起点复现。
- 验收：`tests/test_noise.py::TestNoiseReproducibility` 通过。

## 数据、接口与迁移

- 数据与状态：`NoiseProfile` 为 frozen dataclass，包含 `LiDARNoiseConfig` 与 `CameraNoiseConfig`；传感器私有 `Generator` 持有运行期随机状态。
- 接口与协议：`SimulatedLiDAR`/`SimulatedCamera` 构造器尾部新增关键字参数 `noise` 与 `seed`；既有位置参数及 `capture()` 返回契约不变。
- 迁移与回退：无需数据迁移；默认 `noise=clean`，删除新增关键字即可回退既有调用。后续 P2.5/P2.8 可直接保存 `NoiseProfile.to_metadata()`。
- 安全与隐私：只生成合成随机观测，不处理个人数据、不访问外部系统；随机数不用于密码学或安全决策。

## 追溯

| Spec ID | 验收 | 测试或人工入口 | 实现符号 | 实际验证 | 状态 |
|---|---|---|---|---|---|
| `NOISE-PROFILE-001` | 三档、校验、元数据 | `tests/test_noise.py::TestNoiseProfiles` | `sim/noise.py::NoiseProfile/NOISE_PROFILES` | unittest 通过 | ✅ |
| `NOISE-LIDAR-001` | 距离/丢点/量程抖动 | `tests/test_noise.py::TestLiDARNoise` | `sim/sensor_sim.py::SimulatedLiDAR` | unittest 通过 | ✅ |
| `NOISE-CAMERA-001` | 像素/漏检/误检 | `tests/test_noise.py::TestCameraNoise` | `sim/sensor_camera.py::SimulatedCamera` | unittest 通过 | ✅ |
| `NOISE-SEED-001` | seed 复现与全局 RNG 隔离 | `tests/test_noise.py::TestNoiseReproducibility` | 两个传感器的私有 `Generator` | unittest 通过 | ✅ |

## 待人工确认

- low/high 数值属于实验档位；获得真实传感器日志后应以同一 profile 接口替换参数并新增分布拟合验收。
