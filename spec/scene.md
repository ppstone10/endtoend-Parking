# 场景库 Spec

## 元数据

- Spec ID 前缀：`SCENE`
- 强度：轻量
- 状态：已采纳
- 最后更新：2026-08-24

## 目标

- 提供覆盖真实矿场六类泊车行为的 9 个参数化场景（REQUIREMENTS §3），供任务层、数据管线与实验矩阵消费。
- 车位容差（tol_pos/tol_yaw）即闭环到达判定阈值，场景即精度标准的载体。

## 非目标

- 不做任务采样与难度分层（`sim/tasks.py`，T 层）；不做动态障碍。
- 不做场景 YAML 落盘（P2.2 收尾时按需扩展，当前以构造参数即配置）。

## 边界与约束

- 场景构造器经 `register_scene` 注册进 `SCENE_REGISTRY`，`build_scene(name, **kwargs)` 构建实例。
- `SceneBundle` 携带 env（含语义障碍）、spots（含占用/容差/编号）、spawn_zones（任务层起点采样区）、difficulty_knobs（难度旋钮）。
- 车位占用由场景参数指定，占用车辆以 kind=vehicle 矩形障碍呈现。
- 障碍语义使用 P2.1 体系：悬崖禁入不挡射线、标线可通行、挡墙挡射线。

## 行为与验收

### `SCENE-REG-001`：场景注册与构建

- 前置：S1–S9 九个场景已注册。
- 行为：`build_scene(name, **kwargs)` 按名称与难度旋钮构建实例；未知名称抛 `ValueError`。
- 结果：九场景全部可构建，难度旋钮（车位数/通道宽/占用/岩石数）生效。
- 验收：`tests/test_scenes.py::TestSceneRegistry` 通过。

### `SCENE-VALID-001`：场景自检

- 前置：任意注册场景（默认参数）。
- 行为：`validate_scene` 校验障碍在图内、空闲车位目标位姿无碰撞、起点采样区存在自由位姿。
- 结果：九场景错误列表为空。
- 验收：`tests/test_scenes.py::TestSceneValidation` 通过（自检曾实际捕获 S5/S9 三处车位-禁区几何冲突并修复）。

### `SCENE-SEM-001`：场景语义正确性

- 前置：S4/S5/S8/S9 关键语义。
- 行为：S4 射线打挡墙而非悬崖、悬崖禁入；S5 槽内自由/槽底料口禁区；S8 称重台标线可通行；S9 含三种功能区车位。
- 结果：语义断言全部成立。
- 验收：`tests/test_scenes.py::TestSceneSemantics` 通过。

### `SCENE-VIZ-001`：V1 场景地图渲染

- 前置：全部注册场景。
- 行为：`scripts/render_scenes.py` 渲染 3×3 总览 + 九单图（障碍按 kind 样式：悬崖斜纹、挡墙深色、岩石圆斑、车位编号框），PNG+PDF 双格式。
- 结果：20 个文件落盘 `out/scenes/`，程序化校验（尺寸/内容密度/色彩数）全部通过。
- 验收：2026-08-24 实际渲染 20 文件，校验 10/10 OK。

## 追溯

| Spec ID | 验收 | 测试或人工入口 | 实现符号 | 实际验证 | 状态 |
|---|---|---|---|---|---|
| `SCENE-REG-001` | 九场景注册构建 | `tests/test_scenes.py::TestSceneRegistry` | `sim/scenes.py::SCENE_REGISTRY/build_scene` | unittest 通过 | ✅ |
| `SCENE-VALID-001` | 自检零错误 | `tests/test_scenes.py::TestSceneValidation` | `sim/scenes_validate.py::validate_scene` | unittest 通过（捕获并修复 3 处几何冲突） | ✅ |
| `SCENE-SEM-001` | 语义断言 | `tests/test_scenes.py::TestSceneSemantics` | 场景构造器 | unittest 通过 | ✅ |
| `SCENE-VIZ-001` | V1 图渲染 | `scripts/render_scenes.py` | `viz/world_render.py` | 20 文件、校验 10/10 OK | ✅ |

## 待人工确认

- 场景几何与真实矿场布局的吻合度建议结合文献/现场照片人工复核（S1–S9 规格源自 REQUIREMENTS §3）。
