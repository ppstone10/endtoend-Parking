# 轮式车辆基础模型 Spec

## 元数据

- Spec ID 前缀：`WHEEL-BASE`
- 强度：完整
- 状态：已采纳
- 最后更新：2026-11-04

## 目标

- 为轮式矿卡研究提供与旧履带模型隔离的、可解析验证的低速汽车式运动基础。
- 显式保留转角状态、中心与后轴参考点和可复现配置身份，为后续轮式MPC/规划提供相同契约。

## 非目标

- 本环节不接入MPC、规划器、运行时、数据生成、训练、硬件或默认部署；不认证碰撞安全、泊车成功率或真实车型参数。
- 不模拟轮胎侧偏、滑移、质量惯量、执行器连续动态、刹车距离或真实换挡延迟。

## 边界与约束

- 新接口从`interfaces.wheeled`直接导入；旧接口、包根导出及旧配置JSON语义不变。
- 位姿(x,y,yaw)为世界坐标下车体几何中心；v为后轴纵向有符号速度（也等于中心速度在车身纵轴的投影），不是转弯时中心速度模长。
- 转角和角速度均为弧度单位，左转为正，yaw连续不归一化。轮式状态保存steering_angle，不用停车时的omega倒推转角。
- 后轴位于中心沿车身x轴的`rear_overhang-length/2`；恒速恒转角时按后轴单轨公式精确积分，再变换回中心。
- 执行为离散`sampled_slew_zoh`：请求值先按状态与dt限速、限转角变化，所得v/delta在整个步内保持。变化率仅是采样间差分约束，不宣称连续执行器加速度。

## 行为与验收

### `WHEEL-BASE-001`：严格配置与身份

- 配置记录尺寸、轴距、前后悬、正反速度、转角/转角变化上限、加减速、碰撞余量及参数来源说明。长度=轴距+前悬+后悬（浮点容差1e-9）；尺寸/轴距/上限为有限正数，前后悬/余量有限非负，最大转角严格小于pi/2；派生半径等必须有限。
- 元数据声明schema_version=1、motion_model=kinematic_bicycle、pose_reference=body_center、speed_reference=rear_axle_longitudinal和actuation_model=sampled_slew_zoh。字段缺失、未知、类型错误（包括布尔冒充数字）、不支持声明、旧履带元数据均拒绝。
- 全部字段规范化后计算SHA256身份；完整往返保持身份，任一几何/约束/模型声明变化会改变身份或被拒绝。示例参数必须明确为研究假设。

### `WHEEL-BASE-002`：独立指令与可恢复状态

- 指令含v、steering_angle；状态含x/y/yaw/v/steering_angle，均为有限数，显式数组转换按此顺序，长度严格为5。不把相同长度的旧数组隐式当轮式状态，旧对象也不被step接受。
- 模型无隐含执行状态；复制返回状态后由新模型实例续行，应与原实例相同。状态超出车型速度/转角限制拒绝，不能静默夹紧观测状态；非法dt/对象/非有限输入拒绝且原状态不变。

### `WHEEL-BASE-003`：中心与后轴参考点

- 公开中心→后轴、后轴→中心位姿转换；非对称前后悬、任意有限yaw及正负偏置均往返一致。
- 圆弧验收同时核对后轴曲率和偏置中心轨迹，不能把后轴公式直接用于中心。

### `WHEEL-BASE-004`：速度与转向离散限制

- 目标速度与转角先按配置幅值限幅，转角每步变化不超过max_steering_rate*dt；增速按max_acceleration、减小速度绝对值按max_deceleration。
- 请求反向且当前v非零时，本步目标先设为0，不跨零；下一步可开始反向。无符号阈值捷径，不在同一控制步用反向位移消除停车阶段。
- 停车可改变转角但不得改变车体位姿；请求v=0时若当前仍在运动，则按减速约定继续移动，不能宣称瞬时停住。

### `WHEEL-BASE-005`：汽车式运动推进

- 用本步实际v与delta计算omega=v*tan(delta)/wheelbase；v=0时omega=0，不允许独立原地yaw变化。
- 恒定命令按零阶保持后的后轴精确圆弧积分（直线极限数值稳定）；支持前进/倒车与左右转、连续yaw以及多步组合。输出有限且不修改输入。
- 可查询实际状态的yaw_rate与中心世界速度；中心速度包含偏置产生的侧向分量，不能将其模长混用为v。

### `WHEEL-BASE-006`：旧模型隔离与分阶段使用

- 旧履带配置/元数据、别名、导出及差速行为保持；旧ControlCmd/VehicleState传入轮式step明确报错。
- 新模型不自动进入get_vehicle或旧闭环入口；后续接入须独立规范和验收，失败不得回退旧履带模式。

## 数据、接口与迁移

- `WheeledVehicleConfig`与专用JSON加载器拥有新配置，不给旧VehicleConfig补默认字段，避免改变历史元数据和冻结身份。
- `WheeledControlCmd`、`WheeledVehicleState`是显式新类型；数组数值本身不携带模型身份，只用于已确认上下文下的恢复，持久化必须同时携带配置元数据。
- `KinematicBicycleModel.step(state, command, dt)`返回新的轮式状态。后续MPC必须使用相同推进/限制语义，不自行复制另一套模型。
- 本环节无旧产物迁移、覆写或硬件输出；回退仅移除新模块及其显式调用，不修改旧归档。不得因兼容旧模型而静默接受缺失的轮式身份。
- 无外部服务或个人数据。模型限幅是理论执行规则，不等于实车安全认证。

## 追溯

| Spec ID | 验收 | 测试或人工入口 | 实现符号 | 实际验证 | 状态 |
|---|---|---|---|---|---|
| WHEEL-BASE-001 | 配置/身份严格往返及错配拒绝 | tests.test_wheeled_vehicle.TestWheeledConfig | sim/wheeled_config.py::WheeledVehicleConfig/load_wheeled_vehicle_config | 7项配置测试通过 | ✅ |
| WHEEL-BASE-002 | 独立类型、恢复、异常原子性 | tests.test_wheeled_vehicle.TestWheeledModel | interfaces/wheeled.py；KinematicBicycleModel.step | 轮式25项套件通过，含状态恢复/极值/类型拒绝 | ✅ |
| WHEEL-BASE-003 | 参考点往返及偏置圆弧 | tests.test_wheeled_vehicle.TestWheeledModel | KinematicBicycleModel.center_to_rear_pose/rear_to_center_pose | 正负偏置、非对称外廓、四种圆弧与中心速度导数通过 | ✅ |
| WHEEL-BASE-004 | 采样变化率/换向/停车转向 | tests.test_wheeled_vehicle.TestWheeledModel | KinematicBicycleModel._limited_speed/step | 限幅/变化率/换向/停车通过，含2000步随机序列 | ✅ |
| WHEEL-BASE-005 | 解析正反圆弧、直线极限、中心速度 | tests.test_wheeled_vehicle.TestWheeledModel | sim/wheeled_vehicle.py::KinematicBicycleModel | 后轴圆心/半径及中心导数解析对照通过 | ✅ |
| WHEEL-BASE-006 | 旧行为与新旧对象隔离 | tests.test_wheeled_vehicle.TestLegacyIsolation；tests.test_vehicle_model_acceptance | 新模块直接导入；旧代码不修改 | 旧27项回归+新旧隔离测试通过，六份旧源SHA不变 | ✅ |

## 待人工确认

- 真实矿卡转向结构与标定参数、实物履带平台参数尚未提供；本环节可使用明确标记的研究假设，不作为实车适用性验收。
