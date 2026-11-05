# 设备接入契约 v1

`hum_exo` 管理人机协同 PPO、双 actor 与共同 critic、观测分组、日志、checkpoint、播放与设备 actor 导出。设备清单和适配器构成稳定边界；增加其他外骨骼时，公共代码不需要按设备名称或传感器维度增加分支。

当前人体宿主为 MyoFullBody，CPU MuJoCo 与 GPU MuJoCo Warp 共享设备定义。更换人体模型需要另行提供宿主环境适配。现有网络要求当前帧设备观测，历史窗口可以作为新设备传感器契约扩展，尚未实现自动堆叠。

新增设备时创建 `assets/<device_id>/device.json`，例如：

```json
{
  "schema_version": 1,
  "adapter": "hum_exo.devices.ankle_exo.adapter:AnkleExoAdapter",
  "human_model": "MyoFullBody",
  "visual_body_roots": ["ankle_frame_r", "ankle_frame_l"],
  "actuators": ["ankle_motor_r", "ankle_motor_l"],
  "joints": ["ankle_device_r", "ankle_device_l"],
  "sensors": [
    {"name": "right_encoder", "channels": ["right_ankle.angle"], "units": ["rad"]},
    {"name": "left_encoder", "channels": ["left_ankle.angle"], "units": ["rad"]}
  ]
}
```

`visual_body_roots` 可选，声明设备刚体子树的根名称，网页据此着色和选取设备，不要求膝关节专属命名。省略时按设备关节的子树识别；含固定支架时应显式填写。

`actuators` 决定设备输出顺序。`joints` 声明设备自由度，框架将其排除在人体模仿目标之外。`sensors` 按清单顺序读取 MuJoCo `sensordata`，每个条目的通道数必须匹配实际传感器维度。人体 actor 使用其余仿真及目标观测，设备 actor 只使用设备组，共同 critic 使用完整观测。

在适配器模块实现 `DeviceAdapter`：

```python
from hum_exo.device import DeviceAdapter


class AnkleExoAdapter(DeviceAdapter):
    def build_model(self, output=None):
        # 从 self.spec.asset_root 读取 CAD，将设备挂载到人体。
        # 保存完整 MuJoCo XML，资源路径使用 write_portable_xml。
        # 输出路径通常为 output 或 self.spec.generated_dir / "human_exo.xml"。
        ...

    def extend_reference(self, env):
        # env.th 已转为 NumPy，按关节名称扩展设备 qpos/qvel。
        # 人体目标保持原定义，返回可序列化的误差报告。
        ...

    def preview_axes(self):
        return (("ankle_angle_r", "ankle_device_r"),)

    def metrics(self, model, data):
        return {}
```

`build_model` 和 `extend_reference` 必须实现，后两项可选。设备目录由 manifest 中的完整模块路径自动导入，无需编辑公共 registry。`self.spec.input_names`、`input_units`、`actuators` 和 `joints` 是运行时契约。模型必须保留人体需要的观测和参考站点，全部设备执行器均需在清单中声明。

需要交互式安装调节时，可实现可选的 `create_fitting_session()`，返回符合 `hum_exo.fitting.FittingSession` 的会话。会话负责设备关节运动学、测点距离、贴合求解、干涉检查及安装配置保存。公共 `scripts.hum_exo.fit` 界面读取会话声明的参数、侧别和测点；未实现该接口的设备仍可训练和播放。详见 [FITTING.md](FITTING.md)。

当前播放和导出支持标量转动或移动关节上的理想电机，归一化动作范围为 [-1,1]。设备必须在模型中定义正确的 gear、限幅和 SI 单位。复杂传动、气动或电机动态需要在设备模型和设备测量接口中明确实现，不应复用默认力矩解释。

公共接口版本为 `schema_version=1`。修改传感器顺序、单位、动作顺序或设备自由度属于不兼容变更，应使用新的设备 ID 或进行显式迁移。训练保存 `device.json`、`interface.json` 与模型快照，已有 checkpoint 保持原观测和动作布局。

新设备运行入口：

```bash
./scripts/hum_exo.sh preview --device ankle_exo
./scripts/hum_exo.sh train device=ankle_exo
```

这些命令需要先实现相应 CAD 和适配器。测试中的单电机及三电机设备用于验证公共框架的维度与加载接口，不代表已经提供了踝关节物理资产。

## 新设备接入清单

1. 创建 `assets/hip_exo/` 或 `assets/ankle_exo/`，加入 URDF、网格、安装参数和 `device.json`。
2. 在 `hum_exo/devices/<device_id>/` 实现 `DeviceAdapter`。保留人体站点和观测，声明全部设备关节、电机及真实可用的传感器。
3. 实现 `build_model()` 与 `extend_reference()`。校准坐标系、零位、轴向、质量惯量、电机方向和限幅。
4. 需要安装编辑时实现 `create_fitting_session()`；测点、调节自由度和约束留在设备实现。`preview_axes()` 与 `metrics()` 提供轴线及专用评价量。
5. 用 `preview --device <device_id>` 检查安装，再执行小规模 GPU 训练、保存加载和电机关闭对照评估。
6. 正式训练使用 `train device=<device_id>`，播放自动选择该设备时添加 `play --device <device_id>`。奖励、参考动作和 PPO 参数通过现有配置覆盖。

通用默认值仍选择 `knee_exo`。髋和踝资产尚需实现；无需复制训练器、策略、终端日志、Play 或网页工具。训练输出与设备资产分开保存。
