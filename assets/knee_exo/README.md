# 双侧膝关节外骨骼资产

本包迁自 MyoAssist。`urdf/` 保存左右 URDF，`meshes/` 保存八个 CAD 网格，`device.json` 定义框架接口。`source_mounting.json` 保存原人体安装来源，`mounting.json` 定义当前安装，`imu_mounts.json` 定义四个 IMU，`training_interface.json` 保存设备动力学参数。

当前电机轴向偏置 54.5 mm，上合页 -7.4 度，下合页约 +0.123989 度。相对此前 55 mm 安装，电机内移 0.5 mm，电机旁绑缚上端净内移约 0.702 mm；小腿骨骼最小间隙约 1 mm。大腿肌肉显示重叠约 0.438 mm，手部显示几何无交叉。用户选择优先大腿贴合、手部无干涉，允许减小原先的移动目标。机构、质量、惯量和 URDF 保持原定义。

框架通过 `hum_exo.devices.knee_exo.adapter:KneeExoAdapter` 接入，设备输入为 26 维，输出顺序为 `Exo_R`、`Exo_L`。设备专用模型构建、合页拟合和参考初始化代码位于 `hum_exo/devices/knee_exo/`。

在仓库根目录运行：

```bash
./scripts/hum_exo.sh preview --device knee_exo
./scripts/hum_exo.sh train device=knee_exo
```

完整命令见 [框架说明](../../hum_exo/README.md)。安装和既往验证记录位于 [docs/](docs/MOUNTING.md)。过去记录中的 `knee_exo/` 代码和输出路径保留为历史说明，当前布局以本文和框架说明为准。
