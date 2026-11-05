# 双侧膝关节外骨骼

本设备资产迁自 MyoAssist，CAD 由 SolidWorks 导出。左右设备包含大腿绑缚、电机壳体、电机输出连接件和小腿绑缚，具有上合页、电机膝关节和下合页三个转动自由度。左侧由右侧镜像得到。

## 文件

- `urdf/`、`meshes/`：左右机构定义、质量惯量和 CAD 网格。
- `device.json`：设备适配器、有序传感器通道、执行器、关节及显示部件根节点。
- `mounting.json`：安装基础参数及当前启用配置的路径。
- `fitting_profile.json`：当前启用的交互式安装配置。
- `imu_mounts.json`：四个 IMU 的安装位置和方向。
- `training_interface.json`：电机及被动关节动力学参数。
- `source_mounting.json`：MyoAssist 的安装来源。

当前输入为左右大腿、左右小腿四个 IMU 的三轴加速度和三轴角速度，加右膝角度与角速度，共 26 维。输出顺序为 `Exo_R`、`Exo_L`。

## 安装和使用

有效安装先读取 `mounting.json`，再应用其 `fitting_profile` 指向的配置。以实际 JSON 和构建报告为准。当前交互配置两侧均为轴向偏置 48.5 mm、上合页 -5.4 度、下合页 -2.1 度、绕共同轴旋转 2.3 度。

```bash
./scripts/hum_exo.sh preview --device knee_exo
./scripts/hum_exo.sh fit --device knee_exo
./scripts/hum_exo.sh train device=knee_exo experiment.env_params.num_envs=128 --headless
```

机构构建、安装校准、传感器与参考初始化位于 `hum_exo/devices/knee_exo/`，通过 `KneeExoAdapter` 接入公共框架。安装和接触模型说明见 [MOUNTING.md](docs/MOUNTING.md)，运行参数见 [命令手册](../../readme1.md)。
