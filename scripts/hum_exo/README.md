# 人机协同工具

本目录保存命令行入口与交互界面。训练、评估、导出计算和设备运动学由 `hum_exo` 提供，资产由 `assets/<device_id>` 提供。

在仓库根目录执行：

```bash
./scripts/hum_exo.sh train device=knee_exo experiment.env_params.num_envs=128 --headless
./scripts/hum_exo.sh play --seconds 120
./scripts/hum_exo.sh preview --device knee_exo
./scripts/hum_exo.sh fit --device knee_exo
./scripts/hum_exo.sh export outputs/日期/时间/checkpoints/运行标识/checkpoint_编号
```

`hum_exo.sh` 根据自身位置定位仓库并使用 `.venv/bin/python`。训练入口为 `python -m hum_exo.train`；其他工具分别为 `python -m scripts.hum_exo.play`、`preview`、`fit`、`export`，每个工具支持 `--help`。Play 不填写模型路径时加载最新完整模型。

`file_picker.py` 提供网页保存目录选择器，`fit.py` 提供通用安装界面。设备测点、调节参数和硬约束通过 `DeviceAdapter.create_fitting_session()` 实现。

完整参数和续训示例见 [命令手册](../../readme1.md)，设备接口见 [DEVICE_API.md](../../hum_exo/DEVICE_API.md)。
