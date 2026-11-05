# hum_exo 人机协同训练框架

本框架在 MuscleMimic 的 MyoFullBody、JAX PPO 和 MuJoCo Warp 上提供人体与辅助设备的联合训练。人体 actor 使用人体和参考动作观测，设备 actor 使用清单声明的传感器，共同 critic 使用完整观测。奖励同时考虑动作跟踪、肌肉用力和设备输出。

当前已提供双侧膝关节设备。髋、踝设备通过相同契约接入，需要各自的资产、安装和参考初始化实现。当前人体宿主为 MyoFullBody。

## 目录职责

```text
hum_exo/
  train.py                  GPU 训练与运行快照
  environment.py            人体环境、设备观测和参考动作接入
  policy.py                 独立人体/设备 actor 与共同 critic
  reward.py                 跟踪、肌肉用力和设备代价
  evaluation.py             共享策略评估和参考动作回放
  export.py, controller.py  设备策略导出与 NumPy 部署
  checkpoint.py, paths.py   模型读取和可移动路径
  visualization.py          关节轴线测量与绘制
  fitting.py                设备安装调节会话契约
  devices/<device_id>/      设备专用机构、安装、传感器和参考拟合
scripts/hum_exo/             Play、预览、安装调节、导出工具
assets/<device_id>/          URDF、网格、设备清单和安装配置
outputs/日期/时间/           模型快照、checkpoint、训练配置和日志
generated/<device_id>/     可重建的模型、拟合缓存、预览和评估输出
```

`hum_exo` 的计算模块不依赖 `scripts`。工具调用公共接口，设备适配器实现机构差异；新增设备不复制 PPO、日志、评估或网页界面。

## 运行与接入

在仓库根目录使用 `./scripts/hum_exo.sh`。完整训练、播放、续训和导出命令见 [命令手册](../readme1.md)，工具模块入口见 [scripts/hum_exo](../scripts/hum_exo/README.md)。VS Code 配置位于 [.vscode/launch.json](../.vscode/launch.json)。

新增设备的资产清单、接口和接入步骤见 [DEVICE_API.md](DEVICE_API.md)。安装调节方法见 [FITTING.md](FITTING.md)，当前膝关节资产见 [assets/knee_exo](../assets/knee_exo/README.md)。

## 训练和保存

默认使用 32 个 GPU 环境、每轮 64 步和 3000 万步预算；launch 使用 128 个环境。训练轮数向上取整，至少完成预算。例如 2048 环境对应 229 轮、30,015,488 步，结尾显示真实步数、100% 和 `learning done!`。

准备阶段只显示初始化提示，首次真实更新后出现 Rich 进度条和 PPO 表格。`training_console.log` 保存表格，`training_metrics.jsonl` 保存原始指标及逐轮耗时。Orbax 内部 INFO 留在实验日志，终端保留简要保存通知、警告和错误。W&B 默认关闭。

带 `--headless` 时关闭训练窗口，训练继续使用 GPU。窗口从独立进程读取真实状态，网格显示偏移不参与仿真。GPU 求解器沿用官方默认设置，并写入快照，播放和续训复用这些设置。

每次训练写入新的 `outputs/日期/时间/`，checkpoint 位于 `checkpoints/运行标识/checkpoint_编号/`。同时保存完整人体设备模型、`device.json`、`interface.json`、配置和参考拟合报告。Play 路径留空时，按保存时间选择 `outputs/` 下同设备的最新完整 checkpoint。

## 路径与有效范围

根目录通过 `pyproject.toml` 定位，可修改外层文件夹名称。仓库内路径支持 `project://`，运行快照支持 `run://`；MuJoCo 网格和纹理使用相对路径。Python 包名保持不变。移动环境后如 `.venv` 失效，重新执行 `uv sync --extra cuda`。

设备输入顺序、单位、电机顺序和自由度属于保存契约；不匹配时拒绝加载。续训使用保存的物理模型，避免安装调整改变原实验。更换传感器布局后需要重新训练或显式迁移。

当前实现使用理想电机和数值绑缚约束。安装轴线对齐、参考轨迹拟合和短训练通过，只能证明相应接口和执行流程；步态稳定性、动态绑缚负载、助力收益及硬件安全需要独立评估。肌肉平方激活是用力代理量。
