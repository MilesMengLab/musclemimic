# hum_exo 人机协同训练框架

公共框架位于 `hum_exo/`，设备资产位于 `assets/<device_id>/`，设备专用运动学和参考适配位于 `hum_exo/devices/<device_id>/`。当前膝关节设备为 `knee_exo`。训练复用 MuscleMimic 的 JAX PPO、MyoFullBody 和 MuJoCo Warp。

公共模块负责双 actor 与共同 critic、设备观测隔离、奖励、训练日志、checkpoint、播放与策略导出。传感器维度和电机数量从 `device.json` 读取。后续设备通过清单和适配器接入，契约见 [DEVICE_API.md](DEVICE_API.md)。公共接口采用 v1，设备机构变化留在设备适配器中。

## 运行入口

以下命令在仓库根目录执行。脚本根据自身位置找到项目，不依赖根目录名称，也可从其他目录通过脚本完整路径启动。

```bash
./scripts/hum_exo.sh preview --device knee_exo
./scripts/hum_exo.sh fit --device knee_exo
./scripts/hum_exo.sh train device=knee_exo
./scripts/hum_exo.sh train device=knee_exo --headless
./scripts/hum_exo.sh play --checkpoint runs/knee_exo/日期/时间/checkpoints/checkpoint_编号 --seconds 60
./scripts/hum_exo.sh export runs/knee_exo/日期/时间/checkpoints/checkpoint_编号
```

等价模块入口为 `python -m hum_exo.preview`、`python -m hum_exo.fit`、`python -m hum_exo.train`、`python -m hum_exo.play` 和 `python -m hum_exo.export`。可以继续通过 `./scripts/with_proxy.sh` 包裹命令以应用本地网络设置。

```bash
./scripts/with_proxy.sh ./scripts/hum_exo.sh train device=knee_exo
```

预览显示当前中立位的设备和人体转轴。`--render` 保存三个视角截图与轴线报告。参考动作展示使用 `./scripts/hum_exo.sh play --preview --device knee_exo --seconds 30`，它直接设置拟合参考姿态，不代表策略控制的动力学结果。

安装调节命令 `./scripts/hum_exo.sh fit --device knee_exo` 打开本地浏览器页面，默认地址为 `http://127.0.0.1:8080`。可用 `--port 8081` 更换端口，`--no-browser` 关闭自动打开浏览器。工具使用 CPU MuJoCo 计算静态装配运动学，浏览器显示三维几何，不加载 JAX 或策略，不改变 GPU 训练配置。操作方法见 [FITTING.md](FITTING.md)。

默认训练使用 32 个 GPU 环境、每轮 64 个控制步、3000 万步预算及已下载的三段示例动作。终端采用与 MyoAssist/SB3 相同的无时间戳分组表格、3 位有效数字和 Rich 动态进度条，首次更新包含 JAX/Warp 编译。进度条显示百分比、累计步数、已用时间、剩余时间和 it/s，按每轮 GPU rollout 更新；终端持续刷新，重定向到文件时输出最终进度。表格存入 `training_console.log`，原始指标和逐轮耗时存入 `training_metrics.jsonl`。W&B 默认关闭。

训练命令不带 `--headless` 时打开独立 MuJoCo 窗口，将全部环境按网格显示；带 `--headless` 时不打开窗口。窗口读取训练中各环境的真实位置快照，显示偏移不反馈到仿真。关闭窗口后训练继续。首次编译期间显示等待提示。渲染会增加 CPU/GPU 负载，显示默认最多每秒 10 帧，不强制训练按真实时间运行。可用 `experiment.render.num_envs=1` 只显示一个环境，或 `experiment.render.fps=5` 调低刷新率。程序化 Hydra 配置默认仍不启用显示，CLI 负责转换该开关。

准备阶段只显示“初始化 / 编译中”，首轮真实训练指标返回后才出现进度条。网络参数消息和零步数消息不会触发进度条。续训进度从保存步数继续；表格和 JSONL 的累计时间仍包含初始化及首次编译，进度条计时从首次显示开始。

GPU 环境通过原版 `MjxMyoFullBody` 初始化，默认求解器 `iterations=4`、`ls_iterations=8`，并关闭 `EULERDAMP`，与官方设置一致。实际设置写入实验配置，后续播放使用同一设置。可通过 `experiment.env_params.model_option_conf` 显式覆盖；减少迭代上限会改变数值精度，绑缚负载和稳定性仍需检查。

```bash
./scripts/hum_exo.sh train device=knee_exo experiment.env_params.num_envs=128
./scripts/hum_exo.sh train device=knee_exo experiment.env_params.num_envs=128 --headless
```

本机短基准中，32 环境官方求解器约 1033 步/秒，128 环境约 3049 步/秒，均排除了前四次编译和预热更新。长训练吞吐与收敛效果需继续观察，默认环境数仍为 32。

播放默认使用确定性策略和 CPU MuJoCo。`--no-viewer` 关闭窗口，`--seed` 固定初态，`--motor-off` 施加零设备命令，`--output` 指定评估 JSON。导出权重可通过 `hum_exo.controller.DeviceController` 用 NumPy 推理，输入维度和输出顺序来自导出的 `input_schema.json`。

```python
from hum_exo.controller import DeviceController

controller = DeviceController("generated/knee_exo/export/device_actor.npz")
commands = controller(raw_sensors)
```

续训需要匹配设备 ID、网络尺寸和 rollout 配置，默认配置训练的实验可使用：

```bash
./scripts/hum_exo.sh train device=knee_exo \
  experiment.resume_from=runs/knee_exo/日期/时间/checkpoints/checkpoint_编号 \
  experiment.total_timesteps=60000000
```

总预算包含已经完成的步数。续训复用原模型快照，写入新运行目录。修改传感器顺序、单位或电机顺序后，不允许静默加载不匹配的 checkpoint。

## 目录和路径

```text
hum_exo/                     公共训练、观测、策略、播放和导出
  configs/train.yaml         公共配置
  device.py                  设备契约与自动加载
  paths.py                   根目录发现、模型和 checkpoint 路径
  devices/knee_exo/          膝关节专用机构、安装和参考适配
assets/
  knee_exo/                  URDF、STL、传感器、安装参数和设备清单
runs/<device_id>/            新训练结果
generated/<device_id>/      构建模型、拟合缓存、截图和部署导出
knee_exo/                    旧命令兼容入口及历史输出
```

根目录由 `pyproject.toml` 的位置确定，显示名称由当前根目录名读取。新 XML 的网格和纹理使用相对路径；配置使用 `project://`，可移动的实验配置快照使用 `run://`。checkpoint 根据实际所在的实验目录寻找模型，旧版绝对资产路径会在加载时映射到当前设备资产，原实验 XML 保持不变。

可以修改仓库文件夹名称，Python 包名 `hum_exo`、`musclemimic`、`loco_mujoco` 和模块目录需要保持。它们是导入接口，不随外层目录重命名。`.venv` 内工具脚本可能含环境创建时的绝对 shebang；推荐使用上面的启动脚本，它直接调用当前目录的 `.venv/bin/python`。换机器或 Python 环境失效时重新执行 `uv sync --extra cuda`。外部数据集、用户指定的绝对路径及本地代理端口仍由各自配置负责。

原来的 `python -m knee_exo.preview/train/play/export` 命令保留转发入口。旧 `knee_exo/generated/` 和训练记录未移动。新输出统一使用根目录 `generated/`、`runs/`，无需在设备资产目录保存运行产物。

## 当前膝关节状态与限制

设备使用四个 IMU 加右膝编码器共 26 维输入，输出右、左两个电机。按用户选定的优先级，大腿保持轻微贴合并避开手部。当前电机轴向偏置为 54.5 mm，比此前 55 mm 再内移 0.5 mm；电机旁小腿绑缚上端净内移约 0.702 mm。上、下合页分别为 -7.4 度和 +0.123989 度，小腿骨骼最小间隙约 1 mm。大腿肌肉显示重叠约 0.438 mm，手部不相交。原先 3 mm/2 mm 目标因几何冲突减小，详情见设备安装说明。膝轴继续共轴，URDF 和网格保持原定义。

上述安装和避让结果针对中立位，动态动作下的接触需要另行检查。当前实现是可运行的研究原型。数值 weld 表示理想化绑缚，三自由度设备与人体耦合膝的运动拟合仍有约 17 mm 的峰值位置残差。短训练验证执行流程，稳定步态、动态绑缚负载、助力收益及真实硬件控制需要独立验证。肌肉平方激活是用力代理指标，不等于代谢消耗。

安装细节见 [设备说明](../assets/knee_exo/README.md)，重构验证见 [REFACTOR_VALIDATION.md](REFACTOR_VALIDATION.md)。

求解器修复、动态终端显示和最新贴合的实测记录见 [SOLVER_CONSOLE_VALIDATION.md](SOLVER_CONSOLE_VALIDATION.md)。
