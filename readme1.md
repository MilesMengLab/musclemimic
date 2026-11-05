# 人体与膝关节外骨骼训练命令

当前项目以 MuscleMimic 的 MyoFullBody、JAX PPO 和 MuJoCo Warp 为基础，由 `hum_exo/` 提供人体与外骨骼分开控制、联合训练、播放及设备策略导出。膝关节资产在 `assets/knee_exo/`，设备侧输入为四个 IMU 加右膝编码器共 26 维，输出为左右两个电机命令。

以下命令在当前仓库根目录执行。启动脚本自动使用 `.venv/bin/python`，无需手动激活虚拟环境。项目外层文件夹可以改名，Python 包名和模块目录保持不变。

## 安装环境

首次配置环境时执行：

```bash
uv sync --extra cuda
```

已经安装成功后，可以直接运行后续命令。需要使用已配置的本地代理时，在命令前添加 `./scripts/with_proxy.sh`。

## GPU 训练

128 个并行环境，开启 MuJoCo 网格渲染：

```bash
./scripts/hum_exo.sh train device=knee_exo \
  experiment.env_params.num_envs=128 \
  experiment.total_timesteps=30000000
```

128 个并行环境，关闭渲染：

```bash
./scripts/hum_exo.sh train device=knee_exo \
  experiment.env_params.num_envs=128 \
  experiment.total_timesteps=30000000 \
  --headless
```

训练命令带 `--headless` 时不打开窗口；不带时显示训练状态的网格画面。两者均使用 GPU 训练。关闭渲染窗口后训练继续，Ctrl+C 停止训练进程。渲染会增加系统负载。

环境数量直接修改 `experiment.env_params.num_envs`。上述命令及 launch 配置使用 128，基础配置文件的默认值仍为 32。每轮每个环境默认采样 64 个控制步，128 环境合计 8192 个步数；3000 万步是所有环境累计的总预算。

训练轮数按 `ceil(总预算 / (环境数 × 每轮采样步数))` 向上取整，至少跑满预算。2048 环境、每轮 64 步、3000 万步预算对应 229 轮，最终累计 30,015,488 步。最后进度条显示真实累计步数、100% 和零剩余时间，再打印 `learning done!`。

仅显示其中一个环境，保持 128 个环境训练：

```bash
./scripts/hum_exo.sh train device=knee_exo \
  experiment.env_params.num_envs=128 \
  experiment.render.num_envs=1
```

准备阶段显示“初始化 / 编译中”，第一轮真实训练指标返回后显示动态进度条。新训练输出使用 MuscleMimic 的 `outputs/日期/时间/` 格式，包含模型快照、配置、`checkpoints/运行标识/checkpoint_编号/`、`training_console.log` 和 `training_metrics.jsonl`。

终端中每次模型保存只显示一行路径和累计步数。Orbax 内部 INFO 写入实验日志，警告和错误继续在终端显示。

## 推理播放与评估

直接播放最近保存的完整膝关节模型：

```bash
./scripts/hum_exo.sh play --seconds 120 --seed 0
```

也可输入一次 checkpoint 完整目录路径，留空或只输入空格时同样选择最新模型：

```bash
read -r -p "checkpoint_编号完整目录路径（留空使用最新模型）: " exo_checkpoint
./scripts/hum_exo.sh play \
  --checkpoint "$exo_checkpoint" \
  --seconds 120 \
  --seed 0
```

新路径指向 `outputs/日期/时间/checkpoints/运行标识/checkpoint_编号` 目录，支持绝对路径。自动选择会搜索 `outputs/`，按保存时间选取同设备的完整模型，跳过临时保存目录、缺少保存项的目录和其他训练环境。无需另外指定训练配置或模型 XML。模型使用 Orbax checkpoint 目录。

无窗口评估，复用上述 `exo_checkpoint` 变量：

```bash
./scripts/hum_exo.sh play \
  --checkpoint "$exo_checkpoint" \
  --seconds 120 \
  --seed 0 \
  --no-viewer \
  --output generated/knee_exo/evaluation/policy_metrics.json
```

关闭设备电机的对照评估：

```bash
./scripts/hum_exo.sh play \
  --checkpoint "$exo_checkpoint" \
  --seconds 120 \
  --seed 0 \
  --motor-off \
  --no-viewer \
  --output generated/knee_exo/evaluation/motor_off_metrics.json
```

训练入口使用 `--headless`，播放入口使用 `--no-viewer`。播放中的 `--seconds` 表示总仿真时长，允许回合结束后自动重置。推理默认使用确定性动作，MuJoCo 物理步进在 CPU 上执行，JAX 策略推理使用可用设备。

评估 JSON 包含回报、回合结束标记、膝角、电机力矩、功率、肌肉平方激活和绑缚位置误差。不指定 `--output` 时写入 `generated/knee_exo/play_metrics.json`；需要保留多次结果时，为 `--output` 指定不同文件名。

## 安装预览与可视化调节

查看当前安装和人体、电机轴线：

```bash
./scripts/hum_exo.sh preview --device knee_exo
```

打开网页调整安装配合：

```bash
./scripts/hum_exo.sh fit --device knee_exo
```

网页支持选择保存目录，保存文件自动添加时间后缀。轴线对齐是生成和应用的硬约束，其他干涉只显示警告。应用后，新训练使用启用的安装版本；已有 checkpoint 继续使用训练时的模型快照。

参考动作预览：

```bash
./scripts/hum_exo.sh play --preview --device knee_exo --seconds 120
```

参考预览直接设置参考姿态，用于检查安装随动作变化的效果。训练策略的动力学结果通过 `--checkpoint` 播放检查。

## VS Code 启动

打开项目根目录，在“运行和调试”中选择 `.vscode/launch.json` 的配置，按 F5。配置包含 128 环境训练的渲染和无窗口版本、120 秒推理播放和无窗口评估，以及安装预览和网页调节。

推理路径提示框默认留空，直接回车或只输入空格会播放最新模型，也可填写完整 checkpoint 目录路径。环境数量、总步数和播放时长都直接写在各配置的 `args` 中，可按需修改。解释器和工作目录通过 `${workspaceFolder}` 定位；训练渲染子进程独立运行，调试器只附加主进程。

## 续训与策略导出

续训和导出时填写具体 checkpoint 路径。续训的环境数量、每轮采样设置及网络尺寸应与原 checkpoint 保持一致。对使用上述 128 环境设置训练的模型，可执行：

```bash
read -r -p "续训 checkpoint_编号完整目录路径: " exo_checkpoint
./scripts/hum_exo.sh train device=knee_exo \
  experiment.env_params.num_envs=128 \
  experiment.resume_from="$exo_checkpoint" \
  experiment.total_timesteps=60000000 \
  --headless
```

总预算包含已经完成的步数，续训会写入新的实验目录。导出设备 actor：

```bash
./scripts/hum_exo.sh export "$exo_checkpoint" \
  --output generated/knee_exo/export
```

导出结果包含 `device_actor.npz` 和 `input_schema.json`，记录设备输入顺序、单位、归一化和输出映射。框架说明见 [hum_exo/README.md](hum_exo/README.md)，安装操作见 [hum_exo/FITTING.md](hum_exo/FITTING.md)。

## 目录与工具入口

公共训练和计算位于 `hum_exo/`，工具位于 `scripts/hum_exo/`，设备资产位于 `assets/<device_id>/`。统一使用本文的 `./scripts/hum_exo.sh` 命令；Play、预览、调节、导出模块入口分别为 `scripts.hum_exo.play`、`scripts.hum_exo.preview`、`scripts.hum_exo.fit`、`scripts.hum_exo.export`。新增设备见 [接入契约](hum_exo/DEVICE_API.md)。
