# 框架拆分与路径验证

2026-11-04 将公共逻辑从 `knee_exo/` 提取到 `hum_exo/`，膝关节源资产迁移到 `assets/knee_exo/`。设备机构、安装校准和参考初始化保留在 `hum_exo/devices/knee_exo/`。当前上合页 -6 度、电机偏置 56 mm、左右 URDF 和网格保持原定义。历史运行目录保留。

公共模块不依赖膝关节适配器。设备清单声明有序传感器通道、单位、执行器和设备关节。测试使用单输入单电机、七输入三电机两种接口验证网络构造，另验证清单自动导入、输入隔离、NumPy 部署、路径重定位、真实设备几何和原训练日志格式。

`JAX_PLATFORMS=cpu MUJOCO_GL=egl .venv/bin/python -m pytest hum_exo/tests -q` 通过 18 项测试。`JAX_PLATFORMS=cpu make ci` 通过 489 项测试，25 项跳过、15 项按仓库规则排除。Ruff 和差异空白检查通过。

GPU 验证使用 4 个环境、每轮 8 步、128 总步，人体 actor 与 critic 隐藏层为 `[16]`，设备 actor 为 `[8]`，自动评估使用 2 个环境、每次 4 步。完成训练、评估与 checkpoint 保存，输出位于忽略目录 `generated/framework_check/train/`。该规模验证执行流程，不用于评估步态和助力质量。

在临时目录创建与实际项目名称不同、带空格的 `Human Assist Renamed Project` 测试副本，使用现有依赖环境。副本中新构建设备 XML，正确发现新根目录。加载复制的 checkpoint 播放成功，并从 128 步续训到 160 步，checkpoint_5 保存在新目录。

随后实际将该副本重命名为 `Portable Exo Workspace`，重新加载续训的 checkpoint 播放并导出设备 actor，两者通过。迁移前的旧 checkpoint 也在改名后的副本中完成播放，历史绝对网格路径通过兼容映射找到 `assets/knee_exo/`。原工作目录未重命名。

新模型 XML 中没有机器绝对路径。网格和纹理按 XML 所在目录相对引用。复制续训模型时重新计算相对路径，保持模型的物理定义。实验配置快照使用 `run://`，运行配置支持 `project://`。旧模型的路径修复写入忽略的缓存，不改写原 checkpoint 或 XML。

保留 `python -m knee_exo.train/play/preview/export` 转发入口。可改名的是外层仓库目录；Python 模块目录和包导入名保持稳定。启动脚本直接执行当前根目录 `.venv/bin/python`，避免旧虚拟环境工具脚本 shebang 中的绝对目录。跨机器迁移或 Python 运行环境失效仍需重建依赖环境。自定义外部数据集路径不在仓库内资源重定位的范围内。

当前真实设备只有膝关节。新设备需要实现自己的模型构建和参考适配，并提供完整资产清单。未声称其他外骨骼已完成物理验证，也未启动大规模训练。三自由度设备与人体耦合膝的运动学残差、动态绑缚负载和助力收益仍需独立验证。

后续求解器修复、最新贴合和动态训练终端实测见 [SOLVER_CONSOLE_VALIDATION.md](SOLVER_CONSOLE_VALIDATION.md)，当前安装参数以设备 JSON 为准。
