# 本地代理启动

项目根目录的 `.env.proxy` 保存本机代理地址，格式为
`MUSCLEMIMIC_HTTP_PROXY=http://代理主机:端口`。该文件由 Git 忽略。

在项目根目录使用以下入口启动命令。脚本读取本地配置，移除
`all_proxy` 和 `ALL_PROXY`，为目标进程和子进程设置四个 HTTP 代理变量。
当前 shell 的变量保持原值。

```bash
./scripts/with_proxy.sh uv run --extra cuda musclemimic-download-gmr-caches \
  --dataset-group KIT_KINESIS_TRAINING_MOTIONS

./scripts/with_proxy.sh uv run --extra cuda python fullbody/experiment.py \
  --config-name=conf_fullbody_demo

./scripts/with_proxy.sh uv sync --extra cuda
```

该入口适用于下载、登录、训练和评估，新终端也可以直接使用。
已有运行中的进程保留其启动时的环境。代理客户端需要运行并监听配置的端口。
现有 `NO_PROXY` 和 `no_proxy` 规则继续生效，未匹配排除规则的 HTTP(S)
请求使用所配置的代理。网络路由配置不会改变数据仓库的访问权限。
