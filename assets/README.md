# 外骨骼设备资产

每种设备放在独立的 `assets/<device_id>/` 目录。`device.json` 是框架接入入口，描述版本、适配器、执行器、设备关节及有序传感器通道和单位。当前设备为 [knee_exo](knee_exo/README.md)。

URDF、网格、安装参数和设备说明随设备保存。公共训练逻辑位于 `hum_exo/`；机构运动学、参考初始化和设备专用测量由 `hum_exo/devices/<device_id>/` 中的适配器实现。运行时生成的 XML、图片和导出文件进入根目录 `generated/<device_id>/`，新训练结果进入 `outputs/日期/时间/`。

新增设备的契约和示例见 [设备接入说明](../hum_exo/DEVICE_API.md)。源 URDF 中的 package 名称是 CAD 资源标识，与仓库根目录名无关。
