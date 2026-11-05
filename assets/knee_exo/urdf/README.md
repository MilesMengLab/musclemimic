# 设备来源

本目录 URDF 及相邻 meshes 来自当前 MyoAssist 项目的 knee_exo_hinge。
右侧保留清理后的 SolidWorks 导出关节定义，左侧由右侧镜像得到。
结构为 base_link、motor_housing_link、motor_output_link 和 shank_link，
包含 upper_hinge_joint、knee_joint、lower_hinge_joint。
质量、惯量、网格、关节轴和限位保留原文件定义。

设备接入与命令见 ../README.md。source_mounting.json 是原人体的
校准来源；新人体的候选安装由构建器生成并记录于报告。
