"""Browser-based human/exoskeleton installation editor with real CAD joints."""

# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import json
import threading
import time
import webbrowser
from functools import partial
from html import escape
from pathlib import Path
from queue import Empty, Queue

import mujoco
import numpy as np
import viser

from hum_exo.device import load_device
from hum_exo.file_picker import DirectoryPicker
from hum_exo.geometry import mesh_geometry, rendered_tendon_capsules
from hum_exo.paths import project_name, resolve_path

UI_HZ = 20
POINT_COLORS = ((255, 210, 30), (55, 220, 180), (70, 160, 255), (225, 100, 245))


class TextBlock:
    """Render literal UI text without Markdown parsing, preserving line breaks."""

    def __init__(self, gui: viser.GuiApi, content: str = "") -> None:
        self._handle = gui.add_html("")
        self.content = content

    @property
    def content(self) -> str:
        return self._content

    @content.setter
    def content(self, text: str) -> None:
        self._content = text
        self._handle.content = (
            '<div style="white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px;line-height:1.5">'
            + escape(text)
            + "</div>"
        )


class FittingApp:
    """Viser application; a single worker owns all MuJoCo mutations and file I/O."""

    def __init__(self, device_id: str = "knee_exo", host: str = "127.0.0.1", port: int = 8080) -> None:
        self.adapter = load_device(device_id)
        self.session = self.adapter.create_fitting_session()
        self.server = viser.ViserServer(host=host, port=port, label=f"{project_name()} · 外骨骼安装调节")
        self.queue = Queue()
        self.running = threading.Event()
        self.running.set()
        self._updating_gui = False
        self.meshes = {}
        self.tendons = []
        self.points, self.point_lines, self.labels = [], [], []
        self.axes = []
        self._scene()
        self._gui()
        self._sync()
        self._readouts()
        self.thread = threading.Thread(target=self._loop, name="exo-fitting", daemon=True)
        self.thread.start()

    def _scene(self) -> None:
        model, data = self.session.model, self.session.data
        self.server.scene.set_up_direction("+z")
        self.server.scene.configure_environment_map(environment_intensity=0.8)
        self.server.scene.add_grid(
            "/floor",
            plane="xy",
            width=5,
            height=5,
            cell_size=0.25,
            cell_color=(185, 196, 209),
            section_color=(126, 145, 168),
        )
        for geom in range(model.ngeom):
            if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[geom] != 0:
                continue
            body = int(model.geom_bodyid[geom])
            vertices, faces = mesh_geometry(model, data, geom)
            local = (vertices - data.xpos[body]) @ data.xmat[body].reshape(3, 3)
            is_device = model.body(body).name.startswith("hinge_")
            rgba = model.geom_rgba[geom]
            color = tuple(int(v * 255) for v in rgba[:3]) if is_device else (220, 225, 230)
            handle = self.server.scene.add_mesh_simple(
                f"/bodies/{geom}",
                local.astype(np.float32),
                faces.astype(np.uint32),
                color=color,
                opacity=1.0,
                side="double",
                position=data.xpos[body],
                wxyz=data.xquat[body],
            )
            if is_device:
                handle.on_click(partial(self._pick_event, geom))
            self.meshes[geom] = (body, handle, is_device)
        segments, radii = rendered_tendon_capsules(model, data)
        for radius in np.unique(radii):
            handle = self.server.scene.add_line_segments(
                f"/muscles/{radius}",
                points=segments[radii == radius].astype(np.float32),
                colors=(208, 70, 66),
                thickness=float(radius * 2),
                thickness_units="world",
            )
            self.tendons.append(handle)
        for index, _label in enumerate(self.session.contact_labels):
            color = POINT_COLORS[index % len(POINT_COLORS)]
            self.points.append(self.server.scene.add_icosphere(f"/contacts/{index}", radius=0.003, color=color))
            self.point_lines.append(
                self.server.scene.add_line_segments(
                    f"/contact_lines/{index}", points=np.zeros((1, 2, 3), np.float32), colors=color, thickness=0.001
                )
            )
            self.labels.append(
                self.server.scene.add_label(
                    f"/contact_labels/{index}", text=self.session.contact_labels[index], font_screen_scale=0.8
                )
            )
        for index in range(len(self.adapter.preview_axes()) * 2):
            self.axes.append(
                self.server.scene.add_line_segments(
                    f"/axes/{index}",
                    points=np.zeros((1, 2, 3), np.float32),
                    colors=(60, 220, 100) if index % 2 == 0 else (255, 200, 20),
                    thickness=0.002,
                )
            )

        @self.server.on_client_connect
        def connected(client) -> None:
            client.camera.up_direction = (0, 0, 1)
            client.camera.position = (0.75, -0.85, 0.82)
            client.camera.look_at = (0, 0.15, 0.65)

    def _gui(self) -> None:
        gui = self.server.gui
        gui.configure_theme(titlebar_content=None, dark_mode=True)
        gui.main_panel.dock_right()
        TextBlock(
            gui,
            "安装配合调节\n拖动滑块实时查看；绿色为人体轴，黄色为设备轴。每个绑缚是刚体，多个贴合目标可能相互冲突。",
        )
        self.side = gui.add_dropdown("调节侧", options=self.session.sides, initial_value=self.session.sides[0])
        self.mirror = gui.add_checkbox("同步镜像到另一侧", initial_value=True)
        self.lock_axes = gui.add_checkbox("保持人体 / 电机共轴", initial_value=True)
        self.side.on_update(lambda _: self.queue.put(("side", None)))
        self.lock_axes.on_update(lambda _: self._sliders_event())
        self.mirror.on_update(lambda _: self._sliders_event())
        self.controls = {}
        values = self.session.parameters(self.side.value)
        with gui.add_folder("安装位置与真实关节角度"):
            for control in self.session.controls:
                self.controls[control.key] = gui.add_slider(
                    control.label,
                    min=control.minimum,
                    max=control.maximum,
                    step=control.step,
                    initial_value=float(values[control.key]),
                    disabled=control.advanced,
                )
                self.controls[control.key].on_update(lambda _: self._sliders_event())
        with gui.add_folder("贴合测点与目标间隙"):
            TextBlock(
                gui,
                "彩色点位于设备 CAD 表面。间隙针对该点与同侧红色肌肉显示胶囊；负值表示进入显示几何。完整干涉检查还会检查其他部位、骨骼和手部。",
            )
            self.contact_selected = gui.add_dropdown(
                "重新选点的部位", options=self.session.contact_labels, initial_value=self.session.contact_labels[0]
            )
            self.pick_enabled = gui.add_checkbox("点击设备网格重新放置测点", initial_value=False)
            self.targets, self.enabled = [], []
            for label in self.session.contact_labels:
                with gui.add_folder(label, expand_by_default=False):
                    self.enabled.append(gui.add_checkbox("参与自动贴合", initial_value=True))
                    self.targets.append(gui.add_number("目标间隙 (mm)", initial_value=0.0, step=0.1))
            gui.add_button("按测点目标自动贴合").on_click(lambda _: self.queue.put(("solve", None)))
            gui.add_button("把当前间隙设为目标").on_click(lambda _: self.queue.put(("targets", None)))
        self.readings = TextBlock(gui)
        with gui.add_folder("显示与视角"):
            self.muscles = gui.add_checkbox("显示肌肉", initial_value=True)
            self.muscles.on_update(lambda _: self.queue.put(("display", None)))
            self.bones = gui.add_checkbox("显示骨骼", initial_value=True)
            self.bones.on_update(lambda _: self.queue.put(("display", None)))
            self.opacity = gui.add_slider("设备透明度", min=0.1, max=1, step=0.05, initial_value=1.0)
            self.opacity.on_update(lambda _: self.queue.put(("display", None)))
            view = gui.add_dropdown("视角", options=("斜视", "正面", "侧面"), initial_value="斜视")
            view.on_update(lambda _: self.queue.put(("view", view.value)))
        with gui.add_folder("检查、保存与恢复"):
            gui.add_button("完整干涉检查").on_click(lambda _: self.queue.put(("audit", None)))
            gui.add_button("恢复启动时的安装").on_click(lambda _: self.queue.put(("reset", None)))
            self.output = gui.add_text(
                "草稿路径 / 文件名前缀", initial_value=f"generated/{self.adapter.spec.id}/fitting/draft.json"
            )
            gui.add_button("选择保存目录").on_click(self._choose_directory)
            gui.add_button("加载草稿").on_click(lambda _: self.queue.put(("load", None)))
            gui.add_button("保存草稿与模型").on_click(lambda _: self.queue.put(("save", False)))
            gui.add_button("检查并应用到新训练", color="orange").on_click(lambda _: self.queue.put(("save", True)))
            TextBlock(
                gui,
                "保存自动加时间后缀，保留每个版本。膝轴对齐是生成和应用的硬约束；其他干涉只显示警告。应用会备份原配置；旧 checkpoint 不变。",
            )
        self.status = TextBlock(
            gui,
            "当前显示本地已启用的安装。调整后需要重新应用。"
            if self.session.is_applied()
            else "就绪。当前是静态安装预览，不进行策略推理或动力学步进。",
        )
        self.audit_status = TextBlock(gui)

    def _choose_directory(self, event) -> None:
        if event.client is not None:
            DirectoryPicker(event.client.gui, resolve_path(self.output.value).parent, self._directory_selected)

    def _directory_selected(self, directory: Path) -> None:
        self.output.value = str(directory / Path(self.output.value).name)
        self.status.content = f"保存目录已选择: {directory}。保存时自动添加时间后缀。"

    def _sliders_event(self) -> None:
        if not self._updating_gui:
            self.queue.put(
                (
                    "parameters",
                    (
                        self.side.value,
                        {key: float(handle.value) for key, handle in self.controls.items()},
                        self.mirror.value,
                        self.lock_axes.value,
                    ),
                )
            )

    def _pick_event(self, geom: int, event) -> None:
        if self.pick_enabled.value:
            index = self.session.contact_labels.index(self.contact_selected.value)
            self.queue.put(
                ("pick", (self.side.value, index, geom, np.array(event.ray_origin), np.array(event.ray_direction)))
            )

    def _load_sliders(self) -> None:
        self._updating_gui = True
        try:
            values = self.session.parameters(self.side.value)
            for control in self.session.controls:
                self.controls[control.key].value = float(values[control.key])
                self.controls[control.key].disabled = control.advanced and self.lock_axes.value
        finally:
            self._updating_gui = False

    def _sync(self) -> None:
        model, data = self.session.model, self.session.data
        with self.server.atomic():
            for body, handle, _ in self.meshes.values():
                handle.position = data.xpos[body]
                handle.wxyz = data.xquat[body]
            for index, pair in enumerate(self.adapter.preview_axes()):
                for offset, name in enumerate(pair):
                    joint = model.joint(name).id
                    ends = np.array(
                        [
                            [
                                data.xanchor[joint] - 0.1 * data.xaxis[joint],
                                data.xanchor[joint] + 0.1 * data.xaxis[joint],
                            ]
                        ],
                        dtype=np.float32,
                    )
                    self.axes[index * 2 + offset].points = ends
        self.server.flush()

    def _readouts(self) -> None:
        readings = self.session.contact_readings(self.side.value)
        text = "当前测点间隙\n"
        with self.server.atomic():
            for index, row in enumerate(readings):
                self.points[index].position = row["position"]
                self.labels[index].position = row["position"]
                self.labels[index].text = f"{row['label']} · {row['gap_mm']:.2f} mm"
                self.point_lines[index].points = np.array([[row["position"], row["nearest"]]], dtype=np.float32)
                text += f"{row['label']}：{row['gap_mm']:.3f} mm\n"
        self.readings.content = text

    def _display(self) -> None:
        for _, handle, device in self.meshes.values():
            handle.visible = device or self.bones.value
            if device:
                handle.opacity = float(self.opacity.value)
        for handle in self.tendons:
            handle.visible = self.muscles.value

    def _audit_text(self, report: dict, *, applied: bool = False) -> str:
        text = "完整检查\n"
        if applied:
            text += "当前安装已启用。\n"
        text += "轴线硬约束通过，可以生成和应用。\n" if report["applicable"] else "轴线硬约束未通过，不能生成或应用。\n"
        tolerance = report.get("alignment_tolerance")
        if tolerance:
            text += f"共轴阈值: 轴距 ≤ {tolerance['distance_mm']:.3f} mm，夹角 ≤ {tolerance['angle_deg']:.3f}°\n"
        for failure in report.get("failures", []):
            text += f"- {failure}\n"
        warnings = report.get("warnings", [])
        if warnings:
            text += "软约束警告（不阻止生成或应用）:\n"
            for warning in warnings:
                text += f"- {warning}\n"
        for joint, item in report["alignment"].items():
            text += f"- {joint}：轴距 {item['axis_line_distance_mm']:.4f} mm，夹角 {item['axis_angle_deg']:.4f}°\n"
        for side, links in report["sides"].items():
            for name, item in links.items():
                text += f"- {side} {name}：骨骼间隙 {item['bone_gap_mm']:.3f} mm，交叉数 {item['bone_crossings']}\n"
                if "hand_gap_mm" in item:
                    text += f"  手部 ≥ {item['hand_gap_mm']:.3f} mm，肌肉 ≥ {item['muscle_gap_mm']:.3f} mm\n"
        return text

    def _run_command(self, kind: str, value) -> None:
        if kind in ("parameters", "reset", "solve", "load"):
            self.audit_status.content = "安装参数已改变，请重新运行完整干涉检查。"
        if kind == "parameters":
            self.session.update(*value[:2], mirror=value[2], lock_axes=value[3])
            self._load_sliders()
        elif kind == "side":
            self._load_sliders()
        elif kind == "reset":
            self.session.reset()
            self._load_sliders()
        elif kind == "display":
            self._display()
        elif kind == "pick":
            selected = self.session.pick_contact(*value)
            self.status.content = "测点已放到所选 CAD 表面。" if selected else "请点击所选部位对应的设备部件。"
        elif kind == "targets":
            for handle, row in zip(self.targets, self.session.contact_readings(self.side.value), strict=True):
                handle.value = row["gap_mm"]
        elif kind == "solve":
            self.status.content = "正在按测点目标求解，请稍候。"
            targets = [
                float(h.value) if enabled.value else None for h, enabled in zip(self.targets, self.enabled, strict=True)
            ]
            result = self.session.solve_contacts(
                self.side.value, targets, mirror=self.mirror.value, lock_axes=self.lock_axes.value
            )
            self._load_sliders()
            self.status.content = (
                "拟合完成。各目标残差 (mm)："
                + ", ".join(f"{v:.3f}" for v in result["residual_mm"])
                + "。请运行完整干涉检查。"
            )
        elif kind == "audit":
            self.status.content = "正在检查真实 CAD 表面、肌肉和手部几何，请稍候。"
            report = self.session.audit()
            self.audit_status.content = self._audit_text(report, applied=self.session.is_applied())
            self.status.content = "完整检查完成。"
        elif kind == "save":
            self.status.content = "正在检查并应用安装。" if value else "正在保存草稿与模型。"
            already_applied = bool(value) and self.session.is_applied()
            path = self.session.save(resolve_path(self.output.value), apply=bool(value))
            self.output.value = str(path)
            outcome = "已应用到新训练，原配置已备份" if value else "草稿已保存，默认安装未变"
            if already_applied:
                outcome = "安装位置未变，已启用新保存版本，原配置已备份"
            self.status.content = f"{outcome}：{path}"
            self.audit_status.content = self._audit_text(self.session.last_audit, applied=self.session.is_applied())
            if self.session.last_audit.get("warnings"):
                self.status.content += "\n保存成功，存在软约束警告，请查看检查结果。"
        elif kind == "load":
            self.status.content = "正在加载草稿。"
            self.session.load(resolve_path(self.output.value))
            self._load_sliders()
            self.status.content = "草稿已加载，默认安装未变。"
        elif kind == "view":
            positions = {"斜视": (1.2, -1.7, 1.0), "正面": (0.0, -2.0, 0.8), "侧面": (2.0, 0.15, 0.8)}
            for client in self.server.get_clients().values():
                client.camera.position = positions[value]
                client.camera.look_at = (0.0, 0.15, 0.7)
        self._sync()
        self._readouts()

    def _loop(self) -> None:
        while self.running.is_set():
            try:
                kind, value = self.queue.get(timeout=1 / UI_HZ)
            except Empty:
                continue
            # Coalesce only consecutive slider events; never discard saves or checks.
            if kind == "parameters":
                while not self.queue.empty():
                    next_kind, next_value = self.queue.get_nowait()
                    if next_kind == "parameters":
                        value = next_value
                    else:
                        self._execute(kind, value)
                        kind, value = next_kind, next_value
                        break
            self._execute(kind, value)

    def _execute(self, kind: str, value) -> None:
        try:
            self._run_command(kind, value)
        except Exception as error:
            self.status.content = f"未完成：{error}"
            if kind == "save" and self.session.last_audit is not None:
                self.audit_status.content = self._audit_text(self.session.last_audit, applied=self.session.is_applied())
            print(f"Fitting operation {kind}: {error}", flush=True)

    def close(self) -> None:
        self.running.clear()
        self.thread.join(timeout=2)
        self.server.stop()


def main() -> None:
    """Launch the local fitting UI without loading JAX, a policy or CUDA."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="knee_exo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    app = FittingApp(args.device, args.host, args.port)
    url = f"http://{args.host}:{args.port}"
    print(
        f"Installation editor ready: {url}\nCtrl+C to close. Sliders do not change saved installation until Apply.",
        flush=True,
    )
    if not args.no_browser:
        webbrowser.open(url)
    try:
        while app.running.is_set():
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        app.close()


if __name__ == "__main__":
    main()
