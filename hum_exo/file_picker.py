"""A local directory chooser displayed in the Viser browser interface."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import viser

from hum_exo.paths import resolve_path

DIRECTORY_PLACEHOLDER = "选择子目录"


class DirectoryPicker:
    """Browse the server filesystem without requiring a desktop GUI toolkit."""

    def __init__(self, gui: viser.GuiApi, directory: Path, on_selected: Callable[[Path], None]) -> None:
        self.on_selected = on_selected
        self.current = directory.resolve()
        while not self.current.is_dir() and self.current != self.current.parent:
            self.current = self.current.parent
        self.modal = gui.add_modal("选择保存目录")
        with self.modal:
            self.path = gui.add_text("目录路径", initial_value=str(self.current))
            gui.add_button("进入路径").on_click(lambda _: self.navigate(resolve_path(self.path.value)))
            gui.add_button("上一级目录").on_click(lambda _: self.navigate(self.current.parent))
            self.children = gui.add_dropdown("子目录", options=(DIRECTORY_PLACEHOLDER,))
            gui.add_button("进入所选子目录").on_click(lambda _: self._enter_child())
            self.new_folder = gui.add_text("新文件夹名称", initial_value="")
            gui.add_button("创建文件夹").on_click(lambda _: self._create_folder())
            self.status = gui.add_text(
                "目录提示", initial_value="选择运行脚本所在机器上的目录。", multiline=True, disabled=True
            )
            gui.add_button("使用此目录", color="green").on_click(lambda _: self._choose())
            gui.add_button("取消").on_click(lambda _: self.modal.close())
        self.navigate(self.current)

    def navigate(self, directory: Path) -> None:
        """List child directories and preserve the last selection on errors."""
        try:
            directory = directory.resolve()
            if not directory.is_dir():
                raise ValueError("目录不存在或不可访问")
            entries = sorted((p.name for p in directory.iterdir() if p.is_dir()), key=str.casefold)
            self.current = directory
            self.path.value = str(directory)
            self.children.options = (DIRECTORY_PLACEHOLDER, *entries)
            self.children.value = DIRECTORY_PLACEHOLDER
            self.status.value = f"当前目录: {directory}\n可以进入子目录或直接使用此目录。"
        except (OSError, ValueError) as error:
            self.status.value = f"无法打开目录: {error}"

    def _enter_child(self) -> None:
        if self.children.value != DIRECTORY_PLACEHOLDER:
            self.navigate(self.current / self.children.value)

    def _create_folder(self) -> None:
        name = self.new_folder.value.strip()
        if not name or name in (".", "..") or Path(name).name != name:
            self.status.value = "请输入单个文件夹名称。完整路径请填入目录路径。"
            return
        try:
            directory = self.current / name
            directory.mkdir()
            self.new_folder.value = ""
            self.navigate(directory)
        except OSError as error:
            self.status.value = f"无法创建文件夹: {error}"

    def _choose(self) -> None:
        self.on_selected(self.current)
        self.modal.close()
