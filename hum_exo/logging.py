"""Keep checkpoint internals in log files and concise notices in the terminal."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager

from rich.console import Console
from rich.text import Text

CHECKPOINT_HOOK_LOGGER = "musclemimic.algorithms.common.checkpoint_hooks"
CHECKPOINT_LOGGER_PREFIXES = ("absl", "orbax", "musclemimic.algorithms.common.checkpoint")


class _CheckpointInfoFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        return not any(
            record.name == prefix
            or record.name.startswith(prefix + ".")
            # checkpoint_hooks/checkpoint_manager share a module-name stem.
            or (prefix.endswith(".checkpoint") and record.name.startswith(prefix + "_"))
            for prefix in CHECKPOINT_LOGGER_PREFIXES
        )


class _CheckpointNoticeHandler(logging.Handler):
    def __init__(self, console: Console) -> None:
        super().__init__(logging.INFO)
        self.console = console

    def emit(self, record: logging.LogRecord) -> None:
        event = getattr(record, "checkpoint_event", None)
        if record.name != CHECKPOINT_HOOK_LOGGER or event not in ("saved", "queued"):
            return
        try:
            label = "模型已保存" if event == "saved" else "已提交模型保存"
            message = f"{label}: {record.checkpoint_path} (累计 {record.checkpoint_timestep:,} 步)"
            self.console.print(Text(message), soft_wrap=True)
        except Exception:
            self.handleError(record)


@contextmanager
def checkpoint_console_logging(console: Console | None = None) -> Iterator[Console]:
    """Quiet only terminal INFO during this job; keep file logs and warnings."""
    console = console if console is not None else Console(file=sys.stdout)
    root = logging.getLogger()
    quiet = _CheckpointInfoFilter()
    handlers = [
        handler
        for handler in root.handlers
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
    ]
    notice = _CheckpointNoticeHandler(console)
    for handler in handlers:
        handler.addFilter(quiet)
    root.addHandler(notice)
    try:
        yield console
    finally:
        root.removeHandler(notice)
        notice.close()
        for handler in handlers:
            handler.removeFilter(quiet)
