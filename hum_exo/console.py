"""SB3-style console tables for MuscleMimic's measured PPO metrics."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import BarColumn, Progress, ProgressColumn, Task, TimeElapsedColumn, TimeRemainingColumn
from rich.text import Text
from tqdm.rich import FractionColumn, RateColumn


class BudgetPercentageColumn(ProgressColumn):
    """Reserve 100% for a completed budget rather than rounding up at 99.5%."""

    def render(self, task: Task) -> Text:
        return Text(f"{min(100, int(task.percentage)):>4d}%", style="progress.percentage")


def format_table(groups: dict[str, dict[str, Any]]) -> str:
    """Format grouped metrics as aligned terminal rows."""
    rows = []
    for group, values in sorted(groups.items()):
        rows.append((group + "/", ""))
        for name, value in sorted(values.items()):
            text = f"{value:<8.3g}" if isinstance(value, float) else str(value)
            rows.append(("   " + name, text))
    left = max(len(k) for k, _ in rows)
    right = max(len(v) for _, v in rows)
    rule = "-" * (left + right + 7)
    return "\n".join([rule, *[f"| {k:<{left}} | {v:<{right}} |" for k, v in rows], rule])


class TrainingConsole:
    """Record JSONL metrics and print timing, PPO tables and progress."""

    def __init__(self, config: Any, output: Path, clock: Any = time.monotonic, console: Console | None = None) -> None:
        self.config = config.experiment
        self.output = Path(output) / "training_metrics.jsonl"
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.clock, self.started, self.base_steps = clock, clock(), None
        self.logger = logging.getLogger("hum_exo.training")
        self.terminal = console if console is not None else Console(file=sys.stdout)
        self.progress = Progress(
            BudgetPercentageColumn(),
            BarColumn(bar_width=None),
            FractionColumn(),
            "[",
            TimeElapsedColumn(),
            "<",
            TimeRemainingColumn(),
            ",",
            RateColumn(unit="it"),
            "]",
            console=self.terminal,
            auto_refresh=self.terminal.is_terminal,
        )
        self.task_id = None
        self.last_step = 0
        self.previous_elapsed = 0.0
        self.closed = False
        self.preparing = False
        self.table_log = Path(output) / "training_console.log"

    def prepare(self, initial_step: int = 0) -> None:
        """Show startup status without creating a zero-step training progress bar."""
        if self.preparing or self.task_id is not None:
            return
        self.preparing = True
        self.last_step = initial_step
        self.terminal.print(Text("初始化 / 编译中。首轮训练更新完成后显示进度条。"))

    def start(self, initial_step: int = 0) -> None:
        """Start the same Rich columns used by SB3's progress_bar=True."""
        if self.task_id is not None:
            return
        self.preparing = False
        self.last_step = initial_step
        self.progress.start()
        self.task_id = self.progress.add_task("", total=int(self.config.total_timesteps), completed=initial_step)

    def close(self) -> None:
        """Flush and close once, including exceptions and user interruption."""
        if not self.closed and self.task_id is not None:
            self.progress.stop()
        self.closed = True

    def __call__(self, metrics: dict[str, Any]) -> None:
        if metrics.get("_log_only"):
            self.logger.info("Model parameters: %s", {k: v for k, v in metrics.items() if k.startswith("model/")})
            return
        step = int(metrics["max_timestep"])
        raw = int(metrics.get("jax_raw_timestep", step))
        if raw <= 0:
            return
        self.start(max(0, step - raw))
        if self.base_steps is None:
            self.base_steps = step - raw
        elapsed = max(self.clock() - self.started, 1e-6)
        fps = raw / elapsed
        batch = int(self.config.num_envs * self.config.num_steps)
        values = {
            "rollout": {
                "ep_rew_mean": metrics.get("mean_episode_return", 0.0),
                "ep_len_mean": metrics.get("mean_episode_length", 0.0),
            },
            "time": {
                "fps": int(fps),
                "iterations": step // batch,
                "time_elapsed": int(elapsed),
                "total_timesteps": step,
            },
            "train": {},
        }
        mapping = {
            "kl_divergence": "approx_kl",
            "clipped_ratio_frac": "clip_fraction",
            "explained_variance": "explained_variance",
            "total_loss": "loss",
            "policy_loss": "policy_gradient_loss",
            "policy_std": "std",
            "value_loss": "value_loss",
        }
        for source, target in mapping.items():
            if "ppo/" + source in metrics:
                values["train"][target] = metrics["ppo/" + source]
        if "ppo/entropy" in metrics:
            values["train"]["entropy_loss"] = -metrics["ppo/entropy"]
        values["train"].update(
            learning_rate=metrics.get("learning_rate", 0.0),
            clip_range=self.config.clip_eps,
            clip_range_vf=self.config.clip_eps_vf,
        )
        table = format_table(values)
        # Render a Text object so the table stays literal without disabling
        # Rich markup on the concurrently rendered progress bar.
        self.terminal.print(Text(table), soft_wrap=True)
        with self.table_log.open("a") as stream:
            stream.write(table + "\n")
        self.progress.update(self.task_id, completed=step, refresh=True)
        # Exclude JAX parameter trees supplied for the upstream validation recorder.
        payload = {k: v for k, v in metrics.items() if not k.startswith("_")}
        payload["time/elapsed_seconds"] = elapsed
        payload["time/fps"] = fps
        payload["time/interval_seconds"] = elapsed - self.previous_elapsed
        payload["time/interval_fps"] = (step - self.last_step) / max(elapsed - self.previous_elapsed, 1e-6)
        with self.output.open("a") as stream:
            stream.write(json.dumps(payload) + "\n")
        self.last_step = step
        self.previous_elapsed = elapsed
