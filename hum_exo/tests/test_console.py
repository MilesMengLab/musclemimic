"""Console startup must wait for actual completed training updates."""

import json
from io import StringIO

from omegaconf import OmegaConf
from rich.console import Console

from hum_exo.console import TrainingConsole


def test_preparation_and_zero_step_metrics_do_not_display_progress(tmp_path):
    config = OmegaConf.create(
        {"experiment": {"num_envs": 2, "num_steps": 4, "total_timesteps": 2000, "clip_eps": 0.2, "clip_eps_vf": 0.2}}
    )
    stream = StringIO()
    terminal = Console(file=stream, force_terminal=True, color_system=None, width=140)
    clock = iter([10.0, 12.0])
    logger = TrainingConsole(config, tmp_path, clock=lambda: next(clock), console=terminal)
    try:
        logger.prepare(initial_step=1000)
        logger.prepare(initial_step=1000)
        logger({"_log_only": True, "model/param_count": 100, "max_timestep": 0, "jax_raw_timestep": 0})
        logger({"max_timestep": 1000, "jax_raw_timestep": 0})
        assert stream.getvalue().count("初始化 / 编译中") == 1
        assert logger.task_id is None and not logger.progress.live.is_started
        assert "━" not in stream.getvalue()
        assert not (tmp_path / "training_metrics.jsonl").exists()
        logger({"max_timestep": 1008, "jax_raw_timestep": 8})
        assert logger.progress.live.is_started
        assert logger.progress.tasks[0].completed == 1008
    finally:
        logger.close()
    assert "1,008/2,000" in stream.getvalue()
    metrics = json.loads((tmp_path / "training_metrics.jsonl").read_text())
    assert metrics["time/elapsed_seconds"] == 2
    assert metrics["time/interval_fps"] == 4


def test_interrupted_preparation_leaves_no_zero_step_progress(tmp_path):
    config = OmegaConf.create({"experiment": {"total_timesteps": 30_000_000}})
    stream = StringIO()
    logger = TrainingConsole(config, tmp_path, console=Console(file=stream, force_terminal=True, color_system=None))
    logger.prepare()
    logger.close()
    logger.close()
    assert logger.closed
    assert not logger.progress.live.is_started
    assert "初始化 / 编译中" in stream.getvalue()
    assert "0/30,000,000" not in stream.getvalue()
    assert "━" not in stream.getvalue()
