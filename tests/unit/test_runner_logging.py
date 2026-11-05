"""Check training metric delivery with and without W&B."""

from unittest.mock import Mock

import pytest
import wandb
from omegaconf import OmegaConf

from musclemimic.runner import engine
from musclemimic.runner.logging import ExperimentHooks


@pytest.mark.parametrize("use_wandb", [False, True])
def test_training_metrics_reach_the_selected_sink(monkeypatch, use_wandb):
    """Keep progress observable while preserving resumed global step counts."""
    logger = Mock()
    cloud_log = Mock()
    monkeypatch.setattr(engine, "logger", logger)
    monkeypatch.setattr(wandb, "log", cloud_log)
    config = OmegaConf.create({"experiment": {"reset_logging_timestep": False}})
    callback = engine.build_logging_callback(
        env=None,
        config=config,
        agent_conf=None,
        use_wandb=use_wandb,
        hooks=ExperimentHooks(),
    )

    callback({
        "mean_episode_return": 12.5,
        "mean_episode_length": 60.0,
        "max_timestep": 8192,
        "jax_raw_timestep": 4096,
        "ppo/value_loss": 0.4,
    })

    if use_wandb:
        cloud_log.assert_called_once()
        emitted = cloud_log.call_args.args[0]
        assert cloud_log.call_args.kwargs["step"] == 8192
        logger.info.assert_not_called()
    else:
        logger.info.assert_called_once()
        emitted = logger.info.call_args.args[0]
        cloud_log.assert_not_called()

    assert emitted["Mean Episode Return"] == 12.5
    assert emitted["Current Timestep"] == 8192
    assert emitted["Raw JAX Timestep"] == 4096
    assert emitted["PPO/value_loss"] == 0.4
