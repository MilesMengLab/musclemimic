"""Train a human and manifest-selected device using the upstream PPO."""

from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path

import hydra
import jax
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.console import Console

import hum_exo.reward
from hum_exo.checkpoint import load_checkpoint
from hum_exo.configuration import register_resolvers
from hum_exo.console import TrainingConsole
from hum_exo.device import load_device
from hum_exo.environment import register
from hum_exo.logging import checkpoint_console_logging
from hum_exo.paths import copy_model, portable_path, project_name, resolve_path
from hum_exo.policy import HumanExoPPO
from musclemimic.algorithms.common.checkpoint_hooks import create_jax_checkpoint_host_callback
from musclemimic.runner import engine
from musclemimic.runner.checkpointing import resume_or_fresh

register_resolvers()


def training_arguments(arguments: list[str]) -> list[str]:
    """Translate the CLI's --headless switch into a Hydra viewer override."""
    headless = "--headless" in arguments
    result = [arg for arg in arguments if arg != "--headless"]
    prefix = "experiment.render.enabled="
    explicit = any(arg.startswith(prefix) for arg in result)
    if headless:
        result = [arg for arg in result if not arg.startswith(prefix)]
    if headless or not explicit:
        # Keep overrides in their original positional group for Hydra's parser.
        position = next((i for i, arg in enumerate(result) if "=" in arg or arg.startswith("~")), len(result))
        result.insert(position, prefix + ("false" if headless else "true"))
    return result


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(config: DictConfig) -> None:
    """Configure this job's console while keeping full Hydra file logging."""
    with checkpoint_console_logging() as terminal:
        run_training(config, terminal)


def run_training(config: DictConfig, terminal: Console) -> None:
    """Construct the device task, persist configuration, then run joint PPO."""
    register()
    engine.setup_jax_cache()
    output = Path(HydraConfig.get().runtime.output_dir)
    device = load_device(config.device)
    if (output / "session_config.yaml").exists():
        raise FileExistsError(f"Choose a fresh run directory instead of overwriting an experiment: {output}")
    resume = config.experiment.get("resume_from")
    if resume:
        previous, _, metadata = load_checkpoint(resume)
        if previous.experiment.env_params.device_id != device.spec.id:
            raise ValueError("Resume device must match the checkpoint's device ID")
        with open_dict(config.experiment):
            config.experiment.resume_from = str(resolve_path(resume))
        if metadata.global_timestep >= config.experiment.total_timesteps:
            raise ValueError("total_timesteps is an absolute budget and must exceed the saved global timestep")
        if (
            previous.experiment.actor_hidden_layers != config.experiment.actor_hidden_layers
            or (previous.experiment.exo_hidden_layers != config.experiment.exo_hidden_layers)
            or previous.experiment.critic_hidden_layers != config.experiment.critic_hidden_layers
        ):
            raise ValueError("Resume requires the checkpoint's human, device and critic network sizes")
    model_path = output / "human_exo.xml"
    if resume:
        # A resumed optimizer must continue with the saved physics, even when
        # the installation builder or source CAD has changed since that run.
        saved_model = Path(previous.experiment.env_params.device_model)
        if not saved_model.is_file():
            raise FileNotFoundError(f"Resume requires the original experiment model: {saved_model}")
        copy_model(saved_model, model_path)
        if saved_model.with_suffix(".json").is_file():
            shutil.copy2(saved_model.with_suffix(".json"), model_path.with_suffix(".json"))
    else:
        model_path = device.build_model(model_path)
    with open_dict(config.experiment.env_params):
        config.experiment.env_params.device_model = portable_path(model_path)
    env = engine.instantiate_env(config)
    # Persist effective solver options so CPU playback and later resumes use
    # the same numerical settings as GPU training, rather than XML defaults.
    with open_dict(config.experiment.env_params):
        config.experiment.env_params.model_option_conf = {
            "iterations": int(env._model.opt.iterations),
            "ls_iterations": int(env._model.opt.ls_iterations),
            "disableflags": int(env._model.opt.disableflags),
        }
    val_env = engine.instantiate_validation_env(config)
    (output / "reference_fit.json").write_text(json.dumps(env.reference_fit_report, indent=2) + "\n")
    agent = HumanExoPPO.init_agent_conf(env, config)
    with open_dict(config.experiment):
        config.experiment.checkpoint_dir = str(output / "checkpoints" / engine.generate_run_suffix())
    snapshot = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    snapshot.experiment.env_params.device_model = "run://human_exo.xml"
    snapshot.experiment.checkpoint_dir = (
        "run://" + Path(config.experiment.checkpoint_dir).relative_to(output).as_posix()
    )
    OmegaConf.save(snapshot, output / "session_config.yaml")
    shutil.copy2(device.spec.asset_root / "device.json", output / "device.json")
    report = {
        "human_actions": list(agent.network.human_actions),
        "exo_actions": list(agent.network.exo_actions),
        "exo_input_indices": list(agent.network.exo_indices),
        "human_observation_dim": len(agent.network.human_indices),
        "exo_observation_dim": len(device.spec.input_names),
        "device_joints": len(device.spec.joints),
        "device_id": device.spec.id,
        "inputs": device.spec.input_names,
        "input_units": device.spec.input_units,
        "outputs": device.spec.actuators,
        "control_dt_s": env.dt,
        "model_package": "musclemimic_models",
    }
    (output / "interface.json").write_text(json.dumps(report, indent=2) + "\n")
    console = TrainingConsole(config, output, console=terminal)
    metrics = engine.build_metrics_handler(config, val_env)
    train_fn = resume_or_fresh(
        env,
        agent,
        HumanExoPPO,
        config,
        metrics,
        console,
        logging_interval=1,
        val_env=val_env,
        apply_resume_resets=False,
    )
    logging.getLogger("hum_exo").info(
        "Project %s, device %s. First update compiles JAX/Warp; outputs: %s", project_name(), device.spec.id, output
    )
    initial_step = int(metadata.global_timestep) if resume else 0
    console.prepare(initial_step)
    training_view = None
    try:
        if config.experiment.render.enabled:
            from hum_exo.training_view import TrainingView

            count = int(config.experiment.render.num_envs) or int(config.experiment.num_envs)
            if count < 1 or count > config.experiment.num_envs:
                raise ValueError("render.num_envs must be between 1 and the training environment count, or 0 for all")
            training_view = TrainingView(
                env._model,
                output / "rendering",
                count,
                fps=float(config.experiment.render.fps),
                spacing=float(config.experiment.render.spacing_m),
            )
            env.training_state_callback = training_view
        result = engine.run_training(train_fn, engine.compute_training_rngs(config))
        jax.block_until_ready(result)
        # Wait for the final metrics callback before closing its progress bar.
        jax.effects_barrier()
        console.close()
        print("learning done!", flush=True)
    finally:
        console.close()
        if training_view is not None:
            training_view.close()
        cached = getattr(create_jax_checkpoint_host_callback, "__cached_instance__", None)
        if cached is not None:
            cached[2].close()
            delattr(create_jax_checkpoint_host_callback, "__cached_instance__")


def cli() -> None:
    sys.argv[1:] = training_arguments(sys.argv[1:])
    main()


if __name__ == "__main__":
    cli()
