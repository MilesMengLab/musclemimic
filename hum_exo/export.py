"""Export device actor weights and the sensor contract from a joint checkpoint."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from hum_exo.checkpoint import load_checkpoint
from hum_exo.device import load_device
from hum_exo.paths import resolve_model


def export(checkpoint: str, output: Path | None = None) -> Path:
    """Export the restricted tanh MLP and frozen per-channel normalization."""
    config, state, metadata = load_checkpoint(checkpoint)
    device = load_device(config.experiment.env_params.device_id)
    params = state.train_state.params
    if "actor_exo" not in params:
        raise ValueError("Checkpoint does not contain a device actor")
    # Obtain the saved input mapping from the run that produced this checkpoint.
    model = Path(config.experiment.env_params.device_model)
    schema = json.loads((model.parent / "interface.json").read_text())
    ids = np.array(schema["exo_input_indices"], dtype=int)
    if len(ids) != len(device.spec.input_names):
        raise ValueError("Device manifest input dimension differs from the checkpoint")
    statistics = state.train_state.run_stats["ObservationNorm_0"]
    layers = sorted(params["actor_exo"], key=lambda key: int(key.rsplit("_", 1)[1]))
    payload = {
        "mean": np.asarray(statistics["mean"])[ids],
        "var": np.asarray(statistics["var"])[ids],
        "layers": np.array(len(layers)),
    }
    for i, name in enumerate(layers):
        payload[f"kernel_{i}"] = np.asarray(params["actor_exo"][name]["kernel"])
        payload[f"bias_{i}"] = np.asarray(params["actor_exo"][name]["bias"])
    output = Path(output) if output else device.spec.generated_dir / "export"
    output.mkdir(parents=True, exist_ok=True)
    target = output / "device_actor.npz"
    np.savez(target, **payload)
    compiled = mujoco.MjModel.from_xml_path(str(resolve_model(model)))
    scales, units = [], []
    for name in device.spec.actuators:
        motor = compiled.actuator(name).id
        if compiled.actuator_trntype[motor] != mujoco.mjtTrn.mjTRN_JOINT:
            raise ValueError("Export requires scalar joint motors; add a device-specific transmission exporter")
        joint = compiled.actuator_trnid[motor, 0]
        scales.append(float(compiled.actuator_gear[motor, 0] * compiled.actuator_gainprm[motor, 0]))
        units.append("N" if compiled.jnt_type[joint] == mujoco.mjtJoint.mjJNT_SLIDE else "N*m")
    report = {
        "device_id": device.spec.id,
        "inputs": schema.get("inputs") if isinstance(schema.get("inputs"), list) else device.spec.input_names,
        "units": schema.get("input_units", device.spec.input_units),
        "outputs": schema.get("outputs", device.spec.actuators),
        "output_range": [-1, 1],
        "normalization": "(x-mean)/sqrt(var+1e-8)",
        "control_dt_s": schema.get(
            "control_dt_s",
            config.experiment.env_params.get("timestep", 0.002) * config.experiment.env_params.get("n_substeps", 5),
        ),
        "generalized_force_per_command": scales,
        "generalized_force_units": units,
        "checkpoint": checkpoint,
        "global_timestep": metadata.global_timestep,
        "assumptions": "Input units, frames and output order follow the saved device contract",
    }
    (output / "input_schema.json").write_text(json.dumps(report, indent=2) + "\n")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(export(args.checkpoint, args.output))


if __name__ == "__main__":
    main()
