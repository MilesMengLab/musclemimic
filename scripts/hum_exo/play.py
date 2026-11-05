"""Preview a mounted model or play a human-device checkpoint in MuJoCo."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from omegaconf import open_dict

import hum_exo.reward
from hum_exo.checkpoint import load_checkpoint, resolve_checkpoint
from hum_exo.configuration import compose_config
from hum_exo.device import load_device
from hum_exo.environment import register
from hum_exo.evaluation import evaluate


def main() -> None:
    """Inspect prescribed installation motion or simulate a trained policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", type=str)
    source.add_argument("--preview", action="store_true")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--motor-off", action="store_true", help="Apply zero motor command for an evaluation comparison"
    )
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--device", default="knee_exo", help="Device for reference preview or latest checkpoint selection"
    )
    args = parser.parse_args()
    if args.seconds <= 0 or not np.isfinite(args.seconds):
        parser.error("--seconds must be positive and finite")
    register()
    state = None
    if args.preview:
        print(
            "Reference-motion preview includes fitted passive-joint motion. Use ./scripts/hum_exo.sh preview for neutral installation and joint-axis measurements."
        )
    if not args.preview:
        args.checkpoint = str(resolve_checkpoint(args.checkpoint, device_id=args.device))
        print(f"Playing checkpoint: {args.checkpoint}", flush=True)
        config, state, _ = load_checkpoint(args.checkpoint)
    else:
        config = compose_config(args.device)
        device = load_device(args.device)
        with open_dict(config.experiment.env_params):
            config.experiment.env_params.device_model = str(device.build_model())
    result = evaluate(
        config,
        state,
        seconds=args.seconds,
        seed=args.seed,
        preview=args.preview,
        motor_off=args.motor_off,
        show_viewer=not args.no_viewer,
        output=args.output,
    )
    print(f"Saved {result['steps']} steps; {result['episode_resets']} episode resets: {result['output']}")


if __name__ == "__main__":
    main()
