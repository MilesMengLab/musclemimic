"""Export a device actor and sensor schema from a joint training checkpoint."""

import argparse
from pathlib import Path

from hum_exo.export import export


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(export(args.checkpoint, args.output))


if __name__ == "__main__":
    main()
