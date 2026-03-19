from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def run_subprocess(script: str, extra_args: list[str]) -> int:
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    cmd = [sys.executable, str(PROJECT_ROOT / script), *extra_args]
    return subprocess.call(cmd, cwd=PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified entrypoint for training and CARLA inference. "
        "If no subcommand is provided, CARLA run mode is used."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the LSTM model.")
    train_parser.add_argument("args", nargs=argparse.REMAINDER)

    run_parser = subparsers.add_parser("run", help="Run CARLA with the trained LSTM controller.")
    run_parser.add_argument("args", nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        return argparse.Namespace(command="run", args=[])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command == "train":
        raise SystemExit(run_subprocess("train_lstm.py", args.args))
    if args.command == "run":
        raise SystemExit(run_subprocess("run_carla_controller.py", args.args))
