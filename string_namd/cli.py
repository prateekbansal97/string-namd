# string_namd/cli.py
"""
Command-line interface for the string-namd package.
Provides a single `optimize` command to run the string method workflow.
"""

import argparse
import sys
from pathlib import Path

from .config import Config
from .namd_runner import NamdRunner
from .optimizer import StringOptimizer


def main():
    """
    Entry point for the CLI.

    Reads the configuration file, initializes components, and starts optimization.

    Usage:
        string-namd optimize --config path/to/config.yaml
    """
    parser = argparse.ArgumentParser(
        prog="string-namd",
        description="Run the string method with swarm-of-trajectories in NAMD",
    )
    subparsers = parser.add_subparsers(
        title="Commands",
        dest="command",
        required=True,
    )

    optimize_parser = subparsers.add_parser(
        "optimize", help="Run the string method optimization"
    )
    optimize_parser.add_argument(
        "--config", "-c",
        type=Path,
        required=True,
        help="Path to a YAML or JSON configuration file",
    )

    args = parser.parse_args()

    if args.command == "optimize":
        _run_optimize(args.config)
    else:
        parser.print_help()
        sys.exit(1)


def _run_optimize(config_path: Path) -> None:
    """
    Load configuration and execute the string optimization loop.
    """
    # Load and validate config
    config = Config(config_path)

    # Ensure output directory exists
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize NAMD runner and optimizer
    runner = NamdRunner(
        namd_executable=config.namd_executable,
        output_dir=output_dir,
        template_dir=config.template_dir,
    )
    optimizer = StringOptimizer(config=config, runner=runner)

    # Start optimization
    optimizer.optimize()
