# string_namd/__init__.py
"""
string_namd

A package to run the string method with swarm-of-trajectories in NAMD.
"""

__version__ = "0.1.0"

from pathlib import Path

from .config import Config
from .namd_runner import NamdRunner
from .optimizer import StringOptimizer
from .cli import main as _cli_entry

__all__ = [
    "Config",
    "NamdRunner",
    "StringOptimizer",
]

