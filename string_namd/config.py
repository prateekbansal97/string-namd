# string_namd/config.py
"""
Configuration loader and validator for the string-namd package.

Loads settings from a YAML or JSON file, validates required fields,
and provides typed property accessors for all parameters.
"""

import pathlib
import json
from typing import Any, Dict, Union

import yaml


class Config:
    """
    Loads and validates user configuration from a YAML or JSON file.

    Required keys in the config data:
      - num_images: int
      - num_swarms: int
      - swarm_steps: int
      - num_iterations: int
      - namd_executable: str
      - output_dir: str or Path
      - template_dir: str or Path
    """

    def __init__(self, path: Union[str, pathlib.Path]) -> None:
        """
        Initialize and parse the config file.

        Args:
            path: Path to a YAML or JSON configuration file.
        """
        self.path = pathlib.Path(path)
        self._data = self._load_file()
        self._validate()

    def _load_file(self) -> Dict[str, Any]:
        """
        Read the config file and parse YAML or JSON.

        Returns:
            A dict with configuration parameters.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If parsing fails or format unsupported.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Config file '{self.path}' not found.")

        text = self.path.read_text()
        suffix = self.path.suffix.lower()

        try:
            if suffix in {'.yaml', '.yml'}:
                return yaml.safe_load(text)
            if suffix == '.json':
                return json.loads(text)
        except Exception as e:
            raise ValueError(f"Error parsing config file: {e}")

        raise ValueError(
            f"Unsupported config format '{suffix}'. Use .yaml, .yml, or .json."
        )

    def _validate(self) -> None:
        """
        Ensure all required configuration keys are present.

        Raises:
            KeyError: If any required key is missing.
        """
        required_keys = {
            'num_images',
            'num_swarms',
            'swarm_steps',
            'num_iterations',
            'namd_executable',
            'output_dir',
            'template_dir',
        }
        missing = required_keys - self._data.keys()
        if missing:
            raise KeyError(f"Missing required config keys: {sorted(missing)}")

    @property
    def num_images(self) -> int:
        """Number of images along the string."""
        return int(self._data['num_images'])

    @property
    def num_swarms(self) -> int:
        """Number of swarm replicas per image."""
        return int(self._data['num_swarms'])

    @property
    def swarm_steps(self) -> int:
        """Number of MD steps per swarm iteration."""
        return int(self._data['swarm_steps'])

    @property
    def num_iterations(self) -> int:
        """Total number of string method iterations."""
        return int(self._data['num_iterations'])

    @property
    def namd_executable(self) -> str:
        """Path or name of the NAMD executable to run."""
        return str(self._data['namd_executable'])

    @property
    def output_dir(self) -> pathlib.Path:
        """Directory where output and logs will be written."""
        return pathlib.Path(self._data['output_dir'])

    @property
    def template_dir(self) -> pathlib.Path:
        """Directory containing NAMD and Colvars Jinja2 templates."""
        return pathlib.Path(self._data['template_dir'])

    def to_dict(self) -> Dict[str, Any]:
        """Return the raw configuration data as a dict."""
        return dict(self._data)
