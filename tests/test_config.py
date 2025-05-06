# tests/test_config.py

import pathlib
import json
import yaml
import pytest

from string_namd.config import Config


@pytest.fixture
def tmp_yaml_config(tmp_path):
    data = {
        "num_images": 5,
        "num_swarms": 3,
        "swarm_steps": 10,
        "num_iterations": 2,
        "namd_executable": "namd2",
        "output_dir": "./out",
        "template_dir": "./templates",
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(data))
    return p, data


@pytest.fixture
def tmp_json_config(tmp_path):
    data = {
        "num_images": 8,
        "num_swarms": 4,
        "swarm_steps": 20,
        "num_iterations": 5,
        "namd_executable": "/usr/bin/namd2",
        "output_dir": "/tmp/out",
        "template_dir": "/tmp/templates",
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data))
    return p, data


def test_load_yaml_config(tmp_yaml_config):
    path, data = tmp_yaml_config
    cfg = Config(path)
    assert cfg.num_images == data["num_images"]
    assert cfg.num_swarms == data["num_swarms"]
    assert cfg.swarm_steps == data["swarm_steps"]
    assert cfg.num_iterations == data["num_iterations"]
    assert cfg.namd_executable == data["namd_executable"]
    assert cfg.output_dir == pathlib.Path(data["output_dir"])
    assert cfg.template_dir == pathlib.Path(data["template_dir"])
    assert isinstance(cfg.to_dict(), dict)


def test_load_json_config(tmp_json_config):
    path, data = tmp_json_config
    cfg = Config(path)
    assert cfg.num_images == data["num_images"]
    assert cfg.num_swarms == data["num_swarms"]
    assert cfg.swarm_steps == data["swarm_steps"]
    assert cfg.num_iterations == data["num_iterations"]
    assert cfg.namd_executable == data["namd_executable"]
    assert cfg.output_dir == pathlib.Path(data["output_dir"])
    assert cfg.template_dir == pathlib.Path(data["template_dir"])


def test_missing_required_keys(tmp_path):
    # Write a config with missing keys
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"num_images": 1}))
    with pytest.raises(KeyError) as exc:
        Config(p)
    assert "Missing required config keys" in str(exc.value)


def test_unsupported_format(tmp_path):
    p = tmp_path / "config.txt"
    p.write_text("some random text")
    with pytest.raises(ValueError) as exc:
        Config(p)
    assert "Unsupported config format" in str(exc.value)


def test_file_not_found(tmp_path):
    p = tmp_path / "nonexistent.yaml"
    with pytest.raises(FileNotFoundError):
        Config(p)

