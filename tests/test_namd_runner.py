# tests/test_namd_runner.py

import subprocess
import pytest
from pathlib import Path

from string_namd.namd_runner import NamdRunner


def test_prepare_iteration_dir(tmp_path):
    """
    Ensure that prepare_iteration_dir creates and returns the correct directory.
    """
    runner = NamdRunner(namd_executable="namd2", output_dir=tmp_path)
    iter_path = runner.prepare_iteration_dir(iteration=5)
    expected = tmp_path / "iter005"
    assert iter_path == expected
    assert expected.exists() and expected.is_dir()


def test_run_job_without_log(tmp_path, monkeypatch):
    """
    Test that run_job invokes subprocess.run with correct arguments when no log_path.
    """
    config_file = tmp_path / "test.conf"
    config_file.write_text("dummy")

    calls = []

    def fake_run(cmd, check, stdout=None, stderr=None):
        calls.append(cmd)
        assert cmd == ["namd2", str(config_file)]

    monkeypatch.setattr(subprocess, 'run', fake_run)

    runner = NamdRunner(namd_executable="namd2", output_dir=tmp_path)
    runner.run_job(config_path=config_file)
    assert calls, "subprocess.run was not called"


def test_run_job_with_log(tmp_path, monkeypatch):
    """
    Test that run_job writes output to the specified log file.
    """
    config_file = tmp_path / "test.conf"
    config_file.write_text("dummy")
    log_file = tmp_path / "test.log"

    class DummyFile:
        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): pass
        def write(self, data): pass

    calls = []

    def fake_open(path, mode):  # noqa
        calls.append(path)
        return DummyFile()

    def fake_run(cmd, stdout, stderr, check):  # noqa
        calls.append(cmd)
        assert cmd == ["namd2", str(config_file)]

    monkeypatch.setattr('builtins.open', fake_open)
    monkeypatch.setattr(subprocess, 'run', fake_run)

    runner = NamdRunner(namd_executable="namd2", output_dir=tmp_path)
    runner.run_job(config_path=config_file, log_path=log_file)
    # Check that open was called on log_file and subprocess.run was invoked
    assert str(log_file) in [str(c) for c in calls]


def test_run_parallel(tmp_path, monkeypatch):
    """
    Test that run_parallel launches multiple jobs up to max_workers.
    """
    config_files = []
    for i in range(3):
        f = tmp_path / f"job{i}.conf"
        f.write_text("dummy")
        config_files.append(f)

    calls = []

    def fake_run(cmd, stdout=None, stderr=None, check=None):
        # Append only command to calls
        calls.append(cmd)

    monkeypatch.setattr(subprocess, 'run', fake_run)

    runner = NamdRunner(namd_executable="namd2", output_dir=tmp_path)
    runner.run_parallel(config_paths=config_files, max_workers=2)

    # Ensure each .conf was scheduled (order may vary)
    cmd_lists = [["namd2", str(f)] for f in config_files]
    for expected in cmd_lists:
        assert expected in calls

