# tests/test_optimizer.py

import os
import numpy as np
import pytest

from string_namd.optimizer import StringOptimizer
from string_namd.namd_runner import NamdRunner
from string_namd.io import TemplateIO


class DummyConfig:
    num_images = 2
    num_swarms = 3
    swarm_steps = 5
    num_iterations = 2


def test_initial_string_structure(tmp_path, monkeypatch):
    """
    Test that the initial string coordinates are zero arrays of correct shape.
    """
    # Create dummy templates
    tpl_dir = tmp_path / "templates"
    tpl_dir.mkdir()
    (tpl_dir / "namd.template").write_text("")
    (tpl_dir / "colvars.template").write_text("")

    # Initialize runner and IO
    runner = NamdRunner(namd_executable="namd2", output_dir=tmp_path)
    io = TemplateIO(template_dir=tpl_dir)

    optimizer = StringOptimizer(config=DummyConfig, runner=runner, io=io)
    coords = optimizer._coords
    assert isinstance(coords, np.ndarray)
    assert coords.shape == (DummyConfig.num_images, 3)
    assert np.all(coords == 0)


def test_optimize_dispatches_jobs(tmp_path):
    """
    Test that optimize() dispatches the correct number of jobs for
    equilibration and swarm phases.
    """
    calls = []

    class FakeRunner(NamdRunner):
        def prepare_iteration_dir(self, iteration):
            dir_path = tmp_path / f"iter{iteration:03d}"
            dir_path.mkdir(parents=True, exist_ok=True)
            return dir_path

        def run_job(self, config_path, log_path=None):
            calls.append(config_path.name)

    # Setup templates
    tpl_dir = tmp_path / "templates"
    tpl_dir.mkdir()
    (tpl_dir / "namd.template").write_text("")
    (tpl_dir / "colvars.template").write_text("")

    runner = FakeRunner(namd_executable="namd2", output_dir=tmp_path)
    io = TemplateIO(template_dir=tpl_dir)
    optimizer = StringOptimizer(config=DummyConfig, runner=runner, io=io)

    optimizer.optimize()

    # Expected calls per iteration: num_images equil + num_images*num_swarms swarm
    exp_per_iter = DummyConfig.num_images + DummyConfig.num_images * DummyConfig.num_swarms
    assert len(calls) == DummyConfig.num_iterations * exp_per_iter
    # Verify naming patterns
    assert any(name.startswith("eq_img0") for name in calls)
    assert any(name.startswith("swarm_img1_rep") for name in calls)


def test_update_string_writes_file(tmp_path):
    """
    Test that _update_string aggregates replica endpoints and writes coords file.
    """
    # Prepare fake iteration directory and dummy traj files
    iter_dir = tmp_path / "iter001"
    iter_dir.mkdir(parents=True, exist_ok = True)
    tpl_dir = tmp_path / "templates"
    tpl_dir.mkdir()
    (tpl_dir / "colvars.template").write_text("")
    (tpl_dir / "namd.template").write_text("")
    # Create dummy trajectory files
    for img in range(DummyConfig.num_images):
        for rep in range(DummyConfig.num_swarms):
            f = iter_dir / f"swarm_img{img}_rep{rep}_traj.dcd"
            f.write_text("")
    #for rep in range(DummyConfig.num_swarms):
    #    f = iter_dir / f"swarm_img0_rep{rep}_traj.dcd"
    #    f.write_text("")

    # Subclass optimizer to override extraction logic
    class TestOptimizer(StringOptimizer):
        def _load_initial_string(self):
            return np.zeros((self.config.num_images, 3))

        def _extract_coords(self, traj_path):
            # return coords based on replica index
            idx = int(str(traj_path).split("rep")[1].split("_")[0])
            return np.array([idx, idx, idx])

    # Initialize and run update
    runner = NamdRunner(namd_executable="namd2", output_dir=tmp_path)
    io = TemplateIO(template_dir=tpl_dir)
    optimizer = TestOptimizer(config=DummyConfig, runner=runner, io=io)
    optimizer._coords = np.zeros((DummyConfig.num_images, 3))
    optimizer._update_string(iteration=1)

    coords_file = tmp_path / "iter001" / "string_coords.txt"
    assert coords_file.exists()
    data = np.loadtxt(coords_file)
    assert data.shape == (2, 3)
    assert np.allclose(data[0], [1.0, 1.0, 1.0])
    assert np.allclose(data[1], [1.0, 1.0, 1.0])
