# string_namd/optimizer.py
"""
Core string-method algorithm implementation.
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np

from .config import Config
from .namd_runner import NamdRunner
from .io import TemplateIO


class StringOptimizer:
    """
    Orchestrates the string method optimization loop:
      1. Equilibration phase
      2. Swarm sampling phase
      3. String update phase

    Attributes:
        config:     User configuration parameters
        runner:     NamdRunner instance to dispatch NAMD jobs
        io:         TemplateIO for rendering input files
    """

    def __init__(
        self,
        config: Config,
        runner: NamdRunner,
        io: TemplateIO,
    ) -> None:
        """
        Args:
            config:   Loaded Config instance
            runner:   NamdRunner to launch jobs
            io:       TemplateIO to render templates
        """
        self.config = config
        self.runner = runner
        self.io = io
        # current string coordinates: shape (num_images, dims)
        self._coords = self._load_initial_string()

    def optimize(self) -> None:
        """
        Perform the full string optimization for the configured
        number of iterations.
        """
        for iteration in range(1, self.config.num_iterations + 1):
            print(f"[Iteration {iteration}/{self.config.num_iterations}]")
            self._equilibration_phase(iteration)
            self._swarm_phase(iteration)
            self._update_string(iteration)
        print("String optimization complete.")

    def _load_initial_string(self) -> np.ndarray:
        """
        Load or initialize the coordinates for the string images.

        Returns:
            A NumPy array of shape (num_images, 3).
        """
        # TODO: replace with real loading or interpolation
        return np.zeros((self.config.num_images, 3))

    def _equilibration_phase(self, iteration: int) -> None:
        """
        Render and launch one equilibration job per image.
        """
        iter_dir = self.runner.prepare_iteration_dir(iteration)
        for img in range(self.config.num_images):
            out_conf = iter_dir / f"eq_img{img}.conf"
            self.io.render(
                template_name="namd.template",
                output_path=out_conf,
                context={
                    "iteration": iteration,
                    "image_idx": img,
                    "coords": self._coords[img],
                },
            )
            self.runner.run_job(out_conf)

    def _swarm_phase(self, iteration: int) -> None:
        """
        Render and launch swarm sampling jobs for each image & replica.
        """
        iter_dir = self.runner.prepare_iteration_dir(iteration)
        for img in range(self.config.num_images):
            for rep in range(self.config.num_swarms):
                out_conf = iter_dir / f"swarm_img{img}_rep{rep}.conf"
                self.io.render(
                    template_name="colvars.template",
                    output_path=out_conf,
                    context={
                        "iteration": iteration,
                        "image_idx": img,
                        "replica_idx": rep,
                        "steps": self.config.swarm_steps,
                        "coords": self._coords[img],
                    },
                )
                self.runner.run_job(out_conf)

    def _update_string(self, iteration: int) -> None:
        """
        Post-process swarm results to update the string coordinates.
        Saves updated coords to `string_coords.txt`.
        """
        iter_dir = self.runner.output_dir / f"iter{iteration}"
        new_coords = []
        for img in range(self.config.num_images):
            trajs = list(iter_dir.glob(f"swarm_img{img}_rep*_traj.dcd"))
            # TODO: integrate with MDTraj/MDAnalysis to extract final coords
            rep_coords = [self._extract_coords(p) for p in trajs]
            new_coords.append(np.mean(rep_coords, axis=0))
        self._coords = np.vstack(new_coords)
        np.savetxt(
            iter_dir / "string_coords.txt",
            self._coords,
            header="x y z",
        )

    def _extract_coords(self, traj_path: Path) -> np.ndarray:
        """
        Load a trajectory and return the final-frame coordinates.
        Placeholder: integrate with MDTraj or MDAnalysis.

        Args:
            traj_path: Path to a DCD or similar file.

        Returns:
            1D NumPy array of length 3.
        """
        return np.zeros(3)

