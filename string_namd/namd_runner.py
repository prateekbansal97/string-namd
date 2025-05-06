# string_namd/namd_runner.py
"""
Handles preparation and launching of NAMD jobs based on rendered templates.
"""

import subprocess
from pathlib import Path
from typing import Optional, List


class NamdRunner:
    """
    Runs NAMD jobs by invoking the NAMD executable on configuration files.

    Attributes:
        namd_executable: Path or name of the NAMD binary.
        output_dir:      Base output directory for all iterations.
        template_dir:    Directory containing Jinja2 templates (for reference).
    """

    def __init__(
        self,
        namd_executable: str,
        output_dir: Path,
        template_dir: Optional[Path] = None,
    ) -> None:
        """
        Args:
            namd_executable: Path/name of the NAMD executable.
            output_dir:      Base directory where iteration subfolders will be created.
            template_dir:    (Optional) Directory with templates, for error messages.
        """
        self.namd_executable = namd_executable
        self.output_dir = output_dir
        self.template_dir = template_dir

    def prepare_iteration_dir(self, iteration: int) -> Path:
        """
        Create and return the directory for a given iteration.

        Args:
            iteration: 1-based iteration index.

        Returns:
            Path to the iteration directory.
        """
        iter_dir = self.output_dir / f"iter{iteration:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        return iter_dir

    def run_job(self, config_path: Path, log_path: Optional[Path] = None) -> None:
        """
        Launch a single NAMD job.

        Args:
            config_path: Path to the .conf (Tcl) file.
            log_path:    Optional path to redirect stdout/stderr.
        """
        cmd = [self.namd_executable, str(config_path)]
        if log_path:
            with open(log_path, "w") as log_file:
                subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, check=True)
        else:
            subprocess.run(cmd, check=True)

    def run_parallel(
        self,
        config_paths: List[Path],
        max_workers: int = 4,
    ) -> None:
        """
        Launch multiple NAMD jobs in parallel, up to max_workers at once.

        Args:
            config_paths: List of .conf file paths to run.
            max_workers:  Maximum number of concurrent jobs.
        """
        import concurrent.futures

        def _launch(path: Path) -> None:
            log_file = path.with_suffix(".log")
            self.run_job(path, log_path=log_file)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(_launch, config_paths)
