# string_namd/io.py
"""
Template rendering and file I/O utilities using Jinja2.
"""

import pathlib
from typing import Any, Dict
from jinja2 import Environment, FileSystemLoader, select_autoescape


class TemplateIO:
    """
    Handles loading and rendering of Jinja2 templates for NAMD and Colvars files.

    Attributes:
        template_dir: Directory containing .template files.
        env:          Jinja2 Environment configured to load from template_dir.
    """

    def __init__(self, template_dir: pathlib.Path) -> None:
        """
        Args:
            template_dir: Path to directory containing Jinja2 templates.
        """
        self.template_dir = pathlib.Path(template_dir)
        if not self.template_dir.is_dir():
            raise FileNotFoundError(f"Template directory '{self.template_dir}' not found.")
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=select_autoescape(enabled_extensions=('template',)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render(
        self,
        template_name: str,
        output_path: pathlib.Path,
        context: Dict[str, Any],
    ) -> None:
        """
        Render a template with the given context and write to file.

        Args:
            template_name: Name of the template file (e.g. 'namd.template').
            output_path:   Path to write the rendered file (e.g. 'iteration0.conf').
            context:       Dictionary of variables to substitute in the template.
        """
        template = self.env.get_template(template_name)
        content = template.render(**context)

        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)

