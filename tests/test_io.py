# tests/test_io.py

import pytest
from pathlib import Path
from string_namd.io import TemplateIO


def test_template_dir_not_found(tmp_path):
    """
    Ensure TemplateIO raises FileNotFoundError if template_dir is invalid.
    """
    with pytest.raises(FileNotFoundError):
        TemplateIO(tmp_path / "nonexistent_dir")


def test_render_creates_file(tmp_path):
    """
    Test that render() writes rendered content to output_path with correct context.
    """
    # Setup: create a simple template
    tpl_dir = tmp_path / "templates"
    tpl_dir.mkdir()
    tpl_file = tpl_dir / "test.template"
    tpl_file.write_text("Value: {{ value }}")

    io = TemplateIO(template_dir=tpl_dir)
    output_file = tmp_path / "out.txt"
    io.render(
        template_name="test.template",
        output_path=output_file,
        context={"value": 42},
    )

    assert output_file.exists()
    content = output_file.read_text()
    assert content == "Value: 42"


def test_render_creates_parent_dirs(tmp_path):
    """
    Test that render() creates parent directories for output_path if needed.
    """
    tpl_dir = tmp_path / "templates"
    tpl_dir.mkdir()
    (tpl_dir / "test.template").write_text("Hello {{ name }}!")

    io = TemplateIO(template_dir=tpl_dir)
    nested_out = tmp_path / "nested" / "dir" / "greeting.txt"
    io.render(
        template_name="test.template",
        output_path=nested_out,
        context={"name": "World"},
    )

    assert nested_out.exists()
    assert nested_out.read_text() == "Hello World!"

