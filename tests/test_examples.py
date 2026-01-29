"""Tests to validate example code structure and syntax."""

import ast
import os
from pathlib import Path

import pytest


EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def get_python_files() -> list[Path]:
    """Get all Python files in examples directory."""
    return list(EXAMPLES_DIR.rglob("*.py"))


class TestExamplesSyntax:
    """Test that all example files have valid Python syntax."""

    @pytest.mark.parametrize("python_file", get_python_files(), ids=lambda p: str(p.relative_to(EXAMPLES_DIR)))
    def test_valid_syntax(self, python_file: Path) -> None:
        """Verify each Python file has valid syntax."""
        source = python_file.read_text()
        try:
            ast.parse(source)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in {python_file}: {e}")


class TestExamplesStructure:
    """Test the structure of examples directory."""

    def test_examples_directory_exists(self) -> None:
        """Verify examples directory exists."""
        assert EXAMPLES_DIR.exists(), "examples/ directory should exist"

    def test_has_requirements(self) -> None:
        """Verify requirements.txt exists in examples."""
        req_file = EXAMPLES_DIR / "requirements.txt"
        assert req_file.exists(), "examples/requirements.txt should exist"

    def test_has_readme(self) -> None:
        """Verify README exists in examples."""
        readme = EXAMPLES_DIR / "README.md"
        assert readme.exists(), "examples/README.md should exist"

    def test_example_directories_exist(self) -> None:
        """Verify expected example directories exist."""
        expected_dirs = [
            "loading",
            "transform",
            "tokenize",
            "augment",
            "imbalance",
            "models",
            "training",
            "custom",
            "callback",
            "inferencing",
            "publishing",
        ]
        for dir_name in expected_dirs:
            dir_path = EXAMPLES_DIR / dir_name
            assert dir_path.exists(), f"examples/{dir_name}/ should exist"


class TestDemoScript:
    """Test the demo script."""

    def test_demo_exists(self) -> None:
        """Verify demo.py exists."""
        demo_path = Path(__file__).parent.parent / "demo.py"
        assert demo_path.exists(), "demo.py should exist"

    def test_demo_has_main(self) -> None:
        """Verify demo.py has a main function."""
        demo_path = Path(__file__).parent.parent / "demo.py"
        source = demo_path.read_text()
        tree = ast.parse(source)

        function_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        assert "main" in function_names, "demo.py should have a main() function"
