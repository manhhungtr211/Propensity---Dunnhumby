"""AST and architectural guard for notebooks/eda.ipynb."""

from __future__ import annotations

import ast
import json
from pathlib import Path


def test_eda_notebook_contains_no_forbidden_definitions() -> None:
    """Verify eda.ipynb strictly adheres to the AST rule: zero def, class, lambda in code cells."""
    nb_path = Path("notebooks/eda.ipynb")
    assert nb_path.exists(), "Notebook notebooks/eda.ipynb does not exist"

    with open(nb_path, encoding="utf-8") as f:
        nb_data = json.load(f)

    code_cells = [
        cell["source"] for cell in nb_data["cells"] if cell["cell_type"] == "code"
    ]
    full_code = "\n".join("".join(lines) for lines in code_cells)

    tree = ast.parse(full_code)

    # Disallow function defs, async function defs, class defs, and lambdas
    for node in ast.walk(tree):
        assert not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ), f"Notebook contains forbidden definition: {type(node).__name__}"
