"""AST and structural guard for notebooks/00_data_overview.ipynb."""

import ast
import json
from pathlib import Path


def test_notebook_00_contains_no_local_functions_or_analytical_logic() -> None:
    # Given
    nb_path = Path("notebooks/00_data_overview.ipynb")
    assert nb_path.exists(), "Notebook 00_data_overview.ipynb does not exist"

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

    # Verify build_dunnhumby_overview is called
    calls = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert "build_dunnhumby_overview" in calls
