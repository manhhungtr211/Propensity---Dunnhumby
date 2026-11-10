from __future__ import annotations

import subprocess
import sys

from propensity import ColumnRole, DatasetSpec, DatasetSpecError


def test_public_schema_api_is_available_from_package_root() -> None:
    # Given / When
    public_types = (ColumnRole, DatasetSpec, DatasetSpecError)

    # Then
    assert all(
        public_type.__module__.startswith("propensity.") for public_type in public_types
    )


def test_installed_package_imports_outside_repository(tmp_path) -> None:
    # Given
    command = [sys.executable, "-c", "import propensity; print(propensity.__name__)"]

    # When
    completed = subprocess.run(
        command,
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    # Then
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "propensity"
