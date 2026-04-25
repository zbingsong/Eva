from pathlib import Path

import pytest


@pytest.fixture
def tmp_output_root(tmp_path: Path) -> Path:
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    return output_root


@pytest.fixture
def tmp_output_dir(tmp_output_root: Path) -> Path:
    output_dir = tmp_output_root / "run"
    output_dir.mkdir()
    return output_dir
