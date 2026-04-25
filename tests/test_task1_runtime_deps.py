import ast
from pathlib import Path


def _dependency_name(spec: str) -> str:
    for token in ("<", ">", "=", "!", "~", "["):
        if token in spec:
            return spec.split(token, 1)[0].strip()
    return spec.strip()


def _parse_env_yaml_dependencies() -> tuple[list[str], list[str]]:
    top_level: list[str] = []
    pip_deps: list[str] = []
    in_pip_block = False
    in_dependencies_block = False

    for line in Path("env.yaml").read_text(encoding="utf-8").splitlines():
        if line.startswith("dependencies:"):
            in_dependencies_block = True
            continue

        if not in_dependencies_block:
            continue

        if line.startswith("  - "):
            entry = line[4:].strip()
            in_pip_block = entry == "pip:"
            if not in_pip_block:
                top_level.append(entry)
            continue

        if in_pip_block and line.startswith("      - "):
            pip_deps.append(line[8:].strip())
            continue

        if in_pip_block and line and not line.startswith("      ") and not line.startswith("    "):
            in_pip_block = False

    return top_level, pip_deps


def _parse_setup_install_requires() -> list[str]:
    module = ast.parse(Path("setup.py").read_text(encoding="utf-8"))

    for node in module.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not isinstance(call.func, ast.Name) or call.func.id != "setup":
            continue
        for keyword in call.keywords:
            if keyword.arg != "install_requires" or not isinstance(keyword.value, ast.List):
                continue
            return [
                elt.value
                for elt in keyword.value.elts
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
            ]
    raise AssertionError("setup.py is missing a literal install_requires list")


def test_env_yaml_lists_task1_dependencies_structurally() -> None:
    conda_deps, pip_deps = _parse_env_yaml_dependencies()
    conda_names = {_dependency_name(spec) for spec in conda_deps}
    pip_names = {_dependency_name(spec) for spec in pip_deps}

    assert "openslide" in conda_names
    assert {"openslide-python", "tifffile", "pytest"} <= pip_names


def test_setup_py_lists_runtime_dependencies_structurally() -> None:
    install_requires = _parse_setup_install_requires()
    install_names = {_dependency_name(spec) for spec in install_requires}

    assert {"tifffile", "openslide-python", "omegaconf", "timm"} <= install_names


def test_tmp_output_dir_fixture_creates_writable_directory(tmp_output_dir: Path) -> None:
    assert tmp_output_dir.exists()
    assert tmp_output_dir.is_dir()

    sentinel = tmp_output_dir / "sentinel.txt"
    sentinel.write_text("ok", encoding="utf-8")

    assert sentinel.read_text(encoding="utf-8") == "ok"
