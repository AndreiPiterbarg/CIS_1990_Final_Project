"""Install notebook dependencies and register a local Jupyter kernel."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    venv_python = root / ".venv" / "bin" / "python"
    if not venv_python.exists():
        raise SystemExit(
            "Expected a project virtualenv at .venv/bin/python. "
            "Create it first, then rerun this script."
        )

    requirements = root / "eval" / "requirements-notebook.txt"
    kernel_prefix = root / ".venv"
    kernel_name = "git-explainer-eval"
    display_name = "Python (.venv) - Git Explainer"

    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "-r", str(requirements)],
        check=True,
    )
    subprocess.run(
        [
            str(venv_python),
            "-m",
            "ipykernel",
            "install",
            "--prefix",
            str(kernel_prefix),
            "--name",
            kernel_name,
            "--display-name",
            display_name,
        ],
        check=True,
    )

    print("Notebook dependencies installed.")
    print(f"Kernel registered as {kernel_name!r} under {kernel_prefix}.")
    print("Launch with: .venv/bin/jupyter notebook eval/git_explainer_eval_workbench.ipynb")


if __name__ == "__main__":
    main()
