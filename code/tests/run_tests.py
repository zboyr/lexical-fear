"""Runner for the unit test suite.

Uses pytest when it is installed (some tests rely on the `tmp_path` fixture
and `pytest.mark.parametrize`); otherwise falls back to a dependency-free
loop over plain assertion tests.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def main() -> None:
    count = 0
    for path in sorted(Path(__file__).parent.glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name in sorted(dir(module)):
            if name.startswith("test_"):
                getattr(module, name)()
                count += 1
    print(f"{count} unit tests passed")


if __name__ == "__main__":
    try:
        import pytest  # noqa: F401
    except ImportError:
        main()
    else:
        import sys

        sys.exit(pytest.main(["-q", str(Path(__file__).parent)]))
