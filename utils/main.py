"""Compatibility entry point for launchers that execute ``utils/main.py``.

The upstream project represents this path as a symbolic link to the repository
root's ``main.py``. Some Windows copies materialize the link target as plain
text, which is not valid Python, so this wrapper forwards execution explicitly.
"""

from pathlib import Path
import runpy
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    runpy.run_path(str(PROJECT_ROOT / "main.py"), run_name="__main__")
