"""
conftest.py — pytest configuration and sys.path guard.

This file solves the "editable install shadow bug":
When you run `pip install -e .` from inside the repo directory, Python's
import system may find the local `devmind/` folder before the installed
package, causing ImportError if the package has C extensions or if the
installed version differs.

The fix: ensure the repo root is NOT on sys.path ahead of site-packages.
pytest adds the rootdir to sys.path by default; this conftest removes it.
"""
import sys
import os

# Remove the repo root from sys.path if it was added by pytest's rootdir logic.
# This ensures `import devmind` resolves to the installed package (or the
# editable install's .pth entry), not the raw source directory.
_repo_root = os.path.dirname(os.path.abspath(__file__))
if _repo_root in sys.path:
    sys.path.remove(_repo_root)
    # Re-append at the end so relative imports still work as a last resort
    sys.path.append(_repo_root)
