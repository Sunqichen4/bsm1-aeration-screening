"""Keep pytest's dynamically imported Numba modules out of production caches."""
from pathlib import Path
import os

# conftest imports before test-module collection, including imports of
# run_study.py, whose environment setup deliberately uses setdefault.
_PYTEST_NUMBA_CACHE = Path(__file__).resolve().parents[1] / ".runtime_cache" / "numba_pytest"
_PYTEST_NUMBA_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["NUMBA_CACHE_DIR"] = str(_PYTEST_NUMBA_CACHE)
