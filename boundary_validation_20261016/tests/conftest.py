"""Test-only runtime isolation; no production cache or long ODE integrations."""
from pathlib import Path
import os
import sys


STUDY_ROOT = Path(__file__).resolve().parents[1]
# This must happen before collecting scientific modules.  Inherited production
# cache paths are deliberately replaced, never shared with monkeypatched tests.
os.environ["NUMBA_CACHE_DIR"] = str(STUDY_ROOT / ".runtime_cache" / "tests")
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, str(STUDY_ROOT))
