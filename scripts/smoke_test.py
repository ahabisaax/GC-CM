"""In-process driver for smoke_test.sh.

Runs experiments/run_experiments.py with the given CLI args in this process
(via runpy), then reports peak GPU memory from torch.cuda.max_memory_allocated.
Running in-process is what makes the peak-memory reading possible.

Usage: python scripts/smoke_test.py <run_experiments args...>
Exits non-zero if the underlying run fails.
"""
import os
import runpy
import sys

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)

import torch  # noqa: E402

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

target = os.path.join(REPO_DIR, "experiments", "run_experiments.py")
sys.argv = [target] + sys.argv[1:]

exit_code = 0
try:
    runpy.run_path(target, run_name="__main__")
except SystemExit as e:
    exit_code = int(e.code or 0)
except Exception as e:  # noqa: BLE001
    print(f"SMOKE TEST FAILED: {type(e).__name__}: {e}", file=sys.stderr)
    exit_code = 1

if torch.cuda.is_available():
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"PEAK_GPU_MEM_MB={peak_mb:.1f}")
else:
    print("PEAK_GPU_MEM_MB=n/a (CUDA not available)")

sys.exit(exit_code)
