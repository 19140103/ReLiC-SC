"""Report whether the local machine is ready for the ReLiC-SC pipeline."""

from __future__ import annotations

import importlib
import platform
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def module_version(name: str) -> str:
    try:
        module = importlib.import_module(name)
        return str(getattr(module, "__version__", "installed"))
    except Exception as exc:  # diagnostics should continue after one failure
        return f"MISSING ({exc.__class__.__name__})"


def main() -> int:
    print(f"Project: {ROOT}")
    print(f"OS:      {platform.platform()}")
    print(f"Python:  {sys.version.split()[0]} ({sys.executable})")
    for name in ["torch", "numpy", "scipy", "pandas", "mne", "h5py", "zarr"]:
        print(f"{name:8s} {module_version(name)}")

    try:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"GPU:            {props.name}")
            print(f"GPU memory:     {props.total_memory / 1024**3:.2f} GB")
            print(f"CUDA runtime:   {torch.version.cuda}")
    except Exception:
        pass

    usage = shutil.disk_usage(ROOT)
    print(f"Project drive free: {usage.free / 1024**3:.2f} GB")
    if usage.free < 10 * 1024**3:
        print("WARNING: less than the 10 GB safety reserve remains.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
