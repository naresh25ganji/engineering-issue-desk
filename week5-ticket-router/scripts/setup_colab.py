"""Install the pinned training framework inside a CUDA notebook runtime."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from router_lab.config import FACTORY_REVISION


def main():
    if sys.version_info < (3, 11):
        raise RuntimeError("Python 3.11 or newer is required.")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Select a CUDA GPU runtime before installing GPU training dependencies.")
    print("GPU:", torch.cuda.get_device_name(0), "| torch:", torch.__version__, flush=True)
    factory = ROOT / ".cache/LLaMA-Factory"
    factory.parent.mkdir(parents=True, exist_ok=True)
    if not factory.exists():
        subprocess.run(["git", "clone", "--no-checkout", "https://github.com/hiyouga/LLaMA-Factory.git", str(factory)], check=True)
        subprocess.run(["git", "checkout", "--detach", FACTORY_REVISION], cwd=factory, check=True)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=factory, text=True).strip()
    if actual != FACTORY_REVISION:
        raise RuntimeError("Existing LLaMA Factory checkout has another revision; preserve it and use a new project directory.")
    changed = subprocess.check_output(["git", "diff", "--name-only", "HEAD"], cwd=factory, text=True).strip()
    if changed:
        raise RuntimeError("Existing LLaMA Factory source has changes; use a clean pinned checkout.")
    constraint = ROOT / "requirements-colab.txt"
    environment = {**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(constraint),
                    "-c", str(constraint), "-e", str(factory), "-e", str(ROOT)],
                   check=True, env=environment)
    print("Pinned training framework installed. Training has not started.")


if __name__ == "__main__":
    main()
