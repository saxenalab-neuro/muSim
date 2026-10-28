# Runs the lab's main.py on a computer without a GPU.
#
# Why this exists: the lab's trained checkpoint was saved on a GPU, and
# simulate.py loads checkpoints with plain torch.load(...). On a CPU-only
# machine that fails with "Attempting to deserialize object on a CUDA device"
# (bug B8 in the codebase guide). Rather than change lab code, this wrapper
# tells torch.load to put everything on the CPU, then runs main.py unchanged.
#
# Usage, from the repo root:  python tests/_cpu_main.py <the same args as main.py>
import functools
import os
import runpy
import sys

import torch

# Make the lab's modules importable (main.py expects to run from the repo root)
sys.path.insert(0, os.getcwd())

# From now on, every torch.load(...) call behaves like torch.load(..., map_location="cpu")
torch.load = functools.partial(torch.load, map_location="cpu")

# Pretend main.py was started directly, with our command-line arguments, and run it
sys.argv = ["main.py", *sys.argv[1:]]
runpy.run_path("main.py", run_name="__main__")
