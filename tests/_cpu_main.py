# Runs main.py with checkpoints mapped to CPU, so GPU-saved checkpoints load on
# machines without CUDA. Usage: python tests/_cpu_main.py <main.py args>
import functools
import os
import runpy
import sys

import torch

sys.path.insert(0, os.getcwd())
torch.load = functools.partial(torch.load, map_location="cpu")
sys.argv = ["main.py", *sys.argv[1:]]
runpy.run_path("main.py", run_name="__main__")
