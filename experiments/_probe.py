"""Quick environment probe for the experiment runner."""
import sys
import os

# Make sure the project root is on sys.path so imports resolve from anywhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

print("python:", sys.version)
print("executable:", sys.executable)
print("cwd:", os.getcwd())

import numpy
print("numpy:", numpy.__version__)

from config import get_default_config
from simulation import run_simulation
print("imports ok")

cfg = get_default_config()
print("default seed:", cfg.seed, "epochs:", cfg.num_epochs)
