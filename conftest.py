"""Make the ``rl_small`` package importable when running ``pytest`` from the
repo root without an editable install."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
