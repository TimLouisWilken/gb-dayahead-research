"""Sanity / hyper-parameter checks of single members on the burn-in period only
(train < 2023-07-01, test 2023-07-01 .. 2023-12-31). The evaluation period (2024+) is never touched.

Usage: python src/tune_lstm.py "NeuralMember(arch='tcn', revin=True)"
       python src/tune_lstm.py "GBMember()"
"""
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import torch

from data import DayAheadData
from models import GBMember, LEARMember, MeritOrderMember, NeuralMember, RFMember  # noqa: F401  (used by eval)

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "3")))
spec = sys.argv[1]
d = DayAheadData()
valid = np.where(d.valid)[0]
t0 = d.index_of("2023-07-01")
tr, te = valid[valid < t0], np.arange(t0, d.index_of("2024-01-01"))
t = time.time()
p = eval(spec).fit(d, tr).predict(d, te)
print(f"{spec:70s} MAE {np.abs(p - d.y[te]).mean():6.2f}  ({time.time() - t:.0f}s)", flush=True)
