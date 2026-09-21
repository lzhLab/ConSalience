import numpy as np
from pathlib import Path

for p in [
    Path("data/preprocessed/3Dircadb1/3Dircadb1.3.npz"),
    Path("data/preprocessed/MSD/hepaticvessel_265.npz"),
]:
    if p.exists():
        data = np.load(p)
        print(p)
        print("ct:", data["ct"].shape, data["ct"].dtype, data["ct"].min(), data["ct"].max())
        print("mask:", data["mask"].shape, data["mask"].dtype, data["mask"].min(), data["mask"].max())

