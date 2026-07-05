#!/usr/bin/env python3
"""Per-method test for invented methods. Args: name wt_path sx sy"""

import sys, gc, json, numpy as np, time, os

sys.path.insert(0, ".")
sys.stderr = open(os.devnull, "w")

name = sys.argv[1]
wt_path = sys.argv[2]
sx = int(sys.argv[3])
sy = int(sys.argv[4])

# Import invented methods code
sys.path.insert(0, "/home/mike/Documents/Github/SpectralStream")
from _invented_decomp_methods import *

weight = np.load(wt_path)
tw = weight[:sx, :sy]
t0 = time.time()

try:
    cls = None
    for k in dir():
        if k.lower().replace("_", "") == name.lower().replace("_", "").replace("-", ""):
            cls = eval(k)
            break
    if cls is None:
        title = "".join(w.capitalize() for w in name.replace("-", "_").split("_"))
        if title in dir():
            cls = eval(title)
    if cls is None:
        # Try globals
        cls = globals().get(name)
    if cls is None:
        raise ValueError(f"Class not found for {name}")

    inst = cls() if isinstance(cls, type) else cls
    data, meta = inst.compress(tw)
    recon = inst.decompress(data, meta)

    if recon.shape != tw.shape:
        raise ValueError(f"Shape mismatch: {recon.shape} vs {tw.shape}")

    from spectralstream.core.math_primitives.quality import QualityAssessor

    q = QualityAssessor().assess(tw, recon)
    ratio = float(tw.nbytes / max(len(data), 1))

    print(
        json.dumps(
            {
                "status": "OK",
                "ratio": ratio,
                "cos_sim": float(q.cosine_similarity),
                "rel_err": float(q.relative_error),
                "time": time.time() - t0,
                "shape": str(tw.shape),
            }
        )
    )
except Exception as e:
    import traceback

    print(
        json.dumps({"status": "FAIL", "error": str(e)[:300], "time": time.time() - t0})
    )
