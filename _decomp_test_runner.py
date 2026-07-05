#!/usr/bin/env python3
"""Per-method test subprocess runner. Args: name module class sx sy"""

import sys, gc, json, numpy as np, time, os

sys.path.insert(0, ".")
sys.stderr = open(os.devnull, "w")

name = sys.argv[1]
mod_path = sys.argv[2]
cls_name = sys.argv[3]
wt_path = sys.argv[4]
sx = int(sys.argv[5])
sy = int(sys.argv[6])

weight = np.load(wt_path)
tw = weight[:sx, :sy]
t0 = time.time()

try:
    exec(f"from {mod_path} import {cls_name}")
    cls = eval(cls_name)
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
    print(
        json.dumps({"status": "FAIL", "error": str(e)[:300], "time": time.time() - t0})
    )
