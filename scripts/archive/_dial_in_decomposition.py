#!/usr/bin/env python3
"""DIAL IN DECOMPOSITION METHODS — subprocess runner with timeouts."""

import sys, gc, os, json, subprocess, time, signal

sys.path.insert(0, ".")
import numpy as np

CACHE_DIR = "/tmp/spectralstream_diag"
os.makedirs(CACHE_DIR, exist_ok=True)
RESULTS_FILE = f"{CACHE_DIR}/decomposition_results.json"
WT_PATH = f"{CACHE_DIR}/weight_down_proj.npy"

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: CACHE WEIGHTS
# ═══════════════════════════════════════════════════════════════════════════
if not os.path.exists(WT_PATH):
    from spectralstream.compression.engine.memory_mapped_engine import (
        MemoryMappedTensorEngine,
    )

    mmap = MemoryMappedTensorEngine(
        "/home/mike/Documents/Github/SpectralStream/models/gemma-4-E2B/model.safetensors"
    )
    for name in mmap.get_tensor_names():
        if "down_proj" in name:
            wt = np.array(mmap.get_tensor(name)).astype(np.float32)
            np.save(WT_PATH, wt)
            print(f"Cached: {name} {wt.shape} {wt.nbytes / 1e6:.1f}MB")
            break
    mmap.close()

weight = np.load(WT_PATH)
m, n = weight.shape
print(f"Weight: {m}x{n} {weight.nbytes / 1e6:.1f}MB")

SIZE_MAP = {
    "full": (m, n),
    "medium": (min(1536, m), min(3072, n)),
    "small": (min(768, m), min(1536, n)),
    "tiny": (min(256, m), min(256, n)),
}

# ═══════════════════════════════════════════════════════════════════════════
# METHODS DEFINITION
# ═══════════════════════════════════════════════════════════════════════════
EXISTING_METHODS = [
    (
        "svd_truncated",
        "spectralstream.compression.methods.decomposition.svd",
        "SVDTruncated",
        "medium",
    ),
    (
        "svd_compress",
        "spectralstream.compression.engine._methods",
        "_SVDCompress",
        "medium",
    ),
    (
        "tensor_train",
        "spectralstream.compression.methods.decomposition.tensor_train",
        "TensorTrain",
        "small",
    ),
    (
        "tensor_ring",
        "spectralstream.compression.methods.decomposition.tensor_train",
        "TensorRing",
        "small",
    ),
    (
        "tt_orthogonal",
        "spectralstream.compression.methods.decomposition.tensor_train",
        "TTOrthogonal",
        "small",
    ),
    (
        "tt_svd",
        "spectralstream.compression.methods.decomposition.tensor_train",
        "TTSVD",
        "small",
    ),
    (
        "tucker",
        "spectralstream.compression.methods.decomposition.tucker",
        "TuckerDecomposition",
        "medium",
    ),
    (
        "block_tucker",
        "spectralstream.compression.methods.decomposition.tucker",
        "BlockTucker",
        "medium",
    ),
    (
        "hierarchical_tucker",
        "spectralstream.compression.methods.decomposition.tucker",
        "HierarchicalTucker",
        "medium",
    ),
    (
        "cp_decomposition",
        "spectralstream.compression.methods.decomposition.cp",
        "CPDecomposition",
        "tiny",
    ),
    (
        "butterfly",
        "spectralstream.compression.methods.decomposition.butterfly",
        "Butterfly",
        "tiny",
    ),
    (
        "monarch",
        "spectralstream.compression.methods.decomposition.butterfly",
        "Monarch",
        "small",
    ),
    (
        "kronecker",
        "spectralstream.compression.methods.decomposition.kronecker",
        "Kronecker",
        "small",
    ),
    (
        "cur_decomposition",
        "spectralstream.compression.methods.decomposition.kronecker",
        "CURDecomposition",
        "small",
    ),
    (
        "einsort_tt",
        "spectralstream.compression.methods.decomposition.einsort",
        "EinsortTT",
        "small",
    ),
    (
        "lotr",
        "spectralstream.compression.methods.decomposition.einsort",
        "LOTR",
        "tiny",
    ),
    (
        "h_matrix",
        "spectralstream.compression.methods.decomposition.matrix_approx",
        "HMatrix",
        "small",
    ),
    (
        "nystrom",
        "spectralstream.compression.methods.decomposition.matrix_approx",
        "Nystrom",
        "small",
    ),
    (
        "random_feature",
        "spectralstream.compression.methods.decomposition.matrix_approx",
        "RandomFeature",
        "tiny",
    ),
    (
        "block_diagonal",
        "spectralstream.compression.methods.decomposition.structured_mat",
        "BlockDiagonal",
        "medium",
    ),
    (
        "toeplitz",
        "spectralstream.compression.methods.decomposition.structured_mat",
        "Toeplitz",
        "medium",
    ),
    (
        "hankel",
        "spectralstream.compression.methods.decomposition.structured_mat",
        "Hankel",
        "medium",
    ),
    (
        "tensor_network",
        "spectralstream.compression.methods.decomposition.tensor_network",
        "TensorNetwork",
        "small",
    ),
    (
        "hierarchical_mps",
        "spectralstream.compression.methods.decomposition.tensor_network",
        "HierarchicalMPS",
        "small",
    ),
    (
        "adntn_mera",
        "spectralstream.compression.methods.decomposition.merapeps",
        "ADNTNMERA",
        "tiny",
    ),
    (
        "ipeps_2d",
        "spectralstream.compression.methods.decomposition.merapeps",
        "IPEPS2D",
        "tiny",
    ),
    (
        "hpc_block_svd",
        "spectralstream.compression.methods.novel.hpc_parallel",
        "HPCBlockSVD",
        "small",
    ),
    (
        "mera_adv",
        "spectralstream.compression.methods.novel.quantum.tensor_quantum._meraadv",
        "MERAAdv",
        "tiny",
    ),
    (
        "peps_boundary",
        "spectralstream.compression.methods.novel.quantum.tensor_quantum._pepsboundary",
        "PEPSBoundary",
        "tiny",
    ),
    (
        "qtt_adapt",
        "spectralstream.compression.methods.novel.quantum.tensor_quantum._qttadapt",
        "QTTAdapt",
        "tiny",
    ),
    (
        "tt_cross",
        "spectralstream.compression.methods.novel.quantum.tensor_quantum._ttcross",
        "TTCross",
        "tiny",
    ),
    (
        "dmrg_sweep",
        "spectralstream.compression.methods.novel.quantum.tensor_quantum._dmrgsweep",
        "DMRGSweep",
        "tiny",
    ),
    (
        "singular_val_density",
        "spectralstream.compression.methods.novel.quantum.tensor_quantum._singularvaluedensity",
        "SingularValueDensity",
        "tiny",
    ),
    (
        "matrix_product_op",
        "spectralstream.compression.methods.novel.quantum.tensor_quantum._matrixproductoperator",
        "MatrixProductOperator",
        "tiny",
    ),
    (
        "cross_layer_delta",
        "spectralstream.compression.methods.novel.entropy_info.cross_layer_coding",
        "CrossLayerDeltaCompression",
        "small",
    ),
    (
        "blockwise_cross_layer",
        "spectralstream.compression.methods.novel.entropy_info.cross_layer_coding",
        "BlockwiseCrossLayerDelta",
        "small",
    ),
]

INVENTED_METHODS = [
    ("hierarchical_block_svd", "small"),
    ("cross_layer_svd", "small"),
    ("low_rank_plus_sparse", "small"),
    ("adaptive_svd", "small"),
    ("symplectic_svd", "small"),
]

# ═══════════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════════
results = {}
TIMEOUTS = {"full": 300, "medium": 120, "small": 60, "tiny": 30}


def run_test(args_list, timeout=60):
    try:
        proc = subprocess.run(
            [sys.executable] + args_list,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        for line in proc.stdout.split("\n"):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        return {"status": "FAIL", "error": f"No JSON. stderr: {proc.stderr[:200]}"}
    except subprocess.TimeoutExpired:
        return {"status": "FAIL", "error": "TIMEOUT"}
    except Exception as e:
        return {"status": "FAIL", "error": str(e)[:200]}


# ── Test existing methods ──
print("=" * 100)
print("EXISTING DECOMPOSITION METHODS")
print("=" * 100)

for name, mod_path, cls_name, size_key in EXISTING_METHODS:
    sx, sy = SIZE_MAP[size_key]
    timeout = TIMEOUTS[size_key]
    print(f"\n  {name:<30s} size={sx}x{sy} timeout={timeout}s...", end=" ", flush=True)

    result = run_test(
        ["_decomp_test_runner.py", name, mod_path, cls_name, WT_PATH, str(sx), str(sy)],
        timeout,
    )
    results[name] = result

    if result["status"] == "OK":
        print(
            f"✓ ratio={result['ratio']:>7.1f}:1  cos={result['cos_sim']:.4f}  err={result['rel_err']:.6f}  time={result['time']:.1f}s"
        )
    else:
        print(f"✗ {result.get('error', '')[:80]}")

# ── Test invented methods ──
print("\n" + "=" * 100)
print("INVENTED METHODS")
print("=" * 100)

for name, size_key in INVENTED_METHODS:
    sx, sy = SIZE_MAP[size_key]
    timeout = TIMEOUTS[size_key]
    print(f"\n  {name:<30s} size={sx}x{sy} timeout={timeout}s...", end=" ", flush=True)

    result = run_test(["_decomp_test_new.py", name, WT_PATH, str(sx), str(sy)], timeout)
    results[name] = result

    if result["status"] == "OK":
        print(
            f"✓ ratio={result['ratio']:>7.1f}:1  cos={result['cos_sim']:.4f}  err={result['rel_err']:.6f}  time={result['time']:.1f}s"
        )
    else:
        print(f"✗ {result.get('error', '')[:80]}")

# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("FINAL SUMMARY")
print("=" * 100)

valid = {n: r for n, r in results.items() if r.get("status") == "OK"}
bad = {n: r for n, r in results.items() if r.get("status") != "OK"}

print(f"\n  {len(valid)} passed, {len(bad)} failed")

if valid:
    by_score = sorted(valid.items(), key=lambda x: -x[1]["ratio"] * x[1]["cos_sim"])
    print(f"\n  TOP 15 BY RATIO × QUALITY:")
    for name, r in by_score[:15]:
        score = r["ratio"] * r["cos_sim"]
        print(
            f"    {name:<30s} score={score:>10.1f}  ratio={r['ratio']:>7.1f}:1  cos={r['cos_sim']:.4f}  err={r['rel_err']:.6f}"
        )

    by_ratio = sorted(valid.items(), key=lambda x: -x[1]["ratio"])
    print(f"\n  TOP 10 BY RATIO:")
    for name, r in by_ratio[:10]:
        print(
            f"    {name:<30s} ratio={r['ratio']:>7.1f}:1  cos={r['cos_sim']:.4f}  err={r['rel_err']:.6f}"
        )

    by_cos = sorted(valid.items(), key=lambda x: -x[1]["cos_sim"])
    print(f"\n  TOP 10 BY COSINE SIMILARITY:")
    for name, r in by_cos[:10]:
        print(
            f"    {name:<30s} cos={r['cos_sim']:.4f}  err={r['rel_err']:.6f}  ratio={r['ratio']:>7.1f}:1"
        )

    print(f"\n  Methods achieving 500:1 ratio:")
    extreme = [(n, r) for n, r in valid.items() if r["ratio"] >= 500]
    if extreme:
        for name, r in sorted(extreme, key=lambda x: -x[1]["cos_sim"]):
            print(f"    {name:<30s} ratio={r['ratio']:>7.1f}:1  cos={r['cos_sim']:.4f}")
    else:
        print(
            f"    None — best is {max(valid.items(), key=lambda x: x[1]['ratio'])[1]['ratio']:.0f}:1"
        )

if bad:
    print(f"\n  FAILURES:")
    for name, r in sorted(bad.items()):
        print(f"    {name:<30s} {r.get('error', '')[:100]}")

# Save
clean = {}
for k, v in results.items():
    clean[k] = {
        kk: float(vv)
        if isinstance(vv, (float, np.floating, np.integer)) and not isinstance(vv, bool)
        else vv
        for kk, vv in v.items()
    }
with open(RESULTS_FILE, "w") as f:
    json.dump({"weight_shape": str(weight.shape), "results": clean}, f, indent=2)
print(f"\n  Results → {RESULTS_FILE}")
