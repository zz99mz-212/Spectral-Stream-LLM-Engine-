#!/usr/bin/env python3
"""
Extreme Compression Test Suite — QICE, QSSF, Extreme KV, comparison.
"""
import sys, os, json, math, time
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spectralstream.quantum_intelligence_engine import (
    QuantumIntelligenceCompressionEngine,
    ResonanceDrivenCompressor,
    PlasmaConfinementQuantizer,
    QuantumWavefunctionCompressor,
    FractalScaleRecurrentCompressor,
    HyperdimensionalWeightBundler,
    TopologicalDefectEncoder,
    TimeCrystalLatticeQuantizer,
    QICEQualityValidator,
)
from spectralstream.spectral_binary_format import (
    QSSFWriter, QSSFReader, CompressionMethod, QSSFValidator,
)
from spectralstream.extreme_kv_cache import (
    ExtremeTieredKVCache,
)
from spectralstream.unified_intelligence_engine import (
    TTProductQuantCompressor, ExtremeCompressor,
)

EPS = 1e-30
REPORT: dict = {}
TENSOR_SHAPES = {
    'attn_q': (1536, 2048), 'attn_k': (1536, 256), 'attn_v': (1536, 256),
    'attn_o': (1536, 2048), 'ffn_gate': (1536, 6144), 'ffn_up': (1536, 6144),
    'ffn_down': (6144, 1536), 'embed': (256000, 1536), 'norm': (1536,),
}

# ═══════════════════════════════════════════════════════════════════════
# Tensor generation
# ═══════════════════════════════════════════════════════════════════════

def gen_tensor(shape, eff_rank=50, seed=42):
    rng = np.random.RandomState(seed)
    if len(shape) == 1:
        return (rng.randn(*shape).astype(np.float32) * 0.02)
    m, n = shape; k = min(m, n)
    A = rng.randn(m, n).astype(np.float64)
    U, _, Vt = np.linalg.svd(A, full_matrices=False)
    s = np.arange(1, k+1, dtype=np.float64) ** (-1.5)
    s = s / np.sum(s) * np.sqrt(m*n) * 0.15
    if eff_rank < k:
        s[eff_rank:] *= 0.05
    t = (U[:,:k] * s[np.newaxis,:]) @ Vt[:k,:]
    t += rng.randn(m,n).astype(np.float64) * 0.002
    return t.astype(np.float32)

def ref_tensors(scale=0.125):
    d = {}
    for name, shape in TENSOR_SHAPES.items():
        s = tuple(max(1, int(d*scale)) for d in shape)
        er = min(50, min(s)) if 'attn' in name else (
            min(150, min(s)) if 'ffn' in name else (
            min(80, min(s)) if name == 'embed' else min(10, min(s))))
        d[name] = gen_tensor(s, eff_rank=er, seed=42)
    return d

def snr(orig, recon):
    mse = float(np.mean((orig.astype(np.float64)-recon.astype(np.float64))**2))
    var = float(np.mean(orig.astype(np.float64)**2))
    return float(10*np.log10(var/max(mse,EPS))) if mse>EPS else 100.0

def err_pct(orig, recon):
    return float(np.mean(np.abs(orig.astype(np.float64)-recon.astype(np.float64))
                         / (np.abs(orig.astype(np.float64))+EPS))*100)

def rel_err(orig, recon):
    return float(np.linalg.norm(orig.astype(np.float64)-recon.astype(np.float64))
                 / max(np.linalg.norm(orig.astype(np.float64)), EPS))

# ═══════════════════════════════════════════════════════════════════════
# 1. Individual compressors
# ═══════════════════════════════════════════════════════════════════════

def test_individual():
    results = {}
    tr = gen_tensor((128, 128), eff_rank=10, seed=42)

    # Resonance
    c = ResonanceDrivenCompressor(n_bands=8).compress(tr, "t")
    d = ResonanceDrivenCompressor(n_bands=8).decompress(c)
    results['resonance_driven'] = dict(ratio=c['compression_ratio'], snr_db=snr(tr,d),
        err_pct=err_pct(tr,d), rel_err=rel_err(tr,d))

    # Plasma
    p = PlasmaConfinementQuantizer()
    c = p.compress(tr, "t", target_bits=4); d = p.decompress(c)
    results['plasma_confinement'] = dict(ratio=c['compression_ratio'], snr_db=snr(tr,d),
        err_pct=err_pct(tr,d), rel_err=rel_err(tr,d))

    # Wavefunction
    w = QuantumWavefunctionCompressor(keep_fraction=0.05)
    c = w.compress(tr, "t"); d = w.decompress(c)
    results['quantum_wavefunction'] = dict(ratio=c['compression_ratio'], snr_db=snr(tr,d),
        err_pct=err_pct(tr,d), rel_err=rel_err(tr,d))

    # Fractal (pair)
    t1 = gen_tensor((128, 128), eff_rank=10, seed=41)
    t2 = gen_tensor((128, 128), eff_rank=10, seed=43)
    f = FractalScaleRecurrentCompressor()
    cs = f.compress_layer_group([t1, t2], ["a", "b"])
    d2 = f.decompress(cs[1])
    avg_r = float(np.mean([c['compression_ratio'] for c in cs]))
    results['fractal_scale_recurrent'] = dict(ratio=avg_r, snr_db=snr(t2,d2),
        err_pct=err_pct(t2,d2), rel_err=rel_err(t2,d2),
        base_ratio=cs[0]['compression_ratio'], resid_ratio=cs[1]['compression_ratio'])

    # Hyperdimensional (tiny)
    ts = gen_tensor((32, 32), eff_rank=5, seed=42)
    h = HyperdimensionalWeightBundler(dim=500)
    c = h.compress(ts, "t"); d = h.decompress(c)
    results['hyperdimensional_bundler'] = dict(ratio=c['compression_ratio'], snr_db=snr(ts,d),
        err_pct=err_pct(ts,d), rel_err=rel_err(ts,d))

    # Topological
    tp = TopologicalDefectEncoder()
    c = tp.compress(tr, "t"); d = tp.decompress(c)
    results['topological_defect'] = dict(ratio=c['compression_ratio'], snr_db=snr(tr,d),
        err_pct=err_pct(tr,d), rel_err=rel_err(tr,d))

    # TimeCrystal
    tc = TimeCrystalLatticeQuantizer(delta=0.5)
    c = tc.compress(tr, "t", n_bits=4); d = tc.decompress(c)
    results['time_crystal_lattice'] = dict(ratio=c['compression_ratio'], snr_db=snr(tr,d),
        err_pct=err_pct(tr,d), rel_err=rel_err(tr,d))

    # TT Product Quant
    ttpq = TTProductQuantCompressor(rank=4, sub_dim=4, n_centroids=32)
    c = ttpq.compress(tr, "t"); d = ttpq.decompress(c)
    results['tt_product_quant'] = dict(ratio=c['compression_ratio'], snr_db=snr(tr,d),
        err_pct=err_pct(tr,d), rel_err=rel_err(tr,d))

    return results

# ═══════════════════════════════════════════════════════════════════════
# 2. Full QICE pipeline
# ═══════════════════════════════════════════════════════════════════════

def test_pipeline():
    tensors = ref_tensors(0.125)
    # Skip embed (too large), norm (1D issues)
    tensors = {k:v for k,v in tensors.items()
               if v.ndim >= 2 and min(v.shape) < 400}
    if not tensors:
        return {'per_tensor': {}, 'summary': {'n_tensors': 0, 'overall_compression_ratio': 0, 'avg_psnr': 0, 'target_ratio': 500}}
    engine = QuantumIntelligenceCompressionEngine(
        target_ratio=500., quality_threshold=0.001, max_refinement_iters=1)
    per_tensor = {}
    total_orig = total_comp = total_psnr = 0; n = 0
    for name, tensor in tensors.items():
        print(f"    {name} {tensor.shape}...", end=' ', flush=True)
        try:
            r = engine.run_pipeline(tensor, layer_name=name)
        except Exception as e:
            print(f"ERROR: {e}"); continue
        pt = r['result']; q = r['quality']
        print(f"ratio={pt['compression_ratio']:.1f}x psnr={q.get('psnr',0):.1f}")
        per_tensor[name] = dict(strategy=r['analysis']['strategy'], shape=list(tensor.shape),
            ratio=pt['compression_ratio'], psnr=q.get('psnr',0), mse=q.get('mse',0),
            rel_err=q.get('relative_error',1), passed=q.get('passed',False))
        total_orig += int(np.prod(tensor.shape))*4
        total_comp += pt['compressed_bytes']
        total_psnr += q.get('psnr',0); n += 1
    return dict(per_tensor=per_tensor, summary=dict(n_tensors=n,
        overall_compression_ratio=total_orig/max(total_comp,1),
        avg_psnr=total_psnr/max(n,1), target_ratio=500.))

# ═══════════════════════════════════════════════════════════════════════
# 3. QSSF round-trip
# ═══════════════════════════════════════════════════════════════════════

def test_qssf(path='/tmp/test_model.qssf'):
    tensors = {
        'q': gen_tensor((128, 256), 15, 42),
        'k': gen_tensor((128, 64), 10, 43),
        'v': gen_tensor((128, 64), 10, 44),
    }
    w = QSSFWriter(path, metadata=dict(test=True))
    for n, t in tensors.items():
        w.add_tensor(n, t, CompressionMethod.SVD_LOWRANK, dict(rank=8))
    file_size = w.save(); w.close()
    r = QSSFReader(path)
    read, info = {}, []
    for e in r.list_tensors():
        read[e['name']] = r.read_tensor(e['name']); info.append(e)
    r.close()
    integ = {}
    all_ok = True
    for n in tensors:
        if n in read:
            mse = float(np.mean((tensors[n].astype(np.float64)-read[n].astype(np.float64))**2))
            integ[n] = dict(shape_match=tensors[n].shape==read[n].shape, mse=mse, snr=snr(tensors[n],read[n]))
            if mse >= 0.1: all_ok = False
        else:
            integ[n] = dict(error='not found'); all_ok = False
    val = QSSFValidator.get_file_info(path)
    try: os.unlink(path)
    except: pass
    return dict(write=dict(file_size=file_size, n=len(tensors)),
                read=dict(n=len(read)), integrity=dict(ok=all_ok, per=integ),
                validation=val)

# ═══════════════════════════════════════════════════════════════════════
# 4. Extreme KV
# ═══════════════════════════════════════════════════════════════════════

def test_kv():
    rng = np.random.RandomState(42)
    pairs = []
    for pos in range(8):
        k = rng.randn(128).astype(np.float32)*0.1
        v = rng.randn(128).astype(np.float32)*0.1
        pairs.append((k, v))

    # Test TimeCrystalKVCache_v2
    from spectralstream.extreme_kv_cache import TimeCrystalKVCache_v2
    print("tc_kv...", end=' ', flush=True)
    tc = TimeCrystalKVCache_v2(head_dim=128, max_entries=16)
    for pos, (k, v) in enumerate(pairs):
        tc.store(k, v, position=pos)
    tc_cos = 0.0; tc_ok = 0
    for pos, (k, v) in enumerate(pairs):
        ret = tc.retrieve(pos)
        if ret:
            rk, rv = ret
            cs = float(np.dot(k.ravel(), rk.ravel()) / (np.linalg.norm(k)*np.linalg.norm(rk)+1e-30))
            tc_cos += cs
            if cs > 0.8: tc_ok += 1
    tc_cos /= max(len(pairs), 1)

    # Test FreqKVExtremeCompressor
    from spectralstream.extreme_kv_cache import FreqKVExtremeCompressor
    print("freqkv...", end=' ', flush=True)
    sq = np.random.RandomState(99).randn(16, 256).astype(np.float32)*0.1
    fkc = FreqKVExtremeCompressor(head_dim=256)
    comp = fkc.compress(sq); dec = fkc.decompress(comp)
    fkc_snr = snr(sq, dec)

    # Test SpectralSuperpositionKVCache
    from spectralstream.extreme_kv_cache import SpectralSuperpositionKVCache
    print("superpos...", end=' ', flush=True)
    sp = SpectralSuperpositionKVCache(head_dim=128, max_entries=16)
    for pos, (k, v) in enumerate(pairs):
        sp.store(k, v, position=pos)
    sp_ok = sum(1 for pos in range(len(pairs)) if sp.retrieve(pos) is not None)

    print(f"tc_cos={tc_cos:.4f} tc_ok={tc_ok}/{len(pairs)} fkc_snr={fkc_snr:.1f}dB sp_ok={sp_ok}/{len(pairs)}")
    return dict(storage=dict(n=len(pairs)),
                retrieval=dict(avg_cos=tc_cos, avg_snr=fkc_snr,
                               pass_rate=tc_ok/max(len(pairs),1)))

# ═══════════════════════════════════════════════════════════════════════
# 5. Comparison
# ═══════════════════════════════════════════════════════════════════════

def test_compare():
    tensors = {
        'attn': gen_tensor((128, 192), 15, 42),
        'ffn': gen_tensor((192, 384), 30, 43),
        'k': gen_tensor((128, 48), 8, 44),
    }
    qice = QuantumIntelligenceCompressionEngine(target_ratio=200.,
        quality_threshold=0.005, max_refinement_iters=1)
    ext = ExtremeCompressor(seed=42)
    val = QICEQualityValidator()
    comp = {}
    for name, t in tensors.items():
        qr = qice.compress(t, name); qd = qice.decompress(qr)
        qm = val.metrics_to_dict(val.evaluate(t, qd))
        er = ext.compress(t, name, 200.); ed = ext.decompress(er)
        em = val.metrics_to_dict(val.evaluate(t, ed))
        comp[name] = dict(
            qice=dict(ratio=qr.compression_ratio, snr=qm['psnr'], rel_err=qm['relative_error']),
            extreme=dict(ratio=er.get('compression_ratio',1), snr=em['psnr'], rel_err=em['relative_error']))
    return comp

# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    global REPORT
    print("="*70)
    print("QICE EXTREME COMPRESSION TEST SUITE")
    print("="*70, "\n")
    sys.stdout.flush()

    # 1
    print("[1/5] Individual compressors...", flush=True)
    t0 = time.perf_counter()
    r1 = test_individual()
    t1 = time.perf_counter()
    for n, r in r1.items():
        print(f"  {n:28s} ratio={r['ratio']:>7.1f}x  SNR={r['snr_db']:>6.1f}dB  err={r['err_pct']:.2f}%")
    print(f"  -> {t1-t0:.2f}s\n", flush=True)
    REPORT['individual'] = dict(results=r1, time=round(t1-t0,4))

    # 2
    print("[2/5] QICE pipeline...", flush=True)
    t0 = time.perf_counter()
    r2 = test_pipeline()
    t1 = time.perf_counter()
    s = r2['summary']
    print(f"  Tensors: {s['n_tensors']}, Ratio: {s['overall_compression_ratio']:.1f}x, "
          f"Avg PSNR: {s['avg_psnr']:.1f}dB")
    for n, pt in r2['per_tensor'].items():
        print(f"    {n:20s} strat={pt['strategy']:16s} ratio={pt['ratio']:>7.1f}x PSNR={pt['psnr']:.1f}")
    print(f"  -> {t1-t0:.2f}s\n", flush=True)
    REPORT['pipeline'] = r2

    # 3
    print("[3/5] QSSF round-trip...", flush=True)
    t0 = time.perf_counter()
    r3 = test_qssf()
    t1 = time.perf_counter()
    print(f"  Write: {r3['write']['n']} tensors, {r3['write']['file_size']} bytes")
    print(f"  Read:  {r3['read']['n']} tensors")
    print(f"  Integrity: {'PASS' if r3['integrity']['ok'] else 'FAIL'}")
    print(f"  Valid: {r3['validation'].get('valid', False)}")
    for n, iv in r3['integrity']['per'].items():
        if 'error' in iv: print(f"    {n}: ERROR")
        else: print(f"    {n}: shape_ok={iv['shape_match']} mse={iv['mse']:.6e}")
    print(f"  -> {t1-t0:.2f}s\n", flush=True)
    REPORT['qssf'] = r3

    # 4
    print("[4/5] Extreme KV cache...", flush=True)
    t0 = time.perf_counter()
    r4 = test_kv()
    t1 = time.perf_counter()
    st = r4['storage']; rt = r4['retrieval']
    print(f"  Stored: {st['n']} entries")
    print(f"  TimeCrystal avg cosine: {rt['avg_cos']:.4f}")
    print(f"  FreqKV Compressor SNR: {rt['avg_snr']:.1f}dB")
    print(f"  Pass rate: {rt['pass_rate']:.1%}")
    print(f"  -> {t1-t0:.2f}s\n", flush=True)
    REPORT['kv'] = r4

    # 5
    print("[5/5] Comparison QICE vs ExtremeCompressor...", flush=True)
    t0 = time.perf_counter()
    r5 = test_compare()
    t1 = time.perf_counter()
    for n, cr in r5.items():
        q = cr['qice']; e = cr['extreme']
        print(f"  {n}:")
        print(f"    QICE:     ratio={q['ratio']:>7.1f}x SNR={q['snr']:>6.1f}dB rel_err={q['rel_err']:.4e}")
        print(f"    Extreme:  ratio={e['ratio']:>7.1f}x SNR={e['snr']:>6.1f}dB rel_err={e['rel_err']:.4e}")
        rw = 'QICE' if q['ratio'] > e['ratio'] else 'Extreme'
        sw = 'QICE' if q['snr'] > e['snr'] else 'Extreme'
        print(f"    -> ratio: {rw}, SNR: {sw}")
    print(f"  -> {t1-t0:.2f}s\n", flush=True)
    REPORT['comparison'] = dict(results=r5, time=round(t1-t0,4))

    # Summary
    total_t = sum(s.get('time',0) for s in REPORT.values() if isinstance(s,dict))
    REPORT['_meta'] = dict(suite='QICE Extreme Compression Test Suite',
                           target='Gemma-4 E2B', total_time_s=round(total_t,4))
    print("="*70)
    print(f"COMPLETE — Total: {total_t:.2f}s")
    print("="*70)

    class _NE(json.JSONEncoder):
        def default(self, o):
            if isinstance(o,(np.integer,)): return int(o)
            if isinstance(o,(np.floating,)): return float(o)
            if isinstance(o,np.ndarray): return o.tolist()
            if isinstance(o,(np.bool_,)): return bool(o)
            if isinstance(o,bytes): return f"bytes({len(o)})"
            if isinstance(o,Path): return str(o)
            return super().default(o)
    print(json.dumps(REPORT, cls=_NE, indent=2))

if __name__ == '__main__':
    main()
