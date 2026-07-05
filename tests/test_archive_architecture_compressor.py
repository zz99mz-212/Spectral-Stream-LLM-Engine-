"""
Test Architecture-Aware Compression on Real Gemma-4 Tensors
==========================================================
Validates the architecture compressor on actual model weights,
measuring per-layer compression ratios and error.
"""

import json
import logging
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, '/home/mike/Documents/Github/SpectralStream')

from spectralstream.compression.architecture_compressor import (
    Gemma4Architecture,
    ArchitectureAwareCompressor,
    PerLayerEmbeddingCompressor,
    GlobalAttentionOptimizer,
    GLOBAL_ATTENTION_LAYERS,
    GEMMA4_E2B_ARCH,
)
from spectralstream.compression.unified_quant_system import (
    _compress_int8, _decompress_int8,
    _compress_int4, _decompress_int4,
    _error_metrics,
)

logging.basicConfig(level=logging.INFO, format='%(name)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

GEMMA4_PATH = '/home/mike/Documents/Github/SpectralStream/models/gemma-4-E2B/model.safetensors'


def load_safetensors_info(path):
    with open(path, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header_json = f.read(header_len)
    return json.loads(header_json)


def load_tensor(path, key, info):
    ti = info[key]
    dtype_str = ti['dtype']
    shape = ti['shape']
    begin, end = ti['data_offsets']
    header_len = 8 + len(json.dumps(info).encode('utf-8'))
    data_start = header_len + begin
    data_len = end - begin
    dtype_map = {'BF16': 'uint16', 'F32': 'float32', 'F16': 'float16'}
    np_dtype = dtype_map.get(dtype_str, 'float32')
    with open(path, 'rb') as f:
        f.seek(data_start)
        raw = f.read(data_len)
    if dtype_str == 'BF16':
        raw_u16 = np.frombuffer(raw, dtype=np.uint16)
        f32 = (raw_u16.astype(np.uint32) << 16).view(np.float32)
        return f32.reshape(shape).astype(np.float32)
    elif dtype_str == 'F16':
        return np.frombuffer(raw, dtype=np.float16).reshape(shape).astype(np.float32)
    else:
        return np.frombuffer(raw, dtype=np.float32).reshape(shape)


def test_architecture_parser():
    """Test that Gemma4Architecture correctly identifies layer types."""
    print('=' * 70)
    print('TEST 1: Architecture Parser')
    print('=' * 70)

    arch = Gemma4Architecture()

    # Check layer types
    global_layers = set()
    sliding_layers = set()
    for i in range(35):
        lt = arch.get_layer_type(i)
        if lt == 'global_attention':
            global_layers.add(i)
        else:
            sliding_layers.add(i)

    expected_global = {14, 19, 24, 29, 34}
    assert global_layers == expected_global, f'Global layers mismatch: {global_layers} != {expected_global}'
    print(f'  Global attention layers:  {sorted(global_layers)}')
    print(f'  Sliding window layers:    {sorted(sliding_layers)[:10]}...')
    print(f'  Layer types correct:      PASS')

    # Check tensor classification
    test_names = [
        'model.language_model.layers.14.self_attn.q_proj.weight',
        'model.language_model.layers.0.self_attn.q_proj.weight',
        'model.language_model.layers.5.mlp.gate_proj.weight',
        'model.language_model.layers.0.input_norm.weight',
        'model.embed_tokens.weight',
    ]
    print()
    print('  Tensor Classification:')
    for name in test_names:
        cls = arch.classify_tensor(name)
        print(f'    {name.split(".")[-3]}.{name.split(".")[-1]:20s} -> '
              f'{cls["role"]:25s} imp={cls["importance"]:.2f} '
              f'{"GLOBAL" if cls["is_global"] else "SLIDING"}')
    print()


def test_global_attention_optimizer():
    """Test error budget allocation for global vs sliding layers."""
    print('=' * 70)
    print('TEST 2: Global Attention Optimizer')
    print('=' * 70)

    opt = GlobalAttentionOptimizer()

    components = ['attention_q', 'attention_k', 'attention_v', 'attention_o',
                  'ffn_gate', 'ffn_up', 'ffn_down', 'norm']

    print(f'  {"Component":20s} {"Global Budget":15s} {"Sliding Budget":15s} {"Method":10s}')
    print(f'  {"-"*20} {"-"*15} {"-"*15} {"-"*10}')

    for comp in components:
        g_budget = opt.get_error_budget(comp, is_global=True, importance=1.0)
        s_budget = opt.get_error_budget(comp, is_global=False, importance=0.8)
        method = opt.select_method(comp, is_global=True, importance=1.0, tensor_size_bytes=1024*1024)
        print(f'  {comp:20s} {g_budget*100:13.2f}% {s_budget*100:13.2f}% {method:10s}')

    print()
    print('  Global layers get tighter budgets (lower error allowed)')
    print('  Sliding layers get relaxed budgets (2x more error allowed)')
    print()


def test_block_int8_on_real_tensors():
    """Test Block INT8 compression on real Gemma-4 weight tensors."""
    print('=' * 70)
    print('TEST 3: Block INT8 on Real Gemma-4 Tensors')
    print('=' * 70)

    if not os.path.exists(GEMMA4_PATH):
        print(f'  SKIPPED: Model not found at {GEMMA4_PATH}')
        return

    info = load_safetensors_info(GEMMA4_PATH)
    arch = Gemma4Architecture()

    # Test tensors from different layers
    test_configs = [
        ('model.language_model.layers.0.self_attn.q_proj.weight', 'Sliding Q proj'),
        ('model.language_model.layers.14.self_attn.q_proj.weight', 'Global Q proj'),
        ('model.language_model.layers.0.mlp.gate_proj.weight', 'Small FFN gate'),
        ('model.language_model.layers.20.mlp.gate_proj.weight', 'Large FFN gate'),
        ('model.language_model.layers.0.input_norm.weight', 'Norm'),
    ]

    print(f'  {"Tensor":30s} {"Shape":20s} {"Original":>10s} {"INT8":>8s} {"INT4":>8s} '
          f'{"INT8 Err":>10s} {"INT4 Err":>10s} {"Role":15s}')
    print(f'  {"-"*30} {"-"*20} {"-"*10} {"-"*8} {"-"*8} {"-"*10} {"-"*10} {"-"*15}')

    for key, label in test_configs:
        if key not in info:
            continue
        tensor = load_tensor(GEMMA4_PATH, key, info)
        cls = arch.classify_tensor(key)
        shape_str = f'{tensor.shape[0]}x{tensor.shape[1]}' if tensor.ndim >= 2 else str(tensor.shape)

        # Block INT8
        t0 = time.time()
        block8, _ = _compress_int8(tensor)
        t8 = time.time() - t0
        decomp8 = _decompress_int8(block8, tensor.size).reshape(tensor.shape)
        m8 = _error_metrics(tensor, decomp8)

        # Block INT4
        t0 = time.time()
        block4, _ = _compress_int4(tensor)
        t4 = time.time() - t0
        decomp4 = _decompress_int4(block4, tensor.size).reshape(tensor.shape)
        m4 = _error_metrics(tensor, decomp4)

        orig_mb = tensor.nbytes / (1024 * 1024)
        int8_mb = len(block8) / (1024 * 1024)
        int4_mb = len(block4) / (1024 * 1024)

        print(f'  {label:30s} {shape_str:20s} {orig_mb:8.2f}MB {int8_mb:6.2f}MB {int4_mb:6.2f}MB '
              f'{m8["rel_error"]*100:8.4f}% {m4["rel_error"]*100:8.4f}% {cls["role"]:15s}')

    print()


def test_per_layer_embedding_compressor():
    """Test special handling for per-layer embeddings."""
    print('=' * 70)
    print('TEST 4: Per-Layer Embedding Compressor')
    print('=' * 70)

    ple_comp = PerLayerEmbeddingCompressor()

    # Simulate a per-layer embedding (262144 x 256 for speed)
    # Real is 262144 x 8960, but we use smaller for testing
    rng = np.random.RandomState(42)
    n_vocab = 262144
    n_dim = 256

    # Create realistic embedding: most rows are small, few are large
    tensor = rng.randn(n_vocab, n_dim).astype(np.float32) * 0.02
    # Make ~40% of rows near-zero (rare tokens)
    sparse_mask = rng.random(n_vocab) < 0.4
    tensor[sparse_mask] *= 0.001

    orig_bytes = tensor.nbytes
    print(f'  Simulated PLE: shape={tensor.shape}, {orig_bytes / (1024*1024):.1f} MB')
    print(f'  Sparse rows: {np.mean(sparse_mask)*100:.0f}%')

    # Strategy 1: Sparse + INT4
    block_sp, meta_sp = ple_comp.compress(tensor, name='test_ple', target_ratio=4.0)
    print(f'  Sparse+INT4: {len(block_sp) / (1024*1024):.2f} MB, '
          f'ratio={orig_bytes / max(len(block_sp), 1):.2f}x, '
          f'error={meta_sp["rel_error"]*100:.4f}%')

    # Strategy 2: Plain INT4
    block_int4, meta_int4 = ple_comp._compress_aggressive_int4(tensor, name='test_ple')
    print(f'  Plain INT4:  {len(block_int4) / (1024*1024):.2f} MB, '
          f'ratio={orig_bytes / max(len(block_int4), 1):.2f}x, '
          f'error={meta_int4["rel_error"]*100:.4f}%')

    # Strategy 3: INT8 for reference
    block_int8, _ = _compress_int8(tensor)
    decomp8 = _decompress_int8(block_int8, tensor.size).reshape(tensor.shape)
    m8 = _error_metrics(tensor, decomp8)
    print(f'  INT8 ref:    {len(block_int8) / (1024*1024):.2f} MB, '
          f'ratio={orig_bytes / max(len(block_int8), 1):.2f}x, '
          f'error={m8["rel_error"]*100:.4f}%')

    print()
    print(f'  PLE savings: '
          f'{orig_bytes / (1024*1024):.1f}MB -> '
          f'{len(block_sp) / (1024*1024):.2f}MB '
          f'({orig_bytes / max(len(block_sp), 1):.1f}x compression)')
    print()


def test_architecture_aware_vs_naive():
    """Compare architecture-aware vs naive compression."""
    print('=' * 70)
    print('TEST 5: Architecture-Aware vs Naive Compression')
    print('=' * 70)

    if not os.path.exists(GEMMA4_PATH):
        print(f'  SKIPPED: Model not found at {GEMMA4_PATH}')
        return

    info = load_safetensors_info(GEMMA4_PATH)
    arch = Gemma4Architecture()
    opt = GlobalAttentionOptimizer()

    # Test on a subset of tensors from different layers
    test_keys = []
    for key in sorted(info.keys()):
        if key == '__metadata__':
            continue
        if 'layers.0.' in key or 'layers.14.' in key or 'layers.20.' in key:
            test_keys.append(key)
            if len(test_keys) >= 15:
                break

    naive_total_orig = 0
    naive_total_comp = 0
    naive_errors = []

    aware_total_orig = 0
    aware_total_comp = 0
    aware_errors = []

    for key in test_keys:
        tensor = load_tensor(GEMMA4_PATH, key, info)
        cls = arch.classify_tensor(key)

        # Naive: always INT8
        block_n, _ = _compress_int8(tensor)
        decomp_n = _decompress_int8(block_n, tensor.size).reshape(tensor.shape)
        m_n = _error_metrics(tensor, decomp_n)
        naive_total_orig += tensor.nbytes
        naive_total_comp += len(block_n)
        naive_errors.append(m_n['rel_error'])

        # Architecture-aware: INT8 for global, INT4 for sliding
        if cls['is_global'] or cls['importance'] >= 0.9:
            method = 'int8'
        else:
            method = 'int4'

        if method == 'int8':
            block_a, _ = _compress_int8(tensor)
            decomp_a = _decompress_int8(block_a, tensor.size).reshape(tensor.shape)
        else:
            block_a, _ = _compress_int4(tensor)
            decomp_a = _decompress_int4(block_a, tensor.size).reshape(tensor.shape)
        m_a = _error_metrics(tensor, decomp_a)
        aware_total_orig += tensor.nbytes
        aware_total_comp += len(block_a)
        aware_errors.append(m_a['rel_error'])

    naive_ratio = naive_total_orig / max(naive_total_comp, 1)
    aware_ratio = aware_total_orig / max(aware_total_comp, 1)
    naive_avg_err = sum(naive_errors) / max(len(naive_errors), 1)
    aware_avg_err = sum(aware_errors) / max(len(aware_errors), 1)

    print(f'  Naive (all INT8):      ratio={naive_ratio:.2f}x, avg_error={naive_avg_err*100:.4f}%')
    print(f'  Architecture-aware:    ratio={aware_ratio:.2f}x, avg_error={aware_avg_err*100:.4f}%')
    print(f'  Improvement:           {(aware_ratio / naive_ratio - 1) * 100:+.1f}% ratio, '
          f'{(aware_avg_err / naive_avg_err - 1) * 100:+.1f}% error')
    print()


def test_full_model_compression():
    """Full model compression with architecture awareness."""
    print('=' * 70)
    print('TEST 6: Full Model Architecture-Aware Compression')
    print('=' * 70)

    if not os.path.exists(GEMMA4_PATH):
        print(f'  SKIPPED: Model not found at {GEMMA4_PATH}')
        return

    output_path = '/tmp/test_arch_compressed.sscx'

    compressor = ArchitectureAwareCompressor()
    report = compressor.compress_model(
        GEMMA4_PATH,
        output_path,
        target_ratio=4.0,
        max_error=0.01,
    )

    print(report.summary())
    print()

    # Per-layer breakdown
    print('  Per-Layer Compression:')
    print(f'  {"Layer":>6s} {"Type":18s} {"Tensors":>8s} {"Orig":>10s} {"Comp":>10s} '
          f'{"Ratio":>7s} {"Avg Err":>10s}')
    print(f'  {"-"*6} {"-"*18} {"-"*8} {"-"*10} {"-"*10} {"-"*7} {"-"*10}')

    for layer_idx in sorted(report.per_layer.keys()):
        lr = report.per_layer[layer_idx]
        orig_mb = lr.total_original_bytes / (1024 * 1024)
        comp_mb = lr.total_compressed_bytes / (1024 * 1024)
        print(f'  {layer_idx:6d} {lr.layer_type:18s} {len(lr.tensors):8d} '
              f'{orig_mb:8.2f}MB {comp_mb:8.2f}MB {lr.avg_ratio:7.2f} '
              f'{lr.avg_error*100:8.4f}%')

    # Clean up
    if os.path.exists(output_path):
        os.remove(output_path)

    print()


if __name__ == '__main__':
    test_architecture_parser()
    test_global_attention_optimizer()
    test_block_int8_on_real_tensors()
    test_per_layer_embedding_compressor()
    test_architecture_aware_vs_naive()
    test_full_model_compression()

    print('=' * 70)
    print('ALL TESTS COMPLETE')
    print('=' * 70)
