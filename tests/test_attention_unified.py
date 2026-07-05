import sys

sys.path.insert(0, ".")
try:
    import numpy as np
    from spectralstream.attention.unified_attention import (
        VlasovMeanFieldAttention,
        VlasovFlashAttention,
        GyrokineticAttention,
        SymplecticAttentionIntegrator,
        VlasovHelmholtzDecomposition,
        VlasovAttentionLayer,
        UnifiedAttentionSelector,
        TurbulentCascadeAttention,
        EchoAttention,
        VlasovBlock,
    )
except ImportError as e:
    print(f"Import error: {e}")
    raise


class TestVlasovMeanFieldAttention:
    def test_forward_basic(self):
        attn = VlasovMeanFieldAttention(d_model=64, n_grid=16, n_heads=4)
        q = np.random.randn(8, 16).astype(np.float32)
        k = np.random.randn(8, 16).astype(np.float32)
        v = np.random.randn(8, 16).astype(np.float32)
        out = attn.forward(q, k, v)
        assert out.shape == q.shape

    def test_forward_causal(self):
        attn = VlasovMeanFieldAttention(d_model=64, n_grid=16, causal=True, n_heads=4)
        q = np.random.randn(5, 16).astype(np.float32)
        k = np.random.randn(5, 16).astype(np.float32)
        v = np.random.randn(5, 16).astype(np.float32)
        out = attn.forward(q, k, v)
        assert out.shape == q.shape

    def test_forward_with_mask(self):
        attn = VlasovMeanFieldAttention(d_model=64, n_grid=16, n_heads=4)
        q = np.random.randn(6, 16).astype(np.float32)
        k = np.random.randn(6, 16).astype(np.float32)
        v = np.random.randn(6, 16).astype(np.float32)
        mask = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
        out = attn.forward(q, k, v, mask=mask)
        assert out.shape == q.shape

    def test_spectral_forward(self):
        attn = VlasovMeanFieldAttention(d_model=64, n_grid=16, n_heads=4)
        q = np.random.randn(4, 16).astype(np.float32)
        k = np.random.randn(4, 16).astype(np.float32)
        v = np.random.randn(4, 16).astype(np.float32)
        out = attn.spectral_forward(q, k, v, spectral_rank=4)
        assert out.shape == q.shape

    def test_compute_potential(self):
        attn = VlasovMeanFieldAttention(d_model=64, n_grid=16, n_heads=4)
        k = np.random.randn(8, 16).astype(np.float32)
        phi = attn.compute_potential(k)
        assert phi.shape == (attn.n_grid,)

    def test_return_potential(self):
        attn = VlasovMeanFieldAttention(d_model=64, n_grid=16, n_heads=4)
        q = np.random.randn(4, 16).astype(np.float32)
        k = np.random.randn(4, 16).astype(np.float32)
        v = np.random.randn(4, 16).astype(np.float32)
        out, phi = attn.forward(q, k, v, return_potential=True)
        assert out.shape == q.shape

    def test_2d_input(self):
        attn = VlasovMeanFieldAttention(d_model=64, n_grid=16, n_heads=4)
        q = np.random.randn(3, 16).astype(np.float32)
        k = np.random.randn(3, 16).astype(np.float32)
        v = np.random.randn(3, 16).astype(np.float32)
        out = attn.forward(q, k, v)
        assert out.ndim == 2

    def test_invalid_1d_input(self):
        attn = VlasovMeanFieldAttention(d_model=64, n_grid=16)
        q = np.random.randn(16).astype(np.float32)
        k = np.random.randn(16).astype(np.float32)
        v = np.random.randn(16).astype(np.float32)
        try:
            attn.forward(q, k, v)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_reset_causal_state(self):
        attn = VlasovMeanFieldAttention(d_model=64, n_grid=16)
        attn.reset_causal_state()
        assert attn._causal_state is None


class TestVlasovFlashAttention:
    def test_forward_small(self):
        attn = VlasovFlashAttention(d_model=64, n_grid=16, block_size=32, n_heads=4)
        q = np.random.randn(8, 16).astype(np.float32)
        k = np.random.randn(8, 16).astype(np.float32)
        v = np.random.randn(8, 16).astype(np.float32)
        out = attn.forward(q, k, v)
        assert out.shape == q.shape

    def test_forward_large(self):
        attn = VlasovFlashAttention(d_model=64, n_grid=16, block_size=4, n_heads=4)
        q = np.random.randn(12, 16).astype(np.float32)
        k = np.random.randn(12, 16).astype(np.float32)
        v = np.random.randn(12, 16).astype(np.float32)
        out = attn.forward(q, k, v)
        assert out.shape == q.shape


class TestGyrokineticAttention:
    def test_forward_basic(self):
        attn = GyrokineticAttention(d_model=64, n_grid=16, n_heads=4)
        q = np.random.randn(6, 16).astype(np.float32)
        k = np.random.randn(6, 16).astype(np.float32)
        v = np.random.randn(6, 16).astype(np.float32)
        out = attn.forward(q, k, v)
        assert out.shape == q.shape

    def test_gyrokinetic_split(self):
        attn = GyrokineticAttention(d_model=64, n_grid=16, n_heads=4)
        k = np.random.randn(4, 16).astype(np.float32)
        v = np.random.randn(4, 16).astype(np.float32)
        ks, vs, kf, vf = attn._gyrokinetic_split(k, v)
        assert ks.shape == k.shape
        assert kf.shape == k.shape


class TestSymplecticAttentionIntegrator:
    def test_leapfrog_step(self):
        integrator = SymplecticAttentionIntegrator(dt=0.1, n_substeps=2)
        x = np.random.randn(4, 8).astype(np.float64)
        momentum = np.zeros_like(x)
        identity_force = lambda q: q
        x_new, m_new = integrator.leapfrog_step(x, momentum, identity_force)
        assert x_new.shape == x.shape
        assert m_new.shape == momentum.shape

    def test_integrate_layer(self):
        integrator = SymplecticAttentionIntegrator(dt=0.05)
        x = np.random.randn(4, 8).astype(np.float64)
        identity_force = lambda q: q
        x_new, m_new = integrator.integrate_layer(x, force_fn=identity_force)
        assert x_new.shape == x.shape

    def test_energy_calculation(self):
        integrator = SymplecticAttentionIntegrator()
        x = np.random.randn(4, 8)
        momentum = np.random.randn(4, 8)
        attn_out = np.random.randn(4, 8)
        e = integrator.total_energy(x, momentum, attn_out)
        assert isinstance(e, float)

    def test_reset(self):
        integrator = SymplecticAttentionIntegrator(hamiltonian_monitor=True)
        integrator._energy_history = [1.0, 2.0]
        integrator.reset()
        assert len(integrator._energy_history) == 0


class TestVlasovHelmholtzDecomposition:
    def test_decompose(self):
        dec = VlasovHelmholtzDecomposition(d_model=64, spectral_rank=8)
        field = np.random.randn(6, 16).astype(np.float64)
        irr, sol = dec.decompose(field)
        assert irr.shape == field.shape
        assert sol.shape == field.shape

    def test_combine(self):
        dec = VlasovHelmholtzDecomposition(d_model=64)
        irr = np.random.randn(4, 16)
        sol = np.random.randn(4, 16)
        combined = dec.combine(irr, sol)
        assert combined.shape == irr.shape

    def test_spectral_decompose(self):
        dec = VlasovHelmholtzDecomposition(d_model=64, spectral_rank=4)
        field = np.random.randn(4, 16).astype(np.float64)
        irr, sol, spec = dec.spectral_decompose(field)
        assert irr.shape == field.shape
        assert sol.shape == field.shape


class TestUnifiedAttentionSelector:
    def test_select_short(self):
        sel = UnifiedAttentionSelector(d_model=64, n_grid=16, n_heads=4)
        q = np.random.randn(100, 16).astype(np.float32)
        k = np.random.randn(100, 16).astype(np.float32)
        v = np.random.randn(100, 16).astype(np.float32)
        out = sel.forward(q, k, v)
        assert out.shape == q.shape

    def test_select_medium(self):
        sel = UnifiedAttentionSelector(d_model=64, n_grid=16, n_heads=4)
        q = np.random.randn(1000, 16).astype(np.float32)
        k = np.random.randn(1000, 16).astype(np.float32)
        v = np.random.randn(1000, 16).astype(np.float32)
        out = sel.forward(q, k, v)
        assert out.shape == q.shape
