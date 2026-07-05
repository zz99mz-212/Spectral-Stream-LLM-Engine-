cmake_minimum_required(VERSION 3.20)
project(SpectralStream VERSION 1.0.0 LANGUAGES C CXX)

# ──────────────────────────────────────────────────────────────────────────────
# NOTE: Custom C++ SIMD targets have been removed.
# All SIMD-accelerated kernels (FWHT, DCT, INT4 GEMM, Lloyd-Max) are now
# implemented in pure Python via NumPy's vectorised C/Fortran internals
# (pocketfft, BLAS, einsum).  No pybind11 or C++ compilation is needed.
#
# If you need to re-add C++ extensions later, uncomment the blocks below
# and restore the source files in spectralstream/cpp_ext/.
# ──────────────────────────────────────────────────────────────────────────────

# ---- Optional: llama.cpp submodule ----
# Uncomment to build with embedded llama.cpp instead of llama-cpp-python
# add_subdirectory(llama.cpp EXCLUDE_FROM_ALL)
