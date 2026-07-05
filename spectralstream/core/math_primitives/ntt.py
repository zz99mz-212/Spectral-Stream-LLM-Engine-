"""Number Theoretic Transform (exact integer convolution)."""

from typing import Dict, List


class NTT:
    """NTT for exact integer convolution. Uses NTT-friendly primes where p = c * 2^k + 1."""

    PRIMES: Dict[int, int] = {
        256: 257,
        512: 7681,
        1024: 12289,
        2048: 8380417,
        4096: 167772161,
    }

    def __init__(self, n: int = 1024):
        if n not in self.PRIMES:
            raise ValueError(
                f"n={n} not supported. Choose from {sorted(self.PRIMES.keys())}"
            )
        self.n = n
        self.p = self.PRIMES[n]
        self.g = self._find_primitive_root(self.p)
        self.g_n = pow(self.g, (self.p - 1) // self.n, self.p)
        self.g_n_inv = pow(self.g_n, self.p - 2, self.p)
        self.n_inv = pow(self.n, self.p - 2, self.p)

    @staticmethod
    def _is_primitive_root(g: int, p: int) -> bool:
        if g <= 1 or g >= p:
            return False
        order = p - 1
        factors = set()
        n = order
        f = 2
        while f * f <= n:
            if n % f == 0:
                factors.add(f)
                while n % f == 0:
                    n //= f
            f += 1
        if n > 1:
            factors.add(n)
        return all(pow(g, order // q, p) != 1 for q in factors)

    @staticmethod
    def _find_primitive_root(p: int) -> int:
        for g in range(2, p):
            if NTT._is_primitive_root(g, p):
                return g
        raise ValueError(f"No primitive root found for p={p}")

    def _bit_reverse(self, x: int, bits: int) -> int:
        result = 0
        for _ in range(bits):
            result = (result << 1) | (x & 1)
            x >>= 1
        return result

    def ntt(self, a: list) -> list:
        a = [int(x) % self.p for x in a]
        n = self.n
        bits = n.bit_length() - 1
        for i in range(n):
            ri = self._bit_reverse(i, bits)
            if i < ri:
                a[i], a[ri] = a[ri], a[i]
        length = 2
        while length <= n:
            w = pow(self.g_n, n // length, self.p)
            for start in range(0, n, length):
                wn = 1
                half = length // 2
                for j in range(half):
                    u = a[start + j]
                    v = a[start + j + half] * wn % self.p
                    a[start + j] = (u + v) % self.p
                    a[start + j + half] = (u - v) % self.p
                    wn = (wn * w) % self.p
            length <<= 1
        return a

    def intt(self, a: list) -> list:
        n = self.n
        a = self.ntt(a)
        a = [x * self.n_inv % self.p for x in a]
        a = a[:1] + a[:0:-1] if len(a) > 1 else a
        return a

    def convolve(self, a: list, b: list) -> list:
        n = self.n
        a_pad = a + [0] * (n - len(a))
        b_pad = b + [0] * (n - len(b))
        A = self.ntt(a_pad)
        B = self.ntt(b_pad)
        C = [(A[i] * B[i]) % self.p for i in range(n)]
        return self.intt(C)
