"""PRNG utilities: splitmix64, next_power_of_two."""

_SPLITMIX_CONST1 = 0x9E3779B97F4A7C15
_SPLITMIX_CONST2 = 0xBF58476D1CE4E5B9
_SPLITMIX_CONST3 = 0x94D049BB133111EB


def splitmix64(value: int) -> int:
    v = int(value) & 0xFFFFFFFFFFFFFFFF
    v = (v + _SPLITMIX_CONST1) & 0xFFFFFFFFFFFFFFFF
    v = ((v ^ (v >> 30)) * _SPLITMIX_CONST2) & 0xFFFFFFFFFFFFFFFF
    v = ((v ^ (v >> 27)) * _SPLITMIX_CONST3) & 0xFFFFFFFFFFFFFFFF
    return v ^ (v >> 31)


def next_power_of_two(n: int) -> int:
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return 1 << (n - 1).bit_length()
