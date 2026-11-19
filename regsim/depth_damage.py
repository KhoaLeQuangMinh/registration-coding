"""
Depth damage for the generator: a depth map with a chosen mean surface-tilt error, of one of three kinds
(bumpy = smooth random field, flattened = squeezed toward the median, slanted = tilted plane); the size is found by
bisection. From the shape-trust study (archive branch, goal/shape_trust.py).
"""
import numpy as np
from scipy.ndimage import gaussian_filter

from regsim.p2ilreg import RH, RW, normals


def mean_angle(Na, oka, Nb, okb):
    both = oka & okb
    return float(np.degrees(np.arccos(np.clip((Na[both] * Nb[both]).sum(-1), -1, 1))).mean()) if both.sum() > 20 else 0.0


def distort(Z0, S0, K, kind, deg, rng_seed):
    """Depth map with mean surface-tilt error `deg` (bisection on the distortion size)."""
    N0, ok0 = normals(Z0, K)
    m = np.median(Z0[S0])
    rng = np.random.default_rng(rng_seed)
    if kind == "bumpy":
        G = gaussian_filter(rng.standard_normal(Z0.shape), sigma=RW / 10)
        G /= G[S0].std()
        make = lambda a: Z0 + a * m * G          # noqa: E731
        lo, hi, grows = 0.0, 1.0, True
    elif kind == "flattened":
        make = lambda k: m + k * (Z0 - m)        # noqa: E731
        lo, hi, grows = 0.0, 1.0, False          # smaller k = flatter = larger error
    else:
        u, v = np.meshgrid(np.arange(RW) + 0.5, np.arange(RH) + 0.5)
        phi = rng.uniform(0, 2 * np.pi)
        P = np.cos(phi) * (u - K[0, 2]) / K[0, 0] + np.sin(phi) * (v - K[1, 2]) / K[1, 1]
        P /= P[S0].std()
        make = lambda a: Z0 * (1 + a * P)        # noqa: E731
        lo, hi, grows = 0.0, 0.5, True

    def err(x):
        Z = np.where(S0, np.maximum(make(x), 0.2 * m), 0.0)
        N, ok = normals(Z, K)
        return mean_angle(N, ok & S0, N0, ok0 & S0), Z

    if deg == 0:
        return Z0.copy(), 0.0
    if not grows:
        e0, Zc = err(0.0)
        if e0 < deg:                             # flattening cannot reach this error: use the flattest map
            return Zc, e0
    for _ in range(30):
        x = (lo + hi) / 2
        e, Z = err(x)
        if (e < deg) == grows:
            lo = x
        else:
            hi = x
    return err((lo + hi) / 2)[::-1]
