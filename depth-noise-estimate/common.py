"""
Shared helpers: SERV-CT loading, depth alignment, surface normals and the shape-error decomposition.
"""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

EXPERIMENTS = ("Experiment_1", "Experiment_2")   # 30-degree scope, straight scope (two ex vivo pigs)
NO_REFERENCE = ((255, 255, 0), (0, 0, 255))      # occlusion colours: yellow = no stereo overlap, blue = outside the CT surface
FX_TARGET = 343.0   # focal length (px) of our generator's 480-px render with a 70-degree lens: errors are measured at
                    # the same angular resolution, so their degrees compare with the generator's damage settings


def frames(root):
    """(experiment, frame id) for every left image."""
    for exp in EXPERIMENTS:
        for f in sorted((Path(root) / exp / "Left_rectified").glob("*.png")):
            yield exp, f.stem


def load_frame(root, exp, fid):
    """Left image, CT reference depth (mm), pixels with a reference, and the rectified camera matrix K."""
    d = Path(root) / exp
    rgb = np.array(Image.open(d / "Left_rectified" / f"{fid}.png").convert("RGB"))
    depth = np.array(Image.open(d / "Reference_CT" / "DepthL" / f"{fid}.png")).astype(np.float64) / 256.0
    occ = np.array(Image.open(d / "Reference_CT" / "OcclusionL" / f"{fid}.png").convert("RGB"))
    no_ref = np.zeros(depth.shape, bool)
    for colour in NO_REFERENCE:
        no_ref |= np.all(occ == colour, axis=-1)
    P1 = json.load(open(d / "Rectified_calibration" / f"{fid}.json"))["P1"]["data"]
    K = np.array(P1, float).reshape(3, 4)[:, :3]
    return rgb, depth, (depth > 0) & ~no_ref, K


def align_disparity(pred, depth, valid):
    """
    Depth Anything V2 (relative) predicts disparity up to an unknown scale and shift. Fit both to the true inverse
    depth (least squares on the reference pixels) and return the aligned depth in mm (NaN where the fit is not positive).
    """
    A = np.stack([pred[valid], np.ones(valid.sum())], 1)
    (s, t), *_ = np.linalg.lstsq(A, 1.0 / depth[valid], rcond=None)
    disp = s * pred + t
    return np.where(disp > 1e-9, 1.0 / np.maximum(disp, 1e-9), np.nan), (float(s), float(t))


def resample(depth, valid, K, fx_target=FX_TARGET):
    """Area-resample a depth map to the focal length fx_target, keeping only cells fully covered by valid pixels."""
    k = fx_target / K[0, 0]
    H, W = depth.shape
    size = (max(8, round(W * k)), max(8, round(H * k)))
    w = cv2.resize(valid.astype(np.float64), size, interpolation=cv2.INTER_AREA)
    z = cv2.resize(np.where(valid, depth, 0.0), size, interpolation=cv2.INTER_AREA)
    ok = w > 0.999
    Ks = K.copy()
    Ks[:2] *= np.array([[size[0] / W], [size[1] / H]])
    return np.where(ok, z / np.maximum(w, 1e-9), 0.0), ok, Ks


def normals(Z, K):
    """
    Surface normals from a depth map (scale-free: multiplying Z leaves them unchanged) and where they are valid
    (all four neighbours on the surface, no depth jump). Same definition as regsim.p2ilreg.normals.
    """
    H, W = Z.shape
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    P = np.stack([(u - K[0, 2]) / K[0, 0] * Z, (v - K[1, 2]) / K[1, 1] * Z, Z], -1)
    dx, dy = np.zeros_like(P), np.zeros_like(P)
    dx[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dy[1:-1] = P[2:] - P[:-2]
    n = np.cross(dx, dy)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    ok = Z > 0
    for axis, shift in [(0, 1), (0, -1), (1, 1), (1, -1)]:
        nb = np.roll(Z, shift, axis=axis)
        ok &= (nb > 0) & (np.abs(nb - Z) < 0.05 * Z)
    ok[0] = ok[-1] = ok[:, 0] = ok[:, -1] = False
    return n, ok


def mean_tilt(Za, Zb, K, mask):
    """Mean angle (degrees) between the surface normals of two depth maps, over mask."""
    na, oka = normals(Za, K)
    nb, okb = normals(Zb, K)
    both = oka & okb & mask
    if both.sum() < 50:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip((na[both] * nb[both]).sum(-1), -1, 1))).mean())


def bump_size(resid, mask):
    """Size of the bumps: distance (px) at which the residual's autocorrelation falls to one half."""
    r = np.where(mask, resid - resid[mask].mean(), 0.0)
    m = mask.astype(np.float64)
    f = lambda a: np.real(np.fft.ifft2(np.abs(np.fft.fft2(a, s=(2 * a.shape[0], 2 * a.shape[1]))) ** 2))  # noqa: E731
    ac = f(r) / np.maximum(f(m), 1)
    ac = np.fft.fftshift(ac / ac[0, 0])
    cy, cx = np.array(ac.shape) // 2
    yy, xx = np.indices(ac.shape)
    rad = np.hypot(yy - cy, xx - cx).astype(int)
    prof = np.bincount(rad.ravel(), ac.ravel()) / np.maximum(np.bincount(rad.ravel()), 1)
    below = np.where(prof < 0.5)[0]
    return float(below[0]) if len(below) else float("nan")


def decompose(z_true, z_pred, ok, K):
    """
    Split the shape error of an aligned prediction into the three kinds the generator simulates. Both depths are
    divided by their median (scale is never used), then  z_pred ~ c + k (z_true - 1) + a x + b y + residual :
        flattening  k        (k < 1: relief too shallow)          -> tilt error of  1 + k (z_true - 1)
        slant       a, b     (a tilted plane across the picture)  -> tilt error of  z_true + a x + b y
        bumps       residual (local dents and bulges)             -> tilt error of  z_true + residual
        total                                                     -> tilt error of  z_pred
    x, y are normalised image coordinates.
    """
    zt = z_true / np.median(z_true[ok])
    zp = z_pred / np.median(z_pred[ok])
    H, W = zt.shape
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    x, y = (u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1]
    A = np.stack([np.ones(ok.sum()), zt[ok] - 1, x[ok], y[ok]], 1)
    (c, k, a, b), *_ = np.linalg.lstsq(A, zp[ok], rcond=None)
    resid = np.where(ok, zp - (c + k * (zt - 1) + a * x + b * y), 0.0)
    keep = lambda Z: np.where(ok, Z, 0.0)  # noqa: E731
    return dict(
        flattening_k=float(k),
        slant_per_unit=float(np.hypot(a, b)),
        bump_rms=float(resid[ok].std()),
        bump_size_px=bump_size(resid, ok),
        tilt_total_deg=mean_tilt(keep(zp), keep(zt), K, ok),
        tilt_flattening_deg=mean_tilt(keep(1 + k * (zt - 1)), keep(zt), K, ok),
        tilt_slant_deg=mean_tilt(keep(zt + a * x + b * y), keep(zt), K, ok),
        tilt_bumps_deg=mean_tilt(keep(zt + resid), keep(zt), K, ok),
    ), resid
