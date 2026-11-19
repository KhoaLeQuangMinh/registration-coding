"""
Measure Depth Anything V2's SHAPE error on SERV-CT, split into the three kinds the generator simulates
(flattening, slant, bumps), in degrees of surface-tilt error at the generator's angular resolution.

Per frame: align the prediction to the CT depth (scale + shift of the disparity), resample both to FX_TARGET,
decompose (common.decompose). Writes summary.csv, summary.json and one figure per frame (figures/).

Usage: python analyse.py --data <SERV-CT folder> --pred <predictions folder> --out <results folder>
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

from common import FX_TARGET, align_disparity, decompose, frames, load_frame, resample

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLUMNS = ["tilt_total_deg", "tilt_flattening_deg", "tilt_slant_deg", "tilt_bumps_deg",
           "flattening_k", "slant_per_unit", "bump_rms", "bump_size_px", "disparity_shift", "abs_rel"]


def figure(path, rgb, depth, aligned, resid, ok, row):
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
    lo, hi = np.nanpercentile(depth[depth > 0], [2, 98])
    ax[0].imshow(rgb); ax[0].set_title(f"{row['frame']}")
    ax[1].imshow(np.where(depth > 0, depth, np.nan), vmin=lo, vmax=hi, cmap="turbo"); ax[1].set_title("CT depth (mm)")
    ax[2].imshow(aligned, vmin=lo, vmax=hi, cmap="turbo"); ax[2].set_title("Depth Anything, aligned")
    r = np.where(ok, resid, np.nan); m = np.nanpercentile(np.abs(r), 98)
    ax[3].imshow(r, vmin=-m, vmax=m, cmap="coolwarm")
    ax[3].set_title(f"bumps (tilt {row['tilt_bumps_deg']:.1f} deg); total {row['tilt_total_deg']:.1f} deg")
    for x in ax:
        x.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    rows = []
    for exp, fid in frames(a.data):
        rgb, depth, valid, K = load_frame(a.data, exp, fid)
        pred = np.load(Path(a.pred) / f"{exp}_{fid}.npy").astype(np.float64)
        aligned, (s, t) = align_disparity(pred, depth, valid)
        # true disparity ~ s * pred + t, so the prediction is the true disparity shifted by -t (then scaled):
        # as a fraction of the median true disparity, this is the shift the generator adds before dropping the scale
        disparity_shift = float(-t / np.median(1.0 / depth[valid]))
        valid &= np.isfinite(aligned)
        abs_rel = float(np.mean(np.abs(aligned[valid] - depth[valid]) / depth[valid]))
        zt, okt, Ks = resample(depth, valid, K)
        zp, okp, _ = resample(np.nan_to_num(aligned), valid, K)
        ok = okt & okp
        stats, resid = decompose(zt, zp, ok, Ks)
        row = dict(frame=f"{exp}/{fid}", fx=round(K[0, 0], 1), valid_frac=round(float(valid.mean()), 3), **stats,
                   disparity_shift=disparity_shift, abs_rel=abs_rel)
        rows.append(row)
        figure(out / "figures" / f"{exp}_{fid}.png", rgb, np.where(valid, depth, 0), np.where(valid, aligned, np.nan),
               resid, ok, row)
        print(f"{row['frame']}: tilt total {row['tilt_total_deg']:5.1f} = flattening {row['tilt_flattening_deg']:4.1f} "
              f"/ slant {row['tilt_slant_deg']:4.1f} / bumps {row['tilt_bumps_deg']:4.1f} deg | k {row['flattening_k']:.2f} "
              f"| bump size {row['bump_size_px']:.0f} px | disparity shift {disparity_shift:+.2f} | AbsRel {abs_rel:.3f}", flush=True)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    summary = {c: dict(median=float(np.nanmedian([r[c] for r in rows])), p10=float(np.nanpercentile([r[c] for r in rows], 10)),
                       p90=float(np.nanpercentile([r[c] for r in rows], 90))) for c in COLUMNS}
    summary["frames"] = len(rows)
    summary["fx_target_px"] = FX_TARGET
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print("\nmedian (10th-90th percentile) over", len(rows), "frames")
    for c in COLUMNS:
        s = summary[c]
        print(f"  {c:20s} {s['median']:8.3f}  ({s['p10']:.3f} - {s['p90']:.3f})")


if __name__ == "__main__":
    main()
