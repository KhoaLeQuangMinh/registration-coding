"""
Pilot examples: what a "good-Dice" wrong prediction looks like.

For one random test view, draws 40 wrong predictions per family (same sampling as
pilot_dice_vs_tumour.py) and shows the typical one among those with Dice >= 0.95: the
prediction with the median tumour error (not the worst, to avoid cherry-picking).
  left   camera image: liver label, true vs predicted outline, true vs predicted tumours
  right  top-down view (camera at the bottom): shows the depth error the image cannot show

Usage: python benchmark/pilot_examples.py [--patient 01] [--seed 1]
"""
import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pilot_dice_vs_tumour import (FAMILIES, LIVER, SCALE_RANGE, SCALE_RECIPE_ERROR, SHAFT, TIP,
                                  H, W, apply, dice, evaluate, load_patient, load_view, project,
                                  render_mask, rigid_error, run_icp, sample_tumours, surface_dists,
                                  true_w2c)

INK, MUTED, TRUE_C, PRED_C = "#2b2b2b", "#8a8a85", "#2b2b2b", "#2a78d6"
SHOWN = [f for f in FAMILIES if f != "icp"]  # clean-cloud ICP is always ~0.3 mm, nothing to show


def predict(fam, rng, T_true, pivots, P, vis_pts):
    """One wrong prediction; returns the pose and the cloud the method saw."""
    if fam in pivots:
        return rigid_error(rng, pivots[fam]) @ T_true, vis_pts
    if fam == "scale":
        s = np.exp(rng.uniform(*np.log(SCALE_RANGE)))
        return np.diag([s, s, s, 1.0]) @ T_true, vis_pts
    s = np.exp(rng.uniform(*np.log(SCALE_RECIPE_ERROR)))
    T_init = np.diag([s, s, s, 1.0]) @ rigid_error(rng, pivots["organ"]) @ T_true
    return run_icp(T_init, P, vis_pts * s), vis_pts * s


def outline(ax, mask, color, ls="-"):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        c = np.vstack([c[:, 0], c[:1, 0]])
        ax.plot(c[:, 0], c[:, 1], color=color, lw=1.6, ls=ls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default="01")
    ap.add_argument("--perts", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="benchmark/pilot_out/pilot_examples.png")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    P = load_patient(args.patient)
    tumours, _ = sample_tumours(P, rng, 20)

    # First random test view that passes the same true-pose check as the pilot
    for idx in rng.permutation(P["views"]):
        label, vis_pts = load_view(P, int(idx))
        T_true = true_w2c(P["poses"][f"{idx:05d}"][0])
        ignore = (label == TIP) | (label == SHAFT)
        if (dice(render_mask(T_true, P), label == LIVER, ignore) >= 0.98
                and (surface_dists(T_true, P, vis_pts) > 3.0).mean() <= 0.02):
            break
    true_mask = render_mask(T_true, P)
    in_view = np.zeros(len(tumours), bool)
    pivots = {"camera": np.zeros(3), "visible": vis_pts.mean(axis=0),
              "organ": apply(T_true, P["centroid"][None])[0]}

    verts = np.asarray(P["mesh"].vertices)[::15]
    fig, axes = plt.subplots(len(SHOWN), 2, figsize=(13, 3.4 * len(SHOWN)),
                             gridspec_kw=dict(width_ratios=[1.6, 1]))
    for row, fam in enumerate(SHOWN):
        cands = []
        for _ in range(args.perts):
            T_pred, seen = predict(fam, rng, T_true, pivots, P, vis_pts)
            d, _, err, _, px = evaluate(T_pred, T_true, P, (label, seen, in_view), tumours)
            cands.append((d, err, px, T_pred))
        good = [c for c in cands if c[0] >= 0.95] or sorted(cands, key=lambda c: -c[0])[:1]
        good.sort(key=lambda c: c[1].mean())
        d, err, px, T_pred = good[len(good) // 2]
        n_good = sum(c[0] >= 0.95 for c in cands)

        # --- camera image ---
        ax = axes[row, 0]
        canvas = np.full((H, W, 3), 1.0)
        canvas[label == LIVER] = (0.86, 0.86, 0.84)
        canvas[ignore] = (0.70, 0.70, 0.68)
        ax.imshow(canvas)
        outline(ax, true_mask, TRUE_C)
        outline(ax, render_mask(T_pred, P), PRED_C, "--")
        uv_t, uv_p = project(apply(T_true, tumours), P["K"]), project(apply(T_pred, tumours), P["K"])
        for a, b in zip(uv_t, uv_p):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=MUTED, lw=0.8)
        ax.scatter(uv_t[:, 0], uv_t[:, 1], s=22, color=TRUE_C, zorder=3)
        ax.scatter(uv_p[:, 0], uv_p[:, 1], s=22, facecolors="none", edgecolors=PRED_C, lw=1.4, zorder=3)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        depth = np.abs(apply(T_pred, tumours)[:, 2] - apply(T_true, tumours)[:, 2])
        ax.set_title(f"{fam}:  Dice {d:.3f}   |   tumour error median {np.median(err):.1f} mm "
                     f"(depth {np.median(depth):.1f} mm, on screen {np.nanmedian(px):.0f} px)",
                     color=INK, fontsize=10, loc="left")
        ax.text(6, H - 10, f"typical of {n_good}/{args.perts} predictions with Dice >= 0.95",
                color=MUTED, fontsize=8)

        # --- top-down view: x across, depth z up, camera at the origin ---
        ax = axes[row, 1]
        vt, vp = apply(T_true, verts), apply(T_pred, verts)
        ax.scatter(vt[:, 0], vt[:, 2], s=1, color=TRUE_C, alpha=0.15, linewidths=0)
        ax.scatter(vp[:, 0], vp[:, 2], s=1, color=PRED_C, alpha=0.15, linewidths=0)
        ct, cp = apply(T_true, tumours), apply(T_pred, tumours)
        for a, b in zip(ct, cp):
            ax.plot([a[0], b[0]], [a[2], b[2]], color=MUTED, lw=0.8)
        ax.scatter(ct[:, 0], ct[:, 2], s=22, color=TRUE_C, zorder=3)
        ax.scatter(cp[:, 0], cp[:, 2], s=22, facecolors="none", edgecolors=PRED_C, lw=1.4, zorder=3)
        # Viewing rays through the image edges; the camera sits below the plot at depth 0
        both = np.vstack([vt, vp])
        lo, hi = both.min(axis=0) - 15, both.max(axis=0) + 15
        for u in (0, W):
            x = (u - P["K"][0, 2]) / P["K"][0, 0] * hi[2]
            ax.plot([0, x], [0, hi[2]], color=MUTED, lw=0.8, ls=":")
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[2], hi[2])
        ax.text(0.02, 0.03, "camera below, rays dotted", transform=ax.transAxes, color=MUTED, fontsize=8)
        ax.set_aspect("equal")
        ax.set_xlabel("across the view (mm)", color=INK, fontsize=9)
        ax.set_ylabel("depth from camera (mm)", color=INK, fontsize=9)
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    handles = [plt.Line2D([], [], color=TRUE_C, lw=1.6, label="true liver outline / position"),
               plt.Line2D([], [], color=PRED_C, lw=1.6, ls="--", label="predicted liver outline / position"),
               plt.Line2D([], [], marker="o", color=TRUE_C, ls="", label="true tumour centre"),
               plt.Line2D([], [], marker="o", mfc="none", mec=PRED_C, ls="", label="predicted tumour centre")]
    fig.suptitle(f"Patient {args.patient}, synthetic view {idx:05d}: typical high-Dice wrong predictions",
                 color=INK, y=0.995)
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.982))
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"view {idx:05d}; figure saved to {args.out}")


if __name__ == "__main__":
    main()
