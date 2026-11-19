"""
See what outline_reg.py does on one synthetic training view (true pose known), and why it fails when it does.

Saves registration/out/outline_reg_<patient>_<view>_<condition>.png:
  1 input        what the method gets: liver mask (tan), tools (grey), ridge (green, --ridge only)
  2 answer       the mask with the CT outline at the true pose (green) and at the found pose (red)
  3 refinement   the winning start's CT outline, from the start (light) to the answer (dark red)
  4, 5           the CT seen from the true camera and from the found camera (mask outline in blue)
  6 search       every start after the short refinement: outline fit (px) vs tumour error (mm);
                 black rings = the candidates refined fully, red = the answer, green = the true pose refined
  7 path         fit on the way from the true pose (refined) to the answer, as test_ridge_value.py:
                 a hump = a separate wrong pose, flat = a valley of poses that fit equally
  8 numbers

Failing views of the Kaggle run on the local patients (outline condition unless stated):
  01 789 separate (all conditions) | 01 1590 search | 01 869 search, fixed by --more
  10 1092 connected (all)          | 10 1043 search, fixed by --ridge | 11 865 fails only with --more

Usage: python registration/show_outline_reg.py [--patient 01] [--view 789] [--ridge | --more]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from outline_reg import fit_error, gn_step, make_view, pairs, project, register, render, starts, visible
from pilot_dice_vs_tumour import LIVER, SHAFT, TIP, load_view, rotation_deg, true_w2c
from sim3_icp import rotation
from test_ridge_value import ALT_PX, BARRIER_PX, CONDITIONS, HERE, PATH_STEPS, SUCCESS_MM, patient


def track(scene, view, R, t, rid, iters, tol=1e-3):
    """outline_reg.register, keeping every pose on the way."""
    poses = [(R, t)]
    for _ in range(iters):
        kept = pairs(scene, view, R, t, rid)
        if len(kept) < 2:
            break
        R_new, t_new = gn_step(view["K"], R, t, kept)
        moved = np.linalg.norm(R_new - R) * 100 + np.linalg.norm(t_new - t)
        R, t = R_new, t_new
        poses.append((R, t))
        if moved < tol:
            break
    return poses


def path_fits(scene, view, A, B, rid):
    """Fit at the points of test_ridge_value.path_barrier's path from A to B, ends included."""
    (Ra, ta), (Rb, tb) = A, B
    Ca, Cb = -Ra.T @ ta, -Rb.T @ tb
    rel = Ra @ Rb.T
    w = np.array([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]])
    axis = w / (np.linalg.norm(w) + 1e-12)
    ks = np.linspace(0, 1, PATH_STEPS + 2)
    fits = [fit_error(scene, view, *A, rid)]
    for k in ks[1:-1]:
        R = rotation(axis, -k * rotation_deg(rel)) @ Ra
        fits.append(fit_error(scene, view, *register(scene, view, R, -R @ ((1 - k) * Ca + k * Cb), rid,
                                                     iters=20, rotate=False), rid))
    return ks, np.array(fits + [fit_error(scene, view, *B, rid)])


def shaded(scene, K, size, R, t):
    """The CT seen from pose (R, t), full resolution, shaded by the angle to the ray (nan off the liver)."""
    W, H = size
    v, u = np.mgrid[0:H, 0:W]
    d = np.dstack([(u + 0.5 - K[0, 2]) / K[0, 0], (v + 0.5 - K[1, 2]) / K[1, 1], np.ones((H, W))]).reshape(-1, 3) @ R
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    hit = scene.cast_rays(o3d.core.Tensor(np.hstack([np.broadcast_to(-R.T @ t, d.shape), d]).astype(np.float32)))
    s = np.abs(np.sum(hit["primitive_normals"].numpy() * d, axis=1))
    s[~np.isfinite(hit["t_hit"].numpy())] = np.nan
    return s.reshape(H, W)


def outline(ax, scene, view, R, t, **kw):
    hit = np.isfinite(render(scene, view, R, t)).astype(float)
    ax.contour(view["pix"][..., 0], view["pix"][..., 1], hit, levels=[0.5], **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="dataset/liver-registration")
    ap.add_argument("--patient", default="01")
    ap.add_argument("--view", type=int, default=789)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--ridge", action="store_true")
    g.add_argument("--more", action="store_true")
    args = ap.parse_args()
    cond = "ridge" if args.ridge else "outline_more" if args.more else "outline"
    use_ridge, n_dirs, n_rolls, keep = CONDITIONS[cond]

    P = patient(args.data_root, args.patient)
    scene, K = P["scene"], P["K"]
    label, _ = load_view(P, args.view)
    T = true_w2c(P["poses"][f"{args.view:05d}"][0])
    R0, t0 = T[:3, :3], T[:3, 3]
    view = make_view(K, label == LIVER, (label == TIP) | (label == SHAFT))
    seen = P["ridge3d"][visible(scene, view, R0, t0, P["ridge3d"])]
    view["ridge2d"] = project(K, seen @ R0.T + t0)
    rid = P["ridge3d"] if use_ridge else None
    probes = P["probes"]
    err = lambda R, t: np.linalg.norm(probes @ R.T + t - (probes @ R0.T + t0), axis=1).mean()  # noqa: E731
    fit = lambda p: fit_error(scene, view, *p, rid)  # noqa: E731

    # the search, as outline_reg.register_from_scratch, keeping each start's path
    print(f"patient {args.patient} view {args.view} {cond}: searching {n_dirs} x {n_rolls} starts ...")
    paths = [track(scene, view, *s, rid, 15) for s in starts(scene, view, P["centroid"], n_dirs, n_rolls)]
    order = sorted(range(len(paths)), key=lambda i: fit(paths[i][-1]))[:keep]
    for i in order:
        paths[i] += track(scene, view, *paths[i][-1], rid, 100)[1:]
    win = min(order, key=lambda i: fit(paths[i][-1]))
    F = paths[win][-1]
    T_star = track(scene, view, R0, t0, rid, 100)[-1]
    fit_F, fit_star, err_F = fit(F), fit(T_star), err(*F)
    kind = ""
    if err_F >= SUCCESS_MM:
        kind = "search" if fit_F > fit_star + ALT_PX else "alternative"
    ks, along = path_fits(scene, view, T_star, F, rid)
    if kind == "alternative":
        kind = "separate" if along[1:-1].max() - max(fit_star, fit_F) > BARRIER_PX else "connected"

    # figure
    fig, ax = plt.subplots(2, 4, figsize=(22, 10))
    ax = ax.ravel()
    img = np.ones(label.shape + (3,))
    img[label == LIVER] = (0.93, 0.80, 0.65)
    img[(label == TIP) | (label == SHAFT)] = (0.55, 0.55, 0.55)
    W, H = view["size"]
    for a in ax[:3]:
        a.imshow(img)
    if use_ridge and len(view["ridge2d"]):
        ax[0].plot(*view["ridge2d"].T, ".", color="green", ms=3)
    ax[0].set_title("1 input: liver mask" + (" + ridge (green)" if use_ridge else ""))

    outline(ax[1], scene, view, R0, t0, colors="green", linewidths=2.5)
    outline(ax[1], scene, view, *F, colors="red", linewidths=2)
    ax[1].set_title("2 answer: CT outline at true pose (green) / found (red)")

    steps = paths[win]
    for j, k in enumerate(np.unique(np.linspace(0, len(steps) - 1, 6).astype(int))):
        outline(ax[2], scene, view, *steps[k], colors=[plt.cm.Reds(0.25 + 0.75 * j / 5)], linewidths=1.5)
    ax[2].set_title(f"3 refinement of the winning start ({len(steps) - 1} steps, light -> dark)")

    for a, pose, name in [(ax[3], (R0, t0), "4 CT from the true camera"), (ax[4], F, "5 CT from the found camera")]:
        a.imshow(shaded(scene, K, (W, H), *pose), cmap="gray", vmin=0, vmax=1)
        a.contour(label == LIVER, levels=[0.5], colors="dodgerblue", linewidths=1.5)
        if use_ridge:
            s = P["ridge3d"][visible(scene, view, *pose, P["ridge3d"])]
            a.plot(*project(K, s @ pose[0].T + pose[1]).T, ".", color="green", ms=2)
        a.set_title(name + " (blue = mask outline)")
    for a in ax[:5]:
        a.set_xlim(0, W); a.set_ylim(H, 0); a.set_xticks([]); a.set_yticks([])

    screened = [p[min(15, len(p) - 1)] for p in paths]
    a = ax[5]
    a.scatter([fit(p) for p in screened], [err(*p) for p in screened], s=8, c="grey", alpha=0.5, label="each start after 15 steps")
    a.scatter([fit(paths[i][-1]) for i in order], [err(*paths[i][-1]) for i in order], s=60,
              facecolors="none", edgecolors="black", label=f"the {keep} refined fully")
    a.scatter([fit_F], [err_F], s=150, marker="*", c="red", label="answer (best fit)", zorder=3)
    a.scatter([fit_star], [err(*T_star)], s=150, marker="*", c="green", label="true pose refined", zorder=3)
    a.axhline(SUCCESS_MM, ls="--", c="k", lw=0.8)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("outline fit (px, lower = fits the mask better)"); a.set_ylabel("tumour error (mm)")
    a.legend(fontsize=8); a.set_title("6 search: does a wrong pose fit as well as the truth?")

    a = ax[6]
    a.plot(ks, along, "o-", c="k")
    a.plot([0], [along[0]], "*", c="green", ms=15); a.plot([1], [along[-1]], "*", c="red", ms=15)
    a.set_xlabel("true pose refined (0) -> answer (1)"); a.set_ylabel("outline fit (px)")
    a.set_title("7 path: hump = separate wrong pose, flat = valley")

    ax[7].axis("off")
    ax[7].text(0, 0.95, "\n".join([
        f"patient {args.patient}, view {args.view}, condition {cond}",
        f"tumour error      {err_F:7.1f} mm   ({'success' if err_F < SUCCESS_MM else 'FAIL'}, < {SUCCESS_MM:.0f} mm)",
        f"rotation error    {rotation_deg(F[0] @ R0.T):7.1f} deg",
        f"fit of answer     {fit_F:7.2f} px",
        f"fit of truth*     {fit_star:7.2f} px   (* refined from the true pose)",
        f"truth* error      {err(*T_star):7.1f} mm",
        f"failure kind      {kind or '-'}",
        f"ridge points seen {len(seen)}",
        f"starts            {len(paths)}",
    ]), family="monospace", fontsize=12, va="top")

    out = os.path.join(HERE, "out", f"outline_reg_{args.patient}_{args.view}_{cond}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=70)
    print(f"error {err_F:.1f} mm, failure {kind or '-'}; saved {out}")


if __name__ == "__main__":
    main()
