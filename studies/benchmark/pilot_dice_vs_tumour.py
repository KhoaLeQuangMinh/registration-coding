"""
Pilot: does a high Dice guarantee that the tumour is in the right place?

On synthetic P2I-LReg views (true camera pose known) we plant spherical tumours inside
the CT liver, produce wrong pose predictions, and score every prediction two ways:
  - Dice between the projected CT outline and the liver label (what papers report)
  - tumour error: mm between where the prediction puts each tumour and where it really is
A one-sided surface-fit score (visible cloud -> CT surface, mm) is reported as well,
since that is the other score that can be computed without ground truth.

Wrong predictions come from five families, all sampled the same way:
  camera   rotation about the camera centre + translation  (error in camera orientation)
  visible  rotation about the centroid of the visible surface + translation
  organ    rotation about the centroid of the whole liver + translation
  scale    zoom about the camera centre only (Dice is unchanged by construction; reported, not used)
  icp      point-to-plane ICP of the visible cloud to the CT, started from an 'organ' error.
           These are outputs of a real algorithm, so they are the primary test.
Rotation angle ~ U(0, 15 deg) about a uniformly random axis; translation ~ U(0, 20 mm)
in a uniformly random direction; scale ~ log-U(0.8, 1.25). Views are drawn at random from
each patient's test split, and tumours uniformly from the whole liver volume (not from the
visible part). All ranges and the decision rule below were fixed before the first run.

Pre-registered decision rule (Dice >= 0.90, mean tumour error over all tumours):
  SUPPORTED      >= 10 % of predictions have tumour error > 10 mm
  NOT SUPPORTED  <  2 % have tumour error > 10 mm AND Spearman |rho(Dice, error)| >= 0.8
  INCONCLUSIVE   otherwise

Amendment after a 6-view smoke test, before the full run: on the clean synthetic cloud, ICP
converged to < 0.5 mm from every start, so the 'icp' family cannot produce realistic failures.
Added 'icp_scaled': the same ICP run on the cloud scaled about the camera centre by
s ~ log-U(1.01, 1.32), the 10-90 % range of the PCA scale-recipe error we measured on 238
synthetic views. The decision rule is applied to 'icp_scaled'; 'icp' is still reported.
Simplifications: rigid only (synthetic livers are not deformed), spheres with guessed radius.

Usage: python benchmark/pilot_dice_vs_tumour.py [--patients 01 06 10 11] [--views 25] ...
"""
import argparse
import csv
import os

import cv2
import numpy as np
import open3d as o3d
import yaml

DATA_DIR = "dataset/liver-registration"
H, W = 540, 960                      # synthetic render size
LIVER, TIP, SHAFT = 0, 153, 179      # synthetic label values (102 = background)

MAX_ROT_DEG = 15.0
MAX_TRANS_MM = 20.0
SCALE_RANGE = (0.8, 1.25)
SCALE_RECIPE_ERROR = (1.01, 1.32)    # 10-90 % range of the PCA scale-recipe error on synthetic views
TUMOUR_RADIUS_MM = (5.0, 25.0)       # 1-5 cm diameter; to be replaced by LiTS statistics
DICE_GOOD = 0.90
MARGIN_MM = 10.0
FAMILIES = ["camera", "visible", "organ", "scale", "icp", "icp_scaled"]
PRIMARY = "icp_scaled"


# ==============================================================================
# Loading
# ==============================================================================
def load_patient(pid: str):
    """CT mesh in mm, ray-casting scene, ICP target, K, poses, test views."""
    root = os.path.join(DATA_DIR, pid)
    mesh = o3d.io.read_triangle_mesh(os.path.join(root, "model", "reconstructed_mesh_world_m.obj"))
    mesh.scale(1000.0, center=[0, 0, 0])
    mesh.compute_vertex_normals()

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    target = mesh.sample_points_uniformly(200000, use_triangle_normal=True).voxel_down_sample(2.0)

    with open(os.path.join(root, "syn", "CAM_K.yml")) as f:
        K = np.array(yaml.safe_load(f)["00000"][0]["cam_K"]).reshape(3, 3)
    v, u = np.mgrid[0:H, 0:W]
    pix_dirs = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0], (v + 0.5 - K[1, 2]) / K[1, 1],
                         np.ones((H, W))], axis=-1).reshape(-1, 3)  # one ray per pixel centre
    with open(os.path.join(root, "syn", "camPose.yml")) as f:
        poses = yaml.safe_load(f)
    with open(os.path.join(root, "test_syn.txt")) as f:
        views = [int(line.strip().replace("ptsLiverCam", "")) for line in f if line.strip()]

    return dict(root=root, mesh=mesh, centroid=np.asarray(mesh.vertices).mean(axis=0), scene=scene,
                target=target, K=K, pix_dirs=pix_dirs, poses=poses, views=views)


def true_w2c(pose) -> np.ndarray:
    """4x4 world(mm) -> OpenCV camera(mm), from Blender camera-to-world (-Z forward)."""
    R_cv = np.array(pose["cam_R_c2w"]).reshape(3, 3) @ np.diag([1, -1, -1])
    t = np.array(pose["cam_t_c2w"]).reshape(3)
    T = np.eye(4)
    T[:3, :3] = R_cv.T
    T[:3, 3] = -R_cv.T @ t
    return T


def load_view(P, idx: int):
    label = cv2.imread(os.path.join(P["root"], "syn", "labels", f"lbl{idx:05d}.png"))[..., 0]
    pcd = o3d.io.read_point_cloud(os.path.join(P["root"], "syn", "liverPcds", f"ptsLiverCam{idx:05d}.ply"))
    pcd = pcd.voxel_down_sample(3.0)
    pts = np.asarray(pcd.points) * np.array([1.0, -1.0, -1.0])  # Blender camera -> OpenCV camera
    return label, pts


# ==============================================================================
# Geometry helpers
# ==============================================================================
def apply(T: np.ndarray, X: np.ndarray) -> np.ndarray:
    return X @ T[:3, :3].T + T[:3, 3]


def project(X: np.ndarray, K: np.ndarray) -> np.ndarray:
    z = np.where(X[:, 2] > 1e-6, X[:, 2], np.nan)
    return np.stack([K[0, 0] * X[:, 0] / z + K[0, 2], K[1, 1] * X[:, 1] / z + K[1, 2]], axis=1)


def render_mask(T_pred: np.ndarray, P) -> np.ndarray:
    """Liver outline seen with pose T_pred: cast one ray per pixel into the CT mesh."""
    inv = np.linalg.inv(T_pred)
    dirs = P["pix_dirs"] @ inv[:3, :3].T
    rays = np.concatenate([np.broadcast_to(inv[:3, 3], dirs.shape), dirs], axis=1).astype(np.float32)
    t_hit = P["scene"].cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    return np.isfinite(t_hit).reshape(H, W)


def dice(pred: np.ndarray, liver: np.ndarray, ignore: np.ndarray) -> float:
    """Dice over pixels not covered by a tool (the liver behind a tool is unknown)."""
    p, g = pred & ~ignore, liver & ~ignore
    return 2.0 * (p & g).sum() / max(p.sum() + g.sum(), 1)


def random_rotation(rng, max_deg: float) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    a = np.deg2rad(rng.uniform(0, max_deg))
    Kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * Kx + (1 - np.cos(a)) * Kx @ Kx


def rigid_error(rng, pivot: np.ndarray) -> np.ndarray:
    """4x4 camera-frame error: rotate about the pivot, then translate."""
    R = random_rotation(rng, MAX_ROT_DEG)
    d = rng.normal(size=3)
    t = d / np.linalg.norm(d) * rng.uniform(0, MAX_TRANS_MM)
    E = np.eye(4)
    E[:3, :3] = R
    E[:3, 3] = pivot - R @ pivot + t
    return E


def rotation_deg(R: np.ndarray) -> float:
    return np.rad2deg(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def sample_tumours(P, rng, n: int):
    """Sphere centres uniform in the liver volume, each sphere fully inside the liver."""
    lo, hi = P["mesh"].get_min_bound(), P["mesh"].get_max_bound()
    centres, radii = [], []
    while len(centres) < n:
        c = rng.uniform(lo, hi, size=(4096, 3)).astype(np.float32)
        r = rng.uniform(*TUMOUR_RADIUS_MM, size=4096)
        inside = P["scene"].compute_occupancy(o3d.core.Tensor(c), nsamples=9).numpy() > 0.5
        dist = P["scene"].compute_distance(o3d.core.Tensor(c)).numpy()
        ok = inside & (dist >= r)
        centres += list(c[ok].astype(np.float64))
        radii += list(r[ok])
    return np.array(centres[:n]), np.array(radii[:n])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


# ==============================================================================
# Scoring one predicted pose
# ==============================================================================
def surface_dists(T_pred, P, vis_pts):
    """Distance (mm) of each visible point, taken back to the CT frame with T_pred, to the CT surface."""
    back = apply(np.linalg.inv(T_pred), vis_pts).astype(np.float32)
    return P["scene"].compute_distance(o3d.core.Tensor(back)).numpy()


def evaluate(T_pred, T_true, P, view, tumours):
    label, vis_pts, in_view = view
    liver, ignore = label == LIVER, (label == TIP) | (label == SHAFT)

    d = dice(render_mask(T_pred, P), liver, ignore)

    fit = float(surface_dists(T_pred, P, vis_pts).mean())

    c_true, c_pred = apply(T_true, tumours), apply(T_pred, tumours)
    err = np.linalg.norm(c_pred - c_true, axis=1)
    lateral = np.linalg.norm((c_pred - c_true)[:, :2], axis=1)  # across the view; depth = |dz|
    px = np.linalg.norm(project(c_pred, P["K"]) - project(c_true, P["K"]), axis=1)
    return d, fit, err, lateral, px


def run_icp(T_init, P, vis_pts):
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vis_pts))
    res = o3d.pipelines.registration.registration_icp(
        src, P["target"], 20.0, np.linalg.inv(T_init),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
    return np.linalg.inv(res.transformation)


# ==============================================================================
# Main
# ==============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", nargs="+", default=["01", "06", "10", "11"])
    ap.add_argument("--views", type=int, default=25, help="random test views per patient that pass the check")
    ap.add_argument("--perts", type=int, default=40, help="wrong predictions per family per view")
    ap.add_argument("--tumours", type=int, default=20, help="tumours per patient")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="benchmark/pilot_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rows, per_tumour, excluded = [], [], []
    for pid in args.patients:
        P = load_patient(pid)
        tumours, radii = sample_tumours(P, rng, args.tumours)
        n_used = 0

        for idx in rng.permutation(P["views"]):
            if n_used == args.views:
                break
            label, vis_pts = load_view(P, int(idx))
            T_true = true_w2c(P["poses"][f"{idx:05d}"][0])

            # Sanity check: at the true pose the outline and the cloud must match the CT.
            # Views 2000-2499 of patients 10 and 11 fail it: an extra object hides part of the
            # liver, is labelled background, and its surface is included in the liver cloud.
            d0 = dice(render_mask(T_true, P), label == LIVER, (label == TIP) | (label == SHAFT))
            far0 = (surface_dists(T_true, P, vis_pts) > 3.0).mean()
            if d0 < 0.98 or far0 > 0.02:
                excluded.append((pid, int(idx), d0, far0))
                continue
            n_used += 1

            # A tumour is "in view" if its true centre projects onto the true liver outline
            gt_mask = render_mask(T_true, P)
            uv = project(apply(T_true, tumours), P["K"])
            ok = np.isfinite(uv).all(axis=1)
            ui, vi_ = np.full(len(uv), -1), np.full(len(uv), -1)
            ui[ok], vi_[ok] = np.round(uv[ok, 0]).astype(int), np.round(uv[ok, 1]).astype(int)
            inb = ok & (ui >= 0) & (ui < W) & (vi_ >= 0) & (vi_ < H)
            in_view = np.zeros(len(uv), bool)
            in_view[inb] = gt_mask[vi_[inb], ui[inb]]
            view = (label, vis_pts, in_view)

            print(f"[{pid}] view {idx:05d} ({n_used}/{args.views})  true pose: Dice {d0:.3f}, "
                  f"{100 * far0:.1f}% far points, {in_view.sum()}/{len(tumours)} tumours in view")

            pivots = {"camera": np.zeros(3), "visible": vis_pts.mean(axis=0),
                      "organ": apply(T_true, P["centroid"][None])[0]}
            for fam in FAMILIES:
                for _ in range(args.perts):
                    if fam in pivots:
                        T_pred = rigid_error(rng, pivots[fam]) @ T_true
                    elif fam == "scale":
                        s = np.exp(rng.uniform(*np.log(SCALE_RANGE)))
                        T_pred = np.diag([s, s, s, 1.0]) @ T_true
                    elif fam == "icp":
                        T_init = rigid_error(rng, pivots["organ"]) @ T_true
                        T_pred = run_icp(T_init, P, vis_pts)
                    else:
                        # The method sees a cloud with the wrong scale and aligns it rigidly
                        s = np.exp(rng.uniform(*np.log(SCALE_RECIPE_ERROR)))
                        T_init = np.diag([s, s, s, 1.0]) @ rigid_error(rng, pivots["organ"]) @ T_true
                        T_pred = run_icp(T_init, P, vis_pts * s)

                    seen = (label, vis_pts * s, in_view) if fam == "icp_scaled" else view
                    d, fit, err, lateral, px = evaluate(T_pred, T_true, P, seen, tumours)
                    E = T_pred @ np.linalg.inv(T_true)
                    scale = float(np.cbrt(np.linalg.det(E[:3, :3])))
                    rows.append(dict(
                        patient=pid, view=int(idx), family=fam, dice=d, surface_fit_mm=fit,
                        rot_err_deg=rotation_deg(E[:3, :3] / scale), scale=scale,
                        liver_centroid_err_mm=float(np.linalg.norm(apply(E, pivots["organ"][None])[0] - pivots["organ"])),
                        tumour_err_mean_mm=float(err.mean()), tumour_err_max_mm=float(err.max()),
                        tumour_lateral_mean_mm=float(lateral.mean()),
                        tumour_err_inview_mm=float(err[in_view].mean()) if in_view.any() else float("nan"),
                        tumour_px_inview=float(np.nanmean(px[in_view])) if in_view.any() else float("nan"),
                        n_inview=int(in_view.sum())))
                    per_tumour.append(np.stack([err, radii, in_view], axis=1))

    with open(os.path.join(args.out, "pilot_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    np.save(os.path.join(args.out, "pilot_per_tumour.npy"), np.stack(per_tumour))
    with open(os.path.join(args.out, "excluded_views.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient", "view", "true_pose_dice", "far_point_fraction"])
        w.writerows(excluded)

    print(f"\n{len(excluded)} views drawn but excluded by the true-pose check "
          f"(see excluded_views.csv): " + ", ".join(
              f"{pid}: {sum(e[0] == pid for e in excluded)}" for pid in args.patients))
    summarize(rows, args.out)


def summarize(rows, out_dir):
    lines = []
    col = lambda key, sel: np.array([r[key] for r in sel], float)
    bins = [(0.0, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.01)]
    for fam in FAMILIES:
        sel = [r for r in rows if r["family"] == fam]
        dsc, err = col("dice", sel), col("tumour_err_mean_mm", sel)
        fit, inv = col("surface_fit_mm", sel), col("tumour_err_inview_mm", sel)
        lat = col("tumour_lateral_mean_mm", sel)
        lines.append(f"\n== {fam}  (n = {len(sel)}) ==")
        lines.append(f"Spearman rho  Dice vs tumour error: {spearman(dsc, err):+.2f}   "
                     f"surface fit vs tumour error: {spearman(fit, err):+.2f}")
        lines.append(f"{'Dice bin':<12}{'n':>6}{'median mm':>11}{'P90 mm':>9}{'> 10 mm':>9}"
                     f"{'in-view >10':>13}{'lateral >10':>13}")
        for lo, hi in bins:
            m = (dsc >= lo) & (dsc < hi)
            if not m.any():
                continue
            iv = inv[m][np.isfinite(inv[m])]
            lines.append(f"[{lo:.2f},{min(hi, 1):.2f}){m.sum():>6}{np.median(err[m]):>11.1f}"
                         f"{np.percentile(err[m], 90):>9.1f}{100 * (err[m] > MARGIN_MM).mean():>8.1f}%"
                         f"{(100 * (iv > MARGIN_MM).mean() if len(iv) else float('nan')):>12.1f}%"
                         f"{100 * (lat[m] > MARGIN_MM).mean():>12.1f}%")

    sel = [r for r in rows if r["family"] == PRIMARY]
    dsc, err = col("dice", sel), col("tumour_err_mean_mm", sel)
    good = dsc >= DICE_GOOD
    frac = (err[good] > MARGIN_MM).mean() if good.any() else float("nan")
    rho = spearman(dsc, err)
    if not good.any():
        verdict = f"INCONCLUSIVE (no {PRIMARY} prediction reached Dice >= {DICE_GOOD})"
    elif frac >= 0.10:
        verdict = "SUPPORTED"
    elif frac < 0.02 and abs(rho) >= 0.8:
        verdict = "NOT SUPPORTED"
    else:
        verdict = "INCONCLUSIVE"
    lines.append(f"\nPre-registered test ({PRIMARY}, Dice >= {DICE_GOOD}): n = {good.sum()}, "
                 f"{100 * frac:.1f}% with tumour error > {MARGIN_MM:.0f} mm, rho = {rho:+.2f} -> {verdict}")

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(out_dir, "pilot_summary.txt"), "w") as f:
        f.write(text + "\n")
    plot(rows, out_dir)


def plot(rows, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping the figure (pip install matplotlib)")
        return

    ink, muted, mark = "#2b2b2b", "#8a8a85", "#2a78d6"
    fams = [f for f in FAMILIES if f != "scale"]
    fig, axes = plt.subplots(2, len(fams), figsize=(4 * len(fams), 7), sharey=True)
    for j, fam in enumerate(fams):
        sel = [r for r in rows if r["family"] == fam]
        err = np.array([r["tumour_err_mean_mm"] for r in sel])
        for i, (key, xlabel) in enumerate([("dice", "Dice"), ("surface_fit_mm", "Surface fit (mm)")]):
            ax = axes[i, j]
            ax.scatter([r[key] for r in sel], err, s=6, alpha=0.35, color=mark, linewidths=0)
            ax.axhline(MARGIN_MM, color=muted, lw=1, ls="--")
            if key == "dice":
                ax.axvline(DICE_GOOD, color=muted, lw=1, ls="--")
            ax.set_xlabel(xlabel, color=ink)
            if j == 0:
                ax.set_ylabel("Tumour error (mm)", color=ink)
            if i == 0:
                ax.set_title(fam, color=ink)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.tick_params(colors=muted)
    fig.suptitle("Dice and surface fit against tumour error (dashed: 10 mm margin, Dice 0.90)", color=ink)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pilot_scatter.png"), dpi=130)
    print(f"figure saved to {os.path.join(out_dir, 'pilot_scatter.png')}")


if __name__ == "__main__":
    main()
