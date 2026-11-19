"""
Goal landscape test: does a goal give its best score to the TRUE CT pose, including the true distance?

Phase 1 (code check): draw the CT at each synthetic view's true pose (camPose) with the view's own synthetic camera and
compare with the dataset's mask. A near-perfect match means the camera conversion and units are right.

Phase 2 (the test): draw the CT at the true pose with the patient's REAL laparoscope focal length (P2I-LReg real
CAM_K), which gives the "photo" (mask M, and the true depth x a random factor as the depth-model stand-in: perfect
shape, useless distance). Then move the CT away from the truth one way at a time and score every candidate goal:
    distance   CT moved along the line of sight, x0.5 .. x1.5 of its true distance
    sideways   camera x and y, -50 .. +50 mm
    rotation   about the CT centre, camera x / y / z axes, -45 .. +45 deg
Two conditions: "clean" (only the image border hides anything) and "scope" (a round scope border hides the corners).
Candidate goals (lower = better):
    dice_plain   1 - Dice(CT silhouette, mask)              real size + real camera
    dice_masked  same, ignoring CT pixels outside the visible scope circle
    A1, A2       mean distance (px, 1920-wide scale) of mask pixels to the CT / of visible CT pixels to the mask
    B            mean (1 - cos) between CT surface normals and the photo's normals (scale-free)
Phase 3 (false valleys): on a few views, cameras all around the liver (64 directions x 8 rolls x 5 distances).
Self-P2IR's goal is not computed: letting the CT's size change with distance (its scale step) makes every distance give
the same picture, so it is flat along distance by construction.

Usage (repo root, a Python with open3d):  python goal/goal_landscape.py [--views 20] [--scan-views 3]
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import open3d as o3d
import yaml

DATA = "dataset/liver-registration"
OUT = "goal/out"
PATIENTS = ["01", "06", "10", "11"]
F = np.diag([1.0, -1.0, -1.0])          # Blender camera (-Z forward, +Y up) <-> vision camera (+Z forward, +Y down)
RW, RH, DOWN = 480, 270, 4               # phase 2 render size: the 1920x1080 camera / 4
SCOPE_R = 0.36                           # scope circle radius, fraction of image width (like P2I-LReg patient 06)


def k_matrix(v):
    return np.array(v, float).reshape(3, 3)


class Patient:
    def __init__(self, p):
        mesh = o3d.io.read_triangle_mesh(f"{DATA}/{p}/model/reconstructed_mesh_world_m.obj")
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) * 1000.0)   # metres -> mm (poses are mm)
        self.centre = np.asarray(mesh.vertices).mean(0)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        self.poses = yaml.safe_load(open(f"{DATA}/{p}/syn/camPose.yml"))
        self.syn_k = yaml.safe_load(open(f"{DATA}/{p}/syn/CAM_K.yml"))
        real = k_matrix(list(yaml.safe_load(open(f"{DATA}/{p}/real/CAM_K.yml")).values())[0][0]["cam_K"])
        self.real_k = np.diag([1 / DOWN, 1 / DOWN, 1.0]) @ real                            # at the render size
        self.real_k_full = real

    def true_pose(self, key):
        e = self.poses[key][0]
        Rc2w = np.array(e["cam_R_c2w"]).reshape(3, 3)
        R = F @ Rc2w.T                                    # world -> vision camera
        return R, -R @ np.array(e["cam_t_c2w"])


def render(scene, R, t, K, W, H):
    """Silhouette S (bool) and camera-frame depth Z (mm, 0 = miss) of the CT at pose X_cam = R X + t."""
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1).reshape(-1, 3)
    dn = d / np.linalg.norm(d, axis=1, keepdims=True)
    origin = -R.T @ t
    rays = np.hstack([np.repeat(origin[None], len(d), 0), dn @ R]).astype(np.float32)   # world-frame directions
    th = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    hit = np.isfinite(th)
    Z = np.where(hit, th * dn[:, 2], 0.0).reshape(H, W)
    return hit.reshape(H, W), Z


def normals(Z, K):
    """Surface normals from a depth map (any scale: scaling Z leaves them unchanged); invalid near edges."""
    H, W = Z.shape
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    P = np.stack([(u - K[0, 2]) / K[0, 0] * Z, (v - K[1, 2]) / K[1, 1] * Z, Z], -1)
    dx = np.zeros_like(P); dy = np.zeros_like(P)
    dx[:, 1:-1] = P[:, 2:] - P[:, :-2]; dy[1:-1] = P[2:] - P[:-2]
    n = np.cross(dx, dy)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    ok = Z > 0
    for a, b in [(0, 1), (0, -1), (1, 1), (1, -1)]:        # all 4 neighbours on the surface, no depth jump
        nb = np.roll(Z, b, axis=a)
        ok &= (nb > 0) & (np.abs(nb - Z) < 0.05 * Z)
    ok[0] = ok[-1] = ok[:, 0] = ok[:, -1] = False
    return n, ok


def score(S, Z, M, V, Nm, okm, K, dtM):
    """All candidate goals for one rendered pose against the photo (M mask, V visible area, Nm/okm photo normals)."""
    Sv = S & V
    nS, nM = S.sum(), M.sum()
    inter = (S & M).sum()
    g = dict(dice_plain=1 - 2 * inter / (nS + nM), dice_masked=1 - 2 * (Sv & M).sum() / (Sv.sum() + nM))
    diag = np.hypot(*S.shape)
    if nS:
        dtS = cv2.distanceTransform((~S).astype(np.uint8), cv2.DIST_L2, 5)
        g["A1"] = float(dtS[M].mean()) * DOWN
    else:
        g["A1"] = float(diag) * DOWN
    g["A2"] = float(dtM[Sv].mean()) * DOWN if Sv.any() else 0.0
    g["A"] = g["A1"] + g["A2"]
    if inter:
        Ns, oks = normals(Z, K)
        both = M & S & okm & oks
        g["B"] = float((1 - (Ns[both] * Nm[both]).sum(-1)).mean()) if both.sum() > 20 else 1.0
    else:
        g["B"] = 1.0
    return g


def rot(axis, deg):
    return cv2.Rodrigues(np.eye(3)[axis] * np.radians(deg))[0]


def perturbed(R, t, c_cam, dR=np.eye(3), dt=np.zeros(3)):
    """Move the CT about its centre in the camera frame: X' = dR (X_cam - c) + c + dt."""
    return dR @ R, dR @ (t - c_cam) + c_cam + dt


SWEEPS = {
    "distance": np.linspace(0.5, 1.5, 21),
    "side_x": np.linspace(-50, 50, 21), "side_y": np.linspace(-50, 50, 21),
    "rot_x": np.linspace(-45, 45, 19), "rot_y": np.linspace(-45, 45, 19), "rot_z": np.linspace(-45, 45, 19),
}
GOALS = ["dice_plain", "dice_masked", "A1", "A2", "A", "B"]


def sweep_pose(name, x, R, t, c):
    if name == "distance":
        return perturbed(R, t, c, dt=(x - 1) * c)
    if name.startswith("side"):
        return perturbed(R, t, c, dt=x * np.eye(3)["xy".index(name[-1])])
    return perturbed(R, t, c, dR=rot("xyz".index(name[-1]), x))


def pick_views(p, n, rng):
    keys = [l.strip().replace("ptsLiverCam", "") for l in open(f"{DATA}/{p}/train_syn.txt") if l.strip()]
    keys = [k for k in keys if not (p in ("10", "11") and 2000 <= int(k) < 2500)]   # extra-object views
    return sorted(rng.choice(keys, n, replace=False).tolist())


def phase1(pat, p, keys):
    out = []
    for k in keys:
        R, t = pat.true_pose(k)
        Ks = k_matrix(pat.syn_k[k][0]["cam_K"])
        lbl = cv2.imread(f"{DATA}/{p}/syn/labels/lbl{k}.png")[..., 0]
        H, W = lbl.shape
        S, _ = render(pat.scene, R, t, Ks, W, H)
        tool = (lbl == 153) | (lbl == 179)
        liver = lbl == 0
        a, b = S & ~tool, liver
        out.append(dict(view=k, iou=float((a & b).sum() / max((a | b).sum(), 1)), liver_px=int(b.sum())))
    return out


def phase2(pat, p, k, rng):
    R, t = pat.true_pose(k)
    c = R @ pat.centre + t
    K = pat.real_k
    S0, Z0 = render(pat.scene, R, t, K, RW, RH)
    u, v = np.meshgrid(np.arange(RW) + 0.5, np.arange(RH) + 0.5)
    scope = np.hypot(u - RW / 2, v - RH / 2) <= SCOPE_R * RW
    photo_depth = Z0 * rng.uniform(0.2, 5.0)                            # perfect shape, useless distance
    Nm, okm = normals(photo_depth, K)
    res = dict(view=k, true_distance_mm=float(np.linalg.norm(c)), visible_fraction=float(S0.mean()))
    for cond, V in [("clean", np.ones_like(S0)), ("scope", scope)]:
        M = S0 & V
        if M.sum() < 200:
            res[cond] = None
            continue
        dtM = cv2.distanceTransform((~M).astype(np.uint8), cv2.DIST_L2, 5)
        res[cond] = {}
        for name, xs in SWEEPS.items():
            rows = []
            for x in xs:
                S, Z = render(pat.scene, *sweep_pose(name, x, R, t, c), K, RW, RH)
                rows.append([score(S, Z, M, V, Nm, okm, K, dtM)[g] for g in GOALS])
            res[cond][name] = np.array(rows).tolist()
    return res


def look_at(centre, direction, dist, roll):
    """Camera at centre + dist*direction looking at the centre (vision axes), rolled about the view axis."""
    z = -direction / np.linalg.norm(direction)
    up = np.array([0, 0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0, 0])
    x = np.cross(z, up); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = rot(2, roll) @ np.stack([x, y, z])
    return R, -R @ (centre + dist * direction / np.linalg.norm(direction))


def phase3(pat, p, k):
    R, t = pat.true_pose(k)
    c = R @ pat.centre + t
    K = pat.real_k
    S0, Z0 = render(pat.scene, R, t, K, RW, RH)
    M, V = S0, np.ones_like(S0)
    Nm, okm = normals(Z0, K)
    dtM = cv2.distanceTransform((~M).astype(np.uint8), cv2.DIST_L2, 5)
    true_dir = -R.T @ t - pat.centre
    dist = np.linalg.norm(true_dir); true_dir /= dist
    n = 64
    i = np.arange(n) + 0.5
    phi, th = np.arccos(1 - 2 * i / n), np.pi * (1 + 5 ** 0.5) * i
    dirs = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
    rows = []
    for d in dirs:
        best = None
        for roll in range(0, 360, 45):
            for f in [0.7, 0.85, 1.0, 1.2, 1.4]:
                S, Z = render(pat.scene, *look_at(pat.centre, d, dist * f, roll), K, RW, RH)
                g = score(S, Z, M, V, Nm, okm, K, dtM)
                if best is None or g["A"] < best["A"]:
                    best = dict(g, roll=roll, dist_factor=f)
        rows.append(dict(best, angle_to_true=float(np.degrees(np.arccos(np.clip(d @ true_dir, -1, 1))))))
    return dict(view=k, directions=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", type=int, default=20)
    ap.add_argument("--scan-views", type=int, default=3)
    ap.add_argument("--only", choices=["phase1", "phase2", "phase3"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    out = dict(sweeps={k: v.tolist() for k, v in SWEEPS.items()}, goals=GOALS, patients={})
    for p in PATIENTS:
        pat = Patient(p)
        keys = pick_views(p, a.views, rng)
        po = out["patients"][p] = dict(views=keys, real_k=pat.real_k_full.tolist())
        t0 = time.time()
        if a.only in (None, "phase1"):
            po["phase1"] = phase1(pat, p, keys)
            ious = [r["iou"] for r in po["phase1"]]
            print(f"{p} phase 1: CT drawn at the true pose vs dataset mask: IoU median {np.median(ious):.4f}, "
                  f"worst {min(ious):.4f} ({time.time() - t0:.0f} s)", flush=True)
        if a.only in (None, "phase2"):
            po["phase2"] = [phase2(pat, p, k, rng) for k in keys]
            print(f"{p} phase 2: {len(keys)} views swept ({time.time() - t0:.0f} s)", flush=True)
        if a.only in (None, "phase3"):
            po["phase3"] = [phase3(pat, p, k) for k in keys[:a.scan_views]]
            print(f"{p} phase 3: {a.scan_views} views scanned all around ({time.time() - t0:.0f} s)", flush=True)
        json.dump(out, open(f"{OUT}/landscape.json", "w"))
    print("wrote", f"{OUT}/landscape.json")


if __name__ == "__main__":
    main()
