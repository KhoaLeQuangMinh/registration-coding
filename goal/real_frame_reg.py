"""
Rigid registration on ONE real P2I-LReg photo, from VLM-style starting guesses (propose and verify).

Inputs: the photo's liver label (mask), the depth SHAPE from the released P2I-LReg cloud (Depth Anything; its distance is
ignored), the calibrated camera K (no distortion coefficients are given), the CT at its real size.
The scope circle is found in the photo; CT pixels outside it are hidden, so the mask's border along it is a false edge.
Goal J = A/4.5 + B/0.02 (same weights as the synthetic tests), A = A1 + A2 with A2 counted only inside the scope circle.
Each alternative = one of the 6 parts seen head-on (show_classes.py) with a head direction; its starts are directions
within 30 deg of that view x spins within 45 deg, size + centre matched; the best 8 are polished. "all-around" = the
unrestricted search (256 directions x 16 spins). No ground truth exists for these photos: J says which fits best, the
overlay lets a person judge.
Usage: python goal/real_frame_reg.py --patient 01 --frame 10 --alt 2:up 0:up 1:up
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np
import open3d as o3d
from scipy.optimize import minimize

sys.path.insert(0, "goal")
import show_classes as sc  # noqa: E402
from goal_landscape import look_at, normals, perturbed, render  # noqa: E402

W, H, DOWN = 480, 270, 4
WA, WB = 4.5, 0.02
HEADS = {h: i for i, h in enumerate(sc.HEAD)}


def load_photo(p, f):
    d = f"dataset/liver-registration/{p}/real/labels/frame_{f}_json"
    img = cv2.imread(f"{d}/img.png")
    M = cv2.resize((cv2.imread(f"{d}/label.png")[..., 2] > 0).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    grey = cv2.cvtColor(cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    # not tissue: the dark corners outside the scope and the glaring white scope rim, i.e. very dark or very bright
    # regions that touch the image border (shiny spots ON the liver do not touch the border, so they stay)
    bad = np.zeros_like(M)
    for cand in (grey < 30, grey > 225):
        n, cc = cv2.connectedComponents(cand.astype(np.uint8))
        edge = np.unique(np.concatenate([cc[0], cc[-1], cc[:, 0], cc[:, -1]]))
        bad |= np.isin(cc, edge[edge > 0])
    bad = cv2.dilate(bad.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    V = ~bad | M
    pts = np.asarray(o3d.io.read_point_cloud(f"dataset/liver-registration/{p}/real/liverPcds/frame_{f}_json.ply").points)
    P = pts.reshape(1080, 1920, 3)
    lab_full = cv2.imread(f"{d}/label.png")[..., 2] > 0
    z_cam = np.median(P[~lab_full][:, 2])                                 # background pixels sit at the camera point
    D = np.where(lab_full, P[..., 2] - z_cam, 0)                         # depth up to the cloud's scale
    D = cv2.resize(D.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
    return img, M, V, D


class Frame:
    def __init__(self, p, f):
        self.img, self.M, self.V, D = load_photo(p, f)
        K = np.array(list(__import__("yaml").safe_load(open(f"dataset/liver-registration/{p}/real/CAM_K.yml")).values())[0][0]["cam_K"]).reshape(3, 3)
        self.K = np.diag([1 / DOWN, 1 / DOWN, 1.0]) @ K
        self.N, self.ok = normals(np.where(self.M, D, 0), self.K)
        self.dtM = cv2.distanceTransform((~self.M).astype(np.uint8), cv2.DIST_L2, 5)


def terms(scene, R, t, fr):
    S, Z = render(scene, R, t, fr.K, W, H)
    if not S.any():
        return 1e3, 1.0
    Sv = S & fr.V
    dtS = cv2.distanceTransform((~S).astype(np.uint8), cv2.DIST_L2, 5)
    A = (dtS[fr.M].mean() + (fr.dtM[Sv].mean() if Sv.any() else 0)) * DOWN
    Ns, oks = normals(Z, fr.K)
    both = fr.M & S & fr.ok & oks
    B = float((1 - (Ns[both] * fr.N[both]).sum(-1)).mean()) if both.sum() > 20 else 1.0
    return A, B


def J(A, B):
    return A / WA + B / WB


def size_and_centre(scene, centre, R, t, fr):
    S, _ = render(scene, R, t, fr.K, W, H)
    if S.sum() < 20:
        return R, t
    c = R @ centre + t
    Sv = S & fr.V
    k = np.sqrt(max(Sv.sum(), 20) / fr.M.sum())
    c2 = c * k
    vS, uS = np.argwhere(Sv if Sv.any() else S).mean(0); vM, uM = np.argwhere(fr.M).mean(0)
    c2 = c2 + np.array([(uM - uS) * c2[2] / fr.K[0, 0], (vM - vS) * c2[2] / fr.K[1, 1], 0.0])
    return R, t + (c2 - c)


def polish(scene, R0, t0, centre, fr):
    c0 = R0 @ centre + t0

    def pose(x):
        return perturbed(R0, t0, c0, dR=cv2.Rodrigues(np.radians(x[:3]))[0], dt=x[3:])

    def f(x):
        return J(*terms(scene, *pose(x), fr))

    simplex = np.vstack([np.zeros(6)] + [np.eye(6)[i] * (10 if i < 3 else 20) for i in range(6)])
    r = minimize(f, np.zeros(6), method="Nelder-Mead", options=dict(initial_simplex=simplex, maxfev=300, xatol=0.1, fatol=1e-3))
    return (*pose(r.x), r.fun)


def fib(n):
    i = np.arange(n) + 0.5
    phi, th = np.arccos(1 - 2 * i / n), np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def run_alt(args):
    p, f, alt = args
    m, V, T, C, Nt = sc.load(p)
    lab = sc.parts(V, C, Nt, T, m)
    scene = o3d.t.geometry.RaycastingScene(); scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
    fr = Frame(p, f)
    centre = V.mean(0)
    t0 = time.time()
    starts = []
    if alt == "all-around":
        dist = 180.0
        for d in fib(256):
            for roll in np.arange(0, 360, 22.5):
                starts.append(look_at(centre, d, dist, roll))
    else:
        part, head = alt.split(":")
        R0, _, target = sc.class_pose(V, T, C, Nt, lab, int(part), HEADS[head])
        view = R0[2]                                                      # camera forward
        for d in fib(400):
            if np.degrees(np.arccos(np.clip(d @ -view, -1, 1))) <= 30:
                for spin in np.arange(-45, 46, 15):
                    Rd, td = look_at(target, d, 150.0, 0)
                    # choose the roll so the image-up matches the class camera, then add the spin
                    roll0 = np.degrees(np.arctan2(*(Rd @ R0[1])[[0, 1]][::-1])) if False else 0
                    best, bestang = None, 1e9
                    for r_ in np.arange(0, 360, 5):
                        Rr, tr = look_at(target, d, 150.0, r_)
                        ang = np.degrees(np.arccos(np.clip((Rr[1] @ R0[1]), -1, 1)))
                        if ang < bestang:
                            best, bestang = r_, ang
                    starts.append(look_at(target, d, 150.0, best + spin))
    scored = []
    for R, t in starts:
        R, t = size_and_centre(scene, centre, R, t, fr)
        scored.append((J(*terms(scene, R, t, fr)), R, t))
    scored.sort(key=lambda s: s[0])
    best = None
    for _, R, t in scored[:8]:
        Rf, tf, fv = polish(scene, R, t, centre, fr)
        if best is None or fv < best[2]:
            best = (Rf, tf, fv)
    Rf, tf, fv = best
    A, B = terms(scene, Rf, tf, fr)
    S, Z = render(scene, Rf, tf, fr.K, W, H)
    share = np.zeros(6)
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d_ = np.stack([(u - fr.K[0, 2]) / fr.K[0, 0], (v - fr.K[1, 2]) / fr.K[1, 1], np.ones_like(u)], -1).reshape(-1, 3)
    d_ /= np.linalg.norm(d_, axis=1, keepdims=True)
    rays = np.hstack([np.repeat((-Rf.T @ tf)[None], len(d_), 0), d_ @ Rf]).astype(np.float32)
    prim = scene.cast_rays(o3d.core.Tensor(rays))["primitive_ids"].numpy().reshape(H, W)
    vis = (prim != o3d.t.geometry.RaycastingScene.INVALID_ID) & fr.V
    if vis.any():
        share = np.bincount(lab[prim[vis]], minlength=6) / vis.sum()
    dist = float(np.linalg.norm(Rf @ centre + tf))
    return dict(alt=alt, starts=len(starts), J=float(fv), A=float(A), B=float(B), distance_mm=dist,
                parts={sc.PARTS[k]: round(float(share[k]), 2) for k in range(6) if share[k] > 0.02},
                R=Rf.tolist(), t=tf.tolist(), seconds=round(time.time() - t0))


def overlay(p, f, res, out):
    m, V, T, C, Nt = sc.load(p)
    lab = sc.parts(V, C, Nt, T, m)
    scene = o3d.t.geometry.RaycastingScene(); scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
    fr = Frame(p, f)
    base = cv2.resize(fr.img, (W * 2, H * 2), interpolation=cv2.INTER_AREA)
    Kb = np.diag([2.0, 2.0, 1.0]) @ fr.K
    tiles = []
    for r in res:
        R, t = np.array(r["R"]), np.array(r["t"])
        col = cv2.resize(sc.render_view(scene, Nt, lab, R, t, fr.K), (W * 2, H * 2), interpolation=cv2.INTER_NEAREST)
        S, _ = render(scene, R, t, Kb, W * 2, H * 2)
        im = base.copy()
        im[S] = (0.45 * im[S] + 0.55 * col[S]).astype(np.uint8)
        Mb = cv2.resize(fr.M.astype(np.uint8), (W * 2, H * 2), interpolation=cv2.INTER_NEAREST)
        cnt, _ = cv2.findContours(Mb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(im, cnt, -1, (0, 255, 255), 2)
        txt = [f"start: {r['alt']}", f"score J {r['J']:.2f} (outline {r['A']:.1f} px, shape {r['B']:.3f})",
               f"CT distance {r['distance_mm']:.0f} mm; sees: " + ", ".join(f"{k.split(',')[0]} {v:.0%}" for k, v in r["parts"].items())]
        for i, s in enumerate(txt):
            cv2.putText(im, s, (10, 26 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(im, s, (10, 26 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(im)
    while len(tiles) % 2:
        tiles.append(np.full_like(tiles[0], 255))
    cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default="01")
    ap.add_argument("--frame", default="10")
    ap.add_argument("--alt", nargs="+", default=["2:up", "0:up", "1:up"])
    a = ap.parse_args()
    jobs = [(a.patient, a.frame, x) for x in a.alt + ["all-around"]]
    with Pool(len(jobs)) as pool:
        res = pool.map(run_alt, jobs)
    for r in res:
        print(f"{r['alt']:11s} {r['starts']:5d} starts, {r['seconds']:4d} s: J {r['J']:.2f} (A {r['A']:.1f} px, B {r['B']:.3f}), "
              f"CT at {r['distance_mm']:.0f} mm, sees {r['parts']}")
    json.dump(res, open(f"goal/out/real_{a.patient}_{a.frame}.json", "w"), indent=1)
    overlay(a.patient, a.frame, res, f"goal/out/real_{a.patient}_{a.frame}.png")
    print(f"wrote goal/out/real_{a.patient}_{a.frame}.png")


if __name__ == "__main__":
    main()
