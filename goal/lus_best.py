"""
ORACLE registration of every LLR-LUS image: the best registration the data allows, using the answer.

This is NOT a method (it uses the ground truth); it shows what a correct registration looks like and how close a
rigid / bent CT can get. Information used, per image:
    tumour     the ultrasound tumour profile (camera frame, mm): the CT tumour surface must pass through it
    contact    the tracked probe face (camera frame, mm; lus_oracle.probe_face): it touches the liver surface
    outline    the user's hand-drawn liver mask, true edges only (automatic hidden-edge rule, lus_all.AutoFrame)
    landmarks  Koo et al.'s hand-labelled ridge and ligament: 2D curves in the photo + their vertices on the CT
Score (lower = better):  E = 2 * tumour_mm + contact_mm + outline_px / 4.5 + landmark_px / 18
(landmarks weigh least: Koo's ligament sits ~50 px off its curve even at poses with a 3 mm tumour error)
1. Rigid: 400 directions x 12 rolls with the CT tumour centre put on the profile centre, ranked by the cheap terms,
   the best 200 scored fully, the best 8 (plus our 3 saved poses) polished (Nelder-Mead), then a fine re-polish.
2. Bend (LIBR-style, the same model as the playground's Bend tool): 40 control points on the CT (farthest-point
   sample), each pushing the tissue with a Gaussian of reach SIGMA. Pushes are fitted by iterated least squares to
   the four kinds of correspondence above, with a penalty on the push size; then the pose is re-polished; twice.
Reported per image: tumour error (mean distance from the profile to the nearest point of the CT tumour surface;
depth / sideways), probe contact gap, outline gap A, landmark gap, Dice, how far the bend moved the tissue.
Writes dataset/lus-evaluation/goal_results/oracle_{rigid,bent}/PatientN/<image>.stl + oracle_per_image/*.json and runs
the official score on both.
Usage: python goal/lus_best.py [--workers 6] [--only Patient1/02 ...]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np
import open3d as o3d
import scipy.io as sio
from scipy.optimize import minimize
from scipy.spatial import cKDTree

sys.path.insert(0, "goal"); sys.path.insert(0, "studies/benchmark")
import lus_all as la  # noqa: E402
import lus_edges as le  # noqa: E402
import lus_frame as lf  # noqa: E402
import lus_oracle as lo  # noqa: E402
import real_frame_reg as rf  # noqa: E402
import deform_frame as dfm  # noqa: E402
from goal_landscape import look_at, render  # noqa: E402

LUS = "dataset/lus-evaluation"
DSET = f"{LUS}/data/Dataset"
RES = f"{LUS}/goal_results"
W, H, DOWN = rf.W, rf.H, rf.DOWN
WT, WC, WA, WL = 2.0, 1.0, 4.5, 18.0
NH, SIGMA = 40, 40.0
F_GL = np.diag([1.0, -1.0, -1.0])


# ---------------- data ----------------
def koo_to_ply(p):
    """Rigid transform Koo's Liver.obj -> the dataset's Liver.ply (same model, shifted; ICP from the centroid shift)."""
    # raw "v" lines: Open3D re-orders this OBJ's vertices (it splits them by normal index), and Koo's landmark
    # vertex numbers refer to the file order
    a = np.array([l.split()[1:4] for l in open(f"{LUS}/koo/contours_and_models/{p}/Liver.obj") if l.startswith("v ")], float)
    b = np.asarray(o3d.io.read_triangle_mesh(f"{DSET}/{p}/Preoperative Model/Liver.ply").vertices)
    pa = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(a))
    pb = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(b)); pb.estimate_normals()
    T0 = np.eye(4); T0[:3, 3] = b.mean(0) - a.mean(0)
    r = o3d.pipelines.registration.registration_icp(pa, pb, 10.0, T0, o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                                                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    return a, r.transformation, float(np.asarray(pa.transform(r.transformation).compute_point_cloud_distance(pb)).mean())


def landmarks(p, name, kv, T, Ralign, c, Kf, dist):
    """Koo's ridge / ligament: (type, 2D curve in the undistorted photo (px), 3D points in the body frame)."""
    path = f"{LUS}/koo/contours_and_models/{p}/Annotations/{name}/koo/contours.xml"
    if not os.path.exists(path):
        return []
    x = open(path).read()
    out = []
    for blk in re.findall(r"<contour>(.*?)</contour>", x, re.S):
        typ = re.search(r"<contourType>(.*?)<", blk).group(1)
        mp = re.search(r"<modelPoints>.*?<vertices>(.*?)</vertices>", blk, re.S)
        if typ == "Silhouette" or not mp or not mp.group(1).strip():
            continue
        ip = re.search(r"<imagePoints>.*?<x>(.*?)</x>.*?<y>(.*?)</y>", blk, re.S)
        uv = np.stack([np.array(ip.group(1).split(","), float), np.array(ip.group(2).split(","), float)], 1)
        uv = cv2.undistortPoints(uv.reshape(-1, 1, 2), Kf, dist, P=Kf).reshape(-1, 2)
        idx = np.array(mp.group(1).split(","), int)
        P = kv[idx] @ T[:3, :3].T + T[:3, 3]
        out.append(dict(type=typ, curve=uv, pts=(P - c) @ Ralign.T))
    return out


def probe_and_plane(p, name):
    """Probe face (camera frame, mm) and the ultrasound image rectangle (4 corners, camera frame) + its picture."""
    cal = sio.loadmat(f"{DSET}/{p}/LUS Calibration and Pose/{name}.mat")
    to_cam = lambda u, v: (cal["Rpr"] @ (cal["Rus"] @ np.vstack([cal["SLus"][0, 0] * u, np.zeros_like(u), cal["SAus"][0, 0] * v]) + cal["Tus"]) + cal["Tpr"]).T  # noqa: E731
    arc = lo.probe_face()
    us = cv2.imread(f"{DSET}/{p}/LUS Images/{name}.jpg")
    h, w = us.shape[:2]
    corners = to_cam(np.array([0.0, w, w, 0.0]), np.array([0.0, 0.0, h, h]))
    return to_cam(arc[:, 0], arc[:, 1]), corners, us


class Case:
    def __init__(self, image, align, koo):
        p, name = image.split("/")
        self.image, self.p, self.name = image, p, name
        a = align[p]
        Ralign, c = np.array(a["R"]), np.array(a["c"])
        m, V, T, C, Nt, tum = lf.load_ct(p, Ralign, c)
        self.V, self.T, self.tum = V, T, tum
        self.tumF = np.asarray(o3d.io.read_triangle_mesh(f"{DSET}/{p}/Preoperative Model/Tumour.ply").triangles)
        self.centre = V.mean(0)
        self.fr = la.AutoFrame(image)
        xml = open(f"{DSET}/{p}/Lap Images/CameraParameters").read()
        g = lambda k: float(re.search(f"<{k}>(.*?)</{k}>", xml).group(1))  # noqa: E731
        self.dist = np.array([g("k1"), g("k2"), g("p1"), g("p2"), g("k3")])
        self.Kf = self.fr.Kfull
        self.Wf = self.fr.img.shape[1]
        kv, kT, _ = koo
        self.lms = landmarks(p, name, kv, kT, Ralign, c, self.Kf, self.dist)
        self.face, self.usq, self.usimg = probe_and_plane(p, name)
        self.prof = self.fr.profile
        # control points of the bend: farthest-point sample of the CT surface
        idx = [0]; d = np.full(len(V), np.inf)
        for _ in range(NH - 1):
            d = np.minimum(d, ((V - V[idx[-1]]) ** 2).sum(1)); idx.append(int(d.argmax()))
        self.hp = V[idx]
        self.D = np.zeros((NH, 3))
        self.set_shape(self.D)
        self.Mtrue = None

    # deformation: Gaussian pushes at the control points
    def phi(self, X):
        return np.exp(-((X[:, None] - self.hp[None]) ** 2).sum(2) / (2 * SIGMA ** 2))

    def set_shape(self, D):
        self.D = D
        if not hasattr(self, "phiV"):
            self.phiV, self.phiT = self.phi(self.V), self.phi(self.tum)
        self.Vd, self.tumd = self.V + self.phiV @ D, self.tum + self.phiT @ D
        mesh = o3d.t.geometry.TriangleMesh(); mesh.vertex.positions = o3d.core.Tensor(self.Vd.astype(np.float32))
        mesh.triangle.indices = o3d.core.Tensor(self.T.astype(np.int32))
        self.scene = o3d.t.geometry.RaycastingScene(); self.scene.add_triangles(mesh)
        tm = o3d.t.geometry.TriangleMesh(); tm.vertex.positions = o3d.core.Tensor(self.tumd.astype(np.float32))
        tm.triangle.indices = o3d.core.Tensor(self.tumF.astype(np.int32))
        self.tscene = o3d.t.geometry.RaycastingScene(); self.tscene.add_triangles(tm)

    # ---------------- terms ----------------
    @staticmethod
    def closest(scene, P):
        r = scene.compute_closest_points(o3d.core.Tensor(P.astype(np.float32)))
        return r["points"].numpy()

    def tumour(self, R, t):
        """Profile -> nearest point of the CT tumour surface, camera frame."""
        qb = (self.prof - t) @ R                       # profile in the body frame
        near = self.closest(self.tscene, qb) @ R.T + t
        diff = near - self.prof
        return dict(tumour_mm=float(np.linalg.norm(diff, axis=1).mean()), depth_mm=float(diff[:, 2].mean()),
                    across_mm=float(np.linalg.norm(diff[:, :2], axis=1).mean()))

    def contact(self, R, t):
        fb = (self.face - t) @ R
        return float(np.linalg.norm(self.closest(self.scene, fb) - fb, axis=1).mean())

    def landmark_px(self, R, t, Vsrc=None):
        if not self.lms:
            return 0.0
        errs = []
        for lm in self.lms:
            P = lm["pts"] if Vsrc is None else lm["pts_d"]
            q = (self.Kf @ (P @ R.T + t).T).T
            if (q[:, 2] <= 0).any():
                return 1e3
            uv = q[:, :2] / q[:, 2:]
            errs.append(cKDTree(lm["curve"]).query(uv)[0])
        return float(np.concatenate(errs).mean() * 1920 / self.Wf)

    def cheap(self, R, t):
        return WT * self.tumour(R, t)["tumour_mm"] + WC * self.contact(R, t)

    def full(self, R, t, detail=False):
        A, B = rf.terms(self.scene, R, t, self.fr)
        tu, co, L = self.tumour(R, t), self.contact(R, t), self.landmark_px(R, t, True if self.D.any() else None)
        E = WT * tu["tumour_mm"] + WC * co + A / WA + L / WL
        if detail:
            return E, dict(E=float(E), A=float(A), B=float(B), contact_mm=co, landmark_px=L, **tu)
        return E

    def update_lm(self):
        for lm in self.lms:
            lm["pts_d"] = lm["pts"] + self.phi(lm["pts"]) @ self.D

    def polish(self, R0, t0, maxfev, step_deg, step_mm):
        c0 = R0 @ self.centre + t0

        def pose(x):
            dR = cv2.Rodrigues(np.radians(x[:3]))[0]
            return dR @ R0, dR @ (t0 - c0) + c0 + x[3:]
        simplex = np.vstack([np.zeros(6)] + [np.eye(6)[i] * (step_deg if i < 3 else step_mm) for i in range(6)])
        r = minimize(lambda x: self.full(*pose(x)), np.zeros(6), method="Nelder-Mead",
                     options=dict(initial_simplex=simplex, maxfev=maxfev, xatol=0.05, fatol=1e-3))
        return (*pose(r.x), float(r.fun))

    # ---------------- 1. rigid ----------------
    def rigid(self, saved):
        qbar, cb = self.prof.mean(0), self.tum.mean(0)
        cands = []
        for d in rf.fib(400):
            for roll in range(0, 360, 30):
                R = look_at(self.centre, d, 1.0, roll)[0]
                t = qbar - R @ cb
                cands.append((self.cheap(R, t), R, t))
        cands.sort(key=lambda x: x[0])
        scored = sorted(((self.full(R, t), R, t) for _, R, t in cands[:200]), key=lambda x: x[0])
        pool = [(R, t) for _, R, t in scored[:8]] + [(np.array(s["R"]), np.array(s["t"])) for s in saved]
        pol = sorted((self.polish(R, t, 500, 5.0, 10.0) for R, t in pool), key=lambda x: x[2])
        R, t, _ = pol[0]
        R, t, _ = self.polish(R, t, 300, 1.5, 3.0)
        return R, t

    # ---------------- 2. bend ----------------
    def bend_step(self, R, t, lam):
        """One least-squares update of the pushes D from the current correspondences (pose fixed)."""
        rows, rhs, wts = [], [], []

        def add3(Xrest, target_minus_now, w):        # full 3D pull of body points (rest positions Xrest)
            ph = self.phi(Xrest)
            for k in range(3):
                a = np.zeros((len(Xrest), NH, 3)); a[:, :, k] = ph
                rows.append(a.reshape(len(Xrest), -1)); rhs.append(target_minus_now[:, k]); wts.append(np.full(len(Xrest), w / len(Xrest)))

        def add2(Xrest, dcam, w):                      # image-plane pull (camera x, y only), dcam: n x 2 in mm
            ph = self.phi(Xrest)
            for k in range(2):
                axis = R[k]                            # camera axis k in the body frame
                a = ph[:, :, None] * axis[None, None, :]
                rows.append(a.reshape(len(Xrest), -1)); rhs.append(dcam[:, k]); wts.append(np.full(len(Xrest), w / len(Xrest)))
        # tumour: each profile point <- nearest deformed tumour vertex
        qb = (self.prof - t) @ R
        j = cKDTree(self.tumd).query(qb)[1]
        add3(self.tum[j], qb - self.tumd[j], 4.0)
        # probe face <- nearest deformed liver vertex
        fb = (self.face - t) @ R
        k = cKDTree(self.Vd).query(fb)[1]
        add3(self.V[k], fb - self.Vd[k], 1.0)
        # outline: true mask edges, CT edges, spill (as lus_frame.bend), image plane at the point's depth
        fr = self.fr
        S, Z, prim = dfm.cast(self.Vd, self.T, R, t, fr.K)
        border = np.zeros((H, W), bool); border[:2] = border[-2:] = True; border[:, :2] = border[:, -2:] = True
        hidden = cv2.dilate((~fr.V).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        Mtrue = dfm.edges(fr.M) & ~hidden & ~border
        from scipy.ndimage import distance_transform_edt
        _, (iy, ix) = distance_transform_edt(~Mtrue, return_indices=True)
        Sedge = dfm.edges(S) & fr.V & ~border
        if Sedge.any():
            _, (sy, sx) = distance_transform_edt(~Sedge, return_indices=True)
            src = [(y, x, iy[y, x], ix[y, x]) for y, x in np.argwhere(Sedge)]
            src += [(sy[y, x], sx[y, x], y, x) for y, x in np.argwhere(Mtrue)]
            spill = np.argwhere(S & fr.V & ~fr.M & ~border); spill = spill[::max(1, len(spill) // 1500)]
            src += [(y, x, iy[y, x], ix[y, x]) for y, x in spill]
            src = np.array(src); src = src[np.hypot(src[:, 2] - src[:, 0], src[:, 3] - src[:, 1]) < 60]
            ok = prim[src[:, 0], src[:, 1]] != o3d.t.geometry.RaycastingScene.INVALID_ID
            src = src[ok]
            if len(src):
                z = Z[src[:, 0], src[:, 1]]
                vid = self.T[prim[src[:, 0], src[:, 1]]][:, 0]
                dcam = np.stack([(src[:, 3] - src[:, 1]) * z / fr.K[0, 0], (src[:, 2] - src[:, 0]) * z / fr.K[1, 1]], 1)
                add2(self.V[vid], dcam, 1.0)
        # landmarks: projected (deformed) landmark point -> nearest point of its 2D curve
        for lm in self.lms:
            P = lm["pts"] + self.phi(lm["pts"]) @ self.D
            cam = P @ R.T + t
            uv = (self.Kf @ cam.T).T; uv = uv[:, :2] / uv[:, 2:]
            near = lm["curve"][cKDTree(lm["curve"]).query(uv)[1]]
            dcam = (near - uv) * cam[:, 2:] / self.Kf[0, 0]
            add2(lm["pts"], dcam, 0.5)
        A = np.vstack(rows); b = np.concatenate(rhs); w = np.sqrt(np.concatenate(wts))
        # penalty on the push size (energy stand-in): lam * |D + dD|^2 / NH
        reg = np.sqrt(lam / NH)
        lhs = np.vstack([A * w[:, None], reg * np.eye(NH * 3)])
        rhs_ = np.concatenate([b * w, -reg * self.D.reshape(-1)])
        dD = np.linalg.lstsq(lhs, rhs_, rcond=None)[0].reshape(NH, 3)
        self.set_shape(self.D + 0.7 * dD)

    def bend(self, R, t, lam, rounds=2, iters=10):
        """Alternate push fitting and pose polish; keep only states that lower E (the fit can overshoot)."""
        self.update_lm()
        best = (self.full(R, t), self.D.copy(), R, t)
        for _ in range(rounds):
            for _ in range(iters):
                self.bend_step(R, t, lam)
                self.update_lm()
                e = self.full(R, t)
                if e < best[0]:
                    best = (e, self.D.copy(), R, t)
            self.set_shape(best[1]); self.update_lm()
            R, t = best[2], best[3]
            R, t, e = self.polish(R, t, 200, 1.5, 3.0)
            if e < best[0]:
                best = (e, self.D.copy(), R, t)
        self.set_shape(best[1]); self.update_lm()
        return best[2], best[3]

    def moved(self):
        m = np.linalg.norm(self.phiV @ self.D, axis=1)
        return dict(moved_mean_mm=float(m.mean()), moved_max_mm=float(m.max()),
                    tumour_moved_mm=float(np.linalg.norm(self.phiT @ self.D, axis=1).mean()))

    def dice(self, R, t):
        S, _ = render(self.scene, R, t, self.fr.K, W, H)
        Sv = S & self.fr.V
        return float(2 * (Sv & self.fr.M).sum() / (Sv.sum() + self.fr.M.sum()))


_KOO = {}


def run(job):
    image, align, lam = job
    p, name = image.split("/")
    if p not in _KOO:
        _KOO[p] = koo_to_ply(p)
    t0 = time.time()
    cs = Case(image, align, _KOO[p])
    hint = json.load(open(f"{RES}/hint_per_image/{p}_{name}.json"))
    saved = [hint["A"], hint["no_hint"], min(hint["suggestions"], key=lambda s: s["tumour_mm"])]
    R, t = cs.rigid(saved)
    _, rig = cs.full(R, t, detail=True); rig["dice"] = cs.dice(R, t)
    Rr, tr = R.copy(), t.copy()
    Rb, tb = cs.bend(R, t, lam)
    _, ben = cs.full(Rb, tb, detail=True); ben["dice"] = cs.dice(Rb, tb); ben.update(cs.moved())
    res = dict(image=image, rigid=dict(R=Rr.tolist(), t=tr.tolist(), **rig),
               bent=dict(R=Rb.tolist(), t=tb.tolist(), handles=cs.hp.tolist(), D=cs.D.tolist(), sigma=SIGMA, lam=lam, **ben),
               koo_fit_mm=_KOO[p][2], landmarks=[lm["type"] for lm in cs.lms], seconds=round(time.time() - t0))
    for variant, (Rv, tv, tum) in [("rigid", (Rr, tr, cs.tum)), ("bent", (Rb, tb, cs.tumd))]:
        cam = tum @ Rv.T + tv
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(cam @ F_GL.T), o3d.utility.Vector3iVector(cs.tumF))
        mesh.compute_triangle_normals()
        os.makedirs(f"{RES}/oracle_{variant}/{p}", exist_ok=True)
        o3d.io.write_triangle_mesh(f"{RES}/oracle_{variant}/{p}/{name}.stl", mesh)
    os.makedirs(f"{RES}/oracle_per_image", exist_ok=True)
    json.dump(res, open(f"{RES}/oracle_per_image/{p}_{name}.json", "w"), indent=1)
    print(f"{image}: rigid tumour {rig['tumour_mm']:5.1f} mm (depth {rig['depth_mm']:+5.1f}), contact {rig['contact_mm']:4.1f}, "
          f"outline {rig['A']:4.1f} px, landmarks {rig['landmark_px']:5.1f} px, Dice {rig['dice']:.3f} | bent tumour {ben['tumour_mm']:4.1f}, "
          f"contact {ben['contact_mm']:4.1f}, outline {ben['A']:4.1f}, landmarks {ben['landmark_px']:5.1f}, Dice {ben['dice']:.3f}, "
          f"moved {ben['moved_mean_mm']:.1f}/{ben['moved_max_mm']:.0f} mm  [{res['seconds']} s]", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--lam", type=float, default=1.0)
    a = ap.parse_args()
    align = json.load(open(f"{RES}/align.json"))
    images = a.only or le.images()
    with Pool(min(a.workers, len(images))) as pool:
        pool.map(run, [(i, align, a.lam) for i in images], chunksize=1)
    if not a.only:
        for variant in ["rigid", "bent"]:
            print(f"\n== official LLR-LUS score, oracle {variant} ==", flush=True)
            subprocess.run([sys.executable, "studies/benchmark/lus_evaluation.py", "--method-dir", f"{RES}/oracle_{variant}"], check=False)


if __name__ == "__main__":
    main()
