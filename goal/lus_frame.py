"""
The whole pipeline on ONE LLR-LUS image, scored by the real tumour (ultrasound profile, camera frame, mm).

1. Prepare: undistort the photo (LLR-LUS lens coefficients) and the hand-drawn liver mask; depth SHAPE from the
   Self-P2IR-format cloud of this image (Depth Anything; distance ignored); align the CT liver to body axes
   (x left, y back, z head) by fitting it to the 4 P2I-LReg livers, so the 6 parts are defined (show_classes.py).
2. VLM answer (written by hand into goal/vlm/<image>.json): alternatives (main part + head direction) and false-edge
   boxes (probe, ligament). A box hides only the NON-liver pixels inside it (the probe, the ligament), where the CT may
   continue without penalty; the dark scope border is hidden too.
3. Rigid registration from each alternative (+-30 deg / +-45 deg spin around its class view) and from the no-hint
   all-around search, same goal as before (J = A/4.5 + B/0.02); the lowest J wins.
4. Bending (deform_frame.py's cage, outline + spill pulls) from the winning pose: stiff and soft.
5. Tumour error for EVERY result (also the losers), so we see whether the goal's choice is the most accurate one:
   mean distance from the ultrasound tumour profile to the nearest point of the placed CT tumour (as
   lus_selfp2ir_score.py), split into depth (along the view) and across.
Usage: python goal/lus_frame.py --image Patient1/02
"""
import argparse
import itertools
import json
import re
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np
import open3d as o3d

sys.path.insert(0, "goal"); sys.path.insert(0, "studies/benchmark")
import show_classes as sc  # noqa: E402
import real_frame_reg as rf  # noqa: E402
import deform_frame as dfm  # noqa: E402
from goal_landscape import look_at, normals, render  # noqa: E402
import lus_evaluation as ev  # noqa: E402

LUS = "dataset/lus-evaluation"
DSET = f"{LUS}/data/Dataset"
W, H = rf.W, rf.H


# ---------------- 1. preparation ----------------
def body_axes(p):
    """Rotation + centre putting the LLR-LUS liver into body axes (fit to the P2I-LReg livers, 24 starts each)."""
    V = np.asarray(o3d.io.read_triangle_mesh(f"{DSET}/{p}/Preoperative Model/Liver.ply").vertices)
    c = V.mean(0)

    def pcd(v, n=6000):
        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(v)).farthest_point_down_sample(min(n, len(v)))
    best = None
    for q in ["01", "06", "10", "11"]:
        tv = np.asarray(o3d.io.read_triangle_mesh(f"dataset/liver-registration/{q}/model/reconstructed_mesh_world_m.obj").vertices) * 1000
        tv = tv - tv.mean(0)
        k = np.sqrt((tv ** 2).sum(1).mean() / ((V - c) ** 2).sum(1).mean())
        src, tgt = pcd((V - c) * k), pcd(tv)
        for perm in itertools.permutations(range(3)):
            for s in itertools.product([1, -1], repeat=3):
                M = np.zeros((3, 3))
                for i, j in enumerate(perm):
                    M[i, j] = s[i]
                if np.linalg.det(M) < 0:
                    continue
                T0 = np.eye(4); T0[:3, :3] = M
                r = o3d.pipelines.registration.registration_icp(src, tgt, 25.0, T0, o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                                 o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80))
                mv = o3d.geometry.PointCloud(src).transform(r.transformation)
                d = 0.5 * (np.asarray(mv.compute_point_cloud_distance(tgt)).mean() + np.asarray(tgt.compute_point_cloud_distance(mv)).mean())
                if best is None or d < best[0]:
                    best = (d, q, r.transformation[:3, :3].copy())
    return best[2], c, best[0], best[1]


def load_ct(p, Ralign, c):
    m = o3d.io.read_triangle_mesh(f"{DSET}/{p}/Preoperative Model/Liver.ply")
    m.remove_duplicated_vertices(); m.remove_degenerate_triangles(); m.remove_unreferenced_vertices()   # weld seams (geometry unchanged)
    m.vertices = o3d.utility.Vector3dVector((np.asarray(m.vertices) - c) @ Ralign.T)
    m.compute_triangle_normals(); m.compute_vertex_normals()
    V, T = np.asarray(m.vertices), np.asarray(m.triangles)
    Nt = np.asarray(m.triangle_normals); C = V[T].mean(1)
    if ((C - V.mean(0)) * Nt).sum(1).mean() < 0:
        Nt = -Nt
    tum = np.asarray(o3d.io.read_triangle_mesh(f"{DSET}/{p}/Preoperative Model/Tumour.ply").vertices)
    return m, V, T, C, Nt, (tum - c) @ Ralign.T


class LusFrame:
    def __init__(self, image, vlm):
        p, name = image.split("/")
        xml = open(f"{DSET}/{p}/Lap Images/CameraParameters").read()
        g = lambda k: float(re.search(f"<{k}>(.*?)</{k}>", xml).group(1))
        Kf = np.array([[g("fx"), 0, g("cx")], [0, g("fy"), g("cy")], [0, 0, 1]])
        dist = np.array([g("k1"), g("k2"), g("p1"), g("p2"), g("k3")])
        img = cv2.imread(f"{DSET}/{p}/Lap Images/{name}.png")
        Hf, Wf = img.shape[:2]
        mx, my = cv2.initUndistortRectifyMap(Kf, dist, None, Kf, (Wf, Hf), cv2.CV_32FC1)
        self.img = cv2.remap(img, mx, my, cv2.INTER_LINEAR)
        mask = cv2.imread(f"{LUS}/masks_manual/{p}/{name}.png", 0) > 0
        Mf = cv2.remap(mask.astype(np.uint8), mx, my, cv2.INTER_NEAREST) > 0
        # depth shape from the Self-P2IR-format cloud of this image (background pixels sit at the camera point)
        meta = json.load(open(f"{LUS}/selfp2ir/meta.json"))[image]
        P = np.asarray(o3d.io.read_point_cloud(f"{LUS}/selfp2ir/{meta['id']}/real/liverPcds/{meta['frame']}.ply").points).reshape(Hf, Wf, 3)
        zc = np.median(P[~mask][:, 2])
        Dd = np.where(mask, P[..., 2] - zc, 0).astype(np.float32)
        Du = cv2.remap(Dd, mx, my, cv2.INTER_NEAREST)
        s = W / Wf
        self.K = np.diag([s, s, 1.0]) @ Kf
        self.Kfull = Kf
        self.M = cv2.resize(Mf.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        D = cv2.resize(Du, (W, H), interpolation=cv2.INTER_NEAREST)
        self.N, self.ok = normals(np.where(self.M, D, 0), self.K)
        self.dtM = cv2.distanceTransform((~self.M).astype(np.uint8), cv2.DIST_L2, 5)
        # hidden (not tissue we can see): dark scope border + non-liver pixels inside the VLM's false-edge boxes
        grey = cv2.cvtColor(cv2.resize(self.img, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        n, cc = cv2.connectedComponents((grey < 25).astype(np.uint8))
        edge = np.unique(np.concatenate([cc[0], cc[-1], cc[:, 0], cc[:, -1]]))
        hidden = np.isin(cc, edge[edge > 0])
        for box in vlm.get("false_edges", {}).values():
            if isinstance(box, list):
                x0, y0, x1, y1 = (np.array(box) * s).astype(int)
                bx = np.zeros_like(hidden); bx[y0:y1, x0:x1] = True
                hidden |= bx & ~self.M
        hidden = cv2.dilate(hidden.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        self.V = ~hidden | self.M
        self.profile = ev.profile(DSET, p[len("Patient"):], name).T             # n x 3, camera frame, mm


# ---------------- 3. rigid registration ----------------
_G = {}


def _init(image, vlm_path):
    p = image.split("/")[0]
    vlm = json.load(open(vlm_path))
    Ralign, c, fit, q = body_axes(p)
    m, V, T, C, Nt, tum = load_ct(p, Ralign, c)
    lab = sc.parts(V, C, Nt, T, m)
    scene = o3d.t.geometry.RaycastingScene(); scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
    _G.update(m=m, V=V, T=T, C=C, Nt=Nt, tum=tum, lab=lab, scene=scene, fr=LusFrame(image, vlm), vlm=vlm, fit=fit, q=q)


def run_alt(job):
    image, vlm_path, alt = job
    if not _G:
        _init(image, vlm_path)
    g = _G; fr, scene, V = g["fr"], g["scene"], g["V"]
    centre = V.mean(0)
    t0 = time.time()
    starts = []
    if alt == "no hint":
        for d in rf.fib(256):
            for roll in np.arange(0, 360, 22.5):
                starts.append(look_at(centre, d, 180.0, roll))
    else:
        part, head = alt
        R0, _, target = sc.class_pose(V, g["T"], g["C"], g["Nt"], g["lab"], part, rf.HEADS[head])
        for d in rf.fib(400):
            if np.degrees(np.arccos(np.clip(d @ -R0[2], -1, 1))) <= 30:
                best, bang = 0, 1e9
                for r_ in np.arange(0, 360, 5):
                    Rr, _ = look_at(target, d, 150.0, r_)
                    a_ = np.degrees(np.arccos(np.clip(Rr[1] @ R0[1], -1, 1)))
                    if a_ < bang:
                        best, bang = r_, a_
                for spin in np.arange(-45, 46, 15):
                    starts.append(look_at(target, d, 150.0, best + spin))
    scored = []
    for R, t in starts:
        R, t = rf.size_and_centre(scene, centre, R, t, fr)
        scored.append((rf.J(*rf.terms(scene, R, t, fr)), R, t))
    scored.sort(key=lambda s: s[0])
    best = None
    for _, R, t in scored[:8]:
        Rf, tf, fv = rf.polish(scene, R, t, centre, fr)
        if best is None or fv < best[2]:
            best = (Rf, tf, fv)
    Rf, tf, fv = best
    A, B = rf.terms(scene, Rf, tf, fr)
    return dict(alt=alt if alt == "no hint" else f"{sc.PARTS[alt[0]]}, head {alt[1]}", J=float(fv), A=float(A), B=float(B),
                R=Rf.tolist(), t=tf.tolist(), seconds=round(time.time() - t0), **tumour_error(g["tum"], Rf, tf, fr.profile),
                distance_mm=float(np.linalg.norm(Rf @ centre + tf)))


def tumour_error(tum_body, R, t, prof):
    cam = tum_body @ R.T + t
    near = cam[((prof[:, None] - cam[None]) ** 2).sum(2).argmin(1)]
    diff = near - prof
    return dict(tumour_mm=float(np.linalg.norm(diff, axis=1).mean()), depth_mm=float(diff[:, 2].mean()),
                across_mm=float(np.linalg.norm(diff[:, :2], axis=1).mean()), true_depth_mm=float(prof[:, 2].mean()),
                pred_depth_mm=float(cam[:, 2].mean()))


# ---------------- 4. bending ----------------
def bend(R, t, smooth):
    g = _G; fr, V, T = g["fr"], g["V"], g["T"]
    nodes, Wv, pairs = dfm.cage(V)
    lo = V.min(0); hi = V.max(0); pad = 0.08 * (hi - lo); lo, hi = lo - pad, hi + pad
    n = len(nodes); U = np.zeros((n, 3))
    border = np.zeros((H, W), bool); border[:2] = border[-2:] = True; border[:, :2] = border[:, -2:] = True
    hidden = cv2.dilate((~fr.V).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    Mtrue = dfm.edges(fr.M) & ~hidden & ~border
    from scipy.ndimage import distance_transform_edt
    _, (iy, ix) = distance_transform_edt(~Mtrue, return_indices=True)
    Dm = np.zeros((len(pairs) * 3, n * 3))
    for k, (a, b) in enumerate(pairs):
        for c_ in range(3):
            Dm[k * 3 + c_, a * 3 + c_] = 1; Dm[k * 3 + c_, b * 3 + c_] = -1
    for it in range(dfm.ITERS):
        Vd = V + Wv @ U
        S, Z, prim = dfm.cast(Vd, T, R, t, fr.K)
        Sedge = dfm.edges(S) & fr.V & ~border
        _, (sy, sx) = distance_transform_edt(~Sedge, return_indices=True)
        src = [(y, x, iy[y, x], ix[y, x]) for y, x in np.argwhere(Sedge)]
        src += [(sy[y, x], sx[y, x], y, x) for y, x in np.argwhere(Mtrue)]
        spill = np.argwhere(S & fr.V & ~fr.M & ~border); spill = spill[::max(1, len(spill) // 1500)]
        src += [(y, x, iy[y, x], ix[y, x]) for y, x in spill]
        src = np.array(src); src = src[np.hypot(src[:, 2] - src[:, 0], src[:, 3] - src[:, 1]) < 60]
        rows, b = [], []
        for y, x, ty, tx in src:
            tri = prim[y, x]
            if tri == o3d.t.geometry.RaycastingScene.INVALID_ID:
                continue
            z = Z[y, x]
            rows.append(T[tri][0]); b.append(R.T @ np.array([(tx - x) * z / fr.K[0, 0], (ty - y) * z / fr.K[1, 1], 0.0]))
        Ws = Wv[np.array(rows)].toarray(); b = np.array(b); npts = len(b)
        lhs = np.vstack([np.kron(Ws, np.eye(3)) / np.sqrt(npts), np.sqrt(smooth) * Dm / np.sqrt(len(pairs)), np.sqrt(dfm.SMALL) * np.eye(n * 3) / np.sqrt(n)])
        cur = U.reshape(-1)
        rhs = np.concatenate([b.reshape(-1) / np.sqrt(npts), -np.sqrt(smooth) * Dm @ cur / np.sqrt(len(pairs)), -np.sqrt(dfm.SMALL) * cur / np.sqrt(n)])
        U = U + dfm.STEP * np.linalg.lstsq(lhs, rhs, rcond=None)[0].reshape(n, 3)
    # move the tumour with the same cage (trilinear weights of its points)
    def weights(P):
        gr = (P - lo) / (hi - lo) * (np.array(dfm.GRID) - 1)
        i0 = np.clip(np.floor(gr).astype(int), 0, np.array(dfm.GRID) - 2); f = gr - i0
        out = np.zeros((len(P), 3))
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                    idx = ((i0[:, 0] + dx) * dfm.GRID[1] + (i0[:, 1] + dy)) * dfm.GRID[2] + (i0[:, 2] + dz)
                    out += w[:, None] * U[idx]
        return out
    Vd = V + Wv @ U
    tum_d = g["tum"] + weights(g["tum"])
    S, Z, _ = dfm.cast(Vd, T, R, t, fr.K)
    A, B, dice = dfm.scores(S, Z, fr)
    mv = np.linalg.norm(Wv @ U, axis=1)
    return Vd, tum_d, dict(outline_px=float(A), shape=float(B), dice=float(dice), moved_mean_mm=float(mv.mean()),
                           moved_max_mm=float(mv.max()), tumour_moved_mm=float(np.linalg.norm(tum_d - g["tum"], axis=1).mean()),
                           **tumour_error(tum_d, R, t, fr.profile))


def picture(image, res, bends, out):
    g = _G; fr = g["fr"]
    Wb, Hb = 960, 540
    Kb = np.diag([2.0, 2.0, 1.0]) @ fr.K
    photo = cv2.resize(fr.img, (Wb, Hb), interpolation=cv2.INTER_AREA)
    Mb = cv2.resize(fr.M.astype(np.uint8), (Wb, Hb), interpolation=cv2.INTER_NEAREST)
    mc, _ = cv2.findContours(Mb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    prof = fr.profile
    pu = (Kb @ prof.T).T; pu = pu[:, :2] / pu[:, 2:]
    tiles = []
    items = [(r["alt"], g["V"], g["tum"], np.array(r["R"]), np.array(r["t"]), r) for r in res] + \
            [(f"winner + bending ({k})", Vd, tum, np.array(res[0]["R"]), np.array(res[0]["t"]), s) for k, (Vd, tum, s) in bends.items()]
    for name, Vx, tum, R, t, s in items:
        S, _, _ = dfm.cast(Vx, g["T"], R, t, Kb) if False else (None, None, None)
        m = o3d.t.geometry.TriangleMesh(); m.vertex.positions = o3d.core.Tensor(Vx.astype(np.float32)); m.triangle.indices = o3d.core.Tensor(g["T"].astype(np.int32))
        sc_ = o3d.t.geometry.RaycastingScene(); sc_.add_triangles(m)
        S, _ = render(sc_, R, t, Kb, Wb, Hb)
        im = photo.copy()
        cc, _ = cv2.findContours(S.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(im, mc, -1, (0, 255, 255), 2); cv2.drawContours(im, cc, -1, (255, 255, 0), 2)
        tc = (Kb @ (tum @ R.T + t).T).T; tc = tc[:, :2] / tc[:, 2:]
        for x, y in tc[::max(1, len(tc) // 400)].astype(int):
            cv2.circle(im, (x, y), 2, (60, 60, 255), -1)
        for x, y in pu.astype(int):
            cv2.circle(im, (x, y), 3, (60, 220, 60), -1)
        lines = [name, f"tumour error {s['tumour_mm']:.1f} mm (depth {s['depth_mm']:+.1f}, across {s['across_mm']:.1f})"]
        if "J" in s:
            lines.append(f"goal score J {s['J']:.2f} (outline {s['A']:.1f} px, shape {s['B']:.3f})")
        else:
            lines.append(f"outline {s['outline_px']:.1f} px, liver moved {s['moved_mean_mm']:.1f} / max {s['moved_max_mm']:.1f} mm")
        for i, l in enumerate(lines):
            cv2.putText(im, l, (12, 30 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(im, l, (12, 30 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(im)
    while len(tiles) % 2:
        tiles.append(np.full_like(tiles[0], 255))
    cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="Patient1/02")
    a = ap.parse_args()
    tag = a.image.replace("/", "_")
    vlm_path = f"goal/vlm/{tag}.json"
    vlm = json.load(open(vlm_path))
    alts = [(sc.PARTS.index(x["main"]), x["head"]) for x in vlm["alternatives"]]
    jobs = [(a.image, vlm_path, x) for x in alts] + [(a.image, vlm_path, "no hint")]
    with Pool(len(jobs)) as pool:
        res = pool.map(run_alt, jobs)
    _init(a.image, vlm_path)
    print(f"CT aligned to body axes via P2I-LReg liver {_G['q']} (shape fit {_G['fit']:.1f} mm)")
    res.sort(key=lambda r: r["J"])
    for r in res:
        print(f"{r['alt']:40s} J {r['J']:.2f} (outline {r['A']:4.1f} px, shape {r['B']:.3f}) | tumour {r['tumour_mm']:5.1f} mm "
              f"(depth {r['depth_mm']:+6.1f}, across {r['across_mm']:4.1f}); CT tumour depth {r['pred_depth_mm']:.0f} vs true {r['true_depth_mm']:.0f} mm")
    win = res[0]
    bends = {}
    for name, sm in [("stiff", 4.0), ("soft", 0.5)]:
        Vd, tum_d, s = bend(np.array(win["R"]), np.array(win["t"]), sm)
        bends[name] = (Vd, tum_d, s)
        print(f"winner + bending ({name}): outline {s['outline_px']:.1f} px, Dice {s['dice']:.3f}, shape {s['shape']:.3f}, liver moved "
              f"{s['moved_mean_mm']:.1f} / max {s['moved_max_mm']:.1f} mm, tumour moved {s['tumour_moved_mm']:.1f} mm | tumour error {s['tumour_mm']:.1f} mm "
              f"(depth {s['depth_mm']:+.1f}, across {s['across_mm']:.1f})")
    json.dump(dict(rigid=res, bending={k: v[2] for k, v in bends.items()}), open(f"goal/out/lus_{tag}.json", "w"), indent=1)
    picture(a.image, res, bends, f"goal/out/lus_{tag}.png")
    print(f"wrote goal/out/lus_{tag}.png  (yellow = mask, cyan = CT outline, red dots = placed CT tumour, green = ultrasound tumour)")


if __name__ == "__main__":
    main()
