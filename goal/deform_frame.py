"""
Simple deformation after the rigid registration of one real P2I-LReg photo (first look, no ground truth).

The CT keeps the rigid pose found by real_frame_reg.py and may bend through a coarse cage (free-form deformation:
5 x 5 x 4 control points around the liver, trilinear weights), so the INSIDE moves with the surface.
Each iteration: render the bent CT, match outline points both ways to the nearest TRUE liver edge in the photo (the
scope rim, dark corners and image border are not edges), convert each 2D gap to a 3D move in the image plane at that
point's depth, then solve for the cage moves by least squares with penalties on uneven moves between neighbouring
control points (smoothness) and on move size. CT surface spilling past a true edge is pulled to the nearest true edge. Only the OUTLINE drives the fit; the SHAPE term (surface directions vs
the depth map) is measured before/after as an independent check.
Usage: python goal/deform_frame.py --patient 01 --frame 10 --pose all-around [--smooth 4 --tag _stiff]
"""
import argparse
import json
import sys

import cv2
import numpy as np
import open3d as o3d
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, "goal")
import show_classes as sc  # noqa: E402
from real_frame_reg import DOWN, H, W, Frame  # noqa: E402
from goal_landscape import normals  # noqa: E402

GRID = (5, 5, 4)
SMOOTH, SMALL, STEP, ITERS = 4.0, 0.05, 0.7, 25


def cage(V):
    lo, hi = V.min(0), V.max(0)
    pad = 0.08 * (hi - lo); lo, hi = lo - pad, hi + pad
    axes = [np.linspace(lo[i], hi[i], GRID[i]) for i in range(3)]
    nodes = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    # trilinear weights of every vertex on the 8 surrounding nodes
    g = (V - lo) / (hi - lo) * (np.array(GRID) - 1)
    i0 = np.clip(np.floor(g).astype(int), 0, np.array(GRID) - 2); f = g - i0
    rows, cols, vals = [], [], []
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                idx = ((i0[:, 0] + dx) * GRID[1] + (i0[:, 1] + dy)) * GRID[2] + (i0[:, 2] + dz)
                rows.append(np.arange(len(V))); cols.append(idx); vals.append(w)
    from scipy.sparse import coo_matrix
    Wv = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(V), len(nodes))).tocsr()
    # neighbour pairs on the cage (for smoothness)
    pairs = []
    for a in range(len(nodes)):
        ia = np.unravel_index(a, GRID)
        for d in range(3):
            ib = list(ia); ib[d] += 1
            if ib[d] < GRID[d]:
                pairs.append((a, np.ravel_multi_index(ib, GRID)))
    return nodes, Wv, np.array(pairs)


def cast(Vd, T, R, t, K):
    m = o3d.t.geometry.TriangleMesh()
    m.vertex.positions = o3d.core.Tensor(Vd.astype(np.float32)); m.triangle.indices = o3d.core.Tensor(T.astype(np.int32))
    scene = o3d.t.geometry.RaycastingScene(); scene.add_triangles(m)
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1).reshape(-1, 3)
    dn = d / np.linalg.norm(d, axis=1, keepdims=True)
    rays = np.hstack([np.repeat((-R.T @ t)[None], len(d), 0), dn @ R]).astype(np.float32)
    o = scene.cast_rays(o3d.core.Tensor(rays))
    th = o["t_hit"].numpy(); hit = np.isfinite(th)
    Z = np.where(hit, th * dn[:, 2], 0).reshape(H, W)
    return hit.reshape(H, W), Z, o["primitive_ids"].numpy().reshape(H, W)


def edges(mask):
    return mask & ~(cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)


def scores(S, Z, fr):
    Sv = S & fr.V
    dtS = cv2.distanceTransform((~S).astype(np.uint8), cv2.DIST_L2, 5)
    A = (dtS[fr.M].mean() + (fr.dtM[Sv].mean() if Sv.any() else 0)) * DOWN
    Ns, oks = normals(Z, fr.K)
    both = fr.M & S & fr.ok & oks
    B = float((1 - (Ns[both] * fr.N[both]).sum(-1)).mean())
    inter = (S & fr.M).sum()
    return A, B, 2 * inter / (S.sum() + fr.M.sum())


def main():
    global SMOOTH
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default="01"); ap.add_argument("--frame", default="10"); ap.add_argument("--pose", default="all-around")
    ap.add_argument("--smooth", type=float, default=4.0); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    SMOOTH = a.smooth
    r = [x for x in json.load(open(f"goal/out/real_{a.patient}_{a.frame}.json")) if x["alt"] == a.pose][0]
    R, t = np.array(r["R"]), np.array(r["t"])
    m, V, T, C, Nt = sc.load(a.patient)
    fr = Frame(a.patient, a.frame)
    K = fr.K
    nodes, Wv, pairs = cage(V)
    n = len(nodes)
    U = np.zeros((n, 3))
    # true edges of the photo's mask: not along the scope rim / dark corners / image border
    border = np.zeros((H, W), bool); border[:2] = border[-2:] = True; border[:, :2] = border[:, -2:] = True
    hidden = cv2.dilate((~fr.V).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    Mtrue = edges(fr.M) & ~hidden & ~border
    _, (iy, ix) = distance_transform_edt(~Mtrue, return_indices=True)
    log = []
    D = np.zeros((len(pairs) * 3, n * 3))
    for k, (p, q) in enumerate(pairs):
        for c in range(3):
            D[k * 3 + c, p * 3 + c] = 1; D[k * 3 + c, q * 3 + c] = -1
    Wd = Wv.toarray() if Wv.shape[0] * Wv.shape[1] < 2e7 else None
    for it in range(ITERS + 1):
        Vd = V + Wv @ U
        S, Z, prim = cast(Vd, T, R, t, K)
        A, B, dice = scores(S, Z, fr)
        disp = np.linalg.norm(Wv @ U, axis=1)
        log.append(dict(iteration=it, outline_px=float(A), shape=float(B), dice=float(dice), move_mean_mm=float(disp.mean()), move_max_mm=float(disp.max())))
        print(f"iter {it:2d}: outline gap {A:5.2f} px, shape mismatch {B:.4f}, Dice {dice:.3f}, tissue moved mean {disp.mean():4.1f} mm / max {disp.max():4.1f} mm", flush=True)
        if it == ITERS:
            break
        # outline correspondences, both ways
        Sedge = edges(S) & fr.V & ~border
        _, (sy, sx) = distance_transform_edt(~Sedge, return_indices=True)
        src = []                                                     # (pixel on the CT outline, target pixel)
        for y, x in np.argwhere(Sedge):
            src.append((y, x, iy[y, x], ix[y, x]))
        for y, x in np.argwhere(Mtrue):
            src.append((sy[y, x], sx[y, x], y, x))
        # CT surface spilling past a TRUE liver edge (visible, outside the mask): pull it to the nearest true edge
        spill = np.argwhere(S & fr.V & ~fr.M & ~border)
        spill = spill[::max(1, len(spill) // 1500)]
        for y, x in spill:
            src.append((y, x, iy[y, x], ix[y, x]))
        src = np.array(src)
        gap = np.hypot(src[:, 2] - src[:, 0], src[:, 3] - src[:, 1])
        src = src[gap < 60]                                          # ignore implausible matches (> 160 px at full size)
        rows, b = [], []
        for y, x, ty, tx in src:
            tri = prim[y, x]
            if tri == o3d.t.geometry.RaycastingScene.INVALID_ID:
                continue
            z = Z[y, x]
            dcam = np.array([(tx - x) * z / K[0, 0], (ty - y) * z / K[1, 1], 0.0])   # move in the image plane, mm
            dworld = R.T @ dcam
            vtx = T[tri][np.argmin(np.linalg.norm(Vd[T[tri]] - (Vd[T[tri]].mean(0)), axis=1))]
            rows.append(vtx); b.append(dworld)
        if not rows:
            break
        Ws = Wv[np.array(rows)].toarray()                           # (points x nodes)
        b = np.array(b)
        npts = len(b)
        # least squares over the cage moves dU: data + smoothness + size (per coordinate, same matrices)
        Adata = np.kron(Ws, np.eye(3)) / np.sqrt(npts)
        reg_s = np.sqrt(SMOOTH) * D / np.sqrt(len(pairs))
        reg_0 = np.sqrt(SMALL) * np.eye(n * 3) / np.sqrt(n)
        lhs = np.vstack([Adata, reg_s, reg_0])
        cur = U.reshape(-1)
        rhs = np.concatenate([b.reshape(-1) / np.sqrt(npts), -reg_s @ cur, -reg_0 @ cur])
        dU = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        U = U + STEP * dU.reshape(n, 3)
    Vd = V + Wv @ U
    vol0 = o3d.geometry.TriangleMesh(m).get_volume() if m.is_watertight() else None
    md = o3d.geometry.TriangleMesh(m); md.vertices = o3d.utility.Vector3dVector(Vd)
    vol1 = md.get_volume() if md.is_watertight() else None
    vol = None if vol0 is None else (vol1 / vol0 - 1)
    print("volume change:", "n/a (mesh not watertight)" if vol is None else f"{vol:+.1%}")
    json.dump(dict(log=log, volume_change=vol, cage=GRID, smooth=SMOOTH, small=SMALL), open(f"goal/out/deform_{a.patient}_{a.frame}{a.tag}.json", "w"), indent=1)
    np.save(f"goal/out/deform_{a.patient}_{a.frame}{a.tag}_vertices.npy", Vd.astype(np.float32))

    # pictures: outlines before/after on the photo, and how far each part of the liver moved
    Wb, Hb = 960, 540
    photo = cv2.resize(fr.img, (Wb, Hb), interpolation=cv2.INTER_AREA)
    Mb = cv2.resize(fr.M.astype(np.uint8), (Wb, Hb), interpolation=cv2.INTER_NEAREST)
    mc, _ = cv2.findContours(Mb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    tiles = []
    for name, Vx, row in [("RIGID (before bending)", V, log[0]), ("BENT", Vd, log[-1])]:
        S0, _, _ = cast(Vx, T, R, t, K)
        Sb = cv2.resize(S0.astype(np.uint8), (Wb, Hb), interpolation=cv2.INTER_NEAREST)
        cc, _ = cv2.findContours(Sb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        im = photo.copy(); cv2.drawContours(im, mc, -1, (0, 255, 255), 2); cv2.drawContours(im, cc, -1, (255, 255, 0), 2)
        for i, s in enumerate([name, f"outline gap {row['outline_px']:.1f} px, Dice {row['dice']:.3f}, shape {row['shape']:.4f}",
                               "yellow = liver in photo, cyan = CT outline"]):
            cv2.putText(im, s, (12, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(im, s, (12, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(im)
    # displacement map seen through the camera
    S1, Z1, prim1 = cast(Vd, T, R, t, K)
    disp = np.linalg.norm(Vd - V, axis=1)
    dm = np.zeros((H, W)); dm[S1] = disp[T[prim1[S1], 0]]
    vmax = max(5.0, np.percentile(disp, 99))
    heat = cv2.applyColorMap((np.clip(dm / vmax, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    heat[~S1] = 245
    heat = cv2.resize(heat, (Wb, Hb), interpolation=cv2.INTER_NEAREST)
    for i, s in enumerate(["how far the tissue moved (bent CT, same camera)", f"dark = 0 mm, yellow = {vmax:.0f} mm or more",
                           f"mean {log[-1]['move_mean_mm']:.1f} mm, max {log[-1]['move_max_mm']:.1f} mm"]):
        cv2.putText(heat, s, (12, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.imwrite(f"goal/out/deform_{a.patient}_{a.frame}{a.tag}.png", np.hstack(tiles + [heat]))
    print(f"wrote goal/out/deform_{a.patient}_{a.frame}.png")


if __name__ == "__main__":
    main()
