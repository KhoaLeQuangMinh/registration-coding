"""
See the 48 pose classes (6 visible parts x 8 head directions) on a P2I-LReg liver.

CT axes (checked 2026-11-17 on all 4 local patients): x = patient LEFT (the thick right lobe is at -x), y = BACK (front is
-y), z = HEAD (the hollow, visceral surface faces -z; the dome +z is smooth).
The 6 parts are a ROUGH geometric stand-in for the anatomy (no landmarks are marked on these CTs yet):
    lobes by left-right position: right lobe = the 60 % of the liver furthest right, segment 4 = the next 20 %,
    left lateral section = the 20 % furthest left (the falciform ligament sits roughly at that last cut);
    surfaces by which way they face: dome = facing up (head), underside = facing down (feet),
    right lobe side/back = right lobe facing back or right; everything else facing front = the 3 "front" parts.
Each class = look at one part head-on (from 150 mm along its average outward direction), rolled so the patient's HEAD
appears in one of 8 compass directions in the image (for the dome, where the head axis points at the camera, the
patient's FRONT is used instead).

Usage (repo root):
    python goal/show_classes.py                 interactive Open3D window
        N / P  next / previous part        R / E  head direction clockwise / anticlockwise
        V      look through the class camera (what that class sees)      O  back to the overview
    python goal/show_classes.py --sheet         also write goal/out/classes_sheet.png (all 48 views)
    --patient 06                                other patient (01, 06, 10, 11)
"""
import argparse
import os

import cv2
import numpy as np
import open3d as o3d
import yaml

PARTS = ["left lateral section, front", "segment 4, front", "right lobe, front", "back and right side",
         "underside", "dome"]
COLOURS = np.array([[42, 120, 214], [235, 104, 52], [27, 175, 122], [237, 161, 0], [232, 123, 164], [98, 80, 214]]) / 255.0
GREY = np.array([0.72, 0.72, 0.70])
HEAD = ["up", "up-right", "right", "down-right", "down", "down-left", "left", "up-left"]
DIST = 150.0
W, H = 480, 270


def load(p):
    m = o3d.io.read_triangle_mesh(f"dataset/liver-registration/{p}/model/reconstructed_mesh_world_m.obj")
    m.vertices = o3d.utility.Vector3dVector(np.asarray(m.vertices) * 1000.0)
    m.compute_triangle_normals(); m.compute_vertex_normals()
    V, T = np.asarray(m.vertices), np.asarray(m.triangles)
    Nt = np.asarray(m.triangle_normals)
    C = V[T].mean(1)
    if ((C - V.mean(0)) * Nt).sum(1).mean() < 0:     # outward normals
        Nt = -Nt
    return m, V, T, C, Nt


def parts(V, C, Nt, T=None, m=None):
    """6 clean regions covering the whole liver (per triangle). Facing is read from a heavily smoothed copy of the
    liver so bumps do not flip a region, then specks are removed by a neighbour majority vote."""
    if m is not None:
        sm = o3d.geometry.TriangleMesh(m).filter_smooth_taubin(number_of_iterations=300)
        sm.compute_triangle_normals()
        Ns = np.asarray(sm.triangle_normals)
        if ((C - V.mean(0)) * Ns).sum(1).mean() < 0:
            Ns = -Ns
        lab = _majority(_parts_from(V, C, Ns), T, iters=15)
        return _solid_patches(lab, V, T, C, Ns)
    return _parts_from(V, C, Nt)


def _solid_patches(lab, V, T, C, Ns, w=1.5):
    """Turn rough labels into 6 solid patches: one centre per part (area-weighted position + facing of its rough
    region), every triangle goes to the nearest centre (position / liver size + w x facing difference), then any
    island (a piece not connected to the part's main patch) joins the part that surrounds it."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    area = np.linalg.norm(np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]), axis=1)
    size = np.ptp(V, axis=0).max()
    cen, nrm = [], []
    for k in range(6):
        s_ = lab == k
        cen.append((C[s_] * area[s_, None]).sum(0) / area[s_].sum())
        n = (Ns[s_] * area[s_, None]).sum(0); nrm.append(n / np.linalg.norm(n))
    cen, nrm = np.array(cen), np.array(nrm)
    d = np.linalg.norm(C[:, None] - cen[None], axis=2) / size + w * (1 - Ns @ nrm.T)
    lab = d.argmin(1)
    # triangle adjacency through shared edges
    e = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
    tid = np.tile(np.arange(len(T)), 3)
    key = e[:, 0].astype(np.int64) * (T.max() + 1) + e[:, 1]
    order = np.argsort(key); key, tid = key[order], tid[order]
    same = key[1:] == key[:-1]
    a, b = tid[:-1][same], tid[1:][same]
    for _ in range(60):
        changed = False
        for k in range(6):
            sel = np.flatnonzero(lab == k)
            if len(sel) == 0:
                continue
            idx = -np.ones(len(T), int); idx[sel] = np.arange(len(sel))
            keep = (idx[a] >= 0) & (idx[b] >= 0)
            g = coo_matrix((np.ones(keep.sum()), (idx[a][keep], idx[b][keep])), shape=(len(sel), len(sel)))
            nc, comp = connected_components(g, directed=False)
            if nc <= 1:
                continue
            main = np.bincount(comp, weights=area[sel]).argmax()
            for c_ in range(nc):
                if c_ == main:
                    continue
                piece = sel[comp == c_]
                inside = np.zeros(len(T), bool); inside[piece] = True
                border = np.concatenate([b[inside[a] & ~inside[b]], a[inside[b] & ~inside[a]]])
                if len(border):
                    lab[piece] = np.bincount(lab[border], minlength=6).argmax(); changed = True
        if not changed:
            break
    return lab


def _majority(lab, T, iters):
    """Replace each triangle's part by the most common part among triangles sharing its vertices."""
    nv = T.max() + 1
    for _ in range(iters):
        votes = np.zeros((nv, 6))
        for j in range(3):
            np.add.at(votes, (T[:, j], lab), 1)
        tri_votes = votes[T[:, 0]] + votes[T[:, 1]] + votes[T[:, 2]]
        lab = tri_votes.argmax(1)
    return lab


def _parts_from(V, C, Nt):
    x = C[:, 0]
    q60, q80 = np.quantile(V[:, 0], [0.6, 0.8])
    lobe = np.where(x < q60, 2, np.where(x < q80, 1, 0))          # 2 right lobe, 1 segment 4, 0 left lateral
    # each bit of surface goes to the direction it mostly faces
    dirs = np.array([[0, -1, 0],            # front
                     [-1, 0.5, 0],          # right side / back (patient right = -x, back = +y)
                     [0, 0.3, -1],          # underside (faces feet, a little backwards)
                     [0, 0, 1],             # dome (head)
                     [1, 0, 0],             # left side: rarely visible, no class
                     [0, 1, 0]], float)     # back: rarely visible, no class
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    facing = np.argmax(Nt @ dirs.T, axis=1)
    lab = np.full(len(C), -1)
    lab[facing == 0] = lobe[facing == 0]                            # 0, 1, 2: the three front parts
    lab[facing == 2] = 4                                            # underside
    lab[facing == 3] = 5                                            # dome
    back_or_right = (facing == 1) | (facing == 5)
    lab[back_or_right & (lobe == 2)] = 3                            # right lobe side / back
    upper = C[:, 2] > np.median(V[:, 2])
    lab[back_or_right & (lobe < 2)] = np.where(upper[back_or_right & (lobe < 2)], 5, 4)   # back of the left lobes: dome if high, else underside (faces the stomach)
    lab[facing == 4] = np.where(lobe[facing == 4] == 0, 0, 4)[...]  # left side: left lobe tip, else underside
    return lab


def class_pose(V, T, C, Nt, lab, part, head_idx):
    """World -> camera (vision axes) rotation R and translation t for one class."""
    area = np.linalg.norm(np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]), axis=1)
    sel = lab == part
    n = (Nt[sel] * area[sel, None]).sum(0); n /= np.linalg.norm(n)
    target = (C[sel] * area[sel, None]).sum(0) / area[sel].sum()
    f = -n                                                          # camera looks at the part head-on
    ref = np.array([0, 0, 1.0]) if part != 5 else np.array([0, -1.0, 0])   # head; for the dome: the front
    hp = ref - (ref @ f) * f; hp /= np.linalg.norm(hp)
    theta = np.radians(45 * head_idx)                               # clockwise from image-up
    best = None
    for sign in (1, -1):
        phi = sign * theta
        u = hp * np.cos(phi) + np.cross(f, hp) * np.sin(phi) + f * (f @ hp) * (1 - np.cos(phi))
        y = -u; z = f; xax = np.cross(y, z)
        R = np.stack([xax, y, z])
        c = R @ ref
        ang = np.degrees(np.arctan2(c[0], -c[1])) % 360
        err = min(abs(ang - 45 * head_idx), 360 - abs(ang - 45 * head_idx))
        if best is None or err < best[0]:
            best = (err, R)
    R = best[1]
    centre = target - DIST * f
    return R, -R @ centre, target


def camera_k(p):
    K = np.array(list(yaml.safe_load(open(f"dataset/liver-registration/{p}/real/CAM_K.yml")).values())[0][0]["cam_K"]).reshape(3, 3)
    return np.diag([W / 1920, H / 1080, 1.0]) @ K


def render_view(scene, Nt, lab, R, t, K):
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1).reshape(-1, 3)
    dn = d / np.linalg.norm(d, axis=1, keepdims=True)
    rays = np.hstack([np.repeat((-R.T @ t)[None], len(d), 0), dn @ R]).astype(np.float32)
    prim = scene.cast_rays(o3d.core.Tensor(rays))["primitive_ids"].numpy()
    hit = prim != o3d.t.geometry.RaycastingScene.INVALID_ID
    img = np.full((H * W, 3), 0.97)
    col = np.where((lab[prim[hit]] >= 0)[:, None], COLOURS[np.clip(lab[prim[hit]], 0, 5)], GREY)
    lam = np.abs((Nt[prim[hit]] * (dn[hit] @ R.T @ R)).sum(1))      # shading by facing angle
    img[hit] = col * (0.35 + 0.65 * lam[:, None])
    return (img.reshape(H, W, 3)[..., ::-1] * 255).astype(np.uint8)


def sheet(p, m, V, T, C, Nt, lab):
    scene = o3d.t.geometry.RaycastingScene(); scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
    K = camera_k(p)
    tw, th = W // 2, H // 2
    rows = []
    for part in range(6):
        tiles = []
        for h in range(8):
            R, t, _ = class_pose(V, T, C, Nt, lab, part, h)
            img = cv2.resize(render_view(scene, Nt, lab, R, t, K), (tw, th), interpolation=cv2.INTER_AREA)
            a = np.radians(45 * h)
            cv2.arrowedLine(img, (tw - 22, 22), (int(tw - 22 + 16 * np.sin(a)), int(22 - 16 * np.cos(a))), (30, 30, 30), 2, tipLength=0.4)
            cv2.putText(img, f"#{part * 8 + h + 1:02d}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1, cv2.LINE_AA)
            tiles.append(cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
        head = np.full((th + 2, 170, 3), 255, np.uint8)
        cv2.putText(head, PARTS[part], (4, th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.rectangle(head, (4, th // 2 + 8), (40, th // 2 + 18), tuple(int(c) for c in (COLOURS[part][::-1] * 255)), -1)
        rows.append(np.hstack([head] + tiles))
    top = np.full((24, rows[0].shape[1], 3), 255, np.uint8)
    for h in range(8):
        cv2.putText(top, f"head: {HEAD[h]}", (170 + h * (tw + 2) + 4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1, cv2.LINE_AA)
    out = np.vstack([top] + rows)
    os.makedirs("goal/out", exist_ok=True)
    cv2.imwrite(f"goal/out/classes_sheet_{p}.png", out)
    print(f"wrote goal/out/classes_sheet_{p}.png (arrow in each tile = where the patient's head is in that image; "
          f"for the dome row, where the patient's front is)")


def frustum(R, t, K, colour, size=45.0):
    Cw = -R.T @ t
    corners = [np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0]) * size for u, v in [(0, 0), (W, 0), (W, H), (0, H)]]
    P = [Cw] + [R.T @ c + Cw for c in corners]
    up = R.T @ np.array([0, -(K[1, 2] / K[1, 1]) * size * 1.35, size]) + Cw            # marker above the image's top edge
    P.append(up)
    L = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1], [1, 5], [2, 5]]
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(P), o3d.utility.Vector2iVector(L))
    ls.paint_uniform_color(colour)
    return ls


def interactive(p, m, V, T, C, Nt, lab):
    K = camera_k(p)
    tri_col = np.where((lab >= 0)[:, None], COLOURS[np.clip(lab, 0, 5)], GREY)
    vcol = np.zeros((len(V), 3)); cnt = np.zeros(len(V))
    for j in range(3):
        np.add.at(vcol, T[:, j], tri_col); np.add.at(cnt, T[:, j], 1)
    m.vertex_colors = o3d.utility.Vector3dVector(vcol / np.maximum(cnt, 1)[:, None])
    centre = V.mean(0)
    head_arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=2.5, cone_radius=5, cylinder_height=70, cone_height=18)
    head_arrow.paint_uniform_color([0.15, 0.15, 0.15]); head_arrow.translate(centre + np.array([0, 0, np.ptp(V[:, 2]) / 2 + 10]))
    six = [frustum(*class_pose(V, T, C, Nt, lab, k, 0)[:2], K, COLOURS[k] * 0.8, 30) for k in range(6)]
    state = dict(part=0, head=0, cur=None)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(f"48 pose classes, patient {p}", 1280, 800)
    vis.add_geometry(m)
    vis.add_geometry(head_arrow)
    for f in six:
        vis.add_geometry(f)

    def update(v, look=False):
        R, t, _ = class_pose(V, T, C, Nt, lab, state["part"], state["head"])
        if state["cur"] is not None:
            v.remove_geometry(state["cur"], reset_bounding_box=False)
        state["cur"] = frustum(R, t, K, [0.05, 0.05, 0.05], 55)
        v.add_geometry(state["cur"], reset_bounding_box=False)
        print(f"class #{state['part'] * 8 + state['head'] + 1:02d}: {PARTS[state['part']]}, head {HEAD[state['head']]}"
              + (" (dome: arrow = patient's front)" if state["part"] == 5 else ""), flush=True)
        if look:
            ctr = v.get_view_control()
            prm = ctr.convert_to_pinhole_camera_parameters()
            ext = np.eye(4); ext[:3, :3] = R; ext[:3, 3] = t
            prm.extrinsic = ext
            ww, hh = prm.intrinsic.width, prm.intrinsic.height
            f = K[0, 0] * ww / W
            prm.intrinsic.set_intrinsics(ww, hh, f, f, ww / 2 - 0.5, hh / 2 - 0.5)
            ctr.convert_from_pinhole_camera_parameters(prm, allow_arbitrary=True)
        return False

    def step(key):
        def cb(v):
            if key == "N": state["part"] = (state["part"] + 1) % 6
            if key == "P": state["part"] = (state["part"] - 1) % 6
            if key == "R": state["head"] = (state["head"] + 1) % 8
            if key == "E": state["head"] = (state["head"] - 1) % 8
            return update(v, look=state.get("looking", False))
        return cb

    def look(v):
        state["looking"] = True
        return update(v, look=True)

    def overview(v):
        state["looking"] = False
        v.reset_view_point(True)
        return False

    for k in "NPRE":
        vis.register_key_callback(ord(k), step(k))
    vis.register_key_callback(ord("V"), look)
    vis.register_key_callback(ord("O"), overview)
    print("colours:", ", ".join(f"{PARTS[i]}" for i in range(6)), "(grey = rarely visible, no class)")
    print("black arrow = patient's head; small coloured cameras = the 6 parts; black camera = the current class")
    print("keys: N/P part, R/E head direction, V look through the camera, O overview")
    update(vis)
    vis.run()
    vis.destroy_window()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default="01", choices=["01", "06", "10", "11"])
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--no-window", action="store_true")
    a = ap.parse_args()
    m, V, T, C, Nt = load(a.patient)
    lab = parts(V, C, Nt, T, m)
    print("share of the surface per part:", ", ".join(f"{PARTS[i]} {np.mean(lab == i):.0%}" for i in range(6)), f", none {np.mean(lab < 0):.0%}")
    if a.sheet:
        sheet(a.patient, m, V, T, C, Nt, lab)
    if not a.no_window:
        interactive(a.patient, m, V, T, C, Nt, lab)


if __name__ == "__main__":
    main()
