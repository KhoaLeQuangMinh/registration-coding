"""
P2I-LReg access and drawing helpers shared by the generator, the lab and the previews.

Dataset root: $REGSIM_P2ILREG (default dataset/liver-registration, relative to the repo root); outputs: $REGSIM_OUT
(default out/, git-ignored). Conventions verified on the data (2026-11-04 .. 11-17): CT meshes are in metres and are
converted to mm; CT axes x = patient left, y = back, z = head; synthetic poses are Blender cameras (-Z forward) and are
converted to vision cameras (+Z forward, +Y down); real PLY / label details in the archive branch notes.
"""
import os

import cv2
import numpy as np
import open3d as o3d
import yaml

DATA = os.environ.get("REGSIM_P2ILREG", "dataset/liver-registration")
OUT = os.environ.get("REGSIM_OUT", "out")
F = np.diag([1.0, -1.0, -1.0])          # Blender camera (-Z forward, +Y up) <-> vision camera (+Z forward, +Y down)
RW, RH, DOWN = 480, 270, 4               # drawing size: the 1920 x 1080 camera / 4


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


def rot(axis, deg):
    return cv2.Rodrigues(np.eye(3)[axis] * np.radians(deg))[0]


def perturbed(R, t, c_cam, dR=np.eye(3), dt=np.zeros(3)):
    """Move the CT about its centre in the camera frame: X' = dR (X_cam - c) + c + dt."""
    return dR @ R, dR @ (t - c_cam) + c_cam + dt


def look_at(centre, direction, dist, roll):
    """Camera at centre + dist*direction looking at the centre (vision axes), rolled about the view axis."""
    z = -direction / np.linalg.norm(direction)
    up = np.array([0, 0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0, 0])
    x = np.cross(z, up); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = rot(2, roll) @ np.stack([x, y, z])
    return R, -R @ (centre + dist * direction / np.linalg.norm(direction))


def load_photo(p, f):
    d = f"{DATA}/{p}/real/labels/frame_{f}_json"
    img = cv2.imread(f"{d}/img.png")
    M = cv2.resize((cv2.imread(f"{d}/label.png")[..., 2] > 0).astype(np.uint8), (RW, RH), interpolation=cv2.INTER_NEAREST) > 0
    grey = cv2.cvtColor(cv2.resize(img, (RW, RH), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    # not tissue: the dark corners outside the scope and the glaring white scope rim, i.e. very dark or very bright
    # regions that touch the image border (shiny spots ON the liver do not touch the border, so they stay)
    bad = np.zeros_like(M)
    for cand in (grey < 30, grey > 225):
        n, cc = cv2.connectedComponents(cand.astype(np.uint8))
        edge = np.unique(np.concatenate([cc[0], cc[-1], cc[:, 0], cc[:, -1]]))
        bad |= np.isin(cc, edge[edge > 0])
    bad = cv2.dilate(bad.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    V = ~bad | M
    pts = np.asarray(o3d.io.read_point_cloud(f"{DATA}/{p}/real/liverPcds/frame_{f}_json.ply").points)
    P = pts.reshape(1080, 1920, 3)
    lab_full = cv2.imread(f"{d}/label.png")[..., 2] > 0
    z_cam = np.median(P[~lab_full][:, 2])                                 # background pixels sit at the camera point
    D = np.where(lab_full, P[..., 2] - z_cam, 0)                         # depth up to the cloud's scale
    D = cv2.resize(D.astype(np.float32), (RW, RH), interpolation=cv2.INTER_NEAREST)
    return img, M, V, D
