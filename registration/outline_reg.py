"""
Rigid registration of the CT liver to one image, from the liver outline and, optionally, the ridge.

No point cloud: the CT is in mm and the camera is calibrated, so where the outline is, its shape
and its size fix the whole pose, depth included. Each round pairs points in the image (pixels)
both ways and takes one Gauss-Newton step on the pose (R, t: CT -> camera, mm):
  outline  CT outline seen from the camera  <->  liver outline in the mask
  ridge    CT ridge seen from the camera    <->  ridge drawn in the image
Each set keeps its 90 % closest pairs and counts as a whole (its mean), the ridge times
`ridge_weight`. Pixels next to a tool are ignored: the liver's edge there is unknown.
Search from scratch tries 40 viewing directions x 8 rolls (as module 2), each starting at the
distance where the CT looks as big as the mask.
"""
import numpy as np
import open3d as o3d

from multistart import N_DIRS, N_ROLLS, sphere_directions
from sim3_icp import closest, edge, nearest, rotation

STEP = 4      # coarse pixel grid for rendering the CT outline
TRIM = 0.9


def grow(mask: np.ndarray) -> np.ndarray:
    p = np.pad(mask, 1)
    return mask | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]


def make_view(K: np.ndarray, liver: np.ndarray, tool: np.ndarray, ridge2d=None) -> dict:
    """One image: K, liver and tool masks (full size), ridge pixels (n x 2, optional)."""
    H, W = liver.shape
    v, u = np.mgrid[0:H:STEP, 0:W:STEP]
    pix = np.stack([u + 0.5, v + 0.5], axis=-1)
    dirs = np.dstack([(pix[..., 0] - K[0, 2]) / K[0, 0], (pix[..., 1] - K[1, 2]) / K[1, 1], np.ones(u.shape)])
    unknown = grow(tool[::STEP, ::STEP])
    return dict(K=K, size=(W, H), pix=pix, dirs=dirs, unknown=unknown, liver=liver[::STEP, ::STEP],
                liver_edge=pix[edge(liver[::STEP, ::STEP]) & ~unknown], ridge2d=ridge2d)


def project(K: np.ndarray, Xc: np.ndarray) -> np.ndarray:
    return np.stack([K[0, 0] * Xc[:, 0] / Xc[:, 2] + K[0, 2], K[1, 1] * Xc[:, 1] / Xc[:, 2] + K[1, 2]], axis=1)


def render(scene, view, R, t) -> np.ndarray:
    """Ray length to the CT per grid pixel (inf where the ray misses)."""
    world = view["dirs"].reshape(-1, 3) @ R
    origins = np.broadcast_to(-R.T @ t, world.shape)
    return scene.cast_rays(o3d.core.Tensor(np.hstack([origins, world]).astype(np.float32)))["t_hit"] \
        .numpy().reshape(view["dirs"].shape[:2])


def ct_outline(scene, view, R, t):
    """CT outline seen from the camera: CT points (CT frame) and their pixels."""
    hit = render(scene, view, R, t)
    keep = edge(np.isfinite(hit)) & ~view["unknown"]
    return (view["dirs"][keep] * hit[keep, None] - t) @ R, view["pix"][keep]


def visible(scene, view, R, t, X: np.ndarray) -> np.ndarray:
    """Mask of CT points X seen by the camera: in front, inside the image, not hidden, not under a tool."""
    Xc = X @ R.T + t
    p = project(view["K"], np.where(Xc[:, 2:] > 1e-6, Xc, np.nan))
    W, H = view["size"]
    ok = (Xc[:, 2] > 1e-6) & (p[:, 0] >= 0) & (p[:, 0] < W) & (p[:, 1] >= 0) & (p[:, 1] < H)
    i, j = (np.nan_to_num(p[:, 1]) // STEP).astype(int), (np.nan_to_num(p[:, 0]) // STEP).astype(int)
    ok &= ~view["unknown"][np.clip(i, 0, view["unknown"].shape[0] - 1), np.clip(j, 0, view["unknown"].shape[1] - 1)]
    C = -R.T @ t
    dist = np.linalg.norm(X - C, axis=1)
    rays = np.hstack([np.broadcast_to(C, X.shape), (X - C) / dist[:, None]]).astype(np.float32)
    return ok & (scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy() >= dist - 1.0)


def match(src_px: np.ndarray, dst_px: np.ndarray) -> np.ndarray:
    """Index into dst of the nearest pixel to each src pixel."""
    z = lambda p: np.hstack([p, np.zeros((len(p), 1))])  # noqa: E731
    return nearest(z(dst_px), z(src_px))


def pairs(scene, view, R, t, ridge3d=None, ridge_weight=1.0):
    """Pair sets as (CT points, target pixels, weight of the set)."""
    out = []
    X, px = ct_outline(scene, view, R, t)
    m = view["liver_edge"]
    if len(X) and len(m):
        out.append((X, m[match(px, m)], 0.5))            # CT outline -> liver outline
        out.append((X[match(m, px)], m, 0.5))            # liver outline -> CT outline
    r = view["ridge2d"]
    if ridge3d is not None and r is not None and len(r):
        seen = ridge3d[visible(scene, view, R, t, ridge3d)]
        front = ridge3d[(ridge3d @ R.T + t)[:, 2] > 1e-6]
        if len(seen):
            out.append((seen, r[match(project(view["K"], seen @ R.T + t), r)], 0.5 * ridge_weight))
        if len(front):
            out.append((front[match(r, project(view["K"], front @ R.T + t))], r, 0.5 * ridge_weight))
    kept = []
    for X, target, w in out:
        d = np.linalg.norm(project(view["K"], X @ R.T + t) - target, axis=1)
        k = closest(d, TRIM)
        kept.append((X[k], target[k], w, d[k]))
    return kept


def fit_error(scene, view, R, t, ridge3d=None, ridge_weight=1.0) -> float:
    """Weighted mean pixel distance of the kept pairs (inf if the CT outline is not in view)."""
    kept = pairs(scene, view, R, t, ridge3d, ridge_weight)
    if len(kept) < 2:
        return np.inf
    return sum(w * d.mean() for _, _, w, d in kept)


def gn_step(K, R, t, kept, rotate=True):
    """One Gauss-Newton step: rotation w about the camera centre and shift dt, both in the camera frame."""
    A, b = np.zeros((6, 6)), np.zeros(6)
    for X, target, w, _ in kept:
        Xc = X @ R.T + t
        x, y, z = Xc.T
        Jp = np.zeros((len(X), 2, 3))
        Jp[:, 0, 0], Jp[:, 0, 2] = K[0, 0] / z, -K[0, 0] * x / z ** 2
        Jp[:, 1, 1], Jp[:, 1, 2] = K[1, 1] / z, -K[1, 1] * y / z ** 2
        S = np.zeros((len(X), 3, 3))  # d(Xc) = w x Xc + dt = -[Xc]x w + dt
        S[:, 0, 1], S[:, 0, 2], S[:, 1, 0], S[:, 1, 2], S[:, 2, 0], S[:, 2, 1] = z, -y, -z, x, y, -x
        J = np.concatenate([Jp @ S, Jp], axis=2)
        r = project(K, Xc) - target
        A += w / len(X) * np.einsum("nij,nik->jk", J, J)
        b += w / len(X) * np.einsum("nij,ni->j", J, r)
    free = slice(0, 6) if rotate else slice(3, 6)
    A, b = A[free, free], b[free]
    step = np.zeros(6)
    step[free] = np.linalg.solve(A + 1e-6 * np.trace(A) * np.eye(len(b)), -b)
    angle = np.linalg.norm(step[:3])
    dR = rotation(step[:3] / angle, np.rad2deg(angle)) if angle > 0 else np.eye(3)
    return dR @ R, dR @ t + step[3:]


def register(scene, view, R, t, ridge3d=None, ridge_weight=1.0, iters=100, rotate=True, tol=1e-3):
    """Refine the pose (R, t)."""
    for _ in range(iters):
        kept = pairs(scene, view, R, t, ridge3d, ridge_weight)
        if len(kept) < 2:
            break
        R_new, t_new = gn_step(view["K"], R, t, kept, rotate)
        moved = np.linalg.norm(R_new - R) * 100 + np.linalg.norm(t_new - t)
        R, t = R_new, t_new
        if moved < tol:
            break
    return R, t


def starts(scene, view, centre: np.ndarray, n_dirs: int = N_DIRS, n_rolls: int = N_ROLLS):
    """Poses looking at the CT centre along the ray through the mask's centre, at matching size."""
    m = view["pix"][view["liver"]].mean(axis=0)
    K = view["K"]
    u = np.array([(m[0] - K[0, 2]) / K[0, 0], (m[1] - K[1, 2]) / K[1, 1], 1.0])
    u /= np.linalg.norm(u)
    out = []
    for v in sphere_directions(n_dirs):  # viewing direction in the CT frame
        axis = np.cross(u, v)
        align = rotation(axis / (np.linalg.norm(axis) + 1e-12), np.rad2deg(np.arccos(np.clip(u @ v, -1, 1))))
        for roll in np.arange(n_rolls) * 360.0 / n_rolls:
            R = (rotation(v, roll) @ align).T  # CT -> camera; sends v to u
            area = np.isfinite(render(scene, view, R, -R @ (centre - 300.0 * v))).sum()
            if area:
                dist = 300.0 * np.sqrt(area / view["liver"].sum())
                out.append((R, -R @ (centre - dist * v)))
    return out


def register_from_scratch(scene, view, centre, ridge3d=None, ridge_weight=1.0, n_dirs=N_DIRS, n_rolls=N_ROLLS,
                          screen_iters=15, keep=10):
    """Best pose (R, t) and its fit error."""
    err = lambda p: fit_error(scene, view, *p, ridge3d, ridge_weight)  # noqa: E731
    screened = [register(scene, view, *s, ridge3d, ridge_weight, iters=screen_iters)
                for s in starts(scene, view, centre, n_dirs, n_rolls)]
    finals = [register(scene, view, *p, ridge3d, ridge_weight) for p in sorted(screened, key=err)[:keep]]
    errors = [err(p) for p in finals]
    return finals[int(np.argmin(errors))], min(errors)
