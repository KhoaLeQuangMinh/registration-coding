"""
Module 3: fine alignment of the camera cloud to the CT surface, with scale.

Two-way ICP with scale. Each round pairs points in both directions, solves the similarity
transform for all pairs (Umeyama 1991), and repeats until the cloud stops moving:
  camera -> CT  every camera point should lie on the CT surface
  CT -> camera  every CT point the camera would see should have a camera point on it
With the first direction alone, a cloud that shrinks can always lie snugly somewhere, so the
scale collapses; the second direction penalises the CT surface a shrunk cloud leaves uncovered.
Optionally a third set of pairs uses the liver outline in the image:
  outline       the CT's outline seen from the camera should lie on the liver's outline
A smaller cloud with the camera moved closer satisfies both directions above, but makes the CT
spill over background pixels; the outline pairs pull it back.
The result maps camera points y to the CT frame as X = c Q y + d: the scale c corrects the
camera cloud about the camera centre, and (Q, d) is the camera-to-CT pose.
"""
import numpy as np
import open3d as o3d


def rotation(axis: np.ndarray, deg: float) -> np.ndarray:
    """Rotation by `deg` about the unit vector `axis` (Rodrigues)."""
    a = np.deg2rad(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def umeyama(src: np.ndarray, dst: np.ndarray):
    """Scale c, rotation Q and translation d minimising sum |dst - (c Q src + d)|^2."""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    E = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    Q = U @ E @ Vt
    c = (sig * np.diag(E)).sum() / S.var(axis=0).sum()
    return c, Q, mu_d - c * Q @ mu_s


def visible_ct(scene, rays: np.ndarray, Q: np.ndarray, d: np.ndarray) -> np.ndarray:
    """CT points first hit by rays from the camera at pose (Q, d); rays are camera-frame directions.

    The scale does not move the camera centre, so it does not change the rays.
    """
    dirs = rays @ Q.T
    origins = np.broadcast_to(d, dirs.shape)
    t = scene.cast_rays(o3d.core.Tensor(np.hstack([origins, dirs]).astype(np.float32)))["t_hit"].numpy()
    hit = np.isfinite(t)
    return d + t[hit, None] * dirs[hit]


def nearest(points: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Index of the closest point to each query."""
    nns = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(points.astype(np.float32)))
    nns.knn_index()
    return nns.knn_search(o3d.core.Tensor(queries.astype(np.float32)), 1)[0].numpy()[:, 0]


def closest(dist: np.ndarray, trim: float) -> np.ndarray:
    """Mask of the `trim` fraction of pairs with the smallest distance."""
    return dist <= np.quantile(dist, trim)


def edge(mask: np.ndarray) -> np.ndarray:
    """Pixels of the mask next to a pixel outside it; the image border does not count as outside."""
    p = np.pad(mask, 1, constant_values=True)
    return mask & ~(p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:])


def outline_pairs(scene, outline, c: float, Q: np.ndarray, d: np.ndarray, trim: float):
    """Pairs that pull the CT's outline, seen from the camera, onto the liver's outline in the image.

    outline: (camera-frame direction of every pixel of a coarse grid (h, w, 3), liver mask (h, w)).
    Each CT outline point is paired with the point at the same depth on the ray through the
    nearest pixel of the liver's outline.
    """
    dirs, mask = outline
    world = dirs.reshape(-1, 3) @ Q.T
    t = scene.cast_rays(o3d.core.Tensor(np.hstack([np.broadcast_to(d, world.shape), world])
                                        .astype(np.float32)))["t_hit"].numpy().reshape(mask.shape)
    ct_edge, liver_edge = edge(np.isfinite(t)), np.argwhere(edge(mask))
    if not ct_edge.any() or not len(liver_edge):
        return None
    X = d + t[ct_edge][:, None] * world.reshape(dirs.shape)[ct_edge]  # CT outline points, CT frame
    y = (X - d) @ Q / c                                                # the same, camera frame
    pix = lambda ij: np.hstack([ij, np.zeros((len(ij), 1))])           # noqa: E731
    target = liver_edge[nearest(pix(liver_edge), pix(np.argwhere(ct_edge)))]
    y_on = dirs[target[:, 0], target[:, 1]] * y[:, 2:3]                # on the liver-outline ray, same depth
    dist = c * np.linalg.norm(y_on - y, axis=1)
    keep = closest(dist, trim)
    return y_on[keep], X[keep], dist[keep]


def pairs(cloud: np.ndarray, rays: np.ndarray, scene, c: float, Q: np.ndarray, d: np.ndarray, trim: float,
          outline=None):
    """Matched (camera point, CT point) pairs in both directions and their distances (mm).

    cloud: camera points (mm); rays: camera-frame directions through liver pixels;
    scene: CT RaycastingScene (mm); outline: see `outline_pairs` (None: not used). Each set keeps
    its `trim` fraction of closest pairs, so stray points and tool-covered regions cannot drag the fit.
    """
    X = cloud @ (c * Q).T + d
    ct = scene.compute_closest_points(o3d.core.Tensor(X.astype(np.float32)))["points"].numpy()
    dist = np.linalg.norm(ct - X, axis=1)
    keep = closest(dist, trim)
    src, dst, err = [cloud[keep]], [ct[keep]], [dist[keep]]

    seen = visible_ct(scene, rays, Q, d)
    if len(seen):
        idx = nearest(X, seen)
        dist = np.linalg.norm(X[idx] - seen, axis=1)
        keep = closest(dist, trim)
        src.append(cloud[idx[keep]])
        dst.append(seen[keep])
        err.append(dist[keep])

    found = outline_pairs(scene, outline, c, Q, d, trim) if outline is not None else None
    if found is not None:
        src.append(found[0])
        dst.append(found[1])
        err.append(found[2])
    return np.vstack(src), np.vstack(dst), np.concatenate(err)


def fit_error(cloud: np.ndarray, rays: np.ndarray, scene, c: float, Q: np.ndarray, d: np.ndarray,
              trim: float = 0.9, outline=None) -> float:
    """Mean distance of the kept pairs in both directions: how well a pose explains the view.

    Measured in the camera cloud's own units (divided by the scale c): distances in the CT frame
    shrink with the cloud, so otherwise a shrunk cloud would score well just by being small.
    """
    return pairs(cloud, rays, scene, c, Q, d, trim, outline)[2].mean() / c


def register(cloud: np.ndarray, rays: np.ndarray, scene, c: float, Q: np.ndarray, d: np.ndarray,
             trim: float = 0.9, iters: int = 500, tol_mm: float = 1e-3, outline=None):
    """Refine the starting guess (c, Q, d); arguments as in `pairs`."""
    X = cloud @ (c * Q).T + d
    for _ in range(iters):
        src, dst, _ = pairs(cloud, rays, scene, c, Q, d, trim, outline)
        c, Q, d = umeyama(src, dst)
        X_new = cloud @ (c * Q).T + d
        if np.abs(X_new - X).max() < tol_mm:
            break
        X = X_new
    return c, Q, d
