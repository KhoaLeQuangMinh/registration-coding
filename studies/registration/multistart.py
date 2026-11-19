"""
Module 2: rough pose and size from scratch, by trying many starting guesses.

Candidate cameras look at the CT liver from directions spread evenly over a sphere, with several
rolls about each direction. Each candidate puts the centre of the camera cloud on the first CT
surface point seen from that direction, with the cloud `start_size` times too big (module 3
recovers too-big starts but collapses from too-small ones). Every candidate gets a few rounds of
module 3, the `keep` best by two-way fit are refined fully, and the best fit wins.
"""
import numpy as np
import open3d as o3d

from sim3_icp import fit_error, register, rotation

START_SIZE = 1.25  # too small collapses suddenly, too big degrades gently (capture_range.py --tests size)
N_DIRS, N_ROLLS = 40, 8


def sphere_directions(n: int) -> np.ndarray:
    """n unit vectors spread evenly over the sphere (Fibonacci lattice)."""
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    phi = np.pi * (1 + 5 ** 0.5) * i
    r = np.sqrt(1 - z ** 2)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def candidates(cloud: np.ndarray, scene, centre: np.ndarray, start_size: float, n_dirs: int, n_rolls: int):
    """Starting guesses (c, Q, d), each with the cloud centre on the CT surface facing one direction."""
    m = cloud.mean(axis=0)
    u = m / np.linalg.norm(m)  # direction from the camera to the cloud centre, camera frame
    starts = []
    for v in sphere_directions(n_dirs):  # viewing direction in the CT frame
        far = centre - 1000.0 * v
        t = scene.cast_rays(o3d.core.Tensor(np.hstack([far, v])[None].astype(np.float32)))["t_hit"].numpy()[0]
        if not np.isfinite(t):
            continue
        axis = np.cross(u, v)
        align = rotation(axis / (np.linalg.norm(axis) + 1e-12), np.rad2deg(np.arccos(np.clip(u @ v, -1, 1))))
        for roll in np.arange(n_rolls) * 360.0 / n_rolls:
            Q = rotation(v, roll) @ align  # sends u to v, then spins about v
            starts.append((start_size, Q, far + t * v - start_size * Q @ m))
    return starts


def register_from_scratch(cloud: np.ndarray, rays: np.ndarray, scene, centre: np.ndarray,
                          start_size: float = START_SIZE, n_dirs: int = N_DIRS, n_rolls: int = N_ROLLS,
                          screen_iters: int = 20, keep: int = 10, outline=None):
    """Best pose (c, Q, d), plus all fully refined candidates and their fit errors (mm).

    outline: optional liver outline in the image, passed to module 3 (see sim3_icp.outline_pairs).
    """
    screened = [register(cloud, rays, scene, *s, iters=screen_iters, outline=outline)
                for s in candidates(cloud, scene, centre, start_size, n_dirs, n_rolls)]
    best = sorted(screened, key=lambda p: fit_error(cloud, rays, scene, *p, outline=outline))[:keep]
    finals = [register(cloud, rays, scene, *p, outline=outline) for p in best]
    errors = [fit_error(cloud, rays, scene, *p, outline=outline) for p in finals]
    return finals[int(np.argmin(errors))], finals, errors
