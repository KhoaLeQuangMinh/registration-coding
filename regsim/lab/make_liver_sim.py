"""
Build out/liver_sim.html: a surgical deformation lab with LIBR physics. Left: the laparoscope's view, where you
press and grasp like the surgeon. Right: the whole liver in 3D, deforming the same way. Manipulations can be recorded.

Physics (LIBR-style, after Heiselman & Miga's linearized iterative boundary reconstruction):
  1. the CT liver is filled with cubes of side H_MM, each split into 6 tetrahedra (Kuhn split, conforming)
  2. linear-elastic finite elements (Young's modulus 1, Poisson ratio NU) with the attached region held fixed
     (by default the posterior-cranial "bare area" near the vena cava, ~8 % of the nodes)
  3. a basis of responses: a Gaussian-spread surface force (reach SIGMA_MM) at each of N_CTRL control points
     (farthest-point sample of the free surface), in x, y and z; plus unit gravity in x, y, z and a unit gas push
     (pressure on the free front and lower surfaces)
  4. strain energy of the control-point responses, E_ij = u_i . f_j (for the fit)
  In the browser a tool sets targets ("this point goes there"); the control-point weights that meet the targets with
  the least strain energy are solved for (a small linear system), so the whole organ deforms as one elastic body.
  Linear elasticity: accurate for dents, gas and sag; large lifts that should ROTATE a lobe stretch it instead.
Sizes of gas and gravity responses are normalised (gas 100 % -> 10 mm, head-up 30 deg -> 20 mm at the surface,
uncalibrated placeholders until checked against measured deformations).
Usage (repo root): python -m regsim.lab.make_liver_sim [--patient 06] && open out/liver_sim.html
"""
import argparse
import base64
import json
import os
import time

import numpy as np
import open3d as o3d
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.spatial import cKDTree

from regsim import anatomy as sc
from regsim.p2ilreg import DATA, OUT, Patient

H_MM, NU, N_CTRL, SIGMA_MM, HFOV = 8.0, 0.45, 48, 15.0, 70.0
TETS = [[0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7], [0, 2, 6, 7], [0, 4, 5, 7], [0, 4, 6, 7]]


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def tet_mesh(m, V, Vs):
    scene = o3d.t.geometry.RaycastingScene(); scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
    lo = V.min(0) - H_MM; n = np.ceil((V.max(0) + H_MM - lo) / H_MM).astype(int)
    ijk = np.stack(np.meshgrid(*[np.arange(k) for k in n], indexing="ij"), -1).reshape(-1, 3)
    inside = scene.compute_occupancy(o3d.core.Tensor((lo + (ijk + 0.5) * H_MM).astype(np.float32))).numpy() > 0.5
    keep = set(map(tuple, ijk[inside])) | set(map(tuple, np.floor((Vs - lo) / H_MM).astype(int)))
    cubes = np.array(sorted(keep)); corner = {}; tets = []
    for c in cubes:
        ids = []
        for b in range(8):
            key = (c[0] + (b & 1), c[1] + ((b >> 1) & 1), c[2] + ((b >> 2) & 1))
            ids.append(corner.setdefault(key, len(corner)))
        tets += [[ids[k] for k in tt] for tt in TETS]
    X = lo + np.array(sorted(corner, key=corner.get), float) * H_MM
    tets = np.array(tets)
    d = np.einsum("ij,ij->i", np.cross(X[tets[:, 1]] - X[tets[:, 0]], X[tets[:, 2]] - X[tets[:, 0]]), X[tets[:, 3]] - X[tets[:, 0]])
    tets[d < 0] = tets[d < 0][:, [0, 2, 1, 3]]
    return X, tets, lo, {tuple(c): i for i, c in enumerate(cubes)}


def embed(Xq, X, tets, lo, cube_of):
    tid, w = np.zeros(len(Xq), np.int32), np.zeros((len(Xq), 4))
    for i, x in enumerate(Xq):
        c = cube_of.get(tuple(np.floor((x - lo) / H_MM).astype(int)))
        best = -1e9
        for k in (range(c * 6, c * 6 + 6) if c is not None else range(len(tets))):
            T = X[tets[k]]; l = np.linalg.solve(np.stack([T[1] - T[0], T[2] - T[0], T[3] - T[0]], 1), x - T[0]); l = np.r_[1 - l.sum(), l]
            if l.min() > best:
                best, w[i], tid[i] = l.min(), l, k
    return tid, w


def stiffness(X, tets):
    """Linear-elastic stiffness matrix (E = 1, Poisson NU), linear tetrahedra; also nodal volumes."""
    lam, mu = NU / ((1 + NU) * (1 - 2 * NU)), 1 / (2 * (1 + NU))
    D = np.zeros((6, 6)); D[:3, :3] = lam; D[[0, 1, 2], [0, 1, 2]] += 2 * mu; D[[3, 4, 5], [3, 4, 5]] = mu
    P = X[tets]; Dm = np.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0], P[:, 3] - P[:, 0]], 2)       # nt x 3 x 3 (columns)
    vol = np.linalg.det(Dm) / 6; inv = np.linalg.inv(Dm)                                          # rows = grad N1..3
    g = np.concatenate([-inv.sum(1, keepdims=True), inv], 1)                                      # nt x 4 x 3
    B = np.zeros((len(tets), 6, 12))
    for a in range(4):
        bx, by, bz = g[:, a, 0], g[:, a, 1], g[:, a, 2]; c = 3 * a
        B[:, 0, c] = bx; B[:, 1, c + 1] = by; B[:, 2, c + 2] = bz
        B[:, 3, c] = by; B[:, 3, c + 1] = bx; B[:, 4, c + 1] = bz; B[:, 4, c + 2] = by; B[:, 5, c] = bz; B[:, 5, c + 2] = bx
    Ke = vol[:, None, None] * np.einsum("tki,kl,tlj->tij", B, D, B)
    dofs = (3 * tets[:, :, None] + np.arange(3)).reshape(len(tets), 12)
    Kmat = sp.coo_matrix((Ke.ravel(), (np.repeat(dofs, 12, 1).ravel(), np.tile(dofs, (1, 12)).ravel())), shape=(3 * len(X),) * 2).tocsc()
    nvol = np.zeros(len(X)); np.add.at(nvol, tets.ravel(), np.repeat(vol / 4, 4))
    return Kmat, nvol


def boundary(X, tets):
    faces = {}
    for t, tt in enumerate(tets):
        for f in ([0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]):
            ids = tuple(tt[f]); key = tuple(sorted(ids))
            faces[key] = None if key in faces else (ids, t)
    out = []
    for v in faces.values():
        if v is None:
            continue
        ids, t = v; a, b, c = X[list(ids)]
        n = np.cross(b - a, c - a); other = [k for k in tets[t] if k not in ids][0]
        if n @ (X[other] - a) > 0:
            n = -n
        out.append((ids, n / 2))                                  # area-weighted outward normal
    return out


def build(p):
    t0 = time.time()
    m, V, T, C, Nt = sc.load(p)
    surf = m.simplify_quadric_decimation(10000); surf.remove_degenerate_triangles(); surf.remove_unreferenced_vertices()
    Vs, Ts = np.asarray(surf.vertices), np.asarray(surf.triangles)
    X, tets, lo, cube_of = tet_mesh(m, V, Vs)
    n = len(X)
    Kmat, nvol = stiffness(X, tets)
    bnd = boundary(X, tets)
    on_surf = np.zeros(n, bool); snorm = np.zeros((n, 3))
    for ids, nrm in bnd:
        on_surf[list(ids)] = True; snorm[list(ids)] += nrm
    # attachments: the posterior-cranial region (bare area / vena cava), ~8 % of the nodes
    score = (X[:, 1] - X[:, 1].min()) / np.ptp(X[:, 1]) + (X[:, 2] - X[:, 2].min()) / np.ptp(X[:, 2])
    fixed = score >= np.quantile(score, 0.92)
    free_dof = np.repeat(~fixed, 3)
    # loads: control points (Gaussian-spread surface forces), gravity x/y/z, gas
    cand = np.where(on_surf & ~fixed)[0]
    ctrl = [cand[0]]; dist = np.full(len(cand), np.inf)
    for _ in range(N_CTRL - 1):
        dist = np.minimum(dist, np.linalg.norm(X[cand] - X[ctrl[-1]], axis=1)); ctrl.append(cand[int(dist.argmax())])
    ctrl = np.array(ctrl)
    sidx = np.where(on_surf)[0]
    F = []
    for c in ctrl:
        wgt = np.exp(-np.sum((X[sidx] - X[c]) ** 2, 1) / (2 * SIGMA_MM ** 2)); wgt /= wgt.sum()
        for k in range(3):
            f = np.zeros(3 * n); f[3 * sidx + k] = wgt; F.append(f)
    for k in range(3):                                            # unit gravity along x, y, z (per unit volume)
        f = np.zeros(3 * n); f[3 * np.arange(n) + k] = nvol; F.append(f)
    gas = np.zeros(3 * n)                                         # push on the free front (-y) and lower (-z) surfaces
    for ids, nrm in bnd:
        nn = nrm / (np.linalg.norm(nrm) + 1e-12); w = np.clip(-nn[1] + 0.3 * -nn[2], 0, 1)
        for i in ids:
            gas[3 * i:3 * i + 3] += -nrm * w / 3
    F.append(gas)
    F = np.stack(F, 1)
    Kff = Kmat[free_dof][:, free_dof].tocsc()
    lu = spl.splu(Kff)
    U = np.zeros((3 * n, F.shape[1])); U[free_dof] = lu.solve(F[free_dof])
    nc = 3 * N_CTRL
    Emat = U[:, :nc].T @ F[:, :nc]; Emat = 0.5 * (Emat + Emat.T)    # strain energy of the control responses
    # the basis at the rendered surface and the virtual tumour (embedded in the tetrahedra)
    s_tid, s_w = embed(Vs, X, tets, lo, cube_of)
    lab = sc.parts(V, C, Nt, T, m)
    area = np.linalg.norm(np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]), axis=1); sel = lab == 2
    front_pt = (C[sel] * area[sel, None]).sum(0) / area[sel].sum()
    n_front = (Nt[sel] * area[sel, None]).sum(0); n_front /= np.linalg.norm(n_front)
    tc = front_pt - 25 * n_front
    sph = o3d.geometry.TriangleMesh.create_sphere(12.0, 12); sph.translate(tc)
    tv, tf = np.asarray(sph.vertices), np.asarray(sph.triangles)
    t_tid, t_w = embed(tv, X, tets, lo, cube_of)

    def at(tid, w):                                               # basis at embedded points: (npts*3) x ncases
        Ur = U.reshape(n, 3, -1)
        return np.einsum("pk,pkdc->pdc", w, Ur[tets[tid]]).reshape(-1, U.shape[1])
    Us, Ut = at(s_tid, s_w), at(t_tid, t_w)
    # normalise gas and gravity so their sizes are plausible placeholders (gas 100 % -> 10 mm; head-up 30 deg -> 20 mm)
    surf_max = lambda col: np.linalg.norm(Us[:, col].reshape(-1, 3), axis=1).max()  # noqa: E731
    gas_scale = 10.0 / surf_max(nc + 3)
    g30 = 9.81 * np.array([0.0, np.cos(np.radians(30)) - 1, -np.sin(np.radians(30))])
    grav_mm = np.linalg.norm((Us[:, nc:nc + 3] @ g30).reshape(-1, 3), axis=1).max()
    grav_scale = 20.0 / grav_mm
    Uall = np.vstack([Us, Ut])
    qscale = np.abs(Uall).max(0) / 32000 + 1e-30
    Uq = np.round(Uall / qscale).astype(np.int16)
    fixed_render = np.where(fixed[tets[s_tid]].sum(1) >= 3)[0]
    # a 4x finer surface for drawing: LIBR movement interpolated from the coarse surface (smooth anyway), dents computed on it
    fine = m.simplify_quadric_decimation(40000); fine.remove_degenerate_triangles(); fine.remove_unreferenced_vertices()
    Vf, Tf = np.asarray(fine.vertices), np.asarray(fine.triangles)
    cs = o3d.t.geometry.RaycastingScene(); cmesh = o3d.t.geometry.TriangleMesh(); cmesh.vertex.positions = o3d.core.Tensor(Vs.astype(np.float32))
    cmesh.triangle.indices = o3d.core.Tensor(Ts.astype(np.int32)); cs.add_triangles(cmesh)
    cl = cs.compute_closest_points(o3d.core.Tensor(Vf.astype(np.float32)))
    tri = Ts[cl["primitive_ids"].numpy()]; q = cl["points"].numpy()
    A, B, C_ = Vs[tri[:, 0]], Vs[tri[:, 1]], Vs[tri[:, 2]]
    v0, v1, v2 = B - A, C_ - A, q - A
    d00, d01, d11, d20, d21 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1), (v2 * v0).sum(1), (v2 * v1).sum(1)
    den = d00 * d11 - d01 * d01 + 1e-12; bv = (d11 * d20 - d01 * d21) / den; bw = (d00 * d21 - d01 * d20) / den
    fw = np.stack([1 - bv - bw, bv, bw], 1).clip(0, 1); fw /= fw.sum(1, keepdims=True)
    f_near = cKDTree(Vs).query(Vf)[1]
    # cameras: 6 well-spread P2I-LReg synthetic poses, realistic lens
    pat = Patient(p)
    keys = [l.strip().replace("ptsLiverCam", "") for l in open(f"{DATA}/{p}/train_syn.txt") if l.strip()]
    poses = [pat.true_pose(k) for k in keys]
    dirs = np.array([(-R.T @ t - pat.centre) / np.linalg.norm(-R.T @ t - pat.centre) for R, t in poses])
    pick = [int(np.argmax(dirs @ np.array([0, -1.0, 0])))]
    for _ in range(5):
        pick.append(int(np.argmax(np.min(1 - dirs @ dirs[pick].T, axis=1))))
    fx = 960 / np.tan(np.radians(HFOV) / 2)
    print(f"patient {p}: {n} nodes, {len(tets)} tetrahedra ({H_MM} mm cubes), {fixed.sum()} fixed nodes, {N_CTRL} control points, "
          f"{F.shape[1]} load cases solved in {time.time() - t0:.0f} s; surface {len(Vs)} vertices (drawn: {len(Vf)})")
    return dict(patient=p, nu=NU, sigma_mm=SIGMA_MM, ncase=F.shape[1], nctrl=N_CTRL,
                surf=dict(v=b64(Vs.astype(np.float32)), f=b64(Ts.astype(np.uint32)), nv=len(Vs)),
                fine=dict(v=b64(Vf.astype(np.float32)), f=b64(Tf.astype(np.uint32)), nv=len(Vf), tri=b64(tri.astype(np.uint32)),
                          w=b64(fw.astype(np.float32)), near=b64(f_near.astype(np.uint32))),
                tumour=dict(v=b64(tv.astype(np.float32)), f=b64(tf.astype(np.uint32)), nv=len(tv), centre=tc.tolist()),
                U=b64(Uq), qscale=b64(qscale.astype(np.float32)), E=b64(Emat.astype(np.float64)),
                gas_scale=float(gas_scale), grav_scale=float(grav_scale), ctrl=X[ctrl].tolist(),
                fixed_render=fixed_render.tolist(), cams=[dict(key=keys[i], R=poses[i][0].tolist(), t=poses[i][1].tolist()) for i in pick],
                K=[[fx, 0, 960], [0, fx, 540], [0, 0, 1]], centre=V.mean(0).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default="06")
    a = ap.parse_args()
    data = build(a.patient)
    html = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "liver_sim_template.html")).read().replace("/*__DATA__*/null", json.dumps(data))
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, "liver_sim.html")
    open(out, "w").write(html)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
