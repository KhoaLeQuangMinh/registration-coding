"""
Synthetic training generator (PROTOTYPE): new views of the CT livers with the answer known, damaged the way real
laparoscope images are damaged. One call to Generator.sample() makes one training example.

Steps (every number below is a placeholder range until backed by a published source; see the TODO list at the end):
  1. bend     smooth random pushes at 40 control points (the playground's / lus_best's model), mean tissue movement
              0 (25 % of examples) or 2-20 mm; the overall turn + shift hidden in a random bend is removed, so the
              true pose stays the unbent CT's pose
  2. camera   direction from a target point on the liver: within 80 deg of the patient's front (anterior), the
              target a random liver point facing that way, 40-180 mm away (log-uniform), target placed anywhere in
              the middle 60 % of the picture, any roll; lens 60-90 deg wide (1920 x 1080), principal point off-centre
              by N(0, 60 px), at most 150 px
  3. draw     liver silhouette, depth and triangle ids at 480 x 270
  4. cover    scope circle (50 %), 0-2 tapered tools, smooth fat patches over the liver edge (50 %), a ligament band
              entering at the edge and crossing part of the liver (30 %)
  5. edges    every edge pixel of the visible liver is labelled: real (the liver ends there), covered (the liver goes
              on under something) or border (image or scope edge)
  6. mask     small smooth wobble of the visible-liver mask edge (up to 2.5 px), as a segmentation network would make
  7. depth    true depth damaged (bumpy / flattened / slanted, 0-35 deg mean tilt error; shape_trust.distort), then
              divided by its median (the scale is thrown away, as with Depth Anything)
  8. answer   true R, t, K, the bend, and 64 random points inside the liver in the camera frame (virtual tumours)
The CT frame is P2I-LReg's: mm (converted), centred, x = patient left, y = back, z = head; front = -y.
"""
import cv2
import numpy as np
import open3d as o3d
from scipy.ndimage import gaussian_filter

from regsim import anatomy as sc
from regsim.depth_damage import distort

W, H, IMW, IMH = 480, 270, 1920, 1080
NH = 40
FRONT = np.array([0.0, -1.0, 0.0])

CFG = dict(
    no_bend=0.25, bend_mm=(2.0, 20.0), sigma_mm=(30.0, 60.0),
    cone_deg=80.0, dist_mm=(40.0, 180.0), offset=0.3,
    hfov_deg=(60.0, 90.0), pp_sd=60.0, pp_max=150.0,
    scope=0.5, scope_r=(0.33, 0.62), tools=(0.3, 0.45, 0.25), tool_w=(14, 40),
    fat=0.5, ligament=0.3, ligament_w=(6, 16),
    wobble_px=2.5, depth_kinds=("bumpy", "flattened", "slanted"), depth_deg=(0.0, 35.0),
    min_visible=0.03, n_inside=64,
)


def rot_to(a, b):
    """Rotation taking unit vector a to unit vector b."""
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def umeyama_rigid(A, B):
    """R, t minimising |R A + t - B| (no scale)."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((B - cb).T @ (A - ca))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    return R, cb - R @ ca


class Liver:
    def __init__(self, p):
        m, V, T, C, Nt = sc.load(p)
        self.p, self.V, self.T = p, V, T
        self.centre = V.mean(0)
        m.compute_vertex_normals()
        N = np.asarray(m.vertex_normals)
        self.N = N if ((V - self.centre) * N).sum(1).mean() > 0 else -N      # outward
        idx = [0]; d = np.full(len(V), np.inf)
        for _ in range(NH - 1):
            d = np.minimum(d, ((V - V[idx[-1]]) ** 2).sum(1)); idx.append(int(d.argmax()))
        self.hp = V[idx]
        self.d2 = ((V[:, None] - self.hp[None]) ** 2).sum(2)            # squared distance to the control points
        rest = o3d.t.geometry.RaycastingScene()
        rest.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
        self.rest = rest
        self.lo, self.hi = V.min(0), V.max(0)


class Generator:
    def __init__(self, patients, cfg=None):
        self.cfg = dict(CFG, **(cfg or {}))
        self.livers = {p: Liver(p) for p in patients}

    # ---------------- 1. bend ----------------
    def bend(self, lv, rng):
        c = self.cfg
        inside = self.inside_points(lv, rng)
        if rng.random() < c["no_bend"]:
            return lv.V, inside, dict(bend_mean_mm=0.0, bend_max_mm=0.0, sigma_mm=0.0), np.zeros((NH, 3)), 0.0
        sigma = rng.uniform(*c["sigma_mm"])
        D = rng.standard_normal((NH, 3))
        phi = np.exp(-lv.d2 / (2 * sigma ** 2))
        disp = phi @ D
        target = rng.uniform(*c["bend_mm"])
        k = target / np.linalg.norm(disp, axis=1).mean()
        D *= k
        Vd = lv.V + phi @ D
        Rb, tb = umeyama_rigid(Vd, lv.V)                                     # remove the hidden overall turn + shift
        Vd = Vd @ Rb.T + tb
        phi_in = np.exp(-((inside[:, None] - lv.hp[None]) ** 2).sum(2) / (2 * sigma ** 2))
        inside = (inside + phi_in @ D) @ Rb.T + tb
        mv = np.linalg.norm(Vd - lv.V, axis=1)
        return Vd, inside, dict(bend_mean_mm=float(mv.mean()), bend_max_mm=float(mv.max()), sigma_mm=float(sigma)), D, sigma

    def inside_points(self, lv, rng):
        P = rng.uniform(lv.lo, lv.hi, (self.cfg["n_inside"] * 8, 3)).astype(np.float32)
        occ = lv.rest.compute_occupancy(o3d.core.Tensor(P)).numpy() > 0.5
        return P[occ][: self.cfg["n_inside"]].astype(float)

    # ---------------- 2. camera ----------------
    def camera(self, lv, Vd, rng):
        c = self.cfg
        # view direction (from the liver to the camera): uniform on a cap around the front
        cosmax = np.cos(np.radians(c["cone_deg"]))
        ct = rng.uniform(cosmax, 1.0); ph = rng.uniform(0, 2 * np.pi); st = np.sqrt(1 - ct ** 2)
        e1 = np.array([1.0, 0, 0]); e2 = np.cross(FRONT, e1)
        d = ct * FRONT + st * (np.cos(ph) * e1 + np.sin(ph) * e2)
        facing = np.where(lv.N @ d > 0.2)[0]
        target = Vd[rng.choice(facing)]
        r = float(np.exp(rng.uniform(*np.log(c["dist_mm"]))))
        C = target + r * d
        hfov = rng.uniform(*c["hfov_deg"])
        fx = (IMW / 2) / np.tan(np.radians(hfov) / 2)
        pp = np.clip(rng.normal(0, c["pp_sd"], 2), -c["pp_max"], c["pp_max"])
        K = np.array([[fx, 0, IMW / 2 + pp[0]], [0, fx, IMH / 2 + pp[1]], [0, 0, 1.0]])
        # camera looking at the target, rolled, then turned so the target lands off-centre
        z = (target - C) / np.linalg.norm(target - C)
        ref = np.array([0, 0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0, 0])
        x = np.cross(z, ref); x /= np.linalg.norm(x); y = np.cross(z, x)
        R0 = np.stack([x, y, z])
        roll = rng.uniform(0, 360)
        R0 = cv2.Rodrigues(np.array([0, 0, np.radians(roll)]))[0] @ R0
        u = K[0, 2] + rng.uniform(-c["offset"], c["offset"]) * IMW
        v = K[1, 2] + rng.uniform(-c["offset"], c["offset"]) * IMH
        dc = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0]); dc /= np.linalg.norm(dc)
        R = rot_to(np.array([0, 0, 1.0]), dc) @ R0
        t = -R @ C
        ang = float(np.degrees(np.arccos(np.clip(((C - lv.centre) / np.linalg.norm(C - lv.centre)) @ FRONT, -1, 1))))
        return R, t, K, dict(hfov_deg=float(hfov), pp_off_px=pp.round(0).tolist(), target_dist_mm=r, roll_deg=float(roll),
                             front_angle_deg=ang, centre_dist_mm=float(np.linalg.norm(C - lv.centre)))

    # ---------------- 3. draw ----------------
    @staticmethod
    def draw(Vd, T, R, t, K):
        m = o3d.t.geometry.TriangleMesh()
        m.vertex.positions = o3d.core.Tensor(Vd.astype(np.float32)); m.triangle.indices = o3d.core.Tensor(T.astype(np.int32))
        scene = o3d.t.geometry.RaycastingScene(); scene.add_triangles(m)
        Kc = np.diag([W / IMW, H / IMH, 1.0]) @ K
        u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        d = np.stack([(u - Kc[0, 2]) / Kc[0, 0], (v - Kc[1, 2]) / Kc[1, 1], np.ones_like(u)], -1).reshape(-1, 3)
        dn = d / np.linalg.norm(d, axis=1, keepdims=True)
        rays = np.hstack([np.repeat((-R.T @ t)[None], len(d), 0), dn @ R]).astype(np.float32)
        th = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        hit = np.isfinite(th)
        return hit.reshape(H, W), np.where(hit, th * dn[:, 2], 0.0).reshape(H, W), Kc, scene

    # ---------------- 4. cover ----------------
    def cover(self, S, rng):
        c = self.cfg
        u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        layers, info = {}, {}
        if rng.random() < c["scope"]:
            rad = rng.uniform(*c["scope_r"]) * W
            cx, cy = W / 2 + rng.normal(0, 8), H / 2 + rng.normal(0, 5)
            layers["scope"] = np.hypot(u - cx, v - cy) > rad
            info["scope_radius_frac"] = round(rad / W, 2)
        tools = np.zeros((H, W), np.uint8)
        n = rng.choice(3, p=c["tools"])
        for _ in range(n):
            side = rng.integers(4)
            p0 = [(rng.uniform(0, W), 0), (W, rng.uniform(0, H)), (rng.uniform(0, W), H), (0, rng.uniform(0, H))][side]
            ang = np.arctan2(H / 2 - p0[1], W / 2 - p0[0]) + np.radians(rng.uniform(-50, 50))
            L = rng.uniform(0.3, 0.8) * np.hypot(W, H)
            p1 = (p0[0] + L * np.cos(ang), p0[1] + L * np.sin(ang))
            w = rng.uniform(*c["tool_w"])
            nrm = np.array([-np.sin(ang), np.cos(ang)])
            quad = np.array([np.array(p0) + nrm * w * 0.6, np.array(p1) + nrm * w * 0.35, np.array(p1) - nrm * w * 0.35, np.array(p0) - nrm * w * 0.6])
            cv2.fillPoly(tools, [quad.astype(np.int32)], 1)                  # tapers away from the camera
            cv2.circle(tools, tuple(np.int32(p1)), int(w * 0.45), 1, -1)
        if n:
            layers["tool"] = tools > 0
        info["tools"] = int(n)
        edge = S & ~(cv2.erode(S.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
        ep = np.argwhere(edge)
        yc, xc = np.argwhere(S).mean(0) if S.any() else (H / 2, W / 2)
        if rng.random() < c["fat"] and len(ep):
            # fat / omentum: one or two smooth patches lying over the liver's edge, reaching in from outside
            fat = np.zeros((H, W), np.uint8)
            for _ in range(rng.integers(1, 3)):
                y0, x0 = ep[rng.integers(len(ep))]
                out = np.array([x0 - xc, y0 - yc]); out /= np.linalg.norm(out) + 1e-9      # pointing away from the liver
                tang = np.degrees(np.arctan2(out[1], out[0])) + 90
                for _ in range(rng.integers(2, 5)):
                    a, b = rng.uniform(15, 50), rng.uniform(10, 32)                  # along / across the edge, px
                    sh = out * rng.uniform(-0.3, 1.0) * b + rng.normal(0, 12, 2)
                    cv2.ellipse(fat, (int(x0 + sh[0]), int(y0 + sh[1])), (int(a), int(b)), tang + rng.normal(0, 20), 0, 360, 1, -1)
            fat = cv2.GaussianBlur(fat.astype(np.float32), (0, 0), 6) > 0.5
            layers["fat"] = fat
            info["fat_on_liver_frac"] = round(float((fat & S).sum() / max(S.sum(), 1)), 2)
        if rng.random() < c["ligament"] and len(ep):
            # ligament: a band entering at the liver's edge and crossing part of it, gently curved
            lig = np.zeros((H, W), np.uint8)
            y0, x0 = ep[rng.integers(len(ep))]
            ang = np.arctan2(yc - y0, xc - x0) + np.radians(rng.normal(0, 25)); bend = rng.uniform(-0.003, 0.003)
            L = rng.uniform(0.3, 0.7) * np.hypot(*np.ptp(np.argwhere(S), 0))
            s_ = np.linspace(-20, L, 80)
            xs = x0 + s_ * np.cos(ang) - bend * s_ ** 2 * np.sin(ang); ys = y0 + s_ * np.sin(ang) + bend * s_ ** 2 * np.cos(ang)
            cv2.polylines(lig, [np.stack([xs, ys], 1).astype(np.int32)], False, 1, int(rng.uniform(*c["ligament_w"])))
            layers["ligament"] = lig > 0
            info["ligament"] = True
        return layers, info

    # ---------------- 5-6. edges, mask wobble ----------------
    @staticmethod
    def edges(S, vis, layers):
        """Edge pixels of the visible liver: 1 real (liver ends), 2 covered (liver goes on underneath), 3 border."""
        out = np.zeros((H, W), np.uint8)
        hidden = np.zeros((H, W), bool)
        for k in ("tool", "fat", "ligament"):
            if k in layers:
                hidden |= layers[k]
        scope = layers.get("scope", np.zeros((H, W), bool))
        pad = lambda a, val: np.pad(a, 1, constant_values=val)  # noqa: E731
        Vp, Sp, Hp, Cp = pad(vis, False), pad(S, False), pad(hidden, False), pad(scope, True)
        outside = np.zeros((H + 2, W + 2), bool); outside[0] = outside[-1] = True; outside[:, 0] = outside[:, -1] = True
        rim = vis & ~(cv2.erode(vis.astype(np.uint8), np.ones((3, 3), np.uint8), borderValue=0) > 0)
        for y, x in np.argwhere(rim):
            nb = (slice(y, y + 3), slice(x, x + 3))
            gone = ~Vp[nb]
            if (outside[nb] & gone).any() or (Cp[nb] & gone).any():
                out[y, x] = 3
            elif (gone & Sp[nb]).any():
                out[y, x] = 2
            else:
                out[y, x] = 1
        return out

    def wobble(self, M, rng):
        a = rng.uniform(0, self.cfg["wobble_px"])
        if a < 0.3:
            return M
        dx = gaussian_filter(rng.standard_normal((H, W)), 10); dy = gaussian_filter(rng.standard_normal((H, W)), 10)
        dx *= a / (np.abs(dx).max() + 1e-9); dy *= a / (np.abs(dy).max() + 1e-9)
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        return cv2.remap(M.astype(np.uint8), (u + dx).astype(np.float32), (v + dy).astype(np.float32), cv2.INTER_NEAREST) > 0

    # ---------------- the whole example ----------------
    def sample(self, p, rng, tries=30):
        lv = self.livers[p]
        Vd, inside, binfo, D, sigma = self.bend(lv, rng)
        for _ in range(tries):
            R, t, K, cinfo = self.camera(lv, Vd, rng)
            S, Z, Kc, scene = self.draw(Vd, lv.T, R, t, K)
            if S.mean() < self.cfg["min_visible"] or (Z[S] < 15).any():
                continue
            occ = scene.compute_occupancy(o3d.core.Tensor((-R.T @ t)[None].astype(np.float32))).numpy()[0]
            if occ > 0.5:                                                    # camera inside the liver
                continue
            layers, oinfo = self.cover(S, rng)
            vis = S.copy()
            for k in layers:
                vis &= ~layers[k]
            if vis.mean() < self.cfg["min_visible"]:
                continue
            E = self.edges(S, vis, layers)
            M = self.wobble(vis, rng)                                        # segmentation-like edge error
            if "scope" in layers:
                M &= ~layers["scope"]                                        # never into the black outside the scope
            kind = str(rng.choice(self.cfg["depth_kinds"]))
            level = float(rng.uniform(*self.cfg["depth_deg"]))
            Zd, got = distort(Z, S, Kc, kind, level, int(rng.integers(1 << 30)))
            Zr = np.where(M, Zd, 0.0)
            Zr = np.where(M, Zr / np.median(Zr[M]), 0.0)
            cam_inside = inside @ R.T + t
            ne = np.bincount(E[E > 0], minlength=4)[1:]
            return dict(patient=p, R=R, t=t, K=K, S=S, Z=Z, M=M, E=E, depth=Zr, layers=layers, Vd=Vd, D=D, sigma=sigma,
                        inside_cam=cam_inside,
                        info=dict(**cinfo, **binfo, **oinfo, depth_kind=kind, depth_tilt_err_deg=round(float(got), 1),
                                  visible_frac=round(float(M.mean()), 3),
                                  edge_real=int(ne[0]), edge_covered=int(ne[1]), edge_border=int(ne[2])))
        raise RuntimeError("no valid view after many tries")


# TODO before the real run (the dev-set rule: every simulated effect needs published backing for existing AND its size):
#   camera cone / distances / lenses, tool widths and counts, fat and ligament coverage, bend size (pneumoperitoneum
#   deformation literature), depth damage mix and size (SERV-CT measurement), mask wobble (segmentation error studies).
