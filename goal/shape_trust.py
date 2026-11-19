"""
Which way of comparing SHAPE still finds the true pose when the depth map is wrong by a known amount?

Same 12 synthetic P2I-LReg views as shape_protection.py (truth known; 3 are cut-off close-ups). Nothing from LLR-LUS.

1. Depth made wrong on purpose (from the TRUE depth of the view), three kinds, each scaled by bisection so the mean
   surface-tilt error is 0, 10, 20 or 30 deg:
     bumpy      smooth random dents/bulges added to the depth
     flattened  depth squeezed toward its median (curves get shallower; may not reach 30 deg: the cap is recorded)
     slanted    the whole surface tilted as one plane (wrong focal / perspective)
   then multiplied by a random overall scale (distance is useless, as on real data).
2. Search starts where a correct but COARSE hint would put it: the true camera moved 20 deg around the liver in a
   random direction and spun up to 20 deg; then directions within 30 deg x spins within 45 deg (as lus_hints.py), each
   size + centre matched; the 8 best per goal are polished by Nelder-Mead (300 evaluations); lowest score wins.
   (The perfect-VLM CLASS camera of show_classes.py was tried first, but it can sit 70+ deg from the truth -- it looks
   at the part head-on, real cameras look at it obliquely -- so the truth was outside the search; its distance from
   the truth is still reported for every view.)
3. Goals (all include the outline A/4.5):
     outline   no shape
     tilt      + B/0.02         B = mean(1 - cos) between CT and depth-map surface directions (current goal)
     rank      + Rk/WR          Rk = among pixel pairs whose depth-map depths differ by > 3 % of the median depth,
                                the (soft) fraction the CT puts in the opposite order
     blend     + w B/0.02 + (1-w) Rk/WR,  w = clip(1 - true error / 20 deg, 0, 1)  (PERFECT trust: a ceiling)
   WR is fixed before the search, like 0.02 was: what a 10 deg rotation of the TRUE (undistorted) map costs in Rk.
   With this w the blend equals tilt at 0 deg and rank at >= 20 deg; it is run separately only at 10 deg.
4. Measured: success (within 10 deg and 5 % distance, as before); fooled (a wrong pose scores better than the truth)
   vs missed; ambiguity = of 8 probes (truth tilted 5/10/15/20 deg about 8 in-image axes, size + centre re-matched),
   how many score clearly worse than the truth (by > 0.25 goal units).
Usage: python goal/shape_trust.py [--workers 4] [--smoke]   -> goal/out/shape_trust.{json,txt}
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np
import open3d as o3d
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize

sys.path.insert(0, "goal")
import show_classes as sc  # noqa: E402
from goal_landscape import OUT, RH, RW, DOWN, Patient, look_at, normals, perturbed, render  # noqa: E402

LEVELS = [0, 10, 20, 30]
KINDS = ["bumpy", "flattened", "slanted"]
WA, WB = 4.5, 0.02
TAU = 0.03             # "clearly differ": depth difference > 3 % of the median depth
SOFT = 0.01            # softness of the order test: 1 % of the median rendered depth
MARGIN = 0.25          # "clearly worse" for the ambiguity probes, in goal units
PROBE_ANGLES = [5, 10, 15, 20]
MAXFEV = 300


def mean_angle(Na, oka, Nb, okb):
    both = oka & okb
    return float(np.degrees(np.arccos(np.clip((Na[both] * Nb[both]).sum(-1), -1, 1))).mean()) if both.sum() > 20 else 0.0


def distort(Z0, S0, K, kind, deg, rng_seed):
    """Depth map with mean surface-tilt error `deg` (bisection on the distortion size)."""
    N0, ok0 = normals(Z0, K)
    m = np.median(Z0[S0])
    rng = np.random.default_rng(rng_seed)
    if kind == "bumpy":
        G = gaussian_filter(rng.standard_normal(Z0.shape), sigma=RW / 10)
        G /= G[S0].std()
        make = lambda a: Z0 + a * m * G          # noqa: E731
        lo, hi, grows = 0.0, 1.0, True
    elif kind == "flattened":
        make = lambda k: m + k * (Z0 - m)        # noqa: E731
        lo, hi, grows = 0.0, 1.0, False          # smaller k = flatter = larger error
    else:
        u, v = np.meshgrid(np.arange(RW) + 0.5, np.arange(RH) + 0.5)
        phi = rng.uniform(0, 2 * np.pi)
        P = np.cos(phi) * (u - K[0, 2]) / K[0, 0] + np.sin(phi) * (v - K[1, 2]) / K[1, 1]
        P /= P[S0].std()
        make = lambda a: Z0 * (1 + a * P)        # noqa: E731
        lo, hi, grows = 0.0, 0.5, True

    def err(x):
        Z = np.where(S0, np.maximum(make(x), 0.2 * m), 0.0)
        N, ok = normals(Z, K)
        return mean_angle(N, ok & S0, N0, ok0 & S0), Z

    if deg == 0:
        return Z0.copy(), 0.0
    if not grows:
        e0, Zc = err(0.0)
        if e0 < deg:                             # flattening cannot reach this error: use the flattest map
            return Zc, e0
    for _ in range(30):
        x = (lo + hi) / 2
        e, Z = err(x)
        if (e < deg) == grows:
            lo = x
        else:
            hi = x
    return err((lo + hi) / 2)[::-1]


class View:
    """The photo: true mask + one distorted depth map per (kind, level)."""
    def __init__(self, pat, key, seed):
        self.R, self.t = pat.true_pose(key)
        self.K = pat.real_k
        self.M, Z0 = render(pat.scene, self.R, self.t, self.K, RW, RH)
        self.dtM = cv2.distanceTransform((~self.M).astype(np.uint8), cv2.DIST_L2, 5)
        rng = np.random.default_rng(seed)
        self.photos = {}
        for kind in KINDS:
            for lv in LEVELS:
                if lv == 0 and kind != KINDS[0]:
                    self.photos[(kind, 0)] = self.photos[(KINDS[0], 0)]
                    continue
                Z, achieved = distort(Z0, self.M, self.K, kind, lv, seed * 10 + KINDS.index(kind))
                Z = Z * rng.uniform(0.2, 5.0)
                self.photos[(kind, lv)] = Photo(Z, self.M, self.K, achieved, lv, rng)


class Photo:
    def __init__(self, Z, M, K, achieved, level, rng):
        self.N, self.ok = normals(Z, K)
        self.achieved, self.level = achieved, level
        self.w = float(np.clip(1 - level / 20, 0, 1))
        idx = np.flatnonzero(M.ravel() & (Z.ravel() > 0))
        i, j = rng.choice(idx, 4000), rng.choice(idx, 4000)
        zi, zj = Z.ravel()[i], Z.ravel()[j]
        keep = np.abs(zi - zj) > TAU * np.median(Z[M])
        self.i, self.j, self.sign = i[keep], j[keep], np.sign(zi - zj)[keep]


class Render:
    def __init__(self, pat, R, t, view):
        self.S, self.Z = render(pat.scene, R, t, view.K, RW, RH)
        if not self.S.any():
            self.A = np.hypot(RW, RH) * DOWN
            self.Ns = None
            return
        dtS = cv2.distanceTransform((~self.S).astype(np.uint8), cv2.DIST_L2, 5)
        self.A = (dtS[view.M].mean() + view.dtM[self.S].mean()) * DOWN
        self.Ns, self.oks = normals(self.Z, view.K)
        self.view = view

    def B(self, ph):
        if self.Ns is None:
            return 1.0
        both = self.view.M & self.S & ph.ok & self.oks
        return float((1 - (self.Ns[both] * ph.N[both]).sum(-1)).mean()) if both.sum() > 20 else 1.0

    def Rk(self, ph):
        if self.Ns is None or len(ph.i) < 20:
            return 0.5
        zi, zj = self.Z.ravel()[ph.i], self.Z.ravel()[ph.j]
        both = (zi > 0) & (zj > 0)
        if both.sum() < 20:
            return 0.5
        s = SOFT * np.median(self.Z[self.S])
        x = np.clip(ph.sign[both] * (zi[both] - zj[both]) / s, -30, 30)
        return float((1 / (1 + np.exp(x))).mean())


def J(r, goal, ph, WR):
    a = r.A / WA
    if goal == "outline":
        return a
    if goal == "tilt":
        return a + r.B(ph) / WB
    if goal == "rank":
        return a + r.Rk(ph) / WR
    return a + ph.w * r.B(ph) / WB + (1 - ph.w) * r.Rk(ph) / WR


def size_and_centre(pat, R, t, view):
    S, _ = render(pat.scene, R, t, view.K, RW, RH)
    if S.sum() < 20:
        return R, t
    c = R @ pat.centre + t
    c2 = c * np.sqrt(S.sum() / view.M.sum())
    vS, uS = np.argwhere(S).mean(0); vM, uM = np.argwhere(view.M).mean(0)
    c2 = c2 + np.array([(uM - uS) * c2[2] / view.K[0, 0], (vM - vS) * c2[2] / view.K[1, 1], 0.0])
    return R, t + (c2 - c)


def polish(pat, R0, t0, view, goal, ph, WR):
    c0 = R0 @ pat.centre + t0

    def pose(x):
        return perturbed(R0, t0, c0, dR=cv2.Rodrigues(np.radians(x[:3]))[0], dt=x[3:])

    def f(x):
        return J(Render(pat, *pose(x), view), goal, ph, WR)

    simplex = np.vstack([np.zeros(6)] + [np.eye(6)[i] * (10 if i < 3 else 20) for i in range(6)])
    r = minimize(f, np.zeros(6), method="Nelder-Mead", options=dict(initial_simplex=simplex, maxfev=MAXFEV, xatol=0.1, fatol=1e-3))
    return (*pose(r.x), r.fun)


def true_class(pat, p, R, t, K, cls):
    """What a perfect VLM would say: the most visible part and the head direction (nearest of 8)."""
    m, V, T, C, Nt, lab = cls
    u, v = np.meshgrid(np.arange(RW) + 0.5, np.arange(RH) + 0.5)
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1).reshape(-1, 3)
    dn = d / np.linalg.norm(d, axis=1, keepdims=True)
    rays = np.hstack([np.repeat((-R.T @ t)[None], len(d), 0), dn @ R]).astype(np.float32)
    prim = pat.scene.cast_rays(o3d.core.Tensor(rays))["primitive_ids"].numpy()
    hit = prim != o3d.t.geometry.RaycastingScene.INVALID_ID
    part = int(np.bincount(lab[prim[hit]], minlength=6).argmax())
    ref = np.array([0, 0, 1.0]) if part != 5 else np.array([0, -1.0, 0])
    c = R @ ref
    head = int(round((np.degrees(np.arctan2(c[0], -c[1])) % 360) / 45)) % 8
    return part, head


def class_gap(cls, part, head, R, t):
    """Angle between the perfect-VLM class camera and the true camera, seen from the liver (reported, not used)."""
    m, V, T, C, Nt, lab = cls
    R0, t0, _ = sc.class_pose(V, T, C, Nt, lab, part, head)
    c = V.mean(0)
    a, b = -R0.T @ t0 - c, -R.T @ t - c
    return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def coarse_hint(pat, R, t, rng):
    """A correct but coarse hint: the true view direction tilted 20 deg in a random direction, spin off by up to 20 deg."""
    c = R @ pat.centre + t
    a = rng.uniform(0, 2 * np.pi)
    R1, t1 = perturbed(R, t, c, dR=cv2.Rodrigues(np.radians(20) * np.array([np.cos(a), np.sin(a), 0.0]))[0])
    R1, t1 = perturbed(R1, t1, R1 @ pat.centre + t1, dR=cv2.Rodrigues(np.array([0, 0, np.radians(rng.uniform(-20, 20))]))[0])
    d0 = -R1.T @ t1 - pat.centre                 # direction from the liver centre to the hinted camera
    return R1, d0 / np.linalg.norm(d0)


def starts(d0, up, target):
    """Cameras looking at `target` from within 30 deg of direction d0, roll matched to `up`, then spun +-45 deg."""
    n = 400
    i = np.arange(n) + 0.5
    phi, th = np.arccos(1 - 2 * i / n), np.pi * (1 + 5 ** 0.5) * i
    out = []
    for d in np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1):
        if np.degrees(np.arccos(np.clip(d @ d0, -1, 1))) <= 30:
            best, bang = 0, 1e9
            for r_ in np.arange(0, 360, 5):
                Rr, _ = look_at(target, d, 150.0, r_)
                a_ = np.degrees(np.arccos(np.clip(Rr[1] @ up, -1, 1)))
                if a_ < bang:
                    best, bang = r_, a_
            for spin in np.arange(-45, 46, 15):
                out.append(look_at(target, d, 150.0, best + spin))
    return out


def jobs():
    scan = json.load(open(f"{OUT}/landscape.json"))
    return [(p, v["view"], 1000 * int(p) + n) for p in ["01", "06", "10", "11"] for n, v in enumerate(scan["patients"][p]["phase3"])]


def calibrate(job):
    """Cost of a 10 deg rotation of the TRUE map, in Rk and in B (B should come out near 0.02)."""
    p, key, seed = job
    pat = Patient(p)
    view = View(pat, key, seed)
    ph = view.photos[("bumpy", 0)]
    r0 = Render(pat, view.R, view.t, view)
    c = view.R @ pat.centre + view.t
    dR, dB = [], []
    for ax in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]:
        R, t = perturbed(view.R, view.t, c, dR=cv2.Rodrigues(np.radians(10) * np.array(ax, float))[0])
        r = Render(pat, *size_and_centre(pat, R, t, view), view)
        dR.append(r.Rk(ph) - r0.Rk(ph)); dB.append(r.B(ph) - r0.B(ph))
    return float(np.mean(dR)), float(np.mean(dB))


def goals_for(kind, lv):
    g = ["tilt", "rank"]
    return g + (["blend"] if lv == 10 else [])


def run_view(args):
    job, WR, smoke = args
    p, key, seed = job
    t0 = time.time()
    pat = Patient(p)
    view = View(pat, key, seed)
    m, V, T, C, Nt = sc.load(p)
    cls = (m, V, T, C, Nt, sc.parts(V, C, Nt, T, m))
    part, head = true_class(pat, p, view.R, view.t, view.K, cls)
    cgap = class_gap(cls, part, head, view.R, view.t)
    R0, d0 = coarse_hint(pat, view.R, view.t, np.random.default_rng(seed + 7))
    st = starts(d0, R0[1], pat.centre)
    if smoke:
        st = st[::8]
    true_d = -view.R.T @ view.t - pat.centre
    start_gap = float(np.degrees(np.arccos(np.clip(d0 @ true_d / np.linalg.norm(true_d), -1, 1))))
    rendered = []
    for R, t in st:
        R, t = size_and_centre(pat, R, t, view)
        rendered.append((R, t, Render(pat, R, t, view)))
    truth = Render(pat, view.R, view.t, view)
    truth_c = view.R @ pat.centre + view.t
    probes = {}
    for ang in PROBE_ANGLES:
        probes[ang] = []
        for ax in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
            a = np.array([ax[0], ax[1], 0.0]); a /= np.linalg.norm(a)
            R, t = perturbed(view.R, view.t, truth_c, dR=cv2.Rodrigues(np.radians(ang) * a)[0])
            probes[ang].append(Render(pat, *size_and_centre(pat, R, t, view), view))
    conds = [("outline", "-", 0)] + [(g, k, lv) for k in KINDS for lv in LEVELS for g in goals_for(k, lv)
                                     if not (lv == 0 and k != KINDS[0])]
    res = dict(patient=p, view=key, part=sc.PARTS[part], head=sc.HEAD[head], start_gap_deg=start_gap, n_starts=len(st),
               class_camera_gap_deg=cgap,
               achieved={f"{k}/{lv}": view.photos[(k, lv)].achieved for k in KINDS for lv in LEVELS}, results={})
    global MAXFEV
    if smoke:
        MAXFEV = 40
    for goal, kind, lv in conds:
        ph = view.photos[(KINDS[0] if kind == "-" else kind, lv)]
        scored = sorted(rendered, key=lambda x: J(x[2], goal, ph, WR))[:2 if smoke else 8]
        best = min((polish(pat, R, t, view, goal, ph, WR) for R, t, _ in scored), key=lambda x: x[2])
        Rf, tf, Jf = best
        rot_err = float(np.degrees(np.linalg.norm(cv2.Rodrigues(Rf @ view.R.T)[0])))
        dist_ratio = float(np.linalg.norm(Rf @ pat.centre + tf) / np.linalg.norm(truth_c))
        Jt = J(truth, goal, ph, WR)
        ok = bool(rot_err <= 10 and abs(dist_ratio - 1) <= 0.05)
        worse = {a: int(sum(J(r, goal, ph, WR) > Jt + MARGIN for r in probes[a])) for a in PROBE_ANGLES}
        res["results"][f"{goal}|{kind}|{lv}"] = dict(rot_err=rot_err, dist_ratio=dist_ratio, J=float(Jf), J_truth=float(Jt),
                                                     success=ok, fooled=bool((not ok) and Jf < Jt), probes_worse=worse)
    print(f"{p}/{key} ({sc.PARTS[part]}, {sc.HEAD[head]}, start {start_gap:.0f} deg from truth) {time.time() - t0:.0f} s: " +
          " ".join(f"{k.split('|')[0][0]}{k.split('|')[1][0]}{k.split('|')[2]}:{'OK' if v['success'] else 'x'}{v['rot_err']:.0f}"
                   for k, v in res["results"].items()), flush=True)
    return res


def summary(rows, WR, Bcal):
    n = len(rows)
    L = [f"Shape-trust test: {n} synthetic views, success = within 10 deg and 5 % distance",
         f"WR (10 deg rotation of the true map, in rank units) = {WR:.4f}; same check for tilt B = {Bcal:.4f} (goal uses 0.02)",
         "coarse hint: median {:.0f} deg from the truth (range {:.0f}-{:.0f})".format(
             np.median([r["start_gap_deg"] for r in rows]), min(r["start_gap_deg"] for r in rows), max(r["start_gap_deg"] for r in rows)),
         "for reference, the perfect-VLM CLASS camera's view direction is {} deg from the truth".format(
             ", ".join(f"{r['class_camera_gap_deg']:.0f}" for r in rows)),
         ""]

    def cell(key):
        rs = [r["results"][key] for r in rows]
        s = sum(x["success"] for x in rs); f = sum(x["fooled"] for x in rs)
        w10 = np.mean([x["probes_worse"]["10"] if "10" in x["probes_worse"] else x["probes_worse"][10] for x in rs])
        return f"{s:2d}/{n} (fooled {f:2d}, amb {w10:.1f}/8)"

    L.append(f"outline only: {cell('outline|-|0')}")
    for kind in KINDS:
        ach = [np.median([r["achieved"][f"{kind}/{lv}"] for r in rows]) for lv in LEVELS]
        L.append(f"\n{kind} (achieved mean tilt error: " + ", ".join(f"{lv}->{a:.0f}" for lv, a in zip(LEVELS, ach)) + " deg)")
        for goal in ["tilt", "rank", "blend"]:
            cells = []
            for lv in LEVELS:
                k = f"{goal}|{KINDS[0] if lv == 0 else kind}|{lv}"
                if goal == "blend":
                    k = f"tilt|{KINDS[0]}|0" if lv == 0 else (f"blend|{kind}|10" if lv == 10 else f"rank|{kind}|{lv}")
                cells.append(f"{lv:2d} deg: {cell(k)}")
            L.append(f"  {goal:6s} " + " | ".join(cells))
    L.append("\namb = of 8 poses tilted 10 deg from the truth, how many the goal scores clearly worse (8 = rotation pinned).")
    L.append("blend: w = 1 at 0 deg (= tilt), 0 at >= 20 deg (= rank); only the 10 deg column is a separate run.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="one view, few starts, short polish: check the code runs")
    a = ap.parse_args()
    js = jobs()[:1] if a.smoke else jobs()
    with Pool(a.workers) as pool:
        cal = pool.map(calibrate, js)
        WR = float(np.median([c[0] for c in cal])); Bcal = float(np.median([c[1] for c in cal]))
        print(f"calibration: WR = {WR:.4f}, B for 10 deg = {Bcal:.4f}", flush=True)
        rows = pool.map(run_view, [(j, WR, a.smoke) for j in js], chunksize=1)
    tag = "_smoke" if a.smoke else ""
    json.dump(dict(WR=WR, B_cal=Bcal, rows=rows), open(f"{OUT}/shape_trust{tag}.json", "w"), indent=1)
    txt = summary(rows, WR, Bcal)
    open(f"{OUT}/shape_trust{tag}.txt", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
