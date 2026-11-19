"""
Does the shape term protect the search from wrong poses with a perfect outline, and up to how much shape error?

Same 12 front-view photos as goal_landscape.py phase 3 (truth known). All-around guesses: 256 camera directions x 16
rolls; each guess is first matched to the mask's SIZE (distance from the outline area: area ~ 1/distance^2) and CENTRE
(slid sideways), then scored. The 8 best guesses (different directions) are polished by Nelder-Mead and the lowest
final score wins. Compared:
    outline only      J = A                      (A = A1 + A2, px at 1920-wide scale)
    outline + shape   J = A / 4.5 + B / 0.02     weights fixed beforehand from the one-at-a-time sweeps:
                                                 10 % distance error ~ 4.5 px of A, 10 deg rotation ~ 0.02 of B
The "photo" surface directions are tilted by a SMOOTH random field (like a depth model's error, not pixel noise),
scaled so the mean tilt is 0, 5, 10, 15, 20 or 30 deg (same field shape per view, only its size changes).
Success = final pose within 10 deg of the true rotation and 5 % of the true distance.
Usage: python goal/shape_protection.py      -> goal/out/shape_protection.json + .txt
"""
import json
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize

sys.path.insert(0, "goal")
from goal_landscape import OUT, RH, RW, DOWN, Patient, look_at, normals, perturbed, render  # noqa: E402

LEVELS = [0, 5, 10, 15, 20, 30]
WA, WB = 4.5, 0.02
SW, SH, SDOWN = RW // 2, RH // 2, DOWN * 2          # coarse size for the all-around grid


def tilt_field(Nm, ok, rng):
    e = gaussian_filter(rng.standard_normal(Nm.shape), sigma=(Nm.shape[1] / 10, Nm.shape[1] / 10, 0))
    e /= np.sqrt((e[ok] ** 2).sum(-1).mean()) + 1e-12

    def tilted(deg):
        if deg == 0:
            return Nm
        lo, hi = 0.0, 10.0
        for _ in range(40):                          # scale so the mean angle is `deg`
            s = (lo + hi) / 2
            n = Nm + s * e
            n /= np.linalg.norm(n, axis=-1, keepdims=True)
            ang = np.degrees(np.arccos(np.clip((n[ok] * Nm[ok]).sum(-1), -1, 1))).mean()
            lo, hi = (s, hi) if ang < deg else (lo, s)
        return n
    return tilted


class Photo:
    """Mask, distance map, normals (one per tilt level) at one render size."""
    def __init__(self, S0, Z0, K, normals_by_level, down):
        self.M, self.K, self.down = S0, K, down
        self.dtM = cv2.distanceTransform((~S0).astype(np.uint8), cv2.DIST_L2, 5)
        _, self.ok = normals(Z0, K)
        self.N = normals_by_level


def terms(pat, R, t, ph, W, H):
    S, Z = render(pat.scene, R, t, ph.K, W, H)
    if not S.any():
        return np.hypot(W, H) * ph.down, {lv: 1.0 for lv in ph.N}
    dtS = cv2.distanceTransform((~S).astype(np.uint8), cv2.DIST_L2, 5)
    A = (dtS[ph.M].mean() + ph.dtM[S].mean()) * ph.down
    Ns, oks = normals(Z, ph.K)
    both = ph.M & S & ph.ok & oks
    B = {lv: (float((1 - (Ns[both] * n[both]).sum(-1)).mean()) if both.sum() > 20 else 1.0) for lv, n in ph.N.items()}
    return A, B


def J(A, B, cfg):
    return A if cfg == "outline" else A / WA + B[cfg] / WB


def polish(pat, R0, t0, ph, cfg):
    c0 = R0 @ pat.centre + t0

    def pose(x):
        return perturbed(R0, t0, c0, dR=cv2.Rodrigues(np.radians(x[:3]))[0], dt=x[3:])

    def f(x):
        A, B = terms(pat, *pose(x), ph, RW, RH)
        return J(A, B, cfg)

    simplex = np.vstack([np.zeros(6)] + [np.eye(6)[i] * (10 if i < 3 else 20) for i in range(6)])
    r = minimize(f, np.zeros(6), method="Nelder-Mead", options=dict(initial_simplex=simplex, maxfev=300, xatol=0.1, fatol=1e-3))
    return (*pose(r.x), r.fun)


def size_and_centre(pat, R, t, ph, W, H):
    """Move a guess along its line of sight to match the mask's area, then sideways to match its centre."""
    S, _ = render(pat.scene, R, t, ph.K, W, H)
    if S.sum() < 20:
        return R, t
    c = R @ pat.centre + t
    k = np.sqrt(S.sum() / ph.M.sum())
    c2 = c * k
    vS, uS = np.argwhere(S).mean(0); vM, uM = np.argwhere(ph.M).mean(0)
    c2 = c2 + np.array([(uM - uS) * c2[2] / ph.K[0, 0], (vM - vS) * c2[2] / ph.K[1, 1], 0.0])
    return R, t + (c2 - c)


def run_patient(p):
    scan = json.load(open(f"{OUT}/landscape.json"))
    rng = np.random.default_rng(int(p))
    cfgs = ["outline"] + LEVELS
    rows = []
    n = 256
    i = np.arange(n) + 0.5
    phi, th = np.arccos(1 - 2 * i / n), np.pi * (1 + 5 ** 0.5) * i
    dirs = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
    pat = Patient(p)
    for v in scan["patients"][p].get("phase3", []):
        t_start = time.time()
        k = v["view"]
        R, t = pat.true_pose(k)
        K = pat.real_k
        Ks = np.diag([0.5, 0.5, 1.0]) @ K
        S0, Z0 = render(pat.scene, R, t, K, RW, RH)
        N0, ok0 = normals(Z0 * rng.uniform(0.2, 5.0), K)
        tilted = tilt_field(N0, ok0, rng)
        Nl = {lv: tilted(lv) for lv in LEVELS}
        fine = Photo(S0, Z0, K, Nl, DOWN)
        S0s, Z0s = render(pat.scene, R, t, Ks, SW, SH)
        cN = {}
        for lv, n_ in Nl.items():
            m = cv2.resize(n_, (SW, SH), interpolation=cv2.INTER_AREA)
            cN[lv] = m / (np.linalg.norm(m, axis=-1, keepdims=True) + 1e-12)
        coarse = Photo(S0s, Z0s, Ks, cN, SDOWN)
        dist = np.linalg.norm(-R.T @ t - pat.centre)
        truth_c = R @ pat.centre + t
        grid = []
        for di, d in enumerate(dirs):
            for roll in np.arange(0, 360, 22.5):
                Rg, tg = size_and_centre(pat, *look_at(pat.centre, d, dist, roll), coarse, SW, SH)
                A, B = terms(pat, Rg, tg, coarse, SW, SH)
                grid.append((di, Rg, tg, A, B))
        res = dict(patient=p, view=k)
        for cfg in cfgs:
            order = sorted(grid, key=lambda g: J(g[3], g[4], cfg))
            starts, used = [], set()
            for g in order:
                if g[0] not in used:
                    used.add(g[0]); starts.append(g)
                if len(starts) == 8:
                    break
            best = None
            for _, Rg, tg, _, _ in starts:
                Rf, tf, fval = polish(pat, Rg, tg, fine, cfg)
                if best is None or fval < best[2]:
                    best = (Rf, tf, fval)
            Rf, tf, fval = best
            rot_err = float(np.degrees(np.linalg.norm(cv2.Rodrigues(Rf @ R.T)[0])))
            dist_ratio = float(np.linalg.norm(Rf @ pat.centre + tf) / np.linalg.norm(truth_c))
            A, B = terms(pat, Rf, tf, fine, RW, RH)
            At, Bt = terms(pat, R, t, fine, RW, RH)
            res[str(cfg)] = dict(rot_err=rot_err, dist_ratio=dist_ratio, J=float(fval), A=float(A),
                                 B=float(B[cfg] if cfg != "outline" else B[0]), J_truth=float(J(At, Bt, cfg)),
                                 success=bool(rot_err <= 10 and abs(dist_ratio - 1) <= 0.05))
        rows.append(res)
        print(f"{p}/{k} ({time.time() - t_start:.0f} s): " + "  ".join(
            f"{c}:{'OK' if res[str(c)]['success'] else 'x'}({res[str(c)]['rot_err']:.0f}deg,x{res[str(c)]['dist_ratio']:.2f},"
            f"J {res[str(c)]['J']:.2f} vs truth {res[str(c)]['J_truth']:.2f})" for c in cfgs), flush=True)
    return rows


def main():
    with Pool(4) as pool:
        rows = [r for part in pool.map(run_patient, ["01", "06", "10", "11"]) for r in part]
    json.dump(rows, open(f"{OUT}/shape_protection.json", "w"), indent=1)
    cfgs = ["outline"] + LEVELS
    lines = [f"success (within 10 deg and 5 percent distance) out of {len(rows)} views:"]
    for c in cfgs:
        s = sum(r[str(c)]["success"] for r in rows)
        fooled = sum((not r[str(c)]["success"]) and r[str(c)]["J"] < r[str(c)]["J_truth"] for r in rows)
        lines.append(f"  {('outline only' if c == 'outline' else f'outline + shape, tilt {c:>2} deg'):30s} {s:2d}/{len(rows)}   "
                     f"failures where a wrong pose scores BETTER than the truth: {fooled}; others = search never reached the truth")
    open(f"{OUT}/shape_protection.txt", "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
