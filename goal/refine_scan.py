"""
Fair false-valley check for the all-around scan (goal_landscape.py phase 3).
The scan's grid is coarse (64 directions x 8 rolls x 5 distances), so a grid point near the truth can score badly just
because its roll or distance is off. Here each candidate is polished by a local optimiser (Nelder-Mead on the
outline goal A1 + A2, 6 pose numbers) before comparing:
    near   the best grid point within 20 deg of the true camera direction
    far    the 3 best grid points more than 45 deg away (different directions)
A FALSE VALLEY = a far pose that polishes to an outline score as good as the near one. B (shape) is reported at each
polished pose, to see whether the shape term tells them apart.
Usage: python goal/refine_scan.py   (reads/writes goal/out/)
"""
import json
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, "goal")
from goal_landscape import (OUT, RH, RW, Patient, look_at, normals, perturbed, render, rot, score)  # noqa: E402

import cv2  # noqa: E402


def polish(pat, R0, t0, M, V, Nm, okm, K, dtM):
    c0 = R0 @ pat.centre + t0

    def pose(x):
        dR = cv2.Rodrigues(np.radians(x[:3]))[0]
        return perturbed(R0, t0, c0, dR=dR, dt=x[3:])

    def f(x):
        S, Z = render(pat.scene, *pose(x), K, RW, RH)
        return score(S, Z, M, V, Nm, okm, K, dtM)["A"]

    x0 = np.zeros(6)
    simplex = np.vstack([x0] + [x0 + np.eye(6)[i] * (10 if i < 3 else 20) for i in range(6)])
    r = minimize(f, x0, method="Nelder-Mead", options=dict(initial_simplex=simplex, maxfev=400, xatol=0.1, fatol=0.05))
    R, t = pose(r.x)
    S, Z = render(pat.scene, R, t, K, RW, RH)
    g = score(S, Z, M, V, Nm, okm, K, dtM)
    return R, t, g


def main():
    d = json.load(open(f"{OUT}/landscape.json"))
    lines, rows = [], []
    for p, pv in d["patients"].items():
        pat = Patient(p)
        for v in pv.get("phase3", []):
            k = v["view"]
            R, t = pat.true_pose(k)
            K = pat.real_k
            S0, Z0 = render(pat.scene, R, t, K, RW, RH)
            M, V = S0, np.ones_like(S0)
            Nm, okm = normals(Z0, K)
            dtM = cv2.distanceTransform((~M).astype(np.uint8), cv2.DIST_L2, 5)
            true_dir = -R.T @ t - pat.centre
            dist = np.linalg.norm(true_dir)
            n = len(v["directions"])
            i = np.arange(n) + 0.5
            phi, th = np.arccos(1 - 2 * i / n), np.pi * (1 + 5 ** 0.5) * i
            dirs = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
            D = v["directions"]
            order = np.argsort([r["A"] for r in D])
            near = [j for j in order if D[j]["angle_to_true"] < 20][:1]
            far = [j for j in order if D[j]["angle_to_true"] > 45][:3]
            out = {}
            for tag, js in [("near", near), ("far", far)]:
                for j in js:
                    R0, t0 = look_at(pat.centre, dirs[j], dist * D[j]["dist_factor"], D[j]["roll"])
                    Rf, tf, g = polish(pat, R0, t0, M, V, Nm, okm, K, dtM)
                    cam = -Rf.T @ tf - pat.centre
                    ang = float(np.degrees(np.arccos(np.clip(cam @ true_dir / np.linalg.norm(cam) / dist, -1, 1))))
                    rot_err = float(np.degrees(np.linalg.norm(cv2.Rodrigues(Rf @ R.T)[0])))
                    out.setdefault(tag, []).append(dict(R=Rf.tolist(), t=tf.tolist(), A=g["A"], B=g["B"], dice=1 - g["dice_plain"], view_angle=ang, rot_err=rot_err,
                                                        dist_ratio=float(np.linalg.norm(Rf @ pat.centre + tf) / np.linalg.norm(R @ pat.centre + t))))
            allp = out["near"] + out["far"]
            wrong = [r for r in allp if r["rot_err"] > 30]
            fv = [r for r in wrong if r["A"] <= 2.0]
            verdict = (f"{len(fv)} wrong pose(s) with a near-perfect outline (A <= 2 px); "
                       f"smallest B among them {min(r['B'] for r in fv):.3f} (truth: A 0, B 0)") if fv else "no wrong pose reaches A <= 2 px"
            found = min(allp, key=lambda r: r["A"])
            rows.append(dict(patient=p, view=k, polished=allp, verdict=verdict, best_by_outline_rot_err=found["rot_err"],
                             best_by_outline_B=found["B"]))
            lines.append(f"{p}/{k}: polished poses (rot err deg / dist ratio / A px / Dice / B): " +
                         "; ".join(f"{r['rot_err']:.0f}/{r['dist_ratio']:.2f}/{r['A']:.1f}/{r['dice']:.3f}/{r['B']:.2f}" for r in allp) +
                         f"\n    -> {verdict}; best outline score is {found['rot_err']:.0f} deg off")
            print(lines[-1], flush=True)
    json.dump(rows, open(f"{OUT}/refine_scan.json", "w"), indent=1)
    open(f"{OUT}/refine_scan.txt", "w").write("\n".join(lines))


if __name__ == "__main__":
    main()
