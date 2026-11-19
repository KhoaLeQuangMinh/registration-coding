"""
Fake-perfect-VLM test: does knowing WHICH PART of the liver the photo shows help the search find the true pose?

Same 12 photos, shape tilt 10 deg, same search as shape_protection.py (256 directions x 16 rolls, size + centre matched,
best 8 polished by Nelder-Mead on J = A/4.5 + B/0.02, lowest wins). The liver is cut at its centre along the body axes
(P2I-LReg CT axes: x left-right, y front-back, z head-feet): 2 parts (x), 4 parts (x, z), 8 parts (x, y, z).
Hint = share of the visible liver in each part at the TRUE pose (what a perfect VLM would say).
A guess is kept as a start only if its own visible-part shares overlap the hint by >= 50 % (histogram intersection);
polishing is free (the hint only chooses where the search starts).
Conditions: none | perfect 2 | perfect 4 | perfect 8 | top-2 with a WRONG first answer (8 parts: the left-right mirror
of the truth) and the right answer second (guesses consistent with either are kept; the goal decides).
Usage: python goal/part_hint.py   -> goal/out/part_hint.json + .txt
"""
import json
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np
import open3d as o3d

sys.path.insert(0, "goal")
from goal_landscape import OUT, RH, RW, DOWN, Patient, look_at, normals  # noqa: E402
from shape_protection import (SDOWN, SH, SW, J, Photo, polish, size_and_centre, terms, tilt_field)  # noqa: E402

TILT = 10
OVERLAP = 0.5
CONDS = ["none", "perfect2", "perfect4", "perfect8", "top2_wrong_first"]


def part_labels(pat):
    """Part index (0-7) of every mesh triangle: bit0 = left (x > centre), bit1 = head (z > centre), bit2 = back (y > centre)."""
    mesh = o3d.io.read_triangle_mesh(pat_path(pat.p))
    V = np.asarray(mesh.vertices) * 1000.0
    T = np.asarray(mesh.triangles)
    c = V[T].mean(1) - pat.centre
    return (c[:, 0] > 0).astype(int) + 2 * (c[:, 2] > 0) + 4 * (c[:, 1] > 0)


def pat_path(p):
    return f"dataset/liver-registration/{p}/model/reconstructed_mesh_world_m.obj"


def cast_parts(pat, labels, R, t, K, W, H):
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1).reshape(-1, 3)
    dn = d / np.linalg.norm(d, axis=1, keepdims=True)
    rays = np.hstack([np.repeat((-R.T @ t)[None], len(d), 0), dn @ R]).astype(np.float32)
    prim = pat.scene.cast_rays(o3d.core.Tensor(rays))["primitive_ids"].numpy()
    hit = prim != o3d.t.geometry.RaycastingScene.INVALID_ID
    if not hit.any():
        return np.zeros(8)
    return np.bincount(labels[prim[hit]], minlength=8) / hit.sum()


def coarsen(s8, n):
    if n == 8:
        return s8
    if n == 4:
        return s8[:4] + s8[4:]                       # drop front/back
    return np.array([s8[0::2].sum(), s8[1::2].sum()])  # left/right only


def mirror(s8):
    return s8[[1, 0, 3, 2, 5, 4, 7, 6]]                # swap left and right


def overlap(a, b):
    return np.minimum(a, b).sum()


def run_patient(p):
    scan = json.load(open(f"{OUT}/landscape.json"))
    rng = np.random.default_rng(int(p))                # same tilt fields as shape_protection.py
    pat = Patient(p); pat.p = p
    labels = part_labels(pat)
    n = 256
    i = np.arange(n) + 0.5
    phi, th = np.arccos(1 - 2 * i / n), np.pi * (1 + 5 ** 0.5) * i
    dirs = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
    rows = []
    for v in scan["patients"][p].get("phase3", []):
        t0 = time.time()
        k = v["view"]
        R, t = pat.true_pose(k)
        K = pat.real_k
        Ks = np.diag([0.5, 0.5, 1.0]) @ K
        from goal_landscape import render
        S0, Z0 = render(pat.scene, R, t, K, RW, RH)
        N0, ok0 = normals(Z0 * rng.uniform(0.2, 5.0), K)
        tilted = tilt_field(N0, ok0, rng)
        Nl = {lv: tilted(lv) for lv in [0, 5, 10, 15, 20, 30]}      # keep the RNG sequence identical
        fine = Photo(S0, Z0, K, {TILT: Nl[TILT]}, DOWN)
        S0s, Z0s = render(pat.scene, R, t, Ks, SW, SH)
        m = cv2.resize(Nl[TILT], (SW, SH), interpolation=cv2.INTER_AREA)
        coarse = Photo(S0s, Z0s, Ks, {TILT: m / (np.linalg.norm(m, axis=-1, keepdims=True) + 1e-12)}, SDOWN)
        dist = np.linalg.norm(-R.T @ t - pat.centre)
        truth_c = R @ pat.centre + t
        hint8 = cast_parts(pat, labels, R, t, K, RW, RH)
        grid = []
        for di, d in enumerate(dirs):
            for roll in np.arange(0, 360, 22.5):
                Rg, tg = size_and_centre(pat, *look_at(pat.centre, d, dist, roll), coarse, SW, SH)
                A, B = terms(pat, Rg, tg, coarse, SW, SH)
                grid.append((di, Rg, tg, J(A, B, TILT), cast_parts(pat, labels, Rg, tg, Ks, SW, SH)))
        At, Bt = terms(pat, R, t, fine, RW, RH)
        res = dict(patient=p, view=k, hint8=hint8.tolist(), J_truth=float(J(At, Bt, TILT)))
        for cond in CONDS:
            if cond == "none":
                keep = grid
            elif cond.startswith("perfect"):
                nparts = int(cond[-1])
                h = coarsen(hint8, nparts)
                keep = [g for g in grid if overlap(coarsen(g[4], nparts), h) >= OVERLAP]
            else:
                keep = [g for g in grid if overlap(g[4], hint8) >= OVERLAP or overlap(g[4], mirror(hint8)) >= OVERLAP]
            order = sorted(keep, key=lambda g: g[3])
            starts, used = [], set()
            for g in order:
                if g[0] not in used:
                    used.add(g[0]); starts.append(g)
                if len(starts) == 8:
                    break
            best = None
            for _, Rg, tg, _, _ in starts:
                Rf, tf, fval = polish(pat, Rg, tg, fine, TILT)
                if best is None or fval < best[2]:
                    best = (Rf, tf, fval)
            if best is None:
                res[cond] = dict(kept=0, success=False, rot_err=None, dist_ratio=None, J=None)
                continue
            Rf, tf, fval = best
            rot_err = float(np.degrees(np.linalg.norm(cv2.Rodrigues(Rf @ R.T)[0])))
            dist_ratio = float(np.linalg.norm(Rf @ pat.centre + tf) / np.linalg.norm(truth_c))
            res[cond] = dict(kept=len(keep), rot_err=rot_err, dist_ratio=dist_ratio, J=float(fval),
                             success=bool(rot_err <= 10 and abs(dist_ratio - 1) <= 0.05))
        rows.append(res)
        print(f"{p}/{k} ({time.time() - t0:.0f} s) hint {np.round(hint8, 2).tolist()}: " + "  ".join(
            f"{c}:{'OK' if res[c]['success'] else 'x'}({res[c]['kept']} kept"
            + (f", {res[c]['rot_err']:.0f}deg, x{res[c]['dist_ratio']:.2f}, J {res[c]['J']:.2f} vs truth {res['J_truth']:.2f})" if res[c]['rot_err'] is not None else ")")
            for c in CONDS), flush=True)
    return rows


def main():
    with Pool(4) as pool:
        rows = [r for part in pool.map(run_patient, ["01", "06", "10", "11"]) for r in part]
    json.dump(rows, open(f"{OUT}/part_hint.json", "w"), indent=1)
    lines = [f"shape tilt {TILT} deg; success = within 10 deg and 5 percent distance; {len(rows)} views"]
    hard = {("01", "00031"), ("01", "00077"), ("10", "00146")}
    for c in CONDS:
        s = sum(r[c]["success"] for r in rows)
        sh = sum(r[c]["success"] for r in rows if (r["patient"], r["view"]) in hard)
        fooled = sum((not r[c]["success"]) and r[c]["J"] is not None and r[c]["J"] < r["J_truth"] for r in rows)
        kept = np.median([r[c]["kept"] for r in rows])
        lines.append(f"  {c:18s} {s:2d}/{len(rows)}  (close-up views solved {sh}/3; goal fooled {fooled}; "
                     f"guesses kept: median {kept:.0f} of 4096)")
    open(f"{OUT}/part_hint.txt", "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
