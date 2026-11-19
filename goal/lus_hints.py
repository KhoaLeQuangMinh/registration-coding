"""
VLM-hinted registration on all 46 LLR-LUS images (compare with the no-hint run of lus_all.py).

Hints = goal/vlm/lus_suggestions.json (written from photo + mask only, frozen before this ran; its sha256 is printed).
Everything except the STARTS is identical to lus_all.py: automatic edge rule, goal J = A/4.5 + B/0.02, Nelder-Mead
polish of the 8 best starts. Starts per suggestion: the 6-part class camera (show_classes.py) for its part and head
direction, then directions within 30 deg x spins within 45 deg, each size + centre matched.
Reported per image and as the official score:
  A  hints only:          the lowest-J pose among the 3 suggestions
  B  hints + no hint:     the lowest-J pose among the 3 suggestions and the no-hint winner (propose and verify)
  upper bound:            the suggestion with the lowest TUMOUR error (uses the ground truth: a ceiling, not a method)
Usage: python goal/lus_hints.py [--workers 4]
       HINT_TAG=_outline python goal/lus_hints.py --outline-only     (score = outline only)
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np
import open3d as o3d

sys.path.insert(0, "goal"); sys.path.insert(0, "studies/benchmark")
import lus_all as la  # noqa: E402
import lus_frame as lf  # noqa: E402
import real_frame_reg as rf  # noqa: E402
import show_classes as sc  # noqa: E402
from goal_landscape import look_at  # noqa: E402

OUT = "dataset/lus-evaluation/goal_results"
TAG = os.environ.get("HINT_TAG", "")
SUGG = "goal/vlm/lus_suggestions.json"
_L = {}


def labels():
    if _L.get("p") != la._C["p"]:
        c = la._C
        V, T = c["V"], c["T"]
        m = c["m"]
        m.compute_triangle_normals()
        Nt = np.asarray(m.triangle_normals); C = V[T].mean(1)
        if ((C - V.mean(0)) * Nt).sum(1).mean() < 0:
            Nt = -Nt
        _L.update(p=c["p"], C=C, Nt=Nt, lab=sc.parts(V, C, Nt, T, m))
    return _L


def starts_for(part, head):
    c, L = la._C, labels()
    R0, _, target = sc.class_pose(c["V"], c["T"], L["C"], L["Nt"], L["lab"], sc.PARTS.index(part), rf.HEADS[head])
    out = []
    for d in rf.fib(400):
        if np.degrees(np.arccos(np.clip(d @ -R0[2], -1, 1))) <= 30:
            best, bang = 0, 1e9
            for r_ in np.arange(0, 360, 5):
                Rr, _ = look_at(target, d, 150.0, r_)
                a_ = np.degrees(np.arccos(np.clip(Rr[1] @ R0[1], -1, 1)))
                if a_ < bang:
                    best, bang = r_, a_
            for spin in np.arange(-45, 46, 15):
                out.append(look_at(target, d, 150.0, best + spin))
    return out


def outline_only():
    """Score = outline term only (A / 4.5); the depth map is never used."""
    import cv2
    from goal_landscape import render

    def terms(scene, R, t, fr):
        S, _ = render(scene, R, t, fr.K, rf.W, rf.H)
        if not S.any():
            return 1e3, 0.0
        Sv = S & fr.V
        dtS = cv2.distanceTransform((~S).astype(np.uint8), cv2.DIST_L2, 5)
        return (dtS[fr.M].mean() + (fr.dtM[Sv].mean() if Sv.any() else 0)) * rf.DOWN, 0.0
    rf.terms = terms
    rf.J = lambda A, B: A / rf.WA


def run_image(job):
    image, align, sugg, mode = job
    if mode == "outline":
        outline_only()
    p, name = image.split("/")
    la.setup(p, align)
    c = la._C
    t0 = time.time()
    fr = la.AutoFrame(image)
    centre = c["V"].mean(0)
    res = dict(image=image, suggestions=[])
    for s in sugg:
        scored = []
        for R, t in starts_for(s["part"], s["head"]):
            R, t = rf.size_and_centre(c["scene"], centre, R, t, fr)
            scored.append((rf.J(*rf.terms(c["scene"], R, t, fr)), R, t))
        scored.sort(key=lambda x: x[0])
        pol = sorted((rf.polish(c["scene"], R, t, centre, fr) for _, R, t in scored[:8]), key=lambda x: x[2])
        Rb, tb, Jb = pol[0]
        res["suggestions"].append(dict(part=s["part"], head=s["head"], p=s["p"], J=float(Jb), R=Rb.tolist(), t=tb.tolist(),
                                       **lf.tumour_error(c["tum"], Rb, tb, fr.profile)))
    nh = json.load(open(f"{OUT}/per_image/{p}_{name}.json"))
    res["no_hint"] = dict(J=nh["J"], R=nh["R"], t=nh["t"], **nh["rigid"])
    if mode == "outline":   # judge the no-hint pose by the same outline-only score
        res["no_hint"]["J"] = float(rf.J(*rf.terms(c["scene"], np.array(nh["R"]), np.array(nh["t"]), fr)))
    A = min(res["suggestions"], key=lambda x: x["J"])
    B = min(res["suggestions"] + [res["no_hint"]], key=lambda x: x["J"])
    U = min(res["suggestions"], key=lambda x: x["tumour_mm"])
    res.update(A=dict(tumour_mm=A["tumour_mm"], depth_mm=A["depth_mm"], J=A["J"], R=A["R"], t=A["t"], from_suggestion=A.get("part", "") + " / " + A.get("head", "")),
               B=dict(tumour_mm=B["tumour_mm"], depth_mm=B["depth_mm"], J=B["J"], R=B["R"], t=B["t"], picked="no hint" if B is res["no_hint"] else B["part"] + " / " + B["head"]),
               upper_bound=dict(tumour_mm=U["tumour_mm"], suggestion=U["part"] + " / " + U["head"]),
               seconds=round(time.time() - t0))
    for variant in ["A", "B"]:
        R, t = np.array(res[variant]["R"]), np.array(res[variant]["t"])
        cam = c["tum"] @ R.T + t
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(cam @ la.F_GL.T), o3d.utility.Vector3iVector(c["tum_faces"]))
        mesh.compute_triangle_normals()
        os.makedirs(f"{OUT}/hint{TAG}_{variant}/{p}", exist_ok=True)
        o3d.io.write_triangle_mesh(f"{OUT}/hint{TAG}_{variant}/{p}/{name}.stl", mesh)
    os.makedirs(f"{OUT}/hint{TAG}_per_image", exist_ok=True)
    json.dump(res, open(f"{OUT}/hint{TAG}_per_image/{p}_{name}.json", "w"), indent=1)
    print(f"{image}: no hint {res['no_hint']['tumour_mm']:5.1f} mm | A {res['A']['tumour_mm']:5.1f} | B {res['B']['tumour_mm']:5.1f} (picked {res['B']['picked']}) | "
          f"per suggestion " + ", ".join(f"{s['tumour_mm']:.0f} mm J{s['J']:.2f}" for s in res["suggestions"]) + f" vs no-hint J{res['no_hint']['J']:.2f}  [{res['seconds']} s]", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--outline-only", action="store_true", help="score = outline only, no depth shape")
    a = ap.parse_args()
    mode = "outline" if a.outline_only else "full"
    if mode == "outline" and not TAG:
        print("set HINT_TAG=_outline so the results do not overwrite the full-score run"); return
    raw = open(SUGG).read()
    print("suggestions sha256", hashlib.sha256(raw.encode()).hexdigest(), flush=True)
    sugg = json.loads(raw)["images"]
    align = json.load(open(f"{OUT}/align.json"))
    images = la.le.images()
    with Pool(a.workers) as pool:
        rows = pool.map(run_image, [(i, align, sugg[i]["suggestions"], mode) for i in images], chunksize=3)
    for variant in ["A", "B"]:
        print(f"\n== official LLR-LUS score, hints {'only' if variant == 'A' else '+ no hint (propose and verify)'} ==", flush=True)
        subprocess.run([sys.executable, "studies/benchmark/lus_evaluation.py", "--method-dir", f"{OUT}/hint{TAG}_{variant}"], check=False)
    print("\nper patient (median tumour error, mm; within 10 mm):")
    for p in ["Patient1", "Patient2", "Patient3", "Patient4"]:
        rs = [r for r in rows if r["image"].startswith(p)]
        f = lambda k: [r[k]["tumour_mm"] for r in rs]  # noqa: E731
        print(f"  {p}: no hint {np.median(f('no_hint')):5.1f} ({sum(x < 10 for x in f('no_hint'))}/{len(rs)}) | A {np.median(f('A')):5.1f} ({sum(x < 10 for x in f('A'))}) | "
              f"B {np.median(f('B')):5.1f} ({sum(x < 10 for x in f('B'))}) | upper bound {np.median(f('upper_bound')):5.1f} ({sum(x < 10 for x in f('upper_bound'))})")


if __name__ == "__main__":
    main()
