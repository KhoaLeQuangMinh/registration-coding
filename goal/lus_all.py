"""
Our pipeline on all 46 LLR-LUS images: NO hints, AUTOMATIC edge rule (lus_edges.hidden_regions without hand boxes).

Per image: undistorted photo + the user's manual liver mask + depth SHAPE from the Self-P2IR-format cloud + CT aligned
to body axes (once per patient) -> all-around search (256 directions x 16 spins, size + centre matched, best 8 polished;
goal J = A/4.5 + B/0.02) -> light bending (stiff cage) -> tumour placed in the camera frame.
Also recorded: AMBIGUITY = the best polished pose more than 30 deg away from the winner, its score ratio to the winner
(near 1 = two very different poses explain the image almost equally) and its tumour error.
Writes dataset/lus-evaluation/goal_results/{rigid,bent}/PatientN/<image>.stl (OpenGL camera axes, as Evaluation.m wants),
per-image JSON, then runs the official score (studies/benchmark/lus_evaluation.py) on both.
Usage: python goal/lus_all.py [--workers 4]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np
import open3d as o3d

sys.path.insert(0, "goal"); sys.path.insert(0, "studies/benchmark")
import lus_edges as le  # noqa: E402
import lus_frame as lf  # noqa: E402
import real_frame_reg as rf  # noqa: E402
import deform_frame as dfm  # noqa: E402
from goal_landscape import look_at, normals  # noqa: E402

OUT = "dataset/lus-evaluation/goal_results"
W, H = rf.W, rf.H
F_GL = np.diag([1.0, -1.0, -1.0])


class AutoFrame(lf.LusFrame):
    """LusFrame, but the hidden area comes from the automatic edge rule (no VLM boxes)."""
    def __init__(self, image):
        super().__init__(image, {})
        img, M = le.load(image)
        scope, cover = le.hidden_regions(img, M, {})
        hidden = cv2.resize((scope | cover).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        self.V = ~hidden | self.M


_C = {}


def setup(p, align):
    if _C.get("p") == p:
        return
    a = align[p]
    m, V, T, C, Nt, tum = lf.load_ct(p, np.array(a["R"]), np.array(a["c"]))
    scene = o3d.t.geometry.RaycastingScene(); scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
    tf = np.asarray(o3d.io.read_triangle_mesh(f"{lf.DSET}/{p}/Preoperative Model/Tumour.ply").triangles)
    _C.clear(); _C.update(p=p, m=m, V=V, T=T, tum=tum, tum_faces=tf, scene=scene)


def rot_deg(Ra, Rb):
    return float(np.degrees(np.linalg.norm(cv2.Rodrigues(Ra @ Rb.T)[0])))


def run_image(job):
    image, align = job
    p, name = image.split("/")
    setup(p, align)
    c = _C
    t0 = time.time()
    fr = AutoFrame(image)
    centre = c["V"].mean(0)
    scored = []
    for d in rf.fib(256):
        for roll in np.arange(0, 360, 22.5):
            R, t = rf.size_and_centre(c["scene"], centre, *look_at(centre, d, 180.0, roll), fr)
            scored.append((rf.J(*rf.terms(c["scene"], R, t, fr)), R, t))
    scored.sort(key=lambda s: s[0])
    pol = [rf.polish(c["scene"], R, t, centre, fr) for _, R, t in scored[:8]]
    pol.sort(key=lambda x: x[2])
    Rb, tb, Jb = pol[0]
    alt = next((x for x in pol[1:] if rot_deg(x[0], Rb) > 30), None)
    A, B = rf.terms(c["scene"], Rb, tb, fr)
    lf._G.clear(); lf._G.update(fr=fr, V=c["V"], T=c["T"], tum=c["tum"])
    Vd, tum_d, bent = lf.bend(Rb, tb, 4.0)
    res = dict(image=image, seconds=None, J=float(Jb), A=float(A), B=float(B), distance_mm=float(np.linalg.norm(Rb @ centre + tb)),
               rigid=lf.tumour_error(c["tum"], Rb, tb, fr.profile), bent=bent, R=Rb.tolist(), t=tb.tolist())
    if alt is not None:
        res["ambiguity"] = dict(score_ratio=float(alt[2] / Jb), rot_from_winner_deg=rot_deg(alt[0], Rb),
                                tumour_mm=lf.tumour_error(c["tum"], alt[0], alt[1], fr.profile)["tumour_mm"])
    for variant, tum in [("rigid", c["tum"]), ("bent", tum_d)]:
        cam = tum @ Rb.T + tb
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(cam @ F_GL.T), o3d.utility.Vector3iVector(c["tum_faces"]))
        mesh.compute_triangle_normals()
        os.makedirs(f"{OUT}/{variant}/{p}", exist_ok=True)
        o3d.io.write_triangle_mesh(f"{OUT}/{variant}/{p}/{name}.stl", mesh)
    res["seconds"] = round(time.time() - t0)
    os.makedirs(f"{OUT}/per_image", exist_ok=True)
    json.dump(res, open(f"{OUT}/per_image/{p}_{name}.json", "w"), indent=1)
    amb = res.get("ambiguity")
    print(f"{image}: tumour {res['rigid']['tumour_mm']:5.1f} mm rigid / {bent['tumour_mm']:5.1f} bent "
          f"(depth {res['rigid']['depth_mm']:+6.1f}), J {Jb:.2f}"
          + (f", runner-up {amb['rot_from_winner_deg']:.0f} deg away at x{amb['score_ratio']:.2f} score ({amb['tumour_mm']:.0f} mm)" if amb else "")
          + f"  [{res['seconds']} s]", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    apath = f"{OUT}/align.json"
    align = json.load(open(apath)) if os.path.exists(apath) else {}
    for p in ["Patient1", "Patient2", "Patient3", "Patient4"]:
        if p not in align:
            R, c, fit, q = lf.body_axes(p)
            align[p] = dict(R=R.tolist(), c=c.tolist(), fit_mm=float(fit), ref=q)
            print(f"{p}: CT aligned to body axes via P2I-LReg {q} (fit {fit:.1f} mm)", flush=True)
            json.dump(align, open(apath, "w"), indent=1)
    images = le.images()
    done = {f"{os.path.basename(f)[:-5].replace('_', '/', 1)}" for f in os.listdir(f"{OUT}/per_image")} if os.path.exists(f"{OUT}/per_image") else set()
    todo = [i for i in images if i not in done]
    print(f"{len(todo)} of {len(images)} images to run", flush=True)
    # keep each worker on one patient as long as possible (the CT setup is cached per worker)
    with Pool(a.workers) as pool:
        pool.map(run_image, [(i, align) for i in todo], chunksize=3)
    for variant in ["rigid", "bent"]:
        print(f"\n== official LLR-LUS score, {variant} ==", flush=True)
        subprocess.run([sys.executable, "studies/benchmark/lus_evaluation.py", "--method-dir", f"{OUT}/{variant}"], check=False)


if __name__ == "__main__":
    main()
