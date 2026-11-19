"""
Build dataset/lus-evaluation/register_playground_lus.html: the registration playground on all 46 LLR-LUS images,
with the TRUE tumour position from the tracked ultrasound probe, so a hand registration can be checked.

Per image, exactly what our LUS runs used (lus_all.AutoFrame): undistorted photo, the user's hand-drawn liver mask,
the automatic hidden-edge rule (scope border, covered edges), the depth SHAPE from the Self-P2IR-format cloud, the CT
liver + tumour aligned to body axes (goal_results/align.json). Truth = the ultrasound tumour profile (camera frame, mm).
Tumour error = mean distance from each profile point to the nearest CT tumour vertex (lus_frame.tumour_error).
Reference poses = our saved results (goal_results/hint_per_image, per_image).

LLR-LUS is research-only: the page stays under dataset/ (not committed, not uploaded). Usage, from the repo root:
    python goal/make_playground_lus.py [--workers 4] && open dataset/lus-evaluation/register_playground_lus.html
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

sys.path.insert(0, "goal"); sys.path.insert(0, "studies/benchmark")
import lus_all as la  # noqa: E402
import lus_best as lb  # noqa: E402
import lus_edges as le  # noqa: E402
import lus_frame as lf  # noqa: E402
import show_classes as sc  # noqa: E402
from make_playground import HERE, W, H, b64, png  # noqa: E402

RES = "dataset/lus-evaluation/goal_results"
OUT = "dataset/lus-evaluation/register_playground_lus.html"


def normals_png(N, ok):
    """Normals (as returned by goal_landscape.normals, pointing away) -> camera-facing RGBA png."""
    N = -N
    rgba = np.zeros((H, W, 4), np.uint8)
    rgba[..., :3] = np.clip((N * 0.5 + 0.5) * 255 + 0.5, 0, 255).astype(np.uint8)
    rgba[..., 3] = (ok * 255).astype(np.uint8)
    return png(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))


def patient_mesh(p, align):
    a = align[p]
    m, V, T, C, Nt, tum = lf.load_ct(p, np.array(a["R"]), np.array(a["c"]))
    lab = sc.parts(V, C, Nt, T, m)
    small = m.simplify_quadric_decimation(12000)
    small.remove_degenerate_triangles(); small.remove_unreferenced_vertices()
    Vs, Ts = np.asarray(small.vertices), np.asarray(small.triangles)
    _, idx = cKDTree(C).query(Vs)
    tf = np.asarray(o3d.io.read_triangle_mesh(f"{lf.DSET}/{p}/Preoperative Model/Tumour.ply").triangles)
    classes = []
    for part in range(6):
        for head in range(8):
            R0, t0, _ = sc.class_pose(V, T, C, Nt, lab, part, head)
            classes.append(dict(part=part, head=head, R=R0.tolist(), t=t0.tolist()))
    # the page measures distance from the centre of ITS (decimated) mesh; keep the full mesh's centre so the
    # reference poses and the size step mean the same as in Python
    return dict(v=b64(Vs.astype(np.float32)), f=b64(Ts.astype(np.uint32)), part=b64(lab[idx].astype(np.uint8)),
                centre=V.mean(0).tolist(), nv=len(Vs), nf=len(Ts), classes=classes,
                tumour=dict(v=b64(tum.astype(np.float32)), f=b64(tf.astype(np.uint32))))


def one_image(image):
    p, name = image.split("/")
    fr = la.AutoFrame(image)
    Hf, Wf = fr.img.shape[:2]
    hint = json.load(open(f"{RES}/hint_per_image/{p}_{name}.json"))
    refs = [dict(name="Our pipeline: VLM hints + goal", R=hint["A"]["R"], t=hint["A"]["t"], mm=hint["A"]["tumour_mm"]),
            dict(name="Our pipeline: no hint", R=hint["no_hint"]["R"], t=hint["no_hint"]["t"], mm=hint["no_hint"]["tumour_mm"])]
    best = min(hint["suggestions"], key=lambda s: s["tumour_mm"])
    refs.append(dict(name=f"Our closest pose (hint '{best['part']}, head {best['head']}')", R=best["R"], t=best["t"], mm=best["tumour_mm"]))
    orc = f"{RES}/oracle_per_image/{p}_{name}.json"
    if os.path.exists(orc):                          # lus_best.py: the registration that uses the answer
        o = json.load(open(orc))
        r_, b_ = o["rigid"], o["bent"]
        refs.insert(0, dict(name="Best possible, rigid (uses the answer)", R=r_["R"], t=r_["t"], mm=r_["tumour_mm"], oracle=True,
                            info=f"probe gap {r_['contact_mm']:.1f} mm, outline {r_['A']:.1f} px, Dice {r_['dice']:.2f}"))
        refs.insert(1, dict(name="Best possible, rigid + bend (uses the answer)", R=b_["R"], t=b_["t"], mm=b_["tumour_mm"], oracle=True,
                            bend=dict(p=b_["handles"], d=b_["D"], sigma=b_["sigma"]),
                            info=f"probe gap {b_['contact_mm']:.1f} mm, outline {b_['A']:.1f} px, Dice {b_['dice']:.2f}, "
                                 f"tissue moved {b_['moved_mean_mm']:.0f} mm on average (max {b_['moved_max_mm']:.0f})"))
    # the same tumour error as lus_best.py and the page: profile -> nearest point of the tumour SURFACE
    # (the saved runs logged the nearest VERTEX, a little larger)
    a = json.load(open(f"{RES}/align.json"))[p]
    _, _, _, _, _, tum = lf.load_ct(p, np.array(a["R"]), np.array(a["c"]))
    tmesh = o3d.t.geometry.TriangleMesh(); tmesh.vertex.positions = o3d.core.Tensor(tum.astype(np.float32))
    tmesh.triangle.indices = o3d.core.Tensor(np.asarray(o3d.io.read_triangle_mesh(f"{lf.DSET}/{p}/Preoperative Model/Tumour.ply").triangles).astype(np.int32))
    tsc = o3d.t.geometry.RaycastingScene(); tsc.add_triangles(tmesh)
    for r in refs:
        if not r.get("oracle"):
            R, t = np.array(r["R"]), np.array(r["t"])
            qb = (fr.profile - t) @ R
            near = tsc.compute_closest_points(o3d.core.Tensor(qb.astype(np.float32)))["points"].numpy()
            r["mm"] = float(np.linalg.norm(near - qb, axis=1).mean())
    face, usq, us = lb.probe_and_plane(p, name)
    usw = 480
    us_small = cv2.resize(us, (usw, round(us.shape[0] * usw / us.shape[1])), interpolation=cv2.INTER_AREA)
    return dict(
        id=image, title=f"{p.replace('Patient', 'Patient ')} · image {name}", patient=p, mesh=p, size=[Wf, Hf],
        note=f"LLR-LUS {p}, image {name}: real surgery photo (lens distortion removed), your hand-drawn liver mask, "
             f"and the tracked ultrasound probe's view of the tumour as the answer key.",
        refName="our pipeline's pose (VLM hints)", refIsTruth=False,
        K=fr.Kfull.tolist(), photo=png(cv2.resize(fr.img, (960, 540), interpolation=cv2.INTER_AREA), ".jpg", 85),
        mask=png((fr.M * 255).astype(np.uint8)), visible=png((fr.V * 255).astype(np.uint8)),
        shapes=[dict(name="Depth Anything (Self-P2IR cloud)", normals=normals_png(fr.N, fr.ok & fr.M))],
        ref=dict(R=refs[0]["R"], t=refs[0]["t"]), refs=refs,
        profile=b64(fr.profile.astype(np.float32)), nprofile=len(fr.profile),
        face=b64(face.astype(np.float32)), usq=usq.tolist(), usimg=png(us_small, ".jpg", 80))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    align = json.load(open(f"{RES}/align.json"))
    images = le.images()
    with Pool(a.workers) as pool:
        scenes = pool.map(one_image, images)
        meshes = dict(zip(align, pool.starmap(patient_mesh, [(p, align) for p in align])))
    data = dict(parts=sc.PARTS, heads=sc.HEAD, colours=(sc.COLOURS * 255).round().astype(int).tolist(),
                meshes=meshes, scenes=scenes, lus=True)
    html = open(os.path.join(HERE, "playground_template.html")).read()
    html = html.replace("<title>Registration Playground</title>", "<title>LUS Registration Playground</title>")
    html = html.replace("<h1>Registration Playground</h1>", "<h1>Registration Playground · LLR-LUS</h1>")
    with open(OUT, "w") as fh:
        fh.write(html.replace("/*__DATA__*/null", json.dumps(data)))
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB, {len(scenes)} images)")


if __name__ == "__main__":
    main()
