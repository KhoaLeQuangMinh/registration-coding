"""
Build goal/out/register_playground.html: an interactive page where you move (and bend) the CT liver by hand until it
matches the photo, scored live with the same goal as our tests (J = A/4.5 + B/0.02).

Two scenes:
    real   P2I-LReg 01 / frame 10: real photo, its liver label, Depth Anything shape from the released cloud, real K.
           No ground truth exists; the reference pose is our pipeline's best (all-around) pose from real_frame_reg.py.
    syn    P2I-LReg 06 / synthetic view 00430 drawn with the patient's REAL K (as in shape_trust.py): the true pose is
           known. The "photo" is a shaded render (synthetic views have no RGB). Its depth shape can be perfect or
           distorted (bumpy / flattened / slanted, 20 deg mean tilt error) with shape_trust.distort.

The page embeds patient data, so it stays local (goal/out is not committed). Usage, from the repo root:
    python goal/make_playground.py && open goal/out/register_playground.html
"""
import base64
import json
import os
import sys

import cv2
import numpy as np
import open3d as o3d
import yaml
from scipy.spatial import cKDTree

sys.path.insert(0, "goal")
import show_classes as sc  # noqa: E402
from goal_landscape import Patient, normals, render  # noqa: E402
from real_frame_reg import load_photo  # noqa: E402
from shape_trust import distort  # noqa: E402

W, H, DOWN = 480, 270, 4
HERE = os.path.dirname(os.path.abspath(__file__))


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()


def png(img, ext=".png", q=None):
    ok, buf = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, q] if q else [])
    return f"data:image/{'jpeg' if ext == '.jpg' else 'png'};base64," + base64.b64encode(buf).decode()


def normal_png(Z, K, keep):
    """Normals facing the camera, vision axes (x right, y down, z forward), as RGBA: rgb = n*0.5+0.5, a = valid."""
    N, ok = normals(Z, K)
    N = -N                                                   # normals() points away from the camera
    rgba = np.zeros((H, W, 4), np.uint8)
    rgba[..., :3] = np.clip((N * 0.5 + 0.5) * 255 + 0.5, 0, 255).astype(np.uint8)
    rgba[..., 3] = ((ok & keep) * 255).astype(np.uint8)
    return png(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))


def mesh_block(p):
    m, V, T, C, Nt = sc.load(p)
    lab = sc.parts(V, C, Nt, T, m)
    small = m.simplify_quadric_decimation(9000)
    small.remove_degenerate_triangles(); small.remove_unreferenced_vertices()
    Vs, Ts = np.asarray(small.vertices), np.asarray(small.triangles)
    _, idx = cKDTree(C).query(Vs)                            # part of the nearest original triangle
    poses = []
    for part in range(6):
        for head in range(8):
            R0, t0, _ = sc.class_pose(V, T, C, Nt, lab, part, head)
            poses.append(dict(part=part, head=head, R=R0.tolist(), t=t0.tolist()))
    return dict(v=b64(Vs.astype(np.float32)), f=b64(Ts.astype(np.uint32)), part=b64(lab[idx].astype(np.uint8)),
                centre=V.mean(0).tolist(), nv=len(Vs), nf=len(Ts), classes=poses), m


def visible_png(V):
    return png((V * 255).astype(np.uint8))


def real_scene():
    p, f = "01", "10"
    img, M, V, D = load_photo(p, f)
    K = np.array(list(yaml.safe_load(open(f"dataset/liver-registration/{p}/real/CAM_K.yml")).values())[0][0]["cam_K"]).reshape(3, 3)
    Kc = np.diag([1 / DOWN, 1 / DOWN, 1.0]) @ K
    mesh, _ = mesh_block(p)
    ref = [r for r in json.load(open("goal/out/real_01_10.json")) if r["alt"] == "all-around"][0]
    return dict(
        id="real", title="Real photo: patient 01, frame 10",
        note="Real laparoscope photo. Nobody knows the true pose; the reference is our pipeline's best pose "
             "(the one you judged most plausible), not ground truth.",
        refName="our pipeline's pose", refIsTruth=False,
        K=K.tolist(), photo=png(cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA), ".jpg", 88),
        mask=png((M * 255).astype(np.uint8)), visible=visible_png(V),
        shapes=[dict(name="Depth Anything (released cloud)", normals=normal_png(np.where(M, D, 0), Kc, M))],
        ref=dict(R=ref["R"], t=ref["t"]), mesh=mesh)


def syn_scene():
    p, k = "06", "00430"
    pat = Patient(p)
    R, t = pat.true_pose(k)
    Kc = pat.real_k
    S, Z = render(pat.scene, R, t, Kc, W, H)
    # the "photo": a shaded drawing of the true surface at 960x540 (synthetic views have no RGB)
    Kb = np.diag([2.0, 2.0, 1.0]) @ Kc
    Sb, Zb = render(pat.scene, R, t, Kb, 2 * W, 2 * H)
    Nb, _ = normals(Zb, Kb)
    lam = np.clip(Nb[..., 2], 0, 1)                         # normals() points away: +z = facing the camera
    u, v = np.meshgrid(np.linspace(-1, 1, 2 * W), np.linspace(-1, 1, 2 * H))
    bg = (np.clip(1 - 0.5 * (u ** 2 + v ** 2), 0, 1)[..., None] * np.array([38, 30, 34])).astype(np.float32)
    liver = (0.25 + 0.75 * lam[..., None] ** 1.3) * np.array([70, 80, 165]) + 60 * lam[..., None] ** 30
    img = np.where(Sb[..., None], liver, bg)
    img = np.clip(img + np.random.default_rng(0).normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    shapes = [dict(name="perfect depth", normals=normal_png(Z, Kc, S))]
    for kind in ["bumpy", "flattened", "slanted"]:
        Zd, e = distort(Z, S, Kc, kind, 20, 7)
        shapes.append(dict(name=f"{kind}, {e:.0f}° tilt error", normals=normal_png(Zd, Kc, S)))
    mesh, _ = mesh_block(p)
    return dict(
        id="syn", title="Synthetic view: patient 06, view 00430",
        note="Synthetic: the CT drawn at a known pose with the patient's real camera, so the answer is known. "
             "The 'photo' is a shaded drawing (synthetic views have no colour image).",
        refName="the true pose", refIsTruth=True,
        K=pat.real_k_full.tolist(), photo=png(img, ".jpg", 90),
        mask=png((S * 255).astype(np.uint8)), visible=visible_png(np.ones_like(S)),
        shapes=shapes, ref=dict(R=R.tolist(), t=t.tolist()), mesh=mesh)


def main():
    data = dict(parts=sc.PARTS, heads=sc.HEAD, colours=(sc.COLOURS * 255).round().astype(int).tolist(),
                scenes=[real_scene(), syn_scene()])
    html = open(os.path.join(HERE, "playground_template.html")).read()
    out = os.path.join(HERE, "out", "register_playground.html")
    with open(out, "w") as fh:
        fh.write(html.replace("/*__DATA__*/null", json.dumps(data)))
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
