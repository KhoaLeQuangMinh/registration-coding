"""
Build out/generator_preview.html: our generator's examples next to the dataset's synthetic views and real photos.

    dataset synthetic   P2I-LReg syn views (label + the CT drawn at the true pose for depth)
    our generator       regsim/generator.py examples (seeded)
    real photos         P2I-LReg real frames (photo, label, Depth Anything depth from the released cloud)
Also: camera positions around the liver (patient 06: dataset synthetic vs generator) and the same statistics for each.
Local only (patient data). Usage (repo root): python -m regsim.preview.make_generator_preview
"""
import base64
import json
import os

import cv2
import numpy as np
import yaml
from scipy.spatial import cKDTree

from regsim import anatomy as sc
from regsim import generator as gen
from regsim.p2ilreg import DATA, OUT as OUTDIR, Patient, load_photo, render

W, H = gen.W, gen.H
OUT = os.path.join(OUTDIR, "generator_preview.html")
COL = dict(bg=(34, 30, 28), liver=(150, 120, 215), hidden=(92, 74, 118), tool=(165, 165, 165), fat=(80, 200, 235),
           ligament=(205, 225, 235), scope=(0, 0, 0), real=(80, 220, 80), covered=(0, 140, 255), border=(255, 150, 60))


def png(img):
    return "data:image/png;base64," + base64.b64encode(cv2.imencode(".png", img)[1]).decode()


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def composite(S, M, layers, E):
    im = np.zeros((H, W, 3), np.uint8); im[:] = COL["bg"]
    im[S & ~M] = COL["hidden"]
    for k in ("fat", "ligament", "tool"):
        if k in layers:
            im[layers[k]] = COL[k]
    im[M] = COL["liver"]
    if "scope" in layers:
        im[layers["scope"]] = COL["scope"]
    for code, k in [(1, "real"), (2, "covered"), (3, "border")]:
        im[cv2.dilate((E == code).astype(np.uint8), np.ones((2, 2), np.uint8)) > 0] = COL[k]
    return im


def depth_img(Z, M, lo=None, hi=None):
    z = np.where(M, Z, np.nan)
    if lo is None:
        lo, hi = np.nanpercentile(z, 2), np.nanpercentile(z, 98)
    x = np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1)
    im = cv2.applyColorMap((np.nan_to_num(x) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    im[~M] = (20, 18, 16)
    return im, lo, hi


def edges_for(S, vis, layers):
    return gen.Generator.edges(S, vis, layers)


def mesh_small(p):
    m, V, T, C, Nt = sc.load(p)
    s = m.simplify_quadric_decimation(8000); s.remove_degenerate_triangles(); s.remove_unreferenced_vertices()
    Vs, Ts = np.asarray(s.vertices), np.asarray(s.triangles)
    return dict(v=b64(Vs.astype(np.float32)), f=b64(Ts.astype(np.uint32)), centre=V.mean(0).tolist()), cKDTree(V).query(Vs)[1]


def frust(R, t, K, depth):
    c = [(0, 0), (gen.IMW, 0), (gen.IMW, gen.IMH), (0, gen.IMH)]
    P = [np.array([(u - K[0, 2]) / K[0, 0] * depth, (v - K[1, 2]) / K[1, 1] * depth, depth]) for u, v in c]
    return dict(C=(-R.T @ t).tolist(), corners=[(R.T @ (p - t)).tolist() for p in P])


def stats(rows):
    def rng_(k, f="{:.0f}"):
        v = np.array([r[k] for r in rows if r.get(k) is not None], float)
        return f"{f.format(np.percentile(v, 5))}–{f.format(np.percentile(v, 95))}" if len(v) else "–"
    tot = np.array([[r["edge_real"], r["edge_covered"], r["edge_border"]] for r in rows], float)
    share = tot.sum(0) / tot.sum()
    return dict(hfov=rng_("hfov_deg"), dist=rng_("centre_dist_mm"), front=rng_("front_angle_deg"), vis=rng_("visible_frac", "{:.2f}"),
                tools=f"{100 * np.mean([r.get('tools', 0) > 0 for r in rows]):.0f} %",
                edges=f"{100 * share[0]:.0f} / {100 * share[1]:.0f} / {100 * share[2]:.0f}",
                bend=rng_("bend_mean_mm"), depth=rng_("depth_tilt_err_deg"))


def dataset_view(pat, p, key):
    R, t = pat.true_pose(key)
    Ks = np.array(pat.syn_k[key][0]["cam_K"], float).reshape(3, 3)
    Kc = np.diag([0.5, 0.5, 1.0]) @ Ks                                      # 960 x 540 -> 480 x 270
    S, Z = render(pat.scene, R, t, Kc, W, H)
    lbl = cv2.resize(cv2.imread(f"{DATA}/{p}/syn/labels/lbl{key}.png")[..., 0], (W, H), interpolation=cv2.INTER_NEAREST)
    vis = lbl == 0
    layers = {"tool": (lbl == 153) | (lbl == 179)} if ((lbl == 153) | (lbl == 179)).any() else {}
    E = edges_for(S | vis, vis, layers)
    K = np.diag([4.0, 4.0, 1.0]) @ Kc                                       # in 1920-wide pixels, like the generator
    C = -R.T @ t
    ne = np.bincount(E[E > 0], minlength=4)[1:]
    info = dict(hfov_deg=float(2 * np.degrees(np.arctan(gen.IMW / 2 / K[0, 0]))), centre_dist_mm=float(np.linalg.norm(C - pat.centre)),
                front_angle_deg=float(np.degrees(np.arccos(np.clip((C - pat.centre) / np.linalg.norm(C - pat.centre) @ gen.FRONT, -1, 1)))),
                visible_frac=float(vis.mean()), tools=int(bool(layers)), edge_real=int(ne[0]), edge_covered=int(ne[1]), edge_border=int(ne[2]),
                bend_mean_mm=0.0, depth_tilt_err_deg=0.0)
    return R, t, K, S | vis, Z, vis, layers, E, info


def main():
    rng = np.random.default_rng(7)
    patients = ["01", "06", "10", "11"]
    G = gen.Generator(patients)
    meshes, nearest = {}, {}
    for p in patients:
        meshes[p], nearest[p] = mesh_small(p)
    cards = {"dataset": [], "generator": [], "real": []}

    # dataset synthetic views
    pats = {p: Patient(p) for p in patients}
    for p in patients:
        keys = [l.strip().replace("ptsLiverCam", "") for l in open(f"{DATA}/{p}/train_syn.txt") if l.strip()]
        keys = [k for k in keys if not (p in ("10", "11") and 2000 <= int(k) < 2500)]
        for key in rng.choice(keys, 2, replace=False):
            R, t, K, S, Z, vis, layers, E, info = dataset_view(pats[p], p, key)
            di, lo, hi = depth_img(Z, vis)
            cards["dataset"].append(dict(title=f"Dataset synthetic · {p}/{key}", patient=p, imgs=[png(composite(S, vis, layers, E)), png(di), png(di)],
                                         info=info, R=R.tolist(), t=t.tolist(), K=K.tolist(), frustum=frust(R, t, K, info["centre_dist_mm"] * 1.2)))

    # our generator
    for p in patients:
        for _ in range(3):
            s = G.sample(p, rng)
            Zt = np.where(s["S"], s["Z"], 0)
            dt, lo, hi = depth_img(Zt / np.median(Zt[s["M"]]), s["S"])
            dd, _, _ = depth_img(s["depth"], s["M"], lo, hi)
            info = s["info"]
            R, t, K = s["R"], s["t"], s["K"]
            inside_w = (s["inside_cam"] - t) @ R
            cards["generator"].append(dict(
                title=f"Generator · {p}", patient=p, imgs=[png(composite(s["S"], s["M"], s["layers"], s["E"])), png(dd), png(dt)],
                info=info, R=R.tolist(), t=t.tolist(), K=K.tolist(), frustum=frust(R, t, K, info["centre_dist_mm"] * 1.2),
                bent=b64(s["Vd"][nearest[p]].astype(np.float32)), inside=inside_w[:24].tolist()))

    # real photos (dev patients; LLR-LUS is not used here)
    for p, f in [("06", "12"), ("06", "20"), ("10", "14"), ("11", "15"), ("11", "30"), ("01", "10")]:
        if not os.path.exists(f"{DATA}/{p}/real/labels/frame_{f}_json"):
            continue
        img, M, V, D = load_photo(p, f)
        Kr = np.array(list(yaml.safe_load(open(f"{DATA}/{p}/real/CAM_K.yml")).values())[0][0]["cam_K"]).reshape(3, 3)
        ph = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        cnt, _ = cv2.findContours(M.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(ph, cnt, -1, (0, 220, 255), 1)
        dr, _, _ = depth_img(np.where(M, D / np.median(D[M]), 0), M)
        lab = np.zeros((H, W, 3), np.uint8); lab[:] = COL["bg"]; lab[M] = COL["liver"]
        cards["real"].append(dict(title=f"Real photo · {p}/frame {f}", patient=p, imgs=[png(ph), png(dr), png(lab)],
                                  info=dict(hfov_deg=float(2 * np.degrees(np.arctan(960 / Kr[0, 0]))), visible_frac=float(M.mean()))))

    # camera positions around patient 06: every dataset view vs 250 generator examples, + statistics
    p = "06"; pat = pats[p]
    keys = [l.strip().replace("ptsLiverCam", "") for l in open(f"{DATA}/{p}/train_syn.txt") if l.strip()]
    cam_ds = []
    for k in keys:
        R, t = pat.true_pose(k)
        cam_ds.append(np.r_[-R.T @ t, R[2]].tolist())
    ds_rows = [dataset_view(pat, p, k)[-1] for k in rng.choice(keys, 150, replace=False)]
    cam_gen, gen_rows = [], []
    for _ in range(250):
        s = G.sample(p, rng)
        cam_gen.append(np.r_[-s["R"].T @ s["t"], s["R"][2]].tolist()); gen_rows.append(s["info"])
    real_hfov = []
    for q in ["01", "06", "10", "11"]:
        Kr = np.array(list(yaml.safe_load(open(f"{DATA}/{q}/real/CAM_K.yml")).values())[0][0]["cam_K"]).reshape(3, 3)
        real_hfov.append(2 * np.degrees(np.arctan(960 / Kr[0, 0])))
    real_vis = [c["info"]["visible_frac"] for c in cards["real"]]
    table = dict(dataset=stats(ds_rows), generator=stats(gen_rows),
                 real=dict(hfov=f"{min(real_hfov):.0f}–{max(real_hfov):.0f}", vis=f"{min(real_vis):.2f}–{max(real_vis):.2f}"))

    data = dict(meshes=meshes, cards=cards, cams=dict(patient=p, dataset=cam_ds, generator=cam_gen), table=table,
                colours={k: "#%02x%02x%02x" % v[::-1] for k, v in COL.items()})
    html = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "generator_preview_template.html")).read().replace("/*__DATA__*/null", json.dumps(data))
    os.makedirs(OUTDIR, exist_ok=True)
    open(OUT, "w").write(html)
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")
    print(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
