"""
Pack the LLR-LUS Self-P2IR layout (lus_selfp2ir_prep.py) small enough to upload to Kaggle, and unpack it there.

The clouds are most of the size (4 GB) but carry nothing beyond their depth map: each released-style cloud is
    P = sigma * [(u - W/2) D / fx, (v - H/2) D / fy, D] - shift      (every pixel, D = 0 off the liver)
So `pack` stores D (float32, compressed) instead of the cloud, plus a fingerprint of the cloud, and `unpack`
rebuilds every cloud and checks it against its fingerprint.

  pack     dataset/lus-evaluation/selfp2ir/ -> dataset/lus-evaluation/kaggle_upload/lus-selfp2ir.zip
  unpack   <unzipped upload> -> <out>/ (the layout Self-P2IR's loader reads)

LLR-LUS is research-only: upload it only as a PRIVATE Kaggle dataset, with the authors' agreement.

  centre-ct  <layout> -> <out>/ : the same layout with each CT liver moved to the origin (see centre_ct)

Usage: python studies/benchmark/lus_kaggle_pack.py pack
       python lus_kaggle_pack.py unpack <unzipped folder> <out folder> [<sigma json, e.g. oracle.json>]
       python lus_kaggle_pack.py centre-ct <layout> <out folder>
"""
import json
import os
import shutil
import sys
import zipfile

import numpy as np
import open3d as o3d

SRC = "dataset/lus-evaluation/selfp2ir"
UPLOAD = "dataset/lus-evaluation/kaggle_upload"


def rebuild(D, sigma, shift, fx, fy):
    H, W = D.shape
    v, u = np.mgrid[0:H, 0:W]
    T = np.stack([(u - W / 2) * D / fx, (v - H / 2) * D / fy, D], -1).reshape(-1, 3)
    return sigma * T - np.asarray(shift)


def fingerprint(P):
    return [float(x) for x in np.r_[np.abs(P).mean(0), np.abs(P).max(0)]]  # mm; the cloud is centred, so not sums


def pack():
    meta = json.load(open(f"{SRC}/meta.json"))
    stage = f"{UPLOAD}/lus-selfp2ir"
    shutil.rmtree(stage, ignore_errors=True)
    shutil.copytree(SRC, stage, ignore=shutil.ignore_patterns("liverPcds"))
    for key, m in meta.items():
        P = np.asarray(o3d.io.read_point_cloud(f"{SRC}/{m['id']}/real/liverPcds/{m['frame']}.ply").points)
        W, H = m["size"]
        D = ((P + np.asarray(m["shift_mm"]))[:, 2] / m["sigma"]).reshape(H, W).astype(np.float32)
        err = np.abs(rebuild(D, m["sigma"], m["shift_mm"], m["fx"], m["fy"]) - P).max()
        assert err < 1e-3, f"{key}: depth does not rebuild the cloud (max error {err} mm)"
        os.makedirs(f"{stage}/{m['id']}/real/depth", exist_ok=True)
        np.savez_compressed(f"{stage}/{m['id']}/real/depth/{m['frame']}.npz", D=D)
        m["fingerprint"] = fingerprint(P)
        print(f"{key}: packed, rebuild error {err:.1e} mm", flush=True)
    json.dump(meta, open(f"{stage}/meta.json", "w"), indent=1)
    zpath = f"{UPLOAD}/lus-selfp2ir.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(stage):
            for f in files:
                z.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), UPLOAD))
    shutil.rmtree(stage)
    print(f"wrote {zpath} ({os.path.getsize(zpath) / 1e6:.0f} MB)")


def unpack(src, out, sigma_json=None):
    """Rebuild the clouds. With sigma_json ({image: {"sigma": ...}}, e.g. lus_oracle.py's oracle.json) each cloud
    is then rescaled to that sigma, about the same centre, and meta.json / scale files are updated to match."""
    meta = json.load(open(f"{src}/meta.json"))
    sigmas = json.load(open(sigma_json)) if sigma_json else {}
    shutil.copytree(src, out, ignore=shutil.ignore_patterns("depth"), dirs_exist_ok=True)
    for key, m in meta.items():
        D = np.load(f"{src}/{m['id']}/real/depth/{m['frame']}.npz")["D"].astype(np.float64)
        P = rebuild(D, m["sigma"], m["shift_mm"], m["fx"], m["fy"])
        fp = np.array(fingerprint(P)); ref = np.array(m["fingerprint"])
        assert np.allclose(fp, ref, rtol=0, atol=1e-3), f"{key}: rebuilt cloud differs from the original"
        if key in sigmas:
            centre = np.asarray(m["shift_mm"]) / (m["sigma"] - 1)       # the shift is (sigma - 1) * centre
            m.update(sigma_recipe=m["sigma"], sigma=sigmas[key]["sigma"])
            m["shift_mm"] = ((m["sigma"] - 1) * centre).tolist()
            P = rebuild(D, m["sigma"], m["shift_mm"], m["fx"], m["fy"])
            with open(f"{out}/{m['id']}/real/scale/{m['frame']}.txt", "w") as f:
                f.write(f"{m['sigma']}\n")
        os.makedirs(f"{out}/{m['id']}/real/liverPcds", exist_ok=True)
        o3d.io.write_point_cloud(f"{out}/{m['id']}/real/liverPcds/{m['frame']}.ply",
                                 o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)), write_ascii=False)
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
    print(f"rebuilt {len(meta)} clouds in {out}/, all match their fingerprints"
          + (f"; {len(sigmas)} rescaled to the given sigma" if sigmas else ""))


def centre_ct(src, out):
    """A copy of the layout whose CT livers are moved so their vertex mean is at the origin, as P2I-LReg's
    are (within ~3 cm). Self-P2IR is not indifferent to where the CT sits (frame check, 2026-11-13); the
    clouds are untouched (sigma ignores position). Writes <out>/ct_offsets.json: the mean subtracted, mm."""
    offsets = {}
    for pid in sorted(d for d in os.listdir(src) if d.isdigit()):
        os.makedirs(f"{out}/{pid}/model", exist_ok=True)
        for name in os.listdir(f"{src}/{pid}"):
            if name != "model" and not os.path.exists(f"{out}/{pid}/{name}"):
                os.symlink(os.path.abspath(f"{src}/{pid}/{name}"), f"{out}/{pid}/{name}")
        mesh = o3d.io.read_triangle_mesh(f"{src}/{pid}/model/reconstructed_mesh_world_m.obj")
        V = np.asarray(mesh.vertices)
        offsets[pid] = (V.mean(0) * 1000.0).tolist()
        mesh.vertices = o3d.utility.Vector3dVector(V - V.mean(0))
        o3d.io.write_triangle_mesh(f"{out}/{pid}/model/reconstructed_mesh_world_m.obj", mesh)
    json.dump(offsets, open(f"{out}/ct_offsets.json", "w"), indent=1)
    print("CT livers moved to the origin; offsets (mm):", {k: np.round(v).astype(int).tolist() for k, v in offsets.items()})


if __name__ == "__main__":
    if sys.argv[1:2] == ["pack"]:
        pack()
    elif sys.argv[1:2] == ["unpack"] and len(sys.argv) in (4, 5):
        unpack(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) == 5 else None)
    elif sys.argv[1:2] == ["centre-ct"] and len(sys.argv) == 4:
        centre_ct(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
