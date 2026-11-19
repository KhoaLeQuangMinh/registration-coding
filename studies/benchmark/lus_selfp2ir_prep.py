"""
Put LLR-LUS into the layout Self-P2IR's real-frame loader reads, with point clouds made exactly as the
released P2I-LReg real clouds were (recipe recovered 2026-11-12; rebuilt P2I-LReg clouds within ~1-2 mm):
  1 Depth Anything V1 METRIC indoor model (ZoeDepth, nyu config); it resizes the image to 392 x 518 itself
  2 PIL nearest-neighbour upsampling to the image size
  3 depth 0 outside the liver mask
  4 every pixel back-projected with the IMAGE CENTRE: X = (u - W/2) D / fx, Y = (v - H/2) D / fy, Z = D
  5 sigma = principal-axis range of the CT vertices (mm) / that of the WHOLE cloud
  6 the cloud scaled by sigma ABOUT its own axis-aligned bounding-box centre (so the result is sigma * cloud
    minus a shift of (sigma - 1) * centre). Their renderer scales back about that centre to put the camera
    at the origin; scaling first and then centring (the first version, 2026-11-13) differs by < 1 mm in the
    cloud but puts their renderer's camera in the wrong place, which scrambles their Dice.
No undistortion (P2I-LReg has none). Faithful replication: the quirks (image centre, camera point inside
the scale) are kept on purpose.

Layout written, one folder per patient (Self-P2IR hard-codes 4K images for patient 05, so LUS Patient 3,
the 4K one, is stored as 05):
  <out>/<id>/model/reconstructed_mesh_world_m.obj   LUS Liver.ply, mm -> m
  <out>/<id>/real/CAM_K.yml                         calibrated fx, fy, cx, cy
  <out>/<id>/real/labels/frame_<n>_json/{img,label}.png   label: palette PNG, index 1 = liver
  <out>/<id>/real/liverPcds/frame_<n>_json.ply      every pixel, mm (binary; P2I-LReg's are ASCII)
  <out>/<id>/real/scale/frame_<n>_json.txt          sigma
  <out>/<id>/{test,train}_real.txt                  all frames are test frames
  <out>/meta.json                                   per frame: sigma and the shift (mm), to map results back
All of it is derived from LLR-LUS (research use only), so it stays under dataset/.

Before writing LUS, one released P2I-LReg cloud is rebuilt and compared, as a check that the recipe still holds.

Run from the Depth-Anything repo's metric_depth folder (its zoedepth package must be importable):
  python <repo>/studies/benchmark/lus_selfp2ir_prep.py --repo-root <registration-coding> --weights <dir with
      depth_anything_metric_depth_indoor.pt and depth_anything_vitl14/pytorch_model.bin>
"""
import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import open3d as o3d
import torch
import yaml
from PIL import Image

LUS_IDS = {1: "01", 2: "02", 3: "05", 4: "04"}
PALETTE = [0, 0, 0, 128, 0, 0] + [0] * (256 * 3 - 6)


def depth_model(weights):
    sys.path.insert(0, os.getcwd())
    from zoedepth.models.builder import build_model
    from zoedepth.utils.config import get_config
    os.makedirs("checkpoints", exist_ok=True)
    link = "checkpoints/depth_anything_vitl14.pth"
    if not os.path.exists(link):
        os.symlink(os.path.join(weights, "depth_anything_vitl14", "pytorch_model.bin"), link)
    cfg = get_config("zoedepth", "eval", "nyu")
    cfg.pretrained_resource = f"local::{os.path.join(weights, 'depth_anything_metric_depth_indoor.pt')}"
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return build_model(cfg).to(dev).eval(), dev


def principal_range(X):
    Xc = X - X.mean(0)
    _, V = np.linalg.eigh(np.cov(Xc.T))
    return float(np.ptp(Xc @ V[:, -1]))


def cloud(model, dev, img_bgr, liver, fx, fy, ct_mm):
    """Steps 1-6: the released-style cloud (H*W x 3, mm), sigma, and the shift that was subtracted."""
    H, W = liver.shape
    x = torch.from_numpy(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None] / 255.0).float().to(dev)
    with torch.no_grad():
        out = model(x, dataset="nyu")
    pred = (out["metric_depth"] if isinstance(out, dict) else out).squeeze().cpu().numpy().astype(np.float32)
    D = np.array(Image.fromarray(pred).resize((W, H), Image.NEAREST)) * liver
    v, u = np.mgrid[0:H, 0:W]
    T = np.stack([(u - W / 2) * D / fx, (v - H / 2) * D / fy, D], -1).reshape(-1, 3)
    sigma = principal_range(ct_mm) / principal_range(T)
    centre = (T.max(0) + T.min(0)) / 2               # scaled ABOUT the unscaled cloud's bounding-box centre:
    P = sigma * (T - centre) + centre                # their renderer undoes exactly this to put the camera back at 0
    return P, sigma, (sigma - 1) * centre            # P = sigma * T - shift


def write_ply(path, P, rgb):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P.astype(np.float64)))
    pc.colors = o3d.utility.Vector3dVector(rgb.reshape(-1, 3).astype(np.float64) / 255.0)
    o3d.io.write_point_cloud(path, pc, write_ascii=False)


def check_recipe(model, dev, p2i_root, pid="06", frame="frame_10_json"):
    base = f"{p2i_root}/{pid}/real"
    img = cv2.imread(f"{base}/labels/{frame}/img.png")
    liver = np.array(Image.open(f"{base}/labels/{frame}/label.png")) > 0
    K = np.array(yaml.safe_load(open(f"{base}/CAM_K.yml"))["00000"][0]["cam_K"]).reshape(3, 3)
    ct = np.asarray(o3d.io.read_triangle_mesh(f"{p2i_root}/{pid}/model/reconstructed_mesh_world_m.obj").vertices) * 1000
    P, sigma, shift = cloud(model, dev, img, liver, K[0, 0], K[1, 1], ct)
    rel = np.asarray(o3d.io.read_point_cloud(f"{base}/liverPcds/{frame}.ply").points)
    e = np.linalg.norm(P - rel, axis=1)[liver.reshape(-1)]
    sig_file = float(open(f"{base}/scale/{frame}.txt").read())
    T = (P + shift) / sigma                                  # with the file's sigma: only depth noise should remain
    centre = (T.max(0) + T.min(0)) / 2
    e_file = np.linalg.norm(sig_file * (T - centre) + centre - rel, axis=1)[liver.reshape(-1)]
    print(f"recipe check on P2I-LReg {pid}/{frame}: liver points vs released cloud median {np.median(e):.2f} mm, "
          f"max {e.max():.2f} mm (with the scale file's sigma: median {np.median(e_file):.3f}, max {e_file.max():.3f} mm); "
          f"sigma {sigma:.2f} vs file {sig_file:.2f}")
    assert np.median(e) < 2.0 and np.median(e_file) < 0.05, "the recipe no longer reproduces the released cloud"


def camera(path):
    c = {e.tag: e.text for e in ET.parse(path).getroot()}
    return float(c["fx"]), float(c["fy"]), float(c["cx"]), float(c["cy"]), int(c["width"]), int(c["height"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--weights", required=True)
    args = ap.parse_args()
    R = args.repo_root
    lus, masks, out = f"{R}/dataset/lus-evaluation/data/Dataset", f"{R}/dataset/lus-evaluation/masks_manual", \
        f"{R}/dataset/lus-evaluation/selfp2ir"

    model, dev = depth_model(args.weights)
    check_recipe(model, dev, f"{R}/dataset/liver-registration")

    meta = {}
    for p, pid in LUS_IDS.items():
        src = f"{lus}/Patient{p}"
        fx, fy, cx, cy, W, H = camera(f"{src}/Lap Images/CameraParameters")
        for d in ("model", "real/labels", "real/liverPcds", "real/scale"):
            os.makedirs(f"{out}/{pid}/{d}", exist_ok=True)
        mesh = o3d.io.read_triangle_mesh(f"{src}/Preoperative Model/Liver.ply")
        ct_mm = np.asarray(mesh.vertices).copy()
        mesh.vertices = o3d.utility.Vector3dVector(ct_mm / 1000.0)
        o3d.io.write_triangle_mesh(f"{out}/{pid}/model/reconstructed_mesh_world_m.obj", mesh)
        with open(f"{out}/{pid}/real/CAM_K.yml", "w") as f:
            yaml.safe_dump({"00000": [{"cam_K": [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]}]}, f)
        frames = []
        for name in sorted(os.path.splitext(n)[0] for n in os.listdir(f"{src}/Lap Images") if n.endswith(".png")):
            frame = f"frame_{int(name)}_json"
            img = cv2.imread(f"{src}/Lap Images/{name}.png")
            assert img.shape[:2] == (H, W), f"Patient{p}/{name}: image size differs from its calibration"
            liver = cv2.imread(f"{masks}/Patient{p}/{name}.png", 0) > 0
            P, sigma, shift = cloud(model, dev, img, liver, fx, fy, ct_mm)
            os.makedirs(f"{out}/{pid}/real/labels/{frame}", exist_ok=True)
            cv2.imwrite(f"{out}/{pid}/real/labels/{frame}/img.png", img)
            lab = Image.fromarray(liver.astype(np.uint8), mode="P")
            lab.putpalette(PALETTE)
            lab.save(f"{out}/{pid}/real/labels/{frame}/label.png")
            write_ply(f"{out}/{pid}/real/liverPcds/{frame}.ply", P, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            with open(f"{out}/{pid}/real/scale/{frame}.txt", "w") as f:
                f.write(f"{sigma}\n")
            meta[f"Patient{p}/{name}"] = dict(id=pid, frame=frame, sigma=sigma, shift_mm=shift.tolist(),
                                              size=[W, H], fx=fx, fy=fy, cx=cx, cy=cy)
            frames.append(frame)
            print(f"Patient{p}/{name} -> {pid}/{frame}: sigma {sigma:.1f}, liver {liver.mean() * 100:.0f} % of the image, "
                  f"liver depth {np.percentile(P[liver.reshape(-1), 2] + shift[2], [5, 95]).round().tolist()} mm", flush=True)
        with open(f"{out}/{pid}/test_real.txt", "w") as f:
            f.write("\n".join(frames) + "\n")
        open(f"{out}/{pid}/train_real.txt", "w").close()
    with open(f"{out}/meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {len(meta)} frames to {out}/")


if __name__ == "__main__":
    main()
