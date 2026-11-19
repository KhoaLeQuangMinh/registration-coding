"""
Which single-image depth model gets the DISTANCE right in laparoscopy? (route 1, step 1)

On LLR-LUS, the ultrasound probe touches the liver and is tracked, so the true depth of the liver surface where it
touches is known (lus_oracle.py: "contact depth", per image). Each model predicts metric depth for the whole photo;
we compare its median depth on the visible liver pixels around the probe tip (the same "near" region as the oracle)
with the contact depth. Used for EVALUATION only: the probe is never an input.

Models (one per run, so one broken install cannot stop the others):
  depthpro      Apple Depth Pro                       focal: predicted, and the calibrated one (f_px)
  metric3d      Metric3D v2 ViT-L (torch.hub)         focal: calibrated (its canonical-camera conversion)
  unidepth      UniDepthV2 ViT-L                      camera: predicted, and the calibrated pinhole
  moge          MoGe-2 ViT-L                          field of view: predicted, and from the calibration
  da3metric     Depth Anything 3 METRIC-LARGE         README: metric = focal * output / 300 (two focal readings)
  da3nested     Depth Anything 3 NESTED-GIANT-LARGE-1.1   README: output already in metres
  dav2indoor    Depth Anything V2 Metric-Indoor-Large (transformers)
  dav1indoor    Depth Anything V1 metric indoor = Self-P2IR's depth, before its scale step (from the upload)
Also reported: the depth Self-P2IR's recipe gives there (its cloud after the scale step), from oracle.json.

Images are used as they are (not undistorted), with the calibrated pinhole fx, fy, cx, cy, as in the recipe.

Writes <out>/depth_<model>.json (per image: true contact depth, predicted depth near the probe, whole-liver median)
and <out>/preview_<model>.jpg; `--summary` prints the comparison of every model found in <out>.

Usage: python depth_benchmark.py --lus <unzipped LUS upload> --oracle oracle.json --model depthpro --out out/
       python depth_benchmark.py --lus ... --oracle oracle.json --summary --out out/
"""
import argparse
import glob
import json
import math
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PREVIEW = ["Patient1/02", "Patient2/11", "Patient3/05", "Patient4/02"]


def decode(runs, w, h):
    return np.repeat(np.arange(len(runs)) % 2, runs).astype(bool).reshape(h, w)


def resize_to(depth, w, h, mode=cv2.INTER_LINEAR):
    depth = np.asarray(depth, np.float32)
    return depth if depth.shape == (h, w) else cv2.resize(depth, (w, h), interpolation=mode)


# ---------------- model adapters: each returns {variant name: depth (H x W, metres)} for one RGB image ----------------

def load_depthpro():
    import depth_pro
    from huggingface_hub import hf_hub_download
    os.makedirs("checkpoints", exist_ok=True)
    if not os.path.exists("checkpoints/depth_pro.pt"):
        if os.system("wget -q https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt -O checkpoints/depth_pro.pt") != 0:
            import shutil  # the HF cache path is a symlink into the cache: copy the file, don't move the link
            shutil.copyfile(hf_hub_download("apple/DepthPro", "depth_pro.pt"), "checkpoints/depth_pro.pt")
    model, transform = depth_pro.create_model_and_transforms(device=torch.device(DEV), precision=torch.half)
    model.eval()

    def run(rgb, cam):
        x = transform(Image.fromarray(rgb))
        out = {}
        with torch.no_grad():
            p = model.infer(x)
            out["predicted focal"] = p["depth"].float().cpu().numpy()
            out["focal_px_predicted"] = float(p["focallength_px"])
            try:   # infer() calls f_px.squeeze(), so it must be a tensor, not a float
                p = model.infer(x, f_px=torch.tensor(float(cam["fx"]), device=DEV))
                out["calibrated focal"] = p["depth"].float().cpu().numpy()
            except Exception as e:  # keep the predicted-focal result if this variant fails
                print("  Depth Pro with the calibrated focal failed:", repr(e))
        return out
    return run


def load_metric3d():
    # ":main" skips torch.hub's branch probe (a GitHub 504 there killed one run); retry brief GitHub outages
    for attempt in range(4):
        try:
            model = torch.hub.load("yvanyin/metric3d:main", "metric3d_vit_large", pretrain=True, trust_repo=True)
            break
        except Exception as e:
            print(f"  torch.hub attempt {attempt + 1} failed: {e!r}", flush=True)
            if attempt == 3:
                raise
            time.sleep(30)
    model = model.to(DEV).eval()

    def run(rgb, cam):  # follows the hubconf.py example
        h, w = rgb.shape[:2]
        size = (616, 1064)
        s = min(size[0] / h, size[1] / w)
        img = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_LINEAR)
        fx = cam["fx"] * s
        mean = np.array([123.675, 116.28, 103.53])
        ph, pw = size[0] - img.shape[0], size[1] - img.shape[1]
        pad = [ph // 2, ph - ph // 2, pw // 2, pw - pw // 2]
        img = cv2.copyMakeBorder(img, pad[0], pad[1], pad[2], pad[3], cv2.BORDER_CONSTANT, value=mean.tolist())
        x = torch.from_numpy(((img - mean) / np.array([58.395, 57.12, 57.375])).transpose(2, 0, 1)).float()[None].to(DEV)
        with torch.no_grad():
            d, _, _ = model.inference({"input": x})
        d = d.squeeze()[pad[0]:d.shape[-2] - pad[1], pad[2]:d.shape[-1] - pad[3]]
        d = torch.nn.functional.interpolate(d[None, None], (h, w), mode="bilinear").squeeze()
        return {"calibrated focal": (d * fx / 1000.0).clamp(0, 300).float().cpu().numpy()}
    return run


def load_unidepth():
    from unidepth.models import UniDepthV2
    from unidepth.utils.camera import Pinhole
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").to(DEV).eval()

    def run(rgb, cam):
        x = torch.from_numpy(rgb).permute(2, 0, 1).to(DEV)
        K = torch.tensor([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]], dtype=torch.float32, device=DEV)
        with torch.no_grad():
            a = model.infer(x)["depth"].squeeze().float().cpu().numpy()
            b = model.infer(x, Pinhole(K=K[None]))["depth"].squeeze().float().cpu().numpy()
        return {"predicted camera": a, "calibrated camera": b}
    return run


def load_moge():
    from moge.model.v2 import MoGeModel
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(DEV).eval()

    def run(rgb, cam):
        x = torch.tensor(rgb / 255.0, dtype=torch.float32, device=DEV).permute(2, 0, 1)
        fov = math.degrees(2 * math.atan(rgb.shape[1] / (2 * cam["fx"])))
        out = {}
        with torch.no_grad():
            out["predicted field of view"] = model.infer(x)["depth"].float().cpu().numpy()
            try:
                out["calibrated field of view"] = model.infer(x, fov_x=fov)["depth"].float().cpu().numpy()
            except TypeError as e:
                print("  MoGe infer() does not take fov_x:", e)
        return {k: np.where(np.isfinite(v), v, np.nan) for k, v in out.items()}
    return run


def load_da3(repo):
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained(repo).to(DEV)

    def run(rgb, cam):
        h, w = rgb.shape[:2]
        with torch.no_grad():
            p = model.inference([Image.fromarray(rgb)])
        net = np.asarray(p.depth[0], np.float32)
        if "METRIC" not in repo:
            return {"as output (metres)": resize_to(net, w, h)}
        f = (cam["fx"] + cam["fy"]) / 2
        s = net.shape[1] / w                                # focal at the resolution the network saw
        return {"focal of the original image": resize_to(f * net / 300.0, w, h),
                "focal at the network's resolution": resize_to(f * s * net / 300.0, w, h)}
    return run


def load_dav2indoor():
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    name = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
    proc, model = AutoImageProcessor.from_pretrained(name), AutoModelForDepthEstimation.from_pretrained(name).to(DEV).eval()

    def run(rgb, cam):
        inputs = proc(images=Image.fromarray(rgb), return_tensors="pt").to(DEV)
        with torch.no_grad():
            d = model(**inputs).predicted_depth
        d = torch.nn.functional.interpolate(d[:, None], rgb.shape[:2], mode="bicubic", align_corners=False).squeeze()
        return {"as output": d.float().cpu().numpy()}
    return run


LOADERS = dict(depthpro=load_depthpro, metric3d=load_metric3d, unidepth=load_unidepth, moge=load_moge,
               da3metric=lambda: load_da3("depth-anything/DA3METRIC-LARGE"),
               da3nested=lambda: load_da3("depth-anything/DA3NESTED-GIANT-LARGE-1.1"),
               dav2indoor=load_dav2indoor)


# ---------------------------------------------------------------------------------------------------------------------

def images(lus):
    meta = json.load(open(f"{lus}/meta.json"))
    for key in sorted(meta):
        m = meta[key]
        rgb = cv2.cvtColor(cv2.imread(f"{lus}/{m['id']}/real/labels/{m['frame']}/img.png"), cv2.COLOR_BGR2RGB)
        yield key, m, rgb


def evaluate(name, lus, oracle, out_dir):
    meta = json.load(open(f"{lus}/meta.json"))
    if name == "dav1indoor":
        run = None
    else:
        t0 = time.time()
        run = LOADERS[name]()
        print(f"{name}: model loaded in {time.time() - t0:.0f} s", flush=True)
    rows, previews, maps = {}, {}, {}
    for key, m, rgb in images(lus):
        o = oracle[key]
        W, H = o["size"]
        near = decode(o["near_rle"], W, H)
        liver = np.array(Image.open(f"{lus}/{m['id']}/real/labels/{m['frame']}/label.png")) > 0
        cam = dict(fx=m["fx"], fy=m["fy"], cx=m["cx"], cy=m["cy"])
        if run is None:  # Self-P2IR's own depth, before its scale step: the packed depth map (metres, liver only)
            preds = {"as used by Self-P2IR (before scaling)": np.load(f"{lus}/{m['id']}/real/depth/{m['frame']}.npz")["D"]}
        else:
            preds = run(rgb, cam)
        extra = {k: v for k, v in preds.items() if not isinstance(v, np.ndarray)}
        rows[key] = dict(contact_mm=o["contact_depth_mm"], recipe_mm=o["cloud_depth_near_mm"], **extra, variants={})
        for variant, d in preds.items():
            if not isinstance(d, np.ndarray):
                continue
            d = resize_to(d, W, H) * 1000.0                                                    # mm
            rows[key]["variants"][variant] = dict(near_mm=float(np.nanmedian(d[near])), liver_mm=float(np.nanmedian(d[liver])))
            maps[f"{key}|{variant}"] = cv2.resize(d, (W // 4, H // 4), interpolation=cv2.INTER_AREA).astype(np.float16)
            if key in PREVIEW:
                previews.setdefault(variant, []).append((key, rgb, d, near))
        v0 = next(iter(rows[key]["variants"].values()))
        print(f"  {key}: true {o['contact_depth_mm']:6.1f} mm | " +
              " | ".join(f"{k} {v['near_mm']:7.1f}" for k, v in rows[key]["variants"].items()), flush=True)
    json.dump(rows, open(f"{out_dir}/depth_{name}.json", "w"), indent=1)
    # whole depth maps (mm, quarter resolution, image NOT undistorted), so the SHAPE can be tested in the registration
    np.savez_compressed(f"{out_dir}/maps_{name}.npz", **maps)

    for variant, items in previews.items():  # one strip per variant: photo | depth, true contact depth written on it
        tiles = []
        for key, rgb, d, near in items:
            lo, hi = np.nanpercentile(d[np.isfinite(d)], [2, 98])
            col = cv2.applyColorMap((255 * (1 - np.clip((np.nan_to_num(d, nan=hi) - lo) / max(hi - lo, 1e-6), 0, 1))).astype(np.uint8), cv2.COLORMAP_TURBO)
            col[near] = (255, 255, 255)
            pair = np.hstack([cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (480, 270)), cv2.resize(col, (480, 270))])
            cv2.putText(pair, f"{key}: true {oracle[key]['contact_depth_mm']:.0f} mm, model {np.nanmedian(d[near]):.0f} mm (white = probe area)",
                        (8, 20), 0, 0.5, (255, 255, 255), 1)
            tiles.append(pair)
        cv2.imwrite(f"{out_dir}/preview_{name}_{variant.replace(' ', '_').replace('(', '').replace(')', '')}.jpg",
                    np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(f"{name}: wrote {out_dir}/depth_{name}.json", flush=True)


def summary(out_dir):
    results = {os.path.basename(f)[6:-5]: json.load(open(f)) for f in sorted(glob.glob(f"{out_dir}/depth_*.json"))}
    if not results:
        print("no results yet"); return
    any_rows = next(iter(results.values()))
    keys = sorted(any_rows)
    table = [("Self-P2IR recipe (after its scale step)", {k: any_rows[k]["recipe_mm"] for k in keys})]
    for name, rows in results.items():
        for variant in rows[keys[0]]["variants"]:
            table.append((f"{name}: {variant}", {k: rows[k]["variants"][variant]["near_mm"] for k in keys if variant in rows[k]["variants"]}))
    true = {k: any_rows[k]["contact_mm"] for k in keys}
    print(f"\nDepth of the liver next to the probe: predicted vs TRUE contact depth, {len(keys)} LLR-LUS images\n")
    print(f"{'model':58s} {'|error| median':>14s} {'pred/true median':>16s} {'within 10 %':>11s} {'within 20 %':>11s}  "
          + "  ".join(f"P{p} pred/true" for p in range(1, 5)))
    for label, pred in table:
        ks = [k for k in keys if k in pred and np.isfinite(pred[k])]
        r = np.array([pred[k] / true[k] for k in ks]); e = np.array([abs(pred[k] - true[k]) for k in ks])
        per = [np.median([pred[k] / true[k] for k in ks if k.startswith(f"Patient{p}/")]) for p in range(1, 5)]
        print(f"{label:58s} {np.median(e):11.1f} mm {np.median(r):16.2f} {100 * np.mean(abs(r - 1) < 0.1):9.0f} % {100 * np.mean(abs(r - 1) < 0.2):9.0f} %  "
              + "  ".join(f"{x:13.2f}" for x in per))
    print(f"\ntrue contact depth: median {np.median(list(true.values())):.0f} mm, per patient "
          + ", ".join(f"P{p} {np.median([true[k] for k in keys if k.startswith(f'Patient{p}/')]):.0f}" for p in range(1, 5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lus", required=True)
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--model", choices=list(LOADERS) + ["dav1indoor"])
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--out", default="depth_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.summary:
        summary(args.out)
    else:
        evaluate(args.model, args.lus, json.load(open(args.oracle)), args.out)


if __name__ == "__main__":
    main()
