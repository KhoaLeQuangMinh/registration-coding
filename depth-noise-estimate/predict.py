"""
Run Depth Anything V2 (relative) on every SERV-CT left image and save its raw output (relative disparity, at the
image's own size) as <experiment>_<frame>.npy.

Usage: python predict.py --data <SERV-CT folder> --out <predictions folder> [--model depth-anything/Depth-Anything-V2-Large-hf]
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import pipeline

from common import frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Large-hf")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("depth-estimation", model=a.model, device=device)
    print(f"model {a.model} on {'GPU' if device == 0 else 'CPU'}", flush=True)
    for exp, fid in frames(a.data):
        img = Image.open(Path(a.data) / exp / "Left_rectified" / f"{fid}.png").convert("RGB")
        pred = pipe(img)["predicted_depth"].float()            # model resolution, relative disparity
        pred = pred.reshape(1, 1, *pred.shape[-2:])
        pred = torch.nn.functional.interpolate(pred, size=(img.height, img.width), mode="bicubic", align_corners=False)
        np.save(out / f"{exp}_{fid}.npy", pred[0, 0].cpu().numpy().astype(np.float32))
        print(f"{exp}/{fid}: saved", flush=True)


if __name__ == "__main__":
    main()
