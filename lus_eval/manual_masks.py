"""
Hand-drawn liver masks for LLR-LUS, made in lus_eval/tools/mask_editor.html.

The editor starts every image from SAM's mask (sam_masks.py) and the user corrects or redraws it.
Masks travel as run lengths over the image in row order, starting with a run of background.

  export   SAM's masks (dataset/lus-evaluation/masks/) -> dataset/lus-evaluation/sam_masks.js, read by the editor
  import   the editor's download -> dataset/lus-evaluation/masks_manual/PatientN/NN.png (255 = liver);
           pieces and holes under TIDY_PX pixels (brush leftovers) are dropped / filled
           plus masks_manual/review.html (each image with its mask, and how much it differs from SAM's)

All outputs are derived from LLR-LUS, so they stay under dataset/ (not in the repo).

Usage: python lus_eval/manual_masks.py export
       python lus_eval/manual_masks.py import ~/Downloads/lus_manual_masks.json
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

LUS = "dataset/lus-evaluation"
TIDY_PX = 500  # brush leftovers: on the 2026-11-13 masks all stray pieces and holes were < 500 px or > 1600 px


def encode(mask):
    flat = mask.reshape(-1).astype(np.int8)
    change = np.flatnonzero(np.diff(flat)) + 1
    runs = np.diff(np.r_[0, change, flat.size])
    return ([0] if flat[0] else []) + runs.tolist()


def decode(runs, h, w):
    values = np.arange(len(runs)) % 2
    return np.repeat(values, runs).astype(bool).reshape(h, w)


def tidy(mask, min_px=TIDY_PX):
    """Drop pieces and fill holes (gaps not touching the image border) smaller than min_px: brush leftovers."""
    _, cc, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    small = [i for i in range(1, len(st)) if st[i, 4] < min_px]
    out = mask & ~np.isin(cc, small)
    _, gc, gs, _ = cv2.connectedComponentsWithStats((~out).astype(np.uint8), 4)
    border = set(np.r_[gc[0], gc[-1], gc[:, 0], gc[:, -1]].tolist())
    holes = [i for i in range(1, len(gs)) if i not in border and gs[i, 4] < min_px]
    return out | np.isin(gc, holes), len(small), len(holes)


def export():
    out = {}
    for f in sorted(glob.glob(f"{LUS}/masks/Patient*/*.png")):
        patient, name = f.split("/")[-2], os.path.basename(f)[:-4]
        out[f"{patient}/{name}"] = encode(cv2.imread(f, 0) > 0)
    with open(f"{LUS}/sam_masks.js", "w") as f:
        f.write("const SAM_MASKS = " + json.dumps(out, separators=(",", ":")) + ";\n")
    print(f"wrote {LUS}/sam_masks.js: {len(out)} masks, {os.path.getsize(f'{LUS}/sam_masks.js') // 1024} KB")


def import_(path):
    data = json.load(open(os.path.expanduser(path)))
    out_dir = f"{LUS}/masks_manual"
    os.makedirs(f"{out_dir}/review", exist_ok=True)
    rows = []
    for key in sorted(data):
        patient, name = key.split("/")
        img = cv2.imread(f"{LUS}/data/Dataset/{patient}/Lap Images/{name}.png")
        h, w = img.shape[:2]
        assert sum(data[key]["rle"]) == h * w, f"{key}: mask size does not match the image"
        mask, n_specks, n_holes = tidy(decode(data[key]["rle"], h, w))
        os.makedirs(f"{out_dir}/{patient}", exist_ok=True)
        cv2.imwrite(f"{out_dir}/{patient}/{name}.png", mask.astype(np.uint8) * 255)

        sam_path = f"{LUS}/masks/{patient}/{name}.png"
        sam = cv2.imread(sam_path, 0) > 0 if os.path.exists(sam_path) else np.zeros_like(mask)
        iou = (mask & sam).sum() / max((mask | sam).sum(), 1)
        view = img.copy()
        view[mask] = (0.6 * view[mask] + 0.4 * np.array([255, 170, 0])).astype(np.uint8)
        cv2.drawContours(view, cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0],
                         -1, (255, 255, 255), max(2, w // 480))
        cv2.imwrite(f"{out_dir}/review/{patient}_{name}.jpg",
                    cv2.resize(view, (1280, round(1280 * h / w)), interpolation=cv2.INTER_AREA))
        flags = ("done" if data[key]["done"] else "NOT marked done") + (", edited" if data[key]["edited"] else ", SAM unedited")
        note = (f"{100 * mask.mean():.0f} % of the image; overlap with SAM's mask (IoU) {iou:.2f}; {flags}; "
                f"tidied: {n_specks} specks dropped, {n_holes} pinholes filled")
        print(f"{key}: {note}")
        rows.append((key, f"review/{patient}_{name}.jpg", note))

    with open(f"{out_dir}/review.html", "w") as f:
        f.write("<!doctype html><meta charset='utf-8'><title>LUS manual masks</title>"
                "<body style='background:#16181c;color:#e8eaed;font:14px system-ui;max-width:1300px;margin:20px auto'>"
                "<h1 style='font-size:18px'>LUS liver masks (hand-corrected)</h1>")
        for key, review, note in rows:
            f.write(f"<h2 style='font-size:15px;margin:22px 0 4px'>{key}</h2><div style='color:#9aa3ad'>{note}</div>"
                    f"<img src='{review}' style='width:100%;margin-top:6px'>")
    print(f"{len(rows)} masks written to {out_dir}/; review page {out_dir}/review.html")


if __name__ == "__main__":
    if sys.argv[1:2] == ["export"]:
        export()
    elif sys.argv[1:2] == ["import"] and len(sys.argv) == 3:
        import_(sys.argv[2])
    else:
        print(__doc__)
