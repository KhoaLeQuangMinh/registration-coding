"""
Real vs false edges of the LLR-LUS liver masks (the user's rule: if liver continues beyond a stretch of the mask's
border, that stretch is FALSE; if the liver really ends there, it is REAL).

auto    first pass for all 46 images, in UNDISTORTED full-resolution pixels:
        the mask border is split into stretches of ~STEP px; a stretch is FALSE if most of it lies next to
          - the dark area outside the scope circle, or
          - a "cover": unsaturated (grey / white / black) non-liver pixels touching the mask = probe, tools, ligament;
        otherwise REAL. Optional per-image corrections in goal/vlm/edge_fixes.json:
          {"Patient1/02": {"cover": [[x0,y0,x1,y1], ...], "not_cover": [[...]]}}  (non-liver pixels in a box)
        writes dataset/lus-evaluation/edges/edges.json, undistorted JPGs for the editor, review sheets
sheets  contact sheets (6 images each) for review: goal/out/edge_sheets/sheet_N.jpg
import  <downloaded json from dataset/lus-evaluation/edge_editor.html>  -> edges/edges.json (the reviewed labels)
Usage: python goal/lus_edges.py auto | import ~/Downloads/lus_edges.json
"""
import json
import os
import re
import sys

import cv2
import numpy as np

LUS = "dataset/lus-evaluation"
DSET = f"{LUS}/data/Dataset"
OUT = f"{LUS}/edges"
STEP = 40          # stretch length, px at 1920 wide
NEAR = 14          # a border point is "next to" hidden stuff within this distance, px at 1920 wide


def images():
    meta = json.load(open(f"{LUS}/selfp2ir/meta.json"))
    return sorted(meta, key=lambda k: (k.split("/")[0], int(k.split("/")[1])))


def undistort_maps(p, shape):
    xml = open(f"{DSET}/{p}/Lap Images/CameraParameters").read()
    g = lambda k: float(re.search(f"<{k}>(.*?)</{k}>", xml).group(1))
    K = np.array([[g("fx"), 0, g("cx")], [0, g("fy"), g("cy")], [0, 0, 1]])
    dist = np.array([g("k1"), g("k2"), g("p1"), g("p2"), g("k3")])
    return cv2.initUndistortRectifyMap(K, dist, None, K, (shape[1], shape[0]), cv2.CV_32FC1)


def load(key):
    p, name = key.split("/")
    img = cv2.imread(f"{DSET}/{p}/Lap Images/{name}.png")
    mx, my = undistort_maps(p, img.shape)
    img = cv2.remap(img, mx, my, cv2.INTER_LINEAR)
    M = cv2.remap((cv2.imread(f"{LUS}/masks_manual/{p}/{name}.png", 0) > 0).astype(np.uint8), mx, my, cv2.INTER_NEAREST) > 0
    return img, M


def hidden_regions(img, M, fixes):
    s = img.shape[1] / 1920
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    grey = hsv[..., 2]
    # outside the scope: darkness connected to the image CORNERS only (dark tissue inside the view stays), then a
    # generous margin; plus the image border itself (the mask touching it = liver running out of the picture)
    corners = [grey[:20, :20], grey[:20, -20:], grey[-20:, :20], grey[-20:, -20:]]
    thr = min(60, np.median(np.concatenate([c.ravel() for c in corners])) + 18)
    n, cc = cv2.connectedComponents((grey < thr).astype(np.uint8))
    hs, ws = grey.shape
    seeds = {cc[y, x] for y, x in [(2, 2), (2, ws - 3), (hs - 3, 2), (hs - 3, ws - 3)]} - {0}
    scope = np.isin(cc, list(seeds)) if seeds else np.zeros_like(M)
    scope[:2] = scope[-2:] = True; scope[:, :2] = scope[:, -2:] = True
    scope = cv2.dilate(scope.astype(np.uint8), np.ones((int(60 * s) | 1,) * 2, np.uint8)) > 0
    # covers: unsaturated non-liver pixels (probe shaft, checkerboard, tools, ligament) that touch the mask
    cand = (hsv[..., 1] < 70) & ~M & ~scope
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, cc, st, _ = cv2.connectedComponentsWithStats(cand)
    ring = cv2.dilate(M.astype(np.uint8), np.ones((int(9 * s) | 1,) * 2, np.uint8)) > 0
    cover = np.zeros_like(M)
    for i in range(1, n):
        comp = cc == i
        if st[i, cv2.CC_STAT_AREA] > 0.002 * M.size and (comp & ring).sum() > 30 * s:
            cover |= comp
    for box in fixes.get("cover", []):
        x0, y0, x1, y1 = (np.array(box) * s).astype(int); b = np.zeros_like(M); b[y0:y1, x0:x1] = True
        cover |= b & ~M
    for box in fixes.get("not_cover", []):
        x0, y0, x1, y1 = (np.array(box) * s).astype(int); b = np.zeros_like(M); b[y0:y1, x0:x1] = True
        cover &= ~b
    return scope, cover


def stretches(M, scope, cover):
    s = M.shape[1] / 1920
    near = cv2.dilate((scope | cover).astype(np.uint8), np.ones((int(2 * NEAR * s) | 1,) * 2, np.uint8)) > 0
    cnts, _ = cv2.findContours(M.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = []
    step = max(4, int(STEP * s))
    for c in cnts:
        c = c[:, 0]
        if len(c) < 20:
            continue
        for i in range(0, len(c), step):
            seg = c[i:i + step + 1]
            if len(seg) < 3:
                continue
            frac = near[seg[:, 1], seg[:, 0]].mean()
            out.append(dict(points=seg[::max(1, len(seg) // 8)].tolist() + [seg[-1].tolist()], label="false" if frac > 0.5 else "real"))
    return out


def auto():
    os.makedirs(OUT, exist_ok=True)
    fixes_all = json.load(open("goal/vlm/edge_fixes.json")) if os.path.exists("goal/vlm/edge_fixes.json") else {}
    data = {}
    for key in images():
        img, M = load(key)
        scope, cover = hidden_regions(img, M, fixes_all.get(key, {}))
        segs = stretches(M, scope, cover)
        data[key] = dict(size=[img.shape[1], img.shape[0]], segments=segs)
        p, name = key.split("/")
        os.makedirs(f"{OUT}/img/{p}", exist_ok=True)
        cv2.imwrite(f"{OUT}/img/{p}/{name}.jpg", cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])
        nf = sum(sg["label"] == "false" for sg in segs)
        print(f"{key}: {len(segs)} stretches, {nf} false, {len(segs) - nf} real", flush=True)
    json.dump(data, open(f"{OUT}/edges_auto.json", "w"))
    if not os.path.exists(f"{OUT}/edges.json"):
        json.dump(data, open(f"{OUT}/edges.json", "w"))
    write_js(json.load(open(f"{OUT}/edges.json")))
    sheets(data)


def draw(key, entry, w=640):
    p, name = key.split("/")
    img = cv2.imread(f"{OUT}/img/{p}/{name}.jpg")
    s = 1920 / entry["size"][0]
    for sg in entry["segments"]:
        pts = (np.array(sg["points"]) * s).astype(np.int32)
        cv2.polylines(img, [pts], False, (60, 60, 240) if sg["label"] == "false" else (60, 220, 60), 9)
    for x in range(0, 1920, 240):
        cv2.putText(img, str(x), (x + 6, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    for y in range(270, 1080, 270):
        cv2.putText(img, str(y), (6, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    img = cv2.resize(img, (w, w * 9 // 16), interpolation=cv2.INTER_AREA)
    cv2.putText(img, key, (8, w * 9 // 16 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(img, key, (8, w * 9 // 16 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def sheets(data):
    os.makedirs("goal/out/edge_sheets", exist_ok=True)
    keys = list(data)
    for i in range(0, len(keys), 6):
        tiles = [draw(k, data[k]) for k in keys[i:i + 6]]
        while len(tiles) < 6:
            tiles.append(np.full_like(tiles[0], 255))
        sheet = np.vstack([np.hstack(tiles[j:j + 3]) for j in (0, 3)])
        cv2.imwrite(f"goal/out/edge_sheets/sheet_{i // 6 + 1}.jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    print("wrote goal/out/edge_sheets/sheet_1..%d.jpg (red = false edge, green = real edge)" % ((len(keys) + 5) // 6))


def write_js(data):
    """The editor (dataset/lus-evaluation/edge_editor.html) reads this."""
    open(f"{OUT}/edges_data.js", "w").write("window.EDGES = " + json.dumps(data) + ";")


def do_import(path):
    got = json.load(open(os.path.expanduser(path)))
    data = json.load(open(f"{OUT}/edges.json"))
    for key, entry in got.items():
        data[key]["segments"] = entry["segments"]
    json.dump(data, open(f"{OUT}/edges.json", "w"))
    write_js(data)
    sheets(data)
    print(f"imported {len(got)} images into {OUT}/edges.json")


if __name__ == "__main__":
    if sys.argv[1] == "auto":
        auto()
    elif sys.argv[1] == "sheets":
        sheets(json.load(open(f"{OUT}/edges.json")))
    elif sys.argv[1] == "import":
        do_import(sys.argv[2])
