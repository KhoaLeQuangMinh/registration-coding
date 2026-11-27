"""
Liver masks for the LLR-LUS laparoscope images, from hand-drawn prompts, with SAM (ViT-H).

Self-P2IR needs a liver mask per image and LLR-LUS has none. The prompts come from
lus_eval/tools/prompt_tool.html: per image one box, plus "liver" and "not liver" marks.
A click is one point; a scribble becomes points every 2 % of the image width along the stroke.
SAM gets the box (if drawn) and all points. With a box it returns one mask; without one, its best-scored
of three. (Choosing among SAM's candidates by agreement with the marks was tried and dropped: unmarked
non-liver costs nothing, so it picked masks that swallowed whole fat areas.) Where a not-liver stroke
still lies inside the mask, SAM segments the object under that stroke (stroke = this object, liver points
= not this object) and it is taken out. The mask is then cleaned using the prompts only: pieces holding no liver point are dropped
(SAM leaves specks, even outside the box), and holes holding no not-liver point are filled (glare).
Last, the hand corrections ("add area" / "remove area" loops) are applied as drawn, without SAM.

Convention, as in P2I-LReg's own real labels: the mask is the VISIBLE liver surface. Tissue lying on
the liver (falciform ligament, fat, tools, the probe) is left out even though liver lies beneath it.

Everything written here is derived from LLR-LUS, so it stays under dataset/ (not in the repo):
  dataset/lus-evaluation/masks/PatientN/NN.png         255 = liver
  dataset/lus-evaluation/masks/review/PatientN_NN.jpg  the image with the mask and the prompts drawn on it
  dataset/lus-evaluation/masks/review.html             all review images on one page

Usage: python lus_eval/sam_masks.py --prompts ~/Downloads/lus_prompts.json [dataset/lus-evaluation/requested_cuts.json]
           [--checkpoint dataset/models/sam_vit_h_4b8939.pth] [--lus dataset/lus-evaluation]
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

SPACING = 0.02  # scribble points, as a fraction of the image width


def points(marks, width):
    """All prompt points (n x 2) and their labels (1 liver, 0 not liver)."""
    xy, labels = [], []
    for m in marks:
        if m["kind"] not in ("pos", "neg"):
            continue
        p = np.asarray(m["pts"], float)
        if len(p) > 1:  # resample the stroke at even steps, both ends kept
            along = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
            n = max(2, int(round(along[-1] / (SPACING * width))) + 1)
            at = np.linspace(0, along[-1], n)
            p = np.stack([np.interp(at, along, p[:, 0]), np.interp(at, along, p[:, 1])], axis=1)
        xy += p.tolist()
        labels += [1 if m["kind"] == "pos" else 0] * len(p)
    return np.asarray(xy, float).reshape(-1, 2), np.asarray(labels, int)


def cut_spills(predictor, mask, marks, width):
    """Where a not-liver stroke lies inside the mask, ask SAM for the object under that stroke (its points
    as "this object", every liver point as "not this object") and take that object out of the mask.
    SAM's candidates that would take any liver point are not used. Returns the mask and the pixels removed."""
    h, w = mask.shape
    pos, _ = points([m for m in marks if m["kind"] == "pos"], width)
    removed = 0
    for m in marks:
        if m["kind"] != "neg":
            continue
        neg, _ = points([dict(m, kind="pos")], width)
        hit = mask[neg[:, 1].round().astype(int).clip(0, h - 1), neg[:, 0].round().astype(int).clip(0, w - 1)]
        if not hit.any():
            continue
        with torch.no_grad():
            objs, scores, _ = predictor.predict(point_coords=np.vstack([neg, pos]),
                                                point_labels=np.r_[np.ones(len(neg)), np.zeros(len(pos))].astype(int),
                                                multimask_output=True)
        at = (pos[:, 1].round().astype(int).clip(0, h - 1), pos[:, 0].round().astype(int).clip(0, w - 1))
        ok = [i for i in range(len(objs)) if not objs[i][at].any()]
        if ok:
            obj = objs[max(ok, key=lambda i: scores[i])]
            removed += int((mask & obj).sum())
            mask = mask & ~obj
    return mask, removed


def clean(mask, xy, labels):
    """Keep only the pieces of the mask that hold a liver point, and fill the holes that hold no
    not-liver point (glare on the liver). Uses only the prompts: no size thresholds."""
    h, w = mask.shape
    at = lambda m: (xy[labels == m][:, 1].round().astype(int).clip(0, h - 1),  # noqa: E731
                    xy[labels == m][:, 0].round().astype(int).clip(0, w - 1))
    _, pieces = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    mask = np.isin(pieces, list(set(pieces[at(1)].tolist()) - {0}))
    _, gaps = cv2.connectedComponents((~mask).astype(np.uint8), connectivity=4)
    outside = set(np.r_[gaps[0], gaps[-1], gaps[:, 0], gaps[:, -1]].tolist())
    marked = set(gaps[at(0)].tolist())
    holes = [g for g in range(1, gaps.max() + 1) if g not in outside and g not in marked]
    return mask | np.isin(gaps, holes)


def patch(mask, marks):
    """The hand corrections, in the order drawn: an "add" loop is put into the mask, a "cut" loop taken out."""
    mask = mask.copy()
    for m in marks:
        if m["kind"] in ("add", "cut"):
            area = np.zeros(mask.shape, np.uint8)
            cv2.fillPoly(area, [np.asarray(m["pts"], np.int32)], 1)
            mask = mask | area.astype(bool) if m["kind"] == "add" else mask & ~area.astype(bool)
    return mask


def review_image(img, mask, box, xy, labels, marks=()):
    out = img.copy()
    out[mask] = (0.6 * out[mask] + 0.4 * np.array([255, 200, 0])).astype(np.uint8)  # liver tinted blue
    r = max(2, img.shape[1] // 480)
    cv2.drawContours(out, cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0],
                     -1, (255, 255, 255), r)
    if box is not None:
        cv2.rectangle(out, tuple(box[:2]), tuple(box[2:]), (68, 199, 242), r)
    for m in marks:  # hand corrections: blue = added, orange = removed
        if m["kind"] in ("add", "cut"):
            cv2.polylines(out, [np.asarray(m["pts"], np.int32)], True,
                          (255, 166, 62) if m["kind"] == "add" else (46, 157, 255), r)
    for (x, y), lab in zip(xy.round().astype(int), labels):
        cv2.circle(out, (int(x), int(y)), 3 * r, (255, 255, 255), -1)
        cv2.circle(out, (int(x), int(y)), 2 * r, (113, 191, 47) if lab else (77, 72, 229), -1)
    return cv2.resize(out, (1280, round(1280 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, nargs="+",
                    help="prompt files; marks of the same image are joined, the first file gives the box")
    ap.add_argument("--checkpoint", default="dataset/models/sam_vit_h_4b8939.pth")
    ap.add_argument("--lus", default="dataset/lus-evaluation")
    args = ap.parse_args()

    prompts = {}
    for path in args.prompts:
        for key, P in json.load(open(os.path.expanduser(path))).items():
            if key in prompts:
                prompts[key]["marks"] += P["marks"]
            else:
                prompts[key] = P
    predictor = SamPredictor(sam_model_registry["vit_h"](checkpoint=args.checkpoint).eval())
    out_dir = os.path.join(args.lus, "masks")
    os.makedirs(os.path.join(out_dir, "review"), exist_ok=True)

    rows = []
    for key in sorted(prompts):  # "Patient1/02"
        patient, name = key.split("/")
        img = cv2.imread(os.path.join(args.lus, "data", "Dataset", patient, "Lap Images", name + ".png"))
        P = prompts[key]
        assert img.shape[:2] == (P["h"], P["w"]), f"{key}: the prompts were drawn on another image size"
        xy, labels = points(P["marks"], P["w"])
        box = P.get("box")
        if box is not None or (labels == 1).any():
            predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            with torch.no_grad():
                # with a box, one mask; with points only, SAM's three guesses and the best-scored one
                masks, scores, _ = predictor.predict(point_coords=xy if len(xy) else None,
                                                     point_labels=labels if len(xy) else None,
                                                     box=None if box is None else np.asarray(box, float),
                                                     multimask_output=box is None)
            best = int(np.argmax(scores))
            raw, scores = masks[best], scores[best:best + 1]
        else:  # drawn entirely by hand with "add" loops
            raw, scores = np.zeros(img.shape[:2], bool), np.array([np.nan])
        cleaned, spilled = clean(raw, xy, labels), 0
        if box is not None or (labels == 1).any():  # only conflicts left after the clean-up
            cleaned, spilled = cut_spills(predictor, cleaned, P["marks"], P["w"])
            cleaned = clean(cleaned, xy, labels) if spilled else cleaned
        mask = patch(cleaned, P["marks"])
        os.makedirs(os.path.join(out_dir, patient), exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, patient, name + ".png"), mask.astype(np.uint8) * 255)
        review = f"review/{patient}_{name}.jpg"
        cv2.imwrite(os.path.join(out_dir, review), review_image(img, mask, box, xy, labels, P["marks"]))

        hit = mask[xy[:, 1].round().astype(int).clip(0, P["h"] - 1), xy[:, 0].round().astype(int).clip(0, P["w"] - 1)]
        in_box = "no box" if box is None else f"{100 * mask[box[1]:box[3], box[0]:box[2]].mean():.0f} % of the box"
        note = (f"{100 * mask.mean():.0f} % of the image, {in_box}; "
                f"liver points inside the mask {int(hit[labels == 1].sum())}/{int((labels == 1).sum())}, "
                f"not-liver points inside {int(hit[labels == 0].sum())}/{int((labels == 0).sum())}; "
                f"SAM score {scores[0]:.2f}; spills under not-liver strokes removed {spilled} px; clean-up removed {int((raw & ~cleaned).sum())} px, "
                f"filled {int((cleaned & ~raw).sum())} px; hand corrections added {int((mask & ~cleaned).sum())} px, "
                f"removed {int((cleaned & ~mask).sum())} px")
        print(f"{key}: {note}", flush=True)
        rows.append((key, review, note))

    with open(os.path.join(out_dir, "review.html"), "w") as f:
        f.write("<!doctype html><meta charset='utf-8'><title>LUS liver masks</title>"
                "<body style='background:#16181c;color:#e8eaed;font:14px system-ui;max-width:1300px;margin:20px auto'>"
                "<h1 style='font-size:18px'>LUS liver masks (SAM)</h1>"
                "<p>Blue tint and white line = mask. Yellow = your box. Green dots = liver points, red dots = not liver.</p>")
        for key, review, note in rows:
            f.write(f"<h2 style='font-size:15px;margin:22px 0 4px'>{key}</h2><div style='color:#9aa3ad'>{note}</div>"
                    f"<img src='{review}' style='width:100%;margin-top:6px'>")
    print("wrote", os.path.join(out_dir, "review.html"))


if __name__ == "__main__":
    main()
