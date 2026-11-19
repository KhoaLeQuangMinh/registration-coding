"""
The "slide": move Self-P2IR's registered CT liver along the camera's line of sight until its outline matches the
liver mask, and see what that does to the tumour error. No training, no ground truth: only the mask, the
calibrated camera and the CT (whose real size fixes how big it should look at each distance).

Per image, starting from Self-P2IR's pose (main LUS run, CT as given):
  every point moves by (k - 1) * c, where c is the placed liver's centre: k scales the liver's distance from
  the camera along the ray through its centre; orientation unchanged. k = 1 is Self-P2IR's answer.
  For k = 0.30 ... 1.50 (step 0.01):
    outline Dice  the CT liver drawn with the calibrated camera vs the liver mask, inside the scope's round view
                  (both at 480 px wide)
    tumour error  mean distance from the ultrasound tumour outline to the nearest point of the moved CT tumour
  The slide picks the k with the best outline Dice. Compared with k = 1 and with the oracle run.
  Also, for viewing: the CT tumour shifted (orientation kept) so the ultrasound outline lies on it best --
  what a correct registration would put there. Its leftover misfit is how well the CT tumour's shape agrees
  with the ultrasound slice (a floor for any rigid registration with this orientation).

Writes dataset/lus-evaluation/slide_data.js for dataset/lus-evaluation/slide_explorer.html (LUS-derived:
stays under dataset/) and prints the batch result.

Usage: python studies/benchmark/lus_slide.py
"""
import json
import re

import cv2
import numpy as np
import open3d as o3d

import lus_evaluation as ev

LUS = "dataset/lus-evaluation"
RES = f"{LUS}/selfp2ir_results"
F = np.diag([1.0, -1.0, -1.0])
KS = np.round(np.arange(0.30, 1.5001, 0.01), 2)
SMALL = 480  # width of the images the outline is compared on


def rle(mask):
    flat = mask.reshape(-1).astype(np.int8)
    change = np.flatnonzero(np.diff(flat)) + 1
    runs = np.diff(np.r_[0, change, flat.size])
    return ([0] if flat[0] else []) + runs.tolist()


def tumour_error(T, prof):
    near = T[((prof[:, None] - T[None]) ** 2).sum(2).argmin(1)]
    d = near - prof
    return float(np.linalg.norm(d, axis=1).mean()), float(d[:, 2].mean())


def fit_to_slice(T, prof, iters=50):
    """The shift that moves the tumour (vertices T) so the ultrasound outline lies on it best: start at the
    outline's centre, then repeat: nearest tumour points, shift by their mean offset. Orientation unchanged.
    This is where the tumour REALLY is (given that orientation): what correct registration looks like."""
    shift = prof.mean(0) - T.mean(0)
    for _ in range(iters):
        X = T + shift
        near = X[((prof[:, None] - X[None]) ** 2).sum(2).argmin(1)]
        shift = shift + (prof - near).mean(0)
    return shift, tumour_error(T + shift, prof)[0]


def oracle_errors():
    """Per-image oracle tumour error, computed the same way, from the oracle run."""
    meta = json.load(open(f"{RES}/lus_oracle/meta.json"))
    frames = {(f"{r['patient']:02d}", r["frame"]): r for r in json.load(open(f"{RES}/lus_oracle/real_frames.json"))["frames"]}
    out = {}
    for key, m in meta.items():
        p, name = key.split("/")
        r = frames[(m["id"], m["frame"])]
        V = np.asarray(o3d.io.read_triangle_mesh(f"{ev.DATASET}/{p}/Preoperative Model/Tumour.ply").vertices)
        T = 1000.0 * (V / 1000.0 @ np.array(r["R"]).T + np.array(r["t"])) @ F.T + np.array(m["shift_mm"])
        out[key] = tumour_error(T, ev.sample_profile(ev.profile(ev.DATASET, p[7:], name), ev.M_SAMPLES).T)[0]
    return out


def main():
    meta = json.load(open(f"{LUS}/selfp2ir/meta.json"))
    frames = {(f"{r['patient']:02d}", r["frame"]): r for r in json.load(open(f"{RES}/lus_results/real_frames.json"))["frames"]}
    oracle = oracle_errors()
    data, rows = {}, []
    for key, m in sorted(meta.items()):
        p, name = key.split("/")
        r = frames[(m["id"], m["frame"])]
        place = lambda V: 1000.0 * (V / 1000.0 @ np.array(r["R"]).T + np.array(r["t"])) @ F.T + np.array(m["shift_mm"])  # noqa: E731
        liver = o3d.io.read_triangle_mesh(f"{ev.DATASET}/{p}/Preoperative Model/Liver.ply").simplify_quadric_decimation(2500)
        L, LF = place(np.asarray(liver.vertices)), np.asarray(liver.triangles)
        tum = o3d.io.read_triangle_mesh(f"{ev.DATASET}/{p}/Preoperative Model/Tumour.ply")
        T = place(np.asarray(tum.vertices))
        tum_small = tum.simplify_quadric_decimation(300)
        TS, TF = place(np.asarray(tum_small.vertices)), np.asarray(tum_small.triangles)
        prof = ev.sample_profile(ev.profile(ev.DATASET, p[7:], name), ev.M_SAMPLES).T
        c = L.mean(0)
        ideal_shift, slice_fit = fit_to_slice(T, prof)

        W, H = m["size"]
        s = SMALL / W
        h = round(H * s)
        img = cv2.imread(f"{ev.DATASET}/{p}/Lap Images/{name}.png")
        small = cv2.resize(img, (SMALL, h), interpolation=cv2.INTER_AREA)
        view = cv2.morphologyEx((cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) > 12).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0
        mask = cv2.resize((cv2.imread(f"{LUS}/masks_manual/{p}/{name}.png", 0) > 0).astype(np.uint8), (SMALL, h), interpolation=cv2.INTER_NEAREST) > 0
        mask &= view

        dice, err, depth = [], [], []
        for k in KS:
            X = L + (k - 1) * c
            if (X[:, 2] <= 1).any():
                dice.append(0.0); err.append(np.nan); depth.append(np.nan); continue
            uv = np.stack([(m["fx"] * X[:, 0] / X[:, 2] + m["cx"]) * s, (m["fy"] * X[:, 1] / X[:, 2] + m["cy"]) * s], 1)
            sil = np.zeros((h, SMALL), np.uint8)
            cv2.fillPoly(sil, list(np.round(uv[LF] * 16).astype(np.int32)), 1, lineType=cv2.LINE_8, shift=4)
            sil = (sil > 0) & view
            dice.append(float(200.0 * (sil & mask).sum() / max(sil.sum() + mask.sum(), 1)))
            e, dz = tumour_error(T + (k - 1) * c, prof)
            err.append(e); depth.append(dz)
        dice, err = np.array(dice), np.array(err)
        kb, kt = int(np.argmax(dice)), int(np.nanargmin(err))
        i1 = int(np.argmin(abs(KS - 1.0)))
        rows.append(dict(key=key, err1=err[i1], err_slide=err[kb], k_slide=KS[kb], err_best=err[kt], k_best=KS[kt],
                         dice1=dice[i1], dice_slide=dice[kb], oracle=oracle[key], dist1=float(np.linalg.norm(c))))
        data[key] = dict(
            photo=f"data/Dataset/{p}/Lap%20Images/{name}.png", W=W, H=H, fx=m["fx"], fy=m["fy"], cx=m["cx"], cy=m["cy"],
            small=[SMALL, h], mask=rle(mask), centre=np.round(c, 1).tolist(),
            liver=np.round(L, 1).tolist(), liver_faces=LF.tolist(), tumour=np.round(TS, 1).tolist(), tumour_faces=TF.tolist(),
            profile=np.round(prof, 1).tolist(), ks=KS.tolist(), dice=np.round(dice, 2).tolist(),
            err=[None if np.isnan(x) else round(float(x), 2) for x in err],
            depth=[None if np.isnan(x) else round(float(x), 2) for x in depth],
            k_slide=float(KS[kb]), k_best=float(KS[kt]), oracle_err=round(oracle[key], 1),
            tumour_ideal=np.round(TS + ideal_shift, 1).tolist(), slice_fit=round(slice_fit, 1),
            ideal_offset=round(float(np.linalg.norm(ideal_shift)), 1),
            selfp2ir_dice=round(r["dice"], 1))
        print(f"{key}: k=1 error {err[i1]:5.1f} mm (outline Dice {dice[i1]:4.1f}) | slide k={KS[kb]:.2f} error {err[kb]:5.1f} mm "
              f"(Dice {dice[kb]:4.1f}) | best possible k={KS[kt]:.2f} error {err[kt]:5.1f} | oracle {oracle[key]:5.1f}", flush=True)

    with open(f"{LUS}/slide_data.js", "w") as f:
        f.write("const SLIDE = " + json.dumps(data, separators=(",", ":")) + ";\n")
    a = lambda k: np.array([x[k] for x in rows])  # noqa: E731
    print(f"\nALL 46 (median tumour error): Self-P2IR {np.median(a('err1')):.1f} mm | slide {np.median(a('err_slide')):.1f} mm | "
          f"best possible slide {np.median(a('err_best')):.1f} mm | oracle run {np.median(a('oracle')):.1f} mm")
    print(f"  slide better than Self-P2IR in {(a('err_slide') < a('err1')).sum()}/46; slide k median {np.median(a('k_slide')):.2f} "
          f"(best possible k median {np.median(a('k_best')):.2f})")
    for pid in range(1, 5):
        g = [x for x in rows if x["key"].startswith(f"Patient{pid}/")]
        med = lambda k: np.median([x[k] for x in g])  # noqa: E731
        print(f"  Patient {pid}: Self-P2IR {med('err1'):5.1f} | slide {med('err_slide'):5.1f} (k {med('k_slide'):.2f}) | "
              f"best possible {med('err_best'):5.1f} (k {med('k_best'):.2f}) | oracle {med('oracle'):5.1f}")
    print(f"wrote {LUS}/slide_data.js")


if __name__ == "__main__":
    main()
