# lus_eval: evaluation on LLR-LUS

LLR-LUS (Rabbani et al., 2022) is a set of 4 liver-surgery patients with 46 laparoscope images. Each image comes with a
tracked ultrasound slice of the tumour, which is the ground truth. This folder can do two things:

1. **Score a registration method**: how far its predicted tumours are from the ultrasound tumours (`score.py`).
2. **Make liver masks** for the 46 images (LLR-LUS ships none). All 46 are already drawn by hand; the tools are
   kept so they can be redrawn or checked.

> LLR-LUS is research-use only. Everything derived from it (masks, prompts, results) stays under `dataset/`, which is
> git-ignored. Only code lives here.

## Files

| File | What it does |
|---|---|
| `score.py` | Official LLR-LUS score (TRE in mm, and IC), a step-by-step port of the authors' `Evaluation.m` |
| `sam_masks.py` | Step 1 of masking: turns your box and click prompts into first-guess masks with SAM |
| `manual_masks.py` | Step 2: sends SAM's masks to the editor (`export`), and saves your corrected masks as PNGs (`import`) |
| `tools/prompt_tool.html` | Browser page for drawing the SAM prompts |
| `tools/mask_editor.html` | Browser page for correcting or redrawing each mask |
| `RESULTS.md` | Record of the earlier Self-P2IR study on LLR-LUS (archived; see its note) |

Run everything from the repository root.

## Score a method

A method submits one tumour mesh per image, as a binary STL in OpenGL camera axes:
`<method-dir>/PatientN/<image>.stl`, for example `Patient1/02.stl`.

```bash
python lus_eval/score.py --method-dir "dataset/lus-evaluation/results/Registration Methods/Koo"
```

The output is a TRE and IC per patient, then the overall TRE.

**Check:** scoring the published Koo et al. results reproduces their paper: 9.49 / 38.95 / 25.05 / 18.35 mm,
overall **22.96 mm**, IC 0 of 4 (checked 2026-11-27).

## Liver masks

The final masks, 46 PNGs with 255 = liver, are in **`dataset/lus-evaluation/masks_manual/PatientN/NN.png`**. They follow
P2I-LReg's convention: visible liver surface only, with no falciform ligament, fat, tools or probe. Open
`masks_manual/review.html` to see every image with its mask.

How they were made, and how to redo them:

1. Open `tools/prompt_tool.html`: one box per image, plus "liver" and "not liver" clicks. Download the prompts.
2. `python lus_eval/sam_masks.py --prompts ~/Downloads/lus_prompts.json` writes SAM's masks to
   `dataset/lus-evaluation/masks/`. This needs `dataset/models/sam_vit_h_4b8939.pth` and the `segment-anything` package.
3. `python lus_eval/manual_masks.py export` writes `dataset/lus-evaluation/sam_masks.js`, the editor's starting masks.
4. Open `tools/mask_editor.html`, correct each mask, and press Save. The editor also keeps your work in the browser.
5. `python lus_eval/manual_masks.py import ~/Downloads/lus_manual_masks.json` writes `masks_manual/` and the review
   page. Stray brush specks and pinholes under 500 px are cleaned up.

## Data layout (`dataset/lus-evaluation/`)

| Path | Contents |
|---|---|
| `data/Dataset/PatientN/` | LLR-LUS as downloaded: laparoscope images, ultrasound, calibration, preoperative model |
| `results/Registration Methods/` | Published results of Koo et al. and Adagolodjo et al. (score these as a check) |
| `masks_manual/` | **The hand-drawn liver masks** |
| `annotations/lus_manual_masks.json` | Raw download from the mask editor (the source of `masks_manual/`) |
| `annotations/lus_prompts.json` | The SAM prompts |
| `annotations/goal_vlm/` | Earlier per-image scene annotations (rescued from the old `goal/vlm/`) |
| `masks/` | SAM's first-guess masks |
| `koo_contours.js`, `sam_masks.js` | Data the two browser pages load |
