# Results: Self-P2IR on real patients with tumour ground truth (LLR-LUS)

> **Archived record (2026-11-12 to 2026-11-17).** Only the scorer (`score.py`, formerly `lus_evaluation.py`) and the
> mask tools were kept in this repo. The experiment scripts named below (`lus_oracle.py`, `lus_slide.py`, the Self-P2IR
> prep/score scripts and Kaggle notebooks) are in git tag `archive-2026-11`, under `studies/benchmark/`
> (`git show archive-2026-11:studies/benchmark/<file>`).

**Question:** does Self-P2IR's own score (Dice) tell us where the tumour is, on real patients?
**Model:** Self-P2IR (Zhou et al., TMI 2025), released weights (sha256 `e83a9110…`), code at commit `60d6c1b`, run as the
authors' real-frame tester. Not retrained. Its trained fold has real test patients 01, 02, 13, 17, 18.
**Data:** LLR-LUS (Rabbani et al. 2022): 4 patients, 46 laparoscope images, each with a tracked ultrasound slice of the
tumour (the ground truth). Research use only; derived files stay under `dataset/`.
**Status:** exploratory. Unlike the pre-registered benchmark (`RESULTS.md` in tag `archive-2026-11`), these tests were **not pre-registered**; they were decided one
after another (2026-11-12 to 2026-11-17). Run on Kaggle (2× T4) and on a laptop.

## Summary

| # | Finding | Status |
|---|---|---|
| 1 | Dice hides a 7 cm tumour error | solid |
| 2 | The error comes mostly from the cloud's distance (the scale step), not the network | solid (oracle test) |
| 3 | Dice cannot see depth: the tumour error drops by ~50 mm while Dice stays put | solid (oracle test) |
| 4 | Scene context halves the registration error: a coarse reading of the photo (which liver part is visible, which way the head is) used to start the search takes our method from 65.8 to 32.8 mm | measured, 4 patients, one reader (Claude) |
| 5 | The geometric score chooses wrong on real data: the depth-shape term prefers a pose 30-100 mm worse in 26 of 46 images | measured (score decomposition) |
| 6 | No off-the-shelf metric depth model gets the distance: all put the liver 3-9× too far (0 of 46 within 20 %), even when given the calibrated camera | measured, 5 models (2 more failed to install) |
| 7 | The CT CAN be registered: using the answer (ultrasound tumour, tracked probe, mask, Koo's landmarks), a rigid CT reaches official TRE 4.7 mm with IC 4/4 (a modest bend: 1.8 mm). In patient 2 no rigid pose fits the visible outline, even at 18-44 mm tumour error | oracle (uses the ground truth; not a method) |

Consequence for any new method: **the point clouds scaled by the P2I-LReg formula cannot be trusted for size or distance.**
Consequence of 4-5: **on real images a geometric fit score cannot be used to choose between distant poses; the scene
context has to constrain the search, not just propose candidates for the score to judge.**

## What was run

1. **Liver masks.** LLR-LUS has none; all 46 were drawn by hand (SAM prompts, then corrected in `dataset/lus-evaluation/mask_editor.html`),
   following P2I-LReg's convention: visible liver surface only (no falciform ligament, fat, tools, probe).
2. **Point clouds, as the authors made theirs.** The recipe is unpublished; it was recovered from the released P2I-LReg clouds
   (`lus_selfp2ir_prep.py`) and reproduces them to 0.001 mm with the file's scale:
   - depth from Depth Anything V1 **metric indoor** model (the paper says "relative"), nearest-neighbour upsampled;
   - every pixel back-projected with the **image centre** as principal point;
   - scale `σ = principal-axis length of the CT / that of the whole cloud`, where the whole cloud **includes the background
     pixels, which sit at the camera point**;
   - the cloud scaled by `σ` about its own bounding-box centre.
   Consequence: the farthest liver point always lands about **one CT-liver length** from the camera (0.98×, 84 frames),
   whatever the real distance.
3. **Check of the released model on its own data:** Dice 73.1 ± 8.1 % on its real test patients (paper, mean of 5 folds: 78.9 ± 6.8).
4. **Main run** on LLR-LUS (`selfp2ir_lus_kaggle.ipynb`), CT livers as given, and again with them moved to the origin.
5. **Oracle run** (`selfp2ir_lus_oracle_kaggle.ipynb`): the same, with each cloud's `σ` set so its depth next to the
   ultrasound probe equals the probe's tracked contact depth (`lus_oracle.py`). Uses the probe tracking, which also makes the
   ground truth: an **upper bound, not a method**.
6. **Frame check:** 100 synthetic P2I-LReg views with the CT rotated or moved in Self-P2IR's coordinates.

Tumour error is measured in 3D (mm): per image, the mean distance from the ultrasound tumour outline to the nearest point of
the predicted tumour; per patient, the official LLR-LUS score (`lus_evaluation.py`, a port of `Evaluation.m`).

## Results

**Official LLR-LUS score** (tumour error, mm; IC = all profile points within 1 cm of the tumour):

| Method | Patient 1 | Patient 2 | Patient 3 | Patient 4 | Overall | IC |
|---|---|---|---|---|---|---|
| Self-P2IR (authors' recipe) | 76.8 | 77.2 | 36.7 | 96.7 | **71.9** | 0/4 |
| Self-P2IR, CTs moved to the origin | 88.8 | 81.9 | 46.4 | 95.9 | 78.3 | 0/4 |
| Self-P2IR, true distance (oracle) | 16.1 | 67.1 | 18.7 | 19.1 | **30.2** | 0/4 |
| Koo et al. (published) | 9.5 | 39.0 | 25.1 | 18.4 | 23.0 | 0/4 |
| Adagolodjo et al. (published) | 8.3 | 37.2 | 28.4 | 15.8 | 22.4 | 0/4 |

**Per patient, Dice next to tumour error** (medians; depth = error along the viewing direction, + = too far):

| Patient | Dice | Tumour error | Depth | Sideways | True tumour depth | Predicted | Oracle error | Oracle Dice |
|---|---|---|---|---|---|---|---|---|
| 1 | 64.4 % | 67.7 | +67.1 | 7.3 | 110 | 186 | 14.2 | 68.3 % |
| 2 | 36.6 % | 78.0 | +51.4 | 60.3 | 169 | 218 | 64.1 | 36.5 % |
| 3 | 67.9 % | 33.4 | +32.1 | 9.9 | 165 | 200 | 14.2 | 71.0 % |
| 4 | 67.9 % | 88.6 | +86.0 | 25.2 | 80 | 176 | 9.1 | 62.5 % |
| all 46 | 63.8 % | 72.6 | +58.9 | | | | 20.6 | ~54 % |

## Evidence for each finding

**1. Dice hides a 7 cm tumour error.**
- Dice on LLR-LUS (63.8 %) is close to Self-P2IR's home score (73 %), yet the tumour is 72 mm off, about 3× the classic methods.
- Patients 3 and 4 have the same Dice (67.9 %) and tumour errors of 33 mm and 89 mm.
- Spearman(Dice, tumour error) over 46 images: −0.27. The 25 images with Dice ≥ 60 % have a median error of 67 mm; none is within 10 mm.
- Within Patient 2 the relation reverses: images 19–23 have the best Dice (~50 %) and the worst errors (~100 mm).

**2. The error comes from the distance, not the network.**
- 92 % of the error is depth, and the tumour is too far in **46/46** images.
- Self-P2IR fits its cloud well (median 8.6 mm cloud→CT); the clouds sit 145–180 mm from the camera whatever the truth (true tumour depth 75–173 mm).
- The depth error follows how far the cloud is from the true depth (r = +0.89).
- Oracle: putting only the distance right cuts the error from 71.9 to 30.2 mm overall, and to 9–14 mm for Patients 1, 3 and 4. Better in 40/46 images.
- Not the CT's placement in Self-P2IR's coordinates: Self-P2IR is sensitive to it (synthetic frame check: 99 % → 1 % within 5 mm when the CT moves 0.25 m), but moving the LLR-LUS CTs to P2I-LReg's convention made the result worse (78.3 mm).

**3. Dice cannot see depth.**
- Oracle vs main run: overall Dice ~53 % → ~54 % while the tumour error falls 72.6 → 20.6 mm (median).
- Patient 4: Dice 67.9 → 62.5 % (worse) while the error falls 88.6 → 9.1 mm.

## Caveats

- 4 patients, one tumour each; Patient 2 contributes 21 of the 46 images.
- The oracle sets the depth right next to the probe, ~1 cm above the tumour: it flatters the tumour's depth. 30 mm is an optimistic ceiling.
- Patients 1 and 4: the true-distance clouds are about half the size Self-P2IR was trained on (augmentation ×0.8–1.25); they still improved most.
- The cloud recipe matches the released clouds up to the scale itself: our `σ` is 0.3–0.8 % below the released scale files (cause unknown).
- The LLR-LUS images are not undistorted (P2I-LReg's are not either); the clouds use the image centre, as the released ones do.

## Not yet solid (next work)

- **Patient 2** stays at ~64 mm even at the true distance, mostly sideways (51 mm): a matching failure with little liver visible (~12 % of the image).
- **The slide** (`lus_slide.py`, `dataset/lus-evaluation/slide_explorer.html`): moving Self-P2IR's answer along the line of sight until the
  CT outline best matches the mask, no ground truth. Median error 72.7 → 41.1 mm (best possible along that line: 17.8 mm). Works on
  Patients 1 and 3 (10.7, 15.7 mm), fails on Patient 4 (58.7 mm) because the true-distance outline overflows the visible liver
  (scope border, probe). After the distance is fixed, ~8–12 mm of **sideways** error remains on Patients 1, 3, 4.

## Our method on LLR-LUS (findings 4 and 5)

**Method** (`goal/`; no tracking, no ultrasound, no landmarks; the tumour is only compared at the end):
1. Inputs per image: the photo **undistorted** with the LLR-LUS lens coefficients, the hand-drawn liver mask, the calibrated
   camera, the **depth shape** of the Self-P2IR-format cloud (Depth Anything; its distance is ignored), the CT at its real size.
2. The CT is put into body axes (x left, y back, z head) by fitting it to the 4 P2I-LReg livers (5-8 mm fit), so 6 surface
   parts are defined (left lateral section, segment 4, right lobe front, back/right side, underside, dome).
3. **Hidden area** by a fixed automatic rule (`goal/lus_edges.py`, no hand labels): the dark scope border, the image border,
   and unsaturated non-liver pixels touching the mask (probe, tools). The CT may run under these without penalty.
4. **Score** J = A/4.5 + B/0.02: A = outline distance (mask pixels to the CT outline, visible CT pixels to the mask, px);
   B = mean (1 - cos) between the CT's surface normals and the depth map's (scale-free). Weights fixed on synthetic sweeps.
5. **Search**: starting poses, each moved along its line of sight to match the mask's size and slid to its centre, scored;
   the best 8 polished (Nelder-Mead on rotation + position); lowest J wins. No hint: 256 directions x 16 spins all around.
   With hints: around each suggested class camera (part + head direction), +-30 deg x +-45 deg spin.
6. Light bending (5x5x4 cage) from the winner: changed nothing (65.8 -> 66.1 mm), so results below are rigid.

**Hints** (`goal/vlm/lus_suggestions.json`, sha256 `054747ee...a46f626`): 3 suggestions per image (part, head direction,
probability) written by Claude acting as the VLM from photo + mask sheets only, frozen before the run. They are the same
within a patient (views barely change), so effectively 4 readings. 5 images whose registration overlays Claude had seen
earlier are flagged; excluding them does not change the result.

**Official LLR-LUS score** (mm):

| | Patient 1 | Patient 2 | Patient 3 | Patient 4 | Overall |
|---|---|---|---|---|---|
| Ours, no hint | 21.0 | 95.9 | 70.9 | 75.3 | 65.8 |
| **Ours, hints constrain the search** | **12.8** | **40.7** | **38.8** | **38.9** | **32.8** |
| Ours, hinted and no-hint poses compete on the score J | 21.0 | 94.7 | 70.9 | 67.1 | 63.4 |
| Ours, ... compete on the outline term only | 12.6 | 56.0 | 70.9 | 39.0 | 44.6 |
| Ours, hints constrain the search, score = outline only (no depth model at all) | 22.3 | 47.1 | 65.4 | 37.6 | 43.1 |
| Self-P2IR (as given) | 76.8 | 77.2 | 36.7 | 96.7 | 71.9 |
| Self-P2IR, true distance (oracle) | 16.1 | 67.1 | 18.7 | 19.1 | 30.2 |
| Koo / Adagolodjo (published) | 9.5 / 8.3 | 39.0 / 37.2 | 25.1 / 28.4 | 18.4 / 15.8 | 23.0 / 22.4 |

Per image: median 61.4 -> 33.5 mm with hints (better in 31/46, worse in 5); median |depth error| 26.7 -> 14.7 mm;
Patient 3 (close-ups) 58.8 -> 15.6 mm; Patient 1 7/8 within 10 mm. No patient passes the inclusion test (no method does).

**4. Scene context halves the error.** Without hints the search fails wherever the outline carries little information:
close-ups where the liver fills the frame (Patients 3, 4) and dark, fragmented views (Patient 2); many poses, at different
distances, fit equally well. A coarse reading of the scene (which part, head up) removes most of them.

**5. The geometric score chooses wrong on real data.** In 26 of 46 images the no-hint pose scores better than the hinted
pose although the hinted pose is more than 5 mm (typically 30-100 mm) closer to the tumour. Splitting the score
(`dataset/lus-evaluation/goal_results/score_split.json`): the **shape term** favours the wrong pose in all 26 (alone in 15,
where the outline actually favours the right one; with the outline in 11, all Patient 3 or 2); the outline alone never does.
The shape mismatch at the (better) hinted pose is ~16 deg on Patient 1 (where the method works) but ~22-23 deg on
Patients 3-4 and ~32 deg on Patient 2, beyond the 10-15 deg the synthetic test tolerated. Choosing on the outline only
recovers part of the loss (44.6 mm) but not on the close-ups (Patient 3 stays 70.9), where the outline is uninformative.
Patient 2 also fits badly at every pose (outline spill 20-40 px; 56 % of the image hidden by the dark-scene rule, liver 13 %).
**But the shape still helps locally.** With the hinted search scored on the outline only (no depth model at all;
`HINT_TAG=_outline python goal/lus_hints.py --outline-only`), the official score worsens 32.8 -> 43.1 mm and 33/46 images
get worse: Patient 1 4.4 -> 10.1 mm median (7/8 -> 3/8 within 10 mm), Patient 3 (close-ups) 15.6 -> 65.2 mm; only
Patient 4 improves (38.3 -> 26.9 mm median). With the outline alone, the hinted pose always beats the no-hint pose
(46/46), so the outline is a safe judge between distant poses and the shape is a useful, if noisy, guide for fine
rotation: the shape should refine inside the region the scene context fixes, not choose between regions.

**Caveats.** 4 patients, one tumour each; one reader for the hints, identical within a patient; the hinted pose is not the
true pose, so "shape mismatch at the hinted pose" approximates the mismatch at the truth; Depth Anything V1 metric indoor
(nearest-neighbour upsampled) is the only depth model tried; the automatic edge rule hides real silhouettes in dark scenes.

## Metric depth models on LLR-LUS (finding 6)

Each model predicts depth in millimetres for the raw photo; we take its median on the liver pixels next to the probe
tip and compare with the tracked probe's true contact depth (median 152 mm; P1 94, P2 160, P3 153, P4 71). The probe is
used only to score. Run on Kaggle, 2× T4, 2026-11-17 (`depth_benchmark_kaggle.ipynb`).

| Model (camera) | median predicted / true | within 20 % | P1 | P2 | P3 | P4 |
|---|---|---|---|---|---|---|
| Self-P2IR recipe, after its scale step (uses the CT size) | 1.16 | 59 % | 1.69 | 1.11 | 1.16 | 2.08 |
| Depth Anything V1 metric indoor (Self-P2IR's input, before scaling) | 7.30 | 0 % | 10.5 | 7.1 | 6.8 | 14.1 |
| Depth Anything V2 metric indoor | 9.31 | 0 % | 9.1 | 9.7 | 6.5 | 19.3 |
| Depth Pro, own focal / calibrated focal | 5.24 / **3.00** | 0 % | 3.5 | 2.5 | 2.2 | 8.9 |
| UniDepth V2, own camera / calibrated | 6.76 / 5.96 | 0 % | 6.4 | 5.6 | 3.7 | 8.2 |
| MoGe-2, own field of view / calibrated | 7.65 / 5.91 | 0 % | 12.8 | 5.5 | 5.4 | 12.7 |

The factor is not even constant: for Depth Pro (calibrated) it is 2.2 on P3 and 8.9 on P4, so no single correction
would fix it. The models were trained on rooms and streets, where surfaces are metres away; a liver 7-16 cm from the
lens is outside what they have seen. The distance must come from elsewhere: the camera calibration and the CT's size,
which is what the scale step and our size matching do. Metric3D (GitHub 504 during download) and Depth Anything 3
(Python version pin) did not run; both are fixed for the next run, which also saves every depth map so each model's
SHAPE can be tried in our registration.

## Best possible registration (finding 7; oracle, uses the answer)

`goal/lus_best.py` registers each image with everything available, including the ground truth: score
E = 2 x tumour error (profile -> CT tumour surface, mm) + probe-face gap to the liver surface (mm) + outline gap A / 4.5
+ Koo ridge/ligament landmark gap (px) / 18. Rigid: 400 directions x 12 rolls with the CT tumour on the profile, best 8
plus our 3 saved poses polished. Bend: 40 Gaussian control points (reach 40 mm, the playground's model), pushes fitted by
least squares to the same evidence with a size penalty (lambda 1), alternated with pose polish, kept only if E drops.

| | Patient 1 | Patient 2 | Patient 3 | Patient 4 | Overall | IC |
|---|---|---|---|---|---|---|
| Best rigid (official TRE) | 6.27 | 2.39 | 6.85 | 3.46 | **4.74** | **4/4** |
| Best rigid + bend (official TRE) | 1.84 | 1.82 | 0.94 | 2.70 | **1.82** | **4/4** |
| Ours, hints (for comparison) | 12.8 | 40.7 | 38.8 | 38.9 | 32.8 | 0/4 |

Per image (median): rigid tumour error 3.1 / 1.3 / 2.6 / 2.0 mm (all 46 within 4.1 mm); rigid outline gap 4.1 / 40.1 / 16.0 / 17.3 px;
bend moves the tissue 7 / 28 / 14 / 8 mm on average (max 16 / 78 / 42 / 24).
- Patients 1, 3, 4: one rigid pose fits the tumour, the probe contact and the outline together; a 7-14 mm bend closes the rest.
- Patient 2: the tumour-correct rigid CT is far larger than the visible liver and Koo's landmarks sit 230+ px off. Lowering
  the tumour weight (tested on images 02, 10, 21) never brings the outline below 34 px, even at 18-44 mm tumour error, so no
  rigid pose explains the photo; the bend needed (28 mm mean, up to 87 mm) is not believable. This matches the literature
  reporting LLR-LUS "without P2". Image 21 also has a second pose 169 deg away with a tumour error of 2.7 mm (spin about the tumour).
- Interactive: `python goal/make_playground_lus.py` -> `dataset/lus-evaluation/register_playground_lus.html` (opens on the
  best-possible pose; 3D view with the laparoscope, the ultrasound slice and the probe).

## Reproduce

```
python studies/benchmark/lus_manual_masks.py import ~/Downloads/lus_manual_masks.json      # masks
(cd <Depth-Anything>/metric_depth && python <repo>/studies/benchmark/lus_selfp2ir_prep.py --repo-root <repo> --weights dataset/models)
python studies/benchmark/lus_kaggle_pack.py pack                                           # upload for Kaggle
python studies/benchmark/lus_oracle.py                                                     # oracle scales
python studies/benchmark/kaggle/make_lus_notebook.py [--oracle]                            # notebooks
python studies/benchmark/lus_selfp2ir_score.py --results dataset/lus-evaluation/selfp2ir_results/<run>/real_frames.json --out dataset/lus-evaluation/selfp2ir_tumours/<run>
python studies/benchmark/show_lus_selfp2ir.py --image Patient4/02 --compare koo             # 3D view
python studies/benchmark/lus_slide.py                                                      # slide + explorer data
python goal/lus_all.py                                                                      # ours, no hint (+ official score)
python goal/lus_hints.py                                                                    # ours, VLM hints (+ official score)
python studies/benchmark/kaggle/make_depth_notebook.py                                     # metric depth benchmark (Kaggle)
python goal/lus_best.py                                                                     # best possible registration, uses the answer (+ official score)
python goal/make_playground_lus.py                                                          # interactive page
```
