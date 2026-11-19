# Pre-registration: does Dice tell us where the tumour is? (Self-P2IR)

**Written:** 2026-11-05, before any Self-P2IR prediction has been run or looked at.
**Pilot this builds on:** [pilot_dice_vs_tumour.py](pilot_dice_vs_tumour.py), with invented pose errors. This study uses predictions from a real published model instead.

Any change to this file is listed under *Amendments* with its reason and whether it was made before or after the first run. Analyses dropped by an amendment are still reported.

## Question

Papers on P2I-LReg report Dice (do the 2D outlines overlap?) and a surface distance. Surgeons care about TRE: how far the shown tumour is from the real one. On real frames TRE cannot be measured, so we test on synthetic views, where it can, whether Dice and distance tell us anything about TRE.

## Model

- **Self-P2IR** (Zhou et al., IEEE TMI 2025), official code (github.com/junzastar/Self-P2IR) and the authors' pretrained weights, unchanged.
- Recorded before the run:
  - repo commit `60d6c1b30d57148e6bc0c97fd1fab0f68cc0f079` (2025-12-04);
  - weights `model_best_loss.pth` (the only file released, 182 MB), sha256 `e83a9110bb6a1b1bf7e49e7e0f3a3736b0b01ce9ec76626972a5406afaa963ba`;
  - what the file is: the real-data model at epoch 19. It holds `posenet` (the rigid network) plus the real-data modules. The optimiser holds no state for the `posenet` tensors, so `posenet` was frozen during real-data training (as `main.py` does after loading the synthetic model). The `posenet` weights are therefore the synthetic-trained rigid network, which is what we test;
  - training views: not stored in the file. We assume the standard split (`train_syn.txt`), so `test_syn.txt` views are unseen. This assumption is stated with the results.
- Code used: [selfp2ir_predict.py](selfp2ir_predict.py) runs their synthetic tester's steps unchanged (matching, `get_match(thr=0.15)`, weighted SVD). Its header lists every deviation needed to run it.
- The model returns a pose, possibly with scale. It is trained with random scale ×0.8–1.25, so its rotation can contain a scale. Every prediction is used as a similarity transform: CT point X → s·R·X + t.

## Data

- **Views:** all synthetic test views (`test_syn.txt`) of every patient available on Kaggle that the tested model has not seen in training. Not a subset.
- **Exclusions, fixed in advance:**
  - views that fail the true-pose check (true-pose Dice < 0.98, or > 2 % of cloud points more than 3 mm from the CT). This removes views 2000–2499 of patients 10 and 11, where an extra object is labelled background but included in the liver cloud;
  - views the official loader rejects (fewer than 100 liver pixels or fewer than 1000 correspondences to the true pose). Their test loop silently replaces such a view with a random other view; we skip and record it instead.
  - The number excluded per patient is reported.
- **Probe tumours:**
  - 200 per patient, seed 0;
  - centres uniform through the whole liver volume (not only the visible part);
  - radius 5–25 mm, each sphere fully inside the liver.
  - Each probe answers "if this patient's tumour were here, how far off would it be?" They are not a model of how many tumours a patient has.
- **Condition A, clean:** the input exactly as the official test code prepares synthetic views (true size, `scale = 1.0`).
- **Condition B, recipe scale:**
  - the same views, but the cloud is resized by the PCA scale recipe the authors use for real frames, applied to the synthetic cloud;
  - our recipe implementation is fixed and committed before the run;
  - as a validity check we report how closely it reproduces the released `real/scale/*.txt` values;
  - if the agreement is poor, B results are labelled "approximate recipe";
  - sensitivity runs with fixed factors ×1.05, ×1.11 and ×1.32 are also reported.

## Measures

| Measure | Definition |
|---|---|
| **Dice** (per view) | Overlap of the CT outline rendered with the predicted pose and the liver label, ignoring tool pixels (label 153, 179). Same code as the pilot. |
| **Distance** (per view) | Mean distance (mm) from the visible cloud, taken to the CT frame with the predicted pose, to the CT surface. |
| **TRE** (per probe tumour) | Distance (mm) between the predicted and true position of that tumour centre. |
| **View TRE** (per view) | Mean TRE over the 200 probes. Used wherever one number per view is needed. |

**Good Dice**, two definitions, each applied within a condition:
- **absolute:** Dice ≥ 0.90;
- **relative:** the top 25 % of views by Dice.

Also reported:
- the depth and on-screen (px) parts of TRE;
- TRE for probes in view only;
- results per patient;
- 95 % confidence intervals from resampling patients (patients, not views, are the independent units).

## Hypotheses and what each result means

H1 and H2 are run separately on condition A and condition B.

**H1: Can a good Dice be trusted?**
Among good-Dice views, the % of (view, probe) pairs with TRE > 10 mm (the 1 cm margin), i.e. the chance that a patient's tumour misses the margin when Dice looks good.
- **≥ 10 %** under both good-Dice definitions: a good Dice is not enough.
- **< 2 %** under both: a good Dice is enough for this model.
- **Otherwise** (between, or the two definitions disagree): inconclusive.

Also reported:
- the same at 5 mm, and as a curve over margins from 0 to 10 mm;
- the per-view version: % of good-Dice views with view TRE > 10 mm.

**H2: Does a better Dice mean a lower TRE?**
Spearman correlation between Dice and view TRE across all views.
- **|ρ| ≥ 0.8:** Dice is a good proxy.
- **0.4–0.8:** partial.
- **< 0.4:** weak.

**H3: When the scale is wrong, does Dice notice?**
On the same views, compare B with A: the median change in Dice and the median change in view TRE.
- **Confirmed (Dice is blind):** Dice changes by < 0.02 while view TRE grows by > 5 mm.
- **Not confirmed:** otherwise.

**E1: Which is the better warning sign, Dice or distance?** (exploratory: reported, no verdict)
Spearman correlation of each with view TRE, in A and in B, side by side.

## Expected outcome and results that count against the idea

**Expected:** the paper reports near-perfect synthetic poses (rotation error ~0.19°, translation error ~1.2 mm, Table I). So in condition A we expect high Dice, small TRE and few H1 failures. That would mean the model is accurate on clean data, not that Dice is a good measure. Condition B is the informative test.

Results against the idea, reported as such:
- **H1 < 2 % and H2 ≥ 0.8 in condition B:** even with realistic scale, Dice is a good enough stand-in for this model. The argument for a new metric then rests on other methods or other realistic corruptions.
- **H3 not confirmed:** the scale blind spot from the pilot does not appear with a real model. It is dropped as a headline claim.

## Comparability with published numbers

- The paper's Dice of 78.89 % is measured on real frames, after deformation, with five-fold cross-validation.
- Our Dice is measured on synthetic frames, rigid, with tool pixels ignored.
- The two cannot be compared directly, and no claim of the form "their Dice of X hides Y mm" will be made from this study.

## Rules

- Thresholds are not changed after seeing results.
- All results are reported, including those against the idea.
- Limits stated with the results:
  - synthetic livers are rigid (no deformation);
  - probe tumours are spheres placed uniformly (LiTS location and size statistics come later);
  - the thresholds (10 %, 2 %, 0.8, 5 mm, 0.02, top 25 %) are our choices, made before the run, not literature standards. Only the 10 mm margin is clinical convention.

## Amendments

All made on 2026-11-05, **before any Self-P2IR run.**

1. **Probe tumours raised from 20 to 200 per patient.**
   A precision check on pilot predictions (240 predictions, patients 01 and 10, compared with 2000 probes) found that with 20 probes:
   - the view mean was off by 11.6 % in the worst 5 % of cases;
   - the "> 10 mm" verdict flipped for 2.4 % of predictions, about the size of the 2 % threshold in H1.
   Probes cost nothing to add.

2. **H1 counts each probe on its own instead of the view average.**
   The clinical question is the chance that the patient's tumour misses the margin, and an average can hide individual misses. The per-view version is still reported.

3. **H1 and H2 run on condition B as well as A.**
   The paper's near-perfect synthetic poses mean condition A alone would mostly show that the model is accurate on clean data.

4. **Relative good-Dice definition (top 25 %) added, required to agree with the absolute one.**
   If every view in a condition has Dice ≥ 0.90, the absolute line no longer separates good from bad.

5. **Released weights must be identified, and only unseen views used.**
   The paper trains real-data models per fold and uses patient-specific synthetic data, so the views available for testing depend on which models were released. (Done: see *Model*.)

Made on 2026-11-06, **before any Dice or TRE has been computed for Self-P2IR.**

6. **Condition B uses the recipe as written in the paper, with the range definition, labelled "approximate recipe".**
   The paper does not say how the projections onto the principal axis become one size. Validity check on all 84 local real frames ([scale_recipe.py](scale_recipe.py)): if the released clouds followed the recipe, size(CT) / size(PLY) would be exactly 1.000 on every frame. It is not:
   - range: median 1.112, 10–90 % 0.900–1.313;
   - std: median 1.379, 10–90 % 1.109–1.773.
   The released σ cannot be reproduced from the paper. Range is used because it is the closer of the two. As pre-registered, results from the fixed factors ×1.05, ×1.11 and ×1.32 are reported alongside.

7. **Reproduction check before the hypotheses.**
   Before any Dice or TRE is computed, condition A is compared with the paper on the authors' own registration recall: the % of views whose CT points land, on average, within ρ of their true position, with ρ = 2, 5, 10, 20, 40 mm. The paper reports > 92 % at ρ = 5 mm (Fig. 7a). If our run is far below that, the cause is found and fixed (or reported) before the hypotheses are tested.
   The paper's rotation error (Table I) is not used for this check: their code returns it in radians while the table labels it degrees, so the reference value is ambiguous.

8. **Rejected views are skipped, not replaced** (see *Data*). This differs from the authors' test loop, which substitutes a random view and so can count some views twice.

9. **The verdicts come only from the 21-patient run.**
   A first run on the 4 locally available patients (01, 06, 10, 11) was started on 2026-11-06 to test the whole pipeline end to end. Decided while it was running, before any Dice or TRE was computed:
   - its results are a dry run, reported as preliminary, and no verdict is declared from them;
   - the pre-registered verdicts come only from all patients on Kaggle, with the code and thresholds unchanged;
   - nothing in this file may be changed because of what the dry run shows.
   Reason: confidence intervals resample patients, and 4 patients give intervals too wide to judge the thresholds.

10. **Runs are seeded, including Open3D.**
    Their code samples CT points with Open3D's random generator, which it never seeds. The neighbourhood limits calibrated from those points changed between two runs, `[9, 21, 25, 29]` unseeded vs `[6, 15, 24, 29]` seeded, so their pipeline does not repeat exactly and the limits behind their published numbers cannot be known. We seed numpy, torch and Open3D (seed 0), calibrate as `main.py` does, and record the limits in every predictions file. All conditions of a run must share the same limits; the scorer reports them. The dry-run condition A, started before this change, is rerun with seeding.

Made on 2026-11-06, **after the Kaggle A and B_recipe predictions, before any Dice or TRE has been computed on the 21 patients.** A disclosure only; no analysis changes.

11. **Seeding makes runs repeat on one platform, not across platforms.**
    With the same seed, the neighbourhood limits are `[6, 15, 24, 29]` locally (CPU, macOS) and `[6, 16, 25, 29]` on Kaggle (GPU, Linux). On the 499 views both runs share (patients 01, 06, 10, 11) the predicted poses differ by a median of 0.19 mm and 0.17° (90th percentile 0.41 mm, 0.37°), and the same views are rejected. The confirmatory analysis uses only Kaggle predictions, so every condition shares the Kaggle limits (the scorer checks this).
