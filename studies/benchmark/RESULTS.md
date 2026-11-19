# Results: does Dice tell us where the tumour is? (Self-P2IR)

**Pre-registration:** [PREREGISTRATION.md](PREREGISTRATION.md), committed as `c0cc330` (2026-11-07) before any Dice or TRE was computed on the 21-patient predictions.
**Data:** all 21 P2I-LReg patients, synthetic test views, run on Kaggle (2× T4). Outputs: [selfp2ir_kaggle_out/](selfp2ir_kaggle_out/).
**Scored:** 2026-11-07/08. Thresholds and code as pre-registered; deviations are listed at the end.

## Summary

- The authors' model runs as published: 98.2 % of views within 5 mm (paper: > 92 %).
- With the cloud at its true size (condition A), the model is accurate and Dice is a sufficient check: 0.2 % of probe tumours miss the 1 cm margin among views with Dice ≥ 0.90.
- With the cloud resized by the authors' real-frame recipe (condition B), **51 % of probe tumours miss the 1 cm margin among views with Dice ≥ 0.90**. Under the relative definition (top 25 % of Dice) the figure is 8.6 %, so the pre-registered H1 verdict is **inconclusive**.
- Dice ranks views well under a wrong scale (ρ = −0.89, **good proxy**), and it does notice a wrong scale (Dice −0.06 for +12 mm, **H3 not confirmed**). The pilot's claim that Dice is blind to scale does not hold for this model and is dropped as a headline.
- What the data support: **the value of Dice is not tied to tumour error.** Dice ≥ 0.90 meant 0.2 % misses in one condition and 51 % in another. The error is almost entirely along the viewing direction (depth).

## Reproduction check (before any Dice or TRE)

Condition A, 2,793 views, 7 rejected by the authors' loader.

| ρ (mm) | 2 | 5 | 10 | 20 | 40 |
|---|---|---|---|---|---|
| Registration recall | 90.3 % | **98.2 %** | 99.5 % | 99.9 % | 99.9 % |

Mean CT point error: median 0.87 mm, 90th percentile 1.98 mm. Paper (Fig. 7a): > 92 % at 5 mm. Passed.

## Pre-registered verdicts

2,017 views per condition (same views in every condition), 200 probe tumours per patient. 95 % confidence intervals resample patients.

| | Condition A (true size) | Condition B (recipe scale) |
|---|---|---|
| Median Dice | 0.991 | 0.932 |
| Median probe TRE (depth part) | 0.8 mm (0.5 mm) | 12.9 mm (12.3 mm) |
| **H1** Dice ≥ 0.90: probes > 10 mm | 0.2 % (CI 0.1–0.5), 2,016 views | 51.2 % (CI 42.5–59.8), 1,477 views |
| **H1** top 25 % Dice: probes > 10 mm | 0.1 % (CI 0.0–0.4), 505 views | 8.6 % (CI 3.8–14.8), 505 views |
| **H1 verdict** | **a good Dice is enough** (both < 2 %) | **inconclusive** (51.2 % ≥ 10 %, 8.6 % < 10 %) |
| **H2** Spearman(Dice, view TRE) | −0.49 (CI −0.58 to −0.38) → **partial** | −0.89 (CI −0.91 to −0.87) → **good proxy** |
| **E1** Spearman with view TRE (exploratory) | Dice −0.49, distance +0.68 | Dice −0.89, distance +0.94 |

**H3** (B vs A, same views): median change in Dice −0.059, median change in view TRE +12.0 mm → **not confirmed** (the pre-registered line was a Dice change < 0.02 with a TRE change > 5 mm).

The pre-registered "results against the idea":
- *H1 < 2 % and H2 ≥ 0.8 in condition B*: not met (H1 is 8.6–51.2 %).
- *H3 not confirmed*: met. The scale blind spot from the pilot does not appear with this model.

## Sensitivity: fixed scale factors

| Factor | Median Dice | Median TRE (depth) | H1 Dice ≥ 0.90 | H1 top 25 % | H1 verdict | H2 ρ | H3: ΔDice / ΔTRE |
|---|---|---|---|---|---|---|---|
| ×1.05 | 0.966 | 5.8 mm (5.4) | 10.5 % | 1.2 % | inconclusive | −0.64 partial | −0.023 / +4.9 mm, not confirmed |
| ×1.11 | 0.920 | 13.1 mm (12.4) | 80.0 % | 47.4 % | **not enough** | −0.68 partial | −0.070 / +12.2 mm, not confirmed |
| ×1.32 | 0.776 | 38.7 mm (36.9) | 100 % (91 views) | 100 % | **not enough** | −0.69 partial | −0.215 / +37.9 mm, not confirmed |

## Per patient (condition A and B, from `per_view.csv`)

Every patient shows the same pattern: A near 1 mm, B around 8–19 mm. Among views with Dice ≥ 0.90, between 19 % and 91 % of probes miss 10 mm in B.

| Patient | Views scored | Excluded | A: median Dice | A: median TRE | B: median Dice | B: median TRE | B, Dice ≥ 0.90: probes > 10 mm |
|---|---|---|---|---|---|---|---|
| 01 | 100 | 0 | 0.993 | 0.7 mm | 0.929 | 16.6 mm | 59 % (71 views) |
| 02 | 100 | 0 | 0.993 | 1.0 mm | 0.949 | 12.1 mm | 42 % (74 views) |
| 03 | 100 | 0 | 0.988 | 1.2 mm | 0.907 | 18.3 mm | 60 % (53 views) |
| 04 | 99 | 1 | 0.991 | 0.9 mm | 0.918 | 14.4 mm | 71 % (66 views) |
| 05 | 91 | 8 | 0.989 | 0.9 mm | 0.923 | 14.5 mm | 74 % (73 views) |
| 06 | 99 | 1 | 0.991 | 1.2 mm | 0.957 | 9.9 mm | 40 % (85 views) |
| 07 | 93 | 57 | 0.992 | 0.8 mm | 0.917 | 13.1 mm | 63 % (65 views) |
| 08 | 99 | 51 | 0.992 | 0.7 mm | 0.945 | 8.8 mm | 23 % (70 views) |
| 09 | 93 | 56 | 0.991 | 0.9 mm | 0.954 | 7.8 mm | 19 % (82 views) |
| 10 | 96 | 54 | 0.992 | 0.7 mm | 0.937 | 10.5 mm | 33 % (68 views) |
| 11 | 89 | 60 | 0.986 | 0.8 mm | 0.903 | 17.3 mm | 91 % (50 views) |
| 12 | 100 | 50 | 0.991 | 0.9 mm | 0.959 | 8.1 mm | 28 % (86 views) |
| 13 | 99 | 50 | 0.992 | 0.7 mm | 0.915 | 15.3 mm | 81 % (68 views) |
| 14 | 98 | 52 | 0.992 | 0.8 mm | 0.946 | 9.4 mm | 39 % (85 views) |
| 15 | 100 | 50 | 0.993 | 0.6 mm | 0.947 | 9.2 mm | 37 % (87 views) |
| 16 | 98 | 51 | 0.992 | 0.7 mm | 0.926 | 12.8 mm | 61 % (77 views) |
| 17 | 99 | 51 | 0.992 | 0.8 mm | 0.927 | 15.1 mm | 43 % (61 views) |
| 18 | 95 | 54 | 0.992 | 0.5 mm | 0.926 | 13.6 mm | 72 % (66 views) |
| 19 | 78 | 72 | 0.990 | 0.7 mm | 0.909 | 19.2 mm | 89 % (44 views) |
| 20 | 98 | 52 | 0.993 | 0.7 mm | 0.944 | 9.9 mm | 39 % (81 views) |
| 21 | 93 | 6 | 0.990 | 0.9 mm | 0.922 | 14.4 mm | 62 % (65 views) |

## Exclusions

776 of 2,793 views were excluded by the pre-registered true-pose check, which uses only the true pose and never the model's prediction.

- **About 700: an extra object in the liver cloud.** Every patient with 2,500 synthetic views (07–20) has about 50 test views among 2000–2499 in which an object is labelled background but included in the liver cloud: 15–40 % of cloud points lie more than 3 mm from the CT at the true pose. Patients with 2,000 views (01–06, 21) are not affected. This is a property of the released dataset.
- **About 76: true-pose Dice below 0.98** with a clean cloud (patients 04, 05, 06, 21 and a few in 07–20).
- 7 views rejected by the authors' loader (same views in all conditions).

## Exploratory (decided after seeing the results; no verdicts)

- **Why the two H1 definitions disagree in B.** The recipe gives each view its own scale (median ×1.10). The top 25 % of views by Dice are those where that scale happened to be close to 1 (median |scale − 1| 0.04 vs 0.13 for the rest; rank correlation between |log scale| and Dice −0.92). Dice picks out the lucky views but does not say how far off the others are.
- **Dice bands in condition B:**

  | Dice | Views | Median view TRE | Views with TRE > 10 mm |
  |---|---|---|---|
  | ≥ 0.95 | 681 | 6.7 mm | 13 % |
  | 0.92–0.95 | 525 | 12.1 mm | 75 % |
  | 0.90–0.92 | 271 | 15.5 mm | 100 % |
  | 0.85–0.90 | 345 | 20.8 mm | 100 % |

  In condition A no view had Dice below 0.95. A Dice that would be reported as excellent (0.90–0.92) went with a miss above 1 cm in every view here.
- **Distance tracks TRE better than Dice** in all five conditions (E1).

## Deviations and notes

- **Platform.** Neighbourhood limits are `[6, 16, 25, 29]` on Kaggle and `[6, 15, 24, 29]` locally with the same seed (amendment 11). Poses agree to a median of 0.19 mm on shared views. All confirmatory numbers come from Kaggle.
- **Reporting added after the results.** The pre-registration also promised the on-screen (px) part of TRE, TRE for probes in view only, and per-patient results. The scorer computed the first two but did not print them. After this run a reporting block was added (`extras` in [score_selfp2ir.py](score_selfp2ir.py)). It prints only these promised quantities, and the verdict lines are unchanged (checked on the 4-patient dry run). Per-patient results above come from the confirmatory `per_view.csv`. **The 21-patient px and in-view numbers are pending a re-scoring run on Kaggle**. On the 4-patient dry run: on-screen error median 6.2 px in B (1.3 px in A); 96 % of probes in view; in-view-only H1 52.0 % (B), 0.2 % (A).
- **Dry run.** The 4-patient local run ([selfp2ir_out/](selfp2ir_out/)) was a pipeline test (amendment 9). Its numbers agree with the confirmatory run and were not used for any verdict.

## Limits

- Synthetic, rigid livers; no deformation, no real depth-estimation noise beyond the scale.
- One model. Self-P2IR's final step is rigid (no scale), so a wrong-size cloud cannot be fitted exactly, which is why its Dice notices the scale. A method that estimates scale or depth could fit the outline exactly at the wrong depth; that is untested.
- Probe tumours are spheres placed uniformly in the liver.
- Thresholds (10 %, 2 %, 0.8, 5 mm, 0.02, top 25 %) are our pre-registered choices; only the 10 mm margin is clinical convention.
- Training views of the released weights are not stored in the file; `test_syn.txt` views are assumed unseen.
