# Results: Depth Anything V2 Large on SERV-CT (2026-11-19, SLURM job 2702, GPU)

Shape error after removing scale and shift, measured at the generator's angular resolution (focal 343 px),
16 images of two ex vivo pigs. Median (10th - 90th percentile):

| Measure | Median | 10th - 90th pct | Meaning |
|---|---|---|---|
| **Total tilt error** | **20.4 deg** | 16.1 - 27.9 | how wrong the surface orientation is overall |
| Bumps alone | 17.1 deg | 6.2 - 25.8 | local dents / bulges: the largest part |
| Slant alone | 8.9 deg | 6.0 - 15.9 | a tilted plane across the picture |
| Flattening alone | 5.5 deg | 1.6 - 18.4 | relief too shallow (or too deep) |
| Flattening factor k | 0.82 | 0.36 - 1.09 | 1 = right relief; < 1 = flattened; > 1 = exaggerated |
| Bump size (autocorrelation half-width) | 23 px | 17 - 33 | at focal 343 px; ~ Gaussian sigma 14 px (10 - 20) |
| AbsRel (after scale + shift) | 0.048 | 0.028 - 0.067 | for comparison with the literature |

The three parts are measured one at a time, so they do not add up to the total (they partly cancel or overlap).
Per-image numbers: `summary.csv`; panels: `figures/` (on gpu-a240, `/data/khoalq/registration/depth-noise-estimate/results`).

## What it means

- Real Depth Anything shape error is about **20 deg** (16 - 28), above the ~10 - 15 deg our hand-made shape terms
  tolerated in the synthetic tests, and consistent with the 16 - 32 deg shape mismatch measured on LLR-LUS. The
  depth shape is a weak hint, not something to trust.
- The error is **mostly bumps**, then slant, then mild flattening; all three occur together in the same image.

## Settings for the generator's depth damage (regsim/depth_damage.py), with a margin

| Part | Measured (10th - 90th) | Generator range |
|---|---|---|
| Bumps | 6 - 26 deg, sigma ~10 - 20 px at a 480-px render | 0 - 30 deg, sigma 8 - 24 px (the prototype used 48 px: too smooth) |
| Slant | 6 - 16 deg | 0 - 20 deg |
| Flattening k | 0.36 - 1.09 | 0.3 - 1.2 |
| Combined | total 16 - 28 deg | all three applied together, not one kind per sample |

## Limits

16 images, pig tissue, whole scenes (several organs, not liver only), CT reference ~2 mm accurate (part of the
"bumps" may be reference error). Larger checks: SCARED (by request).
